"""
Generate a distillation corpus by running batch inference on the pre-trained
NotaGen teacher model.

For each metadata prompt sampled from the real training data, we generate one
completion per temperature in GEN_TEMPERATURES.  The lowest-temperature output
is the "chosen" sequence and the highest-temperature output is the "rejected"
sequence, giving (prompt, chosen, rejected) triplets suitable for DPO in
addition to plain SFT pairs.

Output JSONL schema (one object per line):
{
  "prompt":   "<metadata header lines>",
  "chosen":   "<tunebody at low temperature>",
  "rejected": "<tunebody at high temperature>",
  "all_completions": [{"temperature": T, "abc": "..."}, ...]
}
"""

import os
import sys
import json
import time
import random
import argparse

import torch

# Re-use the existing inference utilities without modifying the original repo.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load distillation/config.py explicitly. Clear any cached `config` first so we
# bind to OUR config (not inference/config.py, which lacks these constants).
sys.modules.pop("config", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    NOTAGEN_WEIGHTS_PATH,
    PATCH_STREAM, PATCH_SIZE, PATCH_LENGTH,
    CHAR_NUM_LAYERS, PATCH_NUM_LAYERS, HIDDEN_SIZE,
    PATCH_SAMPLING_BATCH_SIZE,
    ENCODER_BACKBONE, DECODER_BACKBONE, MOTIF_ATTENTION_BIAS,
    CORPUS_OUTPUT_PATH, REAL_TRAIN_JSONL,
    NUM_CORPUS_SAMPLES, GEN_TEMPERATURES, GEN_TOP_K, GEN_TOP_P, GEN_TIMEOUT_SECS,
)
sys.path.pop(0)

# inference/utils.py does `from config import *`; `config` is now cached as the
# distillation config (which supplies the PATCH_SIZE/PATCH_STREAM Patchilizer needs).
sys.path.insert(0, os.path.join(_REPO_ROOT, "inference"))
from utils import Patchilizer, NotaGenLMHeadModel, safe_normalize_probs  # noqa: E402
from samplings import top_k_sampling, top_p_sampling, temperature_sampling
from notagen_core import build_notagen_configs  # noqa: E402  (repo root added to path by utils import)
from abctoolkit.transpose import Key2index, Key2Mode


# ── Model setup ───────────────────────────────────────────────────────────────

def load_notagen(weights_path: str, device: torch.device) -> NotaGenLMHeadModel:
    patch_config, char_config = build_notagen_configs(
        encoder_backbone=ENCODER_BACKBONE, decoder_backbone=DECODER_BACKBONE,
        patch_num_layers=PATCH_NUM_LAYERS, char_num_layers=CHAR_NUM_LAYERS,
        hidden_size=HIDDEN_SIZE, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE,
        motif_attention_bias=MOTIF_ATTENTION_BIAS,
        patch_sampling_batch_size=PATCH_SAMPLING_BATCH_SIZE,
    )
    model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=char_config)
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"Loaded NotaGen from {weights_path}  "
          f"(epoch {ckpt.get('epoch','?')}, loss {ckpt.get('min_eval_loss','?'):.4f})")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ── Metadata extraction ───────────────────────────────────────────────────────

def extract_metadata_prompt(abc_path: str, key: str) -> str | None:
    """Read one transposed ABC file and return only the metadata header lines."""
    folder = os.path.dirname(abc_path)
    name   = os.path.basename(abc_path)
    file_path = os.path.join(folder, key, f"{name}_{key}.abc")
    if not os.path.exists(file_path):
        return None
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
    lines = text.split("\n")
    metadata_lines = []
    for line in lines:
        if line.startswith("[V:") or line.startswith("[r:"):
            break
        # Strip motif lines — the pre-trained model was never trained with them
        if line.startswith("%motif:"):
            continue
        if line.strip():
            metadata_lines.append(line + "\n")
    return "".join(metadata_lines) if metadata_lines else None


def collect_prompts(jsonl_path: str, max_prompts: int) -> list[str]:
    with open(jsonl_path, encoding="utf-8") as f:
        entries = [json.loads(l) for l in f]
    real_entries = [e for e in entries if "synthetic" not in e["path"]]
    random.shuffle(real_entries)

    prompts = []
    seen = set()
    for entry in real_entries:
        prompt = extract_metadata_prompt(entry["path"], entry["key"])
        if prompt and prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
        if len(prompts) >= max_prompts:
            break
    return prompts


# ── Single-sequence generation ────────────────────────────────────────────────

def generate_one(
    model: NotaGenLMHeadModel,
    patchilizer: Patchilizer,
    prompt_text: str,
    temperature: float,
    device: torch.device,
    full_output: bool = False,
) -> str | None:
    """Generate from a metadata prompt.  Returns None on failure.
    full_output=False -> tunebody only (Qwen corpus); True -> the complete ABC
    (prompt metadata + generated %%score/L/Q/M/K + tunebody), [r:] markers
    stripped, suitable for re-encoding."""
    prompt_lines = [line for line in prompt_text.split("\n") if line.strip()]
    prompt_lines = [line + "\n" for line in prompt_lines]

    bos_patch     = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]
    prompt_patches = patchilizer.patchilize_metadata(prompt_lines)
    prompt_patches = [
        [ord(c) for c in p] + [patchilizer.special_token_id] * (PATCH_SIZE - len(p))
        for p in prompt_patches
    ]
    prompt_patches.insert(0, bos_patch)

    input_patches  = torch.tensor(prompt_patches, device=device).reshape(1, -1)
    byte_list      = list("".join(prompt_lines))
    tunebody_flag  = False
    cut_index      = None
    start_time     = time.time()

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            while True:
                predicted_patch = model.generate(
                    input_patches.unsqueeze(0),
                    top_k=GEN_TOP_K,
                    top_p=GEN_TOP_P,
                    temperature=temperature,
                )

                # Force the first tunebody patch to start with [r:0/
                if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith("[r:"):
                    tunebody_flag = True
                    r0 = torch.tensor([ord(c) for c in "[r:0/"]).unsqueeze(0).to(device)
                    temp_patches = torch.cat([input_patches, r0], dim=-1)
                    predicted_patch = model.generate(
                        temp_patches.unsqueeze(0),
                        top_k=GEN_TOP_K,
                        top_p=GEN_TOP_P,
                        temperature=temperature,
                    )
                    predicted_patch = [ord(c) for c in "[r:0/"] + predicted_patch

                # EOS
                if predicted_patch[0] == patchilizer.bos_token_id and predicted_patch[1] == patchilizer.eos_token_id:
                    break

                byte_list.extend(patchilizer.decode([predicted_patch]))

                # Mask padding after EOS within patch
                hit_eos = False
                for j in range(len(predicted_patch)):
                    if hit_eos:
                        predicted_patch[j] = patchilizer.special_token_id
                    if predicted_patch[j] == patchilizer.eos_token_id:
                        hit_eos = True

                predicted_patch = torch.tensor([predicted_patch], device=device)
                input_patches   = torch.cat([input_patches, predicted_patch], dim=1)

                if len(byte_list) > 102400 or time.time() - start_time > GEN_TIMEOUT_SECS:
                    return None

                # Stream: slide context window when full
                if input_patches.shape[1] >= PATCH_LENGTH * PATCH_SIZE:
                    abc_code = "".join(byte_list)
                    abc_lines = [l for l in abc_code.split("\n") if l.strip()]

                    tunebody_start = next(
                        (i for i, l in enumerate(abc_lines) if l.startswith("[r:") or l.startswith("[V:")),
                        None,
                    )
                    if tunebody_start is None or tunebody_start == len(abc_lines) - 1:
                        break

                    meta_part    = "\n".join(abc_lines[:tunebody_start]) + "\n"
                    tunebody_part = [l + "\n" for l in abc_lines[tunebody_start:]]

                    if cut_index is None:
                        cut_index = len(tunebody_part) // 2

                    abc_slice    = meta_part + "".join(tunebody_part[-cut_index:])
                    enc          = patchilizer.encode_generate(abc_slice)
                    flat         = [tok for patch in enc for tok in patch]
                    input_patches = torch.tensor([flat], device=device).reshape(1, -1)

    abc_text = "".join(byte_list)

    import re
    lines = [l for l in abc_text.split("\n") if l.strip()]
    tb_start = next((i for i, l in enumerate(lines) if l.startswith("[V:") or l.startswith("[r:")), None)
    if tb_start is None:
        return None

    if full_output:
        # Whole piece (metadata + tunebody), stream markers [r:N/M] stripped.
        return "\n".join(re.sub(r"^\[r:[^\]]*\]", "", l) for l in lines)

    # Tunebody only (everything from the first [V:/[r: line), [r:N/M] stripped.
    tunebody_lines = [re.sub(r"^\[r:[^\]]*\]", "", l) for l in lines[tb_start:]]
    return "\n".join(tunebody_lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",  default=CORPUS_OUTPUT_PATH)
    parser.add_argument("--weights", default=NOTAGEN_WEIGHTS_PATH)
    parser.add_argument("--samples", type=int, default=NUM_CORPUS_SAMPLES)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model      = load_notagen(args.weights, device)
    patchilizer = Patchilizer()

    prompts = collect_prompts(REAL_TRAIN_JSONL, max_prompts=args.samples)
    print(f"Collected {len(prompts)} unique metadata prompts")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    n_written = 0
    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, prompt in enumerate(prompts):
            completions = []
            for temp in GEN_TEMPERATURES:
                completion = generate_one(model, patchilizer, prompt, temp, device)
                if completion:
                    completions.append({"temperature": temp, "abc": completion})

            if len(completions) < 2:
                print(f"[{i+1}/{len(prompts)}] skipped (too few completions)")
                continue

            record = {
                "prompt":          prompt,
                "chosen":          completions[0]["abc"],   # lowest temperature
                "rejected":        completions[-1]["abc"],  # highest temperature
                "all_completions": completions,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
            print(f"[{i+1}/{len(prompts)}] wrote record {n_written}  "
                  f"(temps: {[c['temperature'] for c in completions]})")

    print(f"\nDone — wrote {n_written} records to {args.output}")


if __name__ == "__main__":
    main()
