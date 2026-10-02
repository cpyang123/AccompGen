"""Cached patch/character decoding with the original trained motif attention bias."""
from dataclasses import dataclass
import gzip
import json
import os
import re
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
from safetensors.torch import load_file
from samplings import top_k_sampling, top_p_sampling, temperature_sampling

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
from notagen_core import NotaGenLMHeadModel, build_notagen_configs, safe_normalize_probs

PATCH_SIZE = 16
MAX_CONTEXT = 1024


def realization_parts(line):
    """Match finetune/utils.py Patchilizer.patchilize_metadata/split_bars.

    Realization tags occupy their own patches; notes use training's bar splits.
    In particular, the final non-voice segment is joined to the preceding bar.
    """
    tag = re.match(r"^(%motif:(?:rhythm:)?abc(?::[a-z_]+)?: )", line)
    if not tag:
        return [line]
    prefix = tag.group(1)
    delimiters = ("|:", "::", ":|", "[|", "||", "|]", "|")
    pieces = [part for part in re.split("(" + "|".join(map(re.escape, delimiters)) + ")", line[len(prefix):]) if part]
    if len(pieces) <= 1:
        return [prefix, *pieces]
    start = 0 if pieces[0] in delimiters else 1
    bars = pieces[:start] + ["".join(pieces[i:i + 2]) for i in range(start, len(pieces), 2)]
    if len(bars) > 1 and "V" not in bars[-1]:
        bars[-2:] = [bars[-2] + bars[-1]]
    return [prefix, *bars]


def encode_prompt(text):
    patches, flags = [[1] * 15 + [2]], [False]
    for line in text.splitlines(keepends=True):
        for part in realization_parts(line):
            ids = list(part.encode("ascii"))
            if len(ids) % PATCH_SIZE:
                ids.append(2)
            for start in range(0, len(ids), PATCH_SIZE):
                patch = ids[start:start + PATCH_SIZE]
                patches.append(patch + [0] * (PATCH_SIZE - len(patch)))
                flags.append(line.startswith("%motif:"))
    return patches, flags


def decode_patch(ids):
    result = []
    for token in ids:
        if token == 2:
            break
        if token >= 32 or token in (9, 10, 13):
            result.append(chr(token))
    return "".join(result)


def load_model():
    manifest = json.loads((ROOT / "checkpoint.json").read_text())
    weights = Path(os.environ.get("ACCOMPGEN_CHECKPOINT", ROOT / manifest.get("export_filename", "model.safetensors")))
    if not weights.exists():
        raise FileNotFoundError("No checkpoint found. Run prepare.py before launching the demo.")
    configs = build_notagen_configs(**manifest["architecture"])
    if weights.name.endswith(".safetensors.gz"):
        with tempfile.TemporaryDirectory(prefix="motigen-model-") as temporary:
            unpacked = Path(temporary) / "model.safetensors"
            with gzip.open(weights, "rb") as source, unpacked.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            state = load_file(str(unpacked))
    elif weights.suffix == ".safetensors":
        state = load_file(str(weights))
    else:
        state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)["model"]
    # Meta initialization avoids allocating a second 2 GB set of parameters.
    with torch.device("meta"):
        model = NotaGenLMHeadModel(encoder_config=configs[0], decoder_config=configs[1])
    model.load_state_dict(state, strict=True, assign=True)
    # GPT2 causal masks are non-persistent buffers, absent from the state dict.
    for module in model.modules():
        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor) and module.bias.is_meta:
            shape = module.bias.shape
            module.bias = torch.tril(torch.ones(shape[-2:], dtype=torch.bool)).view(shape)
        if hasattr(module, "masked_bias") and module.masked_bias.is_meta:
            module.masked_bias = torch.tensor(-1e4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device, dtype=torch.float16 if device == "cuda" else torch.float32).eval()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    return model


class CachedDecoder:
    """Per-request KV state; never shared between visitors."""
    def __init__(self, model):
        self.model = model
        self.past = None

    def encode(self, patches, motif_flags):
        m = self.model
        encoder = m.patch_level_decoder
        ids = torch.tensor([patches], dtype=torch.long, device=m.device)
        embeds = torch.nn.functional.one_hot(ids, num_classes=128).to(m.dtype)
        embeds = encoder.patch_embedding(embeds.reshape(1, -1, 16 * 128))
        bias = torch.tensor(motif_flags, device=m.device, dtype=m.dtype) * m.motif_attention_bias
        encoder._motif_bias = bias[None, None, None, :]
        try:
            output = encoder.base(inputs_embeds=embeds, attention_mask=torch.ones(1, len(motif_flags), device=m.device),
                                  past_key_values=self.past, use_cache=True)
        finally:
            encoder._motif_bias = None
        self.past = output.past_key_values
        return m.patch_proj(output.last_hidden_state[0, -1])

    def patch(self, encoded, rng, temperature=1.2, top_k=9, top_p=0.9, prefix=()):
        base = self.model.char_level_decoder.base
        embeds = encoded.reshape(1, 1, -1)
        if prefix:
            tokens = torch.tensor([prefix], device=self.model.device)
            embeds = torch.cat((embeds, base.get_input_embeddings()(tokens)), dim=1)
        past, result = None, list(prefix)
        while len(result) < PATCH_SIZE:
            output = base(inputs_embeds=embeds, past_key_values=past, use_cache=True)
            past = output.past_key_values
            probs = safe_normalize_probs(torch.softmax(output.logits[0, -1].float(), dim=-1).cpu().numpy())
            probs = safe_normalize_probs(top_k_sampling(probs, top_k=top_k, return_probs=True))
            probs = safe_normalize_probs(top_p_sampling(probs, top_p=top_p, return_probs=True))
            # samplings' temperature transform, sampled with a request-local RNG.
            probs = safe_normalize_probs(temperature_sampling(probs, temperature=temperature, return_probs=True))
            token = int(rng.choice(len(probs), p=probs))
            result.append(token)
            if token == 2:
                result.extend([0] * (PATCH_SIZE - len(result)))
                break
            embeds = base.get_input_embeddings()(torch.tensor([[token]], device=self.model.device))
        return result


@dataclass
class GenerationUpdate:
    text: str
    patches: int
    elapsed: float
    reason: str = ""


def generate(model, prompt, *, temperature=1.2, seed=0, max_bars=16, max_seconds=100, max_patches=800):
    patches, flags = encode_prompt(prompt)
    rng = np.random.default_rng(seed)
    decoder = CachedDecoder(model)
    text, line, body = prompt, "", False
    started, last_yield = time.monotonic(), 0.0
    reason = "Patch limit reached; showing the completed measures."
    with torch.inference_mode():
        encoded = decoder.encode(patches, flags)
        for index in range(max_patches):
            ids = decoder.patch(encoded, rng, temperature)
            fragment = decode_patch(ids)
            if not body and fragment.startswith("[r:"):
                ids = decoder.patch(encoded, rng, temperature, prefix=tuple(b"[r:0/"))
                fragment = decode_patch(ids)
                body = True
            if ids[:2] == [1, 2]:
                reason = "Complete."
                break
            text += fragment
            flags.append((line + fragment).lstrip().startswith("%motif:"))
            line = (line + fragment).rsplit("\n", 1)[-1]
            patches.append(ids)
            elapsed = time.monotonic() - started
            if elapsed - last_yield >= 0.4:
                yield GenerationUpdate(text, index + 1, elapsed)
                last_yield = elapsed
            # Stream-format body lines correspond to complete score measures.
            body_lines = [s for s in text.splitlines(keepends=True) if s.startswith("[r:") and s.endswith("\n")]
            if len(body_lines) >= max_bars:
                reason = f"Finished {max_bars} measures."
                break
            if elapsed >= max_seconds:
                reason = "Time limit reached; showing the completed measures."
                break
            if len(patches) >= MAX_CONTEXT:
                reason = "Context limit reached; showing the completed measures."
                break
            encoded = decoder.encode([ids], flags)
        yield GenerationUpdate(text, index + 1, time.monotonic() - started, reason)
