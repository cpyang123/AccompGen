"""
Two-phase fine-tuning of Qwen2.5-3B-Instruct for ABC music generation.

  Phase 1 — SFT on the NotaGen distillation corpus (general music knowledge).
  Phase 2 — Motif-conditioned fine-tuning on the real + synthetic AccompGen data.

Usage:
  # Phase 1
  python train_qwen.py --phase sft

  # Phase 2 (starts from the SFT checkpoint)
  python train_qwen.py --phase motif

  # Both phases in sequence
  python train_qwen.py --phase both
"""

import os
import sys
import json
import time
import random
import argparse
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_constant_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    QWEN_MODEL_ID,
    QWEN_SFT_OUTPUT_PATH, QWEN_MOTIF_OUTPUT_PATH,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    CORPUS_OUTPUT_PATH,
    REAL_TRAIN_JSONL, REAL_EVAL_JSONL,
    SFT_BATCH_SIZE, SFT_ACCUMULATION_STEPS, SFT_LR, SFT_EPOCHS,
    SFT_WARMUP_STEPS, SFT_MAX_SEQ_LEN,
    MOTIF_BATCH_SIZE, MOTIF_ACCUMULATION_STEPS, MOTIF_LR,
    MOTIF_EPOCHS_SYNTHETIC, MOTIF_EPOCHS_REAL,
    MOTIF_WARMUP_STEPS, MOTIF_MAX_SEQ_LEN,
)

# Needed by inference/utils.py which is imported transitively via finetune for Phase 2
sys.path.insert(0, os.path.join(_REPO_ROOT, "finetune"))

try:
    from abctoolkit.transpose import Key2index, Key2Mode
    Index2Key = {v: k for k, v in Key2index.items() if v not in [1, 11]}
    Mode2Key  = {mode: key for key, modes in Key2Mode.items() for mode in modes}
    _ABC_TOOLKIT = True
except ImportError:
    _ABC_TOOLKIT = False

SYSTEM_PROMPT = (
    "You are a music composition assistant that generates polyphonic ABC notation scores. "
    "Given a header describing period, composer, instrumentation and optionally a motivic pattern, "
    "generate the complete tunebody in ABC notation."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def split_abc(abc_text: str) -> tuple[str, str]:
    """Return (metadata_header, tunebody) for an ABC string."""
    lines = [l for l in abc_text.split("\n") if l.strip()]
    tb_start = next(
        (i for i, l in enumerate(lines) if l.startswith("[V:") or l.startswith("[r:")),
        None,
    )
    if tb_start is None:
        return abc_text, ""
    header   = "\n".join(lines[:tb_start])
    tunebody = "\n".join(lines[tb_start:])
    return header, tunebody


def make_chat_text(tokenizer, header: str, tunebody: str) -> dict:
    """Build input_ids and labels with loss masked on the prompt portion."""
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": header},
        {"role": "assistant", "content": tunebody},
    ]
    # apply_chat_template returns a single string; we need token-level label masking.
    prompt_messages = messages[:-1]
    full_text   = tokenizer.apply_chat_template(messages,        tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    full_ids   = tokenizer(full_text,   add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "labels": labels}


def collate_fn(batch, pad_id: int, max_len: int):
    input_ids_list = [b["input_ids"][:max_len] for b in batch]
    labels_list    = [b["labels"][:max_len]    for b in batch]
    max_seq = max(len(x) for x in input_ids_list)

    input_ids = torch.zeros(len(batch), max_seq, dtype=torch.long)
    labels    = torch.full((len(batch), max_seq), -100, dtype=torch.long)
    attn_mask = torch.zeros(len(batch), max_seq, dtype=torch.long)

    for i, (ids, lbl) in enumerate(zip(input_ids_list, labels_list)):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids)
        labels[i, :n]    = torch.tensor(lbl)
        attn_mask[i, :n] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}


# ── Phase 1 dataset ───────────────────────────────────────────────────────────

class DistilDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.records   = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                # Use the chosen (lowest-temperature) completion for SFT
                self.records.append((obj["prompt"], obj["chosen"]))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        header, tunebody = self.records[idx]
        return make_chat_text(self.tokenizer, header, tunebody)


# ── Phase 2 dataset ───────────────────────────────────────────────────────────

class MotifDataset(Dataset):
    def __init__(self, entries: list[dict], tokenizer, max_len: int):
        self.entries   = entries
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry    = self.entries[idx]
        filepath = entry["path"]

        if _ABC_TOOLKIT:
            ori_key   = Mode2Key[entry["key"]]
            ori_idx   = Key2index[ori_key]
            offsets   = range(-3, 4)
            probs     = [1/16, 2/16, 3/16, 4/16, 3/16, 2/16, 1/16]
            r         = random.random()
            cumsum    = 0
            des_idx   = ori_idx
            for offset, p in zip(offsets, probs):
                cumsum += p
                if r < cumsum:
                    des_idx = (ori_idx + offset) % 12
                    break
            if des_idx == 1:
                des_key = "Db" if random.random() < 0.8 else "C#"
            elif des_idx == 11:
                des_key = "B"  if random.random() < 0.8 else "Cb"
            elif des_idx == 6:
                des_key = "F#" if random.random() < 0.5 else "Gb"
            else:
                des_key = Index2Key[des_idx]
        else:
            des_key = entry["key"]

        folder    = os.path.dirname(filepath)
        name      = os.path.basename(filepath)
        abc_path  = os.path.join(folder, des_key, f"{name}_{des_key}.abc")

        with open(abc_path, encoding="utf-8") as f:
            abc_text = f.read()

        header, tunebody = split_abc(abc_text)
        return make_chat_text(self.tokenizer, header, tunebody)


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, scaler, scheduler, device, accumulation_steps, is_train: bool):
    model.train(is_train)
    total_loss = 0.0
    n_batches  = 0
    pbar = tqdm(loader)

    for step, batch in enumerate(pbar, 1):
        input_ids  = batch["input_ids"].to(device)
        labels     = batch["labels"].to(device)
        attn_mask  = batch["attention_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out  = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss / accumulation_steps

        if is_train:
            scaler.scale(loss).backward()
            if step % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        total_loss += loss.item() * accumulation_steps
        n_batches  += 1
        pbar.set_postfix({"loss": f"{total_loss / n_batches:.4f}"})

    return total_loss / max(n_batches, 1)


def save_lora(model, path: str):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    print(f"Saved LoRA adapter to {path}")


# ── Phase 1: SFT ─────────────────────────────────────────────────────────────

def phase1_sft(args):
    print("\n" + "="*60)
    print("Phase 1 — SFT on distillation corpus")
    print("="*60)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    corpus_path = args.corpus or CORPUS_OUTPUT_PATH
    dataset = DistilDataset(corpus_path, tokenizer, SFT_MAX_SEQ_LEN)
    loader  = DataLoader(
        dataset,
        batch_size=SFT_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id, SFT_MAX_SEQ_LEN),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=SFT_LR)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=SFT_WARMUP_STEPS)
    scaler    = torch.cuda.amp.GradScaler()

    for epoch in range(1, SFT_EPOCHS + 1):
        loss = run_epoch(model, loader, optimizer, scaler, scheduler, device,
                         SFT_ACCUMULATION_STEPS, is_train=True)
        print(f"Epoch {epoch}/{SFT_EPOCHS}  train_loss={loss:.4f}")

    save_lora(model, QWEN_SFT_OUTPUT_PATH)


# ── Phase 2: Motif fine-tuning ────────────────────────────────────────────────

def phase2_motif(args):
    print("\n" + "="*60)
    print("Phase 2 — Motif fine-tuning")
    print("="*60)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Start from the SFT checkpoint if it exists, otherwise from base
    sft_path = args.sft_checkpoint or QWEN_SFT_OUTPUT_PATH
    if os.path.isdir(sft_path):
        print(f"Loading SFT checkpoint from {sft_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, sft_path, is_trainable=True)
    else:
        print("No SFT checkpoint found — fine-tuning from base model")
        base_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_cfg)

    model.print_trainable_parameters()

    with open(REAL_TRAIN_JSONL, encoding="utf-8") as f:
        all_train = [json.loads(l) for l in f]
    with open(REAL_EVAL_JSONL, encoding="utf-8") as f:
        all_eval  = [json.loads(l) for l in f]

    synthetic_train = [e for e in all_train if "synthetic" in e["path"]]
    real_train      = [e for e in all_train if "synthetic" not in e["path"]]

    optimizer = torch.optim.AdamW(model.parameters(), lr=MOTIF_LR)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=MOTIF_WARMUP_STEPS)
    scaler    = torch.cuda.amp.GradScaler()

    def make_loader(entries, shuffle=True):
        ds = MotifDataset(entries, tokenizer, MOTIF_MAX_SEQ_LEN)
        return DataLoader(
            ds,
            batch_size=MOTIF_BATCH_SIZE,
            shuffle=shuffle,
            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id, MOTIF_MAX_SEQ_LEN),
        )

    eval_loader = make_loader(all_eval, shuffle=False)

    # Phase 2a: synthetic data curriculum
    print(f"\n--- Phase 2a: synthetic ({len(synthetic_train)} samples, {MOTIF_EPOCHS_SYNTHETIC} epochs)")
    train_loader = make_loader(synthetic_train)
    best_eval, best_epoch = 1e9, 0
    for epoch in range(1, MOTIF_EPOCHS_SYNTHETIC + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, scheduler, device,
                               MOTIF_ACCUMULATION_STEPS, is_train=True)
        eval_loss  = run_epoch(model, eval_loader,  optimizer, scaler, scheduler, device,
                               1, is_train=False)
        print(f"  Synthetic epoch {epoch}  train={train_loss:.4f}  eval={eval_loss:.4f}")
        if eval_loss < best_eval:
            best_eval, best_epoch = eval_loss, epoch
            save_lora(model, QWEN_MOTIF_OUTPUT_PATH + "_best_synthetic")

    # Load best synthetic checkpoint before real phase
    print(f"\nLoading best synthetic checkpoint (epoch {best_epoch}, eval {best_eval:.4f})")
    model = PeftModel.from_pretrained(
        base_model, QWEN_MOTIF_OUTPUT_PATH + "_best_synthetic", is_trainable=True
    )

    # Phase 2b: real data
    print(f"\n--- Phase 2b: real ({len(real_train)} samples, {MOTIF_EPOCHS_REAL} epochs)")
    train_loader = make_loader(real_train)
    real_eval_entries = [e for e in all_eval if "synthetic" not in e["path"]] or all_eval
    eval_loader  = make_loader(real_eval_entries, shuffle=False)
    best_eval, best_epoch = 1e9, 0

    for epoch in range(1, MOTIF_EPOCHS_REAL + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, scheduler, device,
                               MOTIF_ACCUMULATION_STEPS, is_train=True)
        eval_loss  = run_epoch(model, eval_loader,  optimizer, scaler, scheduler, device,
                               1, is_train=False)
        print(f"  Real epoch {epoch}  train={train_loss:.4f}  eval={eval_loss:.4f}")
        if eval_loss < best_eval:
            best_eval, best_epoch = eval_loss, epoch
            save_lora(model, QWEN_MOTIF_OUTPUT_PATH)

    print(f"\nBest real epoch: {best_epoch}  eval loss: {best_eval:.4f}")
    print(f"Final adapter saved to {QWEN_MOTIF_OUTPUT_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",          choices=["sft", "motif", "both"], default="both")
    parser.add_argument("--corpus",         default=None, help="Path to distillation corpus JSONL")
    parser.add_argument("--sft-checkpoint", default=None, dest="sft_checkpoint")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.phase in ("sft", "both"):
        phase1_sft(args)
    if args.phase in ("motif", "both"):
        phase2_motif(args)


if __name__ == "__main__":
    main()
