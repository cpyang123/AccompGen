"""
Stage 1 — patch-encoder feature distillation.

Distill the frozen GPT-2 patch-level encoder of the pre-trained NotaGen teacher
into a LARGER student encoder, then (in Stage 2) reuse the original, frozen
char-level decoder. Because the patch encoder's output is a continuous per-patch
vector — not a token distribution — there is no tokenizer mismatch and no KL is
required: we match hidden states directly with MSE + cosine (FitNets-style).

Pipeline
--------
  precompute : run the frozen teacher over the corpus once and cache, per patch,
               the input bytes + the teacher's `last_hidden_state` to /usr/xtmp.
  train      : train `student_encoder + projection` to reproduce those cached
               teacher features over valid patch positions. Optionally add an
               end-to-end char-LM term that flows through the frozen char decoder.

Usage
-----
  python distill_patch_encoder.py --stage precompute
  python distill_patch_encoder.py --stage train
  python distill_patch_encoder.py --stage both

The student outputs a vector of dim HIDDEN_SIZE (1280) so it drops straight into
the frozen char decoder for Stage 2 — see notes at the bottom of this file.
"""

import os
import sys
import glob
import json
import time
import random
import argparse
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── Import bootstrap ──────────────────────────────────────────────────────────
# Load distillation/config.py explicitly, then clear it from sys.modules so the
# finetune Patchilizer can import its own `config` (which has MOTIF_LOSS_WEIGHT
# etc.) without a name collision.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

sys.path.insert(0, _HERE)
dcfg = importlib.import_module("config")          # distillation/config.py
sys.path.remove(_HERE)
sys.modules.pop("config", None)

# Reuse the exact training-time tokenizer (motif-aware Patchilizer). Its utils
# does `from config import *` against finetune/config — supply that on the path.
_FINETUNE = os.path.join(_REPO, "finetune")
sys.path.insert(0, _FINETUNE)
Patchilizer = importlib.import_module("utils").Patchilizer
sys.path.remove(_FINETUNE)

from notagen_core import (build_notagen_configs, make_encoder_config,
                          PatchLevelDecoder, NotaGenLMHeadModel)
from abctoolkit.transpose import Key2index, Key2Mode

Index2Key = {index: key for key, index in Key2index.items() if index not in [1, 11]}
Mode2Key  = {mode: key for key, ml in Key2Mode.items() for mode in ml}

PATCH_SIZE  = dcfg.PATCH_SIZE
HIDDEN_SIZE = dcfg.HIDDEN_SIZE   # teacher hidden = char-decoder input dim = projection target


# ── Path / data helpers ───────────────────────────────────────────────────────

def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(_HERE, path))


# All transposed keys present on disk for each piece (15 per file). Used to
# enumerate DISTINCT transpositions per augmentation pass (rather than the
# ±3-semitone random draw used at finetune time), so N passes give N distinct
# (patches, teacher_feats) datapoints instead of duplicates.
CANON_KEYS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "Db", "Ab", "Eb", "Bb", "F", "Gb", "Cb"]


def keys_for_entry(entry: dict) -> list[str]:
    """Keys to use across passes for one piece: original key first (pass 0, so it
    reuses the existing cache), then the remaining keys in a fixed order. Missing
    key files are skipped downstream by load_patches."""
    ori_key = Mode2Key[entry["key"]]
    return [ori_key] + [k for k in CANON_KEYS if k != ori_key]


def load_patches(entry: dict, key: str, patchilizer: Patchilizer):
    """Read the transposed ABC, strip motif lines (the teacher never saw them),
    and encode to patch byte-ids [L, PATCH_SIZE]. Returns None on failure."""
    folder = os.path.dirname(entry["path"])
    name   = os.path.basename(entry["path"])
    abc_path = os.path.join(folder, key, f"{name}_{key}.abc")
    if not os.path.exists(abc_path):
        return None
    with open(abc_path, encoding="utf-8") as f:
        text = f.read()
    text = "\n".join(l for l in text.split("\n") if not l.startswith("%motif:"))
    try:
        id_patches, _weights = patchilizer.encode_train(text)
    except Exception:
        return None
    if not id_patches:
        return None
    return torch.tensor(id_patches, dtype=torch.long)   # [L, PATCH_SIZE]


def load_entries(jsonl_path: str) -> list[dict]:
    with open(_resolve(jsonl_path), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_patches_file(abc_path: str, patchilizer: Patchilizer):
    """Encode a standalone ABC file (e.g. a teacher-generated piece) to patch
    byte-ids [L, PATCH_SIZE]. Returns None on failure."""
    try:
        with open(abc_path, encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return None
    text = "\n".join(l for l in text.split("\n") if not l.startswith("%motif:"))
    try:
        id_patches, _weights = patchilizer.encode_train(text)
    except Exception:
        return None
    if not id_patches:
        return None
    return torch.tensor(id_patches, dtype=torch.long)


# ── Teacher ───────────────────────────────────────────────────────────────────

def load_teacher(device):
    patch_config, char_config = build_notagen_configs(
        encoder_backbone="gpt2", decoder_backbone="gpt2",
        patch_num_layers=dcfg.PATCH_NUM_LAYERS, char_num_layers=dcfg.CHAR_NUM_LAYERS,
        hidden_size=HIDDEN_SIZE, patch_length=dcfg.PATCH_LENGTH, patch_size=PATCH_SIZE,
        motif_attention_bias=0.0, patch_sampling_batch_size=0)
    model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=char_config)
    ckpt = torch.load(dcfg.NOTAGEN_WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"])     # strict — verified key-compatible
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded frozen teacher from {dcfg.NOTAGEN_WEIGHTS_PATH}")
    return model


# ── Phase A: precompute teacher features → /usr/xtmp cache ────────────────────

class EntryDataset(Dataset):
    def __init__(self, entries, patchilizer, pass_idx):
        self.entries = entries
        self.patchilizer = patchilizer
        self.pass_idx = pass_idx

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        keys = keys_for_entry(e)
        if self.pass_idx >= len(keys):     # piece has fewer distinct keys than passes
            return idx, None
        patches = load_patches(e, keys[self.pass_idx], self.patchilizer)
        return idx, patches


def _precompute_collate(batch):
    items = [(i, p) for i, p in batch if p is not None]
    if not items:
        return None
    idxs = [i for i, _ in items]
    lens = [p.shape[0] for _, p in items]
    maxL = max(lens)
    B = len(items)
    patches = torch.zeros(B, maxL, PATCH_SIZE, dtype=torch.long)
    masks   = torch.zeros(B, maxL, dtype=torch.long)
    for k, (_, p) in enumerate(items):
        L = p.shape[0]
        patches[k, :L] = p
        masks[k, :L] = 1
    return idxs, lens, patches, masks


def precompute_split(teacher, entries, split, device, patchilizer, passes):
    out_dir = os.path.join(dcfg.STAGE1_CACHE_DIR, split)
    os.makedirs(out_dir, exist_ok=True)
    n_written = 0
    for p in range(passes):
        ds = EntryDataset(entries, patchilizer, pass_idx=p)
        dl = DataLoader(ds, batch_size=dcfg.STAGE1_PRECOMPUTE_BATCH,
                        shuffle=False, collate_fn=_precompute_collate, num_workers=2)
        for batch in tqdm(dl, desc=f"precompute {split} pass {p}"):
            if batch is None:
                continue
            idxs, lens, patches, masks = batch
            # Skip work if every item in this batch is already cached
            targets = [os.path.join(out_dir, f"{i:06d}_p{p}.pt") for i in idxs]
            if all(os.path.exists(t) for t in targets):
                continue
            patches_d = patches.to(device)
            masks_d   = masks.to(device)
            with torch.inference_mode():
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    feats = teacher.patch_level_decoder(patches_d, masks_d)["last_hidden_state"]
            feats = feats.float().cpu()
            for k, (i, L, tgt) in enumerate(zip(idxs, lens, targets)):
                if os.path.exists(tgt):
                    continue
                torch.save({
                    "patches": patches[k, :L].to(torch.int16),       # [L, PATCH_SIZE]
                    "feats":   feats[k, :L].to(torch.float16),       # [L, HIDDEN_SIZE]
                }, tgt)
                n_written += 1
    print(f"[{split}] cached {n_written} new samples to {out_dir}")


class GenFileDataset(Dataset):
    """Teacher-generated ABC pieces (flat .abc files), encoded to patches."""
    def __init__(self, files, patchilizer):
        self.files = files
        self.patchilizer = patchilizer

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return idx, load_patches_file(self.files[idx], self.patchilizer)


def precompute_generated(teacher, device, patchilizer):
    """Encode teacher-generated pieces from GEN_CORPUS_DIR and cache their teacher
    features into the *train* split, so the student distills on them too. Cache
    files are named gen_<idx>.pt (idx into the sorted file list, so it's stable
    and resumable); real-data shards (NNNNNN_pN.pt) never collide."""
    gen_dir = dcfg.GEN_CORPUS_DIR
    if not getattr(dcfg, "STAGE1_USE_GEN_CORPUS", False):
        return
    if not os.path.isdir(gen_dir):
        print(f"[generated] STAGE1_USE_GEN_CORPUS is on but {gen_dir} does not exist — skipping")
        return
    files = sorted(glob.glob(os.path.join(gen_dir, "**", "*.abc"), recursive=True))
    if not files:
        print(f"[generated] no .abc files under {gen_dir} — run scripts/gen_corpus.sh first")
        return

    out_dir = os.path.join(dcfg.STAGE1_CACHE_DIR, "train")
    os.makedirs(out_dir, exist_ok=True)
    ds = GenFileDataset(files, patchilizer)
    dl = DataLoader(ds, batch_size=dcfg.STAGE1_PRECOMPUTE_BATCH,
                    shuffle=False, collate_fn=_precompute_collate, num_workers=2)
    n_written = 0
    for batch in tqdm(dl, desc=f"precompute generated ({len(files)} files)"):
        if batch is None:
            continue
        idxs, lens, patches, masks = batch
        targets = [os.path.join(out_dir, f"gen_{i:07d}.pt") for i in idxs]
        if all(os.path.exists(t) for t in targets):
            continue
        patches_d = patches.to(device)
        masks_d   = masks.to(device)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                feats = teacher.patch_level_decoder(patches_d, masks_d)["last_hidden_state"]
        feats = feats.float().cpu()
        for k, (i, L, tgt) in enumerate(zip(idxs, lens, targets)):
            if os.path.exists(tgt):
                continue
            torch.save({
                "patches": patches[k, :L].to(torch.int16),
                "feats":   feats[k, :L].to(torch.float16),
            }, tgt)
            n_written += 1
    print(f"[generated] cached {n_written} new samples (of {len(files)} files) to {out_dir}")


# ── Phase B: student + distillation training ──────────────────────────────────

class StudentPatchEncoder(nn.Module):
    """Larger patch encoder + linear projection to the teacher's hidden dim, so
    its output drops into the frozen char decoder unchanged."""

    def __init__(self):
        super().__init__()
        cfg = make_encoder_config(
            dcfg.STUDENT_ENCODER_BACKBONE,
            dcfg.STUDENT_PATCH_NUM_LAYERS, dcfg.STUDENT_HIDDEN_SIZE, dcfg.PATCH_LENGTH,
            patch_size=PATCH_SIZE, motif_attention_bias=0.0)
        self.encoder = PatchLevelDecoder(cfg)
        self.proj = nn.Linear(dcfg.STUDENT_HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, patches, masks):
        h = self.encoder(patches, masks)["last_hidden_state"]   # [B, seq, student_hidden]
        return self.proj(h)                                     # [B, seq, HIDDEN_SIZE]


class FeatCacheDataset(Dataset):
    def __init__(self, split):
        self.files = sorted(glob.glob(os.path.join(dcfg.STAGE1_CACHE_DIR, split, "*.pt")))
        if not self.files:
            raise FileNotFoundError(
                f"No cached features in {os.path.join(dcfg.STAGE1_CACHE_DIR, split)}. "
                f"Run with --stage precompute first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location="cpu")
        return d["patches"].long(), d["feats"].float()


def _train_collate(batch):
    lens = [p.shape[0] for p, _ in batch]
    maxL = max(lens)
    B = len(batch)
    patches = torch.zeros(B, maxL, PATCH_SIZE, dtype=torch.long)
    feats   = torch.zeros(B, maxL, HIDDEN_SIZE, dtype=torch.float)
    masks   = torch.zeros(B, maxL, dtype=torch.long)
    for k, (p, f) in enumerate(batch):
        L = p.shape[0]
        patches[k, :L] = p
        feats[k, :L]   = f
        masks[k, :L]   = 1
    return patches, feats, masks


def feature_loss(student_feats, teacher_feats, masks):
    sel = masks.bool()
    s = student_feats[sel].float()
    t = teacher_feats[sel].float()
    mse = F.mse_loss(s, t)
    cos = (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()
    return mse + dcfg.STAGE1_COS_WEIGHT * cos, mse.detach(), cos.detach()


def char_lm_loss(char_decoder, encoded, patches, masks):
    """End-to-end term: feed student features through the frozen char decoder,
    replicating NotaGenLMHeadModel.forward's shift/masking."""
    m = masks.clone()
    left_shift = m * (m.flip(1).cumsum(1).flip(1) > 1)
    m2 = m.clone()
    m2[:, 0] = 0
    enc = encoded[left_shift == 1]
    tgt = patches[m2 == 1]
    return char_decoder(enc, tgt).loss


def run_epoch(student, char_decoder, loader, optimizer, scheduler, device, is_train):
    student.train(is_train)
    totals = {"loss": 0.0, "mse": 0.0, "cos": 0.0, "lm": 0.0}
    n = 0
    pbar = tqdm(loader, desc="train" if is_train else "eval")
    for step, (patches, feats, masks) in enumerate(pbar, 1):
        patches = patches.to(device); feats = feats.to(device); masks = masks.to(device)
        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                student_feats = student(patches, masks)
                floss, mse, cos = feature_loss(student_feats, feats, masks)
                loss = floss
                lm = torch.zeros((), device=device)
                if dcfg.STAGE1_LAMBDA_LM > 0 and char_decoder is not None:
                    lm = char_lm_loss(char_decoder, student_feats, patches, masks)
                    loss = loss + dcfg.STAGE1_LAMBDA_LM * lm
        if is_train:
            (loss / dcfg.STAGE1_ACCUM_STEPS).backward()
            if step % dcfg.STAGE1_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        totals["loss"] += float(loss); totals["mse"] += float(mse)
        totals["cos"] += float(cos);  totals["lm"] += float(lm)
        n += 1
        pbar.set_postfix({k: f"{v / n:.4f}" for k, v in totals.items()})
    return {k: v / max(n, 1) for k, v in totals.items()}


def save_student(student, path, meta):
    os.makedirs(path, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(path, "student_encoder.pt"))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved student encoder → {path}")


def train(device):
    from transformers import get_constant_schedule_with_warmup

    train_loader = DataLoader(FeatCacheDataset("train"), batch_size=dcfg.STAGE1_BATCH_SIZE,
                              shuffle=True, collate_fn=_train_collate, num_workers=2)
    try:
        eval_loader = DataLoader(FeatCacheDataset("eval"), batch_size=dcfg.STAGE1_BATCH_SIZE,
                                 shuffle=False, collate_fn=_train_collate, num_workers=2)
    except FileNotFoundError:
        eval_loader = None
        print("No eval cache found — training without eval.")

    student = StudentPatchEncoder().to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student encoder: {dcfg.STUDENT_ENCODER_BACKBONE} "
          f"L{dcfg.STUDENT_PATCH_NUM_LAYERS} h{dcfg.STUDENT_HIDDEN_SIZE} "
          f"→ proj {HIDDEN_SIZE} | params={n_params:,}")

    char_decoder = None
    if dcfg.STAGE1_LAMBDA_LM > 0:
        char_decoder = load_teacher(device).char_level_decoder   # frozen
        print(f"End-to-end char-LM term ON (lambda={dcfg.STAGE1_LAMBDA_LM})")

    optimizer = torch.optim.AdamW(student.parameters(), lr=dcfg.STAGE1_LR)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=dcfg.STAGE1_WARMUP_STEPS)

    meta = {
        "student_backbone": dcfg.STUDENT_ENCODER_BACKBONE,
        "student_layers": dcfg.STUDENT_PATCH_NUM_LAYERS,
        "student_hidden": dcfg.STUDENT_HIDDEN_SIZE,
        "proj_out": HIDDEN_SIZE,
        "patch_size": PATCH_SIZE,
    }
    best = float("inf")
    for epoch in range(1, dcfg.STAGE1_EPOCHS + 1):
        tr = run_epoch(student, char_decoder, train_loader, optimizer, scheduler, device, True)
        msg = f"Epoch {epoch}/{dcfg.STAGE1_EPOCHS}  train loss={tr['loss']:.4f} mse={tr['mse']:.4f} cos={tr['cos']:.4f}"
        if eval_loader is not None:
            ev = run_epoch(student, char_decoder, eval_loader, optimizer, scheduler, device, False)
            msg += f"  | eval loss={ev['loss']:.4f} mse={ev['mse']:.4f} cos={ev['cos']:.4f}"
            score = ev["loss"]
        else:
            score = tr["loss"]
        print(msg)
        if score < best:
            best = score
            save_student(student, dcfg.STAGE1_OUTPUT_DIR, {**meta, "epoch": epoch, "score": score})
    print(f"\nDone. Best score: {best:.4f}  Student at {dcfg.STAGE1_OUTPUT_DIR}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["precompute", "train", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if args.stage in ("precompute", "both"):
        patchilizer = Patchilizer()
        teacher = load_teacher(device)
        precompute_split(teacher, load_entries(dcfg.REAL_TRAIN_JSONL), "train", device, patchilizer, dcfg.STAGE1_PASSES)
        precompute_split(teacher, load_entries(dcfg.REAL_EVAL_JSONL),  "eval",  device, patchilizer, 1)
        precompute_generated(teacher, device, patchilizer)   # teacher-generated pieces → train split
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.stage in ("train", "both"):
        train(device)


if __name__ == "__main__":
    main()

# ── Stage 2 hand-off ──────────────────────────────────────────────────────────
# The trained student outputs HIDDEN_SIZE (1280) vectors, so to assemble the full
# upgraded model you load StudentPatchEncoder, take the frozen teacher's
# char_level_decoder, and run NotaGenLMHeadModel.forward's contract:
#   encoded = student(patches, masks)                # [B, seq, 1280]
#   <left-shift/mask exactly as in model.py>         # then char_decoder(enc, tgt)
# Stage 2 then unfreezes the student (char decoder optionally frozen) and
# fine-tunes end-to-end on real + motif data to actually exceed the teacher.
