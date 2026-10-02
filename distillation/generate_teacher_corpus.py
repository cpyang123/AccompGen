"""
Prompt the frozen NotaGen teacher across every valid
Period_Composer_Instrumentation combination to synthesize new ABC pieces — extra
training data for distillation.

Reads the 112 combinations from gradio/prompts.txt and generates GEN_PER_COMBO
pieces per combination. Output: one .abc file per piece (full ABC: prompt
metadata + generated %%score/L/Q/M/K + tunebody) under GEN_CORPUS_DIR/<combo>/.

Generation is the bottleneck (autoregressive, seconds–minutes per piece), so this
is **shardable** for SLURM array jobs and **resumable** (existing files skipped).

Usage:
  python generate_teacher_corpus.py                  # all combos, GEN_PER_COMBO each
  python generate_teacher_corpus.py --shard 3/16     # this job does shard 3 of 16
  python generate_teacher_corpus.py --per-combo 50   # override count (smoke test)
"""

import os
import re
import sys
import time
import argparse

import torch

# Reuse the teacher loader + generation routine (importing runs generate_corpus's
# import bootstrap and leaves sys.modules['config'] = distillation/config.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_corpus as gc          # noqa: E402
import config as cfg                   # noqa: E402  (distillation config, cached by gc import)


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def load_combos(path: str):
    """Parse gradio/prompts.txt → list of (period, composer, instrumentation).
    Composer/instrumentation never contain '_', so a 3-way split is exact."""
    full = path if os.path.isabs(path) else os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), path))
    combos = []
    with open(full, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("_")
            if len(parts) != 3:
                print(f"  skipping malformed combo line: {line!r}")
                continue
            combos.append(tuple(parts))
    return combos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts",   default=cfg.PROMPTS_FILE)
    ap.add_argument("--out",       default=cfg.GEN_CORPUS_DIR)
    ap.add_argument("--per-combo", type=int, default=cfg.GEN_PER_COMBO)
    ap.add_argument("--shard",     default="0/1",
                    help="k/N: this job handles flat work items where idx %% N == k")
    ap.add_argument("--weights",   default=cfg.NOTAGEN_WEIGHTS_PATH)
    args = ap.parse_args()

    k, N = (int(x) for x in args.shard.split("/"))
    assert 0 <= k < N, f"bad shard {args.shard}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combos = load_combos(args.prompts)
    total = len(combos) * args.per_combo
    print(f"{len(combos)} combinations × {args.per_combo} = {total} target pieces | shard {k}/{N} | device {device}")

    model = gc.load_notagen(args.weights, device)
    patchilizer = gc.Patchilizer()
    temps = cfg.GEN_TEMPERATURES

    # Flat work list, then keep only this shard's items (round-robin).
    work = [(ci, pi) for ci in range(len(combos)) for pi in range(args.per_combo)]
    work = [w for i, w in enumerate(work) if i % N == k]
    os.makedirs(args.out, exist_ok=True)

    n_done = n_skip = n_fail = 0
    t0 = time.time()
    for ci, pi in work:
        period, composer, instrumentation = combos[ci]
        combo_dir = os.path.join(args.out, sanitize("_".join(combos[ci])))
        os.makedirs(combo_dir, exist_ok=True)
        out_path = os.path.join(combo_dir, f"{pi:05d}.abc")
        if os.path.exists(out_path):
            n_skip += 1
            continue

        prompt = f"%{period}\n%{composer}\n%{instrumentation}\n"
        temp = temps[pi % len(temps)]
        abc = gc.generate_one(model, patchilizer, prompt, temp, device, full_output=True)
        if abc is None:
            n_fail += 1
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(abc)
        n_done += 1
        if n_done % 20 == 0:
            rate = n_done / max(time.time() - t0, 1e-6)
            print(f"  shard {k}/{N}: {n_done} written, {n_skip} skipped, {n_fail} failed "
                  f"({rate:.2f} pieces/s, last={combos[ci]} t={temp})")

    print(f"Done shard {k}/{N}: wrote {n_done}, skipped {n_skip}, failed {n_fail} → {args.out}")


if __name__ == "__main__":
    main()
