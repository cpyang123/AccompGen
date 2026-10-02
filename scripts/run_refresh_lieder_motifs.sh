#!/bin/bash
#SBATCH -p compsci
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# Refresh ONLY the Lieder V:1 processed .abc files
#   data/abcfiles_processed_v1/<key>/lc*.abc
# with the enharmonic-aware step/skip/leap motif fix (motif/extract.py), WITHOUT
# rebuilding the Irishman-20k portion of the shared v1 indices.
#
# Why this is the minimal step for an irish50k-only rebuild:
#   * The jsonl indices store only {path,key} -- NO motif content -- and the
#     irish50k build (run_build_irish50k.sh, Stage 2) seeds its Lieder index lines
#     by grepping '/lc' from the EXISTING data/abcfiles_processed_v1*.jsonl. Those
#     {path,key} pairs are unchanged by re-preprocessing (same pieces, same keys).
#   * The motif annotations that DO change live inside the per-key .abc files, which
#     this regenerates in place.
#   * So we regenerate the .abc files but DIVERT the freshly written index to a temp
#     dir, leaving data/abcfiles_processed_v1*.jsonl (the 20k-mixed set) untouched.
#
# CPU-only; parallelizes across PREP_NUM_WORKERS. Run via:
#   sbatch scripts/run_refresh_lieder_motifs.sh    (or: bash scripts/run_refresh_lieder_motifs.sh)
set -eo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

eval "$(mamba shell hook --shell bash)" 2>/dev/null || \
  source /home/users/cy232/miniforge3/etc/profile.d/conda.sh
mamba activate notagen 2>/dev/null || conda activate notagen

WORKERS="${SLURM_CPUS_PER_TASK:-8}"
TMPIDX="$(mktemp -d)"
trap 'rm -rf "$TMPIDX"' EXIT

echo "=== Refresh Lieder motifs (workers=$WORKERS) ==="
echo "Throwaway index dir: $TMPIDX  (real v1 indices left untouched)"

# Defaults reproduce the original Lieder preprocessing exactly (ORI_FOLDER=abcfiles,
# AUGMENTED_FOLDER=abcfiles_processed_v1); only the index targets are redirected.
PREP_NUM_WORKERS="$WORKERS" \
PREP_MAIN_INDEX="$TMPIDX/v1.jsonl" \
PREP_TRAIN_INDEX="$TMPIDX/v1_train.jsonl" \
PREP_EVAL_INDEX="$TMPIDX/v1_eval.jsonl" \
python data/2_data_preprocess.py

n_c=$(ls data/abcfiles_processed_v1/C/lc*.abc 2>/dev/null | wc -l)
echo "=== Done. Lieder lc files in key C: $n_c ==="

# Sanity: every '/lc' entry the irish50k build will seed from must resolve to a file
# that now exists (so the rebuild has no dangling Lieder references).
python - "$REPO" <<'PY'
import json, os, sys
repo = sys.argv[1]
miss = total = 0
for split in ('train', 'eval'):
    idx = os.path.join(repo, f'data/abcfiles_processed_v1_{split}.jsonl')
    if not os.path.exists(idx):
        continue
    for line in open(idx):
        line = line.strip()
        if not line or '/lc' not in line:
            continue
        e = json.loads(line)
        total += 1
        # resolve to the original-key file the loader would read
        p, k = e['path'], e['key']
        # map mode->written key the same way the trainer does
        from abctoolkit.transpose import Key2Mode
        m2k = {m: kk for kk, ml in Key2Mode.items() for m in ml}
        wk = m2k.get(k, k)
        f = os.path.join(os.path.dirname(p), wk, f"{os.path.basename(p)}_{wk}.abc")
        if not os.path.exists(f):
            miss += 1
            if miss <= 5:
                print('  MISSING', f)
print(f"Lieder index references: {total} checked, {miss} missing.")
sys.exit(1 if miss else 0)
PY
echo "=== Lieder refresh validated OK ==="
