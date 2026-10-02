#!/bin/bash
#SBATCH -p compsci
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# CPU-only job (no torch/GPU): preprocessing is the bottleneck and parallelizes
# across PREP_NUM_WORKERS. Runtime ~= n_tunes * 0.63s / workers.
#
# Convert the Irishman dataset to ABC and preprocess it into the V:1 (melody)
# training set, MERGING its entries into the existing Lieder v1 indices.
#
# Steps:
#   1. data/irishman_to_abc.py        : raw train/validation JSON  -> normalized ABC
#   2. data/2_data_preprocess.py      : ABC -> 15-key augmented + motif lines, and
#                                       APPENDS entries to data/abcfiles_processed_v1*.jsonl
#
# Raw JSON, normalized ABC, and the 15-key augmented output all live on /usr/xtmp
# (scratch) to stay under the home-dir quota -- only the shared .jsonl indices
# (small) live in the repo. Run on slurm (sbatch scripts/run_preprocess_irishman.sh) or
# directly (bash scripts/run_preprocess_irishman.sh); 216k tunes is a multi-hour job.

set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO/data/2_data_preprocess.py" ] || REPO="$SLURM_SUBMIT_DIR"
[ -f "$REPO/data/2_data_preprocess.py" ] || { echo "cannot locate repo root (REPO=$REPO)"; exit 1; }
XTMP=/usr/xtmp/cy232/accompgen/irishman

eval "$(mamba shell hook --shell bash)" 2>/dev/null || \
  source /home/users/cy232/miniforge3/etc/profile.d/conda.sh
mamba activate notagen 2>/dev/null || conda activate notagen

# 1) Download raw JSON if missing, then convert to normalized ABC.
mkdir -p "$XTMP/raw"
python - <<PY
import os, urllib.request
base="https://huggingface.co/datasets/sander-wood/irishman/resolve/main/"
for f in ("train.json","validation.json"):
    out=os.path.join("$XTMP/raw", f)
    if not (os.path.exists(out) and os.path.getsize(out)>1000):
        print("downloading", f); urllib.request.urlretrieve(base+f, out)
    else:
        print("have", f)
PY
python "$REPO/data/irishman_to_abc.py"

# 2) Preprocess the Irishman ABC into the V:1 set, appending to the Lieder v1 index.
PREP_ORI_FOLDER="$XTMP/abc" \
PREP_INTERLEAVED_FOLDER="$XTMP/abcfiles_inter" \
PREP_AUGMENTED_FOLDER="$XTMP/abcfiles_processed_v1" \
PREP_APPEND_V1=0 \
PREP_INDEX_MODE=append \
PREP_MAIN_INDEX="$REPO/data/abcfiles_processed_v1.jsonl" \
PREP_TRAIN_INDEX="$REPO/data/abcfiles_processed_v1_train.jsonl" \
PREP_EVAL_INDEX="$REPO/data/abcfiles_processed_v1_eval.jsonl" \
PREP_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-2}" \
python "$REPO/data/2_data_preprocess.py"

echo "Done. Irishman entries appended to $REPO/data/abcfiles_processed_v1_{train,eval}.jsonl"
