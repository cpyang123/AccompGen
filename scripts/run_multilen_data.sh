#!/bin/bash
#SBATCH -p compsci
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# Build the multi-length (motif lengths 4-10) dataset end to end:
#   1. back up + clear the shared v1 jsonl indices (2_data_preprocess.py APPENDS
#      to existing train/eval indices, so stale single-length entries must go),
#   2. re-preprocess the Lieder corpus (7 length-variants per piece),
#   3. re-preprocess + merge the Irishman corpus (run_preprocess_irishman.sh),
#   4. regenerate the synthetic crop set (21 crops/piece) on scratch.

set -eo pipefail
# Under sbatch, BASH_SOURCE points at the slurm spool copy of this script, so
# resolve the repo from the submit directory instead.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

eval "$(mamba shell hook --shell bash)" 2>/dev/null || \
  source /home/users/cy232/miniforge3/etc/profile.d/conda.sh
mamba activate notagen 2>/dev/null || conda activate notagen

STAMP=$(date +%Y%m%d_%H%M%S)
for f in data/abcfiles_processed_v1.jsonl \
         data/abcfiles_processed_v1_train.jsonl \
         data/abcfiles_processed_v1_eval.jsonl; do
  if [ -f "$f" ]; then
    cp "$f" "$f.bak_multilen_$STAMP"
    rm "$f"
  fi
done
echo "Backed up + cleared v1 indices (suffix .bak_multilen_$STAMP)"

echo "=== [1/3] Lieder preprocess (multi-length) ==="
# The 7x multi-length output blew the home quota (Errno 122 on the 2026-08-13 run),
# so the Lieder augmented folder moves to scratch like the Irishman one; only the
# small jsonl indices stay in the repo.
LIEDER_OUT=/usr/xtmp/cy232/accompgen/lieder/abcfiles_processed_v1
mkdir -p "$LIEDER_OUT"
PREP_AUGMENTED_FOLDER="$LIEDER_OUT" \
PREP_APPEND_V1=0 \
PREP_MAIN_INDEX="$REPO/data/abcfiles_processed_v1.jsonl" \
PREP_TRAIN_INDEX="$REPO/data/abcfiles_processed_v1_train.jsonl" \
PREP_EVAL_INDEX="$REPO/data/abcfiles_processed_v1_eval.jsonl" \
PREP_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-8}" python data/2_data_preprocess.py

echo "=== [2/3] Irishman convert + preprocess + merge ==="
bash scripts/run_preprocess_irishman.sh

echo "=== [3/3] Synthetic multi-length crops ==="
# Primary source = the scratch Lieder build (NOT the stale single-length home copy);
# the Irishman scratch dir is picked up via the generator's default extra source.
SYNTH_PRIMARY_SOURCE="$LIEDER_OUT" python data/generate_synthetic_motifs.py

echo "=== Multi-length dataset build complete ==="
wc -l data/abcfiles_processed_v1.jsonl data/abcfiles_processed_v1_train.jsonl data/abcfiles_processed_v1_eval.jsonl
