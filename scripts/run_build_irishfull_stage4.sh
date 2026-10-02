#!/bin/bash
#SBATCH -p compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --exclude=compsci-cluster-fitz-38
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# Stage 4 ONLY of the full-Irishman build (run_build_irishfull.sh): synthetic
# generation + index refresh. Stages 1-3 (convert, seed, preprocess) completed in
# job 12084851 and their outputs are on disk; that job was cancelled mid-stage-4
# because generation slowed ~18x by 500k pieces (progressive GC degradation --
# generate_synthetic_motifs.py now disables automatic GC in the loop and logs its
# rate every 5000 sources). The generator wipes its own synth dir and rebuilds
# the synthetic index entries, preserving the real ones, so restarting is safe.

set -eo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
I=/usr/xtmp/cy232/accompgen/irishfull
PROC="$I/abcfiles_processed_v1"
SYNTH="$I/synthetic_motifs_v1"

MAIN_IDX="$REPO/data/abcfiles_processed_v1_irishfull.jsonl"
TRAIN_IDX="$REPO/data/abcfiles_processed_v1_irishfull_train.jsonl"
EVAL_IDX="$REPO/data/abcfiles_processed_v1_irishfull_eval.jsonl"

eval "$(mamba shell hook --shell bash)" 2>/dev/null || \
  source /home/users/cy232/miniforge3/etc/profile.d/conda.sh
mamba activate notagen 2>/dev/null || conda activate notagen

# Sanity: stage 1-3 outputs must exist before we regenerate synthetic data.
test -d "$PROC/C" || { echo "missing $PROC/C"; exit 1; }
test -s "$TRAIN_IDX" || { echo "missing $TRAIN_IDX"; exit 1; }
echo "pre-check OK: proc C files=$(find "$PROC/C" -name '*.abc' | wc -l), train idx=$(wc -l < "$TRAIN_IDX")"

echo "==================== Stage 4: generate synthetic from (Lieder + full Irish) ===================="
SYNTH_OUTPUT_DIR="$SYNTH" \
SYNTH_REAL_INDEX="$MAIN_IDX" \
SYNTH_TRAIN_INDEX="$TRAIN_IDX" \
SYNTH_EVAL_INDEX="$EVAL_IDX" \
SYNTH_EXTRA_SOURCES="$PROC" \
python "$REPO/data/generate_synthetic_motifs.py"

echo "==================== Build complete ===================="
echo "train total: $(wc -l < "$TRAIN_IDX")  synthetic: $(grep -c synthetic_motifs "$TRAIN_IDX")  real: $(grep -vc synthetic_motifs "$TRAIN_IDX")"
echo "eval  total: $(wc -l < "$EVAL_IDX")  synthetic: $(grep -c synthetic_motifs "$EVAL_IDX")  real: $(grep -vc synthetic_motifs "$EVAL_IDX")"
