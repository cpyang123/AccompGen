#!/bin/bash
#SBATCH -p compsci
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# Build an ISOLATED full-Irishman V:1 dataset (real + synthetic): ALL ~216k tunes
# (IRISH_MAX_TUNES=300000 > corpus size, so nothing is capped). Same pipeline as
# run_build_irish50k.sh, landing in irishfull/ scratch dirs and new indices:
#   data/abcfiles_processed_v1_irishfull_{train,eval}.jsonl
# Composition: Lieder real (current-logic labels, refreshed Jun 30) + ~213k Irish
# real + synthetic crops from (Lieder + full Irish).
#
# Scale estimates from the 50k build (46 min preprocess, ~2h synth at 50k):
# ~2.5-3.5h preprocess (16 workers) + ~8h synthetic generation (serial) ≈ 11-12h.

set -eo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
I=/usr/xtmp/cy232/accompgen/irishfull
ABC="$I/abc"
INTER="$I/abcfiles_inter"
PROC="$I/abcfiles_processed_v1"
SYNTH="$I/synthetic_motifs_v1"          # must contain 'synthetic_motifs'

MAIN_IDX="$REPO/data/abcfiles_processed_v1_irishfull.jsonl"
TRAIN_IDX="$REPO/data/abcfiles_processed_v1_irishfull_train.jsonl"
EVAL_IDX="$REPO/data/abcfiles_processed_v1_irishfull_eval.jsonl"

SRC_MAIN="$REPO/data/abcfiles_processed_v1.jsonl"
SRC_TRAIN="$REPO/data/abcfiles_processed_v1_train.jsonl"
SRC_EVAL="$REPO/data/abcfiles_processed_v1_eval.jsonl"

eval "$(mamba shell hook --shell bash)" 2>/dev/null || \
  source /home/users/cy232/miniforge3/etc/profile.d/conda.sh
mamba activate notagen 2>/dev/null || conda activate notagen

WORKERS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$ABC" "$INTER" "$PROC"

echo "==================== Stage 1: convert ALL Irishman tunes -> ABC ===================="
IRISH_MAX_TUNES=300000 IRISH_OUT_DIR="$ABC" python "$REPO/data/irishman_to_abc.py"
echo "Converted ABC files: $(find "$ABC" -name '*.abc' | wc -l)"

echo "==================== Stage 2: seed indices with Lieder (lc) entries ===================="
grep '/lc' "$SRC_MAIN"  > "$MAIN_IDX"
grep '/lc' "$SRC_TRAIN" > "$TRAIN_IDX"
grep '/lc' "$SRC_EVAL"  > "$EVAL_IDX"
echo "Seeded main=$(wc -l < "$MAIN_IDX") train=$(wc -l < "$TRAIN_IDX") eval=$(wc -l < "$EVAL_IDX") Lieder entries."

echo "==================== Stage 3: preprocess full Irishman (append real entries) ===================="
PREP_ORI_FOLDER="$ABC" \
PREP_INTERLEAVED_FOLDER="$INTER" \
PREP_AUGMENTED_FOLDER="$PROC" \
PREP_APPEND_V1=0 \
PREP_INDEX_MODE=append \
PREP_MAIN_INDEX="$MAIN_IDX" \
PREP_TRAIN_INDEX="$TRAIN_IDX" \
PREP_EVAL_INDEX="$EVAL_IDX" \
PREP_NUM_WORKERS="$WORKERS" \
python "$REPO/data/2_data_preprocess.py"
echo "After Irish merge: main=$(wc -l < "$MAIN_IDX") train=$(wc -l < "$TRAIN_IDX") eval=$(wc -l < "$EVAL_IDX")"

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
