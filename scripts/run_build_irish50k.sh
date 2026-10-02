#!/bin/bash
#SBATCH -p compsci
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# Build an ISOLATED 50k-Irish V:1 dataset (real + synthetic) without touching the
# existing 20k data or its shared indices. Produces new indices:
#   data/abcfiles_processed_v1_irish50k_{train,eval}.jsonl  (+ _irish50k.jsonl main)
# containing: Lieder real + 50k Irish real + synthetic crops from (Lieder + 50k Irish).
#
# All heavy outputs land on scratch under /usr/xtmp/.../irish50k/ (isolated from the
# 20k set). The synthetic output dir name MUST contain 'synthetic_motifs' so both the
# index refresh and the train-time synthetic/real split classify it correctly.
#
# Stages:
#   1. irishman_to_abc.py  : raw JSON -> 50k normalized ABC          (IRISH_OUT_DIR)
#   2. seed indices        : copy Lieder (lc) real entries into the new indices
#   3. 2_data_preprocess   : 50k Irish ABC -> 15-key augmented, APPEND real entries
#   4. generate_synthetic  : crop synthetic from (Lieder + 50k Irish), refresh indices

set -eo pipefail

# Under sbatch the script is copied to a spool dir, so BASH_SOURCE is unreliable; SLURM
# runs with cwd = submit dir. Use SLURM_SUBMIT_DIR, falling back to cwd for `bash` runs.
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
I50K=/usr/xtmp/cy232/accompgen/irish50k
ABC="$I50K/abc"
INTER="$I50K/abcfiles_inter"
PROC="$I50K/abcfiles_processed_v1"
SYNTH="$I50K/synthetic_motifs_v1"          # must contain 'synthetic_motifs'

MAIN_IDX="$REPO/data/abcfiles_processed_v1_irish50k.jsonl"
TRAIN_IDX="$REPO/data/abcfiles_processed_v1_irish50k_train.jsonl"
EVAL_IDX="$REPO/data/abcfiles_processed_v1_irish50k_eval.jsonl"

# Lieder real entries already live in the existing v1 indices; reuse those processed
# files (data/abcfiles_processed_v1/<key>/lc*) — only their index lines are copied.
SRC_MAIN="$REPO/data/abcfiles_processed_v1.jsonl"
SRC_TRAIN="$REPO/data/abcfiles_processed_v1_train.jsonl"
SRC_EVAL="$REPO/data/abcfiles_processed_v1_eval.jsonl"

eval "$(mamba shell hook --shell bash)" 2>/dev/null || \
  source /home/users/cy232/miniforge3/etc/profile.d/conda.sh
mamba activate notagen 2>/dev/null || conda activate notagen

WORKERS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$ABC" "$INTER" "$PROC"

echo "==================== Stage 1: convert 50k Irish -> ABC ===================="
IRISH_MAX_TUNES=50000 IRISH_OUT_DIR="$ABC" python "$REPO/data/irishman_to_abc.py"
echo "Converted ABC files: $(ls "$ABC"/*.abc 2>/dev/null | wc -l)"

echo "==================== Stage 2: seed indices with Lieder (lc) entries ===================="
grep '/lc' "$SRC_MAIN"  > "$MAIN_IDX"
grep '/lc' "$SRC_TRAIN" > "$TRAIN_IDX"
grep '/lc' "$SRC_EVAL"  > "$EVAL_IDX"
echo "Seeded main=$(wc -l < "$MAIN_IDX") train=$(wc -l < "$TRAIN_IDX") eval=$(wc -l < "$EVAL_IDX") Lieder entries."

echo "==================== Stage 3: preprocess 50k Irish (append real entries) ===================="
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

echo "==================== Stage 4: generate synthetic from (Lieder + 50k Irish) ===================="
# Base source (Lieder) is data/abcfiles_processed_v1; add the 50k Irish processed dir.
SYNTH_OUTPUT_DIR="$SYNTH" \
SYNTH_REAL_INDEX="$MAIN_IDX" \
SYNTH_TRAIN_INDEX="$TRAIN_IDX" \
SYNTH_EVAL_INDEX="$EVAL_IDX" \
SYNTH_EXTRA_SOURCES="$PROC" \
python "$REPO/data/generate_synthetic_motifs.py"

echo "==================== Build complete ===================="
echo "train total: $(wc -l < "$TRAIN_IDX")  synthetic: $(grep -c synthetic "$TRAIN_IDX")  irish: $(grep -ic irish "$TRAIN_IDX")  lieder: $(grep -c '/lc' "$TRAIN_IDX")"
echo "eval  total: $(wc -l < "$EVAL_IDX")  synthetic: $(grep -c synthetic "$EVAL_IDX")  irish: $(grep -ic irish "$EVAL_IDX")  lieder: $(grep -c '/lc' "$EVAL_IDX")"
