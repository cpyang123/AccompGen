#!/bin/bash
#SBATCH -p compsci
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# Build an ISOLATED 1k-piece V:1 dataset: 800 Lieder pieces (seeded random
# subsample of the 1,217) + the first 200 Irishman tunes (nested within the
# 10k/50k/full subsets). Mirrors run_build_irish50k.sh. Produces:
#   data/abcfiles_processed_v1_irish1k_{train,eval}.jsonl (+ main)
# Heavy outputs land under /usr/xtmp/cy232/accompgen/irish1k/.

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
I1K=/usr/xtmp/cy232/accompgen/irish1k
ABC="$I1K/abc"
INTER="$I1K/abcfiles_inter"
PROC="$I1K/abcfiles_processed_v1"
SYNTH="$I1K/synthetic_motifs_v1"           # must contain 'synthetic_motifs'

MAIN_IDX="$REPO/data/abcfiles_processed_v1_irish1k.jsonl"
TRAIN_IDX="$REPO/data/abcfiles_processed_v1_irish1k_train.jsonl"
EVAL_IDX="$REPO/data/abcfiles_processed_v1_irish1k_eval.jsonl"

SRC_MAIN="$REPO/data/abcfiles_processed_v1.jsonl"
SRC_TRAIN="$REPO/data/abcfiles_processed_v1_train.jsonl"
SRC_EVAL="$REPO/data/abcfiles_processed_v1_eval.jsonl"

eval "$(mamba shell hook --shell bash)" 2>/dev/null || \
  source /home/users/cy232/miniforge3/etc/profile.d/conda.sh
mamba activate notagen 2>/dev/null || conda activate notagen

WORKERS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$ABC" "$INTER" "$PROC"

echo "============ Stage 1: convert first 200 Irish tunes -> ABC ============"
IRISH_MAX_TUNES=200 IRISH_OUT_DIR="$ABC" python "$REPO/data/irishman_to_abc.py"
echo "Converted ABC files: $(ls "$ABC"/*.abc 2>/dev/null | wc -l)"

echo "============ Stage 2: seed indices with 800 subsampled Lieder pieces ============"
SRC_MAIN="$SRC_MAIN" SRC_TRAIN="$SRC_TRAIN" SRC_EVAL="$SRC_EVAL" \
MAIN_IDX="$MAIN_IDX" TRAIN_IDX="$TRAIN_IDX" EVAL_IDX="$EVAL_IDX" \
python - <<'PY'
import json, os, random
random.seed(20260803)
src_main = os.environ['SRC_MAIN']
pieces = sorted({json.loads(l)['path'].rsplit('/', 1)[-1]
                 for l in open(src_main) if '/lc' in l})
keep = set(random.sample(pieces, 800))
print(f'Lieder pieces: {len(pieces)}, keeping {len(keep)}')
for s, d in (('SRC_MAIN', 'MAIN_IDX'), ('SRC_TRAIN', 'TRAIN_IDX'),
             ('SRC_EVAL', 'EVAL_IDX')):
    n = 0
    with open(os.environ[d], 'w') as f:
        for l in open(os.environ[s]):
            if '/lc' in l and json.loads(l)['path'].rsplit('/', 1)[-1] in keep:
                f.write(l); n += 1
    print(f'{d}: {n} Lieder rows')
PY

echo "============ Stage 3: preprocess 200 Irish (append real entries) ============"
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

echo "============ Stage 4: synthetic crops from the 1k pool ============"
SYNTH_OUTPUT_DIR="$SYNTH" \
SYNTH_REAL_INDEX="$MAIN_IDX" \
SYNTH_TRAIN_INDEX="$TRAIN_IDX" \
SYNTH_EVAL_INDEX="$EVAL_IDX" \
SYNTH_EXTRA_SOURCES="$PROC" \
python "$REPO/data/generate_synthetic_motifs.py"

echo "============ Build complete ============"
echo "train total: $(wc -l < "$TRAIN_IDX")  synthetic: $(grep -c synthetic "$TRAIN_IDX")  irish: $(grep -ic irish "$TRAIN_IDX")  lieder: $(grep -c '/lc' "$TRAIN_IDX")"
echo "eval  total: $(wc -l < "$EVAL_IDX")  synthetic: $(grep -c synthetic "$EVAL_IDX")  irish: $(grep -ic irish "$EVAL_IDX")  lieder: $(grep -c '/lc' "$EVAL_IDX")"
