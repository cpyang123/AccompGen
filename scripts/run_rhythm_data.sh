#!/bin/bash
#SBATCH -p compsci
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# Build the RHYTHM-format multi-length dataset (2026-09-22 header: melodic motif
# + rhythmic motif of the same length, see motif/rhythm.py and data/README.md)
# as an ISOLATED copy. Same composition as the inversion-format set the current
# checkpoints were trained on (Lieder + the 20k-tune Irishman conversion, no style
# preamble), so a rhythm-trained model is directly comparable to the `inv` run:
#   real      -> $RHY_ROOT/{lieder,irishman}/abcfiles_processed_v1
#   synthetic -> $RHY_ROOT/synthetic_motifs_multilen_v1
#   indices   -> $RHY_IDX_DIR/abcfiles_processed_v1_$RHY_IDX_TAG{,_train,_eval}.jsonl
# Nothing under the existing lieder/, irishman/ or synthetic_motifs_multilen_v1
# scratch dirs or the shared v1 indices is touched.
#
# Smoke run (tiny inputs, scratch output):
#   RHY_ROOT=/tmp/x RHY_IDX_DIR=/tmp/x RHY_LIEDER_ORI=<2 files> RHY_IRISH_ORI=<2 files> bash scripts/run_rhythm_data.sh
# Real run: sbatch scripts/run_rhythm_data.sh   (~20 min Lieder + ~15 min Irish preprocess
# at 16 workers, then 1-2 h serial synthetic generation).

set -eo pipefail
# Repo root: the script's own dir when run directly; under sbatch BASH_SOURCE is
# the slurm spool copy, so fall back to the submit dir (submit from the repo root).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO/data/2_data_preprocess.py" ] || REPO="$SLURM_SUBMIT_DIR"
[ -f "$REPO/data/2_data_preprocess.py" ] || { echo "cannot locate repo root (REPO=$REPO)"; exit 1; }
cd "$REPO"

RHY_ROOT="${RHY_ROOT:-/usr/xtmp/cy232/accompgen/rhythm}"
RHY_IDX_DIR="${RHY_IDX_DIR:-$REPO/data}"
RHY_IDX_TAG="${RHY_IDX_TAG:-rhythm}"
RHY_LIEDER_ORI="${RHY_LIEDER_ORI:-$REPO/data/abcfiles}"
RHY_IRISH_ORI="${RHY_IRISH_ORI:-/usr/xtmp/cy232/accompgen/irishman/abc}"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

LIEDER_OUT="$RHY_ROOT/lieder/abcfiles_processed_v1"
IRISH_OUT="$RHY_ROOT/irishman/abcfiles_processed_v1"
SYNTH_OUT="$RHY_ROOT/synthetic_motifs_multilen_v1"   # must contain 'synthetic_motifs'
MAIN_IDX="$RHY_IDX_DIR/abcfiles_processed_v1_${RHY_IDX_TAG}.jsonl"
TRAIN_IDX="$RHY_IDX_DIR/abcfiles_processed_v1_${RHY_IDX_TAG}_train.jsonl"
EVAL_IDX="$RHY_IDX_DIR/abcfiles_processed_v1_${RHY_IDX_TAG}_eval.jsonl"

eval "$(mamba shell hook --shell bash)" 2>/dev/null || \
  source /home/users/cy232/miniforge3/etc/profile.d/conda.sh
mamba activate notagen 2>/dev/null || conda activate notagen

test -d "$RHY_LIEDER_ORI" || { echo "missing Lieder source $RHY_LIEDER_ORI"; exit 1; }
test -d "$RHY_IRISH_ORI"  || { echo "missing Irish source $RHY_IRISH_ORI"; exit 1; }
mkdir -p "$LIEDER_OUT" "$IRISH_OUT" "$RHY_ROOT/lieder/abcfiles_inter" "$RHY_ROOT/irishman/abcfiles_inter" "$RHY_IDX_DIR"
# Fresh indices: the preprocess APPENDS to existing train/eval indices.
rm -f "$MAIN_IDX" "$TRAIN_IDX" "$EVAL_IDX"
echo "root=$RHY_ROOT idx=$MAIN_IDX workers=$WORKERS lieder_src=$RHY_LIEDER_ORI irish_src=$RHY_IRISH_ORI"

echo "==================== [1/4] Lieder preprocess (rhythm headers) ===================="
PREP_ORI_FOLDER="$RHY_LIEDER_ORI" \
PREP_INTERLEAVED_FOLDER="$RHY_ROOT/lieder/abcfiles_inter" \
PREP_AUGMENTED_FOLDER="$LIEDER_OUT" \
PREP_APPEND_V1=0 \
PREP_INDEX_MODE=overwrite \
PREP_MAIN_INDEX="$MAIN_IDX" PREP_TRAIN_INDEX="$TRAIN_IDX" PREP_EVAL_INDEX="$EVAL_IDX" \
PREP_NUM_WORKERS="$WORKERS" \
python data/2_data_preprocess.py
echo "after Lieder: main=$(wc -l < "$MAIN_IDX") train=$(wc -l < "$TRAIN_IDX") eval=$(wc -l < "$EVAL_IDX")"

echo "==================== [2/4] Irishman preprocess (append) ===================="
PREP_ORI_FOLDER="$RHY_IRISH_ORI" \
PREP_INTERLEAVED_FOLDER="$RHY_ROOT/irishman/abcfiles_inter" \
PREP_AUGMENTED_FOLDER="$IRISH_OUT" \
PREP_APPEND_V1=0 \
PREP_INDEX_MODE=append \
PREP_MAIN_INDEX="$MAIN_IDX" PREP_TRAIN_INDEX="$TRAIN_IDX" PREP_EVAL_INDEX="$EVAL_IDX" \
PREP_NUM_WORKERS="$WORKERS" \
python data/2_data_preprocess.py
echo "after Irish: main=$(wc -l < "$MAIN_IDX") train=$(wc -l < "$TRAIN_IDX") eval=$(wc -l < "$EVAL_IDX")"

echo "==================== [3/4] Synthetic crops (Lieder + Irish) ===================="
SYNTH_PRIMARY_SOURCE="$LIEDER_OUT" \
SYNTH_EXTRA_SOURCES="$IRISH_OUT" \
SYNTH_OUTPUT_DIR="$SYNTH_OUT" \
SYNTH_REAL_INDEX="$MAIN_IDX" SYNTH_TRAIN_INDEX="$TRAIN_IDX" SYNTH_EVAL_INDEX="$EVAL_IDX" \
python data/generate_synthetic_motifs.py

echo "==================== [4/4] Verify ===================="
# No `ls | head` under pipefail (SIGPIPE would fail the job after the work is done).
RHY_LIEDER_OUT="$LIEDER_OUT" RHY_IRISH_OUT="$IRISH_OUT" RHY_SYNTH_OUT="$SYNTH_OUT" \
RHY_TRAIN_IDX="$TRAIN_IDX" RHY_EVAL_IDX="$EVAL_IDX" python - <<'PY'
import os, json, sys, random
random.seed(0)
dirs = {k: os.environ[k] for k in ('RHY_LIEDER_OUT', 'RHY_IRISH_OUT', 'RHY_SYNTH_OUT')}
bad = 0
for label, d in dirs.items():
    key_dir = os.path.join(d, 'C')
    with os.scandir(key_dir) as it:
        files = [e.path for e in it if e.name.endswith('.abc')]
    sample = random.sample(files, min(400, len(files)))
    with_mel = with_rhy = with_both = non_ascii = 0
    for p in sample:
        raw = open(p, 'rb').read()
        if any(b > 126 for b in raw[:400]):
            non_ascii += 1
        head = raw.decode('utf-8', 'replace').split('[V:', 1)[0]
        m = '%motif:v1:step_skip_leap:' in head
        r = '%motif:v1:rhythm:' in head and '%motif:rhythm:count:' in head and '%motif:rhythm:abc:' in head
        with_mel += m; with_rhy += r; with_both += (m and r)
        if m != r:
            bad += 1
            if bad <= 3: print('MISMATCHED BLOCKS:', p)
    print(f'{label}: {len(files)} files in C; sampled {len(sample)}: melodic {with_mel}, rhythm {with_rhy}, '
          f'both {with_both}, non-ascii {non_ascii}')
    if label == 'RHY_SYNTH_OUT' and with_both != len(sample):
        print('synthetic crops must all carry both blocks'); bad += 1
    if non_ascii:
        bad += 1
    print('--- example', sample[0]); print(open(sample[0], encoding='utf-8').read().split('\n[V:', 1)[0])
for k in ('RHY_TRAIN_IDX', 'RHY_EVAL_IDX'):
    n = syn = 0
    with open(os.environ[k]) as fh:
        for line in fh:
            if not line.strip(): continue
            e = json.loads(line); n += 1; syn += 'synthetic_motifs' in e['path']
            if n <= 3:
                p = os.path.join(e['path'].rsplit('/', 1)[0], 'C', os.path.basename(e['path']) + '_C.abc')
                assert os.path.exists(p), p
    print(f'{k}: {n} entries, {syn} synthetic, {n - syn} real')
    if n == 0: bad += 1
print('VERIFY', 'FAILED' if bad else 'OK'); sys.exit(1 if bad else 0)
PY
echo "==================== Build complete ($(date)) ===================="
