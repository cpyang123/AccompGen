#!/bin/bash
# Full motif-data rebuild after the enharmonic-aware step/skip/leap fix:
#   1. Lieder preprocess (overwrite)   -> data/abcfiles_processed_v1 + fresh v1 indices
#   2. Irishman convert (20k tunes)    -> /usr/xtmp/.../irishman/abc
#   3. Irishman preprocess (append)    -> appends Irishman real entries to the v1 indices
#   4. Synthetic motif generation      -> top-3 motifs/piece, +-5 bars, incl Irishman
#
# Run directly (bash scripts/run_rebuild_motifs.sh). CPU-only; ~2 cores here so it is slow.
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XTMP=/usr/xtmp/cy232/accompgen/irishman
WORKERS="${PREP_NUM_WORKERS:-2}"
cd "$REPO"

eval "$(mamba shell hook --shell bash)" 2>/dev/null || \
  source /home/users/cy232/miniforge3/etc/profile.d/conda.sh
mamba activate notagen 2>/dev/null || conda activate notagen

TS=$(date +%Y%m%d_%H%M%S)
echo "=== [0/4] Back up + reset v1 indices ($TS) ==="
for f in abcfiles_processed_v1 abcfiles_processed_v1_train abcfiles_processed_v1_eval; do
  if [ -f "data/$f.jsonl" ]; then cp "data/$f.jsonl" "data/$f.jsonl.bak_rebuild_$TS"; fi
  rm -f "data/$f.jsonl"   # train/eval append-if-exists, so remove for a clean rebuild
done

echo "=== [1/4] Lieder preprocess (overwrite) — workers=$WORKERS ==="
rm -rf data/abcfiles_processed_v1
PREP_NUM_WORKERS="$WORKERS" python data/2_data_preprocess.py

echo "=== [2/4] Irishman convert (MAX_TUNES=20000) ==="
rm -rf "$XTMP/abc"
python data/irishman_to_abc.py

echo "=== [3/4] Irishman preprocess (append) — workers=$WORKERS ==="
rm -rf "$XTMP/abcfiles_processed_v1"
PREP_ORI_FOLDER="$XTMP/abc" \
PREP_INTERLEAVED_FOLDER="$XTMP/abcfiles_inter" \
PREP_AUGMENTED_FOLDER="$XTMP/abcfiles_processed_v1" \
PREP_APPEND_V1=0 \
PREP_INDEX_MODE=append \
PREP_MAIN_INDEX="$REPO/data/abcfiles_processed_v1.jsonl" \
PREP_TRAIN_INDEX="$REPO/data/abcfiles_processed_v1_train.jsonl" \
PREP_EVAL_INDEX="$REPO/data/abcfiles_processed_v1_eval.jsonl" \
PREP_NUM_WORKERS="$WORKERS" \
python data/2_data_preprocess.py

echo "=== [4/4] Synthetic motif generation (top-3, +-5 bars, incl Irishman) ==="
python data/generate_synthetic_motifs.py

echo "=== DONE ($(date +%Y%m%d_%H%M%S)) ==="
echo "Index line counts:"
wc -l data/abcfiles_processed_v1*.jsonl
