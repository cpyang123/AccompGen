#!/bin/bash
# Submit the per-length stratified motif-containment eval for the two multi-length
# models: 7 lengths x 3 strata (8 motifs x 50 pieces each) + 1 no-prompt base run
# (50 pieces) per model = 44 jobs. Results land under
#   /usr/xtmp/cy232/accompgen/strat_eval_multilen/<tag>/{len<L>_<S>,base}/results.jsonl
# and are aggregated by experiments/make_multilen_strat_results.py.
set -eo pipefail
cd "$(dirname "$0")/.."

W=/usr/xtmp/cy232/accompgen/weights/weights_notagen_
EVAL_ROOT="${EVAL_ROOT:-/usr/xtmp/cy232/accompgen/strat_eval_multilen}"
SUF=_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1.pth
declare -A MODELS=(
  [trans]="${W}multilen4to10_trans_v1_bias4_syn2real5${SUF}"   # I/R/RI-folded dataset
  [multilen]="${W}multilen4to10_v1_bias4_syn2real5${SUF}"      # baseline multilen
  [inv]="${W}multilen4to10_inv_v1_bias4_syn2real5${SUF}"       # inversion-format full FT (2026-09-16, best real epoch 3)
  [rhythm]="${W}multilen4to10_rhythm_v1_bias4_syn2real5${SUF}" # rhythm-format full FT (2026-09-24, best real epoch 3)
)
# Optional extra sbatch flags for every job, e.g. EXTRA_SBATCH="--dependency=afterok:123 --kill-on-invalid-dep=yes"
EXTRA_SBATCH="${EXTRA_SBATCH:-}"

TAGS="${1:-trans multilen}"    # optionally restrict: ./submit_strat_eval_multilen.sh trans
# Optional inference-time motif attention bias override: EVAL_BIAS=5 ./submit... inv
# -> results under <tag>_bias5/ so the default-bias run is untouched.
EVAL_BIAS="${EVAL_BIAS:-}"
# Optional extra bias on the abstract pattern line only: EVAL_PATTERN_EXTRA=2 -> tag suffix _pat2
EVAL_PATTERN_EXTRA="${EVAL_PATTERN_EXTRA:-}"
# Optional rhythm conditioning (rhythm-format checkpoints): EVAL_RHYTHM=1 ./submit... rhythm
# -> every prompt also carries the paired %motif:v1:rhythm: line from notebook/rhythm_sets_multilen.json,
#    results under <tag>_rc/ (melodic-only prompts stay under <tag>/ for comparison with inv/multilen).
EVAL_RHYTHM="${EVAL_RHYTHM:-}"
RHYTHM_SETS="$(pwd)/notebook/rhythm_sets_multilen.json"
if [ -n "$EVAL_RHYTHM" ]; then
  [ -f "$RHYTHM_SETS" ] || { echo "missing $RHYTHM_SETS (run notebook/make_rhythm_sets_multilen.py)"; exit 1; }
fi
for model in $TAGS; do
  weights="${MODELS[$model]}"
  tag="$model${EVAL_BIAS:+_bias$EVAL_BIAS}${EVAL_PATTERN_EXTRA:+_pat$EVAL_PATTERN_EXTRA}${EVAL_RHYTHM:+_rc}"
  [ -f "$weights" ] || { echo "missing weights for $model: $weights"; exit 1; }
  # pre-create all output dirs: concurrent first-starters otherwise race on mkdir over NFS
  for L in 4 5 6 7 8 9 10; do for S in A B C; do mkdir -p "$EVAL_ROOT/$tag/len${L}_$S"; done; done
  mkdir -p "$EVAL_ROOT/$tag/base"
  for L in 4 5 6 7 8 9 10; do
    for S in A B C; do
      jid=$(sbatch --parsable $EXTRA_SBATCH -J "se_${tag}_l${L}${S}" \
        --export=ALL,EVAL_TAG=$tag,EVAL_WEIGHTS=$weights,EVAL_LENGTH=$L,EVAL_STRATUM=$S${EVAL_BIAS:+,EVAL_BIAS=$EVAL_BIAS}${EVAL_PATTERN_EXTRA:+,EVAL_PATTERN_EXTRA=$EVAL_PATTERN_EXTRA}${EVAL_RHYTHM:+,EVAL_RHYTHM_SETS=$RHYTHM_SETS} \
        scripts/run_strat_eval_multilen.sh)
      echo "$tag len$L $S -> $jid"
    done
  done
  jid=$(sbatch --parsable $EXTRA_SBATCH -J "se_${tag}_base" \
    --export=ALL,EVAL_TAG=$tag,EVAL_WEIGHTS=$weights,EVAL_STRATUM=base${EVAL_BIAS:+,EVAL_BIAS=$EVAL_BIAS}${EVAL_PATTERN_EXTRA:+,EVAL_PATTERN_EXTRA=$EVAL_PATTERN_EXTRA}${EVAL_RHYTHM:+,EVAL_RHYTHM_SETS=$RHYTHM_SETS} \
    scripts/run_strat_eval_multilen.sh)
  echo "$tag base -> $jid"
done
