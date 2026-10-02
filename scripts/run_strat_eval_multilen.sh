#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=compsci-cluster-fitz-02   # stale /usr/xtmp view: every job landing there died on mkdir (2026-09-08)
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/strateval-%x-%j.out
#SBATCH --error=slurm_logs/strateval-%x-%j.err
#
# One (model, length, stratum) run of the per-length stratified motif-containment
# eval (experiments/multilen_strat_eval.py). Parameters come from the environment:
#   EVAL_TAG      short model tag -> output under $EVAL_ROOT/$EVAL_TAG/len<L>_<S>/
#   EVAL_WEIGHTS  checkpoint path
#   EVAL_LENGTH   motif length 4..10 (unused for STRATUM=base)
#   EVAL_STRATUM  A | B | C | base
#   EVAL_N        pieces per motif (default 50)
#   EVAL_BIAS     motif attention bias at inference (default 4.0 = training value)
#   EVAL_PATTERN_EXTRA  extra bias on the abstract %motif:v1: line only, added to EVAL_BIAS (default 0)
#   EVAL_RHYTHM_SETS    rhythm sets json -> rhythm-conditioned prompts + rhythm/joint scoring (rhythm ckpts)
# Submit the whole batch with submit_strat_eval_multilen.sh.

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

eval "$(mamba shell hook --shell bash)"
mamba activate notagen

EVAL_ROOT="${EVAL_ROOT:-/usr/xtmp/cy232/accompgen/strat_eval_multilen}"
: "${EVAL_TAG:?}" "${EVAL_WEIGHTS:?}" "${EVAL_STRATUM:?}"
LEN_ARG=""
if [ "$EVAL_STRATUM" != "base" ]; then
  : "${EVAL_LENGTH:?}"
  LEN_ARG="--length $EVAL_LENGTH"
fi

python experiments/multilen_strat_eval.py \
  --weights "$EVAL_WEIGHTS" --stratum "$EVAL_STRATUM" $LEN_ARG \
  --n "${EVAL_N:-50}" --bias "${EVAL_BIAS:-4.0}" --pattern-bias-extra "${EVAL_PATTERN_EXTRA:-0}" \
  ${EVAL_RHYTHM_SETS:+--rhythm-sets "$EVAL_RHYTHM_SETS"} --out-dir "$EVAL_ROOT/$EVAL_TAG"
