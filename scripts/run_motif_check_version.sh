#!/bin/bash
# Headless runner for the curated motif-checker versions.
# Usage:  sbatch scripts/run_motif_check_version.sh <MULTIVOICE|V1_BIAS2|V1_BIAS4|V1_BIAS8>
# Executed copy (with all cell outputs) lands in /usr/xtmp/cy232/accompgen/motifcheck_out/.
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/motifcheck_version-%j.out
#SBATCH --error=slurm_logs/motifcheck_version-%j.err

set -eo pipefail
VERSION="${1:?usage: sbatch scripts/run_motif_check_version.sh <MULTIVOICE|V1_BIAS2|V1_BIAS4|V1_BIAS8>}"
NB="motif_check_${VERSION}.ipynb"

eval "$(mamba shell hook --shell bash)"
mamba activate notagen

cd notebook/
test -f "$NB" || { echo "no such notebook: $NB"; exit 1; }
jupyter nbconvert --to notebook --execute --allow-errors \
    --ExecutePreprocessor.timeout=4800 \
    --output "/usr/xtmp/cy232/accompgen/motifcheck_out/${VERSION}_executed_${SLURM_JOB_ID}.ipynb" \
    "$NB"

echo "EXECUTED_NB=/usr/xtmp/cy232/accompgen/motifcheck_out/${VERSION}_executed_${SLURM_JOB_ID}.ipynb"
