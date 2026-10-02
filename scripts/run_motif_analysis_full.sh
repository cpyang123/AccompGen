#!/bin/bash
#SBATCH -p compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
set -eo pipefail
eval "$(mamba shell hook --shell bash)"
mamba activate notagen
cd notebook/
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=18000 \
    --output /usr/xtmp/cy232/accompgen/motifcheck_out/motif_analysis_full_executed_${SLURM_JOB_ID}.ipynb \
    motif_analysis_full.ipynb
echo "EXECUTED_NB=/usr/xtmp/cy232/accompgen/motifcheck_out/motif_analysis_full_executed_${SLURM_JOB_ID}.ipynb"
