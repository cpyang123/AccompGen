#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/motifcheck_ablv2_nosyn10k_r023-%j.out
#SBATCH --error=slurm_logs/motifcheck_ablv2_nosyn10k_r023-%j.err
set -eo pipefail
eval "$(mamba shell hook --shell bash)"
mamba activate notagen
cd notebook/
jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=4800 \
    --output /usr/xtmp/cy232/accompgen/motifcheck_out/motif_check_ablv2_nosyn10k_r023_executed_${SLURM_JOB_ID}.ipynb \
    motif_check_ablv2_nosyn10k_r023.ipynb
