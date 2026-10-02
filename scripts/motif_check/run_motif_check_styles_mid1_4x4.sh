#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/motifcheck_styles_mid1_4x4-%j.out
#SBATCH --error=slurm_logs/motifcheck_styles_mid1_4x4-%j.err
# Style sweep: full 4x4 model, fixed mid1 motif, 10 period/composer/
# instrumentation prompts x 20 pieces. 200 pieces; generous walltime in case
# orchestral prompts generate long multi-voice scores.
set -eo pipefail
eval "$(mamba shell hook --shell bash)"
mamba activate notagen
cd notebook/
jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=25200 \
    --output /usr/xtmp/cy232/accompgen/motifcheck_out/motif_check_styles_mid1_4x4_executed_${SLURM_JOB_ID}.ipynb \
    motif_check_styles_mid1_4x4.ipynb
