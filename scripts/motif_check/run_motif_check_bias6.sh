#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --output=slurm_logs/motifcheck6-%j.out
#SBATCH --error=slurm_logs/motifcheck6-%j.err

set -eo pipefail
eval "$(mamba shell hook --shell bash)"
mamba activate notagen

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd notebook/
# Larger cell timeout (2h) so the slower +6 generation can finish all 50 pieces.
jupyter nbconvert --to notebook --execute --allow-errors \
    --ExecutePreprocessor.timeout=7200 \
    --output /usr/xtmp/cy232/accompgen/motifcheck_out/v6_executed_${SLURM_JOB_ID}.ipynb \
    motif_check_bias6.ipynb

echo "EXECUTED_NB=/usr/xtmp/cy232/accompgen/motifcheck_out/v6_executed_${SLURM_JOB_ID}.ipynb"
