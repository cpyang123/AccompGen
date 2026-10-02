#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err


set -eo pipefail

# Initialize mamba for non-interactive batch shells, then activate the env.
eval "$(mamba shell hook --shell bash)"
mamba activate notagen

# Fail loudly if the env didn't provide torch, instead of crashing deep in the run.
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py