#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# 1k-subset ablation (800 Lieder + 200 Irishman): bias=4,
# 0 synthetic + 4 real epochs, schedule-matched to the other scales.
set -eo pipefail
eval "$(mamba shell hook --shell bash)"
mamba activate notagen
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
export DATA_TRAIN_INDEX_PATH="../data/abcfiles_processed_v1_irish1k_train.jsonl"
export DATA_EVAL_INDEX_PATH="../data/abcfiles_processed_v1_irish1k_eval.jsonl"
export EXP_TAG="1motif_v1_bias4_irish1k_nosyn04"
export MOTIF_ATTENTION_BIAS=4
export NUM_EPOCHS_SYNTHETIC=0
export NUM_EPOCHS_REAL=4
export WANDB_CACHE_DIR=/usr/xtmp/cy232/accompgen/wandb_cache
cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py
