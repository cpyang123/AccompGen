#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# "- both" with a longer schedule: bias 0, 0 synthetic + 10 real epochs on the
# full-Irishman corpus (the 0+4 version is the current containment champion at
# 92.9%; this tests whether more real epochs help or overfit -- early stopping
# on eval loss picks the best checkpoint either way). ~8h/real epoch => ~3.5d.
# WANDB_CACHE_DIR on xtmp: the end-of-run artifact staging blew the home
# quota on 3 earlier runs (training itself was unaffected).

set -eo pipefail

eval "$(mamba shell hook --shell bash)"
mamba activate notagen
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export DATA_TRAIN_INDEX_PATH="../data/abcfiles_processed_v1_irishfull_train.jsonl"
export DATA_EVAL_INDEX_PATH="../data/abcfiles_processed_v1_irishfull_eval.jsonl"
export EXP_TAG="1motif_v1_bias0_irishfull_nosyn010"
export MOTIF_ATTENTION_BIAS=0
export NUM_EPOCHS_SYNTHETIC=0
export NUM_EPOCHS_REAL=10
export WANDB_CACHE_DIR=/usr/xtmp/cy232/accompgen/wandb_cache

cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py
