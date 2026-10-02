#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# Continue the full 4x4 model (bias4 + curriculum, 4 syn + 4 real) for 6 MORE
# real epochs under a NEW tag (total 4+10), so it is comparable to the
# bias0_nosyn010 run. The original 4x4 checkpoint is never modified: it is
# copied to the new tag's path once (cp -n), then LOAD_FROM_CHECKPOINT +
# SKIP_SYNTHETIC_PHASE resume into the real phase, saving best-of-continuation
# over the copy. Optimizer and LR-schedule state resume from the checkpoint.

set -eo pipefail

eval "$(mamba shell hook --shell bash)"
mamba activate notagen

W=/usr/xtmp/cy232/accompgen/weights
SUF=_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1.pth
cp -n $W/weights_notagen_1motif_v1_bias4_cropsyn_irishfull_4x4$SUF \
      $W/weights_notagen_1motif_v1_bias4_cropsyn_irishfull_4x4_cont6$SUF

export DATA_TRAIN_INDEX_PATH="../data/abcfiles_processed_v1_irishfull_train.jsonl"
export DATA_EVAL_INDEX_PATH="../data/abcfiles_processed_v1_irishfull_eval.jsonl"
export EXP_TAG="1motif_v1_bias4_cropsyn_irishfull_4x4_cont6"
export LOAD_FROM_CHECKPOINT=1
export SKIP_SYNTHETIC_PHASE=1
export NUM_EPOCHS_SYNTHETIC=0
export NUM_EPOCHS_REAL=6
export WANDB_CACHE_DIR=/usr/xtmp/cy232/accompgen/wandb_cache

cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py
