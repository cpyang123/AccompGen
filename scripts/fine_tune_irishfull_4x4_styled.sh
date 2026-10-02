#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=8-00:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# Full recipe (bias 4 + curriculum, 4 synthetic + 4 real epochs) retrained on
# the STYLE-PREAMBLED corpus: every file now carries
# %Period / %Composer / %Art Song above its %motif: lines (see
# data/add_style_preamble.py). Purpose: restore NotaGen's
# period-composer-instrumentation channel, which our earlier fine-tuning washed
# out because neither corpus carried those lines.
#
# NEW EXP_TAG (..._styled) => new WEIGHTS_PATH/LOGS_PATH, so the existing
# irishfull_4x4 checkpoint is untouched and remains the paper's current model.
# Early stopping is the standard behaviour: each phase saves only on improved
# held-out loss, and the real phase resumes from the best synthetic checkpoint
# (which is also preserved separately as ..._phase1.pth).

set -eo pipefail

eval "$(mamba shell hook --shell bash)"
mamba activate notagen
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export DATA_TRAIN_INDEX_PATH="../data/abcfiles_processed_v1_irishfull_train.jsonl"
export DATA_EVAL_INDEX_PATH="../data/abcfiles_processed_v1_irishfull_eval.jsonl"
export EXP_TAG="1motif_v1_bias4_cropsyn_irishfull_4x4_styled"
export MOTIF_ATTENTION_BIAS=4
export NUM_EPOCHS_SYNTHETIC=4
export NUM_EPOCHS_REAL=4
export WANDB_CACHE_DIR=/usr/xtmp/cy232/accompgen/wandb_cache

cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py
