#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=3:00:00
#SBATCH --output=slurm_logs/evalloss_4x4-%j.out
#SBATCH --error=slurm_logs/evalloss_4x4-%j.err

set -eo pipefail
eval "$(mamba shell hook --shell bash)"
mamba activate notagen

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd finetune/
CKPT=/usr/xtmp/cy232/accompgen/weights/weights_notagen_1motif_v1_bias4_cropsyn_irishfull_4x4_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1.pth \
    python eval_loss.py
