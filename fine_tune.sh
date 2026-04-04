#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=20:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err


mamba activate notagen

cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py