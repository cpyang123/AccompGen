#!/bin/bash
#SBATCH -J gencorpus
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-31
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err

# Generate the teacher corpus in parallel across the array. Each task handles one
# shard of the flat (combo, piece) work list; resubmit to resume (existing files
# are skipped). Target = GEN_PER_COMBO*112 ≈ 30k pieces.
#   pieces_per_shard = 30016 / num_shards  must finish in 48h.
#   32 shards -> ~940 pieces/shard -> ~16h @60s/piece, ~8h @30s/piece (comfortable).
#   Smoke-test first to measure your real per-piece time:
#     python distillation/generate_teacher_corpus.py --per-combo 2 --shard 0/1
#   sbatch scripts/gen_corpus.sh

N=${SLURM_ARRAY_TASK_COUNT:-32}

mamba activate notagen

cd distillation/
CUDA_VISIBLE_DEVICES=0 python generate_teacher_corpus.py --shard ${SLURM_ARRAY_TASK_ID}/${N}
