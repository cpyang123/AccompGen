#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# FULL-parameter fine-tune on the RHYTHM-format multi-length dataset built by
# run_rhythm_data.sh (2026-09-22 experiment). Header per length L:
#   %motif:v1:step_skip_leap: <pattern>
#   %motif:v1:rhythm: <duration ratios, rests as Nz>
#   %motif:count: / %motif:abc:                 (melodic)
#   %motif:rhythm:count: / %motif:rhythm:abc:   (rhythmic)
#   %motif:inversion_count: [+ %motif:abc:inversion_instance:]
# Identical recipe to fine_tune_multilen_inv_full_rtx6000.sh (lengths 4-10, 21
# crops/piece, bias4, 2 synthetic + 5 real epochs, LR from config, notagen-bw env
# for the Blackwell GPU); only the data indices and EXP_TAG differ, so the result
# is directly comparable to the `inv` checkpoint. Submit chained after the data
# build:  sbatch --dependency=afterok:<data_jid> --kill-on-invalid-dep=yes $0

set -eo pipefail
# Repo root: the script's own dir when run directly; under sbatch BASH_SOURCE is
# the slurm spool copy, so fall back to the submit dir (submit from the repo root).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO/data/2_data_preprocess.py" ] || REPO="$SLURM_SUBMIT_DIR"
[ -f "$REPO/data/2_data_preprocess.py" ] || { echo "cannot locate repo root (REPO=$REPO)"; exit 1; }
cd "$REPO"
eval "$(mamba shell hook --shell bash)"
mamba activate /usr/xtmp/cy232/envs/notagen-bw

export DATA_TRAIN_INDEX_PATH="$REPO/data/abcfiles_processed_v1_rhythm_train.jsonl"
export DATA_EVAL_INDEX_PATH="$REPO/data/abcfiles_processed_v1_rhythm_eval.jsonl"
export EXP_TAG="multilen4to10_rhythm_v1_bias4_syn2real5"
export USE_LORA=0
export WANDB_CACHE_DIR=/usr/xtmp/cy232/accompgen/wandb_cache

# Fail fast: indices exist and the data really carries the rhythm block; GPU/torch
# pairing is right. Both before loading 500k files.
python - <<'PY'
import os, json, torch
for k in ('DATA_TRAIN_INDEX_PATH', 'DATA_EVAL_INDEX_PATH'):
    p = os.environ[k]; assert os.path.getsize(p) > 0, p
    with open(p) as fh:
        e = json.loads(next(l for l in fh if l.strip()))
    f = os.path.join(e['path'].rsplit('/', 1)[0], 'C', os.path.basename(e['path']) + '_C.abc')
    head = open(f, encoding='utf-8').read().split('[V:', 1)[0]
    assert '%motif:v1:rhythm:' in head and '%motif:rhythm:abc:' in head, f'no rhythm block in {f}'
    print(k, 'ok:', f)
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0))
x = torch.randn(256, 256, device='cuda'); y = (x @ x).sum().item(); assert y == y
print('cuda matmul ok')
PY

cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py
