"""Standalone eval-loss for a saved checkpoint on the real eval split.

Usage:  CKPT=/path/to/weights.pth python eval_loss.py

Loads train-gen.py as a module (model construction, dataloaders, eval_epoch all
live there at module level), loads the checkpoint, and reports eval loss on the
real (non-synthetic) eval set with the motif attention bias ON and OFF. The
random seed is reset before each pass so both see identical transposition draws.
"""
import importlib.util
import json
import os
from copy import deepcopy

import torch

spec = importlib.util.spec_from_file_location(
    'traingen', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train-gen.py'))
tg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tg)

ckpt_path = os.environ['CKPT']
checkpoint = torch.load(ckpt_path, map_location='cpu')
cpu_model = deepcopy(tg.model)
cpu_model.load_state_dict(checkpoint['model'])
tg.model.load_state_dict(cpu_model.state_dict())
print(f"Loaded {ckpt_path}\n  (epoch {checkpoint.get('epoch')}, "
      f"best_epoch {checkpoint.get('best_epoch')}, "
      f"min_eval_loss {checkpoint.get('min_eval_loss')})", flush=True)
checkpoint = None
cpu_model = None

with open(tg.DATA_EVAL_INDEX_PATH, 'r', encoding='utf-8') as f:
    eval_files = [json.loads(line) for line in f]
real_eval = [f for f in eval_files if 'synthetic' not in f['path']]
print(f"Eval files: {len(eval_files)} total, {len(real_eval)} real", flush=True)

loader, _ = tg.make_dataloader(real_eval)
results = {}
for use_weights in (True, False):
    tg.random.seed(0)          # identical transposition draws across passes
    tg.np.random.seed(0)
    torch.manual_seed(0)
    with torch.no_grad():
        loss = tg.eval_epoch(loader, use_motif_weights=use_weights)
    tag = 'bias ON (+%g)' % tg.MOTIF_ATTENTION_BIAS if use_weights else 'bias OFF'
    results[tag] = loss
    print(f"REAL_EVAL_LOSS [{tag}]: {loss:.6f}", flush=True)

print("\nSummary:")
for tag, loss in results.items():
    print(f"  {tag:>14}: {loss:.6f}")
