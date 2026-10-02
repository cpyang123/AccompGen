#!/usr/bin/env python3
"""Matched-bias inference fix, applied to the v1 eval suite for the full model.

Fixes two inference/training asymmetries in inference_patch (verified against
notagen_core.model.generate, which pads per-patch motif weights with 1.0):
  1. the model's self-emitted %motif:abc line was never flagged, so its patches
     received no attention bias (in training they do);
  2. after a stream recut, motif_weights was set to None, disabling the bias
     entirely for the rest of the piece.
The fix maintains a live per-patch flag list during generation (a patch is
flagged iff the text line it starts in begins with '%motif:') and rebuilds the
flags from the re-encoded context after a recut. Suffix: 'fullbf'.
"""
import ast
import json
import re
import stat
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import sys
TAG = sys.argv[1] if len(sys.argv) > 1 else '1motif_v1_bias4_cropsyn_irishfull_4x4'
BIAS = sys.argv[2] if len(sys.argv) > 2 else '4.0'
NAME = sys.argv[3] if len(sys.argv) > 3 else 'fullbf'
WEIGHTS = ('/usr/xtmp/cy232/accompgen/weights/weights_notagen_' + TAG +
           '_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280'
           '_lr_1e-05_batch_1.pth')
V1 = ['', 'top1', 'top2', 'top3', 'top4', 'top5',
      'mid1', 'mid2', 'mid3', 'm0333', 'm0n3']

FLAG_INIT = [
    "motif_flags_live = list(_motif_flags)  # matched-bias fix: live per-patch flags",
    "cur_motif_line = ''",
]
FLAG_APPEND = [
    "# matched-bias fix: flag generated %motif: patches so the emitted",
    "# realization line receives the same attention bias as in training",
    "_flag_src = cur_motif_line if cur_motif_line else next_patch",
    "motif_flags_live.append(_flag_src.lstrip().startswith('%motif:'))",
    "cur_motif_line = next_patch.rsplit('\\n', 1)[-1] if '\\n' in next_patch else cur_motif_line + next_patch",
    "motif_weights = torch.tensor([[3.0 if _f else 1.0 for _f in motif_flags_live]], device=device)",
]
FLAG_REBUILD = [
    "# matched-bias fix: rebuild per-patch motif flags for the re-encoded context",
    "motif_flags_live = []",
    "cur_motif_line = ''",
    "for _pi in range(input_patches.shape[1] // PATCH_SIZE):",
    "    _ptxt = patchilizer.decode([input_patches[0, _pi*PATCH_SIZE:(_pi+1)*PATCH_SIZE].tolist()])",
    "    _flag_src = cur_motif_line if cur_motif_line else _ptxt",
    "    motif_flags_live.append(_flag_src.lstrip().startswith('%motif:'))",
    "    cur_motif_line = _ptxt.rsplit('\\n', 1)[-1] if '\\n' in _ptxt else cur_motif_line + _ptxt",
    "motif_weights = torch.tensor([[3.0 if _f else 1.0 for _f in motif_flags_live]], device=device)",
]

HITS_CELL = '''\
# === Mean occurrences of the target motif(s) per generated piece ===
_ok = [r for r in results if r.get('error') is None]
_tot = sum(c for r in _ok for v in r['hits'].values() for c in v.values())
print(f"Mean motif occurrences per piece: {_tot/max(1,len(_ok)):.2f}  (n={len(_ok)})")
'''

RUNNER = """#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/motifcheck_{name}-%j.out
#SBATCH --error=slurm_logs/motifcheck_{name}-%j.err
set -eo pipefail
eval "$(mamba shell hook --shell bash)"
mamba activate notagen
cd notebook/
jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=4800 \\
    --output /usr/xtmp/cy232/accompgen/motifcheck_out/motif_check_{name}_executed_${{SLURM_JOB_ID}}.ipynb \\
    motif_check_{name}.ipynb
"""


def indent_block(lines, ind):
    return '\n'.join(ind + l for l in lines)


def sub_after(src, pattern, block_lines):
    m = re.search(pattern, src, re.M)
    assert m, pattern
    ind = re.match(r'\s*', m.group(0)).group(0)
    return src[:m.end()] + '\n' + indent_block(block_lines, ind) + src[m.end():], 1


for suf_m in V1:
    suf = f'_{suf_m}' if suf_m else ''
    nb = json.loads((HERE / f'motif_check_noreal_abl_biasoff{suf}_4x4.ipynb').read_text())
    out_name = f'abl_{NAME}{suf}_4x4'
    done = {'fix1': 0, 'fix2': 0, 'fix3': 0, 'w': 0, 'b': 0, 'd': 0}
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if 'INFERENCE_WEIGHTS_PATH' in src:
            src, n = re.subn(r'(?m)^INFERENCE_WEIGHTS_PATH = "[^"]*"',
                             lambda m: f'INFERENCE_WEIGHTS_PATH = "{WEIGHTS}"', src)
            done['w'] += n
            src, n = re.subn(r'(?m)^MOTIF_ATTENTION_BIAS = [0-9.]+',
                             lambda m: f'MOTIF_ATTENTION_BIAS = {BIAS}', src)
            done['b'] += n
        if 'def inference_patch' in src:
            src, n = sub_after(
                src, r'^\s*motif_weights = torch\.tensor\(\[\[3\.0 if _f else 1\.0 '
                     r'for _f in _motif_flags\]\], device=device\)$', FLAG_INIT)
            done['fix1'] += n
            src, n = sub_after(
                src, r'^\s*input_patches = torch\.cat\(\[input_patches, '
                     r'predicted_patch\], dim=1\).*$', FLAG_APPEND)
            done['fix2'] += n
            # pre-existing recut bug: uses undefined context_tunebody_lines
            # (the defined local is context_tunebody_liness); pieces that
            # trigger a recut died with NameError before this fix
            src = src.replace("list(''.join(context_tunebody_lines[-cut_index:]))",
                              "list(''.join(context_tunebody_liness[-cut_index:]))")
            assert src.count('motif_weights = None') == 1
            src = src.replace(
                'motif_weights = None  # prompt motif patches dropped after stream recut',
                'pass  # matched-bias fix: motif weights rebuilt below after re-encoding')
            src, n = sub_after(
                src, r'^\s*input_patches = input_patches\.reshape\(1, -1\)$',
                FLAG_REBUILD)
            done['fix3'] += n
            clean = '\n'.join(l for l in src.splitlines() if not l.strip().startswith('!'))
            ast.parse(clean)
        if 'OUT_DIR = Path(' in src:
            src, n = re.subn(r'OUT_DIR = Path\("[^"]*"\)',
                             lambda m: f'OUT_DIR = Path("motif_check_{out_name}")', src)
            done['d'] += n
        cell['source'] = src.splitlines(keepends=True)
    assert all(v >= 1 for v in done.values()), (out_name, done)
    nb['cells'].append({'cell_type': 'code', 'execution_count': None,
                        'metadata': {}, 'outputs': [],
                        'source': HITS_CELL.splitlines(keepends=True)})
    (HERE / f'motif_check_{out_name}.ipynb').write_text(json.dumps(nb, indent=1))
    sh = ROOT / 'scripts' / 'motif_check' / f'run_motif_check_{out_name}.sh'
    sh.write_text(RUNNER.format(name=out_name))
    sh.chmod(sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    print('wrote', out_name)
