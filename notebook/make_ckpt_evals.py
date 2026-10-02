#!/usr/bin/env python3
"""Eval suite generator for a checkpoint: argv = <exp_tag> <bias> <name>
(defaults = the 10-real-epoch "- both" checkpoint). Produces:
the 11 v1 motifs (paper-table comparability) + the 24 v2 stratified motifs +
one no-prompt base run. Same retarget mechanics as make_ablation_v2.py."""
import json
import re
import stat
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import sys
TAG = sys.argv[1] if len(sys.argv) > 1 else '1motif_v1_bias0_irishfull_nosyn010'
BIAS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
NAME = sys.argv[3] if len(sys.argv) > 3 else 'both10ep'
WEIGHTS = ('/usr/xtmp/cy232/accompgen/weights/weights_notagen_' + TAG +
           '_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280'
           '_lr_1e-05_batch_1.pth')
SETS = json.load(open(HERE / 'motif_sets_v2.json'))
V2 = {f"r{SETS['ranks'][p]:03d}": p
      for p in SETS['A_top10'] + SETS['B_r11to30'] + SETS['C_tail']}
V1 = ['', 'top1', 'top2', 'top3', 'top4', 'top5',
      'mid1', 'mid2', 'mid3', 'm0333', 'm0n3']

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


def retarget(template, out_name, motif_patterns=None):
    nb = json.loads((HERE / template).read_text())
    hits = {'w': 0, 'b': 0, 'd': 0}
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if 'INFERENCE_WEIGHTS_PATH' in src:
            src, n = re.subn(r'(?m)^INFERENCE_WEIGHTS_PATH = "[^"]*"',
                             lambda m: f'INFERENCE_WEIGHTS_PATH = "{WEIGHTS}"', src)
            hits['w'] += n
            src, n = re.subn(r'(?m)^MOTIF_ATTENTION_BIAS = [0-9.]+',
                             lambda m: f'MOTIF_ATTENTION_BIAS = {BIAS}', src)
            hits['b'] += n
        if motif_patterns and 'motif_line = ' in src:
            line = ('%motif:v1:step_skip_leap: '
                    + ' ; '.join(motif_patterns) + ' \\n')
            src, _ = re.subn(r'(?m)^motif_line = "%motif:v1:step_skip_leap: [^"]*"',
                             lambda m: f'motif_line = "{line}"', src)
        if 'OUT_DIR = Path(' in src:
            src, n = re.subn(r'OUT_DIR = Path\("[^"]*"\)',
                             lambda m: f'OUT_DIR = Path("motif_check_{out_name}")', src)
            hits['d'] += n
        cell['source'] = src.splitlines(keepends=True)
    assert all(v >= 1 for v in hits.values()), (out_name, hits)
    nb['cells'].append({'cell_type': 'code', 'execution_count': None,
                        'metadata': {}, 'outputs': [],
                        'source': HITS_CELL.splitlines(keepends=True)})
    (HERE / f'motif_check_{out_name}.ipynb').write_text(json.dumps(nb, indent=1))
    sh = ROOT / 'scripts' / 'motif_check' / f'run_motif_check_{out_name}.sh'
    sh.write_text(RUNNER.format(name=out_name))
    sh.chmod(sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return out_name


names = []
for m in V1:
    suf = f'_{m}' if m else ''
    names.append(retarget(f'motif_check_noreal_abl_biasoff{suf}_4x4.ipynb',
                          f'abl_{NAME}{suf}_4x4'))
for rid, pat in V2.items():
    names.append(retarget('motif_check_noreal_abl_biasoff_4x4.ipynb',
                          f'ablv2_{NAME}_{rid}', [pat]))
names.append(retarget('motif_check_noreal_abl_base_4x4.ipynb',
                      f'ablv2_{NAME}_base', list(V2.values())))
(ROOT / 'scripts' / f'submit_{NAME}.sh').write_text(
    '#!/bin/bash\nset -eo pipefail\ncd "$(dirname "$0")/.."\n'
    f'T=$(sbatch --parsable {sys.argv[4] if len(sys.argv) > 4 else "scripts/fine_tune_irishfull_bias0_nosyn010.sh"})\n'
    'echo "training: $T"\n'
    + '\n'.join(f'sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_{n}.sh'
                for n in names) + '\n')
(ROOT / 'scripts' / f'submit_{NAME}.sh').chmod(0o755)
print(f'{len(names)} eval notebooks + runners; submit via scripts/submit_both10ep.sh')
