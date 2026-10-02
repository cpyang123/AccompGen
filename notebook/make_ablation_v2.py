#!/usr/bin/env python3
"""v2 ablation sweep: 8 checkpoints (full-data + 10k-subset x {full, bias0,
nosyn, both}) x 24 sampled motifs (8 from top-10, 8 from ranks 11-30, 8 from
the tail; piece-annotation ranking, seed 20260731) + one no-prompt base run
per checkpoint. Clones the plain biasoff/base 4x4 templates, swapping
checkpoint, inference bias, motif line, and OUT_DIR, and appends a
mean-occurrences-per-piece summary cell. Also writes SLURM runners and
submit_ablation_v2.sh."""
import json
import re
import stat
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SETS = json.load(open(HERE / 'motif_sets_v2.json'))
MOTIFS = {f"r{SETS['ranks'][p]:03d}": p
          for p in SETS['A_top10'] + SETS['B_r11to30'] + SETS['C_tail']}

WDIR = '/usr/xtmp/cy232/accompgen/weights/weights_notagen_'
WSUF = ('_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280'
        '_lr_1e-05_batch_1.pth')
CONFIGS = {  # name -> (checkpoint tag, inference bias, training dep key)
    'full':     ('1motif_v1_bias4_cropsyn_irishfull_4x4', 4.0, None),
    'bias0':    ('1motif_v1_bias0_cropsyn_irishfull_4x4', 0.0, 'BIAS0FULL'),
    'nosyn':    ('1motif_v1_bias4_irishfull_nosyn04', 4.0, None),
    'both':     ('1motif_v1_bias0_irishfull_nosyn04', 0.0, None),
    'full10k':  ('1motif_v1_bias4_cropsyn_irish10k_44', 4.0, 'T10KFULL'),
    'bias010k': ('1motif_v1_bias0_cropsyn_irish10k_44', 0.0, 'T10KBIAS0'),
    'nosyn10k': ('1motif_v1_bias4_irish10k_nosyn04', 4.0, 'T10KNOSYN'),
    'both10k':  ('1motif_v1_bias0_irish10k_nosyn04', 0.0, 'T10KBOTH'),
}

HITS_CELL = '''\
# === Mean occurrences of the target motif(s) per generated piece ===
_ok = [r for r in results if r.get('error') is None]
_tot = sum(c for r in _ok for v in r['hits'].values() for c in v.values())
print(f"Mean motif occurrences per piece: {_tot/max(1,len(_ok)):.2f}  (n={len(_ok)})")
from collections import Counter
_per = Counter()
for r in _ok:
    for v in r['hits'].values():
        for pat, c in v.items():
            _per[pat] += c
for pat, c in sorted(_per.items()):
    print(f"  {pat}: {c/max(1,len(_ok)):.2f}/piece")
'''

RUNNER = """#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/motifcheck_ablv2_{name}-%j.out
#SBATCH --error=slurm_logs/motifcheck_ablv2_{name}-%j.err
set -eo pipefail
eval "$(mamba shell hook --shell bash)"
mamba activate notagen
cd notebook/
jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=4800 \\
    --output /usr/xtmp/cy232/accompgen/motifcheck_out/motif_check_ablv2_{name}_executed_${{SLURM_JOB_ID}}.ipynb \\
    motif_check_ablv2_{name}.ipynb
"""


def retarget(template, out_name, weights, bias, motif_patterns):
    nb = json.loads((HERE / template).read_text())
    hits = {'w': 0, 'b': 0, 'm': 0, 'd': 0}
    line = '%motif:v1:step_skip_leap: ' + ' ; '.join(motif_patterns) + ' \\n'
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        # NB: replacements go through a lambda so re.subn does NOT interpret
        # backslash escapes in the replacement (motif_line must keep a
        # literal backslash-n inside the string).
        if 'INFERENCE_WEIGHTS_PATH' in src:
            src, n = re.subn(r'(?m)^INFERENCE_WEIGHTS_PATH = "[^"]*"',
                             lambda m: f'INFERENCE_WEIGHTS_PATH = "{weights}"', src)
            hits['w'] += n
            src, n = re.subn(r'(?m)^MOTIF_ATTENTION_BIAS = [0-9.]+',
                             lambda m: f'MOTIF_ATTENTION_BIAS = {bias}', src)
            hits['b'] += n
        if 'motif_line = ' in src:
            src, n = re.subn(r'(?m)^motif_line = "%motif:v1:step_skip_leap: [^"]*"',
                             lambda m: f'motif_line = "{line}"', src)
            hits['m'] += n
        if 'OUT_DIR = Path(' in src:
            src, n = re.subn(r'OUT_DIR = Path\("[^"]*"\)',
                             lambda m: f'OUT_DIR = Path("motif_check_ablv2_{out_name}")', src)
            hits['d'] += n
        cell['source'] = src.splitlines(keepends=True)
    assert all(v >= 1 for v in hits.values()), (out_name, hits)
    nb['cells'].append({'cell_type': 'code', 'execution_count': None,
                        'metadata': {}, 'outputs': [],
                        'source': HITS_CELL.splitlines(keepends=True)})
    (HERE / f'motif_check_ablv2_{out_name}.ipynb').write_text(json.dumps(nb, indent=1))
    sh = ROOT / 'scripts' / 'motif_check' / f'run_motif_check_ablv2_{out_name}.sh'
    sh.write_text(RUNNER.format(name=out_name))
    sh.chmod(sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


submit = ['#!/bin/bash',
          '# Submit the v2 ablation batch (written by make_ablation_v2.py).',
          'set -eo pipefail', 'cd "$(dirname "$0")/.."',
          'T10KFULL=$(sbatch --parsable scripts/fine_tune_irish10k_abl_44.sh)',
          'T10KBIAS0=$(sbatch --parsable scripts/fine_tune_irish10k_abl_bias0_44.sh)',
          'T10KNOSYN=$(sbatch --parsable scripts/fine_tune_irish10k_abl_nosyn04.sh)',
          'T10KBOTH=$(sbatch --parsable scripts/fine_tune_irish10k_abl_bias0_nosyn04.sh)',
          'BIAS0FULL=12243138  # running full-data bias0 training',
          'echo "10k trainings: $T10KFULL $T10KBIAS0 $T10KNOSYN $T10KBOTH"']
count = 0
for cfg, (tag, bias, dep) in CONFIGS.items():
    weights = WDIR + tag + WSUF
    names = [f'{cfg}_{rid}' for rid in MOTIFS] + [f'{cfg}_base']
    for rid, pat in MOTIFS.items():
        retarget('motif_check_noreal_abl_biasoff_4x4.ipynb',
                 f'{cfg}_{rid}', weights, bias, [pat])
    retarget('motif_check_noreal_abl_base_4x4.ipynb',
             f'{cfg}_base', weights, bias, list(MOTIFS.values()))
    depflag = f' --dependency=afterok:${dep}' if dep else ''
    for n in names:
        submit.append(f'sbatch --parsable{depflag} scripts/motif_check/run_motif_check_ablv2_{n}.sh')
    count += len(names)
(ROOT / 'scripts' / 'submit_ablation_v2.sh').write_text('\n'.join(submit) + '\n')
(ROOT / 'scripts' / 'submit_ablation_v2.sh').chmod(0o755)
print(f'{count} notebooks + runners written; motifs: {list(MOTIFS)}')
