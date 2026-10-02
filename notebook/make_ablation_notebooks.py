#!/usr/bin/env python3
"""Generate the ablation-table eval notebooks + SLURM runners by cloning the
existing biasoff 4x4 notebooks and swapping checkpoint / inference bias.

Rows (motif_check_noreal_abl_<row>{_motif}_4x4.ipynb, 11 motifs each):
  norecipe  pretrained backbone + motif prompt, bias 0   (Prompt-only)
  nosyn01   existing 0syn+1real bias4 checkpoint         (- synthetic, old schedule)
  bias0     bias0 4syn+4real checkpoint, bias 0          (- attention bias)
  nosyn04   bias4 0syn+4real checkpoint, bias 4          (- synthetic, schedule-matched)
  both      bias0 0syn+4real checkpoint, bias 0          (- both)
plus abl_base_backbone_4x4: backbone, no motif prompt (base rate per paper text).

The bias0/nosyn04/both checkpoints do not exist yet -- their jobs are
submitted with --dependency=afterok:<training job> (see submit_ablation.sh).
"""
import json
import re
import stat
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

WSUF = ('_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280'
        '_lr_1e-05_batch_1.pth')
WDIR = '/usr/xtmp/cy232/accompgen/weights/'
BACKBONE = ('/usr/xtmp/cy232/accompgen/pretrain/'
            'weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth')

# row -> (weights path, inference bias, slow) ; slow rows get the long
# nbconvert timeout (the untrained backbone blew the 4800s limit on 5 motifs)
ROWS = {
    'norecipe': (BACKBONE, 0.0, True),
    'nosyn01': (WDIR + 'weights_notagen_1motif_v1_bias4_cropsyn_irishfull_nosyn' + WSUF, 4.0, False),
    'bias0':   (WDIR + 'weights_notagen_1motif_v1_bias0_cropsyn_irishfull_4x4' + WSUF, 0.0, False),
    'nosyn04': (WDIR + 'weights_notagen_1motif_v1_bias4_irishfull_nosyn04' + WSUF, 4.0, False),
    'both':    (WDIR + 'weights_notagen_1motif_v1_bias0_irishfull_nosyn04' + WSUF, 0.0, False),
}
MOTIFS = ['', 'top1', 'top2', 'top3', 'top4', 'top5',
          'mid1', 'mid2', 'mid3', 'm0333', 'm0n3']

RUNNER = """#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time={walltime}
#SBATCH --output=slurm_logs/motifcheck_noreal_abl_{name}-%j.out
#SBATCH --error=slurm_logs/motifcheck_noreal_abl_{name}-%j.err
set -eo pipefail
eval "$(mamba shell hook --shell bash)"
mamba activate notagen
cd notebook/
jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout={timeout} \\
    --output /usr/xtmp/cy232/accompgen/motifcheck_out/motif_check_noreal_abl_{name}_executed_${{SLURM_JOB_ID}}.ipynb \\
    motif_check_noreal_abl_{name}.ipynb
"""


def retarget(template: Path, out: Path, weights: str, bias: float):
    nb = json.loads(template.read_text())
    hit_w = hit_b = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if 'INFERENCE_WEIGHTS_PATH' not in src:
            continue
        src, n = re.subn(r'(?m)^INFERENCE_WEIGHTS_PATH = "[^"]*"',
                         f'INFERENCE_WEIGHTS_PATH = "{weights}"', src)
        hit_w += n
        src, n = re.subn(r'(?m)^MOTIF_ATTENTION_BIAS = [0-9.]+',
                         f'MOTIF_ATTENTION_BIAS = {bias}', src)
        hit_b += n
        cell['source'] = src.splitlines(keepends=True)
    assert hit_w == 1 and hit_b == 1, f'{template.name}: weights x{hit_w}, bias x{hit_b}'
    out.write_text(json.dumps(nb, indent=1))


def write_runner(name: str, slow: bool):
    sh = ROOT / 'scripts' / 'motif_check' / f'run_motif_check_noreal_abl_{name}.sh'
    sh.write_text(RUNNER.format(name=name,
                                walltime='6:00:00' if slow else '2:00:00',
                                timeout=18000 if slow else 4800))
    sh.chmod(sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


made = []
for row, (weights, bias, slow) in ROWS.items():
    for motif in MOTIFS:
        suffix = f'_{motif}' if motif else ''
        template = HERE / f'motif_check_noreal_abl_biasoff{suffix}_4x4.ipynb'
        name = f'{row}{suffix}_4x4'
        retarget(template, HERE / f'motif_check_noreal_abl_{name}.ipynb',
                 weights, bias)
        write_runner(name, slow)
        made.append(name)

# base rate on the pretrained backbone (no motif prompt), as the paper text
# describes; the existing abl_base_4x4 measures it on the trained model.
retarget(HERE / 'motif_check_noreal_abl_base_4x4.ipynb',
         HERE / 'motif_check_noreal_abl_base_backbone_4x4.ipynb',
         BACKBONE, 0.0)
write_runner('base_backbone_4x4', slow=True)
made.append('base_backbone_4x4')

print(f'{len(made)} notebooks + runners:')
for n in made:
    print('  ', n)
