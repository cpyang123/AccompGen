#!/usr/bin/env python3
"""v3 = the v2 stratified ablation matrix re-made under the matched-bias
inference protocol (see make_biasfix_evals.py): 8 checkpoints x (24 sampled
motifs + 1 no-prompt base) with the live motif-flag fix, the post-recut flag
rebuild, and the recut NameError fix applied to every notebook. Also emits
fixed 36-eval suites for the two continuation checkpoints (both10ep,
fullcont6), to be chained on their training jobs. Names: motif_check_ablv3_*.
"""
import ast
import json
import re
import stat
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SETS = json.load(open(HERE / 'motif_sets_v2.json'))
MOTIFS = {f"r{SETS['ranks'][p]:03d}": p
          for p in SETS['A_top10'] + SETS['B_r11to30'] + SETS['C_tail']}
V1 = ['', 'top1', 'top2', 'top3', 'top4', 'top5',
      'mid1', 'mid2', 'mid3', 'm0333', 'm0n3']

WDIR = '/usr/xtmp/cy232/accompgen/weights/weights_notagen_'
WSUF = ('_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280'
        '_lr_1e-05_batch_1.pth')
CONFIGS = {  # name -> (tag, bias, dependency job id or None)
    'full':      ('1motif_v1_bias4_cropsyn_irishfull_4x4', 4.0, None),
    'bias0':     ('1motif_v1_bias0_cropsyn_irishfull_4x4', 0.0, None),
    'nosyn':     ('1motif_v1_bias4_irishfull_nosyn04', 4.0, None),
    'both':      ('1motif_v1_bias0_irishfull_nosyn04', 0.0, None),
    'full10k':   ('1motif_v1_bias4_cropsyn_irish10k_44', 4.0, None),
    'bias010k':  ('1motif_v1_bias0_cropsyn_irish10k_44', 0.0, None),
    'nosyn10k':  ('1motif_v1_bias4_irish10k_nosyn04', 4.0, None),
    'both10k':   ('1motif_v1_bias0_irish10k_nosyn04', 0.0, None),
    'both10ep':  ('1motif_v1_bias0_irishfull_nosyn010', 0.0, '12279620'),
    'fullcont6': ('1motif_v1_bias4_cropsyn_irishfull_4x4_cont6', 4.0, '12279673'),
}

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
#SBATCH --output=slurm_logs/motifcheck_ablv3_{name}-%j.out
#SBATCH --error=slurm_logs/motifcheck_ablv3_{name}-%j.err
set -eo pipefail
eval "$(mamba shell hook --shell bash)"
mamba activate notagen
cd notebook/
jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=4800 \\
    --output /usr/xtmp/cy232/accompgen/motifcheck_out/motif_check_ablv3_{name}_executed_${{SLURM_JOB_ID}}.ipynb \\
    motif_check_ablv3_{name}.ipynb
"""


def indent_block(lines, ind):
    return '\n'.join(ind + l for l in lines)


def sub_after(src, pattern, block_lines):
    m = re.search(pattern, src, re.M)
    assert m, pattern
    ind = re.match(r'\s*', m.group(0)).group(0)
    return src[:m.end()] + '\n' + indent_block(block_lines, ind) + src[m.end():]


def retarget(template, out_name, weights, bias, motif_patterns):
    """motif_patterns=None keeps the template's own motif_line (v1 suites)."""
    nb = json.loads((HERE / template).read_text())
    line = ('%motif:v1:step_skip_leap: ' + ' ; '.join(motif_patterns) + ' \\n'
            if motif_patterns is not None else None)
    done = set()
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if 'INFERENCE_WEIGHTS_PATH' in src:
            src, nw = re.subn(r'(?m)^INFERENCE_WEIGHTS_PATH = "[^"]*"',
                              lambda m: f'INFERENCE_WEIGHTS_PATH = "{weights}"', src)
            src, nb_ = re.subn(r'(?m)^MOTIF_ATTENTION_BIAS = [0-9.]+',
                               lambda m: f'MOTIF_ATTENTION_BIAS = {bias}', src)
            if nw:
                assert nw == 1 and nb_ == 1
                done.add('cfg')
        if 'def inference_patch' in src:
            src = sub_after(src, r'^\s*motif_weights = torch\.tensor\(\[\[3\.0 if _f else 1\.0 '
                                 r'for _f in _motif_flags\]\], device=device\)$', FLAG_INIT)
            src = sub_after(src, r'^\s*input_patches = torch\.cat\(\[input_patches, '
                                 r'predicted_patch\], dim=1\).*$', FLAG_APPEND)
            src = src.replace("list(''.join(context_tunebody_lines[-cut_index:]))",
                              "list(''.join(context_tunebody_liness[-cut_index:]))")
            src = src.replace(
                'motif_weights = None  # prompt motif patches dropped after stream recut',
                'pass  # matched-bias fix: motif weights rebuilt below after re-encoding')
            src = sub_after(src, r'^\s*input_patches = input_patches\.reshape\(1, -1\)$',
                            FLAG_REBUILD)
            done.add('fix')
        if 'motif_line = ' in src and 'parse_motif_prompt' in src:
            if line is not None:
                src, n = re.subn(r'(?m)^motif_line = "%motif:v1:step_skip_leap: [^"]*"',
                                 lambda m: f'motif_line = "{line}"', src)
                assert n == 1
            done.add('motif')
        if 'OUT_DIR = Path(' in src:
            src, n = re.subn(r'OUT_DIR = Path\("[^"]*"\)',
                             lambda m: f'OUT_DIR = Path("motif_check_ablv3_{out_name}")', src)
            assert n >= 1
            done.add('dir')
        clean = '\n'.join(l for l in src.splitlines() if not l.strip().startswith('!'))
        ast.parse(clean)
        cell['source'] = src.splitlines(keepends=True)
    assert done == {'cfg', 'fix', 'motif', 'dir'}, (out_name, done)
    nb['cells'].append({'cell_type': 'code', 'execution_count': None,
                        'metadata': {}, 'outputs': [],
                        'source': HITS_CELL.splitlines(keepends=True)})
    (HERE / f'motif_check_ablv3_{out_name}.ipynb').write_text(json.dumps(nb, indent=1))
    sh = ROOT / 'scripts' / 'motif_check' / f'run_motif_check_ablv3_{out_name}.sh'
    sh.write_text(RUNNER.format(name=out_name))
    sh.chmod(sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


submit = ['#!/bin/bash', '# v3 matched-bias ablation matrix (make_ablation_v3.py).',
          'set -eo pipefail', 'cd "$(dirname "$0")/.."']
count = 0
for cfg, (tag, bias, dep) in CONFIGS.items():
    weights = WDIR + tag + WSUF
    depflag = f' --dependency=afterany:{dep}' if dep else ''
    names = []
    for rid, pat in MOTIFS.items():
        retarget('motif_check_noreal_abl_biasoff_4x4.ipynb', f'{cfg}_{rid}',
                 weights, bias, [pat])
        names.append(f'{cfg}_{rid}')
    retarget('motif_check_noreal_abl_base_4x4.ipynb', f'{cfg}_base',
             weights, bias, list(MOTIFS.values()))
    names.append(f'{cfg}_base')
    if dep:  # continuation ckpts: also the 11 v1 paper-table motifs
        for m in V1:
            suf = f'_{m}' if m else ''
            retarget(f'motif_check_noreal_abl_biasoff{suf}_4x4.ipynb',
                     f'{cfg}_v1{suf}', weights, bias, None)
            names.append(f'{cfg}_v1{suf}')
    for n in names:
        submit.append(f'sbatch --parsable{depflag} scripts/motif_check/run_motif_check_ablv3_{n}.sh')
    count += len(names)
(ROOT / 'scripts' / 'submit_ablation_v3.sh').write_text('\n'.join(submit) + '\n')
(ROOT / 'scripts' / 'submit_ablation_v3.sh').chmod(0o755)
print(f'{count} notebooks + runners written')
