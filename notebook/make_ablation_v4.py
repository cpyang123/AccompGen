#!/usr/bin/env python3
"""v4 = top-up evals so the scaling grid's frequency bands have round
boundaries in the FULL-corpus ranking (217,146 annotated pieces, 210 distinct
patterns): A = ranks 1-10, B = 11-40, C = 41+.

The v2/v3 bands were sampled from the pre-expansion pool, and the full
Irishman corpus reorders the ranking, leaving A/B/C overlapping (B and C ended
up with near-identical frequency distributions). Re-banding the 24 motifs
already evaluated fills A with 6 and B with 6 of the 8 wanted; this script
generates the 4 missing motifs across all 12 cells of the grid (4 recipes x 3
data scales) so every band is sampled to 8 (C keeps all 12 it already has).

Everything else -- template, matched-bias fix, decoding, scoring -- is reused
verbatim from make_ablation_v3.retarget, so the new cells are protocol-
identical to the existing ones. Names: motif_check_ablv4_<cfg>_g<rank>, where
<rank> is the motif's rank in the full-corpus ordering (the ablv4 prefix and
the g-for-global rank keep these clear of the ablv3 files, whose r<NNN> ids
refer to the old ranking).
"""
import importlib.util
import json
import stat
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_spec = importlib.util.spec_from_file_location('mkv3', HERE / 'make_ablation_v3.py')
_mkv3 = importlib.util.module_from_spec(_spec)
_mkv3.__name__ = 'mkv3'
# make_ablation_v3 writes files at import time; run it in a scratch cwd-free way
# by only borrowing the pieces we need.
src = (HERE / 'make_ablation_v3.py').read_text()
ns = {'__name__': 'mkv3', '__file__': str(HERE / 'make_ablation_v3.py')}
exec(src.split("submit = ['#!/bin/bash'")[0], ns)      # definitions only, no emit
retarget = ns['retarget']
RUNNER = ns['RUNNER'].replace('ablv3', 'ablv4')
WDIR, WSUF = ns['WDIR'], ns['WSUF']

# the 4 motifs needed to fill A (ranks 1-10) and B (ranks 11-40) to 8 each,
# drawn with a fixed seed from the unevaluated patterns in each band
NEW = json.load(open(HERE / 'motif_sets_v4.json'))['new']

CONFIGS = {   # 12 cells: 4 recipes x 3 scales. All checkpoints already exist.
    'full':      ('1motif_v1_bias4_cropsyn_irishfull_4x4', 4.0),
    'bias0':     ('1motif_v1_bias0_cropsyn_irishfull_4x4', 0.0),
    'nosyn':     ('1motif_v1_bias4_irishfull_nosyn04', 4.0),
    'both':      ('1motif_v1_bias0_irishfull_nosyn04', 0.0),
    'full10k':   ('1motif_v1_bias4_cropsyn_irish10k_44', 4.0),
    'bias010k':  ('1motif_v1_bias0_cropsyn_irish10k_44', 0.0),
    'nosyn10k':  ('1motif_v1_bias4_irish10k_nosyn04', 4.0),
    'both10k':   ('1motif_v1_bias0_irish10k_nosyn04', 0.0),
    'full1k':    ('1motif_v1_bias4_cropsyn_irish1k_44', 4.0),
    'bias01k':   ('1motif_v1_bias0_cropsyn_irish1k_44', 0.0),
    'nosyn1k':   ('1motif_v1_bias4_irish1k_nosyn04', 4.0),
    'both1k':    ('1motif_v1_bias0_irish1k_nosyn04', 0.0),
}

submit = ['#!/bin/bash', '# v4 band top-up (make_ablation_v4.py).',
          'set -eo pipefail', 'cd "$(dirname "$0")/.."']
count = 0
for cfg, (tag, bias) in CONFIGS.items():
    weights = WDIR + tag + WSUF
    for gid, pat in NEW.items():
        name = f'{cfg}_{gid}'
        retarget('motif_check_noreal_abl_biasoff_4x4.ipynb', name, weights, bias, [pat])
        # retarget wrote ablv3-named artifacts; move them to the v4 namespace
        (HERE / f'motif_check_ablv3_{name}.ipynb').rename(HERE / f'motif_check_ablv4_{name}.ipynb')
        nbp = HERE / f'motif_check_ablv4_{name}.ipynb'
        nbp.write_text(nbp.read_text().replace(f'motif_check_ablv3_{name}',
                                               f'motif_check_ablv4_{name}'))
        sh = ROOT / 'scripts' / 'motif_check' / f'run_motif_check_ablv4_{name}.sh'
        sh.write_text(RUNNER.format(name=name))
        sh.chmod(sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        (ROOT / 'scripts' / 'motif_check' / f'run_motif_check_ablv3_{name}.sh').unlink(missing_ok=True)
        submit.append(f'sbatch --parsable scripts/motif_check/run_motif_check_ablv4_{name}.sh')
        count += 1
(ROOT / 'scripts' / 'submit_ablation_v4.sh').write_text('\n'.join(submit) + '\n')
(ROOT / 'scripts' / 'submit_ablation_v4.sh').chmod(0o755)
print(f'{count} notebooks + runners written ({len(NEW)} motifs x {len(CONFIGS)} cells)')
