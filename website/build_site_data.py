import json, re, glob, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from motif.extract import abc_to_pitches_with_bars, detect_key_field

NB_DIR = '/usr/xtmp/cy232/accompgen/motifcheck_out'
ABC_ROOT = '/home/users/cy232/Music_Research/AccompGen/notebook'


def _collapse(seq):
    out = []
    for p in seq:
        if not out or out[-1] != p:
            out.append(p)
    return out


def _longest_run(needle, hay):
    best = 0
    for i in range(len(needle)):
        for j in range(len(hay)):
            k = 0
            while i + k < len(needle) and j + k < len(hay) and needle[i + k] == hay[j + k]:
                k += 1
            best = max(best, k)
    return best


def realization_partially_contained(abc, realization, min_run=4):
    """True if >= min_run consecutive collapsed pitches of the model's own
    %motif:abc realization appear verbatim (exact pitch) in the piece body."""
    if not realization:
        return False
    try:
        body, *_ = abc_to_pitches_with_bars(abc)
        real, *_ = abc_to_pitches_with_bars(realization, key=detect_key_field(abc))
    except Exception:
        return False
    cr, cb = _collapse(real), _collapse(body)
    return _longest_run(cr, cb) >= min(min_run, len(cr))

CONDITIONS = [
    # tag suffix (notebook base name part), motif, tier, corpus rank note
    ('top1', '0,-1,-1,-1', 'top', 'rank 1'),
    ('top2', '0,1,1,1',    'top', 'rank 2'),
    ('top3', '0,1,-1,-1',  'top', 'rank 3'),
    ('top4', '0,-1,-1,1',  'top', 'rank 4'),
    ('top5', '0,1,1,-1',   'top', 'rank 5'),
    ('',     '0,3,-1,-1',  'mid', 'rank 14'),
    ('mid1', '0,1,-1,1',   'mid', 'rank 15'),
    ('mid2', '0,-1,-2,2',  'mid', 'rank 20'),
    ('mid3', '0,2,-2,-1',  'mid', 'rank 27'),
    ('m0333','0,3,3,3',    'tail', 'nearly absent from training'),
    ('m0n3', '0,-3,-3,-3', 'tail', 'nearly absent from training'),
]

# Source runs: the "fullbf" suite (2026-08-02, Slurm jobs 12281166-76) evaluates
# the canonical irishfull_4x4 checkpoint on the paper's 11 v1 motifs under the
# *matched-bias* inference path. It supersedes the earlier
# motif_check_noreal_irishfull_*_4x4 runs, whose inference dropped the motif
# attention bias after the first stream recut (and never flagged the model's own
# %motif:abc line), understating conditioning. Do not mix the two: the old
# 200-piece tail top-ups (m0333b/m0n3b) are retired because the fixed runs supply
# enough qualifying tail pieces on their own.
def nb_for(tag):
    base = 'motif_check_abl_fullbf' + (('_'+tag) if tag else '') + '_4x4'
    hits = sorted(glob.glob(f'{NB_DIR}/{base}_executed_*.ipynb'))
    return hits[-1], base

examples = []
stats = []
for tag, motif, tier, rank in CONDITIONS:
    try:
        path, base = nb_for(tag)
    except IndexError:
        print(f'-- no executed notebook for {tag or "mid0"}, skipping')
        continue
    nb = json.load(open(path))
    # summary cell
    summary = realization = None
    for c in nb['cells']:
        if c['cell_type'] != 'code': continue
        src = ''.join(c['source'])
        txt = ''.join(''.join(o.get('text',[])) for o in c.get('outputs',[]))
        if src.startswith('# Summary'): summary = txt
        if src.startswith('# === Per-piece realization'): realization = txt

    m = re.search(r'Contain any target motif:\s+(\d+)/(\d+)', summary)
    contain, total = int(m.group(1)), int(m.group(2))

    # per-piece body hits: "[00] HIT ..." then "V:1  (...) xN" (may be several patterns; single here)
    hits = {}
    for pm in re.finditer(r'\[(\d+)\] (HIT|miss)[^\n]*\n((?:\s+V:1[^\n]*\n)*)', summary):
        idx = int(pm.group(1))
        n = sum(int(x) for x in re.findall(r'x(\d+)', pm.group(3)))
        hits[idx] = n if pm.group(2)=='HIT' else 0

    # realization per piece
    real = {}
    for rm in re.finditer(r"\[(\d+)\] body_hit=(\w)\s+realization=(\w+)[^\n]*\n\s+realization: '([^\n]*)'", realization):
        real[int(rm.group(1))] = {'faithful': rm.group(3)=='FAITHFUL', 'abc': rm.group(4)}

    n_faithful = sum(1 for v in real.values() if v['faithful'])
    stats.append({'motif': motif, 'tier': tier, 'rank': rank,
                  'body_rate': f'{contain}/{total}', 'faithful_rate': f'{n_faithful}/{len(real)}'})

    # pick 2: body hit + realization (partially) used in the body,
    # prefer faithful realization, then by hit count
    order = sorted(hits, key=lambda i: (hits[i]>0, real.get(i,{}).get('faithful',False), hits[i]), reverse=True)
    def qualifies(i):
        if hits[i] == 0: return False
        if not real.get(i,{}).get('faithful', False): return False
        abc_path = f'{ABC_ROOT}/{base}/motif_check_Romantic_Schubert_Franz_Art_Song_{i:02d}.abc'
        return realization_partially_contained(open(abc_path).read(), real.get(i,{}).get('abc',''))
    picked = [i for i in order if qualifies(i)][:2]
    for i in picked:
        abc_path = f'{ABC_ROOT}/{base}/motif_check_Romantic_Schubert_Franz_Art_Song_{i:02d}.abc'
        abc = open(abc_path).read()
        examples.append({
            'id': f'{tag or "mid0"}_{i:02d}', 'motif': motif, 'tier': tier, 'rank': rank,
            'piece': i, 'body_hits': hits[i],
            'faithful': real.get(i,{}).get('faithful'), 'realization': real.get(i,{}).get('abc',''),
            'abc': abc,
        })

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples.json')
json.dump({'examples': examples, 'stats': stats}, open(out,'w'), indent=1)
print(f'{len(examples)} examples selected')
for s in stats: print(s)
