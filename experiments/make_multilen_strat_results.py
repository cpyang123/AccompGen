#!/usr/bin/env python3
"""Aggregate the per-length stratified eval (experiments/multilen_strat_eval.py) into a
results table: per model x length, piece-level containment averaged over the 24 motifs
(exact and orbit definitions), mean orbit occurrences per piece, and the no-prompt base
rate; then the average over lengths 4-10.

    python experiments/make_multilen_strat_results.py [--root /usr/xtmp/.../strat_eval_multilen]

Writes experiments/multilen_strat_results.csv and prints a summary.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT_DEFAULT = '/usr/xtmp/cy232/accompgen/strat_eval_multilen'
LENGTHS = range(4, 11)
STRATA = ['A', 'B', 'C']


def load(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def summarize_model(model_dir: Path):
    rows = []
    base = [r for r in load(model_dir / 'base' / 'results.jsonl') if r.get('ok')]
    for L in LENGTHS:
        per_motif = {}
        n_missing = 0
        for S in STRATA:
            recs = load(model_dir / f'len{L}_{S}' / 'results.jsonl')
            if not recs:
                n_missing += 1
                continue
            by_p = defaultdict(list)
            for r in recs:
                by_p[r['prompt']].append(r)
            for p, rs in by_p.items():
                ok = [r for r in rs if r.get('ok')]
                if not ok:
                    continue
                per_motif[p] = {
                    'stratum': S,
                    'rhythm_prompt': next((r['rhythm_prompt'] for r in ok if r.get('rhythm_prompt')), None),
                    'n_ok': len(ok), 'n_fail': len(rs) - len(ok),
                    'exact_contain': sum(1 for r in ok if r['hits'][p]['exact']) / len(ok),
                    'orbit_contain': sum(1 for r in ok if r['hits'][p]['orbit']) / len(ok),
                    'exact_occ': sum(r['hits'][p]['exact'] for r in ok) / len(ok),
                    'orbit_occ': sum(r['hits'][p]['orbit'] for r in ok) / len(ok),
                    'realized': sum(1 for r in ok if r.get('emitted_realization')) / len(ok),
                    # inversion-format metrics (None-safe for pre-format runs)
                    'inv_contain': sum(1 for r in ok if r['hits'][p]['by_form'].get('I', 0)) / len(ok),
                    'inv_occ': sum(r['hits'][p]['by_form'].get('I', 0) for r in ok) / len(ok),
                    'inst_emitted': sum(1 for r in ok if r.get('inversion_instance_correct') is not None) / len(ok),
                    'inst_correct': (lambda e: (sum(1 for r in e if r['inversion_instance_correct']) / len(e)) if e else None)(
                        [r for r in ok if r.get('inversion_instance_correct') is not None]),
                    'count_match': (lambda d: (sum(1 for r in d if r['declared_count'] == r['hits'][p]['exact']) / len(d)) if d else None)(
                        [r for r in ok if r.get('declared_count') is not None]),
                    'inv_count_match': (lambda d: (sum(1 for r in d if r['declared_inversion_count'] == r['hits'][p]['by_form'].get('I', 0)) / len(d)) if d else None)(
                        [r for r in ok if r.get('declared_inversion_count') is not None]),
                }
                # rhythm-conditioned runs (<tag>_rc): containment of the paired rhythm prompt and of
                # the joint contour+rhythm motif; None for melodic-only runs
                rr = [(r, r['rhythm_hits'][r['rhythm_prompt']]) for r in ok
                      if r.get('rhythm_prompt') and isinstance(r.get('rhythm_hits', {}).get(r['rhythm_prompt']), dict)]
                per_motif[p].update({
                    'rhythm_contain': (sum(1 for _, h in rr if h['exact']) / len(rr)) if rr else None,
                    'rhythm_occ': (sum(h['exact'] for _, h in rr) / len(rr)) if rr else None,
                    'joint_contain': (sum(1 for _, h in rr if h['joint']) / len(rr)) if rr else None,
                    'rhythm_count_match': (lambda d: (sum(1 for r, h in d if r['declared_rhythm_count'] == h['exact']) / len(d)) if d else None)(
                        [(r, h) for r, h in rr if r.get('declared_rhythm_count') is not None]),
                })
        if not per_motif:
            continue
        n = len(per_motif)
        mean = lambda k: sum(m[k] for m in per_motif.values()) / n
        def mean_opt(k):
            vals = [m[k] for m in per_motif.values() if m.get(k) is not None]
            return sum(vals) / len(vals) if vals else None
        row = {'length': L, 'n_motifs': n, 'strata_missing': n_missing,
               'pieces_ok': sum(m['n_ok'] for m in per_motif.values()),
               'pieces_fail': sum(m['n_fail'] for m in per_motif.values()),
               'exact_contain': mean('exact_contain'), 'orbit_contain': mean('orbit_contain'),
               'exact_occ': mean('exact_occ'), 'orbit_occ': mean('orbit_occ'),
               'realized': mean('realized'),
               'inv_contain': mean('inv_contain'), 'inv_occ': mean('inv_occ'),
               'inst_emitted': mean('inst_emitted'), 'inst_correct': mean_opt('inst_correct'),
               'count_match': mean_opt('count_match'), 'inv_count_match': mean_opt('inv_count_match'),
               'rhythm_contain': mean_opt('rhythm_contain'), 'rhythm_occ': mean_opt('rhythm_occ'),
               'joint_contain': mean_opt('joint_contain'), 'rhythm_count_match': mean_opt('rhythm_count_match')}
        for S in STRATA:
            ms = [m for m in per_motif.values() if m['stratum'] == S]
            row[f'orbit_contain_{S}'] = (sum(m['orbit_contain'] for m in ms) / len(ms)) if ms else None
            row[f'exact_contain_{S}'] = (sum(m['exact_contain'] for m in ms) / len(ms)) if ms else None
        if base:
            pats = list(per_motif)
            denom = len(base) * len(pats)
            row['base_exact'] = sum(1 for r in base for p in pats if r['hits'].get(p, {}).get('exact')) / denom
            row['base_orbit'] = sum(1 for r in base for p in pats if r['hits'].get(p, {}).get('orbit')) / denom
            row['base_n'] = len(base)
            rpats = sorted({m['rhythm_prompt'] for m in per_motif.values() if m.get('rhythm_prompt')})
            rbase = [r for r in base if isinstance(r.get('rhythm_hits'), dict) and 'error' not in r['rhythm_hits']]
            if rpats and rbase:
                row['base_rhythm'] = sum(1 for r in rbase for p in rpats
                                         if r['rhythm_hits'].get(p, {}).get('exact')) / (len(rbase) * len(rpats))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT_DEFAULT)
    args = ap.parse_args()
    root = Path(args.root)
    out_csv = Path(__file__).resolve().parent / 'multilen_strat_results.csv'
    all_rows = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rows = summarize_model(model_dir)
        if not rows:
            continue
        for r in rows:
            r['model'] = model_dir.name
        all_rows += rows
        keys = ['exact_contain', 'orbit_contain', 'exact_occ', 'orbit_occ', 'realized',
                'base_exact', 'base_orbit', 'inv_contain', 'inv_occ', 'inst_emitted',
                'inst_correct', 'count_match', 'inv_count_match',
                'rhythm_contain', 'rhythm_occ', 'joint_contain', 'rhythm_count_match', 'base_rhythm']
        avg = {'model': model_dir.name, 'length': 'avg4-10',
               'n_motifs': sum(r['n_motifs'] for r in rows),
               'strata_missing': sum(r['strata_missing'] for r in rows),
               'pieces_ok': sum(r['pieces_ok'] for r in rows),
               'pieces_fail': sum(r['pieces_fail'] for r in rows)}
        for k in keys:
            vals = [r[k] for r in rows if r.get(k) is not None]
            avg[k] = sum(vals) / len(vals) if vals else None
        all_rows.append(avg)

        print(f'\n== {model_dir.name} ==')
        print(f"{'len':>7} {'motifs':>6} {'exact%':>7} {'orbit%':>7} {'occ/pc':>7} "
              f"{'baseEx%':>8} {'baseOrb%':>9} {'A/B/C orbit%':>20} {'fail':>5}"
              f"{'inv%':>7} {'inv/pc':>7} {'inst%':>6} {'instOK%':>8} {'cnt=%':>6} {'icnt=%':>7}"
              f"{'rhy%':>6} {'joint%':>7} {'baseRhy%':>9} {'rcnt=%':>7}")
        for r in rows + [avg]:
            if r['length'] != 'avg4-10':
                abc = ' / '.join(f"{100*r[f'orbit_contain_{S}']:.0f}" if r.get(f'orbit_contain_{S}') is not None
                                 else '-' for S in STRATA)
            else:
                abc = ''
            be = f"{100*r['base_exact']:.1f}" if r.get('base_exact') is not None else '-'
            bo = f"{100*r['base_orbit']:.1f}" if r.get('base_orbit') is not None else '-'
            pct = lambda k: f"{100*r[k]:.1f}" if r.get(k) is not None else '-'
            occ = f"{r['inv_occ']:.2f}" if r.get('inv_occ') is not None else '-'   # hoisted: nested same-quote f-strings need py3.12
            print(f"{str(r['length']):>7} {r['n_motifs']:>6} {100*r['exact_contain']:>7.1f} "
                  f"{100*r['orbit_contain']:>7.1f} {r['orbit_occ']:>7.2f} {be:>8} {bo:>9} "
                  f"{abc:>20} {r['pieces_fail']:>5}"
                  f"{pct('inv_contain'):>7} {occ:>7} "
                  f"{pct('inst_emitted'):>6} {pct('inst_correct'):>8} {pct('count_match'):>6} {pct('inv_count_match'):>7}"
                  f"{pct('rhythm_contain'):>6} {pct('joint_contain'):>7} {pct('base_rhythm'):>9} {pct('rhythm_count_match'):>7}")

    if all_rows:
        fields = sorted({k for r in all_rows for k in r},
                        key=lambda k: (k not in ('model', 'length'), k))
        with open(out_csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
        print('\nwrote', out_csv)


if __name__ == '__main__':
    main()
