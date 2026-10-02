#!/usr/bin/env python3
"""Stratified motif-containment eval for the multi-length models, one (length, stratum)
per run — the per-length analogue of the v2 4-note ablation eval.

Protocol (mirrors the v2 notebooks): for every target motif, generate N pieces from the
abstract `%motif:v1:step_skip_leap: <pattern>` line ALONE (noreal regime; the model writes
its own %motif:abc realization and the whole piece), then count occurrences of the motif
in the generated V:1 melody with the shared backbone (motif/extract.py).

Two hit definitions are recorded per piece:
  exact  — occurrences of the prompted pattern itself (the v2 definition),
  orbit  — occurrences of the pattern OR its inversion / retrograde / retrograde-inversion
           (the definition the trans model was trained under; see
           count_interval_motifs(fold_transformations=True)).
Both are computed with folding OFF and summed over the orbit, so which member the model
happened to emit first never matters.

    python experiments/multilen_strat_eval.py --weights W.pth --length 6 --stratum A \
        --out-dir /usr/xtmp/.../strat_eval/<tag> [--n 50]
    python experiments/multilen_strat_eval.py --weights W.pth --stratum base ...
        (no prompt at all: the base rate of every target motif of every length, from one
         batch of N unprompted pieces)

Rhythm-conditioned regime (--rhythm-sets, rhythm-format checkpoints 2026-09): prompt i of a
(length, stratum) additionally carries `%motif:v1:rhythm: <ratios>` = rhythm prompt i of the
same (length, stratum) in the rhythm sets file. Per piece the record then also holds
  rhythm_hits  — occurrences of the prompted rhythm pattern in the generated melody
                 (motif/rhythm.py backbone: beat = 1/M-denominator, rests as '1z'),
  joint_hits   — exact melodic occurrences whose first and last note coincide with a
                 rhythm occurrence (the motif appears with BOTH its contour and its rhythm),
and the base run scores the base rate of every rhythm target as well.

Resumable: (pattern, index) pairs already present in the run's jsonl are skipped.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'notebook'))

import multilen_gen as mg                      # noqa: E402  (model + generation loop)
from motif import extract as mx                # noqa: E402  (single motif backbone)

STRATA = {'A': 'A_top10', 'B': 'B_r11to30', 'C': 'C_tail'}


def pat_tuple(s):
    return tuple(int(x) for x in s.split(','))


def sanitize(s):
    # '-' -> 'm' first: dropping the sign made e.g. 0,1,-1 and 0,1,1 share a filename, so the
    # later motif's pieces overwrote the earlier one's .abc files (results.jsonl was unaffected).
    return re.sub(r'[^A-Za-z0-9]+', '_', s.replace('-', 'm')).strip('_')


def motif_hits(abc_text, patterns):
    """{pattern_str: {'exact': n, 'orbit': n, 'by_form': {label: n}}} for the generated
    melody. One backbone pass per distinct motif length."""
    v1, key = mg._v1_text_and_key(abc_text)
    pitches, bars, char_idx, *_ = mx.abc_to_pitches_with_bars(v1, key=key)
    by_len, occ_by_len = {}, {}
    for L in sorted({len(pat_tuple(p)) for p in patterns}):
        if len(pitches) >= L:
            counts, occs = mx.count_interval_motifs(
                list(pitches), list(bars), window_notes=L, stride_notes=1,
                interval_mode=mg.INTERVAL_MODE, fold_transformations=False)
        else:
            counts, occs = {}, {}
        by_len[L] = counts
        occ_by_len[L] = occs
    out = {}
    for p in patterns:
        pat = pat_tuple(p)
        counts = by_len[len(pat)]
        # (start char, end char) of every exact occurrence in the stripped V:1 text, for the
        # joint contour+rhythm check
        spans = [(char_idx[o['start_note']][0], char_idx[o['end_note']][0])
                 for o in occ_by_len[len(pat)].get(pat, [])]
        forms = mx._pattern_transforms(pat)
        by_form, seen = {}, set()
        for lbl, form in zip(mx.TRANSFORM_LABELS, forms):
            if form in seen:          # degenerate pattern: identical forms counted once
                continue
            seen.add(form)
            c = counts.get(form, 0)
            if c:
                by_form[lbl] = c
        out[p] = {'exact': counts.get(forms[0], 0), 'orbit': sum(by_form.values()),
                  'by_form': by_form, '_spans': spans}
    return out, key, len(pitches)


def rhythm_hits(abc_text, rhythms, melodic_hits=None):
    """{rhythm_str: {'exact': n, 'joint': n}} — occurrences of each rhythm pattern in the
    generated melody; 'joint' = exact melodic occurrences (from motif_hits) starting AND ending
    on the same note as some rhythm occurrence. Windows are compared by the start character of
    their first and last note in the stripped V:1 text, which both backbones share."""
    out = {}
    mel_spans = set()
    if melodic_hits:
        for h in melodic_hits.values():
            mel_spans.update(h.get('_spans', []))
    for r in rhythms:
        n, occs, events = mg.rhythm_hits_in_piece(abc_text, mg.parse_rhythm_pattern(r))
        joint = 0
        for o in occs:
            span = (events[o['start_event']]['span'][0], events[o['end_event']]['span'][0])
            joint += span in mel_spans
        out[r] = {'exact': n, 'joint': joint}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--sets', default=str(ROOT / 'notebook' / 'motif_sets_multilen.json'))
    ap.add_argument('--length', type=int, help='motif length 4..10 (ignored for --stratum base)')
    ap.add_argument('--stratum', required=True, choices=['A', 'B', 'C', 'base'])
    ap.add_argument('--n', type=int, default=50, help='pieces per motif (or total, for base)')
    ap.add_argument('--bias', type=float, default=4.0, help='motif attention bias at inference')
    ap.add_argument('--pattern-bias-extra', type=float, default=0.0,
                    help="extra attention bias on the abstract %%motif:v1: pattern line only, added on top of --bias")
    ap.add_argument('--rhythm-sets', default=None,
                    help='rhythm sets json (notebook/rhythm_sets_multilen.json): condition every prompt on '
                         'the paired %%motif:v1:rhythm: line and score rhythm / joint containment')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-minutes', type=float, default=8.0, help='per-piece generation cap')
    args = ap.parse_args()

    sets = json.load(open(args.sets))
    rsets = json.load(open(args.rhythm_sets)) if args.rhythm_sets else None
    rhythm_targets = None   # base run: every rhythm pattern of every length
    rhythm_for = {}         # stratum runs: melodic prompt -> paired rhythm prompt
    if args.stratum == 'base':
        run_name = 'base'
        prompts = [None]
        targets = [p for k, s in sets.items() if k.startswith('len')
                   for p in s['A_top10'] + s['B_r11to30'] + s['C_tail']]
        if rsets:
            rhythm_targets = [p for k, s in rsets.items() if k.startswith('len')
                              for p in s['A_top10'] + s['B_r11to30'] + s['C_tail']]
    else:
        assert args.length is not None, '--length required for strata A/B/C'
        run_name = f'len{args.length}_{args.stratum}'
        prompts = sets[f'len{args.length}'][STRATA[args.stratum]]
        targets = None  # per prompt: the prompted pattern itself
        if rsets:
            rp = rsets[f'len{args.length}'][STRATA[args.stratum]]
            assert len(rp) >= len(prompts), f'rhythm set {run_name} has {len(rp)} < {len(prompts)} prompts'
            rhythm_for = {p: r for p, r in zip(prompts, rp)}
            for r in rp:
                # rests ('1z') sit inside a window but never count toward its length
                n_notes = sum(1 for t in mg.parse_rhythm_pattern(r) if not t.endswith('z'))
                assert n_notes == args.length, (r, args.length)

    out_dir = Path(args.out_dir) / run_name
    # Many runs start in the same second and share parents on NFS: pathlib/os.makedirs can
    # raise FileExistsError in that race, and freshly allocated nodes can even report a
    # just-created parent as missing (stale attribute cache) -> retry on any OSError until
    # the directory is actually visible (up to ~5 min).
    for attempt in range(30):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            print(f'  mkdir attempt {attempt}: {e!r}', flush=True)
        if out_dir.is_dir():
            break
        time.sleep(2 + attempt)
    else:
        raise RuntimeError(f'could not create {out_dir}')
    jsonl = out_dir / 'results.jsonl'
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r['prompt'], r['index']))
            except json.JSONDecodeError:
                pass
    print(f'run {run_name}: {len(prompts)} prompt(s) x {args.n} pieces; {len(done)} already done')

    model, patchilizer, device = mg.load_model(args.weights, motif_attention_bias=args.bias)

    t0 = time.time()
    n_new = 0
    with open(jsonl, 'a') as fh:
        for prompt in prompts:
            prompt_key = prompt if prompt is not None else ''
            rhythm = rhythm_for.get(prompt) if prompt is not None else None
            prompt_lines = mg.build_prompt(pat_tuple(prompt), rhythm=rhythm) if prompt is not None else []
            tgt = [prompt] if prompt is not None else targets
            rtgt = [rhythm] if rhythm is not None else (rhythm_targets if prompt is None else None)
            for i in range(args.n):
                if (prompt_key, i) in done:
                    continue
                # stable per-(seed, prompt, index) seed (hash() of str is salted per process)
                digest = hashlib.md5(f'{args.seed}|{prompt_key}|{i}'.encode()).digest()
                seed = int.from_bytes(digest[:4], 'little')
                torch.manual_seed(seed); random.seed(seed)
                t1 = time.time()
                abc, raw = mg.generate_piece(model, patchilizer, device, prompt_lines,
                                             print_tune=False, max_minutes=args.max_minutes,
                                             pattern_bias_extra=args.pattern_bias_extra)
                rec = {'prompt': prompt_key, 'index': i, 'seed': seed,
                       'gen_seconds': round(time.time() - t1, 1)}
                hits = None
                if abc is None:
                    rec.update({'ok': False, 'error': 'generation failed (length/time cap)'})
                else:
                    stem = f'{sanitize(prompt_key) or "base"}_{i:02d}'
                    (out_dir / f'{stem}.abc').write_text(abc, encoding='utf-8')
                    hits, key, n_notes = motif_hits(abc, tgt)
                    hdr = mg.extract_generated_header(raw)
                    rhits = None
                    if rtgt:
                        try:
                            rhits = rhythm_hits(abc, rtgt, hits)
                        except Exception as e:      # unparsable rhythm (odd tuplet etc.): record, don't die
                            rhits = {'error': repr(e)}
                    for h in hits.values():
                        h.pop('_spans', None)
                    rec.update({'ok': True, 'file': stem + '.abc', 'key': key, 'n_notes': n_notes,
                                'emitted_realization': hdr['abc'],
                                'emitted_header': hdr,
                                'hits': hits})
                    if rhythm is not None:
                        rec['rhythm_prompt'] = rhythm
                    if rhits is not None:
                        rec['rhythm_hits'] = rhits
                        if rhythm is not None and rhythm in rhits:
                            rec['declared_rhythm_count'] = hdr['rhythm_count']
                    if prompt is not None:
                        # inversion-format checks: is the self-written inversion instance a
                        # real inversion of the prompted motif, and do the declared counts
                        # match what the melody actually contains?
                        pat = pat_tuple(prompt)
                        inst = hdr['inversion_instance']
                        rec['inversion_instance_correct'] = (
                            mg.inversion_instance_is_correct(inst, pat, key) if inst else None)
                        rec['declared_count'] = hdr['count']
                        rec['declared_inversion_count'] = hdr['inversion_count']
                fh.write(json.dumps(rec) + '\n'); fh.flush()
                n_new += 1
                if prompt is not None and hits is not None:
                    h = hits[prompt]
                    tag = 'HIT' if h['exact'] else ('ORBIT' if h['orbit'] else 'miss')
                    ic = rec.get('inversion_instance_correct')
                    inst_tag = '-' if ic is None else ('instOK' if ic else 'instBAD')
                    rtag = ''
                    if rhythm is not None and isinstance(rec.get('rhythm_hits', {}).get(rhythm), dict):
                        rh = rec['rhythm_hits'][rhythm]
                        rtag = f' rhythm={rh["exact"]} joint={rh["joint"]} rdecl={rec.get("declared_rhythm_count")}'
                    print(f'  {prompt} [{i:02d}] {tag} exact={h["exact"]} inv={h["by_form"].get("I", 0)} '
                          f'orbit={h["orbit"]} declared={rec.get("declared_count")}/{rec.get("declared_inversion_count")} {inst_tag} '
                          f'{h["by_form"]}{rtag} ({rec["gen_seconds"]}s)', flush=True)
                else:
                    print(f'  {prompt_key or "base"} [{i:02d}] {"ok" if abc else "FAIL"} '
                          f'({rec["gen_seconds"]}s)', flush=True)

    # ---- summary -----------------------------------------------------------------
    recs = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    ok = [r for r in recs if r.get('ok')]
    print(f'\n=== {run_name}: {len(ok)}/{len(recs)} pieces ok, {n_new} new, '
          f'{(time.time()-t0)/60:.0f} min ===')
    if args.stratum == 'base':
        for L in range(4, 11):
            pats = [p for p in targets if len(pat_tuple(p)) == L]
            ex = sum(1 for r in ok for p in pats if r['hits'][p]['exact']) / max(1, len(ok) * len(pats))
            ob = sum(1 for r in ok for p in pats if r['hits'][p]['orbit']) / max(1, len(ok) * len(pats))
            print(f'  base rate len{L}: exact {100*ex:.1f}%  orbit {100*ob:.1f}%  '
                  f'({len(pats)} motifs x {len(ok)} pieces)')
            if rhythm_targets:
                rpats = [p for p in rhythm_targets
                         if sum(1 for t in mg.parse_rhythm_pattern(p) if not t.endswith('z')) == L]
                okr = [r for r in ok if isinstance(r.get('rhythm_hits'), dict) and 'error' not in r['rhythm_hits']]
                rx_ = sum(1 for r in okr for p in rpats if r['rhythm_hits'].get(p, {}).get('exact')) / max(1, len(okr) * len(rpats))
                print(f'  base rate len{L} rhythm: contain {100*rx_:.1f}%  ({len(rpats)} rhythms x {len(okr)} pieces)')
    else:
        for p in prompts:
            rs = [r for r in ok if r['prompt'] == p]
            if not rs:
                continue
            ex = sum(1 for r in rs if r['hits'][p]['exact'])
            ob = sum(1 for r in rs if r['hits'][p]['orbit'])
            occ = sum(r['hits'][p]['orbit'] for r in rs) / len(rs)
            rline = ''
            rp = rhythm_for.get(p)
            if rp is not None:
                rr = [r['rhythm_hits'][rp] for r in rs if isinstance(r.get('rhythm_hits', {}).get(rp), dict)]
                if rr:
                    rline = (f'  | rhythm {rp}: contain {sum(1 for x in rr if x["exact"])}/{len(rr)}'
                             f'  joint {sum(1 for x in rr if x["joint"])}/{len(rr)}')
            print(f'  {p}: exact {ex}/{len(rs)}  orbit {ob}/{len(rs)}  orbit-occ/piece {occ:.2f}{rline}')


if __name__ == '__main__':
    main()
