"""Re-run the motif checker over the already-generated `noreal` pieces using the
current motif/extract.py backbone, and rewrite the results CSV.

Only the checker (motif containment) changed; generation and realization
faithfulness are untouched. So we:
  - recompute body containment (V:1 contains the target motif) by re-checking the
    saved .abc files for each job's model with the fixed backbone, and
  - reuse the realization-faithfulness numbers from the executed notebooks (the
    checker does not affect them).

Run:  python experiments/recheck_noreal.py
"""
import csv, glob, os, re, sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
from motif.extract import (count_interval_motifs, abc_to_pitches_with_bars as _a2p,
                           motif_pattern_relative_to_first, _coarsen_pattern)

import make_results_csv as M  # reuse RUNS, OUT_DIR, TARGET, stream_text, FIELDS

TARGET = M.TARGET                       # (0, 3, -1, -1)
INTERVAL_MODE = "step_skip_leap"
ABC_ROOT = os.path.join(PROJ_ROOT, "notebook")
NEW_CSV = os.path.join(os.path.dirname(__file__), "motif_check_noreal_results_rechecked.csv")


# --- checker, copied verbatim from the notebooks (split + per-voice scan) -------
def split_voices_from_abc(abc_text: str) -> Dict[str, str]:
    voice_to_chunks: Dict[str, List[str]] = defaultdict(list)
    for line in abc_text.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r'^[A-WYZ]:', s) or s.startswith('%%') or s.startswith('%'):
            continue
        s = re.sub(r'^\[r:[^\]]*\]', '', s)
        parts = re.split(r'(\[V:[^\]]+\])', s)
        current = None
        for part in parts:
            vm = re.match(r'^\[V:([^\]]+)\]$', part)
            if vm:
                current = vm.group(1).strip()
            elif current is not None and part.strip():
                voice_to_chunks[current].append(part)
    return {v: ' '.join(chunks) for v, chunks in voice_to_chunks.items()}


W = len(TARGET)


def _count_new(pitches, bars) -> int:
    """Fixed backbone: collapses immediate repeats, slides window."""
    counts, _ = count_interval_motifs(
        pitches, bars, window_notes=W, stride_notes=1, interval_mode=INTERVAL_MODE,
    )
    return counts.get(TARGET, 0)


def _count_old(pitches) -> int:
    """Previous backbone behaviour: repeats -> None sentinel, skip windows with None."""
    proc, prev, first = [], None, True
    for p in pitches:
        if not first and p == prev:
            proc.append(None)
        else:
            proc.append(p); prev = p; first = False
    cnt = Counter()
    for st in range(0, len(proc) - W + 1):
        w = proc[st:st + W]
        if None in w:
            continue
        cnt[_coarsen_pattern(motif_pattern_relative_to_first(w), INTERVAL_MODE)] += 1
    return cnt.get(TARGET, 0)


def _v1_pitches(abc_text):
    v1 = split_voices_from_abc(abc_text).get("1", "")
    if not v1:
        return None, None
    pitches, bars, *_ = _a2p(v1)
    if len(pitches) < W:
        return None, None
    return pitches, bars


def recheck_model_dir(model: str) -> Tuple[Dict[int, bool], Dict[int, bool]]:
    """Return ({i: V:1 hit under NEW checker}, {i: V:1 hit under OLD checker})
    for the local .abc files (same pieces, both checkers)."""
    d = os.path.join(ABC_ROOT, f"motif_check_noreal_{model}")
    new_hit, old_hit = {}, {}
    for path in glob.glob(os.path.join(d, "*.abc")):
        m = re.search(r"_(\d+)\.abc$", os.path.basename(path))
        if not m:
            continue
        i = int(m.group(1))
        abc = open(path, encoding="utf-8").read()
        pitches, bars = _v1_pitches(abc)
        if pitches is None:
            new_hit[i] = old_hit[i] = False
            continue
        new_hit[i] = _count_new(pitches, bars) > 0
        old_hit[i] = _count_old(pitches) > 0
    return new_hit, old_hit


def old_faith_and_realiz(model, job):
    """Pull checker-independent numbers from the executed notebook."""
    fs = glob.glob(f"{M.OUT_DIR}/motif_check_noreal_{model}_executed_{job}.ipynb")
    if not fs:
        return {}, ("", "", ""), ("", "", ""), "", ""
    txt = M.stream_text(fs[0])

    def frac(pat):
        mm = re.search(pat + r"\s*(\d+)/(\d+)\s*\(([\d.]+)%\)", txt)
        return (mm.group(1), mm.group(2), mm.group(3)) if mm else ("", "", "")

    faith = {int(mm.group(1)): int(mm.group(2)) >= 1
             for mm in re.finditer(r"\[(\d+)\] body_hit=\w\s+realization=\w+\s+matched=(\d+)/\d+", txt)}
    r1 = frac(r"Realization matches >=1 target:")
    ra = frac(r"Realization matches ALL targets:")

    def num(pat):
        mm = re.search(pat, txt)
        return mm.group(1) if mm else ""
    gen_ok = num(r"Generated OK:\s*(\d+)")
    emitted = num(r"emitted a %motif:abc line:\s*(\d+)")
    return faith, r1, ra, gen_ok, emitted


def _orig_body_n():
    """body_n per model from the historical CSV (to detect piece-set match)."""
    out = {}
    p = os.path.join(os.path.dirname(__file__), "motif_check_noreal_results.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            try:
                out[r["model"]] = int(r["body_n"])
            except (KeyError, ValueError):
                pass
    return out


def main():
    orig_body = _orig_body_n()
    rows = []
    hdr = f"{'model':9s} {'old(local)':>10s} {'new(local)':>10s} {'delta':>6s}  {'pieces==job?':>12s}"
    print(hdr); print("-" * len(hdr))
    for model, bias, job in M.RUNS:
        new_body, old_body = recheck_model_dir(model)
        faith, (r1n, n, r1p), (ran, _, rap), gen_ok, emitted = old_faith_and_realiz(model, job)
        ntot = int(n) if n else (len(new_body) or 50)

        bn_new = sum(1 for i in new_body if new_body[i])
        bn_old = sum(1 for i in old_body if old_body[i])
        # The job's pieces are still on disk iff the OLD checker on local files
        # reproduces the historical CSV body_n. Only then is the crosstab (which
        # reuses the job's realization-faithfulness) validly aligned to the pieces.
        pieces_match = (model in orig_body and bn_old == orig_body[model])

        if pieces_match:
            fb = fbm = ubh = neither = 0
            for i in range(ntot):
                fa = faith.get(i, False)
                bv = new_body.get(i, False)
                fb += fa and bv
                fbm += fa and not bv
                ubh += (not fa) and bv
                neither += (not fa) and (not bv)
            cross = dict(faithful_and_body=fb, faithful_body_miss=fbm,
                         unfaithful_body_hit=ubh, neither=neither)
            r1n_o, r1p_o, ran_o, rap_o, gen_o, em_o = r1n, r1p, ran, rap, gen_ok, emitted
        else:
            # pieces differ from the recorded job -> faithfulness/realiz columns
            # would be misaligned; leave them blank rather than report wrong numbers.
            cross = dict(faithful_and_body="", faithful_body_miss="",
                         unfaithful_body_hit="", neither="")
            r1n_o = r1p_o = ran_o = rap_o = gen_o = em_o = ""

        rows.append({
            "model": model, "bias": bias, "regime": M.REGIME, "n": ntot,
            "generated_ok": gen_o, "emitted_motif_abc": em_o,
            "realiz_ge1_n": r1n_o, "realiz_ge1_pct": r1p_o,
            "realiz_all_n": ran_o, "realiz_all_pct": rap_o,
            "body_n": bn_new, "body_pct": round(100 * bn_new / ntot, 1),
            **cross, "job": (job if pieces_match else f"{job}*"),
        })
        flag = "yes" if pieces_match else "NO (newer)"
        print(f"{model:9s} {bn_old:>7d}/{ntot:<2d} {bn_new:>7d}/{ntot:<2d} {bn_new - bn_old:>+6d}  {flag:>12s}")

    with open(NEW_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=M.FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {NEW_CSV}")
    print("* job marked with * => local pieces were regenerated after that job; "
          "body_pct reflects the current local pieces, crosstab/realiz left blank.")


if __name__ == "__main__":
    main()
