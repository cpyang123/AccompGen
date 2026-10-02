"""Build a CSV of the motif-check summary metrics across models.

Reads the executed notebooks in /usr/xtmp/cy232/accompgen/motifcheck_out/ and
writes one tidy row per model. Re-run after new jobs finish to refresh.

    python experiments/make_results_csv.py
"""
import ast, csv, glob, json, os, re

# Body containment + crosstab are restricted to voice V:1 (the melody), so the
# multivoice `orig` model is comparable to the single-voice models (whose only
# voice is V:1, so this restriction leaves them unchanged). Computed from the
# per-piece `hits={voice: {pattern: count}}` printed by the generation loop, not
# the any-voice summary line.
TARGET = (0, 3, -1, -1)

OUT_DIR = "/usr/xtmp/cy232/accompgen/motifcheck_out"
CSV_PATH = os.path.join(os.path.dirname(__file__), "motif_check_noreal_results.csv")

# USE_REALIZED_MOTIF=False (abstract-only), \n-fixed prompt + fixed checker, n=50,
# motif (0,3,-1,-1), Schubert Art Song. model -> (bias, executed-notebook job id).
RUNS = [
    ("orig",     2, 11916571),   # multivoice (any-voice body containment, inflated)
    ("v1",       2, 11916570),
    ("cropsyn",  4, 11916063),   # no Irish
    ("irish10k", 4, 11916065),
    ("irish20k", 4, 11916066),
]
REGIME = "False"   # abstract %motif:v1 only, no realization handed in

FIELDS = ["model", "bias", "regime", "n", "generated_ok", "emitted_motif_abc",
          "realiz_ge1_n", "realiz_ge1_pct", "realiz_all_n", "realiz_all_pct",
          "body_n", "body_pct", "faithful_and_body", "faithful_body_miss",
          "unfaithful_body_hit", "neither", "job"]


def stream_text(path):
    nb = json.load(open(path))
    out = []
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                out.append("".join(o.get("text", [])))
    return "".join(out)


def parse_run(model, bias, job):
    fs = glob.glob(f"{OUT_DIR}/motif_check_noreal_{model}_executed_{job}.ipynb")
    row = {"model": model, "bias": bias, "regime": REGIME, "job": job}
    if not fs:
        return row  # job not finished yet -> blank metric cells
    txt = stream_text(fs[0])

    def num(pat):
        m = re.search(pat, txt)
        return m.group(1) if m else ""

    def frac(pat):
        m = re.search(pat + r"\s*(\d+)/(\d+)\s*\(([\d.]+)%\)", txt)
        return (m.group(1), m.group(2), m.group(3)) if m else ("", "", "")

    # Realization faithfulness is voice-independent -> read from the summary.
    r1n, n, r1p = frac(r"Realization matches >=1 target:")
    ran, _, rap = frac(r"Realization matches ALL targets:")

    # Body containment + crosstab from per-piece V:1 hits (not any-voice summary).
    hits = {int(m.group(1)): ast.literal_eval(m.group(2))
            for m in re.finditer(r"\[piece (\d+)\] contains_motif=\w+\s+hits=(\{.*\})", txt)}
    faith = {int(m.group(1)): int(m.group(2)) >= 1
             for m in re.finditer(r"\[(\d+)\] body_hit=\w\s+realization=\w+\s+matched=(\d+)/\d+", txt)}
    ntot = int(n) if n else (len(hits) or 50)

    def body_v1(i):  # does voice '1' contain the target motif?
        d = hits.get(i, {})
        return "1" in d and TARGET in d["1"]

    bn = sum(1 for i in hits if body_v1(i))
    fb = fbm = ubh = neither = 0
    for i in range(ntot):
        fa, bv = faith.get(i, False), body_v1(i)
        fb += fa and bv
        fbm += fa and not bv
        ubh += (not fa) and bv
        neither += (not fa) and (not bv)

    row.update({
        "n": n,
        "generated_ok": num(r"Generated OK:\s*(\d+)"),
        "emitted_motif_abc": num(r"emitted a %motif:abc line:\s*(\d+)"),
        "realiz_ge1_n": r1n, "realiz_ge1_pct": r1p,
        "realiz_all_n": ran, "realiz_all_pct": rap,
        "body_n": bn, "body_pct": round(100 * bn / ntot, 1),
        "faithful_and_body": fb, "faithful_body_miss": fbm,
        "unfaithful_body_hit": ubh, "neither": neither,
    })
    return row


def main():
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for model, bias, job in RUNS:
            w.writerow(parse_run(model, bias, job))
    print(f"wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
