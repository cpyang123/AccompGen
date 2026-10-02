"""Build a multi-motif results table for the irish50k vs irish20k_evalfix comparison.

Each run is a `noreal` motif-conditioning check (abstract %motif:v1 only, bias4, 50
pieces, Schubert Art Song) executed on the cluster; we parse the executed notebook's
stream output. All runs use the fixed motif/extract.py backbone, so rows are directly
comparable across models AND motifs (unlike the original single-motif CSV).

Body containment is restricted to voice V:1 (the melody), computed from the per-piece
`hits={voice: {pattern: count}}` lines (not the any-voice summary).

    python experiments/make_multimotif_results.py
"""
import ast, csv, glob, json, os, re

OUT_DIR = "/usr/xtmp/cy232/accompgen/motifcheck_out"
CSV_PATH = os.path.join(os.path.dirname(__file__), "motif_check_multimotif_results.csv")
REGIME = "False"   # abstract %motif:v1 only, no realization handed in
BIAS = 4

# (model, motif_tuple, executed-notebook glob). The glob matches whatever job id
# produced the executed notebook; if several exist the most recent is used.
RUNS = [
    # initial hand-picked motifs
    ("irish50k",         (0, 3, -1, -1),  "motif_check_noreal_irish50k_executed_*.ipynb"),
    ("irish50k",         (0, 3, 3, 3),    "motif_check_noreal_irish50k_m0333_executed_*.ipynb"),
    ("irish50k",         (0, -3, -3, -3), "motif_check_noreal_irish50k_m0n3_executed_*.ipynb"),
    ("irish20k_evalfix", (0, 3, -1, -1),  "motif_check_noreal_irish20k_evalfix_executed_*.ipynb"),
    ("irish20k_evalfix", (0, 3, 3, 3),    "motif_check_noreal_irish20k_evalfix_m0333_executed_*.ipynb"),
    ("irish20k_evalfix", (0, -3, -3, -3), "motif_check_noreal_irish20k_evalfix_m0n3_executed_*.ipynb"),
    # top-5 most frequent step_skip_leap motifs in the real training distribution
    ("irish50k",         (0, -1, -1, -1), "motif_check_noreal_irish50k_top1_executed_*.ipynb"),
    ("irish50k",         (0, 1, 1, 1),    "motif_check_noreal_irish50k_top2_executed_*.ipynb"),
    ("irish50k",         (0, 1, -1, -1),  "motif_check_noreal_irish50k_top3_executed_*.ipynb"),
    ("irish50k",         (0, -1, -1, 1),  "motif_check_noreal_irish50k_top4_executed_*.ipynb"),
    ("irish50k",         (0, 1, 1, -1),   "motif_check_noreal_irish50k_top5_executed_*.ipynb"),
    ("irish20k_evalfix", (0, -1, -1, -1), "motif_check_noreal_irish20k_evalfix_top1_executed_*.ipynb"),
    ("irish20k_evalfix", (0, 1, 1, 1),    "motif_check_noreal_irish20k_evalfix_top2_executed_*.ipynb"),
    ("irish20k_evalfix", (0, 1, -1, -1),  "motif_check_noreal_irish20k_evalfix_top3_executed_*.ipynb"),
    ("irish20k_evalfix", (0, -1, -1, 1),  "motif_check_noreal_irish20k_evalfix_top4_executed_*.ipynb"),
    ("irish20k_evalfix", (0, 1, 1, -1),   "motif_check_noreal_irish20k_evalfix_top5_executed_*.ipynb"),
    # mid-distribution motifs (ranks 15/20/27 of 179)
    ("irish50k",         (0, 1, -1, 1),   "motif_check_noreal_irish50k_mid1_executed_*.ipynb"),
    ("irish50k",         (0, -1, -2, 2),  "motif_check_noreal_irish50k_mid2_executed_*.ipynb"),
    ("irish50k",         (0, 2, -2, -1),  "motif_check_noreal_irish50k_mid3_executed_*.ipynb"),
    ("irish20k_evalfix", (0, 1, -1, 1),   "motif_check_noreal_irish20k_evalfix_mid1_executed_*.ipynb"),
    ("irish20k_evalfix", (0, -1, -2, 2),  "motif_check_noreal_irish20k_evalfix_mid2_executed_*.ipynb"),
    ("irish20k_evalfix", (0, 2, -2, -1),  "motif_check_noreal_irish20k_evalfix_mid3_executed_*.ipynb"),
    # irish50k r2 = retrain on the Jun-30 fully label-consistent rebuild (reproducibility)
    ("irish50k_r2",      (0, 3, -1, -1),  "motif_check_noreal_irish50k_r2_executed_*.ipynb"),
    ("irish50k_r2",      (0, 3, 3, 3),    "motif_check_noreal_irish50k_r2_m0333_executed_*.ipynb"),
    ("irish50k_r2",      (0, -3, -3, -3), "motif_check_noreal_irish50k_r2_m0n3_executed_*.ipynb"),
    ("irish50k_r2",      (0, -1, -1, -1), "motif_check_noreal_irish50k_r2_top1_executed_*.ipynb"),
    ("irish50k_r2",      (0, 1, 1, 1),    "motif_check_noreal_irish50k_r2_top2_executed_*.ipynb"),
    ("irish50k_r2",      (0, 1, -1, -1),  "motif_check_noreal_irish50k_r2_top3_executed_*.ipynb"),
    ("irish50k_r2",      (0, -1, -1, 1),  "motif_check_noreal_irish50k_r2_top4_executed_*.ipynb"),
    ("irish50k_r2",      (0, 1, 1, -1),   "motif_check_noreal_irish50k_r2_top5_executed_*.ipynb"),
    ("irish50k_r2",      (0, 1, -1, 1),   "motif_check_noreal_irish50k_r2_mid1_executed_*.ipynb"),
    ("irish50k_r2",      (0, -1, -2, 2),  "motif_check_noreal_irish50k_r2_mid2_executed_*.ipynb"),
    ("irish50k_r2",      (0, 2, -2, -1),  "motif_check_noreal_irish50k_r2_mid3_executed_*.ipynb"),
    # full-Irishman model (~216k tunes; 1 synthetic + 1 real epoch, token-matched)
    ("irishfull",        (0, 3, -1, -1),  "motif_check_noreal_irishfull_executed_*.ipynb"),
    ("irishfull",        (0, 3, 3, 3),    "motif_check_noreal_irishfull_m0333_executed_*.ipynb"),
    ("irishfull",        (0, -3, -3, -3), "motif_check_noreal_irishfull_m0n3_executed_*.ipynb"),
    ("irishfull",        (0, -1, -1, -1), "motif_check_noreal_irishfull_top1_executed_*.ipynb"),
    ("irishfull",        (0, 1, 1, 1),    "motif_check_noreal_irishfull_top2_executed_*.ipynb"),
    ("irishfull",        (0, 1, -1, -1),  "motif_check_noreal_irishfull_top3_executed_*.ipynb"),
    ("irishfull",        (0, -1, -1, 1),  "motif_check_noreal_irishfull_top4_executed_*.ipynb"),
    ("irishfull",        (0, 1, 1, -1),   "motif_check_noreal_irishfull_top5_executed_*.ipynb"),
    ("irishfull",        (0, 1, -1, 1),   "motif_check_noreal_irishfull_mid1_executed_*.ipynb"),
    ("irishfull",        (0, -1, -2, 2),  "motif_check_noreal_irishfull_mid2_executed_*.ipynb"),
    ("irishfull",        (0, 2, -2, -1),  "motif_check_noreal_irishfull_mid3_executed_*.ipynb"),
]

# Frequency of each tested motif in the real training distribution (piece-level
# %motif:v1 lines; ~7k-piece Irish sample + all Lieder, reference key C; 179 distinct
# patterns). rank/count make the sheet self-documenting: 'top' motifs are ranks 1-5,
# 'mid' motifs sit mid-distribution, and the leap motifs are tail/absent.
MOTIF_FREQ = {  # motif -> (rank, count)
    (0, -1, -1, -1): (1, 2586),
    (0, 1, 1, 1):    (2, 1580),
    (0, 1, -1, -1):  (3, 301),
    (0, -1, -1, 1):  (4, 276),
    (0, 1, 1, -1):   (5, 251),
    (0, 3, -1, -1):  (14, 78),
    (0, 1, -1, 1):   (15, 75),
    (0, -1, -2, 2):  (20, 53),
    (0, 2, -2, -1):  (27, 38),
    (0, 3, 3, 3):    (None, 0),
    (0, -3, -3, -3): (None, 0),
}

FIELDS = ["model", "motif", "freq_rank", "train_freq", "bias", "regime", "n",
          "generated_ok", "emitted_motif_abc",
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


def freq_cols(motif):
    rank, cnt = MOTIF_FREQ.get(motif, ("", ""))
    return {"freq_rank": rank if rank is not None else "tail", "train_freq": cnt}


def parse_run(model, motif, nb_name):
    row = {"model": model, "motif": ",".join(map(str, motif)), "bias": BIAS, "regime": REGIME,
           **freq_cols(motif)}
    fs = sorted(glob.glob(os.path.join(OUT_DIR, nb_name)), key=os.path.getmtime)
    if not fs:
        return row  # not found -> blank metrics
    path = fs[-1]  # most recent execution
    row["job"] = re.search(r"_executed_(\d+)\.ipynb", os.path.basename(path)).group(1)
    txt = stream_text(path)

    def num(pat):
        m = re.search(pat, txt)
        return m.group(1) if m else ""

    def frac(pat):
        m = re.search(pat + r"\s*(\d+)/(\d+)\s*\(([\d.]+)%\)", txt)
        return (m.group(1), m.group(2), m.group(3)) if m else ("", "", "")

    r1n, n, r1p = frac(r"Realization matches >=1 target:")
    ran, _, rap = frac(r"Realization matches ALL targets:")

    # Body containment (V:1) + crosstab from per-piece output.
    hits = {int(m.group(1)): ast.literal_eval(m.group(2))
            for m in re.finditer(r"\[piece (\d+)\] contains_motif=\w+\s+hits=(\{.*\})", txt)}
    faith = {int(m.group(1)): int(m.group(2)) >= 1
             for m in re.finditer(r"\[(\d+)\] body_hit=\w\s+realization=\w+\s+matched=(\d+)/\d+", txt)}
    ntot = int(n) if n else (len(hits) or 50)

    def body_v1(i):
        d = hits.get(i, {})
        return "1" in d and motif in d["1"]

    bn = sum(1 for i in hits if body_v1(i))
    fb = fbm = ubh = neither = 0
    for i in range(ntot):
        fa, bv = faith.get(i, False), body_v1(i)
        fb += fa and bv
        fbm += fa and not bv
        ubh += (not fa) and bv
        neither += (not fa) and (not bv)

    row.update({
        "n": n or ntot,
        "generated_ok": num(r"Generated OK:\s*(\d+)"),
        "emitted_motif_abc": num(r"emitted a %motif:abc line:\s*(\d+)"),
        "realiz_ge1_n": r1n, "realiz_ge1_pct": r1p,
        "realiz_all_n": ran, "realiz_all_pct": rap,
        "body_n": bn, "body_pct": round(100 * bn / ntot, 1),
        "faithful_and_body": fb, "faithful_body_miss": fbm,
        "unfaithful_body_hit": ubh, "neither": neither,
    })
    return row


# The earlier models (orig, v1, cropsyn, irish10k, irish20k) were only tested on
# motif 0,3,-1,-1. Their fixed-backbone body numbers live in the rechecked CSV
# (recheck_noreal.py re-checked the saved pieces with the corrected checker), which
# is methodologically consistent with the fresh runs above. Fold those rows in here
# so the consolidated table has every model in one place.
LEGACY_MOTIF = "0,3,-1,-1"
RECHECKED_CSV = os.path.join(os.path.dirname(__file__), "motif_check_noreal_results_rechecked.csv")


def legacy_rows():
    if not os.path.exists(RECHECKED_CSV):
        return []
    out = []
    for r in csv.DictReader(open(RECHECKED_CSV)):
        row = {k: r.get(k, "") for k in FIELDS}
        row["motif"] = LEGACY_MOTIF
        row.update(freq_cols((0, 3, -1, -1)))
        out.append(row)
    return out


def main():
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for row in legacy_rows():          # existing single-motif models (0,3,-1,-1)
            w.writerow(row)
        for model, motif, nb in RUNS:      # fresh multi-motif runs
            w.writerow(parse_run(model, motif, nb))
    print(f"wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
