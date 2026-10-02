# MotiGen example showcase (peer-review supplementary site)

Fully static site presenting 20 generated examples from the largest MotiGen
configuration (`irishfull_4x4`: Lieder + full ~216k Irishman, 4 synthetic + 4
real epochs, trained-in motif attention bias +4), drawn from the 50-piece
`fullbf` evaluation runs of 2026-08-02 (Slurm jobs 12281166–12281176), one run
per motif.

These runs supersede the earlier `motif_check_noreal_irishfull_*_4x4` runs the
site previously used. The old inference path dropped the motif attention bias
after the first stream recut and never flagged the model's own `%motif:abc`
line, understating conditioning; `fullbf` evaluates the *same checkpoint* under
the matched-bias path. Containment rises across the board (e.g. rank-14
`0,3,-1,-1` 28/50 → 50/50; the near-unseen `0,-3,-3,-3` 6/50 → 24/50), which is
why the 200-piece tail top-up runs the site needed before are retired: both
near-out-of-distribution motifs now yield qualifying pieces from 50 generations.

Every showcased piece (1) contains the prompted motif in its melody, (2) has a
faithful self-composed `%motif:abc` realization, and (3) uses that realization
in the piece: at least a motif-length (4-note) span of the realization's
pitches appears verbatim (exact pitch, immediate repeats collapsed) in the
body; up to two pieces per run. The `top2` motif (`0,1,1,1`) is absent because
that checkpoint never emits a `%motif:abc` line for it (0/50 here, and ~0/250
across independent sweeps — a deterministic checkpoint anomaly, not sampling).

No build step and no network dependency: `abcjs` is vendored, scores render and
play in-browser from the raw ABC. Open `index.html` directly, or serve the
directory (e.g. `python -m http.server`).

Deployment: the site is published from the separate repo
`/home/users/cy232/Music_Research/motigen-site` (GitHub Pages). This directory
is the build home; after regenerating, copy `index.html`, `data.js`,
`fig_encoding.png`, `abc/`, and `xml/` into that repo and commit.

- `index.html` — the page (vanilla JS, no framework)
- `data.js` — embedded example ABC + metadata (generated)
- `abc/` — per-example raw `.abc` sources (generated)
- `xml/` — per-example MusicXML downloads (generated from `abc/` via
  `python ../gradio/abc2xml.py -o xml abc/*.abc`; rerun after regenerating `abc/`)
- `fig_encoding.png` — interval-class encoding figure (copy of the paper's
  `latex/figures/fig_3_encoding.png`)
- `abcjs-basic-min.js` — vendored abcjs 6.4.3 (MIT, © Paul Rosen & Gregory Dyke)
- `build_site_data.py` — regenerates `examples.json` by parsing the executed
  evaluation notebooks on `/usr/xtmp/cy232/accompgen/motifcheck_out/` and the
  generated pieces in `notebook/motif_check_noreal_irishfull_*4x4/`, applying
  the selection rule above (containment is checked with the
  `motif/extract.py` backbone).
- `build_motif_spans.py` — annotates `examples.json` with motif highlight
  spans (cross-checked against the backbone) and emits `data.js` and `abc/`.

Build order: `build_site_data.py` → `build_motif_spans.py` → abc2xml → copy to
`motigen-site`.

Anonymity note: the page names no authors and points only at abcjs.net, but a
GitHub Pages URL exposes the hosting account name — for double-blind review,
host under an anonymous account (or anonymous.4open.science).
