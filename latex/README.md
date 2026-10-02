# Paper outline — NeurIPS Creative AI track

Outline of the motif-conditioning paper in NeurIPS format (the Creative AI
track uses the standard NeurIPS style; papers are ~4 pages excluding
references, plus a demo/artifact link).

## Files
- `main.tex` — the outline. Gray `[Planned: ...]` markers flag content still to
  be written; numbers in tables come from
  `../experiments/motif_check_multimotif_results.csv` (regenerate with
  `python experiments/make_multimotif_results.py`).
- `neurips_2023.sty` — official NeurIPS style (2024/2025 URLs are dead; the
  layout is unchanged). Swap in the current year's file when available and
  update the `\usepackage` line.
- `references.bib` — seed bibliography with TODO placeholders.

## Build
```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Style-mode options (in main.tex)
- `\usepackage{neurips_2023}` — submission (anonymous + line numbers)
- `\usepackage[preprint]{neurips_2023}` — preprint
- `\usepackage[final]{neurips_2023}` — camera-ready (current setting, for drafting)
