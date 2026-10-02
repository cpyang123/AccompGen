# Motif experiments — manifest

Archive of the motif-conditioning experiment set (notebooks in `notebook/`,
launch scripts archived in `experiments/scripts/`; the live copies are under the top-level `scripts/`, results in the CSVs here). All checks: 50
generations per (model, motif), abstract-only `%motif:v1` prompt, Romantic /
Schubert / Art Song header, fixed `motif/extract.py` backbone (post repeat-collapse
fix), V:1 body containment. Regenerate results with
`python experiments/make_multimotif_results.py`.

## Models (checkpoints on /usr/xtmp/cy232/accompgen/weights/)
| tag | data | schedule | notes |
|---|---|---|---|
| `irish20k_evalfix` | Lieder + 20k Irish (old-logic labels) | 10 syn + 10 real | matched-eval fixes |
| `irish50k` | Lieder + 50k Irish, isolated indices | 4 syn + 3 real | run 1 |
| `irish50k_r2` | same, Jun-30 label-consistent rebuild | 4 syn + 3 real | reproducibility retrain |
| `irishfull` | Lieder + full ~216k Irish | 1 syn + 1 real | token-matched to 50k runs |
| `irishfull_4x4` | same | 4 syn + 4 real | long run (job 12178993) |
| `irishfull_nosyn` | same | 0 syn + 1 real | curriculum ablation (job 12185343) |

## Motif set (11), by training-distribution rank
top-5: `0,-1,-1,-1` `0,1,1,1` `0,1,-1,-1` `0,-1,-1,1` `0,1,1,-1` (ranks 1–5) ·
mid: `0,3,-1,-1`(14) `0,1,-1,1`(15) `0,-1,-2,2`(20) `0,2,-2,-1`(27) ·
tail: `0,3,3,3` `0,-3,-3,-3` (absent).
Notebook tags: base name = `0,3,-1,-1`; `_m0333`/`_m0n3` = tail; `_top1..5`; `_mid1..3`.

## Ablations (jobs 12185320–12185343)
- `abl_base` — no motif prompt; the same 50 pieces scored against all 11 targets
  (unconditioned base rate).
- `abl_norecipe_*` — pretrained NotaGen + naive `%motif` prompt, bias 0 (11 motifs).
- `abl_biasoff_*` — `irishfull` weights, prompt kept, inference bias 0 (11 motifs).
- `irishfull_nosyn` — training ablation: no synthetic phase.

## Analysis
- `notebook/motif_analysis_full.ipynb` — full-corpus motif distribution
  (rank–frequency, coverage, Irish-vs-Lieder, pattern shape); writes
  `motif_distribution_full.csv` + `figures/`.

## Results files
- `motif_check_multimotif_results.csv` — consolidated grid (all models × motifs,
  `freq_rank`/`train_freq` columns; legacy single-motif rows folded in).
- `motif_check_noreal_results{,_rechecked}.csv` — original 5-model single-motif
  tables (pre-/post-checker-fix bodies; kept for provenance).
- `paper_figures.pptx` (+ `make_paper_figures_pptx.py`) — editable figure deck.

## Provenance notes
- Executed notebooks (with outputs) live on scratch:
  `/usr/xtmp/cy232/accompgen/motifcheck_out/`.
- `irish20k_evalfix` trained on pre-fix (old-logic) motif labels; 50k-r2 and all
  irishfull models trained on current-logic labels (Jun-30+ rebuilds) — a label
  confound when comparing 20k vs larger scales.
- Archived copies here are no-clobber snapshots; the five old-generation
  notebooks (orig/v1/cropsyn/irish10k/irish20k) predate the current set and back
  the legacy CSV rows.
