# Handoff: AccompGen / MotiGen, from the Duke cluster to the next one

Written 2026-10-02 by the Claude Code agent that worked on this project at Duke (user:
Peter Yang, `cy232`). It is addressed to the next agent (and to Peter as a checklist). It
carries over the working state, the decisions already made, the saved memories, and the
cluster-specific facts that must be re-mapped. Read `README.md` first for the project itself;
this file is about *how we work* and *what is in flight*.

If you keep a persistent memory, seed it from the **Memories carried over** section below.

---

## 1. How Peter works with the agent (preferences and standing instructions)

- **Smoke-test before any batch submit.** Run the pipeline end to end on one sample (one
  piece, two files, `EVAL_N=1`) and only then chain the batch behind it with
  `EXTRA_SBATCH="--dependency=afterok:<jid> --kill-on-invalid-dep=yes"`. This came from a
  real incident where 7 jobs ran 10-21 h and crashed on a trivial KeyError at the end. Make
  the smoke run cover the edge stratum (C has rest tokens) and not just the easy case.
- **Never overwrite an existing checkpoint or dataset.** New `EXP_TAG` per run; isolated
  scratch dirs and new index files for new data formats (the `rhythm` build is the model).
- **Design decisions Peter made explicitly (do not re-ask):** for rhythmic motifs the beat is
  `1/(meter denominator)` literally; initial `M:` rules the piece; no `M:` means the `L:` unit
  is the beat; rests are `<ratio>z` tokens inside a window but never count toward the length;
  edge rests are excluded; no collapsing of repeats; no inversion or transforms for rhythm;
  ties merge; chords are one event; grace notes are dropped; exact fractions never decimals; a
  length gets both melodic and rhythmic blocks or neither; the synthetic generator was to
  include everything. Pattern-line bias (`EVAL_PATTERN_EXTRA`) is **additive** to the uniform
  bias, not a replacement.
- **Deploying the Space is Peter's call.** `hf_space/deploy.py` refuses to overwrite; a new
  Space name is needed. Do not deploy unasked.
- Peter likes short status updates while long jobs run, concrete numbers with confidence
  intervals when comparing configurations, and results written back into memory/notes.
- Keep `README.md` and this file current when state changes.

## 2. State at handoff (2026-10-02)

**Checkpoints** (scratch `weights/`, each with a `logs_notagen_<tag>_*.txt` per-epoch log):

| Tag | What | Status |
|---|---|---|
| `multilen4to10_v1_bias4_syn2real5` | multi-length baseline | evaluated (`multilen`) |
| `multilen4to10_trans_v1_bias4_syn2real5` | I/R/RI folded | evaluated (`trans`) |
| `multilen4to10_inv_v1_bias4_syn2real5` | inversion header, full FT, best real epoch 3, eval loss 0.150 | evaluated, on the live Space |
| `..._inv_..._lora16` | LoRA r=16 variant | trained, not in the stratified eval |
| `multilen4to10_rhythm_v1_bias4_syn2real5` | rhythm header, full FT, best real epoch 3, eval loss 0.1417; `_phase1.pth` = synthetic epoch 2 (0.0948) | eval jobs done, **not aggregated** |
| `1motif_v1_*` series (28 tags) | legacy single-length models and ablations | finished campaign, see `experiments/MANIFEST.md` |

**Eval outputs** (scratch `strat_eval_multilen/<tag>/`): `multilen`, `trans`, `inv`,
`inv_bias5..8`, `inv_pat1`, `inv_pat2`, `inv_bias5_pat1`, `rhythm`, `rhythm_rc`. The CSV
`experiments/multilen_strat_results.csv` (2026-09-21) has every tag except the two rhythm
ones. Smoke outputs sit in `strat_eval_inv_smoke/` and `strat_eval_rhythm_smoke/` so the
aggregator does not pick them up as models.

**Demo:** Space `cpyang/motigen` serves `inv`. The local `hf_space/` bundle has the `rhythm`
checkpoint exported (`checkpoint.json`, sha `bc779e4c...`), with the rhythm textbox and
rhythm-aware verification wired in and tested (`pytest hf_space/tests`), not deployed. To go
back to `inv` locally: `python prepare.py --checkpoint <inv .pth>`.

**Paper:** `latex/main.tex`, title *Getting Motif-ated*, NeurIPS Creative AI format,
`[final]` style for drafting. Results tables still reflect the legacy single-length campaign.

**No jobs are running.** Four stale `m4_repor*` jobs in the queue are
`DependencyNeverSatisfied` leftovers and can be cancelled.

## 3. Open threads, in suggested order

1. **Aggregate the rhythm evals:** `python experiments/make_multilen_strat_results.py`
   (py3.10-safe) and read `rhy%`, `joint%`, `baseRhy%`, `rcnt=%` for `rhythm` (melodic-only
   prompts, comparable with `inv`) and `rhythm_rc` (rhythm-conditioned). Caveat: melodic and
   rhythmic training motifs were found independently (same snippet only ~20% of the time), so
   *joint* containment is not what training optimized; decide what "rhythm containment" should
   mean before drawing conclusions.
2. **Decide on the Space:** keep `inv` or publish the rhythm build under a new name.
3. **Training schedule:** the rhythm run's real phase overfit monotonically after epoch 3
   (eval 0.142 to 0.212 by epoch 7). Use 2-3 real epochs or early stopping next time.
4. **Bias knobs:** uniform bias 5 is the only significant single improvement (+3.0 pts exact
   containment on `inv`, CI [+1.7, +4.4]); pattern-line +1 adds ~+1 n.s.; anything above +1 on
   the pattern line hurts lengths 7-8; bias 8 degenerates (pieces 2x longer, time-cap failures).
   The bottleneck is the model's own `%motif:abc` realization: valid 84% at length 4 falling
   to 33% at length 10, mostly 1-2 wrong intervals at the pattern ends.
5. **Offline metrics not in the pipeline** (realization-in-piece containment, instance
   containment, count MAE) were computed ad hoc from saved `.abc` + `results.jsonl` with
   `multilen_gen.find_realization_in_piece` / `realization_matches_pattern`. Consider adding
   them to `multilen_strat_eval.py`.
6. **Paper:** fold multilen/inv/rhythm results in; refresh figures.

## 4. Memories carried over

These are the agent's persistent memory files for this project, lightly edited so that
script paths reflect the new `scripts/` layout. Keep the dated structure when extending them.

### accompgen-multilen-eval-pipeline

The eval loop is `experiments/multilen_strat_eval.py` (generate N pieces per motif, count
exact/orbit/inversion hits), submitted per (length 4-10 x stratum A/B/C) + base = 22 jobs
by `scripts/submit_strat_eval_multilen.sh <tag>`; aggregate with
`experiments/make_multilen_strat_results.py`. Results under
`<scratch>/strat_eval_multilen/<tag>/`. Checkpoints in `<scratch>/weights/`.

For a new checkpoint: add a tag to the `MODELS` map in the submit script, submit a 1-piece
smoke run (`EVAL_N=1`, point `EVAL_ROOT` at a separate smoke dir), then
`EXTRA_SBATCH="--dependency=afterok:<jid> --kill-on-invalid-dep=yes" scripts/submit_strat_eval_multilen.sh <tag>`.
Jobs used `--gres=gpu:a5000:1` in the `notagen` env; Blackwell GPUs need `notagen-bw`.

Results (2026-09-21): avg len 4-10 exact/orbit containment: `inv` 48.6/55.7, `multilen`
47.7/54.6, `trans` 45.8/53.6. Inversion containment unchanged (~20%). Header self-report is
weak: inversion instance emitted 41%, correct 32% of those; declared rectus count matches
11%, inversion count 58%.

Bias sweep on `inv`: `EVAL_BIAS=5` gives exact 51.6 / orbit 58.7 (+3.0 pts paired over 168
motifs, CI [+1.7, +4.4]), pieces ~4% longer; 6 = 48.0, 7 = 37.6, 8 degenerates. Bias is a
config value at inference, not a learned weight. `EVAL_PATTERN_EXTRA=x` adds bias on the
`%motif:v1:` line only (`motif_bias_extra` in `notagen_core/model.py`, `pattern_bias_extra`
in `multilen_gen.generate_piece`): +1 raises realization validity 63.6 to 67.1% but
containment only +0.9/+1.1 (CIs cross 0); +2 hurts (-3.7). Best so far: bias 5 + pattern +1
(`inv_bias5_pat1`): 52.7 / 59.9.

Filename collision bug (fixed 2026-09-21): `sanitize()` dropped `-`, so motifs differing only
in sign shared `.abc` filenames (96/168 motifs). `results.jsonl` metrics were never affected;
anything re-reading `.abc` files from runs before the fix (`inv`, `inv_bias5-8`, `multilen`,
`trans`, first `inv_pat1`) must use only motifs with no later same-name motif in the run.
Corrected realization-in-piece on non-overwritten pieces: 61.0 strict / 71.7 fuzzy at bias 4,
65.1 / 75.5 at bias 5. The aggregator once had a py3.12-only nested f-string; keep it 3.10-safe.

### accompgen-rhythmic-motifs

Every `_len<L>` header carries a rhythmic motif of the same length next to the melodic one
(`motif/rhythm.py`). Header order: `%motif:v1:step_skip_leap:`, `%motif:v1:rhythm:`,
`%motif:count:`, `%motif:abc:`, `%motif:rhythm:count:`, `%motif:rhythm:abc:`, then the
inversion lines. Design decisions: see section 1.

Pipeline: `scripts/run_rhythm_data.sh` (CPU, 16 workers, 6 h) builds an isolated copy under
`<scratch>/rhythm/{lieder,irishman}/abcfiles_processed_v1` + `synthetic_motifs_multilen_v1`
with indices `data/abcfiles_processed_v1_rhythm{,_train,_eval}.jsonl` (same composition as
the inv set: Lieder + 20k-tune Irish conversion, no style preamble), ending with a verify
step. `scripts/fine_tune_multilen_rhythm_full_rtx6000.sh` (31 h on an RTX Pro 6000) is the
inv recipe with the rhythm indices and tag `multilen4to10_rhythm_v1_bias4_syn2real5`. Both
were smoke-run on 2 Lieder + 2 Irish files before submission. Readers keyed on the
`%motif:v1:` prefix must skip `%motif:v1:rhythm:` (`multilen_gen._read_motif_header` does).

Eval wiring (2026-09-28): `notebook/make_rhythm_sets_multilen.py` writes
`notebook/rhythm_sets_multilen.json` (rhythm patterns per length, A/B/C scheme, seed
20260928; top ranks are uniform 1/2s, 1s, 1/4s). `multilen_strat_eval.py --rhythm-sets`
pairs melodic prompt i with rhythm prompt i, sends `%motif:v1:rhythm:` right after the
contour line, records `rhythm_hits[r] = {exact, joint}` (joint = exact melodic occurrence
whose first/last note chars coincide with a rhythm occurrence; approximate) and
`declared_rhythm_count`. Submit tag `rhythm` = melodic-only prompts; `EVAL_RHYTHM=1` gives
`rhythm_rc`. Three `rhythm_rc` jobs (len 7C/8C/10C) first died on a length assertion because
tail rhythm sets contain rest tokens (`1z`) that do not count toward the length; the check is
now rest-aware and those three were resubmitted and completed. Lesson: the smoke covered
len6_A only, which has no rests.

Irish corpus rhythm motifs are mostly all-1s (uniform eighths in 6/8), so the rhythm signal
may be weak there; Lieder is more varied. Local 1-piece smokes: with rhythm conditioning the
model contained the rhythm in 6/8 pieces and the joint motif in 2/8 at len6_A; unprompted, it
writes its own rhythm lines.

UI: `hf_space/motifs.build_prompt` returns (prompt, pattern, snippet, rhythm); abstract mode
takes an optional rhythm string (trailing API arg), concrete mode always emits the rhythm
block; verification with a rhythm requires contour AND durations scaled by the score's `M:`
denominator. The hf_space and gradio prompt builders otherwise emit melodic-only prompts.

### accompgen-motigen-local-hosting (Duke OnDemand specifics, adapt)

The `notagen` env needed `gradio==5.49.1` (5.17 crashes on `/gradio_api/info`) and
`music21==9.7.1` for playback. Launch from `hf_space/`:
`no_proxy='*' PORT=7860 SERVER_NAME=127.0.0.1 GRADIO_ROOT_PATH="<proxy prefix, no trailing slash>" python app.py`.
The model loads lazily on the first request (~10 s from the gz export). Drive it headlessly
with `gradio_client.Client("http://127.0.0.1:7860/")`, `api_name="/generate"`, args: mode,
abstract, notes-dataframe dict, style label, bars, tempo, temperature, seed, occurrences,
rhythm. With the rhythm checkpoint: 12 candidates in 48 s, 1 eligible under contour+rhythm
verification.

### test-pipeline-before-batch-submit (feedback)

Before kicking off batch job sets, run the full pipeline end to end on a single sample. Seven
of twenty long eval jobs once crashed at the final aggregation step on a trivial KeyError
after 10-21 h each. Gate batches on a smoke job (`--dependency=afterok`).

## 5. Cluster-specific facts to re-map (Duke values)

| Item | Duke value | Where it appears |
|---|---|---|
| Scratch root | `/usr/xtmp/cy232/accompgen` | `scripts/*.sh`, `finetune/config.py`, `hf_space/prepare.py`, `experiments/*.py`, `notebook/*.py`, the `.jsonl` indices (absolute paths) |
| Pretrained NotaGen-X | `<scratch>/pretrain/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth` | `finetune/config.py` `PRETRAINED_PATH` |
| Conda | `mamba shell hook` with fallback `source /home/users/cy232/miniforge3/etc/profile.d/conda.sh`; env `notagen`; Blackwell env `/usr/xtmp/cy232/envs/notagen-bw` | scripts |
| SLURM | partitions `compsci` (CPU), `compsci-gpu`; gres `gpu:a5000:1` (eval, 24 GB OK), `gpu:a6000:1`, `gpu:rtx_pro_6000:1` (training); `--exclude=compsci-cluster-fitz-02` (stale scratch mount) | scripts |
| Interactive GPU | `srun -p compsci-gpu -G 1 --pty bash -l` | upstream README note |
| W&B | `WANDB_KEY` env var (or `wandb login`); `WANDB_CACHE_DIR=<scratch>/wandb_cache`. The old key was committed in history (commit da53c4f) and should be rotated | `finetune/config.py` |
| Home quota | full; heavy outputs must go to scratch | build scripts already do this |

Scratch tree to migrate (names under the scratch root): `weights/` (checkpoints + logs,
essential), `pretrain/` (NotaGen-X, essential), `irishman/abc` (converted Irish source,
regenerable from HF `sander-wood/irishman`), `lieder/`, `irishman/`, `rhythm/`,
`synthetic_motifs_multilen_v1/`, `synthetic_motifs_v1/` (processed data; regenerable in
~6 h from `data/abcfiles` + `irishman/abc`), `strat_eval_multilen/` and `strat_eval_*_smoke/`
(eval outputs, needed for any re-analysis), `motifcheck_out/` (executed legacy notebooks,
provenance only), `irish1k/ irish50k/ irishfull/` (legacy single-length sets), `distil/`
(experimental), `wandb_cache/`, `wandb_local/` (droppable). After copying, rewrite the
absolute paths in `data/*.jsonl` with `sed` to the new root.

The repo's own Lieder sources (`data/abcfiles`, `data/Lieder`) and the processed single-length
Lieder dir (`data/abcfiles_processed_v1`) are gitignored; copy them along with the repo.

## 6. Conventions and gotchas

- Submit from the repo root (`sbatch scripts/x.sh`); `submit_*.sh` helpers `cd` there
  themselves. Under `sbatch`, `BASH_SOURCE` is the spool copy, so scripts fall back to
  `$SLURM_SUBMIT_DIR` to find the repo.
- `2_data_preprocess.py` APPENDS to existing index files; always back up and clear (the build
  scripts do). The `.jsonl` indices store only `{path,key}`; motif annotations live inside the
  per-key `.abc` files.
- Pre-create output dirs before launching concurrent jobs; first-starters raced on `mkdir`
  over NFS.
- No `ls | head` under `set -o pipefail` on huge directories: the SIGPIPE fails the job after
  the work is done and strands anything chained `afterok`.
- The whole alphabet must stay within 128 ASCII chars (the char-level decoder); the verify
  steps check for non-ASCII bytes.
- Lengths in the eval are counted in notes; rest tokens in rhythm sets do not count.
- The legacy notebook evals (`scripts/motif_check/`) are generated files; edit the
  `notebook/make_*.py` generators, not the scripts.
- `experiments/scripts/` is a frozen archive for `MANIFEST.md`; `experiments/MANIFEST.md`
  also notes a label confound between the `irish20k_evalfix` model and later rebuilds.
- `.claude/settings.json` in the repo allows `python3` invocations and adds `notebook/` as
  an extra directory; re-create permissions as needed on the new machine.

## 7. First actions on the new cluster

1. Clone the repo, copy the scratch tree, rewrite the index paths, install the env, run
   `pytest hf_space/tests` and `python data/test_rhythm_motif.py` as a sanity check.
2. Remap the strings in section 5 (grep command in `scripts/README.md`).
3. Smoke-run `scripts/run_strat_eval_multilen.sh` on one piece
   (`EVAL_N=1 EVAL_TAG=smoke EVAL_WEIGHTS=<inv .pth> EVAL_STRATUM=A EVAL_LENGTH=6 EVAL_ROOT=<scratch>/strat_eval_smoke`)
   before trusting the GPU/env pairing.
4. Then pick up section 3, item 1.
