# AccompGen / MotiGen: motif-conditioned symbolic music generation

A research fork of [NotaGen](https://github.com/ElectricAlexis/NotaGen) that adds
**motif conditioning**: the prompt carries a short, transposition-invariant melodic
pattern (4 to 10 notes, optionally with a rhythm), and the fine-tuned model is trained
to develop that motif through the piece. The user-facing name of the model and demo is
**MotiGen**; the paper draft is *Getting Motif-ated: Controllable AI Compositions from
Injected Motif Prompts* (`latex/`). State of the project as of **2026-10-02**.

Two ingredients make the conditioning work:

1. **A `%motif:` header block** in the ABC prompt (see *Data format*), annotated on
   every training piece by `motif/extract.py` (melodic) and `motif/rhythm.py` (rhythmic).
2. **A trained-in motif attention bias**: a scalar (+4) added to the attention logits of
   the prompt's motif patches in the patch-level encoder (`notagen_core/model.py`), applied
   identically at training and inference. Training is a two-phase curriculum: synthetic
   crops (the motif excerpt plus a few bars of context) first, then full real pieces.

## Project status

**Done**

- Multi-length (4-10 note) motif datasets from OpenScore Lieder (1,217 pieces) and the
  Irishman folk corpus (~216k tunes, V:1 melody only), with synthetic crops.
- Four full fine-tunes of NotaGen-large on that data, all comparable (2 synthetic + 5 real
  epochs, bias 4): `multilen` (baseline), `trans` (inversion/retrograde folded into the
  canonical motif), `inv` (explicit inversion count + instance in the header), `rhythm`
  (adds a rhythmic motif block). Plus a LoRA variant of `inv`.
- Stratified containment evaluation (`experiments/multilen_strat_eval.py`): per motif length
  and training-frequency stratum, 50 pieces per motif, exact and orbit containment, inversion
  containment, header self-report accuracy, and rhythm hits for rhythm checkpoints.
- Inference-time bias sweep on `inv`: uniform bias 5 is the best single change
  (+3.0 pts exact containment, 95% CI [+1.7, +4.4]); an extra +1 on the pattern line only
  helps realization validity but not containment significantly; bias 6+ degrades.
- The MotiGen Gradio Space (`hf_space/`, deployed as `cpyang/motigen` with the `inv`
  checkpoint); a static supplementary site (`website/`); the paper draft (`latex/`).
- A large legacy campaign of single-length (4-note) models and ablations
  (`experiments/MANIFEST.md`, `experiments/*.csv`).

**Open**

- The 44 `rhythm` / `rhythm_rc` eval jobs completed on 2026-09-28 but have **not been
  aggregated** yet: run `python experiments/make_multilen_strat_results.py` and read the
  `rhy%` / `joint%` columns. Decide how rhythm containment should be scored.
- The local `hf_space/` bundle currently exports the `rhythm` checkpoint
  (`hf_space/checkpoint.json`); the live Space still serves `inv`. Deploying needs a new
  Space name (`deploy.py` refuses to overwrite).
- The real phase of the `rhythm` run overfit after epoch 3 (5 real epochs is too many);
  future runs should use 2-3 real epochs or early stopping.
- Paper draft and figures need the multilen / rhythm results folded in.
- Distillation to a larger patch encoder (`distillation/`) is experimental and not part of
  any reported result.

Headline numbers (average over lengths 4-10, piece-level containment of the prompted motif
in the generated melody; exact / orbit, in %):

| Model tag | Exact | Orbit | Notes |
|---|---|---|---|
| `multilen` | 47.7 | 54.6 | baseline |
| `trans` | 45.8 | 53.6 | I/R/RI folded |
| `inv` | 48.6 | 55.7 | inversion header; current Space model |
| `inv`, bias 5 | 51.6 | 58.7 | inference bias only |
| `inv`, bias 5 + pattern +1 | 52.7 | 59.9 | best configuration so far |
| `rhythm`, `rhythm_rc` | pending | pending | jobs done, aggregation not run |

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | **All shell scripts** (SLURM jobs and submit helpers). `scripts/motif_check/` holds the 952 generated legacy eval jobs. See `scripts/README.md`. |
| `data/` | Preprocessing (`2_data_preprocess.py`), Irishman conversion, synthetic crop generator, style preamble tool, tests, and the small `.jsonl` train/eval indices. Heavy outputs live on scratch. `data/README.md` documents the pipeline and header format. |
| `motif/` | `extract.py` (step/skip/leap motif extraction, enharmonic-aware, transformations) and `rhythm.py` (rhythmic motif backbone). |
| `notagen_core/` | Model definition with the motif attention-bias hook and pluggable backbones. Vendored into `hf_space/vendor/`. |
| `finetune/` | `train-gen.py` (two-phase curriculum, LoRA option), `config.py` (env-overridable), `eval_loss.py`. |
| `experiments/` | Current eval (`multilen_strat_eval.py`), aggregators, results CSVs, figure deck builder, `MANIFEST.md` for the legacy campaign, archived launch scripts. |
| `notebook/` | Motif-check notebooks (legacy eval), their generators (`make_*.py`), `multilen_gen.py` (generation helper used by the eval), motif/rhythm prompt-set builders. |
| `hf_space/` | MotiGen Gradio Space: `app.py`, `engine.py`, `motifs.py`, `verification.py`, `prepare.py` (export checkpoint), `deploy.py`, tests. |
| `website/` | Static example showcase (peer-review supplementary site). |
| `latex/` | Paper draft (NeurIPS Creative AI format). |
| `distillation/` | Experimental patch-encoder distillation (teacher corpus + stage-1 training). |
| `exploratory/` | Motif distribution analysis notebook. |
| `slurm_logs/` | Job logs (`slurm-<job>-<id>.out`, `strateval-<name>-<id>.out`); gitignored. |
| `pretrain/`, `RL/`, `clamp2/`, `inference/`, `gradio/` | Retained from upstream NotaGen (pre-training, CLaMP-DPO, original demo). Not used by the motif experiments; see the upstream README for their usage. |
| `HANDOFF.md` | Working notes, decisions, and gotchas for whoever (or whichever agent) continues the project. |

## Environment

```bash
conda create --name notagen python=3.10
conda activate notagen
conda install pytorch==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install accelerate optimum
pip install -r requirements.txt            # transformers 4.40, abctoolkit, samplings, wandb ...
pip install jupyter nbconvert pytest        # notebook evals, hf_space tests
pip install gradio==5.49.1 music21==9.7.1  # only to run hf_space/ locally (overrides the 5.17 pin)
```

A second environment (`notagen-bw`, newer torch) was needed for Blackwell GPUs
(RTX Pro 6000, sm_120); the cu11.8 torch 2.3 build does not run there. Scripts that
target that GPU activate it explicitly.

Pretrained starting point: NotaGen-X large
(`weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth`, from the
[NotaGen Hugging Face repo](https://huggingface.co/ElectricAlexis/NotaGen)); its path is
`PRETRAINED_PATH` in `finetune/config.py`.

## Cluster conventions

- **Submit everything from the repository root**: `sbatch scripts/<job>.sh`. SLURM log paths
  and the `cd finetune/` / `cd notebook/` steps are relative to the submission directory.
- Each experiment gets a **new `EXP_TAG`**; existing checkpoints are never overwritten.
  Checkpoints and per-epoch logs land in the scratch `weights/` directory as
  `weights_notagen_<tag>_<arch>.pth` (plus `_phase1.pth` for the end of the synthetic phase)
  and `logs_notagen_<tag>_<arch>.txt`.
- **Smoke-test before fanning out**: run the job on one piece / two files first, then chain the
  batch behind it with `EXTRA_SBATCH="--dependency=afterok:<jid> --kill-on-invalid-dep=yes"`.
- Heavy data (processed ABC folders, synthetic crops, eval outputs, executed notebooks,
  checkpoints) lives under a scratch root (`/usr/xtmp/cy232/accompgen` at Duke). Only the
  `.jsonl` indices are in the repo, and they contain absolute paths into that root.
- Porting: `scripts/README.md` lists every cluster-specific string (scratch root, partitions,
  GPU gres names, env activation) to remap on a new machine.

## Data format

Each training file is an ABC piece whose header carries the motif block. For the rhythm-format
set (the most recent), per length `L` in 4..10 there is one file `<piece>_len<L>_<key>.abc`
per key (15 keys) with:

```
%motif:v1:step_skip_leap: 0,1,-1,-3        # abstract pattern: 0 start, ±1 step, ±2 skip, ±3 leap
%motif:v1:rhythm: 1,1/2,1/2,2              # duration ratios to the beat (meter denominator); Nz = rest
%motif:count: 6                            # occurrences of the melodic motif in the piece
%motif:abc: ...                            # concrete excerpt of the motif
%motif:rhythm:count: 4
%motif:rhythm:abc: ...
%motif:inversion_count: 2                  # inv/rhythm formats only
%motif:abc:inversion_instance: ...         # only when inversion_count > 0
```

Older sets differ only in which lines exist: `multilen` (melodic lines only), `trans`
(`%motif:abc` carries `I:`/`R:`/`RI:` examples), `inv` (adds the inversion lines).
Synthetic crops are the motif excerpt with a few bars of context, 3 per piece per length
(21 per piece). The train/eval split is at piece level before the per-length expansion.

Build scripts (`scripts/`): `run_multilen_data.sh` (shared v1 indices),
`run_rhythm_data.sh` (isolated rhythm copy), `run_preprocess_irishman.sh` (Irish
conversion + merge), `run_build_irish{1k,50k,full}.sh` (legacy single-length sets).
`2_data_preprocess.py` **appends** to existing indices; the build scripts back up and clear
them first. Details and env overrides: `data/README.md`.

## Training

`finetune/train-gen.py` reads everything from `finetune/config.py`, which is env-overridable:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_TRAIN_INDEX_PATH`, `DATA_EVAL_INDEX_PATH` | `../data/abcfiles_processed_v1_{train,eval}.jsonl` | Index files |
| `EXP_TAG` | `multilen4to10_v1_bias4_syn2real5` | Checkpoint / log name |
| `NUM_EPOCHS_SYNTHETIC`, `NUM_EPOCHS_REAL` | 2, 5 | Curriculum phases (each saves on improved eval loss) |
| `MOTIF_ATTENTION_BIAS` | 4.0 | Trained-in bias on motif patches (0 for the bias ablation) |
| `USE_LORA`, `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT` | 0, 16, 32, 0.05 | LoRA on attention/MLP projections of both encoder and decoder; checkpoints store merged weights plus the adapter |
| `LEARNING_RATE` | 1e-5 | 2e-4 for LoRA runs |
| `LOAD_FROM_CHECKPOINT`, `SKIP_SYNTHETIC_PHASE` | 0, 0 | Resume into the real phase from a saved synthetic checkpoint |

Each `scripts/fine_tune_*.sh` is one recipe (see `scripts/README.md`). A full run on the
multi-length set takes about 31 h on an RTX Pro 6000 (data build 6 h on 16 CPUs). Weights &
Biases logging is on by default; the key comes from the `WANDB_KEY` environment variable
(or a prior `wandb login`).

## Evaluation

The current loop is `scripts/submit_strat_eval_multilen.sh <tag ...>`, which submits, per
model tag, 7 lengths x 3 strata (A = frequent, B = mid, C = rare motifs in the training
distribution; 8 motifs x 50 pieces each) plus one no-prompt base run of 50 pieces = 22 jobs
of `scripts/run_strat_eval_multilen.sh`. Knobs via environment:

- `EVAL_BIAS=5` inference bias override (tag suffix `_bias5`),
- `EVAL_PATTERN_EXTRA=1` extra bias on the `%motif:v1:` line only (`_pat1`),
- `EVAL_RHYTHM=1` rhythm-conditioned prompts from `notebook/rhythm_sets_multilen.json` (`_rc`),
- `EVAL_N=1` for smoke runs, `EXTRA_SBATCH` for dependencies.

Outputs: `<scratch>/strat_eval_multilen/<tag>/{len<L>_<S>,base}/results.jsonl` plus the
generated `.abc` files. Aggregate with `python experiments/make_multilen_strat_results.py`
into `experiments/multilen_strat_results.csv`. Note that runs before 2026-09-21 have
overwritten `.abc` files for sign-only motif name collisions (fixed in `sanitize()`); the
`results.jsonl` metrics were never affected.

Legacy evaluation (single-length models, 2026-06 to 2026-08) executed one notebook per
(model, motif) headlessly: `scripts/motif_check/run_motif_check_*.sh`, generated by
`notebook/make_*.py`, results consolidated by `experiments/make_multimotif_results.py` and
`make_results_csv.py`. `experiments/MANIFEST.md` documents the models, motif set and ablations.

## Demo

`hf_space/` is the MotiGen Space (Gradio 5.49.1). `python prepare.py [--checkpoint <pth>]`
exports a checkpoint to `model.safetensors.gz` + `checkpoint.json`; `python deploy.py <repo>`
publishes the allowlisted bundle. The app accepts an abstract contour (drag editor, 4-10
notes, optional rhythm string) or a concrete note list, generates several candidates, verifies
the motif in the output, and shows score, playback, ABC and MusicXML. Run locally with
`PORT=7860 SERVER_NAME=127.0.0.1 python app.py` from `hf_space/` (the model loads on the
first request); drive it headlessly with `gradio_client` on `api_name="/generate"`.
Tests: `pytest hf_space/tests`.

## Paper

`latex/main.tex` (NeurIPS Creative AI track, build with `pdflatex main && bibtex main &&
pdflatex main && pdflatex main`). Figures come from `experiments/make_paper_figures_pptx.py`
(editable deck `experiments/paper_figures.pptx`) and `experiments/figures/`.

## Upstream

NotaGen (pre-training on 1.6M pieces, fine-tuning with period-composer-instrumentation
prompts, CLaMP-DPO reinforcement learning) by Wang et al., 2025:
[paper](https://arxiv.org/abs/2502.18008), [weights](https://huggingface.co/ElectricAlexis/NotaGen),
[code](https://github.com/ElectricAlexis/NotaGen). The `pretrain/`, `RL/`, `clamp2/`,
`inference/` and `gradio/` directories and `data/{1_batch_xml2abc,3_batch_abc2xml}.py` come
from there; the upstream README documents them.

```bibtex
@misc{wang2025notagenadvancingmusicalitysymbolic,
      title={NotaGen: Advancing Musicality in Symbolic Music Generation with Large Language Model Training Paradigms},
      author={Yashan Wang and Shangda Wu and Jianhuai Hu and Xingjian Du and Yueqi Peng and Yongxin Huang and Shuai Fan and Xiaobing Li and Feng Yu and Maosong Sun},
      year={2025},
      eprint={2502.18008},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2502.18008},
}
```
