"""Helpers for multilen_demo.ipynb: generate ONE piece from the multi-length motif model.

The multi-length model (EXP_TAG multilen4to10_v1_bias4_syn2real5) was trained on
preamble-free V:1 files whose header is exactly

    %motif:v1:step_skip_leap: <pattern>
    %motif:abc: <realization>
    %%score 1
    L:...  Q:...  M:...  K:...  V:1 ...
    [V:1]...

so the prompt we feed it is just the two %motif lines; the model writes the rest.
A motif can be given three ways (see the notebook):
  * an abstract step/skip/leap pattern of length 4-10  -> realized from real training data
    in the requested key (or algorithmically if that pattern was never seen),
  * your own ABC realization                             -> abstracted into a pattern,
  * a random real motif of a chosen length.

Everything model-side is imported from the repo (notagen_core, finetune/utils.py,
motif/extract.py); nothing here re-implements the model.
"""
from __future__ import annotations

import os
import re
import sys
import glob
import time
import random
import importlib
import importlib.util
import subprocess
from pathlib import Path

import torch

NB_DIR = Path(__file__).resolve().parent
PROJ_ROOT = NB_DIR.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from motif import extract as mx  # single motif backbone

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
BEST_CKPT = ('/usr/xtmp/cy232/accompgen/weights/weights_notagen_multilen4to10_v1_bias4_syn2real5'
             '_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1.pth')
# Transformation-aware retrain (I/R/RI folded into the canonical motif, 2026-09): same recipe,
# best real-phase checkpoint. Containment ≈ BEST_CKPT (see experiments/multilen_strat_results.csv).
TRANS_CKPT = ('/usr/xtmp/cy232/accompgen/weights/weights_notagen_multilen4to10_trans_v1_bias4_syn2real5'
              '_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1.pth')

# Real multi-length training data (one file per piece, length and key).
REAL_DIRS = [
    '/usr/xtmp/cy232/accompgen/lieder/abcfiles_processed_v1',
    '/usr/xtmp/cy232/accompgen/irishman/abcfiles_processed_v1',
]
ALL_KEYS = ['Cb', 'C', 'C#', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
MOTIF_LENGTHS = range(4, 11)
INTERVAL_MODE = 'step_skip_leap'
PATCH_SIZE = 16
PATCH_LENGTH = 1024

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _load_finetune_utils():
    """Import finetune/utils.py (Patchilizer) with finetune/ on sys.path so its
    `from config import *` resolves to finetune/config.py."""
    ft = str(PROJ_ROOT / 'finetune')
    if ft not in sys.path:
        sys.path.insert(0, ft)
    for m in ('config', 'utils'):
        sys.modules.pop(m, None)
    return importlib.import_module('utils')


def load_model(weights_path: str = BEST_CKPT, motif_attention_bias: float = 4.0, device=None):
    """Build the gpt2/gpt2 NotaGen and load a fine-tuned checkpoint. Returns (model, patchilizer, device)."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ft_utils = _load_finetune_utils()
    from notagen_core import build_notagen_configs, NotaGenLMHeadModel

    patch_config, char_config = build_notagen_configs(
        encoder_backbone='gpt2', decoder_backbone='gpt2',
        patch_num_layers=20, char_num_layers=6, hidden_size=1280,
        patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE,
        motif_attention_bias=motif_attention_bias, patch_sampling_batch_size=0)
    model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=char_config)
    ckpt = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    info = {k: ckpt.get(k) for k in ('epoch', 'best_epoch', 'min_eval_loss', 'phase')}
    del ckpt
    model = model.to(device).eval()
    print(f'Loaded {os.path.basename(weights_path)}\n  checkpoint info: {info}\n  device: {device}, '
          f'params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M')
    return model, ft_utils.Patchilizer(stream=True), device


# ---------------------------------------------------------------------------
# Motif utilities
# ---------------------------------------------------------------------------

def parse_pattern(text) -> tuple:
    """'0,1,1,-2,3' / '0 1 1 -2 3' / (0,1,1,-2,3) -> (0, 1, 1, -2, 3). Validates length + values."""
    if isinstance(text, (list, tuple)):
        pat = tuple(int(x) for x in text)
    else:
        pat = tuple(int(x) for x in re.split(r'[,\s]+', str(text).strip()) if x != '')
    if not pat or pat[0] != 0:
        raise ValueError(f'pattern must start with 0 (the reference note): {pat}')
    if any(abs(x) > 3 for x in pat):
        raise ValueError(f'step_skip_leap values are 0, ±1 (step), ±2 (skip), ±3 (leap): {pat}')
    if len(pat) not in MOTIF_LENGTHS:
        raise ValueError(f'motif length {len(pat)} not in 4..10')
    return pat


def abstract_realization(abc_snippet: str, key: str) -> tuple:
    """Turn a concrete ABC fragment into its step/skip/leap pattern (repeated notes collapsed).
    Uses the same backbone as data preprocessing so the result matches training labels."""
    pitches, *_ = mx.abc_to_pitches_with_bars(abc_snippet, key=key)
    if len(pitches) < 2:
        raise ValueError(f'could not parse at least 2 notes from {abc_snippet!r}')
    pat_chrom = mx.motif_pattern_relative_to_first(list(pitches))
    return tuple(int(x) for x in mx._coarsen_pattern(pat_chrom, INTERVAL_MODE))


def _iter_real_files(length: int, key: str):
    for d in REAL_DIRS:
        yield from glob.iglob(os.path.join(d, key, f'*_len{length}_{key}.abc'))


def _read_motif_header(path: str):
    """Return (pattern_tuple, abc_snippet) from a processed file's %motif lines, or None."""
    pat = abc = None
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if line.startswith('%motif:v1:') and not line.startswith('%motif:v1:rhythm:'):
                pat = tuple(int(x) for x in line.split(':', 3)[-1].replace(';', ' ').split(',') if x.strip())
            elif line.startswith('%motif:abc: '):     # NOT '%motif:abc:inversion_instance:'
                abc = line[len('%motif:abc:'):].strip()
            elif line.startswith('[V:'):
                break
    if pat is None or not abc:
        return None
    return pat, abc


def find_real_realizations(pattern: tuple, key: str, max_hits: int = 20, max_files: int | None = None):
    """Scan the real _len<L> files in `key` for pieces whose motif IS `pattern`; return their
    %motif:abc realizations (deduplicated, in scan order). One scan is ~21k small files."""
    L = len(pattern)
    hits, seen = [], set()
    for n, path in enumerate(_iter_real_files(L, key)):
        if max_files is not None and n >= max_files:
            break
        got = _read_motif_header(path)
        if got and got[0] == pattern and got[1] not in seen:
            seen.add(got[1]); hits.append(got[1])
            if len(hits) >= max_hits:
                break
    return hits


def algorithmic_realization(pattern: tuple, key: str, seed: int | None = None) -> str:
    """Fallback realization via data/generate_synthetic_motifs.pattern_to_abc."""
    spec = importlib.util.spec_from_file_location(
        'gen_syn', PROJ_ROOT / 'data' / 'generate_synthetic_motifs.py')
    gs = importlib.util.module_from_spec(spec); spec.loader.exec_module(gs)
    if seed is not None:
        random.seed(seed)
    return gs.pattern_to_abc(list(pattern), start_note=gs.KEY_TO_START_NOTE.get(key, 'C'))


def random_real_motif(length: int, key: str, seed: int | None = None):
    """Pick a random real piece's length-L motif in `key`. Returns (pattern, abc, source_path)."""
    files = list(_iter_real_files(length, key))
    if not files:
        raise FileNotFoundError(f'no real _len{length} files for key {key}')
    rng = random.Random(seed)
    for _ in range(50):
        p = rng.choice(files)
        got = _read_motif_header(p)
        if got:
            return got[0], got[1], p
    raise RuntimeError('could not find a file with a motif header')


def build_prompt(pattern: tuple, realization: str | None = None,
                 rhythm: tuple | list | str | None = None) -> list:
    """Prompt lines in the exact training format (trailing space, newline).

    realization=None -> only the abstract `%motif:v1:` line is sent and the model writes its
    own `%motif:abc:` realization (see extract_generated_realization). Otherwise both lines
    are sent; the pair is checked for consistency first.

    rhythm (rhythm-format checkpoints, 2026-09): ratio tokens of the rhythmic motif
    (('1','1/2','1/2','2') or the header string '1,1/2,1/2,2'); emitted as the
    `%motif:v1:rhythm:` line right after the contour line, where training puts it. The
    model then writes `%motif:count:` / `%motif:abc:` / `%motif:rhythm:count:` /
    `%motif:rhythm:abc:` itself."""
    pat_str = ','.join(str(x) for x in pattern)
    lines = [f'%motif:v1:{INTERVAL_MODE}: {pat_str} \n']
    if rhythm is not None:
        rhythm_str = rhythm if isinstance(rhythm, str) else ','.join(str(t) for t in rhythm)
        lines.append(f'%motif:v1:rhythm: {rhythm_str} \n')
    if realization is not None:
        lines.append(f'%motif:abc: {realization} \n')
    return lines


def parse_rhythm_pattern(text) -> tuple:
    """'1,1/2,1/2,2' (header spelling, rests as '1z') -> ('1','1/2','1/2','2'); validated."""
    from motif import rhythm as rx
    toks = tuple(t.strip() for t in str(text).split(',') if t.strip())
    if not toks:
        raise ValueError(f'empty rhythm pattern {text!r}')
    for t in toks:
        rx.parse_ratio(t)   # raises on a malformed ratio
    return toks


def realization_matches_pattern(realization: str, pattern: tuple, key) -> bool:
    """True if `pattern` occurs as a contiguous window of `realization`'s note contour
    (this is exactly what a training-data %motif:abc line guarantees for its %motif:v1 line;
    the realization may carry a few surrounding notes)."""
    pitches, bars, *_ = mx.abc_to_pitches_with_bars(realization, key=key)
    if len(pitches) < len(pattern):
        return False
    counts, _ = mx.count_interval_motifs(list(pitches), list(bars), window_notes=len(pattern),
                                         stride_notes=1, interval_mode=INTERVAL_MODE,
                                         fold_transformations=False)
    return counts.get(tuple(pattern), 0) > 0


def check_prompt_consistency(pattern: tuple, realization: str, key) -> None:
    """Raise if the abstract pattern contradicts the concrete realization it is paired with."""
    if not realization_matches_pattern(realization, pattern, key):
        raise ValueError(
            f"pattern {pattern} is NOT the contour of realization {realization!r} in key {key} "
            f"(its own contour is {abstract_realization(realization, key)}); refusing to send a "
            f"contradictory prompt")


def extract_generated_realization(raw_text: str) -> str | None:
    """The `%motif:abc:` line the model wrote for itself (None if it produced none)."""
    for line in (raw_text or '').splitlines():
        if line.startswith('%motif:abc: '):
            return line[len('%motif:abc:'):].strip()
    return None


def extract_generated_header(raw_text: str) -> dict:
    """Every %motif line the model wrote, parsed (inversion-format training data, 2026-09):
      {'pattern': tuple|None, 'count': int|None, 'abc': str|None,
       'inversion_count': int|None, 'inversion_instance': str|None}
    Missing / unparsable lines give None."""
    out = {'pattern': None, 'count': None, 'abc': None, 'inversion_count': None, 'inversion_instance': None,
           # rhythm-format lines (rhythm checkpoints, 2026-09); None when absent
           'rhythm': None, 'rhythm_count': None, 'rhythm_abc': None}
    def _int(s):
        m = re.match(r'\s*(-?\d+)', s)
        return int(m.group(1)) if m else None
    for line in (raw_text or '').splitlines():
        if line.startswith('%motif:v1:rhythm:'):
            if out['rhythm'] is None:
                out['rhythm'] = tuple(t.strip() for t in line[len('%motif:v1:rhythm:'):].split(',') if t.strip()) or None
        elif line.startswith('%motif:v1:') and out['pattern'] is None:
            try:
                out['pattern'] = tuple(int(x) for x in line.split(':', 3)[-1].split(',') if x.strip())
            except ValueError:
                pass
        elif line.startswith('%motif:rhythm:count:') and out['rhythm_count'] is None:
            out['rhythm_count'] = _int(line[len('%motif:rhythm:count:'):])
        elif line.startswith('%motif:rhythm:abc: ') and out['rhythm_abc'] is None:
            out['rhythm_abc'] = line[len('%motif:rhythm:abc:'):].strip()
        elif line.startswith('%motif:count:') and out['count'] is None:
            out['count'] = _int(line[len('%motif:count:'):])
        elif line.startswith('%motif:abc: ') and out['abc'] is None:
            out['abc'] = line[len('%motif:abc:'):].strip()
        elif line.startswith('%motif:inversion_count:') and out['inversion_count'] is None:
            out['inversion_count'] = _int(line[len('%motif:inversion_count:'):])
        elif line.startswith('%motif:abc:inversion_instance:') and out['inversion_instance'] is None:
            out['inversion_instance'] = line[len('%motif:abc:inversion_instance:'):].strip()
        elif line.startswith('[V:'):
            break
    return out


def inversion_pattern(pattern: tuple) -> tuple:
    """The inversion (every interval negated) of a step/skip/leap pattern."""
    return mx._pattern_transforms(tuple(int(x) for x in pattern))[1]


def inversion_instance_is_correct(instance: str, pattern: tuple, key) -> bool:
    """True if the model's `%motif:abc:inversion_instance:` excerpt really contains the
    inversion of `pattern` as a contiguous window of its contour (same check as
    realization_matches_pattern, applied to the inverted pattern)."""
    try:
        return realization_matches_pattern(instance, inversion_pattern(pattern), key)
    except Exception:
        return False


def generated_key(abc_text: str):
    """K: field of a generated piece (the model picks the key), or None."""
    return _v1_text_and_key(abc_text)[1] if abc_text else None


# ---------------------------------------------------------------------------
# Generation (adapted from the motif-check notebooks' inference_patch)
# ---------------------------------------------------------------------------

def _rest_unreduce_v1(abc_lines):
    """V:1-only variant of inference/inference.py:rest_unreduce — strips [r:..]
    stream markers; nothing else needs unreducing with a single voice."""
    out = []
    for line in abc_lines:
        if line.startswith('[V:') or line.startswith('[r:'):
            line = re.sub(r'^\[r:[^\]]*\]', '', line)
        out.append(line)
    return out


PATTERN_LINE_PREFIX = '%motif:v1:'


class _MotifLineFlagger:
    """Per-patch motif flags for the attention bias, decided the way training decides them:
    a patch is a motif patch iff the LINE it belongs to starts with '%motif:'. Training sees
    whole lines; at generation the line arrives patch by patch, so the decision is made on the
    line text INCLUDING the current patch and, once a line proves to be a %motif line, its
    earlier patches are re-flagged too (the weights vector is rebuilt from `flags` every
    step, so this retro-flagging costs nothing). A patch boundary inside '%motif:' can
    therefore never leave a motif patch unbiased."""

    def __init__(self, flags=None, pattern_flags=None):
        self.flags = flags if flags is not None else []
        # parallel flags: patch belongs to the abstract '%motif:v1:' pattern line
        self.pattern_flags = pattern_flags if pattern_flags is not None else [False] * len(self.flags)
        self.cur_line = ''            # text of the current line before the next patch
        self.line_start = len(self.flags)   # index in flags of the current line's first patch

    def add(self, patch_text: str):
        head = patch_text.split('\n', 1)[0]
        line_text = self.cur_line + head
        is_motif = line_text.lstrip().startswith('%motif:')
        self.flags.append(is_motif)
        if is_motif:
            for j in range(self.line_start, len(self.flags) - 1):
                self.flags[j] = True
        is_pattern = line_text.lstrip().startswith(PATTERN_LINE_PREFIX)
        self.pattern_flags.append(is_pattern)
        if is_pattern:
            for j in range(self.line_start, len(self.pattern_flags) - 1):
                self.pattern_flags[j] = True
        if '\n' in patch_text:
            self.cur_line = patch_text.rsplit('\n', 1)[-1]
            self.line_start = len(self.flags)
        else:
            self.cur_line = line_text
        return self.flags


def generate_piece(model, patchilizer, device, prompt_lines, *, top_k=9, top_p=0.9, temperature=1.2,
                   max_minutes=10, print_tune=True, motif_weight=3.0, pattern_bias_extra=0.0):
    """Generate one piece from `prompt_lines` (list of header lines ending in '\\n').

    Returns (abc_text, raw_text): `abc_text` is the cleaned piece (X:1 header, %motif lines
    dropped, stream markers removed) ready for abc2xml; `raw_text` is the model's verbatim output.
    Returns (None, raw) if generation failed (length/time cap).

    `pattern_bias_extra` (inference-only): attention bias ADDED on top of the model's uniform
    %motif: bias for the patches of the abstract '%motif:v1:' pattern line only."""
    bos_patch = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]
    start_time = time.time()

    prompt_patches = patchilizer.patchilize_metadata(prompt_lines)
    byte_list = list(''.join(prompt_lines))
    context_tunebody, metadata_bytes = [], []
    if print_tune:
        print(''.join(byte_list), end='')
    prompt_patches = [[ord(c) for c in p] + [patchilizer.special_token_id] * (PATCH_SIZE - len(p))
                      for p in prompt_patches]
    prompt_patches.insert(0, bos_patch)

    # per-patch motif flags -> attention bias on %motif: patches (matches training)
    flags = [False]
    for line in prompt_lines:
        flags += [line.lstrip().startswith('%motif:')] * len(patchilizer.split_patches(line))
    def _weights(fl):
        return torch.tensor([[motif_weight if f else 1.0 for f in fl]], device=device)
    motif_weights = _weights(flags)
    pflags = [False]
    for line in prompt_lines:
        pflags += [line.lstrip().startswith(PATTERN_LINE_PREFIX)] * len(patchilizer.split_patches(line))
    flagger = _MotifLineFlagger(flags, pflags)   # flags generated patches line-wise, like training
    def _extra():
        if not pattern_bias_extra:
            return None
        return torch.tensor([[pattern_bias_extra if f else 0.0 for f in flagger.pattern_flags]], device=device)

    input_patches = torch.tensor(prompt_patches, device=device).reshape(1, -1)
    tunebody_flag = False
    failure = False

    def _gen(inp):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
            return model.generate(inp.unsqueeze(0), top_k=top_k, top_p=top_p,
                                  temperature=temperature, motif_weights=motif_weights,
                                  motif_bias_extra=_extra())

    with torch.inference_mode():
        while True:
            predicted = _gen(input_patches)
            if not tunebody_flag and patchilizer.decode([predicted]).startswith('[r:'):
                tunebody_flag = True
                r0 = torch.tensor([ord(c) for c in '[r:0/']).unsqueeze(0).to(device)
                predicted = [ord(c) for c in '[r:0/'] + _gen(torch.concat([input_patches, r0], axis=-1))
            if predicted[0] == patchilizer.bos_token_id and predicted[1] == patchilizer.eos_token_id:
                break  # EOS
            next_patch = patchilizer.decode([predicted])
            for ch in next_patch:
                byte_list.append(ch)
                (context_tunebody if tunebody_flag else metadata_bytes).append(ch)
                if print_tune:
                    print(ch, end='')

            ended = False
            for j in range(len(predicted)):
                if ended:
                    predicted[j] = patchilizer.special_token_id
                if predicted[j] == patchilizer.eos_token_id:
                    ended = True
            input_patches = torch.cat([input_patches, torch.tensor([predicted], device=device)], dim=1)

            flags = flagger.add(next_patch)
            motif_weights = _weights(flags)

            if len(byte_list) > 102400 or time.time() - start_time > max_minutes * 60:
                failure = True
                break

            if input_patches.shape[1] >= PATCH_LENGTH * PATCH_SIZE:
                # context full: keep metadata + last half of the tunebody (stream generation)
                print('\n[stream: re-encoding context]')
                tb = ''.join(context_tunebody)
                if '\n' not in tb:
                    failure = True
                    break
                tb_lines = tb.split('\n')
                tb_lines = ([l + '\n' for l in tb_lines[:-1]] + [tb_lines[-1]]) if not tb.endswith('\n') \
                    else [l + '\n' for l in tb_lines]
                cut = len(tb_lines) // 2
                sl = ''.join(metadata_bytes) + ''.join(tb_lines[-cut:])
                enc = patchilizer.encode_generate(sl)
                input_patches = torch.tensor([[x for p in enc for x in p]], device=device).reshape(1, -1)
                flagger = _MotifLineFlagger([])
                for pi in range(input_patches.shape[1] // PATCH_SIZE):
                    ptxt = patchilizer.decode([input_patches[0, pi*PATCH_SIZE:(pi+1)*PATCH_SIZE].tolist()])
                    flags = flagger.add(ptxt)
                motif_weights = _weights(flags)
                context_tunebody = list(''.join(tb_lines[-cut:]))

    raw = ''.join(byte_list)
    if failure:
        return None, raw
    lines = [l + '\n' for l in raw.split('\n') if l]
    lines = _rest_unreduce_v1(lines)
    lines = [l for l in lines if not (l.startswith('%') and not l.startswith('%%'))]
    return 'X:1\n' + ''.join(lines), raw


# ---------------------------------------------------------------------------
# Post-processing: containment check, saving, MusicXML
# ---------------------------------------------------------------------------

def _v1_text_and_key(abc_text: str):
    key = None
    v1 = []
    for line in abc_text.splitlines():
        m = re.match(r'^K:\s*(\S+)', line)
        if m and key is None:
            key = m.group(1)
        if line.startswith('[V:1]') or line.startswith('[r:'):
            v1.append(re.sub(r'^\[r:[^\]]*\]', '', line).replace('[V:1]', ''))
    return '\n'.join(v1), key


def count_motif_in_piece(abc_text: str, pattern: tuple) -> int:
    """How many times the generated melody contains `pattern` (same window logic as the
    evaluation checker: repeated notes collapsed, key signature honoured)."""
    v1, key = _v1_text_and_key(abc_text)
    pitches, bars, *_ = mx.abc_to_pitches_with_bars(v1, key=key)
    if len(pitches) < len(pattern):
        return 0
    counts, _ = mx.count_interval_motifs(list(pitches), list(bars), window_notes=len(pattern),
                                         stride_notes=1, interval_mode=INTERVAL_MODE,
                                         fold_transformations=False)
    return counts.get(tuple(pattern), 0)


def rhythm_hits_in_piece(abc_text: str, rhythm: tuple):
    """(count, occurrences, events) of the rhythmic pattern `rhythm` (ratio tokens) in the
    generated V:1 melody, measured the way the data pipeline measures it: beat = 1/(initial
    M: denominator), inline L: honoured, rests inside a window carry 'z'. The V:1 text has no
    M:/L: lines of its own, so they are read from the full piece header."""
    from motif import rhythm as rx
    v1, _ = _v1_text_and_key(abc_text)
    meter = rx.detect_meter_field(abc_text)
    unit = rx.detect_unit_length_field(abc_text) or rx.default_unit_length(meter)
    events, _s, _bs = rx.abc_to_rhythm_events(v1, unit_length=unit, meter=meter)
    L = len(rhythm)
    if sum(1 for e in events if not e['rest']) < L:
        return 0, [], events
    counts, occs = rx.count_rhythm_motifs(events, window_notes=L, stride_notes=1)
    return counts.get(tuple(rhythm), 0), occs.get(tuple(rhythm), []), events


def count_motif_forms_in_piece(abc_text: str, pattern: tuple) -> dict:
    """{'exact': n, 'orbit': n, 'by_form': {'original'|'I'|'R'|'RI': n}} — occurrences of the
    pattern and of its inversion / retrograde / retrograde-inversion in the generated melody.
    Counted with transformation folding OFF and summed over the distinct orbit members, so the
    result does not depend on which form the model happened to write first (the same metric
    as the stratified eval, experiments/multilen_strat_eval.py)."""
    v1, key = _v1_text_and_key(abc_text)
    pitches, bars, *_ = mx.abc_to_pitches_with_bars(v1, key=key)
    pat = tuple(int(x) for x in pattern)
    by_form = {}
    if len(pitches) >= len(pat):
        counts, _ = mx.count_interval_motifs(list(pitches), list(bars), window_notes=len(pat),
                                             stride_notes=1, interval_mode=INTERVAL_MODE,
                                             fold_transformations=False)
        seen = set()
        for lbl, form in zip(mx.TRANSFORM_LABELS, mx._pattern_transforms(pat)):
            if form in seen:      # degenerate patterns: R == I, or RI == original
                continue
            seen.add(form)
            c = counts.get(form, 0)
            if c:
                by_form[lbl] = c
    return {'exact': by_form.get('original', 0), 'orbit': sum(by_form.values()), 'by_form': by_form}


def save_piece(abc_text: str, out_dir, stem: str, to_xml: bool = True):
    """Write <out_dir>/<stem>.abc and (optionally) .xml via notebook/abc2xml.py. Returns paths."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    abc_path = out_dir / f'{stem}.abc'
    abc_path.write_text(abc_text, encoding='utf-8')
    xml_path = None
    if to_xml:
        r = subprocess.run([sys.executable, str(NB_DIR / 'abc2xml.py'), '-o', str(out_dir), str(abc_path)],
                           capture_output=True, text=True)
        cand = out_dir / f'{stem}.xml'
        if cand.exists():
            xml_path = cand
        else:
            print('abc2xml failed:', (r.stderr or r.stdout)[-500:])
    return abc_path, xml_path


# ---------------------------------------------------------------------------
# Fuzzy realization check: did the concrete motif notes make it into the score?
# ---------------------------------------------------------------------------

def _pitch_seq(abc: str, key, collapse_repeats: bool = True):
    """Absolute semitone pitches (+ bar index + char span in the stripped text) of an ABC
    fragment. Immediate repeats are collapsed by default, matching the motif definition."""
    pitches, bars, spans, s, _ = mx.abc_to_pitches_with_bars(abc, key=key)
    out = []
    for p, b, sp in zip(pitches, bars, spans):
        if collapse_repeats and out and int(p) == out[-1][0]:
            continue
        out.append((int(p), b, sp, getattr(p, 'dia', None)))  # (semitones, bar, char span, staff position)
    return out, s


def _levenshtein(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def find_realization_in_piece(abc_text: str, realization: str, realization_key: str, *,
                              threshold: float = 0.8, run_threshold: float = 0.7,
                              collapse_repeats: bool = True, slack: int = 2):
    """Fuzzy-search the generated melody for the prompt's concrete realization.

    Both the realization and the melody are reduced to pitch sequences; every melody window
    of length len(realization)±slack is compared to the realization by the edit distance of
    their INTERVAL sequences (semitone steps between successive notes), so a transposed
    occurrence still scores 1.0 (the transposition is reported), and one wrong / extra /
    missing note only lowers the score a little. Rhythm is ignored on purpose.

    Returns a dict:
      similarity        best window score in [0, 1] (1.0 = identical contour)
      exact_pitch       True if the best window is the very same pitches (transposition 0)
      transposition     semitones from the realization to the best window's first note
      start_bar         0-based bar (line) of the best window in the generated melody
      matched_abc       the generated notes that matched
      n_windows_over    how many windows score >= threshold (≈ number of occurrences)
      longest_run_notes longest stretch of realization notes quoted verbatim (transposition
                        allowed) anywhere in the melody, with its bar / notes / start offset
      verdict           'exact' | 'transposed' | 'diatonic' (same letters, other key) |
                        'fuzzy' (similarity >= threshold OR the
                        quoted run covers >= run_threshold of the realization) | 'absent'
    """
    ref, _ = _pitch_seq(realization, realization_key, collapse_repeats)
    v1, key = _v1_text_and_key(abc_text)
    mel, s = _pitch_seq(v1, key, collapse_repeats)
    if len(ref) < 2 or len(mel) < 2:
        return {'similarity': 0.0, 'verdict': 'absent', 'note': 'could not parse notes'}

    def _ints(seq, idx):
        vals = [x[idx] for x in seq]
        if any(v is None for v in vals):
            return None
        return [b - a for a, b in zip(vals, vals[1:])]

    ref_p = [x[0] for x in ref]
    mel_p = [x[0] for x in mel]
    L = len(ref_p)
    # interval sequences in semitone space (strict) and staff-position space (fuzzy: a
    # 2nd is a 2nd whether major/minor, so key changes / accidentals don't break a match)
    spaces = {'semitone': (_ints(ref, 0), _ints(mel, 0))}
    if _ints(ref, 3) is not None and _ints(mel, 3) is not None:
        spaces['diatonic'] = (_ints(ref, 3), _ints(mel, 3))

    best = None          # (sim, -|trans|, -|w-L|, st, w, trans, space)
    n_over = 0
    for space, (ref_int, mel_int) in spaces.items():
        for w in range(max(2, L - slack), L + slack + 1):
            for st in range(0, len(mel_p) - w + 1):
                win_int = mel_int[st:st + w - 1]
                d = _levenshtein(ref_int, win_int)
                sim = 1.0 - d / max(len(ref_int), len(win_int), 1)
                if sim >= threshold and space == 'semitone':
                    n_over += 1
                trans = mel_p[st] - ref_p[0]
                # prefer semitone matches on ties so 'exact' is reported when it truly is
                cand = (sim, space == 'semitone', -abs(trans), -abs(w - L), -st, w, trans, space)
                if best is None or cand > best:
                    best = cand
    sim, _, _, _, neg_st, w, trans, space = best
    st = -neg_st
    c0, c1 = mel[st][2][0], mel[st + w - 1][2][1]
    matched = re.sub(r'\s+', ' ', s[c0:c1]).strip()
    # Longest contiguous run of realization notes quoted verbatim (up to transposition), in
    # the fuzziest space available: longest common substring of interval sequences, +1 note.
    ref_int, mel_int = spaces.get('diatonic', spaces['semitone'])
    run_len, run_ref_i, run_mel_i = 0, 0, 0
    if ref_int and mel_int:
        prev = [0] * (len(mel_int) + 1)
        for i in range(1, len(ref_int) + 1):
            cur = [0] * (len(mel_int) + 1)
            for j in range(1, len(mel_int) + 1):
                if ref_int[i - 1] == mel_int[j - 1]:
                    cur[j] = prev[j - 1] + 1
                    if cur[j] > run_len:
                        run_len, run_ref_i, run_mel_i = cur[j], i - cur[j], j - cur[j]
            prev = cur
    run_notes = run_len + 1 if run_len else 0
    run_frac = run_notes / L
    if run_notes:
        r0, r1 = mel[run_mel_i][2][0], mel[run_mel_i + run_len][2][1]
        run_abc, run_bar = re.sub(r'\s+', ' ', s[r0:r1]).strip(), mel[run_mel_i][1]
        run_offset = run_ref_i
    else:
        run_abc, run_bar, run_offset = '', None, None
        r0 = r1 = None

    if sim >= 0.999 and space == 'semitone':
        verdict = 'exact' if trans == 0 else 'transposed'
    elif sim >= 0.999:
        verdict = 'diatonic'   # same scale-degree contour, different key/mode/accidentals
    elif sim >= threshold or run_frac >= run_threshold:
        verdict = 'fuzzy'
    else:
        verdict = 'absent'
    return {'_best_span': (c0, c1), '_run_span': ((r0, r1) if run_notes else None),
            'similarity': round(sim, 3), 'match_space': space,
            'exact_pitch': verdict == 'exact',
            'transposition': trans, 'start_bar': mel[st][1], 'matched_abc': matched,
            'n_windows_over': n_over, 'verdict': verdict, 'threshold': threshold,
            'longest_run_notes': run_notes, 'realization_notes': L, 'longest_run_frac': round(run_frac, 2),
            'longest_run_abc': run_abc, 'longest_run_bar': run_bar, 'longest_run_from_note': run_offset}


def describe_realization_match(m: dict) -> str:
    """One-line human summary of find_realization_in_piece()."""
    if 'note' in m:
        return f"realization check: {m['note']}"
    tag = {'exact': '✅ exact realization found',
           'transposed': f"✅ realization found transposed by {m['transposition']:+d} semitones",
           'diatonic': f"✅ realization found on the same scale degrees (different key/mode; first note {m['transposition']:+d} st)",
           'fuzzy': f"🟡 fuzzy match (similarity {m['similarity']:.2f} in {m['match_space']} space, transposition {m['transposition']:+d} st)",
           'absent': f"❌ realization not found (best similarity {m['similarity']:.2f})"}[m['verdict']]
    run = (f"; longest verbatim run {m['longest_run_notes']}/{m['realization_notes']} notes"
           + (f" (bar {m['longest_run_bar'] + 1}: \"{m['longest_run_abc']}\")" if m['longest_run_notes'] else ''))
    return (f"{tag} — bar {m['start_bar'] + 1}: \"{m['matched_abc']}\"  "
            f"[{m['n_windows_over']} window(s) ≥ {m['threshold']:.2f}]{run}")


# ---------------------------------------------------------------------------
# Score rendering in the notebook (abcjs, vendored in website/abcjs-basic-min.js)
# ---------------------------------------------------------------------------

_ABCJS_JS = None


def _abcjs_source() -> str:
    global _ABCJS_JS
    if _ABCJS_JS is None:
        _ABCJS_JS = (PROJ_ROOT / 'website' / 'abcjs-basic-min.js').read_text(encoding='utf-8')
    return _ABCJS_JS


def abc_for_display(abc_text: str) -> str:
    """The generated ABC has one bar per '[V:1]' line, which abcjs treats as hard system
    breaks. Join the body lines (same trick as website/index.html) so abcjs wraps measures
    itself; the first '[V:1]' right after the 'V:1' header line must stay on its own line."""
    def _sub(m):
        off = m.start()
        return m.group(0) if abc_text[max(0, off - 4):off] == '\nV:1' else ' '
    return re.sub(r'\n\[V:1\]', _sub, abc_text)


def _strip_with_map(abc: str):
    """Re-implementation of motif.extract.strip_voice_and_text that also returns, for every
    char of the stripped text, its index in the ORIGINAL text (-1 for synthesised newlines).
    Asserted equal to the backbone's output so highlight offsets can never drift from it."""
    text, idx = abc, list(range(len(abc)))
    for pat in (r'\".*?\"', r'\[V:[^\]]*\]', r'![^!\n]*!', r'\+[^+\n]*\+'):
        nt, ni, pos = [], [], 0
        for m in re.finditer(pat, text):
            nt.append(text[pos:m.start()]); ni.extend(idx[pos:m.start()]); pos = m.end()
        nt.append(text[pos:]); ni.extend(idx[pos:])
        text, idx = ''.join(nt), ni
    kept_t, kept_i = [], []
    pos = 0
    for line in text.split('\n'):
        st = line.strip()
        seg_idx = idx[pos:pos + len(line)]
        pos += len(line) + 1
        if not st or st.startswith('%') or re.match(r'^[A-Za-z]:', st):
            continue
        kept_t.append(line); kept_i.append(seg_idx)
    out_t, out_i = '\n'.join(kept_t), []
    for k, seg in enumerate(kept_i):
        if k:
            out_i.append(-1)
        out_i.extend(seg)
    assert out_t == mx.strip_voice_and_text(abc), 'strip map diverged from motif.extract'
    return out_t, out_i


def motif_spans(display_abc: str, pattern: tuple, key=None):
    """Char spans [(start, end), ...] in `display_abc` of every occurrence of `pattern`
    (same window logic as the checker), one span per occurrence covering its notes."""
    stripped, omap = _strip_with_map(display_abc)
    pitches, bars, note_spans, s, _ = mx.abc_to_pitches_with_bars(display_abc, key=key)
    assert s == stripped
    if len(pitches) < len(pattern):
        return []
    counts, occ = mx.count_interval_motifs(list(pitches), list(bars), window_notes=len(pattern),
                                           stride_notes=1, interval_mode=INTERVAL_MODE,
                                         fold_transformations=False)
    spans = []
    for o in occ.get(tuple(pattern), []):
        a = note_spans[o['start_note']][0]
        b = note_spans[o['end_note']][1]
        # map stripped offsets -> original offsets (skip synthesised newlines)
        oa = next(omap[i] for i in range(a, len(omap)) if omap[i] >= 0)
        ob = next(omap[i] for i in range(b - 1, -1, -1) if omap[i] >= 0) + 1
        spans.append((oa, ob))
    return spans


def _window_spans(display_abc: str, m: dict):
    """Char span(s) in `display_abc` for a find_realization_in_piece() result: the best window,
    plus the longest verbatim run (both are recorded as stripped-text offsets)."""
    stripped, omap = _strip_with_map(display_abc)
    def _to_orig(a, b):
        oa = next(omap[i] for i in range(a, len(omap)) if omap[i] >= 0)
        ob = next(omap[i] for i in range(b - 1, -1, -1) if omap[i] >= 0) + 1
        return (oa, ob)
    return [_to_orig(*m[k]) for k in ('_best_span', '_run_span') if m.get(k)]


def render_score(abc_text: str, *, pattern: tuple | None = None, realization: str | None = None,
                 realization_key=None, title: str | None = None, staffwidth: int = 900,
                 measures_per_line: int = 4, show_abc: bool = False, player: bool = True,
                 highlight_transforms: bool = False):
    """Render `abc_text` as engraved notation inline in the notebook using abcjs (inlined
    from website/abcjs-basic-min.js, so no network is needed), with an in-browser player
    (▶ Play / ■ Stop + progress bar; the showcase site's self-contained WebAudio voice — no
    soundfont download) and motif highlighting:
      pattern      -> every occurrence of the abstract motif is coloured RED
      highlight_transforms -> occurrences of its inversion / retrograde / retrograde-inversion
                      are coloured ORANGE (legend gives the count per form)
      realization  -> the best fuzzy match of the concrete realization (find_realization_in_piece)
                      is coloured BLUE (needs realization_key)
    Returns an IPython HTML object — leave it as the last expression of a cell."""
    import json as _json
    import uuid as _uuid
    from IPython.display import HTML
    uid = _uuid.uuid4().hex
    div_id, btn_id, bar_id, wrap_id = f'abcjs_{uid}', f'play_{uid}', f'bar_{uid}', f'wrap_{uid}'
    disp = abc_for_display(abc_text)
    if title:
        disp = disp.replace('X:1\n', f'X:1\nT:{title}\n', 1) if disp.startswith('X:1\n') else f'X:1\nT:{title}\n' + disp
    opts = {'responsive': 'resize', 'staffwidth': staffwidth, 'paddingtop': 4, 'paddingbottom': 4,
            'wrap': {'minSpacing': 1.8, 'maxSpacing': 2.7, 'preferredMeasuresPerLine': measures_per_line}}
    has_q = '\nQ:' in abc_text
    motif_sp = motif_spans(disp, tuple(pattern)) if pattern else []
    trans_sp = []   # [(label, spans)] for the distinct transformed forms
    if pattern and highlight_transforms:
        seen = {tuple(int(x) for x in pattern)}
        for lbl, form in zip(mx.TRANSFORM_LABELS[1:], mx._pattern_transforms(tuple(pattern))[1:]):
            if form in seen:
                continue
            seen.add(form)
            trans_sp.append((lbl, motif_spans(disp, form)))
    real_sp, m = [], None
    if realization:
        m = find_realization_in_piece(disp, realization, realization_key)
        if m.get('verdict', 'absent') != 'absent':
            real_sp = _window_spans(disp, m)
    legend = ''
    if pattern or realization:
        legend = ('<div style="font-family:sans-serif;font-size:12px;color:#555;margin:2px 0 4px 0">'
                  + (f'<span style="color:#d22">&#9632;</span> motif {tuple(pattern)} &times;{len(motif_sp)} &nbsp; ' if pattern else '')
                  + ''.join(f'<span style="color:#e80">&#9632;</span> {lbl} &times;{len(sp)} &nbsp; ' for lbl, sp in trans_sp)
                  + (f'<span style="color:#26c">&#9632;</span> realization match ({m["verdict"]}, sim {m["similarity"]:.2f})' if m else '')
                  + '</div>')
    controls = f"""
<div style="display:flex;align-items:center;gap:10px;margin:6px 0 4px 0;font-family:sans-serif;font-size:13px">
  <button id="{btn_id}" style="padding:4px 12px;border-radius:5px;border:1px solid #888;background:#f4f4f4;cursor:pointer">&#9654; Play</button>
  <div style="flex:1;height:6px;background:#ddd;border-radius:3px;overflow:hidden;max-width:420px"><div id="{bar_id}" style="height:100%;width:0%;background:#4a7"></div></div>
  <span style="color:#666">in-browser synth (WebAudio), tempo from Q: or auto</span>
</div>""" if player else ''
    html = f"""
<div id="{wrap_id}" style="background:#fff;padding:8px;border-radius:6px;max-width:100%;overflow-x:auto">
<style>#{wrap_id} .motif-note {{ fill:#d22 !important; stroke:#d22 !important; }} #{wrap_id} .real-note {{ fill:#26c !important; stroke:#26c !important; }}
#{wrap_id} .trans-note {{ fill:#e80 !important; stroke:#e80 !important; }}
#{wrap_id} .motif-note.real-note {{ fill:#a0d !important; stroke:#a0d !important; }}</style>
{controls}{legend}
<div id="{div_id}"></div>
{'<pre style="font-size:11px;white-space:pre-wrap">' + disp.replace('<','&lt;') + '</pre>' if show_abc else ''}
</div>
<script>
(function(){{
  var box = document.getElementById("{div_id}");
  if (!window.ABCJS) {{
    // abcjs ships as UMD; the notebook webview exposes an AMD `define`, so a plain
    // include would register a module instead of window.ABCJS. Evaluate it in a scope
    // that forces the CommonJS branch and capture the export ourselves.
    try {{
      var __m = {{exports: {{}}}};
      (new Function("module", "exports", "define", {_json.dumps(_abcjs_source())}))(__m, __m.exports, undefined);
      window.ABCJS = __m.exports && (__m.exports.renderAbc ? __m.exports : (__m.exports.abcjs || __m.exports.default));
    }} catch (e) {{ box.innerHTML = "<em>could not load abcjs: " + e + "</em>"; return; }}
  }}
  var vo;
  try {{ vo = window.ABCJS.renderAbc("{div_id}", {_json.dumps(disp)}, {_json.dumps(opts)})[0]; }}
  catch (e) {{ box.innerHTML = "<em>abcjs failed: " + e + "</em>"; return; }}
  // ---- highlight (ported from website/index.html): colour note glyphs whose source
  //      char range intersects a motif / realization span ----
  function highlight(spans, cls) {{
    if (!spans.length || !vo || !vo.lines) return;
    vo.lines.forEach(function(line){{ (line.staff||[]).forEach(function(staff){{
      (staff.voices||[]).forEach(function(voice){{ voice.forEach(function(el){{
        if (el.el_type !== 'note' || el.rest || el.startChar == null) return;
        if (!spans.some(function(sp){{ return el.startChar < sp[1] && el.endChar > sp[0]; }})) return;
        if (el.abselem && el.abselem.elemset)
          el.abselem.elemset.forEach(function(g){{ g.classList && g.classList.add(cls); }});
      }}); }}); }}); }});
  }}
  highlight({_json.dumps([sp for _, sps in trans_sp for sp in sps])}, 'trans-note');
  highlight({_json.dumps(motif_sp)}, 'motif-note');
  highlight({_json.dumps(real_sp)}, 'real-note');
  if (!{str(player).lower()}) return;

  // ---- player (ported from website/index.html: abcjs sequencer -> WebAudio voice) ----
  var btn = document.getElementById("{btn_id}"), bar = document.getElementById("{bar_id}");
  var P = window.__multilenPlayer = window.__multilenPlayer || {{ctx: null, playing: null}};
  function stop() {{
    if (!P.playing) return;
    P.playing.nodes.forEach(function(n){{ try {{ n.stop(); }} catch (e) {{}} }});
    clearInterval(P.playing.timer);
    if (P.playing.bar) P.playing.bar.style.width = '0%';
    P.playing.btn.innerHTML = '&#9654; Play';
    P.playing = null;
  }}
  function deriveQpm(seq, hasQ) {{
    if (hasQ && seq.tempo) return seq.tempo;
    var durs = [];
    seq.tracks.forEach(function(t){{ t.forEach(function(e){{ if (e.cmd === 'note') durs.push(e.duration); }}); }});
    if (!durs.length) return seq.tempo || 110;
    durs.sort(function(a,b){{return a-b;}});
    var med = durs[Math.floor(durs.length/2)];
    return Math.max(75, Math.min(150, 240 * med / 0.25));
  }}
  function eventsFromSeq(seq, qpm) {{
    var spw = 240 / (qpm || seq.tempo || 100), events = [], total = 0;
    seq.tracks.forEach(function(track){{ track.forEach(function(ev){{
      if (ev.cmd !== 'note') return;
      events.push({{pitch: ev.pitch, volume: ev.volume, start: ev.start*spw, dur: ev.duration*spw}});
      total = Math.max(total, (ev.start + ev.duration) * spw);
    }}); }});
    return {{events: events, total: total}};
  }}
  function play(events, total) {{
    stop();
    if (!events.length) return;
    if (!P.ctx) P.ctx = new (window.AudioContext || window.webkitAudioContext)();
    var ac = P.ctx;
    var ready = ac.state === 'suspended' ? ac.resume() : Promise.resolve();
    ready.then(function(){{
      if (P.playing) return;
      var comp = ac.createDynamicsCompressor(); comp.connect(ac.destination);
      var master = ac.createGain(); master.gain.value = 1.4; master.connect(comp);
      var bus = ac.createGain();
      [[700,1.0,4],[1220,0.7,5],[2600,0.35,6]].forEach(function(f){{
        var bp = ac.createBiquadFilter(); bp.type='bandpass'; bp.frequency.value=f[0]; bp.Q.value=f[2];
        var bg = ac.createGain(); bg.gain.value=f[1]; bus.connect(bp); bp.connect(bg); bg.connect(master);
      }});
      var lp = ac.createBiquadFilter(); lp.type='lowpass'; lp.frequency.value=1800;
      var lpg = ac.createGain(); lpg.gain.value=0.6; bus.connect(lp); lp.connect(lpg); lpg.connect(master);
      var t0 = ac.currentTime + 0.08;
      var lfo = ac.createOscillator(); lfo.type='sine'; lfo.frequency.value=5.2; lfo.start(t0); lfo.stop(t0+total+1);
      var nodes = [master, lfo];
      events.forEach(function(ev){{
        var start = t0 + ev.start, dur = Math.max(0.05, ev.dur - 0.02);
        var freq = 440 * Math.pow(2, (ev.pitch - 69) / 12);
        var osc = ac.createOscillator(); osc.type='sawtooth'; osc.frequency.value=freq;
        var vib = ac.createGain(); vib.gain.setValueAtTime(0, start);
        vib.gain.linearRampToValueAtTime(freq*0.007, start + Math.min(0.25, dur));
        lfo.connect(vib); vib.connect(osc.frequency);
        var g = ac.createGain(); var vol = (ev.volume || 90) / 127 * 0.9; var atk = Math.min(0.06, dur*0.4);
        g.gain.setValueAtTime(0, start); g.gain.linearRampToValueAtTime(vol, start+atk);
        g.gain.setTargetAtTime(0, start + dur*0.75, Math.max(0.05, dur*0.15));
        osc.connect(g); g.connect(bus); osc.start(start); osc.stop(start + dur + 0.5); nodes.push(osc);
      }});
      btn.innerHTML = '&#9632; Stop';
      var timer = setInterval(function(){{
        var t = ac.currentTime - t0;
        if (t >= total) {{ stop(); return; }}
        bar.style.width = Math.max(0, Math.min(100, t/total*100)) + '%';
      }}, 120);
      P.playing = {{nodes: nodes, timer: timer, btn: btn, bar: bar}};
    }});
  }}
  btn.addEventListener('click', function(){{
    if (P.playing && P.playing.btn === btn) {{ stop(); return; }}
    try {{
      var seq = vo.setUpAudio({{}});
      var qpm = deriveQpm(seq, {str(has_q).lower()});
      var r = eventsFromSeq(seq, qpm);
      play(r.events, r.total);
    }} catch (e) {{ btn.textContent = 'playback failed: ' + e; }}
  }});
}})();
</script>"""
    return HTML(html)
