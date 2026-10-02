"""Helper utilities for demo_new.ipynb and motif_check.ipynb.

Two main entry points:

  - realize_motif_line(motif_v1_line, key, ...): given a `%motif:v1:` line,
    produce a `%motif:v1:` + `%motif:abc:` prompt by either looking up the
    pattern in the training data (same key) or generating it algorithmically.
    Useful when you want to feed the model a pre-realized motif instead of
    letting it invent the realization itself.

  - inspect_motif_attention(model, patchilizer, period, composer,
    instrumentation, additional_prompts, motif_bias_value): run one forward
    pass through the patch-level encoder with output_attentions=True and
    print per-layer statistics on how much attention non-motif queries pay
    to motif key positions. Pass motif_bias_value=0.0 to see the no-bias
    baseline (i.e. what model.generate() currently uses at inference time).

Both functions assume the notebook has already imported torch, defined the
NotaGen model, the Patchilizer class, and the constants PATCH_SIZE, device,
and MOTIF_ATTENTION_BIAS in the global scope.
"""
from __future__ import annotations

import os
import re
import sys
import random
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Motif realization (lookup in training data, else algorithmic)
# ---------------------------------------------------------------------------

TRAIN_DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'abcfiles_processed'
_MOTIF_INDEX = None


def build_motif_index(abc_dir=TRAIN_DATA_DIR):
    """Scan every training file under abc_dir; return a dict mapping
    (key, abstract_pattern_tuple) -> list[realization_str]."""
    idx = defaultdict(list)
    abc_dir = Path(abc_dir)
    if not abc_dir.exists():
        return {}
    for key_dir in sorted(abc_dir.iterdir()):
        if not key_dir.is_dir():
            continue
        key = key_dir.name
        for fpath in key_dir.glob('*.abc'):
            try:
                content = fpath.read_text(encoding='utf-8')
            except Exception:
                continue
            pat_line = abc_line = None
            for line in content.splitlines():
                if line.startswith('%motif:v1:step_skip_leap:'):
                    pat_line = line[len('%motif:v1:step_skip_leap:'):].strip()
                elif line.startswith('%motif:abc:'):
                    abc_line = line[len('%motif:abc:'):].strip()
                if pat_line and abc_line:
                    break
            if not (pat_line and abc_line):
                continue
            pats = [p.strip() for p in pat_line.split(';')]
            abcs = [a.strip() for a in abc_line.split(';')]
            if len(pats) != len(abcs):
                continue
            for p, a in zip(pats, abcs):
                try:
                    pat_t = tuple(int(x) for x in p.split(','))
                except ValueError:
                    continue
                idx[(key, pat_t)].append(a)
    return dict(idx)


def get_motif_index():
    global _MOTIF_INDEX
    if _MOTIF_INDEX is None:
        print('Building motif index from training data...')
        _MOTIF_INDEX = build_motif_index()
        n = sum(len(v) for v in _MOTIF_INDEX.values())
        print(f'  Indexed {n} realizations across {len(_MOTIF_INDEX)} (key, pattern) entries.')
    return _MOTIF_INDEX


def parse_motif_v1_line(line):
    """Parse '%motif:v1:step_skip_leap: 0,2,2,2 ; 0,3,3,3 ; ...'
    -> ('step_skip_leap', [(0,2,2,2), (0,3,3,3), ...])."""
    m = re.match(r'%?\s*motif:v\d+:([a-zA-Z_]+):\s*(.*)', line.strip().lstrip('%').strip())
    if not m:
        raise ValueError(f'Cannot parse motif v1 line: {line!r}')
    mode = m.group(1)
    body = m.group(2)
    patterns = []
    for chunk in body.split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        ints = []
        for tok in re.split(r'[,\s]+', chunk):
            if not tok:
                continue
            try:
                ints.append(int(tok))
            except ValueError:
                break
        if ints:
            patterns.append(tuple(ints))
    return mode, patterns


def _algorithmic_realize(pattern_tuple, key):
    """Fallback realization via data/generate_synthetic_motifs.py."""
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / 'data'))
    from generate_synthetic_motifs import pattern_to_abc, KEY_TO_START_NOTE
    return pattern_to_abc(
        list(pattern_tuple),
        start_note=KEY_TO_START_NOTE.get(key, 'C'),
        duration_mode='uniform',
    )


def realize_motif_line(motif_v1_line, key, use_training=True,
                       allow_algorithmic=True, verbose=True):
    """Produce a full motif prompt (%motif:v1: line + %motif:abc: line) by
    pre-realizing each abstract pattern.

    Strategy per pattern:
      1. Look up (key, pattern) in the training-data index. If any
         realization exists, sample one.
      2. Else, if allow_algorithmic, generate one via pattern_to_abc.
      3. Else, leave that realization empty.

    Returns (prompt_string, [source_label_per_pattern]).
    """
    mode, patterns = parse_motif_v1_line(motif_v1_line)
    idx = get_motif_index() if use_training else {}
    realizations, sources = [], []
    for pat in patterns:
        candidates = idx.get((key, pat), [])
        if candidates:
            realizations.append(random.choice(candidates))
            sources.append(f'training[{key}]')
        elif allow_algorithmic:
            realizations.append(_algorithmic_realize(pat, key))
            sources.append('algorithmic')
        else:
            realizations.append('')
            sources.append('missing')

    pat_line_text = ' ; '.join(','.join(map(str, p)) for p in patterns)
    abc_line_text = ' ; '.join(realizations)
    prompt = f'%motif:v1:{mode}: {pat_line_text}\n%motif:abc: {abc_line_text}\n'
    if verbose:
        print(f'Realized motif prompt for key={key}:')
        for p, r, s in zip(patterns, realizations, sources):
            print(f'  {p} -> {r!r}  [{s}]')
    return prompt, sources


# ---------------------------------------------------------------------------
# Attention-weight inspection
# ---------------------------------------------------------------------------

def _build_prompt_patches_with_mask(patchilizer, lines, patch_size):
    """Return ([patches_text], [is_motif_flag]) for a prompt."""
    bos_patch = chr(patchilizer.bos_token_id) * (patch_size - 1) + chr(patchilizer.eos_token_id)
    patches_text = [bos_patch]
    motif_flags = [False]
    for line in lines:
        line_patches = patchilizer.split_patches(line)
        patches_text += line_patches
        motif_flags += [line.startswith('%motif:')] * len(line_patches)
    return patches_text, motif_flags


@torch.no_grad()
def _resolve_notebook_globals():
    """Walk up the call stack to find a frame whose globals look like the
    notebook (has 'device', 'PATCH_SIZE', 'MOTIF_ATTENTION_BIAS'). Returns
    that dict, or the helper-module globals as a last resort."""
    frame = sys._getframe(1)
    while frame is not None:
        g = frame.f_globals
        if all(k in g for k in ('device', 'PATCH_SIZE', 'MOTIF_ATTENTION_BIAS')):
            return g
        frame = frame.f_back
    return globals()


def inspect_motif_attention(model, patchilizer, period, composer, instrumentation,
                            additional_prompts, motif_bias_value=None,
                            device=None, patch_size=None, motif_attention_bias=None):
    """Forward-pass the prompt through the patch-level encoder with
    output_attentions=True and print per-layer attention stats.

    `motif_bias_value`: scalar bias added to attention logits at motif key
    positions. Defaults to MOTIF_ATTENTION_BIAS (the value the model was
    trained with). Pass 0.0 to see the no-bias baseline.

    `device`, `patch_size`, `motif_attention_bias`: pulled from the notebook's
    globals (walking up the call stack) if not provided.
    """
    if device is None or patch_size is None or motif_attention_bias is None:
        nb_globals = _resolve_notebook_globals()
        if device is None:
            device = nb_globals['device']
        if patch_size is None:
            patch_size = nb_globals['PATCH_SIZE']
        if motif_attention_bias is None:
            motif_attention_bias = nb_globals['MOTIF_ATTENTION_BIAS']
    if motif_bias_value is None:
        motif_bias_value = motif_attention_bias

    full_lines = [f'%{period}\n', f'%{composer}\n', f'%{instrumentation}\n'] + list(additional_prompts)
    patches_text, motif_flags = _build_prompt_patches_with_mask(patchilizer, full_lines, patch_size)

    ids = []
    for patch in patches_text:
        ids.append([ord(c) for c in patch] + [patchilizer.special_token_id] * (patch_size - len(patch)))
    seq = len(ids)
    patches_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    motif_mask = torch.tensor([motif_flags], dtype=torch.float, device=device)

    motif_bias = motif_mask * float(motif_bias_value)

    pld = model.patch_level_decoder if hasattr(model, 'patch_level_decoder') else model.module.patch_level_decoder
    patches_oh = torch.nn.functional.one_hot(patches_tensor, num_classes=128).to(pld.dtype)
    patches_oh = patches_oh.reshape(1, seq, patch_size * 128)
    patches_emb = pld.patch_embedding(patches_oh.to(pld.device))

    pld._motif_bias = motif_bias[:, None, None, :]
    try:
        out = pld.base(inputs_embeds=patches_emb, output_attentions=True, return_dict=True)
        attns = out.attentions  # tuple of [1, heads, seq, seq] per layer
    finally:
        pld._motif_bias = None

    motif_keys = motif_mask[0].bool()
    nonmotif_keys = ~motif_keys
    n_motif = int(motif_keys.sum().item())
    n_nonmotif = int(nonmotif_keys.sum().item())
    motif_pos = motif_keys.nonzero(as_tuple=True)[0].tolist()
    print(f'Sequence length:     {seq} patches')
    print(f'Motif key positions: {n_motif}/{seq}  indices={motif_pos}')
    print(f'Bias applied:        +{motif_bias_value}')
    print()
    header = f"{'layer':>5}  {'share->motif':>13}  {'avg/motif key':>16}  {'avg/non-motif':>16}  {'log10 ratio':>12}"
    print(header)
    print('-' * len(header))
    results = []
    for li, attn in enumerate(attns):
        a = attn[0].mean(0)              # [q, k] averaged across heads
        nm_q = nonmotif_keys
        a_q = a[nm_q] if nm_q.any() else a
        share = a_q[:, motif_keys].sum(-1).mean().item() if n_motif else 0.0
        avg_motif = a_q[:, motif_keys].mean().item() if n_motif else 0.0
        avg_non = a_q[:, nonmotif_keys].mean().item() if n_nonmotif else 0.0
        log10_ratio = float(np.log10(avg_motif / avg_non)) if (avg_motif > 0 and avg_non > 0) else float('nan')
        print(f'{li:>5}  {share:>13.4f}  {avg_motif:>16.3e}  {avg_non:>16.3e}  {log10_ratio:>12.2f}')
        results.append({
            'layer': li,
            'share_to_motif': share,
            'avg_per_motif_key': avg_motif,
            'avg_per_non_motif_key': avg_non,
            'log10_ratio': log10_ratio,
        })
    return results


def compare_bias_on_off(model, patchilizer, period, composer, instrumentation,
                        additional_prompts, **kwargs):
    """Convenience: run inspect_motif_attention with bias ON (trained value)
    and again with bias OFF (0.0), printing both."""
    nb_globals = _resolve_notebook_globals()
    device = kwargs.pop('device', nb_globals.get('device'))
    patch_size = kwargs.pop('patch_size', nb_globals.get('PATCH_SIZE'))
    motif_attention_bias = kwargs.pop('motif_attention_bias',
                                      nb_globals.get('MOTIF_ATTENTION_BIAS'))
    if device is None or patch_size is None or motif_attention_bias is None:
        raise RuntimeError(
            "Could not resolve device / PATCH_SIZE / MOTIF_ATTENTION_BIAS from "
            "the notebook globals. Pass them explicitly as keyword arguments.")
    print(f'=== With training bias (MOTIF_ATTENTION_BIAS = {motif_attention_bias:.2f}) ===')
    res_on = inspect_motif_attention(model, patchilizer, period, composer, instrumentation,
                                     additional_prompts, motif_bias_value=motif_attention_bias,
                                     device=device, patch_size=patch_size,
                                     motif_attention_bias=motif_attention_bias, **kwargs)
    print()
    print('=== With bias forced to 0.0 (what generate() currently uses) ===')
    res_off = inspect_motif_attention(model, patchilizer, period, composer, instrumentation,
                                      additional_prompts, motif_bias_value=0.0,
                                      device=device, patch_size=patch_size,
                                      motif_attention_bias=motif_attention_bias, **kwargs)
    print()
    print('How to read this:')
    print('  share->motif       fraction of attention from non-motif queries landing on motif keys')
    print('  avg/motif key      that share / (# motif keys) -> per-key probability mass')
    print('  avg/non-motif      same per-key probability mass for the rest of the sequence')
    print('  log10 ratio        positive => motif keys get systematically more attention each')
    return res_on, res_off
