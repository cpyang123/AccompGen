#!/usr/bin/env python3
"""Prepend NotaGen-style `%Period / %Composer / %Instrumentation` lines to the
processed ABC files, above the `%motif:` prompt lines.

Why: our fine-tuning corpora (OpenScore Lieder, Irishman) carry no such
preamble, while every eval and the Gradio demo prompt with one. NotaGen's own
pipeline never generated these lines either -- it preserved them from source
files that already had them -- so the channel was simply never reinforced.

Assignment rules
----------------
* Instrumentation is ``Art Song`` for every piece (matches the evaluation
  prompt).
* Lieder (``lc*``): composer and period come from the piece's own source ABC.
  The source ``C:`` field is resolved to a canonical ``Lastname, Firstname``
  and a period, preferring NotaGen's own Art Song vocabulary (gradio/
  prompts.txt) so the prompt stays in-vocabulary; otherwise the OpenScore
  ``composers.tsv`` supplies the canonical name and the birth year supplies
  the period.
* Irishman (``irish_*``): ``%Folk`` / ``%Irishman``.
* Synthetic crops (``synth_*``): the crop copies its source piece's header, so
  the source corpus is read off the ``V:1`` line -- Lieder voices carry
  attributes (``V:1 treble nm="Voice"``), Irishman voices are a bare ``V:1``.
  Lieder-derived crops are matched back to their source piece by locating a
  tunebody bar line in the Lieder index, which recovers the real composer.

The script is idempotent (a file that already starts with a preamble is left
alone) and writes atomically. Use --dry-run first; it reports the assignment
histogram and any unresolved pieces without touching anything.

Examples
--------
  python data/add_style_preamble.py --dry-run \\
      --real data/abcfiles_processed_v1 \\
      --real /usr/xtmp/cy232/accompgen/irishman/abcfiles_processed_v1 \\
      --synthetic /usr/xtmp/cy232/accompgen/synthetic_motifs_v1
"""
import argparse
import csv
import os
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

LIEDER_SOURCE_DIR = os.path.join(HERE, 'abcfiles')
COMPOSERS_TSV = os.path.join(HERE, 'Lieder', 'data', 'composers.tsv')
PROMPTS_TXT = os.path.join(REPO, 'gradio', 'prompts.txt')

INSTRUMENTATION = 'Art Song'
IRISH_PERIOD, IRISH_COMPOSER = 'Folk', 'Irishman'
DEFAULT_PERIOD = 'Romantic'          # OpenScore Lieder is a Romantic-era corpus
# NotaGen only knows three periods; split the remainder by birth year.
BAROQUE_BEFORE, CLASSICAL_BEFORE = 1710, 1770


try:                                    # same transliteration the corpus uses
    from unidecode import unidecode as _translit
except ImportError:                     # fallback: strip combining marks
    def _translit(s):
        return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()


def to_ascii(s):
    """Fold to pure ASCII.

    NotaGen represents music as raw chars one-hot over a 128-symbol alphabet,
    and `2_data_preprocess.py` unidecodes every line for exactly this reason.
    A preamble containing 'Valérie' or a U+2019 apostrophe makes ord(c) > 127,
    which indexes past the embedding and dies as a CUDA device-side assert.
    """
    return ''.join(c for c in _translit(s) if 32 <= ord(c) < 127)


def _norm_tokens(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    return [t for t in re.sub(r'[^a-z ]', ' ', s.lower()).split() if len(t) > 1]


def _surname_first(s):
    """Best guess at the surname token of a ``C:`` field, as a 1-tuple.

    "Lastname, Firstname" -> the part before the comma; otherwise the last
    token of "Firstname ... Lastname" (handles "Fanny (Mendelssohn) Hensel").
    """
    head = s.split(',')[0] if ',' in s else s
    toks = _norm_tokens(head)
    if not toks:
        return ()
    return (toks[0],) if ',' in s else (toks[-1],)


def _period_from_birth(born):
    try:
        y = int(str(born)[:4])
    except (TypeError, ValueError):
        return DEFAULT_PERIOD
    if y < BAROQUE_BEFORE:
        return 'Baroque'
    if y < CLASSICAL_BEFORE:
        return 'Classical'
    return 'Romantic'


class ComposerResolver:
    """Map a source ``C:`` field to (period, canonical composer)."""

    def __init__(self):
        # Both tables are surname -> [(composer, period, token set), ...].
        # A surname can be shared (Clara vs Robert Schumann), so candidates are
        # kept as a list and disambiguated by given-name overlap at lookup.
        self.vocab = {}
        with open(PROMPTS_TXT) as fh:
            for line in fh:
                parts = line.strip().split('_')
                if len(parts) == 3 and parts[2] == INSTRUMENTATION:
                    period, composer = parts[0], parts[1]
                    toks = _norm_tokens(composer)
                    if toks:
                        self.vocab.setdefault(toks[0], []).append(
                            (composer, period, set(toks)))
        self.tsv = {}
        with open(COMPOSERS_TSV, newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh, delimiter='\t'):
                canonical = row['path'].replace('_', ' ').strip()
                toks = _norm_tokens(canonical)
                if toks:
                    self.tsv.setdefault(toks[0], []).append(
                        (canonical, _period_from_birth(row.get('born')),
                         set(toks)))
        self.unresolved = Counter()

    @staticmethod
    def _best(cands, want):
        """Pick the candidate sharing the most name tokens with `want`."""
        if len(cands) == 1:
            return cands[0][0], cands[0][1]
        best = max(cands, key=lambda c: len(c[2] & want))
        return best[0], best[1]

    def resolve(self, c_field):
        # Try the surname FIRST, then any other token. Order matters: a first
        # name can collide with a different composer's surname (the corpus has
        # both "Franz Schubert" and "Robert Franz", so scanning left-to-right
        # would resolve Schubert to "Franz, Robert").
        want = set(_norm_tokens(c_field))
        for toks in (_surname_first(c_field), _norm_tokens(c_field)):
            for t in toks:                  # prefer NotaGen's vocabulary
                if t in self.vocab:
                    composer, period = self._best(self.vocab[t], want)
                    return period, composer
            for t in toks:                  # else the OpenScore table
                if t in self.tsv:
                    composer, period = self._best(self.tsv[t], want)
                    return period, composer
        self.unresolved[c_field] += 1
        if ',' in c_field:                  # already "Lastname, Firstname"
            return DEFAULT_PERIOD, c_field.strip()
        bits = c_field.split()
        if len(bits) >= 2:
            return DEFAULT_PERIOD, f'{bits[-1]}, {" ".join(bits[:-1])}'
        return DEFAULT_PERIOD, c_field.strip() or 'Anonymous'


def read_c_field(path):
    try:
        with open(path, encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                if line.startswith('C:'):
                    return line[2:].strip()
                if line.startswith(('[V:', 'V:')):
                    break
    except OSError:
        pass
    return ''


def build_lieder_preambles(resolver):
    """piece stem (e.g. 'lc5988112') -> (period, composer)."""
    out = {}
    if not os.path.isdir(LIEDER_SOURCE_DIR):
        return out
    for fname in os.listdir(LIEDER_SOURCE_DIR):
        if not fname.endswith('.abc'):
            continue
        stem = fname[:-4]
        c = read_c_field(os.path.join(LIEDER_SOURCE_DIR, fname))
        out[stem] = resolver.resolve(c) if c else (DEFAULT_PERIOD, 'Anonymous')
    return out


# --------------------------------------------------------------------------
# file rewriting
# --------------------------------------------------------------------------

def _is_preamble_line(line):
    """A bare '%Word' comment that is neither a %motif: line nor an ABC '%%'."""
    return (line.startswith('%') and not line.startswith('%%')
            and not line.startswith('%motif:'))


def split_header(lines):
    """Return (existing_preamble, rest) for an already-read file."""
    i = 0
    while i < len(lines) and _is_preamble_line(lines[i]):
        i += 1
    return lines[:i], lines[i:]


def patch_text(text, period, composer):
    """Insert/replace the preamble. Returns (new_text, action)."""
    lines = text.split('\n')
    existing, rest = split_header(lines)
    want = [f'%{to_ascii(period)}', f'%{to_ascii(composer)}',
            f'%{to_ascii(INSTRUMENTATION)}']
    if existing == want:
        return text, 'already-correct'
    action = 'replaced' if existing else 'added'
    return '\n'.join(want + rest), action


def patch_file(path, period, composer, dry_run):
    try:
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
    except OSError as e:
        return f'error:{type(e).__name__}'
    new_text, action = patch_text(text, period, composer)
    if action == 'already-correct' or dry_run:
        return action
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(new_text)
    os.replace(tmp, path)
    return action


# --------------------------------------------------------------------------
# synthetic-crop provenance
# --------------------------------------------------------------------------

BAR_RE = re.compile(r'^\[V:1\](.*)$')


def _first_bars(text, n=3):
    bars = []
    for line in text.split('\n'):
        m = BAR_RE.match(line)
        if m and m.group(1).strip():
            bars.append(m.group(1).strip())
            if len(bars) >= n:
                break
    return bars


def build_lieder_bar_index(real_dirs, key):
    """bar text -> piece stem, over Lieder pieces in one key."""
    index = {}
    for d in real_dirs:
        kd = os.path.join(d, key)
        if not os.path.isdir(kd):
            continue
        for fname in os.listdir(kd):
            if not fname.startswith('lc'):
                continue
            stem = fname.split('_')[0]
            try:
                with open(os.path.join(kd, fname), encoding='utf-8') as fh:
                    text = fh.read()
            except OSError:
                continue
            for line in text.split('\n'):
                m = BAR_RE.match(line)
                if m:
                    bar = m.group(1).strip()
                    if bar:
                        index.setdefault(bar, stem)
    return index


def _voice_has_attributes(text):
    for line in text.split('\n'):
        if line.startswith('V:1'):
            return line.strip() != 'V:1'
        if line.startswith('[V:'):
            break
    return False


def process_synth_key(args):
    """Patch every crop in one key directory. Returns a Counter of actions."""
    synth_key_dir, real_dirs, key, lieder_preambles, dry_run = args
    stats = Counter()
    bar_index = None                      # built lazily: most crops are Irish
    try:
        fnames = os.listdir(synth_key_dir)
    except OSError:
        return stats
    for fname in fnames:
        if not fname.endswith('.abc'):
            continue
        path = os.path.join(synth_key_dir, fname)
        try:
            with open(path, encoding='utf-8') as fh:
                text = fh.read()
        except OSError:
            stats['error'] += 1
            continue
        if _voice_has_attributes(text):
            if bar_index is None:
                bar_index = build_lieder_bar_index(real_dirs, key)
            stem = next((bar_index[b] for b in _first_bars(text) if b in bar_index),
                        None)
            if stem and stem in lieder_preambles:
                period, composer = lieder_preambles[stem]
                stats['lieder-matched'] += 1
            else:
                period, composer = DEFAULT_PERIOD, 'Anonymous'
                stats['lieder-UNMATCHED'] += 1
        else:
            period, composer = IRISH_PERIOD, IRISH_COMPOSER
            stats['irish'] += 1
        new_text, action = patch_text(text, period, composer)
        stats[action] += 1
        if action != 'already-correct' and not dry_run:
            tmp = f'{path}.tmp{os.getpid()}'
            with open(tmp, 'w', encoding='utf-8') as fh:
                fh.write(new_text)
            os.replace(tmp, path)
    return stats


def process_real_key(args):
    real_key_dir, lieder_preambles, dry_run = args
    stats = Counter()
    try:
        fnames = os.listdir(real_key_dir)
    except OSError:
        return stats
    for fname in fnames:
        if not fname.endswith('.abc'):
            continue
        stem = fname.split('_')[0]
        if stem.startswith('irish'):
            period, composer = IRISH_PERIOD, IRISH_COMPOSER
            stats['irish'] += 1
        elif stem in lieder_preambles:
            period, composer = lieder_preambles[stem]
            stats['lieder'] += 1
        else:
            period, composer = DEFAULT_PERIOD, 'Anonymous'
            stats['UNKNOWN-stem'] += 1
        stats[patch_file(os.path.join(real_key_dir, fname), period, composer,
                         dry_run)] += 1
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--real', action='append', default=[],
                    help='processed real-data dir (contains <key>/ subdirs)')
    ap.add_argument('--synthetic', action='append', default=[],
                    help='synthetic crop dir (contains <key>/ subdirs)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit-keys', type=int, default=0,
                    help='process only the first N key dirs (smoke test)')
    args = ap.parse_args()
    if not args.real and not args.synthetic:
        ap.error('give at least one --real or --synthetic directory')

    resolver = ComposerResolver()
    lieder = build_lieder_preambles(resolver)
    print(f'Lieder source pieces resolved: {len(lieder)}')
    per = Counter(p for p, _ in lieder.values())
    print(f'  periods: {dict(per)}')
    top = Counter(c for _, c in lieder.values()).most_common(5)
    print(f'  most common composers: {top}')
    if resolver.unresolved:
        print(f'  NOT in either vocabulary ({len(resolver.unresolved)} names, '
              f'{sum(resolver.unresolved.values())} pieces) -> '
              f'"{DEFAULT_PERIOD}" + reformatted name; e.g. '
              f'{resolver.unresolved.most_common(3)}')

    def keys_of(d):
        ks = sorted(k for k in os.listdir(d) if os.path.isdir(os.path.join(d, k)))
        return ks[:args.limit_keys] if args.limit_keys else ks

    total = Counter()
    jobs_real, jobs_synth = [], []
    for d in args.real:
        for k in keys_of(d):
            jobs_real.append((os.path.join(d, k), lieder, args.dry_run))
    for d in args.synthetic:
        for k in keys_of(d):
            jobs_synth.append((os.path.join(d, k), args.real, k, lieder,
                               args.dry_run))

    if jobs_real:
        print(f'\nreal: {len(jobs_real)} key dirs')
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for st in ex.map(process_real_key, jobs_real):
                total.update(st)
    if jobs_synth:
        print(f'synthetic: {len(jobs_synth)} key dirs')
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for st in ex.map(process_synth_key, jobs_synth):
                total.update(st)

    print('\n=== ' + ('DRY RUN (nothing written)' if args.dry_run else 'DONE') + ' ===')
    for k, v in sorted(total.items()):
        print(f'  {k:20s} {v}')
    if total.get('lieder-UNMATCHED'):
        print('  NOTE: unmatched Lieder crops fell back to '
              f'"{DEFAULT_PERIOD} / Anonymous".')
    return 0


if __name__ == '__main__':
    sys.exit(main())
