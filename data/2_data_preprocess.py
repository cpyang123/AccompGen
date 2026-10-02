"""Dataset preprocessing.

This script is designed to be executed as:
    python data/2_data_preprocess.py

So paths are resolved relative to this file's directory.
"""

ORI_FOLDER = 'abcfiles'  # Relative to data/
INTERLEAVED_FOLDER = 'abcfiles_inter'   # Relative to data/
AUGMENTED_FOLDER = 'abcfiles_processed'   # Relative to data/
EVAL_SPLIT = 0.1    # The ratio of eval data

# When True, keep only the V:1 (melody) part: other voices are dropped from the
# %%score line / V: headers and the tunebody, and every V:1 bar is written out —
# including all-rest bars, since there is no other voice left to carry the bar.
# Output goes to AUGMENTED_FOLDER + '_v1' (with matching *_v1.jsonl indices) so the
# full multi-voice dataset is not overwritten. Keep in sync with VOICE1_ONLY in
# data/generate_synthetic_motifs.py so synthetic entries land in the same index.
VOICE1_ONLY = True

import os
import sys
import re
import json
import shutil
import random
from tqdm import tqdm
from abctoolkit.utils import (
    remove_information_field, 
    remove_bar_no_annotations, 
    Quote_re, 
    Barlines,
    extract_metadata_and_parts, 
    extract_global_and_local_metadata,
    extract_barline_and_bartext_dict)
from abctoolkit.convert import unidecode_abc_lines
from abctoolkit.rotate import rotate_abc
from abctoolkit.check import check_alignment_unrotated
from abctoolkit.transpose import Key2index, transpose_an_abc_text

# Ensure project root is on sys.path so `import motif` works when running from data/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from motif import get_best_motifs_per_length, get_best_rhythm_motifs_per_length
from motif import rhythm as rhythm_mod

# Number of top motifs to keep per piece. The %motif header format (count /
# abc / inversion_count / inversion_instance, plus the rhythm:count / rhythm:abc
# lines of the rhythmic motif of the same length) describes exactly ONE motif, so
# this must stay 1 here; generate_synthetic_motifs.py uses its own TOP_N for how
# many crops to emit per piece.
TOP_N_MOTIFS = 1
assert TOP_N_MOTIFS == 1

# Motif lengths (distinct successive notes) to condition on. For each length L we
# emit a separate real datapoint <name>_len<L>_<key>.abc: same tunebody, but the
# %motif header holds the best motif of exactly L notes. Keep in sync with
# MOTIF_LENGTHS in data/generate_synthetic_motifs.py.
MOTIF_LENGTHS = list(range(4, 11))

# Optional env overrides so this same script can preprocess a second corpus (e.g.
# the Irishman dataset) into a different output folder while appending to the SAME
# train/eval index files. Defaults reproduce the original Lieder behavior exactly.
# Absolute override paths pass through os.path.join unchanged; PREP_APPEND_V1 lets an
# absolute AUGMENTED_FOLDER opt out of the automatic '_v1' suffix. See
# scripts/run_preprocess_irishman.sh.
ORI_FOLDER = os.environ.get('PREP_ORI_FOLDER', ORI_FOLDER)
INTERLEAVED_FOLDER = os.environ.get('PREP_INTERLEAVED_FOLDER', INTERLEAVED_FOLDER)
AUGMENTED_FOLDER = os.environ.get('PREP_AUGMENTED_FOLDER', AUGMENTED_FOLDER)
EVAL_SPLIT = float(os.environ.get('PREP_EVAL_SPLIT', EVAL_SPLIT))
APPEND_V1 = os.environ.get('PREP_APPEND_V1', '1' if VOICE1_ONLY else '0') == '1'
INDEX_MODE = os.environ.get('PREP_INDEX_MODE', 'overwrite')  # 'overwrite' | 'append'

# resolve folders relative to this script
DATA_DIR = os.path.dirname(__file__)
ORI_FOLDER = os.path.join(DATA_DIR, ORI_FOLDER)
INTERLEAVED_FOLDER = os.path.join(DATA_DIR, INTERLEAVED_FOLDER)
AUGMENTED_FOLDER = os.path.join(DATA_DIR, AUGMENTED_FOLDER)
if APPEND_V1:
    AUGMENTED_FOLDER += '_v1'

os.makedirs(INTERLEAVED_FOLDER, exist_ok=True)
os.makedirs(AUGMENTED_FOLDER, exist_ok=True)
for key in Key2index.keys():
    key_folder = os.path.join(AUGMENTED_FOLDER, key)
    os.makedirs(key_folder, exist_ok=True)


def abc_preprocess_pipeline(abc_path):

    with open(abc_path, 'r', encoding='utf-8') as f:
        abc_lines = f.readlines()

    # delete blank lines
    abc_lines = [line for line in abc_lines if line.strip() != '']

    # unidecode
    abc_lines = unidecode_abc_lines(abc_lines)

    # clean information field
    abc_lines = remove_information_field(abc_lines=abc_lines, info_fields=['X:', 'T:', 'C:', 'W:', 'w:', 'Z:', '%%MIDI'])

    # delete bar number annotations
    abc_lines = remove_bar_no_annotations(abc_lines)

    # delete \"
    for i, line in enumerate(abc_lines):
        if re.search(r'^[A-Za-z]:', line) or line.startswith('%'):
            continue
        else:
            if r'\"' in line:
                abc_lines[i] = abc_lines[i].replace(r'\"', '')

    # delete text annotations with quotes
    for i, line in enumerate(abc_lines):
        quote_contents = re.findall(Quote_re, line)
        for quote_content in quote_contents:
            for barline in Barlines:
                if barline in quote_content:
                    line = line.replace(quote_content, '')
                    abc_lines[i] = line

    # check bar alignment
    try:
        _, bar_no_equal_flag, _ = check_alignment_unrotated(abc_lines)
        if not bar_no_equal_flag:
            print(abc_path, 'Unequal bar number')
            raise Exception
    except:
        raise Exception

    # deal with text annotations: remove too long text annotations; remove consecutive non-alphabet/number characters
    for i, line in enumerate(abc_lines):
        quote_matches = re.findall(r'"[^"]*"', line)
        for match in quote_matches:
            if match == '""':
                line = line.replace(match, '')
            if match[1] in ['^', '_']:
                sub_string = match
                pattern = r'([^a-zA-Z0-9])\1+'
                sub_string = re.sub(pattern, r'\1', sub_string)
                if len(sub_string) <= 40:
                    line = line.replace(match, sub_string)
                else:
                    line = line.replace(match, '')
        abc_lines[i] = line

    abc_name = os.path.splitext(os.path.split(abc_path)[-1])[0]

    # transpose
    metadata_lines, part_text_dict = extract_metadata_and_parts(abc_lines)
    global_metadata_dict, local_metadata_dict = extract_global_and_local_metadata(metadata_lines)
    if global_metadata_dict['K'][0] == 'none':
        global_metadata_dict['K'][0] = 'C'
    ori_key = global_metadata_dict['K'][0]

    interleaved_abc = rotate_abc(abc_lines)
    interleaved_path = os.path.join(INTERLEAVED_FOLDER, abc_name + '.abc')
    with open(interleaved_path, 'w') as w:
        w.writelines(interleaved_abc)

    for key in Key2index.keys():
        transposed_abc_text = transpose_an_abc_text(abc_lines, key)
        transposed_abc_lines = transposed_abc_text.split('\n')
        transposed_abc_lines = list(filter(None, transposed_abc_lines))
        transposed_abc_lines = [line + '\n' for line in transposed_abc_lines]

        # extract motifs from V:1 (full piece): one best melodic AND one best
        # rhythmic motif per length in MOTIF_LENGTHS, each length becoming its
        # own output file below. A length gets both blocks or neither.
        motif_line_by_len = {}
        try:
            # Use abctoolkit's header/parts split so we reliably get V:1 music content
            _metadata_lines, part_text_dict = extract_metadata_and_parts(transposed_abc_lines)
            v1_text = part_text_dict.get('V:1')
            if v1_text:
                motifs_by_len = get_best_motifs_per_length(v1_text,
                window_range=(MOTIF_LENGTHS[0], MOTIF_LENGTHS[-1] + 1),
                interval_mode="step_skip_leap",
                # v1_text comes from extract_metadata_and_parts and has no K: line,
                # so the key signature must be passed in explicitly. Without it,
                # notes altered only by the key signature are mis-parsed and the
                # same piece yields different motifs across transpositions.
                key=key,
                # Only the inversion folds into the canonical motif (retrograde
                # forms count as separate motifs); occurrences are tracked per form.
                fold_forms=("I",),
                )
                # Rhythmic motifs: durations are read in units of L: and
                # expressed as ratios to 1/(initial meter denominator). Both
                # fields live in the header, which the V:1 part text no longer
                # carries, so they are passed in explicitly (like `key` above).
                _meta_text = '\n'.join(l.rstrip('\n') for l in _metadata_lines)
                rhythms_by_len = get_best_rhythm_motifs_per_length(
                    v1_text,
                    window_range=(MOTIF_LENGTHS[0], MOTIF_LENGTHS[-1] + 1),
                    meter=rhythm_mod.detect_meter_field(_meta_text),
                    unit_length=rhythm_mod.detect_unit_length_field(_meta_text),
                )
                for motif_len, top_motifs in motifs_by_len.items():
                    r_top = rhythms_by_len.get(motif_len)
                    if not r_top:
                        continue   # both blocks or neither
                    # One motif per length (TOP_N_MOTIFS == 1): the header format
                    # below does not compose over several motifs.
                    m = top_motifs[0]
                    r = r_top[0]
                    pat_str = ",".join(map(str, m["pattern"]))
                    r_pat_str = ",".join(r["pattern"])
                    by_form = m.get("count_by_form", {})
                    n_rectus = by_form.get("original", m["count"])
                    n_inv = by_form.get("I", 0)
                    # Flatten snippets to one line: an embedded newline would wrap
                    # the line and its continuation could be mis-read as a metadata
                    # field by downstream parsers.
                    abc_str = " ".join(m["abc"].split())
                    r_abc_str = " ".join(r["abc"].split())
                    # Order: both abstract motifs, then each one's count and
                    # realization, then the melodic inversion lines.
                    lines = [
                        f"%motif:v1:step_skip_leap: {pat_str} ",
                        f"%motif:v1:rhythm: {r_pat_str} ",
                        f"%motif:count: {n_rectus} ",
                        f"%motif:abc: {abc_str} ",
                        f"%motif:rhythm:count: {r['count']} ",
                        f"%motif:rhythm:abc: {r_abc_str} ",
                        f"%motif:inversion_count: {n_inv} ",
                    ]
                    inv_snip = m.get("transforms", {}).get("I")
                    if n_inv and inv_snip:
                        lines.append(f"%motif:abc:inversion_instance: {' '.join(inv_snip.split())} ")
                    motif_line_by_len[motif_len] = "\n".join(lines) + "\n"
        except Exception as e:
            print(f"Motif extraction failed for {abc_name} {key}: {e}")

        # rest reduction
        metadata_lines, prefix_dict, left_barline_dict, bar_text_dict, right_barline_dict = \
            extract_barline_and_bartext_dict(transposed_abc_lines)
        
        if VOICE1_ONLY:
            # Keep only V:1: reduce the %%score line to voice 1 and drop the other
            # voices' V: header lines.
            v1_metadata_lines = []
            for mline in metadata_lines:
                if mline.startswith('%%score') or mline.startswith('score'):
                    v1_metadata_lines.append(re.sub(r'^(%*score).*', r'\g<1> 1', mline.rstrip()) + '\n')
                elif mline.startswith('V:') and re.match(r'^V:\s*1\b', mline) is None:
                    continue
                else:
                    v1_metadata_lines.append(mline)
            metadata_lines = v1_metadata_lines

        # Where the motif header goes: *just before* the score or %%score line if
        # present, else before the first V: line
        insert_idx = len(metadata_lines)
        score_idx = -1
        for idx, mline in enumerate(metadata_lines):
            if mline.startswith('score') or mline.startswith('%%score'):
                score_idx = idx
                break

        if score_idx != -1:
            insert_idx = score_idx
        else:
            for idx, mline in enumerate(metadata_lines):
                if mline.startswith('V:'):
                    insert_idx = idx
                    break

        # Build the tunebody once — it is identical across motif lengths.
        tunebody_lines = []
        for i in range(len(bar_text_dict['V:1'])):
            line = ''
            for symbol in prefix_dict.keys():
                if VOICE1_ONLY and symbol != 'V:1':
                    continue
                valid_flag = False
                for char in bar_text_dict[symbol][i]:
                    if char.isalpha() and not char in ['Z', 'z', 'X', 'x']:
                        valid_flag = True
                        break
                if VOICE1_ONLY:
                    # With no other voice left to carry an all-rest bar, dropping it
                    # would silently shorten the piece — keep every V:1 bar.
                    valid_flag = True
                if valid_flag:
                    if i == 0:
                        part_patch = '[' + symbol + ']' + prefix_dict[symbol] + left_barline_dict[symbol][0] + bar_text_dict[symbol][0] + right_barline_dict[symbol][0]
                    else:
                        part_patch = '[' + symbol + ']' + bar_text_dict[symbol][i] + right_barline_dict[symbol][i]
                    line += part_patch
            line += '\n'
            tunebody_lines.append(line)

        # One real datapoint per motif length: same tunebody, length-L motif header.
        # A file is written for EVERY length even when that length yielded no motif
        # (header omitted), so the loader can always resolve <name>_len<L> in any key.
        for motif_len in MOTIF_LENGTHS:
            out_metadata = list(metadata_lines)
            motif_line = motif_line_by_len.get(motif_len)
            if motif_line:
                out_metadata.insert(insert_idx, motif_line)

            reduced_abc_name = f'{abc_name}_len{motif_len}_{key}'
            reduced_abc_path = os.path.join(AUGMENTED_FOLDER, key, reduced_abc_name + '.abc')

            with open(reduced_abc_path, 'w', encoding='utf-8') as w:
                w.writelines(out_metadata + tunebody_lines)

    return abc_name, ori_key





# Optional parallelism for large corpora (e.g. Irishman). Each file is processed
# independently and writes to its own per-key paths, so the loop is embarrassingly
# parallel. Default PREP_NUM_WORKERS=1 keeps the original sequential behavior (and
# its per-file traceback prints) byte-for-byte.
NUM_WORKERS = int(os.environ.get('PREP_NUM_WORKERS', '1'))


def _process_one_file(file):
    """Worker: preprocess one ABC file. Returns the index entry, or None on failure."""
    ori_abc_path = os.path.join(ORI_FOLDER, file)
    try:
        abc_name, ori_key = abc_preprocess_pipeline(ori_abc_path)
    except Exception:
        return None
    return {'path': os.path.join(AUGMENTED_FOLDER, abc_name), 'key': ori_key}


if __name__ == '__main__':

    data = []
    file_list = os.listdir(ORI_FOLDER)

    if NUM_WORKERS > 1:
        from multiprocessing import Pool
        n_fail = 0
        with Pool(NUM_WORKERS) as pool:
            for res in tqdm(pool.imap_unordered(_process_one_file, file_list, chunksize=16),
                            total=len(file_list)):
                if res is None:
                    n_fail += 1
                else:
                    data.append(res)
        if n_fail:
            print(f'{n_fail}/{len(file_list)} files failed to pre-process (skipped).')
    else:
        for file in tqdm(file_list):
            ori_abc_path = os.path.join(ORI_FOLDER, file)
            try:
                abc_name, ori_key = abc_preprocess_pipeline(ori_abc_path)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(ori_abc_path, 'failed to pre-process.')
                continue

            data.append({
                'path': os.path.join(AUGMENTED_FOLDER, abc_name),
                'key': ori_key
            })

    # Split train/eval at the PIECE level, then expand each piece into its
    # per-length entries (<path>_len4 .. _len10). Splitting after expansion would
    # leak the same tunebody (under different motif-length headers) across splits.
    random.shuffle(data)

    def expand_lengths(pieces):
        return [{'path': p['path'] + f'_len{motif_len}', 'key': p['key']}
                for p in pieces for motif_len in MOTIF_LENGTHS]

    eval_data = expand_lengths(data[ : int(EVAL_SPLIT * len(data))])
    train_data = expand_lengths(data[int(EVAL_SPLIT * len(data)) : ])
    data = expand_lengths(data)

    # Index targets default to <AUGMENTED_FOLDER>*.jsonl, but can be redirected to a
    # shared index (e.g. the Lieder v1 index) so a second corpus merges into it.
    data_index_path = os.environ.get('PREP_MAIN_INDEX', AUGMENTED_FOLDER + '.jsonl')
    eval_index_path = os.environ.get('PREP_EVAL_INDEX', AUGMENTED_FOLDER + '_eval.jsonl')
    train_index_path = os.environ.get('PREP_TRAIN_INDEX', AUGMENTED_FOLDER + '_train.jsonl')

    # Default ('overwrite'): main index is rewritten (original behavior). 'append':
    # add to the existing main index instead of clobbering the other corpus's entries.
    main_mode = 'a' if (INDEX_MODE == 'append' and os.path.exists(data_index_path)) else 'w'
    with open(data_index_path, main_mode, encoding='utf-8') as w:
        for d in data:
            w.write(json.dumps(d) + '\n')
    if os.path.exists(eval_index_path):
        with open(eval_index_path, 'a', encoding='utf-8') as w:
            for d in eval_data:
                w.write(json.dumps(d) + '\n')
    else:
        with open(eval_index_path, 'w', encoding='utf-8') as w:
            for d in eval_data:
                w.write(json.dumps(d) + '\n')
                
    if os.path.exists(train_index_path):
        with open(train_index_path, 'a', encoding='utf-8') as w:
            for d in train_data:
                w.write(json.dumps(d) + '\n')
    else:
        with open(train_index_path, 'w', encoding='utf-8') as w:
            for d in train_data:
                w.write(json.dumps(d) + '\n')

    

