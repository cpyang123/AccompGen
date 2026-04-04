"""Dataset preprocessing.

This script is designed to be executed as:
    python data/2_data_preprocess.py

So paths are resolved relative to this file's directory.
"""

ORI_FOLDER = 'abcfiles'  # Relative to data/
INTERLEAVED_FOLDER = 'abcfiles_inter'   # Relative to data/
AUGMENTED_FOLDER = 'abcfiles_processed'   # Relative to data/
EVAL_SPLIT = 0.1    # The ratio of eval data 

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

from motif import get_best_motifs

# resolve folders relative to this script
DATA_DIR = os.path.dirname(__file__)
ORI_FOLDER = os.path.join(DATA_DIR, ORI_FOLDER)
INTERLEAVED_FOLDER = os.path.join(DATA_DIR, INTERLEAVED_FOLDER)
AUGMENTED_FOLDER = os.path.join(DATA_DIR, AUGMENTED_FOLDER)

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

        # extract motif from V:1 (full piece)
        motif_line = None
        try:
            # Use abctoolkit's header/parts split so we reliably get V:1 music content
            _metadata_lines, part_text_dict = extract_metadata_and_parts(transposed_abc_lines)
            v1_text = part_text_dict.get('V:1')
            if v1_text:
                top_motifs = get_best_motifs(v1_text, 
                # interval_mode= 'chromatic',
                # interval_mode="diatonic",
                interval_mode="step_skip_leap"
                # interval_mode="contour",
                # interval_mode="contour"
                )
                if top_motifs:
                    motif_strs = []
                    motif_abc_strs = []
                    for m in top_motifs[:3]:
                        pat_str = ",".join(map(str, m["pattern"]))
                        abc_str = m["abc"]
                        motif_abc_strs.append(f"{abc_str} ")
                        motif_strs.append(f"{pat_str} ")
                    motif_line = f"%motif:v1:step_skip_leap: {' ; '.join(motif_strs)}\n%motif:abc: {' ; '.join(motif_abc_strs)}\n"
        except Exception as e:
            print(f"Motif extraction failed for {abc_name} {key}: {e}")

        # rest reduction
        metadata_lines, prefix_dict, left_barline_dict, bar_text_dict, right_barline_dict = \
            extract_barline_and_bartext_dict(transposed_abc_lines)
        
        # Insert motif line into metadata
        if motif_line:
            # Insert *just before* the score or %%score line if present, else before the first V: line
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
            
            metadata_lines.insert(insert_idx, motif_line)

        reduced_abc_lines = metadata_lines
        for i in range(len(bar_text_dict['V:1'])):
            line = ''
            for symbol in prefix_dict.keys():
                valid_flag = False
                for char in bar_text_dict[symbol][i]:
                    if char.isalpha() and not char in ['Z', 'z', 'X', 'x']:
                        valid_flag = True
                        break
                if valid_flag:
                    if i == 0:
                        part_patch = '[' + symbol + ']' + prefix_dict[symbol] + left_barline_dict[symbol][0] + bar_text_dict[symbol][0] + right_barline_dict[symbol][0]
                    else:
                        part_patch = '[' + symbol + ']' + bar_text_dict[symbol][i] + right_barline_dict[symbol][i]
                    line += part_patch
            line += '\n'
            reduced_abc_lines.append(line)
            
            reduced_abc_name = abc_name + '_' + key
            reduced_abc_path = os.path.join(AUGMENTED_FOLDER, key, reduced_abc_name + '.abc')
        
            with open(reduced_abc_path, 'w', encoding='utf-8') as w:
                w.writelines(reduced_abc_lines)

    return abc_name, ori_key





if __name__ == '__main__':
    
    data = []
    file_list = os.listdir(ORI_FOLDER)
    
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

    random.shuffle(data)
    eval_data = data[ : int(EVAL_SPLIT * len(data))]
    train_data = data[int(EVAL_SPLIT * len(data)) : ]

    data_index_path = AUGMENTED_FOLDER + '.jsonl'
    eval_index_path = AUGMENTED_FOLDER + '_eval.jsonl'
    train_index_path = AUGMENTED_FOLDER + '_train.jsonl'


    with open(data_index_path, 'w', encoding='utf-8') as w:
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

    

