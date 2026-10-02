import torch
import random
import bisect
import json
import re
import numpy as np
from config import *
from transformers import GPT2Model, GPT2LMHeadModel, LlamaModel, LlamaForCausalLM, PreTrainedModel
from samplings import top_p_sampling, top_k_sampling, temperature_sampling
from tokenizers import Tokenizer


class Patchilizer:
    def __init__(self, stream=PATCH_STREAM):
        self.stream = stream
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.special_token_id = 0

    def split_bars(self, body_lines):
        """
        Split a body of music into individual bars.
        """
        new_bars = []
        try:
            for line in body_lines:
                line_bars = re.split(self.regexPattern, line)
                line_bars = list(filter(None, line_bars))
                new_line_bars = []

                if len(line_bars) == 1:
                    new_line_bars = line_bars
                else:
                    if line_bars[0] in self.delimiters:
                        new_line_bars = [line_bars[i] + line_bars[i + 1] for i in range(0, len(line_bars), 2)]
                    else:
                        new_line_bars = [line_bars[0]] + [line_bars[i] + line_bars[i + 1] for i in range(1, len(line_bars), 2)]
                    if 'V' not in new_line_bars[-1]:
                        new_line_bars[-2] += new_line_bars[-1]  
                        new_line_bars = new_line_bars[:-1]
                new_bars += new_line_bars
        except:
            pass

        return new_bars

    def split_patches(self, abc_text, patch_size=PATCH_SIZE, generate_last=False):
        if not generate_last and len(abc_text) % patch_size != 0:
            abc_text += chr(self.eos_token_id)
        patches = [abc_text[i : i + patch_size] for i in range(0, len(abc_text), patch_size)]
        return patches

    def patch2chars(self, patch):
        """
        Convert a patch into a bar.
        """
        bytes = ''
        for idx in patch:
            if idx == self.eos_token_id:
                break
            if idx < self.eos_token_id:
                pass
            bytes += chr(idx)
        return bytes
        

    def patchilize_metadata(self, metadata_lines, return_weights=False):

        metadata_patches = []
        patch_weights = []
        for line in metadata_lines:
            is_motif = line.startswith('%motif:')
            weight = MOTIF_LOSS_WEIGHT if is_motif else 1.0

            abc_tag = re.match(r'^(%motif:(?:rhythm:)?abc(?::[a-z_]+)?: )', line)   # '%motif:abc: ', '%motif:abc:inversion_instance: ' or '%motif:rhythm:abc: '
            if abc_tag:
                # The preamble tag should be patched as metadata text
                # But the actual abc notes following it should be patched as tunebody (using split_bars and then split_patches)
                prefix = abc_tag.group(1)
                abc_part = line[len(prefix):]

                # patch the prefix as metadata
                prefix_patches = self.split_patches(prefix)
                metadata_patches += prefix_patches
                patch_weights += [weight] * len(prefix_patches)

                # patch the abc part using tunebody logic (split into bars, then patches)
                bars = self.split_bars([abc_part])
                for bar in bars:
                    bar_patches = self.split_patches(bar)
                    metadata_patches += bar_patches
                    patch_weights += [weight] * len(bar_patches)
            else:
                line_patches = self.split_patches(line)
                metadata_patches += line_patches
                patch_weights += [weight] * len(line_patches)

        if return_weights:
            return metadata_patches, patch_weights
        return metadata_patches
    
    def patchilize_tunebody(self, tunebody_lines, encode_mode='train'):

        tunebody_patches = []
        bars = self.split_bars(tunebody_lines)
        if encode_mode == 'train':
            for bar in bars:
                tunebody_patches += self.split_patches(bar)
        elif encode_mode == 'generate':
            for bar in bars[:-1]:
                tunebody_patches += self.split_patches(bar)
            tunebody_patches += self.split_patches(bars[-1], generate_last=True)
       
        return tunebody_patches

    def encode_train(self, abc_text, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=True, cut=True):

        lines = abc_text.split('\n')
        lines = list(filter(None, lines))
        lines = [line + '\n' for line in lines]

        tunebody_index = -1
        for i, line in enumerate(lines):
            if line.startswith('[V:'):
                tunebody_index = i
                break
            if line.startswith('%motif:') and tunebody_index == -1:
                pass # keep looking

        metadata_lines = lines[ : tunebody_index]
        tunebody_lines = lines[tunebody_index : ]

        if self.stream:
            tunebody_lines = ['[r:' + str(line_index) + '/' + str(len(tunebody_lines) - line_index - 1) + ']' + line for line_index, line in
                                enumerate(tunebody_lines)]    

        metadata_patches, metadata_weights = self.patchilize_metadata(metadata_lines, return_weights=True)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='train')
        tunebody_weights = [1.0] * len(tunebody_patches)

        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)
            eos_patch = chr(self.bos_token_id) + chr(self.eos_token_id) * (patch_size - 1)

            metadata_patches = [bos_patch] + metadata_patches
            metadata_weights = [1.0] + metadata_weights
            tunebody_patches = tunebody_patches + [eos_patch]
            tunebody_weights = tunebody_weights + [1.0]

        if self.stream:
            if len(metadata_patches) + len(tunebody_patches) > patch_length:
                available_cut_indexes = [0] + [index + 1 for index, patch in enumerate(tunebody_patches) if '\n' in patch]
                line_index_for_cut_index = list(range(len(available_cut_indexes)))
                end_index = len(metadata_patches) + len(tunebody_patches) - patch_length
                biggest_index = bisect.bisect_left(available_cut_indexes, end_index)
                available_cut_indexes = available_cut_indexes[:biggest_index + 1]

                if len(available_cut_indexes) == 1:
                    choices = ['head']
                elif len(available_cut_indexes) == 2:
                    choices = ['head', 'tail']
                else:
                    choices = ['head', 'tail', 'middle']
                choice = random.choice(choices)
                if choice == 'head':
                    patches = metadata_patches + tunebody_patches[0:]
                    weights = metadata_weights + tunebody_weights[0:]
                else:
                    if choice == 'tail':
                        cut_index = len(available_cut_indexes) - 1
                    else:
                        cut_index = random.choice(range(1, len(available_cut_indexes) - 1))

                    line_index = line_index_for_cut_index[cut_index]
                    stream_tunebody_lines = tunebody_lines[line_index : ]

                    stream_tunebody_patches = self.patchilize_tunebody(stream_tunebody_lines, encode_mode='train')
                    if add_special_patches:
                        stream_tunebody_patches = stream_tunebody_patches + [eos_patch]
                    stream_tunebody_weights = [1.0] * len(stream_tunebody_patches)
                    patches = metadata_patches + stream_tunebody_patches
                    weights = metadata_weights + stream_tunebody_weights
            else:
                patches = metadata_patches + tunebody_patches
                weights = metadata_weights + tunebody_weights
        else:
            patches = metadata_patches + tunebody_patches
            weights = metadata_weights + tunebody_weights

        if cut:
            patches = patches[ : patch_length]
            weights = weights[ : patch_length]

        # encode to ids
        id_patches = []
        for patch in patches:
            id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
            id_patches.append(id_patch)

        return id_patches, weights

    def encode_generate(self, abc_code, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=True):

        lines = abc_code.split('\n')
        lines = list(filter(None, lines))
    
        tunebody_index = None
        for i, line in enumerate(lines):
            if line.startswith('[V:') or line.startswith('[r:'):
                tunebody_index = i
                break
            if line.startswith('%motif:') and tunebody_index is None:
                pass # keep looking
    
        metadata_lines = lines[ : tunebody_index]
        tunebody_lines = lines[tunebody_index : ]   
    
        metadata_lines = [line + '\n' for line in metadata_lines]
        if self.stream:
            if not abc_code.endswith('\n'): 
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [tunebody_lines[-1]]
            else:
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]
        else:
            tunebody_lines = [line + '\n' for line in tunebody_lines]
    
        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='generate')
    
        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)

            metadata_patches = [bos_patch] + metadata_patches
    
        patches = metadata_patches + tunebody_patches
        patches = patches[ : patch_length]

        # encode to ids
        id_patches = []
        for patch in patches:
            if len(patch) < PATCH_SIZE and patch[-1] != chr(self.eos_token_id):
                id_patch = [ord(c) for c in patch]
            else:
                id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
            id_patches.append(id_patch)
        
        return id_patches

    def decode(self, patches):
        """
        Decode patches into music.
        """
        return ''.join(self.patch2chars(patch) for patch in patches)


# ── Centralized, backbone-agnostic model classes ─────────────────────────────
# PatchLevelDecoder / CharLevelDecoder / NotaGenLMHeadModel now live in the
# shared notagen_core package, so a new backbone (gpt2, llama, ...) can be
# dropped in purely via config (ENCODER_BACKBONE / DECODER_BACKBONE). The
# Patchilizer above intentionally stays per-directory.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from notagen_core import (  # noqa: E402,F401
    PatchLevelDecoder, CharLevelDecoder, NotaGenLMHeadModel, safe_normalize_probs,
)
