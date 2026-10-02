from pathlib import Path
import sys
import wave
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motifs import DEFAULT_NOTES, abstract_pattern, concrete_motif, concrete_rhythm, build_prompt, rhythm_pattern
from engine import CachedDecoder, encode_prompt, decode_patch
from outputs import clean_abc, export_result
from notagen_core import NotaGenLMHeadModel, build_notagen_configs


@pytest.mark.parametrize("length", range(4, 11))
def test_abstract_lengths(length):
    value = ",".join(["0"] + ["+1"] * (length - 1))
    assert len(abstract_pattern(value).split(",")) == length


@pytest.mark.parametrize("value", ["0,1,2", "0," + ",".join(["1"] * 10), "1,1,1,1", "0,0,1,1", "0,4,1,1", "0,1,x,1", "0,1,1,1\n%injected", ""])
def test_invalid_abstract(value):
    with pytest.raises(ValueError):
        abstract_pattern(value)


def test_concrete_spelling_octaves_and_durations():
    pattern, snippet = concrete_motif([["C4", "1"], ["D#4", "1/2"], ["F4", "1/3"], ["Bb3", "2"]])
    assert pattern == "0,1,2,-3"
    assert snippet == "C2 ^D F2/3 _B,4"
    assert concrete_motif(DEFAULT_NOTES)[0] == "0,3,-1,-1"


@pytest.mark.parametrize("rows", [[], [["C4", "1"]] * 4, [["Z4", "1"]] * 4,
                                    [["C4", "-1"]] * 4, [["C4", "0"]] * 4,
                                    [["C4", "1/0"]] * 4, [["C4", "nan"]] * 4])
def test_bad_concrete(rows):
    with pytest.raises(ValueError):
        concrete_motif(rows)


def test_concrete_prompt_order_and_patch_bias():
    prompt, pattern, snippet, rhythm = build_prompt("Concrete notes", "", DEFAULT_NOTES, ("Romantic", "Schubert, Franz", "Art Song"))
    # Training header order: contour, rhythm, count, abc, rhythm count, rhythm abc, inversion.
    assert (prompt.index("%motif:v1:step_skip_leap:") < prompt.index("%motif:v1:rhythm:") < prompt.index("%motif:count:")
            < prompt.index("%motif:abc:") < prompt.index("%motif:rhythm:count:") < prompt.index("%motif:rhythm:abc:")
            < prompt.index("%motif:inversion_count:"))
    assert rhythm == "1,1/2,1/2,2" and f"%motif:v1:rhythm: {rhythm} \n" in prompt
    assert f"%motif:rhythm:abc: {snippet} \n" in prompt
    assert "L:1/8\nM:4/4\nK:C\n" in prompt
    patches, flags = encode_prompt(prompt)
    assert len(patches) == len(flags)
    assert all(len(patch) == 16 for patch in patches)
    assert "".join(decode_patch(patch) for patch in patches) == prompt
    assert any(flags) and not flags[0]


@pytest.mark.parametrize("tag", ["%motif:abc: ", "%motif:abc:inversion_instance: "])
def test_realization_uses_training_patch_boundaries(tag):
    # Expected segments from finetune/utils.py Patchilizer, including its
    # final non-voice bar merge. Tag and ABC must never share a patch.
    prompt = tag + "C2 D2 | E2 F2 | G2 A2 \n"
    segments = [tag, "C2 D2 ", "| E2 F2 | G2 A2 \n"]
    expected = [[1] * 15 + [2]]
    for segment in segments:
        ids = list(segment.encode("ascii"))
        if len(ids) % 16:
            ids.append(2)
        expected.extend(ids[i:i+16] + [0] * (16-len(ids[i:i+16])) for i in range(0,len(ids),16))
    patches, flags = encode_prompt(prompt)
    assert patches == expected
    assert flags == [False] + [True] * (len(expected)-1)
    assert "".join(map(decode_patch, patches)) == prompt


def test_realization_attention_bias_is_applied_to_logits():
    model = tiny_model()
    patches, flags = encode_prompt("%Romantic\n%motif:abc: C2 G F E4 \nK:C\n")
    ids = torch.tensor([patches])
    encoder = model.patch_level_decoder
    seen = []
    # Observe actual attention masks received by each GPT-2 attention layer.
    hooks = [block.attn.register_forward_pre_hook(
        lambda module,args,kwargs: seen.append(kwargs['attention_mask'].clone()),
        with_kwargs=True) for block in encoder.base.h]
    with torch.inference_mode():
        biased = CachedDecoder(model).encode(patches, flags)
        unbiased = CachedDecoder(model).encode(patches, [False]*len(flags))
    for hook in hooks:
        hook.remove()
    assert not torch.allclose(biased, unbiased)
    expected = torch.tensor(flags).float()*4
    for mask in seen[:len(encoder.base.h)]:
        torch.testing.assert_close(mask.flatten(), expected)
    for mask in seen[len(encoder.base.h):]:
        assert torch.count_nonzero(mask) == 0


def tiny_model():
    torch.manual_seed(1)
    configs = build_notagen_configs(encoder_backbone="gpt2", decoder_backbone="gpt2",
                                    patch_num_layers=2, char_num_layers=2, hidden_size=64,
                                    patch_length=1024, patch_size=16, motif_attention_bias=4,
                                    patch_sampling_batch_size=0)
    return NotaGenLMHeadModel(encoder_config=configs[0], decoder_config=configs[1]).eval()


def test_cached_patch_logits_match_original_with_growing_motif_bias():
    model = tiny_model()
    patches, flags = encode_prompt("%Romantic\n%motif:v1:step_skip_leap: 0,3,-1,-1 \n%motif:abc: C2 G | F E4 \nL:1/8\n")
    cached = CachedDecoder(model)
    with torch.inference_mode():
        cached.encode(patches[:2], flags[:2])
        for stop in range(3, len(patches) + 1):
            actual = cached.encode([patches[stop - 1]], flags[:stop])
            ids = torch.tensor([patches[:stop]])
            bias = torch.tensor([flags[:stop]]).float() * 4
            expected = model.patch_level_decoder(ids, torch.ones(1, stop), bias).last_hidden_state[0, -1]
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_cached_character_probabilities_match_original():
    model = tiny_model()
    encoded = torch.randn(64)
    tokens = [1, 67, 50, 32, 68]
    base = model.char_level_decoder.base
    past = None
    with torch.inference_mode():
        for i in range(len(tokens)):
            embeds = encoded.reshape(1, 1, -1) if i == 0 else base.get_input_embeddings()(torch.tensor([[tokens[i]]]))
            output = base(inputs_embeds=embeds, past_key_values=past, use_cache=True)
            past = output.past_key_values
            actual = torch.softmax(output.logits[0, -1], -1)
            expected = model.char_level_decoder.generate(encoded, torch.tensor(tokens[:i + 1]))
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)


def test_export_xml_audio_and_request_isolation():
    raw = "%motif:abc: C G F E\n%%score 1\nL:1/8\nM:4/4\nK:C\nV:1\n[r:0/0][V:1]C2 G F E4|\n"
    one, two = export_result(raw, 120), export_result(raw, 120)
    assert one[-1] == two[-1] == ""
    assert one[2] != two[2]
    root = ET.parse(one[3]).getroot()
    assert len(root.findall(".//note/pitch")) == 4
    assert "%motif:abc: C G F E" in one[0] and "[r:" not in one[0]
    assert 'sandbox="allow-scripts"' in one[5]
    with wave.open(one[4]) as audio:
        assert audio.getnframes() > 22050
        pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
        assert abs(pcm).max() > 1000


def test_partial_measures_and_include_directives_not_exported():
    raw = "%%abc-include /etc/passwd\nI:abc-include /etc/passwd\nK:C\nV:1\n[r:0/4][V:1]CDEF|\n[r:1/3][V:1]GAB"
    abc = clean_abc(raw)
    assert "include" not in abc and "GAB" not in abc


@pytest.mark.parametrize('mode', ['Abstract motif', 'Concrete notes'])
def test_motif_prefill_reaches_decoder_and_survives_abc_export(monkeypatch, mode):
    import engine
    from verification import find_motif
    prompt, pattern, snippet, rhythm = build_prompt(mode, '0,3,-1,-1', DEFAULT_NOTES,
                                                    ('Romantic', 'Schubert, Franz', 'Art Song'))
    received = []
    class Decoder:
        def __init__(self, model):
            pass
        def encode(self, patches, flags):
            received.append((patches, flags))
            return None
        def patch(self, *args):
            return [1, 2] + [0] * 14  # End generation immediately after prefill.
    monkeypatch.setattr(engine, 'CachedDecoder', Decoder)
    updates = list(engine.generate(None, prompt))
    assert len(received) == 1 and updates[-1].text == prompt
    patches, flags = received[0]
    assert ''.join(map(decode_patch, patches)) == prompt
    conditioned_text = ''.join(decode_patch(p) for p, flag in zip(patches, flags) if flag)
    assert f'%motif:v1:step_skip_leap: {pattern}' in conditioned_text
    if mode == 'Concrete notes':
        assert f'%motif:abc: {snippet}' in conditioned_text
    raw = prompt + 'L:1/8\nM:4/4\nK:C\nV:1\n[V:1]C2 G F E4|\n'
    abc = clean_abc(raw)
    assert clean_abc(abc) == abc
    for line in prompt.splitlines():
        if line.startswith('%motif:'):
            assert line.strip() in abc.splitlines()
    assert find_motif(abc, mode, pattern, DEFAULT_NOTES).count == 1


def test_preserved_motif_metadata_does_not_enable_include_directives():
    raw = '%motif:abc: C G F E\n%%abc-include /etc/passwd\nI:abc-include /etc/passwd\nL:1/8\nK:C\nV:1\n[V:1]C G F E|\n'
    abc = clean_abc(raw)
    assert '%motif:abc: C G F E' in abc
    assert 'abc-include' not in abc


def test_concrete_barlines_survive_prompt_encoding_without_changing_contour():
    rows = DEFAULT_NOTES[:2] + [['|', '']] + DEFAULT_NOTES[2:] + [['|', '']]
    prompt, pattern, snippet, rhythm = build_prompt('Concrete notes', '', rows, ('Romantic',))
    assert pattern == concrete_motif(DEFAULT_NOTES)[0]
    assert rhythm == concrete_rhythm(DEFAULT_NOTES) == '1,1/2,1/2,2'
    assert snippet == 'C2 G | F E4 |'
    patches, _ = encode_prompt(prompt)
    assert ''.join(map(decode_patch, patches)) == prompt
    assert f'%motif:abc: {snippet}' in prompt


@pytest.mark.parametrize('count', [4, 10, 11])
def test_barlines_do_not_count_toward_note_limits(count):
    rows = []
    for i in range(count):
        rows.extend([['C4' if i % 2 == 0 else 'D4', '1'], ['|', '']])
    if count > 10:
        with pytest.raises(ValueError, match='4–10 notes'):
            concrete_motif(rows)
    else:
        assert len(concrete_motif(rows)[0].split(',')) == count


@pytest.mark.parametrize('rows', [
    [['|', '']] + DEFAULT_NOTES,
    DEFAULT_NOTES[:2] + [['|', ''], ['|', '']] + DEFAULT_NOTES[2:],
    DEFAULT_NOTES + [['|', '1']],
])
def test_invalid_barlines_rejected(rows):
    with pytest.raises(ValueError, match='barline|barlines'):
        concrete_motif(rows)


def test_concrete_naturals_only_cancel_accidentals_within_the_measure():
    from verification import find_motif
    rows = [['C#4', '1'], ['D4', '1'], ['C4', '1'], ['E4', '1']]
    pattern, snippet = concrete_motif(rows)
    assert snippet == '^C2 D2 =C2 E2'
    abc = 'X:1\nL:1/8\nM:4/4\nK:C\nV:1\n[V:1]' + snippet + '|\n'
    assert find_motif(abc, 'Concrete notes', pattern, rows).count == 1
    rows.insert(2, ['|', ''])
    assert concrete_motif(rows)[1] == '^C2 D2 | C2 E2'


def test_concrete_accidentals_do_not_force_naturals_in_other_octaves():
    rows = [['C#4', '1'], ['C5', '1'], ['D4', '1'], ['E4', '1']]
    assert concrete_motif(rows)[1] == '^C2 c2 D2 E2'


def test_abstract_rhythm_line_is_optional_and_follows_contour():
    style = ('Romantic', 'Schubert, Franz', 'Art Song')
    prompt, pattern, snippet, rhythm = build_prompt('Abstract motif', '0,3,-1,-1', DEFAULT_NOTES, style)
    assert rhythm is None and '%motif:v1:rhythm:' not in prompt and snippet is None
    assert prompt.endswith('%motif:v1:step_skip_leap: 0,3,-1,-1 \n')
    prompt, pattern, snippet, rhythm = build_prompt('Abstract motif', '0,3,-1,-1', DEFAULT_NOTES, style, 3, ' 1, 1/2 ,1/2, 2 ')
    assert rhythm == '1,1/2,1/2,2'
    assert prompt.endswith('%motif:v1:step_skip_leap: 0,3,-1,-1 \n%motif:v1:rhythm: 1,1/2,1/2,2 \n')
    patches, flags = encode_prompt(prompt)
    assert ''.join(map(decode_patch, patches)) == prompt
    # Both motif lines receive the trained attention bias; the style lines do not.
    conditioned = ''.join(decode_patch(p) for p, flag in zip(patches, flags) if flag)
    assert '%motif:v1:rhythm: 1,1/2,1/2,2' in conditioned and 'Romantic' not in conditioned


@pytest.mark.parametrize('value', ['1,1,1', '1,1,1,1,1', '1,1,1,0', '1,1,1,1z', '1,1,1,1.5', '1,1,1,9', '1,1,1,1/5', 'a,b,c,d'])
def test_invalid_rhythm(value):
    with pytest.raises(ValueError):
        rhythm_pattern(value, 4)


def test_rhythm_canonical_spelling():
    assert rhythm_pattern('2/2,2/4,3/2,4', 4) == '1,1/2,3/2,4'
    assert rhythm_pattern('', 4) is None and rhythm_pattern(None, 4) is None
