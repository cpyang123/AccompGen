"""Backbone registry for the NotaGen hierarchical model.

The NotaGen architecture is backbone-agnostic: the patch-level encoder and the
char-level decoder are each an off-the-shelf HuggingFace causal LM.  This module
lets you "drop in" a new backbone (e.g. Llama) in place of the original GPT-2
without touching the model code — pick a name from ``available_backbones()`` and
set ``ENCODER_BACKBONE`` / ``DECODER_BACKBONE`` in your config.

To register a new architecture, add an entry to ``_REGISTRY`` mapping a name to
its config factory, encoder class (a bare ``*Model``) and decoder class (a
``*ForCausalLM`` / ``*LMHeadModel`` with a language-modeling head).

All runtime constants the model needs (patch size, motif bias, etc.) are stashed
as attributes on the returned config objects via ``make_encoder_config`` /
``make_decoder_config``, so the model classes never import the project config.
"""

from transformers import (
    GPT2Config, GPT2Model, GPT2LMHeadModel,
    LlamaConfig, LlamaModel, LlamaForCausalLM,
)


# ── Per-architecture config factories ────────────────────────────────────────
# Each takes the same four positional args so the registry can call them
# uniformly.  We force eager attention because the motif attention-bias hook
# (see model.py) adds a scalar to the *additive* attention mask before softmax —
# SDPA / flash attention do not expose that mask and would silently drop it.

def _make_gpt2_config(num_layers, hidden, vocab, max_pos):
    return GPT2Config(
        num_hidden_layers=num_layers,
        n_embd=hidden,
        num_attention_heads=hidden // 64,
        max_length=max_pos,
        max_position_embeddings=max_pos,
        vocab_size=vocab,
        attn_implementation="eager",
    )


def _make_llama_config(num_layers, hidden, vocab, max_pos):
    return LlamaConfig(
        num_hidden_layers=num_layers,
        hidden_size=hidden,
        intermediate_size=4 * hidden,
        num_attention_heads=hidden // 64,
        num_key_value_heads=hidden // 64,
        max_position_embeddings=max_pos,
        vocab_size=vocab,
        attn_implementation="eager",
    )


_REGISTRY = {
    "gpt2":  dict(config=_make_gpt2_config,  encoder=GPT2Model,  decoder=GPT2LMHeadModel),
    "llama": dict(config=_make_llama_config, encoder=LlamaModel, decoder=LlamaForCausalLM),
}


def available_backbones():
    return list(_REGISTRY)


def _spec(name):
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown backbone {name!r}. Available: {available_backbones()}. "
            f"Register new ones in notagen_core/backbones.py:_REGISTRY."
        )
    return _REGISTRY[name]


# ── Config builders (stash runtime constants on the config) ───────────────────

def make_encoder_config(name, num_layers, hidden, max_pos, *,
                        patch_size, motif_attention_bias):
    """Config for the patch-level encoder. vocab_size=1 — it consumes
    ``inputs_embeds`` only, never token ids."""
    cfg = _spec(name)["config"](num_layers, hidden, vocab=1, max_pos=max_pos)
    cfg.backbone = name
    cfg.patch_size = patch_size
    cfg.motif_attention_bias = motif_attention_bias
    return cfg


def make_decoder_config(name, num_layers, hidden, max_pos, *,
                        vocab=128, patch_sampling_batch_size=0):
    """Config for the char-level decoder (vocab = byte alphabet, 128)."""
    cfg = _spec(name)["config"](num_layers, hidden, vocab=vocab, max_pos=max_pos)
    cfg.backbone = name
    cfg.patch_sampling_batch_size = patch_sampling_batch_size
    return cfg


def build_notagen_configs(*, encoder_backbone, decoder_backbone,
                          patch_num_layers, char_num_layers, hidden_size,
                          patch_length, patch_size, motif_attention_bias,
                          patch_sampling_batch_size,
                          encoder_hidden_size=None, decoder_hidden_size=None):
    """One-stop builder so every entry point constructs configs identically.

    ``encoder_hidden_size`` / ``decoder_hidden_size`` default to ``hidden_size``.
    Set ``encoder_hidden_size`` larger to use a wider patch encoder (e.g. a
    distilled student); NotaGenLMHeadModel then auto-inserts a projection from
    the encoder width down to the decoder width.
    """
    enc_h = encoder_hidden_size or hidden_size
    dec_h = decoder_hidden_size or hidden_size
    encoder_config = make_encoder_config(
        encoder_backbone, patch_num_layers, enc_h, patch_length,
        patch_size=patch_size, motif_attention_bias=motif_attention_bias,
    )
    decoder_config = make_decoder_config(
        decoder_backbone, char_num_layers, dec_h, patch_size + 1,
        patch_sampling_batch_size=patch_sampling_batch_size,
    )
    return encoder_config, decoder_config


# ── Model construction + introspection ───────────────────────────────────────

def build_encoder(config):
    """Bare encoder model (no LM head) from a config built above."""
    return _spec(config.backbone)["encoder"](config)


def build_decoder(config):
    """Causal-LM decoder (with LM head) from a config built above."""
    return _spec(config.backbone)["decoder"](config)


def iter_layers(base):
    """Transformer block list of a *bare* encoder model, across backbones.

    GPT2Model exposes ``.h``; LlamaModel exposes ``.layers``.
    """
    if hasattr(base, "h"):
        return base.h
    if hasattr(base, "layers"):
        return base.layers
    raise AttributeError(
        f"Cannot locate transformer layers on {type(base).__name__}; "
        f"extend notagen_core.backbones.iter_layers for this backbone."
    )
