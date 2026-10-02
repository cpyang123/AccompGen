"""Shared NotaGen model core: backbone-agnostic hierarchical model + registry."""

from .backbones import (
    available_backbones,
    make_encoder_config,
    make_decoder_config,
    build_notagen_configs,
    build_encoder,
    build_decoder,
    iter_layers,
)
from .model import (
    PatchLevelDecoder,
    CharLevelDecoder,
    NotaGenLMHeadModel,
    safe_normalize_probs,
)

__all__ = [
    "available_backbones",
    "make_encoder_config",
    "make_decoder_config",
    "build_notagen_configs",
    "build_encoder",
    "build_decoder",
    "iter_layers",
    "PatchLevelDecoder",
    "CharLevelDecoder",
    "NotaGenLMHeadModel",
    "safe_normalize_probs",
]
