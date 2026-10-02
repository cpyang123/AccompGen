"""Backbone-agnostic NotaGen hierarchical model.

This is the single source of truth for the model classes.  The patch-level
encoder and char-level decoder backbones are constructed via
``notagen_core.backbones`` and can be swapped (GPT-2, Llama, ...) purely through
config — see ``ENCODER_BACKBONE`` / ``DECODER_BACKBONE``.

All runtime constants (patch size, motif attention bias, patch-sampling batch
size) are read from the config objects, which ``backbones.make_*_config``
populates.  This module deliberately does NOT import the project ``config`` so it
stays importable from any working directory.

Backbone porting notes
-----------------------
Only three things here are backbone-specific, and all are handled portably:
  * hidden size           -> ``config.hidden_size`` (GPT2Config aliases n_embd)
  * input embedding table -> ``base.get_input_embeddings().weight``
  * transformer block list -> ``backbones.iter_layers(base)`` (for the motif hook)
The motif attention-bias hook adds a scalar to the additive attention mask, which
requires eager attention (enforced in backbones.py).
"""

import random

import numpy as np
import torch
from transformers import PreTrainedModel
from samplings import top_p_sampling, top_k_sampling, temperature_sampling

from .backbones import build_encoder, build_decoder, iter_layers

# Byte alphabet size: NotaGen represents music as raw chars (one-hot over 128).
BYTE_VOCAB = 128


class PatchLevelDecoder(PreTrainedModel):
    """Patch-level encoder: generates patch features auto-regressively."""

    def __init__(self, config):
        super().__init__(config)
        self.patch_size = config.patch_size
        self.patch_embedding = torch.nn.Linear(self.patch_size * BYTE_VOCAB, config.hidden_size)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.base = build_encoder(config)
        # [batch, 1, 1, seq_len] scalar bias added to QK attention logits before softmax
        self._motif_bias = None
        for block in iter_layers(self.base):
            block.register_forward_pre_hook(self._inject_motif_bias, with_kwargs=True)

    def _inject_motif_bias(self, module, args, kwargs):
        if self._motif_bias is not None:
            mask = kwargs.get('attention_mask')
            if mask is not None:
                kwargs['attention_mask'] = mask + self._motif_bias.to(mask.dtype)
        return args, kwargs

    def forward(self,
                patches: torch.Tensor,
                masks=None,
                motif_attention_bias=None) -> torch.Tensor:
        """
        :param patches: the patches to be encoded
        :param masks: the masks for the patches
        :param motif_attention_bias: [batch, seq_len] scalar bias added to QK^T logits before softmax
        :return: the encoded patches
        """
        patches = torch.nn.functional.one_hot(patches, num_classes=BYTE_VOCAB).to(self.dtype)
        patches = patches.reshape(len(patches), -1, self.patch_size * BYTE_VOCAB)
        patches = self.patch_embedding(patches.to(self.device))

        # Broadcast bias to [batch, 1, 1, seq_len] to match the extended attention mask shape
        self._motif_bias = motif_attention_bias[:, None, None, :] if motif_attention_bias is not None else None

        try:
            if masks is None:
                result = self.base(inputs_embeds=patches)
            else:
                result = self.base(inputs_embeds=patches, attention_mask=masks)
        finally:
            self._motif_bias = None

        return result


class CharLevelDecoder(PreTrainedModel):
    """Char-level decoder: generates chars within each patch, conditioned on the
    encoded patch feature."""

    def __init__(self, config):
        super().__init__(config)
        self.special_token_id = 0
        self.bos_token_id = 1
        self.patch_sampling_batch_size = config.patch_sampling_batch_size
        self.base = build_decoder(config)

    def forward(self,
                encoded_patches: torch.Tensor,
                target_patches: torch.Tensor):
        """
        :param encoded_patches: the encoded patches
        :param target_patches: the target patches
        :return: the output of the model
        """
        # preparing the labels for model training
        target_patches = torch.cat((torch.ones_like(target_patches[:, 0:1]) * self.bos_token_id, target_patches), dim=1)

        target_masks = target_patches == self.special_token_id
        labels = target_patches.clone().masked_fill_(target_masks, -100)

        # masking the labels for model training
        target_masks = torch.ones_like(labels)
        target_masks = target_masks.masked_fill_(labels == -100, 0)

        # select patches
        if self.patch_sampling_batch_size != 0 and self.patch_sampling_batch_size < target_patches.shape[0]:
            indices = list(range(len(target_patches)))
            random.shuffle(indices)
            selected_indices = sorted(indices[:self.patch_sampling_batch_size])

            target_patches = target_patches[selected_indices, :]
            target_masks = target_masks[selected_indices, :]
            encoded_patches = encoded_patches[selected_indices, :]

        # get input embeddings
        inputs_embeds = torch.nn.functional.embedding(target_patches, self.base.get_input_embeddings().weight)

        # concatenate the encoded patches with the input embeddings
        inputs_embeds = torch.cat((encoded_patches.unsqueeze(1), inputs_embeds[:, 1:, :]), dim=1)

        return self.base(inputs_embeds=inputs_embeds, attention_mask=target_masks, labels=labels)

    def generate(self,
                 encoded_patch: torch.Tensor,   # [hidden_size]
                 tokens: torch.Tensor):  # [1]
        """
        :param encoded_patch: the encoded patch
        :param tokens: already generated tokens in the patch
        :return: the probability distribution of next token
        """
        encoded_patch = encoded_patch.reshape(1, 1, -1)  # [1, 1, hidden_size]
        tokens = tokens.reshape(1, -1)

        # Get input embeddings
        tokens = torch.nn.functional.embedding(tokens, self.base.get_input_embeddings().weight)

        # Concatenate the encoded patch with the input embeddings
        tokens = torch.cat((encoded_patch, tokens[:, 1:, :]), dim=1)

        # Get output from model
        outputs = self.base(inputs_embeds=tokens)

        # Get probabilities of next token
        probs = torch.nn.functional.softmax(outputs.logits.squeeze(0)[-1], dim=-1)

        return probs


def safe_normalize_probs(probs):
    epsilon = 1e-12
    probs = np.array(probs, dtype=np.float64)
    probs = np.where(np.isnan(probs) | (probs < 0), 0, probs)
    probs = probs + epsilon
    s = probs.sum()
    if s > 0:
        probs = probs / s
    else:
        probs = np.zeros_like(probs)
        probs[0] = 1.0
    return probs


class NotaGenLMHeadModel(PreTrainedModel):
    """NotaGen: a hierarchical language model = patch-level encoder + char-level
    decoder.  Backbones for each level are chosen via the config objects."""

    def __init__(self, encoder_config, decoder_config):
        super().__init__(encoder_config)
        self.special_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.patch_size = encoder_config.patch_size
        self.motif_attention_bias = encoder_config.motif_attention_bias
        self.patch_level_decoder = PatchLevelDecoder(encoder_config)
        self.char_level_decoder = CharLevelDecoder(decoder_config)
        # Bridge a wider encoder to the char decoder's input width. When the two
        # match (the original NotaGen), this is a parameter-free Identity, so the
        # state_dict is unchanged and legacy gpt2 checkpoints load strict.
        if encoder_config.hidden_size != decoder_config.hidden_size:
            self.patch_proj = torch.nn.Linear(encoder_config.hidden_size, decoder_config.hidden_size)
        else:
            self.patch_proj = torch.nn.Identity()

    def forward(self,
                patches: torch.Tensor,
                masks: torch.Tensor,
                motif_weights: torch.Tensor = None):
        """
        :param patches: the patches to be encoded
        :param masks: the masks for the patches
        :param motif_weights: per-patch loss weights (motif patches weighted higher)
        :return: the decoded patches
        """
        patches = patches.reshape(len(patches), -1, self.patch_size)

        # Build [batch, seq_len] scalar bias: motif_attention_bias for motif patch key positions, 0 elsewhere
        motif_attention_bias = None
        if motif_weights is not None:
            motif_attention_bias = (motif_weights > 1.0).float() * self.motif_attention_bias

        encoded_patches = self.patch_level_decoder(patches, masks, motif_attention_bias)["last_hidden_state"]
        encoded_patches = self.patch_proj(encoded_patches)

        left_shift_masks = masks * (masks.flip(1).cumsum(1).flip(1) > 1)
        masks[:, 0] = 0

        encoded_patches = encoded_patches[left_shift_masks == 1]
        patches = patches[masks == 1]

        return self.char_level_decoder(encoded_patches, patches)

    def generate(self,
                 patches: torch.Tensor,
                 top_k=0,
                 top_p=1,
                 temperature=1.0,
                 motif_weights: torch.Tensor = None):
        """
        :param patches: the patches to be encoded
        :param motif_weights: optional [*, n_patches] per-patch weights. Patches with
            weight > 1.0 (the prompt's %motif: patches) receive the same attention
            bias used during training, so generation matches the training regime.
            The vector is padded with 1.0 (non-motif) if shorter than the current
            patch count, so the caller only needs to supply the fixed prompt weights.
        :return: the generated patch (list of token ids)
        """
        if patches.shape[-1] % self.patch_size != 0:
            tokens = patches[:, :, -(patches.shape[-1] % self.patch_size):].squeeze(0, 1)
            tokens = torch.cat((torch.tensor([self.bos_token_id], device=self.device), tokens), dim=-1)
            patches = patches[:, :, :-(patches.shape[-1] % self.patch_size)]
        else:
            tokens = torch.tensor([self.bos_token_id], device=self.device)

        patches = patches.reshape(len(patches), -1, self.patch_size)  # [bs, seq, patch_size]
        n_patches = patches.shape[1]

        # Reproduce the training-time motif attention bias. The bias is only injected
        # when an attention_mask is present (the hook adds to it), so we pass an
        # all-ones mask — equivalent to no mask for the encoder, but it lets the hook fire.
        masks = None
        motif_attention_bias = None
        if motif_weights is not None:
            mw = motif_weights.reshape(1, -1).to(device=patches.device, dtype=torch.float)
            if mw.shape[1] < n_patches:
                mw = torch.cat([mw, torch.ones(1, n_patches - mw.shape[1], device=mw.device)], dim=1)
            else:
                mw = mw[:, :n_patches]
            motif_attention_bias = (mw > 1.0).float() * self.motif_attention_bias
            masks = torch.ones(1, n_patches, dtype=torch.long, device=patches.device)

        encoded_patches = self.patch_level_decoder(patches, masks, motif_attention_bias)["last_hidden_state"]
        encoded_patches = self.patch_proj(encoded_patches)                          # [bs, seq, dec_hidden]
        generated_patch = []

        while True:
            prob = self.char_level_decoder.generate(encoded_patches[0][-1], tokens).cpu().detach().numpy()  # [128]
            prob = safe_normalize_probs(prob)
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True)
            prob = safe_normalize_probs(prob)
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True)
            prob = safe_normalize_probs(prob)
            token = temperature_sampling(prob, temperature=temperature)
            generated_patch.append(token)

            if len(tokens) >= self.patch_size:
                break
            else:
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)

        return generated_patch
