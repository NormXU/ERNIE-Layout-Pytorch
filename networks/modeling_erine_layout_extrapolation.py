# -*- coding:utf-8 -*-
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
# edited by Nuo Xu (Norm Inui)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


""" Modeling classes for ErnieLayout model."""
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (QuestionAnsweringModelOutput, TokenClassifierOutput)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_erine_layout import ErnieLayoutConfig
from .visual_backbone import ResNetCustomized

logger = logging.get_logger(__name__)
__all__ = [
    'ErnieLayoutModel', "ErnieLayoutPretrainedModel", "ErnieLayoutForTokenClassification",
    "ErnieLayoutForSequenceClassification", "ErnieLayoutForPretraining", "ErnieLayoutForQuestionAnswering"
]

warning_message = 'You may select either Alibi positional bias or T5 relative positional bias or RoPE, but please refrain from choosing more than one option simultaneously'

def exists(val):
    return val is not None


def pad_at_dim(t, pad, dim=-1, value=0.):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)


# Based on https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, seq_len):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    position_ids = torch.arange(seq_len).expand((1, -1))
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class AlibiPositionalBias(nn.Module):

    def __init__(self, heads, total_heads, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads

        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent=False)
        self.register_buffer('bias', None, persistent=False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):

        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(
            2 * closest_power_of_2)[0::2][:heads - closest_power_of_2]

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=0)
        self.register_buffer('bias', bias, persistent=False)

        return self.bias


class LearnedAlibiPositionalBias(AlibiPositionalBias):

    def __init__(self, heads, total_heads):
        super().__init__(heads, total_heads)
        log_slopes = torch.log(self.slopes)
        self.learned_logslopes = nn.Parameter(log_slopes)

    def forward(self, i, j):
        h, device = self.heads, self.device

        def get_slopes(param):
            return pad_at_dim(param.exp(), (0, h - param.shape[0]), dim=-2)

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            bias = self.bias[..., :i, :j]
        else:
            bias = self.get_bias(i, j, device)
            self.register_buffer('bias', bias, persistent=False)

        slopes = get_slopes(self.learned_logslopes)
        bias = bias * slopes

        return bias


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.cos_cached.size()[2]:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

class MixedNTKScalingRotaryEmbedding(RotaryEmbedding):
    """
        copied from LLamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla
        'fix_based' is inspired from  https://normxu.github.io/Rethinking-Rotary-Position-Embedding-2/
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, b=0.75, device=None):
        self.b = b
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if seq_len > self.max_position_embeddings:
            k = seq_len / self.max_position_embeddings
            a = np.log(k) / (self.dim // 2 ** self.b)
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            inv_freq /= (a * torch.arange(1, self.dim // 2 + 1).float().to(device) ** self.b).exp()
            self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """
        copied from LLamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla
        'fix_based' is inspired from  https://normxu.github.io/Rethinking-Rotary-Position-Embedding-2/
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0, fix_base=False):
        self.scaling_factor = scaling_factor
        self.fix_base = fix_base
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if seq_len > self.max_position_embeddings:
            lamda_factor = (
                                   (self.scaling_factor * seq_len / self.max_position_embeddings) - (
                                       self.scaling_factor - 1)
                           ) ** (self.dim / (self.dim - 2))
            base = self.base * lamda_factor
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            if self.fix_base:
                inv_freq = inv_freq * 1.0 / lamda_factor
            self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """Copied from LLamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for small
    absolute relative_position and larger buckets for larger absolute relative_positions. All relative positions
    >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket. This should
    allow for more graceful generalization to longer sequences than the model has been trained on.

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """

    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # Now n is in the range [0, inf)
    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) *
                                (num_buckets - max_exact)).to(torch.long)

    val_if_large = torch.minimum(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret


class ErnieLayoutPooler(nn.Module):

    def __init__(self, hidden_size, with_pool):
        super(ErnieLayoutPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.with_pool = with_pool

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.with_pool == 'tanh':
            pooled_output = self.activation(pooled_output)
        return pooled_output


class ErnieLayoutEmbeddings(nn.Module):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, config):
        super(ErnieLayoutEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))


    def _calc_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        return left_position_embeddings, upper_position_embeddings, right_position_embeddings, \
               lower_position_embeddings, h_position_embeddings, w_position_embeddings


    def forward(
            self,
            input_ids,
            bbox=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
    ):

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        words_embeddings = inputs_embeds

        if position_ids is None:
            ones = torch.ones_like(input_ids)
            seq_length = torch.cumsum(ones, dim=-1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True

        position_embeddings = self.position_embeddings(position_ids)

        x1, y1, x2, y2, h, w = self.embeddings._calc_spatial_position_embeddings(bbox)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (words_embeddings + position_embeddings + x1 + y1 + x2 + y2 + h + w + token_type_embeddings)

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ErnieLayoutPretrainedModel(PreTrainedModel):
    """
       An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
       models.
       """

    config_class = ErnieLayoutConfig
    base_model_prefix = "ernie_layout"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ErnieLayoutEncoder):
            module.gradient_checkpointing = value


class ErnieLayoutSelfOutput(nn.Module):

    def __init__(self, config):
        super(ErnieLayoutSelfOutput, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.drop_path_rate = 0.0 if not hasattr(config, "hidden_dropout_prob") else config.hidden_dropout_prob
        self.drop_path = DropPath(self.drop_path_rate) if self.drop_path_rate > 0.0 else nn.Identity()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.drop_path(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ErnieLayoutSelfAttention(nn.Module):

    def __init__(self, config):
        super(ErnieLayoutSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                             f"heads ({config.num_attention_heads})")
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias


        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.max_position_embeddings = config.max_position_embeddings

        self.use_entropy_scale = config.use_entropy_scale
        self.use_alibi = config.use_alibi
        self.use_rope_attention_bias = config.use_rope_attention_bias


        if config.use_alibi:
            alibi_num_heads = config.alibi_num_heads if hasattr(config, "alibi_num_heads") else self.num_attention_heads
            assert alibi_num_heads <= self.num_attention_heads, 'number of ALiBi heads must be less than the total number of heads'
            if config.learnable_alibi:
                self.alibi = LearnedAlibiPositionalBias(heads=alibi_num_heads, total_heads=self.num_attention_heads)
            else:
                self.alibi = AlibiPositionalBias(heads=alibi_num_heads, total_heads=self.num_attention_heads)

        if config.use_rope_attention_bias:
            max_position_embeddings = config.max_position_embeddings
            if config.consequent_visual_bias:
                max_position_embeddings += config.image_feature_pool_shape[0] * config.image_feature_pool_shape[1]
            else:
                visual_max_position_embeddings = config.image_feature_pool_shape[0] * \
                                                 config.image_feature_pool_shape[1]
                self.visual_rotary_emb = self._init_rope(config, visual_max_position_embeddings)
            self.rotary_emb = self._init_rope(config, max_position_embeddings)

    def _init_rope(self, config, max_position_embeddings):
        if config.rope_scaling_factor is None:
            rotary_emb = RotaryEmbedding(self.attention_head_size, max_position_embeddings=max_position_embeddings)
        else:
            scaling_type = config.rope_type
            scaling_factor = config.rope_scaling_factor
            if scaling_type == "linear":
                rotary_emb = LinearScalingRotaryEmbedding(
                    self.attention_head_size, max_position_embeddings=max_position_embeddings,
                    scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                rotary_emb = DynamicNTKScalingRotaryEmbedding(
                    self.attention_head_size, max_position_embeddings=max_position_embeddings,
                    scaling_factor=scaling_factor,
                    fix_base=config.fix_base
                )
            elif scaling_type == "mixed_base":
                rotary_emb = MixedNTKScalingRotaryEmbedding(
                    self.attention_head_size, max_position_embeddings=max_position_embeddings,
                    b=config.b
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
        return rotary_emb

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def compute_qkv(self, hidden_states):
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        return q, k, v


    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                past_key_value=None,
                output_attentions=False,
                rel_pos=None,
                rel_2d_pos=None):
        q, k, v = self.compute_qkv(hidden_states)

        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        seq_len = key_layer.shape[-2]
        # Start of applying RoPE on query & key
        if self.use_rope_attention_bias:
            # whether visual component share PE with layout component
            if not self.config.consequent_visual_bias:
                visual_seq_len = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
                cos_visual, sin_visual = self.visual_rotary_emb(value_layer, seq_len=visual_seq_len)
                cos_text, sin_text = self.rotary_emb(value_layer, seq_len=seq_len - visual_seq_len)
                cos = torch.cat([cos_text, cos_visual], dim=-2)
                sin = torch.cat([sin_text, sin_visual], dim=-2)
            else:
                cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)

            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, seq_len)
        # End of applying RoPE on query & key

        query_layer = query_layer / math.sqrt(self.attention_head_size)
        if self.use_entropy_scale:
            query_layer *= ((torch.arange(0, seq_len) + 1)[None, None, :, None].log() / np.log(
                self.max_position_embeddings)).clip(1).to(device=query_layer.device, dtype=query_layer.dtype)
        # [BSZ, NAT, L, L]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.has_relative_attention_bias:
            attention_scores += rel_pos
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos

        # start of Alibi
        if self.use_alibi:
            i, j = map(lambda t: t.shape[-2], (q, k))
            attention_scores += self.alibi(i, j)
        # end of Alibi

        attention_scores = attention_scores.float().masked_fill_(attention_mask.to(torch.bool),
                                                                 torch.finfo(attention_scores.dtype).min)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class ErnieLayoutAttention(nn.Module):

    def __init__(self, config):
        super(ErnieLayoutAttention, self).__init__()
        self.self = ErnieLayoutSelfAttention(config)
        self.output = ErnieLayoutSelfOutput(config)

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                past_key_value=None,
                output_attentions=False,
                rel_pos=None,
                rel_2d_pos=None):

        self_outputs = self.self(hidden_states,
                                 attention_mask,
                                 head_mask,
                                 past_key_value,
                                 output_attentions,
                                 rel_pos=rel_pos,
                                 rel_2d_pos=rel_2d_pos)
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        if output_attentions:
            outputs = [
                          attention_output,
                      ] + self_outputs[1:]
        else:
            outputs = [attention_output]
        return outputs


class ErnieLayoutEncoder(nn.Module):

    def __init__(self, config):
        super(ErnieLayoutEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([ErnieLayoutLayer(config) for _ in range(config.num_hidden_layers)])

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        assert not (
                self.has_relative_attention_bias and (
                    self.config.use_alibi or self.config.use_rope_attention_bias)), \
            warning_message
        assert not (self.config.use_rope_attention_bias and self.config.use_alibi), warning_message

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_onehot_size = config.rel_pos_bins
            self.rel_pos_bias = nn.Linear(self.rel_pos_onehot_size, config.num_attention_heads, bias=False)

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_2d_pos_onehot_size = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)

    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).type_as(hidden_states)
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            past_key_values=None,
            output_attentions=False,
            output_hidden_states=False,
            bbox=None,
            position_ids=None
    ):
        all_hidden_states = () if output_hidden_states else None

        rel_pos = self._cal_1d_pos_emb(hidden_states, position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(hidden_states, bbox) if self.has_spatial_attention_bias else None

        hidden_save = dict()
        hidden_save["input_hidden_states"] = hidden_states

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # gradient_checkpointing is set as False here so we remove some codes here
            hidden_save["input_attention_mask"] = attention_mask
            hidden_save["input_layer_head_mask"] = layer_head_mask
            layer_outputs = layer_module(hidden_states,
                                         attention_mask,
                                         layer_head_mask,
                                         past_key_value,
                                         output_attentions,
                                         rel_pos=rel_pos,
                                         rel_2d_pos=rel_2d_pos)

            hidden_states = layer_outputs[0]

            hidden_save["{}_data".format(i)] = hidden_states

        return hidden_states,


class ErnieLayoutIntermediate(nn.Module):

    def __init__(self, config):
        super(ErnieLayoutIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if config.hidden_act == "gelu":
            self.intermediate_act_fn = nn.GELU()
        else:
            assert False, "hidden_act is set as: {}, please check it..".format(config.hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ErnieLayoutOutput(nn.Module):

    def __init__(self, config):
        super(ErnieLayoutOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.drop_path_rate = 0.0 if not hasattr(config, "drop_path_rate") else config.drop_path_rate
        self.drop_path = DropPath(self.drop_path_rate) if self.drop_path_rate > 0.0 else nn.Identity()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.drop_path(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ErnieLayoutLayer(nn.Module):

    def __init__(self, config):
        super(ErnieLayoutLayer, self).__init__()
        # since chunk_size_feed_forward is 0 as default, no chunk is needed here.
        self.seq_len_dim = 1
        self.attention = ErnieLayoutAttention(config)
        self.add_cross_attention = False  # default as false
        self.intermediate = ErnieLayoutIntermediate(config)
        self.output = ErnieLayoutOutput(config)

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            past_key_value=None,
            output_attentions=False,
            rel_pos=None,
            rel_2d_pos=None
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states,
                                                attention_mask,
                                                head_mask,
                                                output_attentions=output_attentions,
                                                past_key_value=self_attn_past_key_value,
                                                rel_pos=rel_pos,
                                                rel_2d_pos=rel_2d_pos)
        attention_output = self_attention_outputs[0]
        layer_output = self.feed_forward_chunk(attention_output)

        if output_attentions:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
            outputs = [
                          layer_output,
                      ] + list(outputs)
        else:
            outputs = [layer_output]
        return outputs


class VisualBackbone(nn.Module):

    def __init__(self, config):
        super(VisualBackbone, self).__init__()

        self.backbone = ResNetCustomized(layers=101)

        self.register_buffer("pixel_mean", torch.tensor([103.53, 116.28, 123.675]).reshape([3, 1, 1]))
        self.register_buffer("pixel_std", torch.tensor([57.375, 57.12, 58.395]).reshape([3, 1, 1]))

        self.pool = nn.AdaptiveAvgPool2d(config.image_feature_pool_shape[:2])

    def forward(self, images):
        images_input = (images - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        features = self.pool(features).flatten(start_dim=2).transpose(1, 2).contiguous()
        return features


class ErnieLayoutModel(ErnieLayoutPretrainedModel):
    """
    The bare ErnieLayout Model outputting raw hidden-states.

    This model inherits from :class:`~torchnlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `torch.nn.Layer <https://www.torchtorch.org.cn/documentation
    /docs/en/api/torch/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (`int`):
            Vocabulary size of the XLNet model. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling ErnieLayoutModel.
        hidden_size (`int`, optional):
            Dimensionality of the encoder layers and the pooler layer. Defaults to ``768``.
        num_hidden_layers (`int`, optional):
            Number of hidden layers in the Transformer encoder. Defaults to ``12``.
        num_attention_heads (`int`, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to ``12``.
        intermediate_size (`int`, optional):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            Defaults to ``3072``.
        hidden_act (`str`, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other torch supported activation functions
            are supported. Defaults to ``"gelu"``.
        hidden_dropout_prob (`float`, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to ``0.1``.
        attention_probs_dropout_prob (`float`, optional):
            The dropout probability for all fully connected layers in the pooler.
            Defaults to ``0.1``.
        initializer_range (`float`, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Defaults to ``0.02``.
    """

    def __init__(
            self,
            config,
    ):
        super(ErnieLayoutModel, self).__init__(config)
        with_pool = 'tanh',
        self.config = config
        self.embeddings = ErnieLayoutEmbeddings(config)
        self.encoder = ErnieLayoutEncoder(config)
        self.pooler = ErnieLayoutPooler(config.hidden_size, with_pool)

        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        self.visual = VisualBackbone(config)
        self.visual_proj = nn.Linear(config.image_feature_pool_shape[-1], config.hidden_size)
        self.visual_act_fn = nn.GELU()
        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = nn.Parameter(nn.Embedding(1, config.hidden_size).weight[0])
        self.visual_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.visual_dropout = nn.Dropout(config.hidden_dropout_prob)

    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids):
        words_embeddings = self.embeddings.word_embeddings(input_ids)
        x1, y1, x2, y2, h, w = self.embeddings._calc_spatial_position_embeddings(bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + x1 + y1 + x2 + y2 + w + h + token_type_embeddings
        if position_ids is not None:
            position_embeddings = self.embeddings.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_img_embeddings(self, image, bbox, position_ids):
        x1, y1, x2, y2, h, w = self.embeddings._calc_spatial_position_embeddings(bbox)
        embeddings = x1 + y1 + x2 + y2 + w + h
        if image is not None:
            visual_embeddings = self.visual_act_fn(self.visual_proj(self.visual(image)))
            embeddings += visual_embeddings
        if position_ids is not None:
            position_embeddings = self.embeddings.position_embeddings(position_ids)
            embeddings += position_embeddings
        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.visual_dropout(embeddings)
        return embeddings

    def _calc_visual_bbox(self, image_feature_pool_shape, bbox, device, final_shape):
        visual_bbox_x = torch.div(
            torch.arange(
                0,
                1000 * (image_feature_pool_shape[1] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            ),
            self.config.image_feature_pool_shape[1],
            rounding_mode="floor",
        )
        visual_bbox_y = torch.div(
            torch.arange(
                0,
                1000 * (self.config.image_feature_pool_shape[0] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            ),
            self.config.image_feature_pool_shape[0],
            rounding_mode="floor",
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(image_feature_pool_shape[0], 1),
                visual_bbox_y[:-1].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(image_feature_pool_shape[0], 1),
                visual_bbox_y[1:].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, bbox.size(-1))

        visual_bbox = visual_bbox.repeat(final_shape[0], 1, 1)

        return visual_bbox

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        num_position_embeds_diff = new_num_position_embeddings - self.config.max_position_embeddings
        self.embeddings.position_ids = torch.arange(new_num_position_embeddings).expand((1, -1))

        # no resizing needs to be done if the length stays the same
        if num_position_embeds_diff == 0:
            return

        logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
        self.config.max_position_embeddings = new_num_position_embeddings

        old_position_embeddings_weight = self.embeddings.position_embeddings.weight

        self.embeddings.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size)

        with torch.no_grad():
            if num_position_embeds_diff > 0:
                self.embeddings.position_embeddings.weight[:-num_position_embeds_diff] = old_position_embeddings_weight
            else:
                self.embeddings.position_embeddings.weight = old_position_embeddings_weight[:num_position_embeds_diff]

    def forward(
            self,
            input_ids=None,
            bbox=None,
            image=None,
            token_type_ids=None,
            position_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_hidden_states=False,
            output_attentions=False
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if self.config.use_alibi or self.config.use_rope_attention_bias:
            position_ids = None
        else:
            if position_ids is None:
                seq_length = input_shape[1]
                position_ids = self.embeddings.position_ids[:, :seq_length]
                position_ids = position_ids.expand(input_shape)

        if bbox is None:
            bbox = torch.zeros(input_shape + [4], dtype=torch.long, device=device)

        text_layout_emb = self._calc_text_embeddings(
            input_ids=input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )


        visual_shape = list(input_shape)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
        visual_shape = torch.Size(visual_shape)
        final_shape = list(input_shape)
        final_shape[1] = final_shape[1] + visual_shape[1]
        final_shape = torch.Size(final_shape)

        visual_bbox = self._calc_visual_bbox(self.config.image_feature_pool_shape, bbox, device, final_shape)
        final_bbox = torch.cat([bbox, visual_bbox], dim=1)


        visual_position_ids = torch.arange(0, visual_shape[1], dtype=torch.long,
                                           device=device).repeat(input_shape[0], 1)
        if position_ids is None:
            final_position_ids = visual_position_ids
        else:
            final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)

        visual_attention_mask = torch.ones(visual_shape, device=device)
        final_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)

        visual_emb = self._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )
        final_emb = torch.cat([text_layout_emb, visual_emb], dim=1)

        extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            final_emb,
            extended_attention_mask,
            bbox=final_bbox,
            position_ids=final_position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class ErnieLayoutForSequenceClassification(ErnieLayoutPretrainedModel):

    def __init__(self, config):
        super(ErnieLayoutForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.label_smooth = getattr(config, "label_smooth", 0.0)
        self.ernie_layout = ErnieLayoutModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)
        self.init_weights()

    def get_input_embeddings(self):
        return self.ernie_layout.embeddings.word_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        self.ernie_layout.resize_position_embeddings(new_num_position_embeddings)

    def forward(
            self,
            input_ids=None,
            bbox=None,
            image=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None,
    ):
        input_shape = input_ids.size()
        device = input_ids.device

        visual_shape = list(input_shape)
        visual_shape[1] = self.ernie_layout.config.image_feature_pool_shape[0] * \
                          self.ernie_layout.config.image_feature_pool_shape[1]
        visual_shape = torch.Size(visual_shape)
        final_shape = list(input_shape)
        final_shape[1] = final_shape[1] + visual_shape[1]
        final_shape = torch.Size(final_shape)
        visual_bbox = self._calc_visual_bbox(self.config.image_feature_pool_shape, bbox, device, final_shape)

        visual_position_ids = torch.arange(0, visual_shape[1], dtype=torch.long,
                                           device=device).repeat(input_shape[0], 1)

        initial_image_embeddings = self.ernie_layout._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )

        outputs = self.ernie_layout(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        seq_length = input_ids.shape[1]
        # sequence out and image out
        sequence_output, final_image_embeddings = outputs[0][:, :seq_length], outputs[0][:, seq_length:]

        cls_final_output = sequence_output[:, 0, :]

        # average-pool the visual embeddings
        pooled_initial_image_embeddings = initial_image_embeddings.mean(dim=1)
        pooled_final_image_embeddings = final_image_embeddings.mean(dim=1)
        # concatenate with cls_final_output
        sequence_output = torch.cat([cls_final_output, pooled_initial_image_embeddings, pooled_final_image_embeddings],
                                    dim=1)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = logits,

        if labels is not None:
            loss_fct = CrossEntropyLoss(label_smoothing=self.label_smooth)

            loss = loss_fct(logits.reshape([-1, self.num_labels]), labels.reshape([
                -1,
            ]))

            outputs = (loss,) + outputs

        return outputs


class ErnieLayoutPredictionHead(nn.Module):
    """
    Bert Model with a `language modeling` head on top for CLM fine-tuning.
    """

    def __init__(self, hidden_size, vocab_size, activation, embedding_weights=None):
        super(ErnieLayoutPredictionHead, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder_weight = self.create_parameter(shape=[vocab_size, hidden_size],
                                                    dtype=self.transform.weight.dtype,
                                                    is_bias=False) if embedding_weights is None else embedding_weights
        self.decoder_bias = self.create_parameter(shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = torch.reshape(hidden_states, [-1, hidden_states.shape[-1]])
            hidden_states = hidden_states.gather(0, masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = torch.matmul(hidden_states, self.decoder_weight) + self.decoder_bias
        return hidden_states


class ErnieLayoutPretrainingHeads(nn.Module):

    def __init__(self, hidden_size, vocab_size, activation, embedding_weights=None):
        super(ErnieLayoutPretrainingHeads, self).__init__()
        self.predictions = ErnieLayoutPredictionHead(hidden_size, vocab_size, activation, embedding_weights)

    def forward(self, sequence_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        return prediction_scores


class ErnieLayoutForPretraining(ErnieLayoutPretrainedModel):

    def __init__(self, ernie_layout, config):
        super(ErnieLayoutForPretraining, self).__init__(config)
        self.ernie_layout = ernie_layout
        self.cls = ErnieLayoutPretrainingHeads(self.ernie_layout.config.hidden_size,
                                               self.ernie_layout.config.vocab_size,
                                               self.ernie_layout.config.hidden_act,
                                               embedding_weights=self.ernie_layout.embeddings.word_embeddings.weight)

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        self.ernie_layout.resize_position_embeddings(new_num_position_embeddings)

    def forward(self,
                input_ids=None,
                bbox=None,
                image=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                masked_positions=None):
        outputs = self.ernie_layout(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output, masked_positions)
        return prediction_scores


class ErnieLayoutForTokenClassification(ErnieLayoutPretrainedModel):

    def __init__(self, config):
        super(ErnieLayoutForTokenClassification, self).__init__(config)
        self.num_classes = config.num_classes
        self.ernie_layout = ErnieLayoutModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        self.init_weights()

    def get_input_embeddings(self):
        return self.ernie_layout.embeddings.word_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        self.ernie_layout.resize_position_embeddings(
            new_num_position_embeddings)

    def forward(
            self,
            input_ids=None,
            bbox=None,
            pixel_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None,
    ):
        outputs = self.ernie_layout(
            input_ids=input_ids,
            bbox=bbox,
            image=pixel_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        seq_length = input_ids.size()[1]
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_classes)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )


class ErnieLayoutForQuestionAnswering(ErnieLayoutPretrainedModel):

    def __init__(self, config):
        super(ErnieLayoutForQuestionAnswering, self).__init__(config)
        self.ernie_layout = ErnieLayoutModel(config)
        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_classes)
        self.init_weights()

    def get_input_embeddings(self):
        return self.ernie_layout.embeddings.word_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        self.ernie_layout.resize_position_embeddings(new_num_position_embeddings)

    def forward(self,
                input_ids=None,
                bbox=None,
                images=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                start_positions=None,
                end_positions=None,
                **kwargs):
        seq_length = input_ids.size(1)
        attention_mask = attention_mask[:, :seq_length]
        outputs = self.ernie_layout(
            input_ids=input_ids,
            bbox=bbox,
            image=images,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        seq_length = input_ids.size()[1]
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)

        if token_type_ids is not None:
            span_mask = -token_type_ids * 1e8
        else:
            span_mask = 0

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1) + span_mask
        end_logits = end_logits.squeeze(-1) + span_mask

        # outputs = (start_logits, end_logits) + outputs[2:]

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clip(0, ignored_index)
            end_positions = end_positions.clip(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits)
