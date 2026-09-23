# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-local layer-split Wan-style transformer for WaveServe experimental serving."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors

from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import paged_write_attn


def stage_layer_range(num_layers: int, group: int, groups: int) -> tuple[int, int]:
    """Even split of ``num_layers`` across ``groups`` (stage-local G, not pp_world)."""
    if groups < 1:
        raise ValueError("groups must be positive")
    if not 0 <= group < groups:
        raise ValueError(f"group {group} out of range for {groups}")
    return (num_layers * group) // groups, (num_layers * (group + 1)) // groups


class StageWanSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)
        self.scale = head_dim**-0.5

    def forward(self, hidden: torch.Tensor, kv_ctx: Any | None) -> torch.Tensor:
        b, s, _ = hidden.shape
        qkv = self.qkv(hidden).view(b, s, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        if kv_ctx is not None:
            inputs = kv_ctx.to_layer_inputs() if hasattr(kv_ctx, "to_layer_inputs") else kv_ctx
            outs = [
                paged_write_attn(inputs, query[i], key[i], value[i], None, None, self.scale)
                for i in range(b)
            ]
            attn = torch.stack(outs, dim=0)
        else:
            scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * self.scale
            probs = torch.softmax(scores, dim=-1).to(value.dtype)
            attn = torch.einsum("bhqk,bkhd->bqhd", probs, value)
        return self.o_proj(attn.flatten(-2))


class StageWanBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = StageWanSelfAttention(dim, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))

    def forward(self, hidden: torch.Tensor, kv_ctx: Any | None) -> torch.Tensor:
        hidden = hidden + self.attn(self.norm1(hidden), kv_ctx)
        return hidden + self.ffn(self.norm2(hidden))


class StageWanTransformer(nn.Module):
    """Wan-like DiT with stage-local PP slices and paged Latest-KV attention."""

    def __init__(
        self,
        *,
        num_layers: int = 30,
        dim: int = 1536,
        num_heads: int = 12,
        ffn_dim: int = 8960,
        in_channels: int = 16,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        layer_groups: int = 1,
        pp_rank: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.patch_size = patch_size
        self.in_features = in_channels * math.prod(patch_size)
        self.layer_groups = layer_groups
        self.pp_rank = pp_rank
        g = 0 if layer_groups == 1 else pp_rank % layer_groups
        self.start_layer, self.end_layer = stage_layer_range(num_layers, g, layer_groups)
        self.is_stage_first = g == 0
        self.is_stage_last = g == layer_groups - 1
        if self.is_stage_first:
            self.patch_embed = nn.Linear(self.in_features, dim)
        else:
            self.patch_embed = PPMissingLayer()
        blocks: list[nn.Module] = []
        for idx in range(num_layers):
            if self.start_layer <= idx < self.end_layer:
                blocks.append(StageWanBlock(dim, num_heads, self.head_dim, ffn_dim))
            else:
                blocks.append(PPMissingLayer())
        self.blocks = nn.ModuleList(blocks)
        if self.is_stage_last:
            self.proj_out = nn.Linear(dim, self.in_features)
        else:
            self.proj_out = PPMissingLayer()

    @property
    def local_num_layers(self) -> int:
        return max(0, self.end_layer - self.start_layer)

    def _resolve_hidden(
        self,
        hidden_states: torch.Tensor | IntermediateTensors | None,
        intermediate_tensors: IntermediateTensors | None,
    ) -> torch.Tensor:
        if isinstance(hidden_states, IntermediateTensors):
            intermediate_tensors = hidden_states
            hidden_states = None
        if intermediate_tensors is not None:
            return intermediate_tensors["hidden_states"]
        if hidden_states is None:
            raise RuntimeError("stage transformer received no hidden states")
        if self.is_stage_first and hidden_states.size(-1) == self.in_features:
            return self.patch_embed(hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor | IntermediateTensors | None = None,
        kv_contexts: list[Any] | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden = self._resolve_hidden(hidden_states, intermediate_tensors)
        local_count = self.local_num_layers
        for idx in range(self.start_layer, self.end_layer):
            ctx = None
            if kv_contexts is not None:
                ctx = kv_contexts[idx - self.start_layer] if len(kv_contexts) == local_count else kv_contexts[idx]
            hidden = self.blocks[idx](hidden, ctx)
        if self.is_stage_last:
            return self.proj_out(hidden)
        return IntermediateTensors({"hidden_states": hidden})
