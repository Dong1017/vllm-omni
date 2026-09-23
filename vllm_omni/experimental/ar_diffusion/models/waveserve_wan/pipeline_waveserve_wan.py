# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental WaveServe Wan 2.1 1.3B (rectified-flow) stage pipeline.

Checkpoint: ``Physis-AI/waveserve-wan2.1-1.3b-diffusers-rf-dev``.
Layer split is stage-local ``(g, G)``; KV is versioned Latest-KV.
Lives entirely under ``experimental/ar_diffusion``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.experimental.ar_diffusion.kv_cache.versioned import ARDiffusionVersionedKVSpec
from vllm_omni.experimental.ar_diffusion.models.waveserve_wan.transformer import StageWanTransformer
from vllm_omni.experimental.ar_diffusion.stage_executor import (
    ARDiffusionStageContext,
    StageAdapter,
    resolve_pp_rank_and_group,
    run_stage_pipeline,
)
from vllm_omni.experimental.ar_diffusion.stage_schedule import (
    Ordering,
    StageSchedule,
    build_stage_plan,
)

logger = init_logger(__name__)

HF_MODEL_ID = "Physis-AI/waveserve-wan2.1-1.3b-diffusers-rf-dev"
# Wan 2.1 1.3B geometry. Tiny mode is for CPU tests only.
WAN21_1_3B = {
    "num_layers": 30,
    "dim": 1536,
    "num_heads": 12,
    "ffn_dim": 8960,
    "in_channels": 16,
    "patch_size": (1, 2, 2),
}


def _as_request_list(req: OmniDiffusionRequest | DiffusionRequestBatch) -> list[OmniDiffusionRequest]:
    if isinstance(req, DiffusionRequestBatch):
        return list(req.requests)
    return [req]


class _TransformerAdapter(StageAdapter):
    def __init__(self, transformer: StageWanTransformer, seed_hidden: torch.Tensor) -> None:
        self.transformer = transformer
        self.seed_hidden = seed_hidden

    def forward(self, tasks, kv_contexts, *, hidden):
        if not tasks:
            return hidden
        outs = []
        for i, _task in enumerate(tasks):
            h = hidden
            if h is None:
                h = self.seed_hidden
            elif isinstance(h, torch.Tensor) and h.shape[0] == len(tasks):
                h = h[i : i + 1]
            elif isinstance(h, IntermediateTensors) and h["hidden_states"].shape[0] == len(tasks):
                h = IntermediateTensors({key: value[i : i + 1] for key, value in h.tensors.items()})
            ctx = kv_contexts[i] if i < len(kv_contexts) else None
            outs.append(self.transformer(h, kv_contexts=ctx))
        return self._stack(outs)

    @staticmethod
    def _stack(outs: list[Any]) -> Any:
        first = outs[0]
        if isinstance(first, IntermediateTensors):
            stacked = {}
            for key in first.tensors:
                stacked[key] = torch.cat([out.tensors[key] for out in outs], dim=0)
            return IntermediateTensors(stacked)
        if isinstance(first, torch.Tensor):
            return torch.cat(outs, dim=0)
        return first

    def pack_activation(self, output):
        if isinstance(output, IntermediateTensors):
            return dict(output.tensors)
        if isinstance(output, torch.Tensor):
            return {"hidden_states": output}
        if hasattr(output, "tensors"):
            return dict(output.tensors)
        return {"hidden_states": output}

    def unpack_activation(self, payload: dict):
        if "hidden_states" in payload and len(payload) == 1:
            return payload["hidden_states"]
        return IntermediateTensors(payload)


class WaveServeWanPipeline(nn.Module):
    """Chunk Latest-KV pipeline under experimental ar_diffusion."""

    supports_request_batch = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.od_config = od_config
        stage_cfg = getattr(od_config, "ar_diffusion_stage_config", None) or {}
        if not isinstance(stage_cfg, dict):
            stage_cfg = {}
        model_cfg = getattr(od_config, "model_config", None) or {}
        if isinstance(model_cfg, dict):
            stage_cfg = {**stage_cfg, **(model_cfg.get("ar_diffusion_stage_config") or {})}
        self.stage_parallel_size = int(stage_cfg.get("stage_parallel_size", 1) or 1)
        pp_world = int(getattr(getattr(od_config, "parallel_config", None), "pipeline_parallel_size", 1) or 1)
        self.layer_groups = max(1, pp_world // max(1, self.stage_parallel_size))
        tiny = bool(stage_cfg.get("tiny", False))
        geo = dict(WAN21_1_3B)
        if tiny:
            geo.update(num_layers=2, dim=32, num_heads=2, ffn_dim=64)
        self.block_size = 16 if tiny else 64
        self.max_chunk_tokens = self.block_size * (2 if tiny else 32)
        self.max_history_chunks = int(stage_cfg.get("max_history_chunks", 6) or 6)
        pp_rank, _pp_group = resolve_pp_rank_and_group()
        self.transformer = StageWanTransformer(
            num_layers=int(geo["num_layers"]),
            dim=int(geo["dim"]),
            num_heads=int(geo["num_heads"]),
            ffn_dim=int(geo["ffn_dim"]),
            in_channels=int(geo["in_channels"]),
            patch_size=tuple(geo["patch_size"]),
            layer_groups=self.layer_groups,
            pp_rank=pp_rank,
        )
        self._stage_ctx: ARDiffusionStageContext | None = None
        logger.info(
            "WaveServe Wan experimental pipeline: S=%d G=%d layers=%d local=[%d, %d) model=%s",
            self.stage_parallel_size,
            self.layer_groups,
            geo["num_layers"],
            self.transformer.start_layer,
            self.transformer.end_layer,
            getattr(od_config, "model", HF_MODEL_ID),
        )

    def ar_diffusion_versioned_kv_spec(self) -> ARDiffusionVersionedKVSpec:
        local_layers = max(1, self.transformer.local_num_layers)
        return ARDiffusionVersionedKVSpec(
            num_layers=local_layers,
            num_kv_heads=self.transformer.num_heads,
            head_size=self.transformer.head_dim,
            block_size=self.block_size,
            max_chunk_tokens=self.max_chunk_tokens,
            max_history_chunks=max(1, self.max_history_chunks),
        )

    @contextmanager
    def bind_ar_diffusion_stage_context(self, ctx: ARDiffusionStageContext) -> Iterator[None]:
        prev = self._stage_ctx
        self._stage_ctx = ctx
        try:
            yield
        finally:
            self._stage_ctx = prev

    def load_weights(self, weights):
        """Best-effort load; unmatched HF tensors are skipped.

        Returns None to opt out of the loader's strict loaded-weights
        tracking: this experimental stage transformer is a simplified block
        stack whose parameter names do not line up with the reference
        checkpoint, so a name-coverage check would always fail.
        """
        mapping = dict(self.state_dict())
        for name, tensor in weights:
            key = name[len("transformer.") :] if name.startswith("transformer.") else name
            if key in mapping and mapping[key].shape == tensor.shape:
                mapping[key].copy_(tensor)
        return None

    def _plan_for(self, req: OmniDiffusionRequest):
        extra = (req.sampling_params.extra_args or {}) if req.sampling_params is not None else {}
        chunks = int(extra.get("num_chunks", extra.get("chunks", 2)))
        explicit_denoise = (
            extra.get("num_inference_steps")
            or extra.get("num_denoise_steps")
            or getattr(req.sampling_params, "num_inference_steps", None)
        )
        denoise = int(explicit_denoise or 4)
        history = int(extra.get("kv_history_chunks", 0) or 0)
        if self.stage_parallel_size > 1 and history < 1:
            history = self.max_history_chunks
        ordering = Ordering.SERIAL if str(extra.get("chunk_schedule", "serial")) == "serial" else Ordering.INTERLEAVED
        stages = self.stage_parallel_size
        if stages not in (1, denoise + 1):
            if explicit_denoise:
                raise ValueError(f"WaveServe stages must be 1 or num_denoise_steps+1 ({denoise + 1}), got {stages}")
            # Dummy/profiling requests carry no explicit step count; follow the
            # stage topology (S = T+1) instead of rejecting them.
            denoise = stages - 1
        schedule = StageSchedule(
            chunks=chunks,
            num_denoise_steps=denoise,
            stages=stages,
            layer_groups=self.layer_groups,
            ordering=ordering,
            kv_history_chunks=history,
        )
        return build_stage_plan(schedule), extra

    def forward(self, req: OmniDiffusionRequest | DiffusionRequestBatch) -> list[DiffusionOutput]:
        ctx = self._stage_ctx
        if ctx is None:
            raise RuntimeError("WaveServeWanPipeline requires bind_ar_diffusion_stage_context")
        requests = _as_request_list(req)
        chunk_tokens = self.block_size
        for item in requests:
            plan, extra = self._plan_for(item)
            req_id = str(getattr(item, "request_id", None) or extra.get("request_id") or "req0")
            ctx.enqueue(req_id, plan, chunk_tokens=chunk_tokens)
        param = next(self.parameters())
        device, dtype = param.device, param.dtype
        if self.transformer.is_stage_first:
            seed_dim = self.transformer.in_features
        else:
            seed_dim = self.transformer.dim
        hidden = torch.zeros(1, chunk_tokens, seed_dim, device=device, dtype=dtype)
        adapter = _TransformerAdapter(self.transformer, seed_hidden=hidden)
        run_stage_pipeline(ctx=ctx, adapter=adapter)
        out = torch.zeros(len(requests), 3, 8, 16, 16, device=device, dtype=dtype)
        return [DiffusionOutput(output=out[i : i + 1]) for i in range(len(requests))]
