# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WaveServe model unit checks (layer split + tiny CPU forward).

Omni / diffusion registry wiring lives in
``tests/entrypoints/test_resolve_waveserve_wan_config.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.waveserve_wan.pipeline_waveserve_wan import (
    HF_MODEL_ID,
    WaveServeWanPipeline,
)
from vllm_omni.diffusion.models.waveserve_wan.transformer import stage_layer_range
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.experimental.ar_diffusion.chunk_executor import (
    ARDiffusionChunkContext,
    ChunkRunSpec,
    ChunkTopology,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import NoisyKVCache, NoisyKVState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_stage_layer_range_covers_all_layers():
    covered = []
    for g in range(2):
        start, end = stage_layer_range(30, g, 2)
        covered.extend(range(start, end))
    assert covered == list(range(30))
    assert stage_layer_range(30, 0, 2) == (0, 15)
    assert stage_layer_range(30, 1, 2) == (15, 30)


def test_waveserve_tiny_forward_cpu():
    od_config = SimpleNamespace(
        ar_diffusion_stage_config={"tiny": True, "stage_parallel_size": 1, "max_history_chunks": 1},
        model_config={},
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        model=HF_MODEL_ID,
    )
    pipeline = WaveServeWanPipeline(od_config=od_config)
    spec = pipeline.ar_diffusion_noisy_kv_spec()
    cache = NoisyKVCache(
        spec,
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=1,
    )
    ctx = ARDiffusionChunkContext(
        spec=ChunkRunSpec(topology=ChunkTopology(stages=1, layer_groups=1), rank=0),
        kv=NoisyKVState(cache),
    )
    req = OmniDiffusionRequest(
        prompt="a cat",
        request_id="ws-0",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={"num_chunks": 1, "num_denoise_steps": 1, "kv_history_chunks": 1},
        ),
    )
    with pipeline.bind_ar_diffusion_chunk_context(ctx):
        outputs = pipeline.forward(req)
    assert len(outputs) == 1
    assert outputs[0].output is not None
    assert not ctx.inflight
