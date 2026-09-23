# SPDX-License-Identifier: Apache-2.0
"""AR-Diffusion engine-level KV cache helpers.

Thin glue over vLLM's paged KV stack (``KVCacheManager`` / ``BlockPool`` /
``SlidingWindowManager``), used by the AR-Diffusion Engine to manage KV for
AR-diffusion models.

Layout: ``config`` (public knob) · ``paged`` (engine-generic paging mechanics +
chunk-window eviction spec) · ``manager`` (the ARDiffusionKVCache orchestrator + its
request adapter / pool builders) · ``state`` (the model-facing ARDiffusionKVState bridge).
"""

from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.paged import (
    ChunkWindowManager,
    ChunkWindowSpec,
    allocate_kv_pool_with_views,
    chunk_slot_mapping,
    compute_slot_mapping,
    pool_write_chunk,
    resident_block_ids,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
    ARDiffusionPagedForwardContext,
    ARDiffusionPagedLayerContext,
    ARDiffusionPagedLayerInputs,
    ar_diffusion_paged_attention,
    paged_write_attn,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.versioned import (
    ARDiffusionVersionedKVSpec,
    VersionedKVCache,
    VersionedKVState,
)

__all__ = [
    "ARDiffusionKVCache",
    "ARDiffusionKVConfig",
    "ARDiffusionPagedForwardContext",
    "ARDiffusionPagedLayerContext",
    "ARDiffusionPagedLayerInputs",
    "ARDiffusionRequestAdapter",
    "ARDiffusionVersionedKVSpec",
    "ChunkWindowManager",
    "ChunkWindowSpec",
    "VersionedKVCache",
    "VersionedKVState",
    "allocate_kv_pool_with_views",
    "ar_diffusion_paged_attention",
    "build_kv_manager",
    "chunk_slot_mapping",
    "compute_num_blocks",
    "compute_slot_mapping",
    "paged_write_attn",
    "pool_write_chunk",
    "resident_block_ids",
]


def __getattr__(name: str):
    if name in {
        "ARDiffusionKVCache",
        "ARDiffusionRequestAdapter",
        "build_kv_manager",
        "compute_num_blocks",
    }:
        from vllm_omni.experimental.ar_diffusion.kv_cache import manager

        return getattr(manager, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
