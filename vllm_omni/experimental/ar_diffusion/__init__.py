# SPDX-License-Identifier: Apache-2.0
"""AR-Diffusion Engine (AR-Diffusion).

The AR-Diffusion engine: a ``DiffusionEngine`` subclass that adds engine-level KV
cache management for autoregressive / chunked diffusion models. Selected via
``OmniDiffusionConfig.engine_backend = "ar_diffusion"``.
"""

from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionCrossAttentionKVSpec,
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
    SupportsARDiffusionPipeline,
    SupportsARDiffusionStagePipeline,
    SupportsARDiffusionWarmup,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    AR_DIFFUSION_TICK_KEY,
    ARDiffusionChunkMetadata,
    ARDiffusionControlInput,
    ARDiffusionTickRequest,
)

__all__ = [
    "ARDiffusionKVBranchSpec",
    "ARDiffusionCrossAttentionKVSpec",
    "ARDiffusionEngine",
    "ARDiffusionOmniTickConsumer",
    "ARDiffusionSession",
    "ARDiffusionSessionManager",
    "ARDiffusionWorkerLifecycle",
    "AR_DIFFUSION_TICK_KEY",
    "ARDiffusionChunkMetadata",
    "ARDiffusionControlInput",
    "ARDiffusionTickRequest",
    "ARDiffusionKVCacheSpec",
    "SupportsARDiffusionPipeline",
    "SupportsARDiffusionStagePipeline",
    "SupportsARDiffusionWarmup",
]


def __getattr__(name: str):
    if name == "ARDiffusionEngine":
        from vllm_omni.experimental.ar_diffusion.engine import ARDiffusionEngine

        return ARDiffusionEngine
    if name == "ARDiffusionOmniTickConsumer":
        from vllm_omni.experimental.ar_diffusion.consumer import ARDiffusionOmniTickConsumer

        return ARDiffusionOmniTickConsumer
    if name in {
        "ARDiffusionSession",
        "ARDiffusionSessionManager",
        "ARDiffusionWorkerLifecycle",
    }:
        from vllm_omni.experimental.ar_diffusion import session

        return getattr(session, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
