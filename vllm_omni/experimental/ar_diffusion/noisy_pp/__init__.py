# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Noisy chunk step-wise PP as a composable strategy over AR-Diffusion.

Three layers, each independently testable and evolvable (see README.md):

- ``scheduler``: pure-function slot planning (USP2 slot model + noisy
  dependency rules).
- ``kv_manager``: clean/noisy KV lifecycle over the AR-Diffusion paged
  KV stack, producer-aware.
- ``kv_connector``: asynchronous cross-stage KV transport.

Both prior experiment branches (noisy-chunk-pp-layer-step on Dong1017 and
ShengDev forks) are subsumed here; their production paths stay frozen
until this abstraction reaches parity on the 2xH200 benchmark matrix.
"""

from vllm_omni.experimental.ar_diffusion.noisy_pp.scheduler import (
    CLEAN_PASS,
    NoisyPPSlot,
    NoisyPPTask,
    SourcePolicy,
    plan_noisy_pp,
)

__all__ = [
    "CLEAN_PASS",
    "NoisyPPSlot",
    "NoisyPPTask",
    "SourcePolicy",
    "plan_noisy_pp",
]
