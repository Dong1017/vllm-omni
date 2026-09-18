# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Layer 3: KV transfer for noisy-PP, aligned with the vLLM connector stack.

This module does **not** reinvent a transport protocol. The vLLM
``KVConnectorBase_V1`` lifecycle (``start_load_kv`` / ``wait_for_layer_load``
/ ``save_kv_layer`` / ``wait_for_save`` / ``get_finished``) already defines
async KV delivery with per-layer waits; ``vllm_omni/distributed/omni_connectors``
ships a rank-aware ZMQ transfer manager; and ``diffusion_kv/kv_connector.py``
shows the assembly pattern for diffusion-side connectors via
``KVConnectorFactory``.

What noisy-PP adds on top is a *semantics contract*, not a transport:

1. Version identity: transfers address :class:`VersionRef` handles
   ``(chunk, kind, producer)`` from layer 2, not raw layer names.
2. Consumer-side availability: ``wait_for_layer_load`` must complete before
   the reading attention consumes the window -- identical guarantee to the
   synchronous ``Work.wait()`` the experiment branches issue per slot, but
   issued per layer so transfer overlaps compute.
3. Retirement safety: the manager (layer 2) may only free a version after
   ``get_finished`` confirms all outstanding consumer-side loads drained --
   refcounts cover in-flight fetches.

Until the transport lands, ``LocalNoisyKVConnector`` covers world == 1 and
CPU tests; the ZMQ path reuses ``kv_transfer_manager`` primitives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.experimental.ar_diffusion.noisy_pp.kv_manager import NoisyKVManager, VersionRef

logger = init_logger(__name__)


class NoisyKVTransfer(Protocol):
    """Minimal surface noisy-PP needs from any transport backend.

    Implementations adapt an existing engine -- vLLM's
    ``KVConnectorBase_V1`` hooks, the omni ZMQ manager, or a future
    RDMA/Nixl backend -- to version-addressed KV. They must not manage
    lifecycle (that is layer 2) or scheduling (layer 1).
    """

    def publish(self, ref: VersionRef, tensor: Any, dst_rank: int) -> None:
        """Queue an immutable published version toward ``dst_rank``."""
        ...

    def fetch(self, ref: VersionRef, src_rank: int) -> Any:
        """Return a handle whose ``wait_for_layer_load``-style await
        completes before attention consumes the window."""
        ...

    def drained(self) -> bool:
        """True when no in-flight transfers remain (retirement gate)."""
        ...


class LocalNoisyKVConnector:
    """Single-process transport: versions are resident in-process, so
    publish/fetch are no-ops around the manager's storage. Used for
    world == 1 and CPU tests."""

    def __init__(self, manager: NoisyKVManager) -> None:
        self._manager = manager

    def publish(self, ref: VersionRef, tensor: Any, dst_rank: int) -> None:
        if dst_rank != 0:
            raise ValueError("local connector has no remote destination")

    def fetch(self, ref: VersionRef, src_rank: int) -> Any:
        if src_rank != 0:
            raise ValueError("local connector has no remote source")
        # The tensor is already in the manager's paged pool; the adapter
        # reads it through ARDiffusionKVState.get_kv_caches.
        return None

    def drained(self) -> bool:
        return True
