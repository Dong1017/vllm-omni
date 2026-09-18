# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Layer 3: asynchronous KV transfer between PP stages.

The experiment branches transfer KV implicitly today: activations ride
the per-slot tensor-dict P2P alongside the KV producer's forward, and a
synchronous ``Work.wait()`` guarantees availability before the next
slot. This connector makes KV transfer explicit and asynchronous:

- ``publish_async`` returns immediately; the payload leaves on a
  transfer stream while the producer continues computing.
- ``fetch_async`` returns a Future the consumer awaits right before
  attention, letting the transfer overlap the consumer's earlier work.

Correctness contract (mirrors the synchronous design it replaces):

1. A version's bytes are immutable once published.
2. The manager (layer 2) may only retire a version after every
   outstanding fetch future for it has completed -- refcounting covers
   in-flight fetches.
3. Awaits happen on the consumer's compute stream; the connector is
   responsible for stream-ordering the delivered tensor onto that
   stream (record/wait event pair), never a host-side synchronize.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vllm_omni.experimental.ar_diffusion.noisy_pp.kv_manager import KVTensor, VersionRef


class KVTensorFuture(Protocol):
    """Awaitable KV delivery resolved on the consumer's compute stream."""

    def ready(self) -> bool: ...

    def wait(self) -> None:
        """Stream-ordered block until the tensor is usable on the current
        stream. Must not synchronize the host beyond the event wait."""
        ...

    def result(self) -> KVTensor: ...


class NoisyKVConnector(Protocol):
    """Transport for one PP group's KV traffic."""

    def publish_async(self, ref: VersionRef, tensor: KVTensor, dst: int) -> KVTensorFuture:
        """Hand a published version to the transport toward ``dst``."""
        ...

    def fetch_async(self, ref: VersionRef, src: int) -> KVTensorFuture:
        """Request a remote version; await the returned future before use."""
        ...


class LocalNoisyKVConnector:
    """Single-process connector: versions are already resident, futures
    resolve immediately. Used for world == 1 and for CPU tests."""

    def publish_async(self, ref, tensor, dst):
        if dst != 0:
            raise ValueError("local connector has no remote destination")
        return _ImmediateFuture(tensor)

    def fetch_async(self, ref, src):
        if src != 0:
            raise ValueError("local connector has no remote source")
        raise NotImplementedError("binding to NoisyKVManager storage pending")


class _ImmediateFuture:
    def __init__(self, tensor) -> None:
        self._tensor = tensor

    def ready(self) -> bool:
        return True

    def wait(self) -> None:
        return None

    def result(self):
        return self._tensor
