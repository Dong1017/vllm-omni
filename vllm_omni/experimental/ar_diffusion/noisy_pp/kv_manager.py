# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Layer 2: noisy-PP KV lifecycle over the AR-Diffusion KV stack.

Clean and noisy versions live in separate named KV branches of the
``ar_diffusion`` paged cache (``ARDiffusionKVBranchSpec``): clean passes
publish long-lived anchors with window eviction; denoise steps publish
short-lived versions that retire when their consumers finish. The index
that the two experiment branches hand-maintained (per-rank dicts keyed by
``(task, producer)``) becomes this manager's private implementation; the
rest of the stack only sees :class:`VersionRef` handles.

Producer identity (which tower published a version) is orthogonal to the
clean/noisy split and preserved in every handle -- the dual-tower contract
from the noisy-chunk-pp review (#2) survives the abstraction.

This module ships types and the manager contract first; storage binding
to ``ar_diffusion.kv_cache`` lands with the wan adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

VersionKind = Literal["latest", "clean"]


@dataclass(frozen=True)
class VersionRef:
    """Handle to one published KV version. Opaque outside this module."""

    chunk: int
    kind: Literal["clean", "noisy"]
    producer: str
    revision: int  # monotonic publish counter; distinguishes re-publishes


class KVTensor(Protocol):
    """Model-shaped K/V payload; the wan adapter defines the concrete form."""

    def num_bytes(self) -> int: ...


@dataclass
class KVStats:
    live_versions: int = 0
    peak_live_versions: int = 0
    peak_retained_bytes: int = 0
    retired_versions: int = 0


class NoisyKVManager:
    """Per-rank KV lifecycle for one request session.

    Concurrency contract: single-threaded per PP rank, same as the
    experiment branches. Cross-rank coordination happens through the
    plan (layer 1) and the connector (layer 3), not here.
    """

    def __init__(self, *, producers: tuple[str, ...], history_chunks: int) -> None:
        if not producers:
            raise ValueError("noisy KV requires at least one producer identity")
        if history_chunks < 1:
            raise ValueError("noisy KV requires a positive history window")
        self._producers = producers
        self._history_chunks = history_chunks
        self._refcount: dict[VersionRef, int] = {}
        self._stats = KVStats()

    def plan_consumers(self, sources: Sequence[VersionRef]) -> None:
        """Register the frozen consumption plan for this rank.

        Called once with the resolved source list before the first slot:
        every referenced version's refcount is incremented; publish-time
        retirement fires when a version's count is zero.
        """
        raise NotImplementedError("storage binding pending wan adapter")

    def publish(self, task_chunk: int, producer: str, kind: Literal["clean", "noisy"], kv: KVTensor) -> VersionRef:
        """Store one version and return its handle. Duplicate publishes for
        the same (chunk, producer, kind) raise."""
        raise NotImplementedError("storage binding pending wan adapter")

    def resolve_sources(self, task_chunk: int, policy: VersionKind, producer: str) -> tuple[VersionRef, ...]:
        """Resolve the frozen plan to concrete handles for one reading task.

        ``latest`` resolves to the newest noisy version of each window
        chunk at call time; ``clean`` resolves to the window's clean
        versions. Missing versions raise -- the contract is never to
        silently shorten history (review #2).
        """
        raise NotImplementedError("storage binding pending wan adapter")

    def release(self, ref: VersionRef) -> None:
        """Drop one consumer reference; retire at zero. Retiring a version
        with outstanding fetch futures is the connector's contract breach."""
        raise NotImplementedError("storage binding pending wan adapter")

    def stats(self) -> KVStats:
        return self._stats
