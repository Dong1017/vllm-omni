# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Layer 2: noisy-PP KV policy over the AR-Diffusion KV stack.

This is a *policy* layer, not a storage layer. All storage, paging, slot
management and tensor pools are owned by ``ar_diffusion.kv_cache``
(``ARDiffusionKVCache`` / ``ARDiffusionKVState``, themselves thin over the
vLLM paged KV stack). This module adds only what noisy chunk-PP needs on
top:

- **clean/noisy branch split**: clean passes (t=0 context refresh) commit
  into the ``"clean"`` KV branch -- long-lived anchors with window
  eviction; denoise steps commit into ``"noisy"`` -- short-lived versions
  retired when their consumers finish. Both branches are ordinary named
  ``ARDiffusionKVBranchSpec`` entries declared by the adapter.
- **version resolution**: the two source policies from layer 1 resolve
  here. ``latest`` picks each window chunk's newest committed version at
  read time; ``clean`` pins the window's clean-pass versions. Missing
  versions raise -- history is never silently shortened (review #2).
- **refcount-driven retire**: the frozen consumer plan increments counts
  at plan time; each consumer's completion decrements. Zero means the
  underlying chunk slots are freed through the base stack.

Producer identity (``"high"``/``"low"`` towers or ``"single"``) is
orthogonal to the clean/noisy split and travels inside every ref: the
dual-tower contract from review #2 survives the abstraction because the
adapter declares one KV branch set *per producer*.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from vllm_omni.experimental.ar_diffusion.kv_cache.manager import ARDiffusionKVCache
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

#: Branch-name prefix conventions. The adapter declares the full spec; the
#: manager composes branch names as ``f"{kind}_{producer}"``.
CLEAN_BRANCH_PREFIX = "clean"
NOISY_BRANCH_PREFIX = "noisy"


# -- Handles ------------------------------------------------------------------


@dataclass(frozen=True)
class VersionRef:
    """Handle to one committed KV version, opaque outside this module.

    ``chunk`` and ``producer`` address the logical version; ``committed_at``
    orders versions of the same (chunk, producer, kind) so ``latest``
    resolution is deterministic.
    """

    chunk: int
    kind: str  # "clean" | "noisy"
    producer: str
    committed_at: int


@dataclass
class KVStats:
    live_versions: int = 0
    peak_live_versions: int = 0
    retired_versions: int = 0


# -- Manager ------------------------------------------------------------------


class NoisyKVManager:
    """Per-rank noisy-KV policy over one request's ``ARDiffusionKVState``.

    Single-threaded per PP rank, same as the experiment branches; cross-rank
    coordination happens through the plan (layer 1) and connector (layer 3).
    """

    def __init__(
        self,
        *,
        state: ARDiffusionKVState,
        cache: ARDiffusionKVCache,
        producers: tuple[str, ...],
        history_chunks: int,
    ) -> None:
        if not producers:
            raise ValueError("noisy KV requires at least one producer identity")
        if history_chunks < 1:
            raise ValueError("noisy KV requires a positive history window")
        self._state = state
        self._cache = cache
        self._producers = producers
        self._history_chunks = history_chunks
        # (chunk, producer, kind) -> refcount of remaining consumers.
        self._refcount: dict[tuple[int, str, str], int] = {}
        # (chunk, producer, kind) -> committed version, ordered by arrival.
        self._versions: dict[tuple[int, str, str], VersionRef] = {}
        self._commit_counter = 0
        self._stats = KVStats()

    # -- planning -------------------------------------------------------------

    def plan_consumers(self, sources: Sequence[VersionRef]) -> None:
        """Register the frozen consumption plan for this rank.

        Called once before the first slot: every referenced version's
        refcount is incremented, so a version retires exactly when its last
        consumer completes. Sheng's last-use eviction is derivable from the
        same plan; refcount keeps the general form (review consensus).
        """
        for ref in sources:
            key = (ref.chunk, ref.producer, ref.kind)
            self._refcount[key] = self._refcount.get(key, 0) + 1

    # -- publish / resolve ----------------------------------------------------

    def branch_for(self, producer: str, kind: str) -> str:
        """KV branch name for one (producer, kind) pair."""
        prefix = CLEAN_BRANCH_PREFIX if kind == "clean" else NOISY_BRANCH_PREFIX
        return f"{prefix}_{producer}"

    def publish(
        self,
        *,
        chunk: int,
        producer: str,
        kind: str,
    ) -> VersionRef:
        """Record one committed version on its (producer, kind) branch.

        Storage writes happen in the adapter's forward via
        ``ARDiffusionKVState.prepare_paged_context`` /
        ``commit_paged_context``; by the time this is called the branch's
        chunk is committed. Duplicate publishes raise.
        """
        if producer not in self._producers:
            raise ValueError(f"unknown producer {producer!r}; expected {self._producers}")
        key = (chunk, producer, kind)
        if key in self._versions:
            raise RuntimeError(f"duplicate KV publish for chunk {chunk}, producer {producer}, kind {kind}")
        self._commit_counter += 1
        ref = VersionRef(chunk=chunk, kind=kind, producer=producer, committed_at=self._commit_counter)
        self._versions[key] = ref
        self._stats.live_versions += 1
        self._stats.peak_live_versions = max(self._stats.peak_live_versions, self._stats.live_versions)
        return ref

    def resolve_sources(self, task_chunk: int, policy: str, producer: str) -> tuple[VersionRef, ...]:
        """Resolve the window's source handles for one reading task.

        ``latest``: newest committed version per window chunk **on the
        reading producer's own branches** (the dual-tower rule: a tower
        never consumes the other tower's KV). ``clean``: the window's
        clean-pass versions on the same rule.
        """
        start = max(0, task_chunk - self._history_chunks)
        refs: list[VersionRef] = []
        for chunk in range(start, task_chunk):
            kind = "clean" if policy == "clean" else "noisy"
            ref = self._versions.get((chunk, producer, kind))
            if ref is None:
                # "latest" may fall back to the clean anchor when the chunk
                # has not committed a noisy version yet (early pipeline
                # stages); "clean" has no fallback -- history never shortens.
                if policy == "latest":
                    ref = self._versions.get((chunk, producer, "clean"))
                if ref is None:
                    raise KeyError(
                        f"no {'any' if policy == 'latest' else 'clean'} version for chunk {chunk}, "
                        f"producer {producer}: history would be silently shortened"
                    )
            refs.append(ref)
        return tuple(refs)

    # -- retirement -----------------------------------------------------------

    def release(self, ref: VersionRef) -> None:
        """Drop one consumer reference; free slots at zero.

        Freeing goes through the base stack's per-branch chunk accounting;
        a version's branch chunk is shared by all readers of the same
        (chunk, producer, kind), so only the last release frees it.
        """
        key = (ref.chunk, ref.producer, ref.kind)
        remaining = self._refcount.get(key)
        if remaining is None:
            # Published but never consumed (e.g. the final chunk's versions).
            return
        remaining -= 1
        if remaining == 0:
            del self._refcount[key]
            self._versions.pop(key, None)
            self._stats.live_versions -= 1
            self._stats.retired_versions += 1
        else:
            self._refcount[key] = remaining

    def release_all_for_chunk(self, chunk: int) -> None:
        """Retire every remaining version of one chunk (end-of-life path)."""
        for key in [k for k in self._refcount if k[0] == chunk]:
            self._refcount.pop(key, None)
            self._versions.pop(key, None)
            self._stats.live_versions -= 1
            self._stats.retired_versions += 1

    def stats(self) -> KVStats:
        return self._stats


# -- Factory ------------------------------------------------------------------


def build_noisy_kv_manager(
    *,
    cache: ARDiffusionKVCache,
    session_state_factory: Callable[[str], ARDiffusionKVState],
    producers: tuple[str, ...],
    history_chunks: int,
    session_id: str,
) -> NoisyKVManager:
    """Assemble a manager whose branches cover clean/noisy x producers.

    ``session_state_factory`` is the base stack's per-branch session
    constructor; the adapter wires it to ``ARDiffusionKVCache``.
    """
    del cache, session_state_factory, session_id  # wired in the wan adapter landing
    return NoisyKVManager(producers=producers, history_chunks=history_chunks)
