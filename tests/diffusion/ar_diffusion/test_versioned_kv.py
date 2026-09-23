# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for versioned KV last-use eviction (k=1 overlap)."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.kv_cache.versioned import (
    ARDiffusionVersionedKVSpec,
    VersionedKVCache,
    VersionedKVState,
)
from vllm_omni.experimental.ar_diffusion.stage_schedule import (
    Inflight,
    Ordering,
    StagePlan,
    StageSchedule,
    build_stage_plan,
    incoming_transfers,
    union_wait_ready,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _cache(**kwargs) -> VersionedKVCache:
    spec = ARDiffusionVersionedKVSpec(
        num_layers=2,
        num_kv_heads=2,
        head_size=4,
        block_size=4,
        max_chunk_tokens=4,
        max_history_chunks=6,
    )
    return VersionedKVCache(
        spec,
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=kwargs.get("layer_groups", 2),
        max_batch_size=kwargs.get("max_batch_size", 1),
    )


def test_capacity_uses_k1():
    cache = _cache(layer_groups=2, max_batch_size=2)
    # R * (H + 2 + G) = 2 * (6 + 2 + 2) = 20
    assert cache.capacity == 20


def test_prepare_evict_releases_last_use():
    plan = build_stage_plan(
        StageSchedule(
            chunks=4,
            num_denoise_steps=2,
            stages=3,
            layer_groups=2,
            ordering=Ordering.INTERLEAVED,
            kv_history_chunks=6,
        )
    )
    cache = _cache()
    state = VersionedKVState(cache)
    state.bind_rank(0, None)
    state.begin_request("A", plan, chunk_tokens=4, t0=0)
    inflight = (Inflight(req="A", t0=0, plan=plan),)
    state.set_inflight(inflight)
    # Slot 0 writes (0,0)
    state.prepare((("A", (0, 0)),))
    assert state.resident_versions == 1
    state.evict(0)
    # last-use of (0,0) on r0 is later than slot 0
    assert state.resident_versions == 1
    state.prepare((("A", (1, 0)),))
    state.prepare((("A", (2, 0)),))
    # after slot 2, (0,0) should still be resident until evict(2)
    before = state.resident_versions
    state.evict(2)
    assert state.resident_versions < before or (0, 0) not in [(k[1], k[2]) for k in cache.pool.keys if k[0] == "A"]


def _vertical_plan() -> StagePlan:
    return build_stage_plan(
        StageSchedule(
            chunks=4,
            num_denoise_steps=2,
            stages=3,
            layer_groups=2,
            ordering=Ordering.INTERLEAVED,
            kv_history_chunks=6,
        )
    )


def test_wait_ready_matches_incoming_transfers():
    """I9②: the awaited set is exactly slot T's inbound, read at slot T+1."""
    plan = _vertical_plan()
    inflight = (Inflight(req="A", t0=0, plan=plan),)
    for slot in range(plan.num_slots):
        incoming = sorted(incoming_transfers(plan._transfers, slot, 0), key=lambda x: x.version)
        ready = sorted(union_wait_ready(inflight, slot, 0), key=lambda item: item[1])
        assert [x.version for x in incoming] == [version for _req, version in ready]


def test_wait_ready_is_empty_without_kv_history():
    plan = build_stage_plan(
        StageSchedule(
            chunks=2,
            num_denoise_steps=1,
            stages=1,
            layer_groups=2,
            ordering=Ordering.INTERLEAVED,
            kv_history_chunks=0,
        )
    )
    inflight = (Inflight(req="A", t0=0, plan=plan),)
    for slot in range(plan.num_slots):
        assert union_wait_ready(inflight, slot, 0) == frozenset()
