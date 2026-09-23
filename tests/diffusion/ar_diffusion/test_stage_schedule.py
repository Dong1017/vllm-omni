# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for experimental stage timetable and Latest-KV plan."""

from __future__ import annotations

import pytest

from vllm_omni.experimental.ar_diffusion.stage_schedule import (
    Ordering,
    StageSchedule,
    build_stage_plan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _s1(chunks, world, ordering, *, kv=False, steps=3) -> StageSchedule:
    return StageSchedule(
        chunks=chunks,
        num_denoise_steps=steps,
        stages=1,
        layer_groups=world,
        ordering=ordering,
        kv_history_chunks=6 if kv else 0,
    )


@pytest.mark.parametrize("world", [1, 2, 3, 4])
@pytest.mark.parametrize("chunks", [1, 2, 3, 6])
@pytest.mark.parametrize("schedule", ["serial", "stepwise"])
def test_s1_builds_without_idle_slot(world, chunks, schedule):
    ordering = Ordering.SERIAL if schedule == "serial" else Ordering.INTERLEAVED
    plan = build_stage_plan(_s1(chunks, world, ordering))
    assert plan.num_slots >= chunks
    for t in range(plan.num_slots):
        assert any(plan.task(t, rank) is not None for rank in range(world))


@pytest.mark.parametrize("world", [1, 2, 3, 4])
@pytest.mark.parametrize("chunks", [1, 2, 3, 6])
def test_s1_serial_kv_sources_are_clean(world, chunks):
    plan = build_stage_plan(_s1(chunks, world, Ordering.SERIAL, kv=True))
    for t in range(plan.num_slots):
        for rank in range(world):
            task = plan.task(t, rank)
            if task is None:
                continue
            for src in plan.sources(task, rank):
                assert src.version[1] == plan.schedule.num_denoise_steps
                assert src.owner == rank


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("chunks", [1, 2, 4])
def test_s1_interleaved_kv_uses_latest_completed(world, chunks):
    plan = build_stage_plan(_s1(chunks, world, Ordering.INTERLEAVED, kv=True))
    for t in range(plan.num_slots):
        for rank in range(world):
            task = plan.task(t, rank)
            if task is None:
                continue
            g = rank
            for src in plan.sources(task, rank):
                assert plan.completion_slot(src.version, g) < t
                assert src.owner == rank


def test_s1_kv_transfers_empty():
    plan = build_stage_plan(_s1(4, 2, Ordering.INTERLEAVED, kv=True))
    for t in range(plan.num_slots):
        assert plan.transfers(t) == ()


def test_vertical_diagonal_example():
    schedule = StageSchedule(
        chunks=4,
        num_denoise_steps=2,
        stages=3,
        layer_groups=2,
        ordering=Ordering.INTERLEAVED,
        kv_history_chunks=6,
    )
    plan = build_stage_plan(schedule)
    assert plan.task(0, 0) == (0, 0)
    assert plan.task(3, 0) == (3, 0)
    assert plan.task(2, 2) == (0, 1)
    srcs = plan.sources((3, 0), 0)
    versions = {src.version for src in srcs}
    assert (2, 0) in versions
    assert (1, 0) in versions
    assert (0, 1) in versions
    owners = {src.version: src.owner for src in srcs}
    assert owners[(0, 1)] == 2
    xfers = plan.transfers(2)
    assert any(x.version == (0, 1) and x.src == 2 and x.dst == 0 for x in xfers)
    clean_srcs = plan.sources((3, 2), 4)
    assert {s.version for s in clean_srcs} == {(2, 2), (1, 2), (0, 2)}
    assert all(s.owner == 4 for s in clean_srcs)


def test_rejects_intermediate_s():
    with pytest.raises(ValueError, match="stages must be 1"):
        StageSchedule(
            chunks=4,
            num_denoise_steps=4,
            stages=2,
            layer_groups=1,
            ordering=Ordering.SERIAL,
            kv_history_chunks=4,
        )


def test_i2_sources_strictly_earlier():
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
    for t in range(plan.num_slots):
        for rank in range(plan.world):
            task = plan.task(t, rank)
            if task is None:
                continue
            g = rank % 2
            for src in plan.sources(task, rank):
                assert plan.completion_slot(src.version, g) < t
