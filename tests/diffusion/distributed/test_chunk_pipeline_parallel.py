# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for chunk dependencies and the N-stage layer schedule."""

import pytest

from vllm_omni.diffusion.distributed.chunk_pipeline_parallel import (
    ChunkKVContext,
    evict_kv_versions,
    plan_chunk_pipeline,
    plan_clean_kv_sources,
    plan_kv_last_use,
    plan_latest_kv_sources,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("chunks", [1, 2, 3, 6])
def test_latest_kv_never_reads_same_slot_or_future_chunk(chunks):
    slots = plan_chunk_pipeline(chunks, 2, kv=True)
    sources = plan_latest_kv_sources(slots, 6)
    completed = [{}, {}]
    for tick, tasks in enumerate(slots):
        for rank, task in enumerate(tasks):
            if task is None:
                continue
            chunk, step = task
            expected = []
            for previous in range(chunk):
                versions = [s for (c, s), when in completed[rank].items() if c == previous and when < tick]
                if versions:
                    expected.append((previous, max(versions)))
            assert sources[rank][task] == tuple(expected)
            if step:
                assert (chunk, step - 1) in completed[1]
        for rank, task in enumerate(tasks):
            if task is not None:
                assert task not in completed[rank]
                completed[rank][task] = tick
    expected_tasks = {(c, s) for c in range(chunks) for s in range(4)}
    assert all(set(stage) == expected_tasks for stage in completed)


@pytest.mark.parametrize("world", [1, 2, 3, 4])
@pytest.mark.parametrize("chunks", [1, 2, 3, 6])
@pytest.mark.parametrize("num_denoise_steps", [3, 8])
def test_serial_kv_reads_only_clean_history(world, chunks, num_denoise_steps):
    slots = plan_chunk_pipeline(chunks, world, "serial", kv=True, num_denoise_steps=num_denoise_steps)
    sources = plan_clean_kv_sources(slots, 6, num_denoise_steps)
    published = [set() for _ in range(world)]
    for tasks in slots:
        for rank, task in enumerate(tasks):
            if task is None:
                continue
            chunk, _ = task
            expected = tuple((c, num_denoise_steps) for c in range(max(0, chunk - 6), chunk))
            assert sources[rank][task] == expected
            assert set(expected) <= published[rank]
            published[rank].add(task)


def test_serial_kv_history_window_drops_chunks_outside_h():
    slots = plan_chunk_pipeline(4, 2, "serial", kv=True)
    sources = plan_clean_kv_sources(slots, 1, 3)
    for rank in range(2):
        assert sources[rank][(1, 0)] == ((0, 3),)
        assert sources[rank][(2, 2)] == ((1, 3),)
        assert sources[rank][(3, 3)] == ((2, 3),)


def test_latest_kv_includes_clean_versions_after_pair_finishes():
    sources = plan_latest_kv_sources(plan_chunk_pipeline(4, 2, kv=True), 6)
    for rank in range(2):
        assert sources[rank][(1, 0)] == ((0, 0),)
        assert sources[rank][(2, 0)] == ((0, 3), (1, 3))


def test_latest_kv_eight_step_clean_version_is_denoise_count():
    sources = plan_latest_kv_sources(plan_chunk_pipeline(4, 2, kv=True, num_denoise_steps=8), 6)
    for rank in range(2):
        assert sources[rank][(1, 0)] == ((0, 0),)
        assert sources[rank][(2, 0)] == ((0, 8), (1, 8))
        expected_tasks = {(c, s) for c in range(4) for s in range(9)}
        planned = {task for tasks in plan_chunk_pipeline(4, 2, kv=True, num_denoise_steps=8) for task in tasks if task}
        assert planned == expected_tasks


def test_kv_attention_matches_explicit_history_and_keeps_versions_separate():
    import torch
    import torch.nn.functional as F

    rng = torch.Generator().manual_seed(7)
    tensors = [torch.randn(1, 5, 2, 4, generator=rng) for _ in range(7)]
    k0, v0, k1, v1, key, value, query = tensors
    layers = {}
    ChunkKVContext(layers, (0, 0), ()).append(15, k0, v0)
    ChunkKVContext(layers, (0, 3), ()).append(15, k1, v1)
    k, v = ChunkKVContext(layers, (1, 0), ((0, 0),)).append(15, key, value)
    actual = F.scaled_dot_product_attention(query.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2), torch.cat([k0, key], 1).transpose(1, 2), torch.cat([v0, value], 1).transpose(1, 2)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert layers[15][(0, 3)][0] is k1
    with pytest.raises(KeyError):
        ChunkKVContext(layers, (1, 1), ((0, 0),)).append(16, key, value)
    with pytest.raises(RuntimeError, match="Duplicate KV"):
        ChunkKVContext(layers, (1, 0), ()).append(15, key, value)


@pytest.mark.parametrize("world", [1, 2, 3, 4])
@pytest.mark.parametrize("chunks", [1, 2, 3, 6])
@pytest.mark.parametrize("history", [1, 6])
@pytest.mark.parametrize("schedule", ["serial", "stepwise"])
@pytest.mark.parametrize("num_denoise_steps", [3, 8])
def test_kv_last_use_eviction_keeps_needed_sources(world, chunks, history, schedule, num_denoise_steps):
    import torch

    execution = plan_chunk_pipeline(chunks, world, schedule, kv=True, num_denoise_steps=num_denoise_steps)
    if schedule == "serial":
        planned = plan_clean_kv_sources(execution, history, num_denoise_steps)
    else:
        planned = plan_latest_kv_sources(execution, history)
    last_use = plan_kv_last_use(execution, planned)
    dummy = torch.zeros(1, 1, 1, 1)
    for rank in range(world):
        layers = {}
        published = set()
        for slot_idx, tasks in enumerate(execution):
            task = tasks[rank]
            if task is None:
                continue
            ChunkKVContext(layers, task, planned[rank][task]).append(0, dummy, dummy)
            published.add(task)
            assert set(layers[0]) == {version for version in published if last_use[rank][version] >= slot_idx}
            evict_kv_versions(
                layers, [version for version, use in last_use[rank].items() if use == slot_idx]
            )
            remaining = set(layers.get(0, {}))
            assert remaining == {version for version in published if last_use[rank][version] > slot_idx}
            for later in execution[slot_idx + 1 :]:
                future = later[rank]
                if future is not None:
                    assert set(planned[rank][future]) & published <= remaining
        assert not layers


def test_serial_evicts_noisy_immediately_and_keeps_clean_for_later_chunks():
    stepwise = plan_chunk_pipeline(4, 2, "stepwise", kv=True)
    serial = plan_chunk_pipeline(4, 2, "serial", kv=True)
    latest = plan_latest_kv_sources(stepwise, 6)
    clean = plan_clean_kv_sources(serial, 6, 3)
    stepwise_use = plan_kv_last_use(stepwise, latest)
    serial_use = plan_kv_last_use(serial, clean)
    for rank in range(2):
        assert latest[rank][(1, 0)] == ((0, 0),)
        assert clean[rank][(1, 0)] == ((0, 3),)
        assert clean[rank][(2, 0)] == ((0, 3), (1, 3))
        stepwise_clean = next(idx for idx, tasks in enumerate(stepwise) if tasks[rank] == (0, 3))
        serial_noisy = next(idx for idx, tasks in enumerate(serial) if tasks[rank] == (0, 0))
        serial_clean = next(idx for idx, tasks in enumerate(serial) if tasks[rank] == (0, 3))
        serial_reader = next(idx for idx, tasks in enumerate(serial) if tasks[rank] == (1, 0))
        assert stepwise_use[rank][(0, 0)] < stepwise_clean
        assert serial_use[rank][(0, 0)] == serial_noisy
        assert serial_use[rank][(0, 0)] < serial_clean
        assert serial_use[rank][(0, 3)] == next(
            idx for idx, tasks in enumerate(serial) if tasks[rank] == (3, 3)
        )
        assert serial_use[rank][(0, 3)] > serial_reader
        assert stepwise_use[rank][(0, 3)] > stepwise_clean


@pytest.mark.parametrize("world", [1, 2, 3, 4])
@pytest.mark.parametrize("chunks", [1, 2, 3, 6])
@pytest.mark.parametrize("schedule", ["serial", "stepwise"])
def test_each_layer_stage_executes_once_after_dependencies(world, chunks, schedule):
    slots = plan_chunk_pipeline(chunks, world, schedule)
    expected = {(chunk, step) for chunk in range(chunks) for step in range(3)}
    executed = [{} for _ in range(world)]
    completed = {}
    for slot_idx, tasks in enumerate(slots):
        assert len(tasks) == world
        assert any(task is not None for task in tasks)
        if schedule == "serial":
            assert sum(task is not None for task in tasks) == 1
        for stage, task in enumerate(tasks):
            if task is None:
                continue
            assert task in expected
            assert task not in executed[stage]
            chunk, step = task
            if stage == 0:
                if step > 0:
                    assert completed[chunk, step - 1] < slot_idx
            else:
                assert executed[stage - 1][task] < slot_idx
            executed[stage][task] = slot_idx
        if tasks[-1] is not None:
            completed[tasks[-1]] = slot_idx
    assert set(completed) == expected
    assert all(set(stage) == expected for stage in executed)
    # N stages mean N times as many local forward calls, not N times as many DiT steps.
    assert sum(len(stage) for stage in executed) == world * chunks * 3


@pytest.mark.parametrize("chunks", [1, 2, 3, 6])
@pytest.mark.parametrize("world", [2, 3, 4])
def test_serial_and_stepwise_execute_the_same_jobs(chunks, world):
    jobs = []
    for schedule in ("serial", "stepwise"):
        slots = plan_chunk_pipeline(chunks, world, schedule)
        jobs.append([{task for tasks in slots if (task := tasks[stage]) is not None} for stage in range(world)])
    assert jobs[0] == jobs[1]


def test_two_chunk_two_stage_order():
    assert plan_chunk_pipeline(2, 2) == [
        ((0, 0), None),
        ((1, 0), (0, 0)),
        ((0, 1), (1, 0)),
        ((1, 1), (0, 1)),
        ((0, 2), (1, 1)),
        ((1, 2), (0, 2)),
        (None, (1, 2)),
    ]


def test_two_chunk_three_stage_order():
    # Completion only happens on the last stage, so step N+1 waits a bubble.
    assert plan_chunk_pipeline(2, 3) == [
        ((0, 0), None, None),
        ((1, 0), (0, 0), None),
        (None, (1, 0), (0, 0)),
        ((0, 1), None, (1, 0)),
        ((1, 1), (0, 1), None),
        (None, (1, 1), (0, 1)),
        ((0, 2), None, (1, 1)),
        ((1, 2), (0, 2), None),
        (None, (1, 2), (0, 2)),
        (None, None, (1, 2)),
    ]


def test_stepwise_wave_width_matches_world():
    # Four stages interleave four chunks per step before the next wave.
    slots = plan_chunk_pipeline(8, 4, "stepwise")
    launched = [tasks[0] for tasks in slots if tasks[0] is not None]
    assert launched[:8] == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
    ]
    assert (4, 0) in launched
    assert launched.index((4, 0)) > launched.index((3, 2))


def test_six_chunks_fill_two_stages_in_nineteen_slots():
    slots = plan_chunk_pipeline(6, 2)
    assert len(slots) == 19
    assert sum(all(task is not None for task in tasks) for tasks in slots) == 17
    assert slots[0][1] is None
    assert slots[-1][0] is None


def test_single_chunk_waits_for_each_last_stage():
    assert plan_chunk_pipeline(1, 2) == [
        ((0, 0), None),
        (None, (0, 0)),
        ((0, 1), None),
        (None, (0, 1)),
        ((0, 2), None),
        (None, (0, 2)),
    ]


def test_single_chunk_drains_three_stages():
    assert plan_chunk_pipeline(1, 3) == [
        ((0, 0), None, None),
        (None, (0, 0), None),
        (None, None, (0, 0)),
        ((0, 1), None, None),
        (None, (0, 1), None),
        (None, None, (0, 1)),
        ((0, 2), None, None),
        (None, (0, 2), None),
        (None, None, (0, 2)),
    ]
