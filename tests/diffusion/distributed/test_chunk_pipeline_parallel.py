# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for chunk dependencies and the two-stage layer schedule."""

import pytest

from vllm_omni.diffusion.distributed.chunk_pipeline_parallel import plan_chunk_pipeline


@pytest.mark.parametrize("chunks", [1, 2, 3, 6])
def test_latest_kv_never_reads_same_slot_or_future_chunk(chunks):
    from vllm_omni.diffusion.distributed.chunk_pipeline_parallel import plan_latest_kv_sources

    slots = plan_chunk_pipeline(chunks, 1, 2, kv=True)
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
    # The serial replay must have every explicitly selected historical version.
    available = [set(), set()]
    for tasks in plan_chunk_pipeline(chunks, 1, 2, "serial", kv=True):
        for rank, task in enumerate(tasks):
            if task is not None:
                assert set(sources[rank][task]) <= available[rank]
                available[rank].add(task)


def test_latest_kv_includes_clean_versions_after_pair_finishes():
    from vllm_omni.diffusion.distributed.chunk_pipeline_parallel import plan_latest_kv_sources

    sources = plan_latest_kv_sources(plan_chunk_pipeline(4, 1, 2, kv=True), 6)
    for rank in range(2):
        assert sources[rank][(1, 0)] == ((0, 0),)
        assert sources[rank][(2, 0)] == ((0, 3), (1, 3))


def test_kv_attention_matches_explicit_history_and_keeps_versions_separate():
    import torch
    import torch.nn.functional as F

    from vllm_omni.diffusion.distributed.chunk_pipeline_parallel import ChunkKVContext

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


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("chunks", [1, 2, 3, 6])
@pytest.mark.parametrize("gap", [1, 2])
@pytest.mark.parametrize("schedule", ["serial", "stepwise"])
def test_each_layer_stage_executes_once_after_dependencies(world, chunks, gap, schedule):
    slots = plan_chunk_pipeline(chunks, gap, world, schedule)
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
                if chunk > 0 and step >= gap:
                    assert completed[chunk - 1, step - gap] < slot_idx
            else:
                assert executed[stage - 1][task] < slot_idx
            executed[stage][task] = slot_idx
        if tasks[-1] is not None:
            completed[tasks[-1]] = slot_idx
    assert set(completed) == expected
    assert all(set(stage) == expected for stage in executed)
    # Two stages mean twice as many local forward calls, not twice as many DiT steps.
    assert sum(len(stage) for stage in executed) == world * chunks * 3


@pytest.mark.parametrize("gap", [1, 2])
@pytest.mark.parametrize("chunks", [1, 2, 3, 6])
def test_serial_and_stepwise_execute_the_same_jobs(gap, chunks):
    jobs = []
    for schedule in ("serial", "stepwise"):
        slots = plan_chunk_pipeline(chunks, gap, 2, schedule)
        jobs.append([{task for tasks in slots if (task := tasks[stage]) is not None} for stage in range(2)])
    assert jobs[0] == jobs[1]


def test_gap_one_two_chunk_stage_order():
    assert plan_chunk_pipeline(2, 1, 2) == [
        ((0, 0), None),
        ((1, 0), (0, 0)),
        ((0, 1), (1, 0)),
        ((1, 1), (0, 1)),
        ((0, 2), (1, 1)),
        ((1, 2), (0, 2)),
        (None, (1, 2)),
    ]


@pytest.mark.parametrize("gap", [1, 2])
def test_six_chunks_fill_two_stages_in_nineteen_slots(gap):
    slots = plan_chunk_pipeline(6, gap, 2)
    assert len(slots) == 19
    assert sum(all(task is not None for task in tasks) for tasks in slots) == 17
    assert slots[0][1] is None
    assert slots[-1][0] is None


@pytest.mark.parametrize("gap", [1, 2])
def test_single_chunk_waits_for_each_last_stage(gap):
    assert plan_chunk_pipeline(1, gap, 2) == [
        ((0, 0), None),
        (None, (0, 0)),
        ((0, 1), None),
        (None, (0, 1)),
        ((0, 2), None),
        (None, (0, 2)),
    ]
