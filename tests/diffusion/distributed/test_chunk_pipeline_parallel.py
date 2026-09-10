# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for chunk dependencies and the two-stage layer schedule."""

import pytest

from vllm_omni.diffusion.distributed.chunk_pipeline_parallel import plan_chunk_pipeline


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
