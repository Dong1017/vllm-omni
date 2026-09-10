# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for the condition dependencies implied by the chunk schedule."""

import pytest

from vllm_omni.diffusion.distributed.chunk_pipeline_parallel import slot_task


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("chunks", [1, 2, 3, 6])
@pytest.mark.parametrize("lag", [1, 2])
def test_slot_task_condition_dependencies(world, chunks, lag):
    # A serial schedule is an upper bound; do not duplicate the runner's slot
    # count calculation or its model/communication implementation.
    schedule = [[slot_task(slot, rank, chunks, world) for rank in range(world)] for slot in range(3 * chunks + world)]
    execution_slot = {}
    for slot, tasks in enumerate(schedule):
        for rank, task in enumerate(tasks):
            if task is None:
                continue
            chunk, step = task
            assert 0 <= chunk < chunks and 0 <= step < 3
            assert rank == chunk % world
            assert task not in execution_slot, f"Repeated execution of {task}"
            execution_slot[task] = slot
    assert set(execution_slot) == {(chunk, step) for chunk in range(chunks) for step in range(3)}
    assert all(task is None for task in schedule[-1])
    for chunk in range(chunks):
        assert execution_slot[chunk, 0] < execution_slot[chunk, 1] < execution_slot[chunk, 2]

    # Independent contract: adjacent chunk k+1 at step s+lag consumes k.s.
    consumer_for = {(chunk, step): (chunk + 1, step + lag) for chunk in range(chunks - 1) for step in range(3 - lag)}
    source_for = {consumer: source for source, consumer in consumer_for.items()}
    caches = [{} for _ in range(world)]
    sent = set()
    consumed = set()
    bidirectional_slots = 0

    for slot, tasks in enumerate(schedule):
        # Consume before delivering this slot's results: a same-slot producer
        # cannot satisfy an input required before the consumer's forward.
        for rank, task in enumerate(tasks):
            if task not in source_for:
                continue
            source = source_for[task]
            assert source in caches[rank], f"{task} needs uncached {source} at slot {slot}"
            assert caches[rank].pop(source) == execution_slot[source] < slot
            assert task not in consumed
            consumed.add(task)

        sends = {(rank, consumer_for[task][0] % world, task) for rank, task in enumerate(tasks) if task in consumer_for}
        # A receiving rank must post at the producer's slot, even when its own
        # task is idle or belongs to an earlier chunk. Pair by endpoint AND key.
        receives = set()
        for receiver in range(world):
            producer = (receiver - 1) % world
            source = tasks[producer]
            if source in consumer_for:
                receives.add((producer, receiver, source))
        assert sends == receives
        if world == 2 and len(sends) == 2:
            bidirectional_slots += 1
            assert {(src, dst) for src, dst, _ in sends} == {(0, 1), (1, 0)}

        for _, receiver, source in receives:
            assert source not in sent
            assert source not in caches[receiver]
            assert execution_slot[consumer_for[source]] > slot
            caches[receiver][source] = slot
            sent.add(source)

    # In world=1 these logical sends are local cache handoffs, not network IO.
    assert len(sent) == (chunks - 1) * (3 - lag)
    assert sent == set(consumer_for)
    assert consumed == set(source_for)
    assert all(not cache for cache in caches)
    if world == 2 and chunks >= 3 and lag == 1:
        assert bidirectional_slots > 0
