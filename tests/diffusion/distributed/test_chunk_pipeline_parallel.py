# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for chunk dependencies, the two-stage layer schedule, and the KV runner."""

from types import SimpleNamespace

import pytest
import torch
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.distributed.chunk_pipeline_parallel import (
    plan_chunk_pipeline,
    run_noisy_chunk_pipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


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


class _StubChunkScheduler:
    def predict_clean(self, model_output, sample, timestep):
        del timestep
        return sample - model_output

    def add_noise(self, clean_sample, noise, timestep):
        del timestep
        return clean_sample + noise


class _FakeCudaStream:
    pass


class _FakeCudaEvent:
    def __init__(self, enable_timing=False):
        del enable_timing

    def record(self, stream=None):
        del stream

    def elapsed_time(self, other):
        del other
        return 0.0


class _CompletedWork:
    def wait(self):
        return None


def _patch_cpu_chunk_pp_runtime(monkeypatch, rank_in_group=0, world_size=1, peer_payload_shape=None):
    pp_stub = SimpleNamespace(rank_in_group=rank_in_group, world_size=world_size, device_group=None, cpu_group=None)
    if world_size == 2:
        # Simulate the peer stage. Stage one (rank 0) receives feedback
        # payloads ({"latents": ...}); stage two (rank 1) receives forward
        # payloads carrying the hidden states and the model input.
        def fake_isend(payload, dst=None):
            del dst, payload
            return [_CompletedWork()]

        def fake_irecv(src=None):
            del src
            if rank_in_group == 0:
                payload = {"latents": torch.zeros(peer_payload_shape)}
            else:
                payload = {
                    "hidden_states": torch.zeros(peer_payload_shape),
                    "model_input": torch.zeros(peer_payload_shape),
                }
            return payload, [_CompletedWork()], [lambda: None]

        pp_stub.isend_tensor_dict = fake_isend
        pp_stub.irecv_tensor_dict = fake_irecv
    monkeypatch.setattr("vllm_omni.diffusion.distributed.chunk_pipeline_parallel.get_pp_group", lambda: pp_stub)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _FakeCudaStream())
    monkeypatch.setattr(torch.cuda, "Event", _FakeCudaEvent)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.accelerator, "max_memory_allocated", lambda *a, **k: 0)
    if world_size == 2:
        monkeypatch.setattr(
            "vllm_omni.diffusion.distributed.chunk_pipeline_parallel.dist.barrier", lambda *a, **k: None
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.distributed.chunk_pipeline_parallel.dist.all_gather_object",
            lambda out, payload, group=None: out.__setitem__(slice(None), [payload] * len(out)),
        )


# Peaks for 6 chunks, 4 steps, H-history conditioning. The stepwise reference
# freezes sources before each slot; serial replays the identical versions. At
# H=1 the stepwise peak is 3 live versions (5 for serial); at the formal
# benchmark's H=6 it is 6 (9 for serial). Each rank executes every task once,
# so 24 versions are published and evicted per rank (48 across PP2).
_EXPECTED_PEAKS = {
    (1, "stepwise"): 3,
    (1, "serial"): 5,
    (6, "stepwise"): 6,
    (6, "serial"): 9,
}


@pytest.mark.parametrize("rank_in_group", [0, 1])
@pytest.mark.parametrize(
    ("kv_history_chunks", "schedule", "expected_peak"), sorted((*k, v) for k, v in _EXPECTED_PEAKS.items())
)
def test_chunk_kv_evicts_versions_and_reports_peak(
    monkeypatch, rank_in_group: int, kv_history_chunks: int, schedule: str, expected_peak: int
) -> None:
    """KV versions are freed once their last consumer's forward completes."""
    chunks = 6
    t_l = 2
    layers_count = 3
    shape = (1, 4, t_l, 1, 1)
    _patch_cpu_chunk_pp_runtime(monkeypatch, rank_in_group=rank_in_group, world_size=2, peer_payload_shape=shape)
    initial_latents = torch.zeros((1, 4, chunks * t_l, 1, 1), dtype=torch.float32)
    step_noises = [torch.zeros_like(initial_latents), torch.zeros_like(initial_latents)]
    captured: dict[str, object] = {}
    # Identifiable, non-zero K/V per (chunk, step) version.
    versions: dict[tuple[int, int], torch.Tensor] = {}
    checked_histories = 0

    def fake_predict_noise(
        model_input, timestep, temporal_offset, step_idx, intermediate_tensors=None, kv_context=None
    ):
        del timestep, temporal_offset, step_idx, intermediate_tensors
        nonlocal checked_histories
        if kv_context is not None:
            captured["kv_layers"] = kv_context.layers
            version = torch.full_like(model_input, kv_context.task[0] * 10 + kv_context.task[1])
            versions[kv_context.task] = version
            for layer in range(layers_count):
                if layer == layers_count - 1 and kv_context.sources:
                    # The consumed history must carry the identified version
                    # values of the frozen sources, in source order.
                    history_key, history_value = kv_context.append(layer, version, version)
                    expected = [versions[source] for source in kv_context.sources]
                    torch.testing.assert_close(history_key, torch.cat([*expected, version], dim=1))
                    torch.testing.assert_close(history_value, torch.cat([*expected, version], dim=1))
                    checked_histories += 1
                else:
                    kv_context.append(layer, version, version)
        if rank_in_group == 0:
            # Stage one hands its intermediates to the peer via the payload.
            return IntermediateTensors({"hidden_states": torch.zeros_like(model_input)})
        return torch.zeros_like(model_input)

    result, metrics = run_noisy_chunk_pipeline(
        predict_noise=fake_predict_noise,
        scheduler=_StubChunkScheduler(),
        timesteps=torch.tensor([900.0, 500.0, 100.0]),
        shape=shape,
        chunks=chunks,
        cond_frames=t_l,
        gap=1,
        seed=0,
        device=torch.device("cpu"),
        schedule=schedule,
        initial_latents=initial_latents,
        step_noises=step_noises,
        kv_history_chunks=kv_history_chunks,
    )

    del result
    kv_layers = captured["kv_layers"]
    # Each rank publishes chunks*4 versions; the top-level metrics sum the
    # per-rank evictions across the two stages (24 + 24 = 48) and take the
    # max live peak.
    assert len(versions) == chunks * 4
    assert metrics["kv_evicted_versions"] == chunks * 4 * 2
    assert metrics["kv_live_versions"] == expected_peak
    # At least one task consumed a non-empty history and its content matched.
    assert checked_histories > 0
    # Nothing survives the request.
    assert all(not cache for cache in kv_layers.values())


def test_chunk_kv_reports_retained_bytes_peak_before_release(monkeypatch) -> None:
    """The bytes peak is sampled when the version is resident, not after eviction."""
    _patch_cpu_chunk_pp_runtime(monkeypatch)
    shape = (1, 4, 2, 1, 1)
    initial_latents = torch.zeros((1, 4, 2, 1, 1), dtype=torch.float32)
    step_noises = [torch.zeros_like(initial_latents), torch.zeros_like(initial_latents)]
    version_bytes = 0

    def fake_predict_noise(
        model_input, timestep, temporal_offset, step_idx, intermediate_tensors=None, kv_context=None
    ):
        nonlocal version_bytes
        del timestep, temporal_offset, step_idx, intermediate_tensors
        if kv_context is not None:
            version = torch.full_like(model_input, kv_context.task[0] * 10 + kv_context.task[1])
            version_bytes = version.numel() * version.element_size()
            kv_context.append(0, version, version)
        return torch.zeros_like(model_input)

    result, metrics = run_noisy_chunk_pipeline(
        predict_noise=fake_predict_noise,
        scheduler=_StubChunkScheduler(),
        timesteps=torch.tensor([900.0, 500.0, 100.0]),
        shape=shape,
        chunks=1,
        cond_frames=shape[2],
        gap=1,
        seed=0,
        device=torch.device("cpu"),
        schedule="stepwise",
        initial_latents=initial_latents,
        step_noises=step_noises,
        kv_history_chunks=1,
    )

    del result
    # A single-chunk request publishes 4 versions (3 denoise + 1 clean) and a
    # single version is resident when each release entry samples the peak.
    assert metrics["kv_evicted_versions"] == 4
    assert metrics["kv_live_versions"] == 1
    assert metrics["kv_retained_bytes_est"] == 2 * version_bytes
