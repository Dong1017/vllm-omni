# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for chunk dependencies, the two-stage layer schedule, and the KV runner."""

import queue
import threading
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


def test_latest_kv_sources_reject_unfinished_history():
    from vllm_omni.diffusion.distributed.chunk_pipeline_parallel import plan_latest_kv_sources

    # A broken plan that schedules chunk 1 before chunk 0 has run anywhere
    # must be rejected instead of silently conditioning on a short history.
    slots = [((0, 0), None), (None, (1, 0))]
    with pytest.raises(RuntimeError, match=r"history chunks \[0\]"):
        plan_latest_kv_sources(slots, 6)


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


# Peaks for 6 chunks, 4 steps, H-history conditioning, default kv_source_policy
# "latest". Sources are frozen against the stepwise reference plan under both
# schedules (serial is the replay control), but serial's consumption order
# releases frozen versions later than stepwise: every chunk runs all its steps
# before the next chunk starts, so predecessors' later versions stay resident
# until the reading chunk begins. Stepwise peak 3 at H=1 / 6 at H=6; serial
# replay peak 5 at H=1 / 9 at H=6. The clean policy's serial peaks (2 at H=1 /
# 6 at H=6) are covered by the dedicated clean-semantics test. Each rank
# executes every task once, so 24 versions are published and evicted per rank
# (48 across PP2).
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


def test_clean_kv_sources_target_only_clean_passes():
    from vllm_omni.diffusion.distributed.chunk_pipeline_parallel import plan_clean_kv_sources

    slots = plan_chunk_pipeline(3, 1, 2, "serial", kv=True, num_denoise_steps=3)
    sources = plan_clean_kv_sources(slots, 2, 3)
    for rank in (0, 1):
        for task, source in sources[rank].items():
            chunk = task[0]
            assert source == tuple((c, 3) for c in range(max(0, chunk - 2), chunk))


def test_chunk_kv_serial_semantics_selects_clean_versions(monkeypatch) -> None:
    """With the clean policy, serial tasks read the predecessors' clean-pass KV.

    The replay policy would freeze the latest finished version before each
    slot; the clean policy always points at (chunk, clean_step) instead.
    """
    chunks, t_l, layers_count = 3, 2, 3
    shape = (1, 4, t_l, 1, 1)
    _patch_cpu_chunk_pp_runtime(monkeypatch, rank_in_group=1, world_size=2, peer_payload_shape=shape)
    initial_latents = torch.zeros((1, 4, chunks * t_l, 1, 1), dtype=torch.float32)
    step_noises = [torch.zeros_like(initial_latents) for _ in range(4)]
    checked = 0

    def fake_predict_noise(
        model_input, timestep, temporal_offset, step_idx, intermediate_tensors=None, kv_context=None
    ):
        nonlocal checked
        del timestep, temporal_offset, step_idx, intermediate_tensors
        if kv_context is not None:
            for layer in range(layers_count):
                kv_context.append(
                    layer,
                    torch.full_like(model_input, kv_context.task[1]),
                    torch.full_like(model_input, kv_context.task[1]),
                )
            for source in kv_context.sources:
                # Every source version must be a clean pass (step index 5 ==
                # num_denoise_steps, the extra t=0 forward) frozen by the
                # planner.
                assert source[1] == 5, source
                checked += 1
        return torch.zeros_like(model_input)

    result, metrics = run_noisy_chunk_pipeline(
        predict_noise=fake_predict_noise,
        scheduler=_StubChunkScheduler(),
        timesteps=torch.tensor([900.0, 700.0, 500.0, 300.0, 100.0]),
        shape=shape,
        chunks=chunks,
        cond_frames=t_l,
        gap=1,
        seed=0,
        device=torch.device("cpu"),
        schedule="serial",
        initial_latents=initial_latents,
        step_noises=step_noises,
        kv_history_chunks=2,
        kv_source_policy="clean",
    )

    del result
    assert checked > 0
    assert metrics["kv_source_policy"] == "clean"
    assert metrics["kv_evicted_versions"] == chunks * (4 + 2) * 2


def test_chunk_kv_clean_policy_rejects_stepwise(monkeypatch) -> None:
    """The clean policy depends on serial ordering; stepwise must reject it."""
    _patch_cpu_chunk_pp_runtime(monkeypatch, rank_in_group=0, world_size=1, peer_payload_shape=(1, 4, 2, 1, 1))
    with pytest.raises(ValueError, match="clean KV source policy requires the serial schedule"):
        run_noisy_chunk_pipeline(
            predict_noise=lambda *args, **kwargs: torch.zeros(1, 4, 2, 1, 1),
            scheduler=_StubChunkScheduler(),
            timesteps=torch.tensor([900.0]),
            shape=(1, 4, 2, 1, 1),
            chunks=2,
            cond_frames=2,
            gap=1,
            seed=0,
            device=torch.device("cpu"),
            schedule="stepwise",
            initial_latents=torch.zeros((1, 4, 4, 1, 1)),
            step_noises=[torch.zeros((1, 4, 4, 1, 1))],
            kv_history_chunks=1,
            kv_source_policy="clean",
        )


def test_chunk_kv_serial_latest_replays_stepwise_sources(monkeypatch) -> None:
    """Serial + latest is the replay control: identical sources to stepwise.

    The version consumed by a (chunk, step) task must be the same under both
    schedules so a serial re-run isolates the schedule's contribution without
    changing the attention history.
    """
    chunks, t_l, layers_count = 3, 2, 2
    shape = (1, 4, t_l, 1, 1)
    _patch_cpu_chunk_pp_runtime(monkeypatch, rank_in_group=1, world_size=2, peer_payload_shape=shape)
    initial_latents = torch.zeros((1, 4, chunks * t_l, 1, 1), dtype=torch.float32)
    step_noises = [torch.zeros_like(initial_latents) for _ in range(2)]
    observed = {}

    def fake_predict_noise(
        model_input, timestep, temporal_offset, step_idx, intermediate_tensors=None, kv_context=None
    ):
        del timestep, temporal_offset, step_idx, intermediate_tensors
        if kv_context is not None:
            observed[kv_context.task] = tuple(kv_context.sources)
            for layer in range(layers_count):
                kv_context.append(
                    layer,
                    torch.full_like(model_input, kv_context.task[1]),
                    torch.full_like(model_input, kv_context.task[1]),
                )
        return torch.zeros_like(model_input)

    common = dict(
        predict_noise=fake_predict_noise,
        scheduler=_StubChunkScheduler(),
        timesteps=torch.tensor([900.0, 500.0, 100.0]),
        shape=shape,
        chunks=chunks,
        cond_frames=t_l,
        gap=1,
        seed=0,
        device=torch.device("cpu"),
        initial_latents=initial_latents,
        step_noises=step_noises,
        kv_history_chunks=2,
        kv_source_policy="latest",
    )
    _, serial_metrics = run_noisy_chunk_pipeline(schedule="serial", **common)
    serial_sources = dict(observed)
    observed.clear()
    _, stepwise_metrics = run_noisy_chunk_pipeline(schedule="stepwise", **common)

    assert serial_sources == observed
    assert serial_metrics["kv_source_policy"] == "latest"
    assert stepwise_metrics["kv_source_policy"] == "latest"


def test_chunk_kv_world_two_matches_whole_transformer(monkeypatch) -> None:
    """The two-stage layer-split KV run must reproduce the single-process result.

    The rank split only partitions layers, and every random draw is shared
    through initial_latents and step_noises, so the final latent is expected
    to be bit-identical across world sizes -- the chunk-pipeline counterpart
    of WaveServe's "vertical and horizontal produce identical latents" check.
    """
    chunks, t_l = 2, 2
    shape = (1, 4, t_l, 1, 1)
    initial_latents = torch.randn((1, 4, chunks * t_l, 1, 1), generator=torch.Generator().manual_seed(11))
    step_noises = [
        torch.randn(initial_latents.shape, generator=torch.Generator().manual_seed(101 + step)) for step in range(2)
    ]
    common = dict(
        scheduler=_StubChunkScheduler(),
        timesteps=torch.tensor([900.0, 500.0, 100.0]),
        shape=shape,
        chunks=chunks,
        cond_frames=t_l,
        gap=1,
        seed=0,
        device=torch.device("cpu"),
        schedule="stepwise",
        initial_latents=initial_latents,
        step_noises=step_noises,
        kv_history_chunks=2,
    )

    # Two threads with rank-aware stubs and clone-forwarding queues: the two
    # stages actually exchange the tensors they produced, like the real P2P
    # wait, so a dropped or swapped payload breaks the bit-equality check.
    # (timesteps len 3 → step_noises len 2 = last_denoise; stepwise uses the
    # latest-version planner, so this also exercises the pre-existing path.)
    exchange = {0: queue.Queue(), 1: queue.Queue()}
    rank_state = threading.local()

    split_stages = False

    def fake_predict_noise(
        model_input, timestep, temporal_offset, step_idx, intermediate_tensors=None, kv_context=None
    ):
        del timestep, temporal_offset, step_idx, intermediate_tensors
        if kv_context is not None:
            version = torch.full_like(model_input, kv_context.task[0] * 10 + kv_context.task[1])
            kv_context.append(0, version, version)
        # Only the split two-stage run hands intermediates to the peer via the
        # payload; the whole-transformer baseline returns the noise tensor.
        if split_stages and getattr(rank_state, "rank", 0) == 0:
            return IntermediateTensors({"hidden_states": torch.zeros_like(model_input)})
        return torch.zeros_like(model_input)

    _patch_cpu_chunk_pp_runtime(monkeypatch, world_size=1)
    whole, _ = run_noisy_chunk_pipeline(predict_noise=fake_predict_noise, **common)
    split_stages = True

    class _ThreadedPP:
        world_size = 2
        device_group = None
        cpu_group = None

        @property
        def rank_in_group(self):
            return rank_state.rank

        def isend_tensor_dict(self, payload, dst=None):
            del dst
            exchange[rank_state.rank].put({key: value.clone() for key, value in payload.items()})
            return [_CompletedWork()]

        def irecv_tensor_dict(self, src=None):
            del src
            payload = exchange[1 - rank_state.rank].get()
            return payload, [_CompletedWork()], [lambda: None]

    monkeypatch.setattr("vllm_omni.diffusion.distributed.chunk_pipeline_parallel.get_pp_group", lambda: _ThreadedPP())
    monkeypatch.setattr("vllm_omni.diffusion.distributed.chunk_pipeline_parallel.dist.barrier", lambda *a, **k: None)
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.chunk_pipeline_parallel.dist.all_gather_object",
        lambda out, payload, group=None: out.__setitem__(slice(None), [payload] * len(out)),
    )

    results = {}

    def run_rank(rank):
        rank_state.rank = rank
        results[rank] = run_noisy_chunk_pipeline(predict_noise=fake_predict_noise, **common)

    threads = [threading.Thread(target=run_rank, args=(rank,)) for rank in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Only rank zero holds the final latents; rank one returns placeholder zeros.
    torch.testing.assert_close(results[0][0], whole, rtol=0, atol=0)
