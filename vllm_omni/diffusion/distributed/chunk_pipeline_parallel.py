# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Noisy chunk scheduling over the existing one- or two-stage layer pipeline."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.distributed.parallel_state import get_pp_group

ChunkStep = tuple[int, int]

# Request extra_args keys shared by the engine dummy request, the pipeline,
# the benchmark client and tests. Keep one source of truth; the engine dummy
# request historically sent "chunk_lag" while the pipeline read "chunk_gap",
# so the canonical key is now CHUNK_GAP_KEY everywhere.
CHUNK_SCHEDULE_KEY = "chunk_schedule"
CHUNK_FRAMES_KEY = "chunk_frames"
CHUNK_COND_FRAMES_KEY = "chunk_cond_frames"
CHUNK_GAP_KEY = "chunk_gap"
CHUNK_CONDITIONING_KEY = "chunk_conditioning"
KV_HISTORY_CHUNKS_KEY = "kv_history_chunks"
KV_SOURCE_POLICY_KEY = "kv_source_policy"
COLLECT_PP_METRICS_KEY = "collect_pp_metrics"
EXPERIMENT_ITERATION_KEY = "experiment_iteration"

# KV source policies, decoupled from the slot schedule:
# - "latest": each chunk reads the highest finished version of its history
#   chunks (``plan_latest_kv_sources``). Valid under both schedules; serial
#   + latest is the replay control that isolates the schedule's contribution.
# - "clean": each chunk reads only the clean pass of its history chunks
#   (``plan_clean_kv_sources``). Requires the serial schedule, whose strict
#   ordering guarantees the clean pass has run; this is the Self Forcing
#   semantics aligned with the paper.
KV_SOURCE_POLICIES = ("latest", "clean")


def plan_chunk_pipeline(
    chunks: int,
    gap: int,
    world: int,
    schedule: str = "stepwise",
    *,
    kv: bool = False,
    num_denoise_steps: int = 3,
) -> list[tuple[ChunkStep | None, ...]]:
    """Plan stage work using only results completed before the current slot."""
    if chunks < 1 or gap not in (1, 2) or world not in (1, 2):
        raise ValueError("Chunk pipeline requires positive chunks, gap 1/2 and one/two stages")
    if num_denoise_steps < 1:
        raise ValueError("Chunk pipeline requires a positive number of denoise steps")
    steps = num_denoise_steps + 1 if kv else num_denoise_steps
    if schedule == "serial":
        jobs = [(chunk, step) for chunk in range(chunks) for step in range(steps)]
    elif schedule == "stepwise":
        jobs = [
            (chunk, step)
            for first in range(0, chunks, 2)
            for step in range(steps)
            for chunk in range(first, min(first + 2, chunks))
        ]
    else:
        raise ValueError("Chunk schedule must be 'serial' or 'stepwise'")

    slots = []
    completed = set()
    pending = None
    cursor = 0
    while cursor < len(jobs) or pending is not None:
        launched = None
        if cursor < len(jobs) and not (schedule == "serial" and pending is not None):
            chunk, step = jobs[cursor]
            dependencies = {(chunk, step - 1)} if step else set()
            if not kv and chunk > 0 and step >= gap:
                dependencies.add((chunk - 1, step - gap))
            if dependencies <= completed:
                launched = jobs[cursor]
                cursor += 1
        slot = (launched,) if world == 1 else (launched, pending)
        if all(task is None for task in slot):
            raise RuntimeError("Chunk schedule cannot satisfy the next task's dependencies")
        slots.append(slot)
        if slot[-1] is not None:
            completed.add(slot[-1])
        pending = launched if world == 2 else None
    return slots


def plan_latest_kv_sources(slots, history_chunks: int):
    """Freeze latest completed versions BEFORE a slot, separately per layer stage.

    The history of each chunk concatenates the highest finished denoise or
    clean step of its history chunks as of the slot's position in the plan.
    Valid under both schedules: under stepwise this is the algorithm as
    designed, under serial it is the replay control (identical versions to
    stepwise, no parallelism). Every history chunk in the window must have a
    finished version on the reading rank. The extra final step is a forward
    at t=0 on the clean sample; it does not denoise.
    """
    if history_chunks < 1:
        raise ValueError("KV history must contain at least one chunk")
    latest = [{} for _ in slots[0]]
    sources = [{} for _ in slots[0]]
    for tasks in slots:
        for rank, task in enumerate(tasks):
            if task is not None:
                chunk, _ = task
                window = range(max(0, chunk - history_chunks), chunk)
                missing = [c for c in window if c not in latest[rank]]
                if missing:
                    raise RuntimeError(
                        f"Task {task} on rank {rank} is scheduled before history chunks {missing} "
                        "have a finished version on that rank"
                    )
                sources[rank][task] = tuple((c, latest[rank][c]) for c in window)
        for rank, task in enumerate(tasks):
            if task is not None:
                chunk, step = task
                latest[rank][chunk] = step
    return sources


def plan_clean_kv_sources(slots, history_chunks: int, clean_step: int):
    """Serial Self Forcing: each chunk concatenates predecessors' clean KV only.

    Serial ordering guarantees every chunk in the window has run its clean pass
    (task (chunk, clean_step)) before the reading chunk starts, so the sources
    do not depend on slot completion. The clean pass publishes its KV and does
    not denoise.
    """
    if history_chunks < 1:
        raise ValueError("KV history must contain at least one chunk")
    if clean_step < 0:
        raise ValueError("Clean KV step must be non-negative")
    sources = [{} for _ in slots[0]]
    for tasks in slots:
        for rank, task in enumerate(tasks):
            if task is None:
                continue
            chunk, _ = task
            start = max(0, chunk - history_chunks)
            sources[rank][task] = tuple((c, clean_step) for c in range(start, chunk))
    return sources


@dataclass
class ChunkKVContext:
    """Request-local, stage-local cache of post-normalization/post-RoPE K and V."""

    layers: dict
    task: ChunkStep
    sources: tuple[ChunkStep, ...]

    def append(self, layer: int, key: torch.Tensor, value: torch.Tensor):
        cache = self.layers.setdefault(layer, {})
        if self.task in cache:
            raise RuntimeError(f"Duplicate KV publication for layer {layer}, task {self.task}")
        history = [cache[source] for source in self.sources]
        cache[self.task] = (key.contiguous(), value.contiguous())
        if not history:
            return key, value
        return (
            torch.cat([pair[0] for pair in history] + [key], dim=1),
            torch.cat([pair[1] for pair in history] + [value], dim=1),
        )


def _tensor_bytes(payload: dict[str, torch.Tensor] | None) -> int:
    return sum(value.numel() * value.element_size() for value in payload.values()) if payload else 0


def _kv_retained_bytes(kv_layers: dict) -> int:
    """Logical bytes of the K/V tensors currently cached (numel * element_size),
    not allocator reservations; the cache stores post-RoPE tensors per layer."""
    total = 0
    for cache in kv_layers.values():
        for key, value in cache.values():
            total += key.numel() * key.element_size() + value.numel() * value.element_size()
    return total


@torch.inference_mode()
def run_noisy_chunk_pipeline(
    *,
    predict_noise: Callable,
    scheduler,
    timesteps: torch.Tensor,
    shape: tuple[int, ...],
    chunks: int,
    cond_frames: int,
    gap: int,
    seed: int,
    device: torch.device,
    schedule: str = "stepwise",
    layer_range: tuple[int, int] | None = None,
    initial_latents: torch.Tensor | None = None,
    step_noises: list[torch.Tensor] | None = None,
    kv_history_chunks: int | None = None,
    kv_source_policy: str = "latest",
) -> tuple[torch.Tensor, dict]:
    """Run local layers and exchange activations/state in a common direction order.

    The callback accepts (model_input, timestep, temporal_offset, step_idx,
    intermediate_tensors=None). It returns IntermediateTensors on the first of
    two stages, and a noise-prediction Tensor on the last stage. Each chunk has
    its own RNG; only the last stage draws the two scheduler re-noising samples.
    """
    pp = get_pp_group()
    rank, world = pp.rank_in_group, pp.world_size
    kv = kv_history_chunks is not None
    num_denoise_steps = len(timesteps)
    last_denoise = num_denoise_steps - 1
    slots = plan_chunk_pipeline(chunks, gap, world, schedule, kv=kv, num_denoise_steps=num_denoise_steps)
    kv_layers = {}
    kv_sources = None
    kv_remaining = None
    kv_live = 0
    kv_peak_live = 0
    kv_evicted = 0
    kv_peak_retained_bytes = 0
    if kv:
        if kv_source_policy not in KV_SOURCE_POLICIES:
            raise ValueError(f"KV source policy must be one of {KV_SOURCE_POLICIES}")
        if kv_source_policy == "clean" and schedule != "serial":
            raise ValueError(
                "The clean KV source policy requires the serial schedule: only its strict "
                "ordering guarantees every history chunk has run its clean pass"
            )
        # The "latest" policy always freezes sources against the stepwise
        # reference plan: under stepwise that is the algorithm as designed,
        # under serial it makes the run a replay control whose attention
        # history is identical to the stepwise schedule's. The "clean" policy
        # reads the serial plan's own completion order instead.
        if kv_source_policy == "clean":
            kv_sources = plan_clean_kv_sources(slots, kv_history_chunks, num_denoise_steps)[rank]
        else:
            reference_slots = slots
            if schedule != "stepwise":
                reference_slots = plan_chunk_pipeline(
                    chunks, gap, world, "stepwise", kv=True, num_denoise_steps=num_denoise_steps
                )
            kv_sources = plan_latest_kv_sources(reference_slots, kv_history_chunks)[rank]
        if initial_latents is None or step_noises is None or len(step_noises) != last_denoise:
            raise ValueError(
                "KV execution requires the shared full-video initial sample and one re-noising tensor per transition"
            )
        # Each task's stored KV is consumed by the later same-rank tasks that
        # list it in their frozen sources. Zero-consumer tasks free immediately.
        kv_remaining = {task: 0 for task in kv_sources}
        for sources in kv_sources.values():
            for source in sources:
                kv_remaining[source] += 1
    if num_denoise_steps < 1 or not 1 <= cond_frames <= shape[2]:
        raise ValueError("Chunk pipeline requires DMD timesteps and a valid condition prefix")
    t_l, length = shape[2], cond_frames
    current, generators, cache, clean_chunks = {}, {}, {}, {}
    records, slot_records = [], []
    # CUDA events are resolved after the request completes. Resolving each
    # event in the hot loop would make the CPU wait for every stage forward.
    pending_timings = []
    comm_ms = 0.0
    activation_bytes = sample_bytes = feedback_bytes = condition_bytes = 0
    stage_input = None
    trace_enabled = os.environ.get("VLLM_OMNI_CHUNK_PP_TRACE") == "1"

    def trace_scope(name: str):
        return torch.profiler.record_function(name) if trace_enabled else nullcontext()

    def has_consumer(task):
        chunk, step = task
        return not kv and chunk + 1 < chunks and step + gap < num_denoise_steps

    def accept_feedback(task, payload):
        chunk, step = task
        current[chunk] = payload["latents"]
        if step == last_denoise:
            clean_chunks[chunk] = current[chunk]
        if has_consumer(task):
            source = current[chunk][:, :, -length:] if gap == 1 else payload["condition"]
            cache[task] = source.to(torch.bfloat16).contiguous()

    def initialize_chunk(chunk):
        if kv:
            if rank == 0:
                current[chunk] = initial_latents[:, :, chunk * t_l : (chunk + 1) * t_l].contiguous()
            return
        generator = torch.Generator(device=device).manual_seed(seed + chunk * 100003)
        generators[chunk] = generator
        initial = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        if rank == 0:
            current[chunk] = initial
        # Rank one consumes the same initialization draw; its exact sample
        # arrives with model_input, and this generator owns the later draws.

    def barrier():
        if world == 2:
            dist.barrier(group=pp.device_group)

    def release_kv(task):
        nonlocal kv_live, kv_peak_live, kv_evicted, kv_peak_retained_bytes
        # Sample both peaks at the residency maximum, before any release: the
        # task just published its version, so bytes mirror the live count.
        kv_live += 1
        kv_peak_live = max(kv_peak_live, kv_live)
        kv_peak_retained_bytes = max(kv_peak_retained_bytes, _kv_retained_bytes(kv_layers))
        for source in kv_sources[task]:
            kv_remaining[source] -= 1
            if kv_remaining[source] == 0:
                kv_live -= 1
                kv_evicted += 1
                for layer_cache in kv_layers.values():
                    layer_cache.pop(source)
        if kv_remaining[task] == 0:
            kv_live -= 1
            kv_evicted += 1
            for layer_cache in kv_layers.values():
                layer_cache.pop(task)

    # Keep metrics collection out of the timed request path. ``timesteps`` is
    # resident on the accelerator, so converting it in a per-slot record would
    # otherwise introduce a device-to-host synchronization for every forward.
    timestep_values = tuple(float(value) for value in timesteps.detach().cpu().flatten().tolist())
    torch.accelerator.synchronize(device)
    barrier()
    torch.accelerator.reset_peak_memory_stats(device)
    # All chunk forwards and P2P waits are issued on this one stream, so
    # slots execute in issue order and a per-slot Work.wait() is enough to
    # keep consumption ordered; the per-slot CUDA events only timestamp the
    # forwards (Event.record records a time, it synchronizes nothing).
    stream = torch.cuda.current_stream(device)
    origin = torch.cuda.Event(enable_timing=True)
    origin.record(stream)
    start_wall = time.perf_counter()
    for slot_idx, tasks in enumerate(slots):
        task = tasks[rank]
        forward_payload = feedback_payload = None
        record = None
        if task is not None:
            chunk, step = task
            clean_pass = kv and step == num_denoise_steps
            t = timesteps.new_zeros(()) if clean_pass else timesteps[step]
            if step == 0:
                initialize_chunk(chunk)
            has_prefix = not kv and chunk > 0 and step >= gap
            offset = chunk * t_l - (length if has_prefix else 0)
            intermediate = None
            if rank == 0:
                model_input = current[chunk]
                if has_prefix:
                    cond = cache.pop((chunk - 1, step - gap)).float()
                    if gap == 2:
                        cond_generator = torch.Generator(device=device).manual_seed(
                            seed + 1000000007 + chunk * 100003 + step * 1009
                        )
                        noise = torch.randn(cond.shape, generator=cond_generator, device=device, dtype=torch.float32)
                        cond = scheduler.add_noise(cond, noise, t)
                    model_input = torch.cat([cond, model_input], dim=2)
            else:
                model_input = stage_input["model_input"]
                intermediate = IntermediateTensors(
                    {key: value for key, value in stage_input.items() if key != "model_input"}
                )
            first, last = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            first.record(stream)
            kwargs = {"intermediate_tensors": intermediate}
            if kv:
                kwargs["kv_context"] = ChunkKVContext(kv_layers, task, kv_sources[task])
            with trace_scope(f"chunk_pp.forward.slot{slot_idx}.rank{rank}.chunk{chunk}.step{step}"):
                prediction = predict_noise(model_input, t.expand(shape[0]), offset, step, **kwargs)
            last.record(stream)
            record = {
                "chunk_idx": chunk,
                "step_idx": step,
                "slot_idx": slot_idx,
                "rank": rank,
                "stage_idx": rank,
                "timestep": 0.0 if clean_pass else timestep_values[step],
                "forward_ms": 0.0,
                "stage_start_ms": 0.0,
                "stage_end_ms": 0.0,
                "comm_wait_ms": 0.0,
                "input_latent_frames": model_input.shape[2],
            }
            pending_timings.append((record, first, last))
            if kv:
                record.update(
                    pass_kind="clean_kv" if clean_pass else "denoise",
                    kv_sources=[list(source) for source in kv_sources[task]],
                    kv_history_latent_frames=len(kv_sources[task]) * t_l,
                )
                release_kv(task)
            records.append(record)
            if world == 2 and rank == 0:
                forward_payload = {**prediction.tensors, "model_input": model_input}
            else:
                prediction = prediction[:, :, -t_l:]
                clean = model_input if clean_pass else scheduler.predict_clean(prediction, model_input[:, :, -t_l:], t)
                if kv and not clean_pass:
                    clean = clean.to(prediction.dtype)
                if step < last_denoise:
                    noise = (
                        step_noises[step][:, :, chunk * t_l : (chunk + 1) * t_l].contiguous()
                        if kv
                        else torch.randn(shape, generator=generators[chunk], device=device, dtype=torch.float32)
                    )
                    updated = scheduler.add_noise(clean, noise, timesteps[step + 1])
                else:
                    updated = clean
                feedback_payload = {"latents": updated}
                if gap == 2 and has_consumer(task):
                    feedback_payload["condition"] = clean[:, :, -length:].to(torch.bfloat16).contiguous()
                if world == 1:
                    accept_feedback(task, feedback_payload)

        exchange_ms = 0.0
        if world == 2:
            exchange_start = time.perf_counter()
            handles = []
            postprocessors = []
            received_feedback = received_forward = None
            # The dictionary APIs send metadata synchronously. Both ranks must
            # use this SAME direction order, never issue two opposite sends first.
            if tasks[1] is not None:
                if rank == 1:
                    feedback_bytes += _tensor_bytes({"latents": feedback_payload["latents"]})
                    condition_bytes += _tensor_bytes(
                        {"condition": feedback_payload["condition"]} if "condition" in feedback_payload else None
                    )
                    with trace_scope(f"chunk_pp.metadata.send_feedback.slot{slot_idx}"):
                        handles.extend(pp.isend_tensor_dict(feedback_payload, dst=0))
                else:
                    with trace_scope(f"chunk_pp.metadata.recv_feedback.slot{slot_idx}"):
                        received_feedback, work, postprocess = pp.irecv_tensor_dict(src=1)
                    postprocessors.extend(postprocess)
                    handles.extend(work)
            if tasks[0] is not None:
                if rank == 0:
                    activation_bytes += _tensor_bytes(
                        {key: value for key, value in forward_payload.items() if key != "model_input"}
                    )
                    sample_bytes += _tensor_bytes({"model_input": forward_payload["model_input"]})
                    with trace_scope(f"chunk_pp.metadata.send_forward.slot{slot_idx}"):
                        handles.extend(pp.isend_tensor_dict(forward_payload, dst=1))
                else:
                    with trace_scope(f"chunk_pp.metadata.recv_forward.slot{slot_idx}"):
                        received_forward, work, postprocess = pp.irecv_tensor_dict(src=0)
                    postprocessors.extend(postprocess)
                    handles.extend(work)
            # Keep all outgoing and incoming payloads alive until transfer ends.
            with trace_scope(f"chunk_pp.p2p.wait.slot{slot_idx}"):
                for handle in handles:
                    handle.wait()
            for postprocess in postprocessors:
                postprocess()
            # ``Work.wait`` establishes the dependency from the P2P work to
            # this stream. A full stream synchronization here only blocks the
            # host before it can enqueue the next ready slot.
            if received_feedback is not None:
                accept_feedback(tasks[1], received_feedback)
            stage_input = received_forward
            exchange_ms = (time.perf_counter() - exchange_start) * 1000
            comm_ms += exchange_ms
        if record is not None:
            record["comm_wait_ms"] = exchange_ms
        slot_records.append({"slot_idx": slot_idx, "task": task, "comm_wait_ms": exchange_ms})
    if cache:
        raise RuntimeError(f"Unconsumed chunk conditions: {list(cache)}")
    torch.accelerator.synchronize(device)
    for record, first, last in pending_timings:
        record["forward_ms"] = first.elapsed_time(last)
        record["stage_start_ms"] = origin.elapsed_time(first)
        record["stage_end_ms"] = origin.elapsed_time(last)
    barrier()
    local_wall = (time.perf_counter() - start_wall) * 1000
    payload = {
        "rank": rank,
        "stage_idx": rank,
        "layer_range": list(layer_range) if layer_range is not None else None,
        "steps": records,
        "slots": slot_records,
        "denoise_wall_ms": local_wall,
        "comm_wait_ms": comm_ms,
        "comm_wait_scope": (
            "host wall time of the per-slot tensor-dict exchange, including dict metadata, "
            "P2P submission and waits, and postprocessing; not device-communication-only time"
        ),
        "condition_bytes_sent": condition_bytes,
        "activation_bytes_sent": activation_bytes,
        "sample_bytes_sent": sample_bytes,
        "feedback_bytes_sent": feedback_bytes,
        "total_payload_bytes_sent": activation_bytes + sample_bytes + feedback_bytes + condition_bytes,
        "denoise_peak_allocated_bytes": torch.accelerator.max_memory_allocated(device),
    }
    if kv:
        payload.update(
            kv_live_versions=kv_peak_live,
            kv_evicted_versions=kv_evicted,
            kv_retained_bytes_est=kv_peak_retained_bytes,
        )
    all_metrics = [payload]
    if world == 2:
        all_metrics = [None, None]
        dist.all_gather_object(all_metrics, payload, group=pp.cpu_group)

    gather_start = time.perf_counter()
    full_shape = (*shape[:2], t_l * chunks, *shape[3:])
    if rank == 0:
        result = torch.cat([clean_chunks[chunk] for chunk in range(chunks)], dim=2)
    else:
        result = torch.zeros(full_shape, device=device, dtype=torch.float32)
    torch.accelerator.synchronize(device)
    metrics = {
        "execution": "whole_request_layer_chunk_pipeline",
        "schedule": schedule,
        "gap": gap,
        "lag": gap,
        "world_size": world,
        "chunks": chunks,
        "chunk_latent_frames": t_l,
        "cond_frames": length,
        "seed": seed,
        "latent_shape": list(shape),
        "ranks": all_metrics,
        "forward_call_unit": "layer_stage" if world == 2 else "whole_transformer",
        "stage_time_basis": "rank_local_cuda_event_origin; clocks are not aligned across ranks",
        "activation_bytes_include_model_input": False,
        "condition_bytes_scope": "separate bf16 clean-prefix payload for gap2; gap1 derives prefix from feedback",
        "denoise_wall_ms": max(item["denoise_wall_ms"] for item in all_metrics),
        "forward_total_ms": sum(row["forward_ms"] for item in all_metrics for row in item["steps"]),
        "comm_wait_rank_sum_ms": sum(item["comm_wait_ms"] for item in all_metrics),
        "condition_bytes_sent": sum(item["condition_bytes_sent"] for item in all_metrics),
        "activation_bytes_sent": sum(item["activation_bytes_sent"] for item in all_metrics),
        "sample_bytes_sent": sum(item["sample_bytes_sent"] for item in all_metrics),
        "feedback_bytes_sent": sum(item["feedback_bytes_sent"] for item in all_metrics),
        "total_payload_bytes_sent": sum(item["total_payload_bytes_sent"] for item in all_metrics),
        "latent_gather_ms": (time.perf_counter() - gather_start) * 1000,
    }
    if kv:
        metrics.update(
            conditioning="per_layer_latest_kv",
            kv_history_chunks=kv_history_chunks,
            kv_source_policy=kv_source_policy,
            kv_version_policy=(
                "each chunk reads the highest finished version of its history chunks "
                "(serial + latest is the replay control of the stepwise schedule)"
                if kv_source_policy == "latest"
                else "each chunk reads only the clean pass of its history chunks (Self Forcing)"
            ),
            kv_live_versions=max(item["kv_live_versions"] for item in all_metrics),
            kv_evicted_versions=sum(item["kv_evicted_versions"] for item in all_metrics),
            kv_retained_bytes_est=max(item["kv_retained_bytes_est"] for item in all_metrics),
            kv_retained_bytes_scope=(
                "logical K/V bytes (numel * element_size) per layer stage, sampled at each "
                "release entry; summed across layers, not allocator reserved bytes"
            ),
            clean_kv_forward=True,
            clean_kv_forward_total_ms=sum(
                row["forward_ms"] for item in all_metrics for row in item["steps"] if row["pass_kind"] == "clean_kv"
            ),
            noise_contract="shared full-video initial sample and BF16 re-noising; native BF16 clean update",
            cond_frames=0,
            gap=None,
            lag=None,
        )
    return result, metrics
