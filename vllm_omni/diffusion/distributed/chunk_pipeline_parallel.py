# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chunk scheduling over existing layer-split PP stages.

Layer weights are partitioned at model build time. This module schedules
temporal ``(chunk, step)`` work, optional KV history, and stage communication.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.distributed.parallel_state import get_pp_group

# (chunk_idx, step_idx); with kv, step == num_denoise_steps is the clean pass.
ChunkStep = tuple[int, int]


def plan_chunk_pipeline(
    chunks: int,
    world: int,
    schedule: str = "stepwise",
    *,
    kv: bool = False,
    num_denoise_steps: int = 3,
) -> list[tuple[ChunkStep | None, ...]]:
    """Plan per-slot work for each PP stage.

    Returns slots of length ``world``. A job enters stage 0 and advances one
    stage per slot; completion on the last stage unlocks dependents.

    ``serial`` runs one chunk at a time. ``stepwise`` interleaves waves of
    ``world`` chunks. With ``kv``, each chunk includes one extra clean step.
    """
    if chunks < 1 or world < 1:
        raise ValueError("Chunk pipeline requires positive chunks and pipeline_parallel_size >= 1")
    if num_denoise_steps < 1:
        raise ValueError("Chunk pipeline requires a positive number of denoise steps")
    steps = num_denoise_steps + 1 if kv else num_denoise_steps
    if schedule == "serial":
        jobs = [(chunk, step) for chunk in range(chunks) for step in range(steps)]
    elif schedule == "stepwise":
        jobs = [
            (chunk, step)
            for first in range(0, chunks, world)
            for step in range(steps)
            for chunk in range(first, min(first + world, chunks))
        ]
    else:
        raise ValueError("Chunk schedule must be 'serial' or 'stepwise'")

    slots = []
    completed = set()
    carry = [None] * (world - 1)
    cursor = 0
    while cursor < len(jobs) or any(task is not None for task in carry):
        launched = None
        if cursor < len(jobs) and not (schedule == "serial" and any(task is not None for task in carry)):
            chunk, step = jobs[cursor]
            dependencies = {(chunk, step - 1)} if step else set()
            if dependencies <= completed:
                launched = jobs[cursor]
                cursor += 1
        slot = (launched, *carry)
        if all(task is None for task in slot):
            raise RuntimeError("Chunk schedule cannot satisfy the next task's dependencies")
        slots.append(slot)
        if slot[-1] is not None:
            completed.add(slot[-1])
        carry = list(slot[:-1])
    return slots


def plan_latest_kv_sources(slots, history_chunks: int):
    """Latest-KV sources for stepwise: freeze history before each slot, per stage.

    Each task reads the highest step already published on that rank for prior
    chunks in the window ``H``. Same-slot publications are not visible yet.
    """
    if history_chunks < 1:
        raise ValueError("KV history must contain at least one chunk")
    latest = [{} for _ in slots[0]]
    sources = [{} for _ in slots[0]]
    for tasks in slots:
        for rank, task in enumerate(tasks):
            if task is not None:
                chunk, _ = task
                sources[rank][task] = tuple(
                    (c, latest[rank][c]) for c in range(max(0, chunk - history_chunks), chunk) if c in latest[rank]
                )
        for rank, task in enumerate(tasks):
            if task is not None:
                chunk, step = task
                latest[rank][chunk] = step
    return sources


def plan_clean_kv_sources(slots, history_chunks: int, clean_step: int):
    """Clean-KV sources for serial Self Forcing: predecessors use ``clean_step`` only."""
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


def plan_kv_last_use(slots, sources_by_rank):
    """Last slot on each rank that still references a published KV version."""
    last_use = [{} for _ in slots[0]]
    for slot_idx, tasks in enumerate(slots):
        for rank, task in enumerate(tasks):
            if task is None:
                continue
            last_use[rank][task] = max(last_use[rank].get(task, -1), slot_idx)
            for source in sources_by_rank[rank][task]:
                last_use[rank][source] = max(last_use[rank].get(source, -1), slot_idx)
    return last_use


def evict_kv_versions(layers: dict, versions) -> None:
    """Remove KV versions from the per-layer cache; drop empty layer entries."""
    empty = []
    for layer, cache in layers.items():
        for version in versions:
            cache.pop(version, None)
        if not cache:
            empty.append(layer)
    for layer in empty:
        del layers[layer]


@dataclass
class ChunkKVContext:
    """Stage-local post-norm / post-RoPE K/V cache for one attention call."""

    layers: dict
    task: ChunkStep
    sources: tuple[ChunkStep, ...]

    def append(self, layer: int, key: torch.Tensor, value: torch.Tensor):
        """Store this task's K/V and return history concatenated on dim=1."""
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
    """Byte size of tensors in a communication payload."""
    return sum(value.numel() * value.element_size() for value in payload.values()) if payload else 0


@torch.inference_mode()
def run_chunk_pipeline(
    *,
    predict_noise: Callable,
    predict_clean: Callable,
    add_noise: Callable,
    timesteps: torch.Tensor,
    shape: tuple[int, ...],
    chunks: int,
    seed: int,
    device: torch.device,
    schedule: str = "stepwise",
    layer_range: tuple[int, int] | None = None,
    initial_latents: torch.Tensor | None = None,
    step_noises: list[torch.Tensor] | None = None,
    kv_history_chunks: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Run the chunk timetable on this PP rank.

    All ranks share the same slots; rank ``r`` executes ``tasks[r]``.
    ``predict_noise`` returns ``IntermediateTensors`` on non-last stages and a
    noise tensor on the last stage. ``predict_clean(pred, sample, t)`` and
    ``add_noise(sample, noise, t)`` perform the last-stage latent update.
    Per slot, communicate last→0 feedback, then stage i→i+1 activations.
    KV mode requires shared ``initial_latents`` and ``step_noises``.
    """
    pp = get_pp_group()
    rank, world = pp.rank_in_group, pp.world_size
    last_rank = world - 1
    kv = kv_history_chunks is not None
    num_denoise_steps = len(timesteps)
    last_denoise = num_denoise_steps - 1
    slots = plan_chunk_pipeline(chunks, world, schedule, kv=kv, num_denoise_steps=num_denoise_steps)
    kv_layers = {}
    kv_sources = None
    kv_last_use = None
    kv_peak_versions = 0
    if kv:
        if schedule == "serial":
            sources_by_rank = plan_clean_kv_sources(slots, kv_history_chunks, num_denoise_steps)
        else:
            sources_by_rank = plan_latest_kv_sources(slots, kv_history_chunks)
        kv_sources = sources_by_rank[rank]
        kv_last_use = plan_kv_last_use(slots, sources_by_rank)[rank]
        if initial_latents is None or step_noises is None or len(step_noises) != last_denoise:
            raise ValueError(
                "KV execution requires the shared full-video initial sample "
                "and one re-noising tensor per transition"
            )
    if num_denoise_steps < 1:
        raise ValueError("Chunk pipeline requires a positive number of denoise steps")
    t_l = shape[2]
    current, generators, clean_chunks = {}, {}, {}
    records, slot_records = [], []
    comm_ms = 0.0
    activation_bytes = sample_bytes = feedback_bytes = 0
    stage_input = None

    def accept_feedback(task, payload):
        chunk, step = task
        current[chunk] = payload["latents"]
        if step == last_denoise:
            clean_chunks[chunk] = current[chunk]

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
        # Non-first ranks share the RNG stream; the sample arrives via model_input.

    def barrier():
        if world > 1:
            dist.barrier(group=pp.device_group)

    torch.accelerator.synchronize(device)
    barrier()
    torch.accelerator.reset_peak_memory_stats(device)
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
            offset = chunk * t_l
            intermediate = None
            if rank == 0:
                model_input = current[chunk]
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
            prediction = predict_noise(model_input, t.expand(shape[0]), offset, step, **kwargs)
            last.record(stream)
            last.synchronize()
            if kv:
                kv_peak_versions = max(kv_peak_versions, max((len(cache) for cache in kv_layers.values()), default=0))
                evict_kv_versions(
                    kv_layers, [version for version, use in kv_last_use.items() if use == slot_idx]
                )
            record = {
                "chunk_idx": chunk,
                "step_idx": step,
                "slot_idx": slot_idx,
                "rank": rank,
                "stage_idx": rank,
                "timestep": float(t.item()),
                "forward_ms": first.elapsed_time(last),
                "stage_start_ms": origin.elapsed_time(first),
                "stage_end_ms": origin.elapsed_time(last),
                "comm_wait_ms": 0.0,
                "input_latent_frames": model_input.shape[2],
            }
            if kv:
                record.update(
                    pass_kind="clean_kv" if clean_pass else "denoise",
                    kv_sources=[list(source) for source in kv_sources[task]],
                    kv_history_latent_frames=len(kv_sources[task]) * t_l,
                )
            records.append(record)
            if rank != last_rank:
                forward_payload = {**prediction.tensors, "model_input": model_input}
            else:
                prediction = prediction[:, :, -t_l:]
                sample = model_input[:, :, -t_l:]
                clean = sample if clean_pass else predict_clean(prediction, sample, t)
                if kv and not clean_pass:
                    clean = clean.to(prediction.dtype)
                if step < last_denoise:
                    noise = (
                        step_noises[step][:, :, chunk * t_l : (chunk + 1) * t_l].contiguous()
                        if kv
                        else torch.randn(shape, generator=generators[chunk], device=device, dtype=torch.float32)
                    )
                    updated = add_noise(clean, noise, timesteps[step + 1])
                else:
                    updated = clean
                feedback_payload = {"latents": updated}
                if world == 1:
                    accept_feedback(task, feedback_payload)

        exchange_ms = 0.0
        if world > 1:
            exchange_start = time.perf_counter()
            handles = []
            postprocessors = []
            received_feedback = received_forward = None
            # Same order on every rank: last→0 feedback, then i→i+1 forward.
            if tasks[last_rank] is not None:
                if rank == last_rank:
                    feedback_bytes += _tensor_bytes({"latents": feedback_payload["latents"]})
                    handles.extend(pp.isend_tensor_dict(feedback_payload, dst=0))
                elif rank == 0:
                    received_feedback, work, postprocess = pp.irecv_tensor_dict(src=last_rank)
                    postprocessors.extend(postprocess)
                    handles.extend(work)
            for src in range(last_rank):
                dst = src + 1
                if tasks[src] is None:
                    continue
                if rank == src:
                    activation_bytes += _tensor_bytes(
                        {key: value for key, value in forward_payload.items() if key != "model_input"}
                    )
                    sample_bytes += _tensor_bytes({"model_input": forward_payload["model_input"]})
                    handles.extend(pp.isend_tensor_dict(forward_payload, dst=dst))
                elif rank == dst:
                    received_forward, work, postprocess = pp.irecv_tensor_dict(src=src)
                    postprocessors.extend(postprocess)
                    handles.extend(work)
            for handle in handles:
                handle.wait()
            for postprocess in postprocessors:
                postprocess()
            stream.synchronize()
            if received_feedback is not None:
                accept_feedback(tasks[last_rank], received_feedback)
            if received_forward is not None:
                stage_input = received_forward
            exchange_ms = (time.perf_counter() - exchange_start) * 1000
            comm_ms += exchange_ms
        if record is not None:
            record["comm_wait_ms"] = exchange_ms
        slot_records.append({"slot_idx": slot_idx, "task": task, "comm_wait_ms": exchange_ms})
    torch.accelerator.synchronize(device)
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
        "activation_bytes_sent": activation_bytes,
        "sample_bytes_sent": sample_bytes,
        "feedback_bytes_sent": feedback_bytes,
        "total_payload_bytes_sent": activation_bytes + sample_bytes + feedback_bytes,
        "denoise_peak_allocated_bytes": torch.accelerator.max_memory_allocated(device),
    }
    if kv:
        payload["kv_resident_peak_versions"] = kv_peak_versions
    all_metrics = [payload]
    if world > 1:
        all_metrics = [None] * world
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
        "world_size": world,
        "chunks": chunks,
        "chunk_latent_frames": t_l,
        "seed": seed,
        "latent_shape": list(shape),
        "ranks": all_metrics,
        "forward_call_unit": "layer_stage" if world > 1 else "whole_transformer",
        "stage_time_basis": "rank_local_cuda_event_origin; clocks are not aligned across ranks",
        "activation_bytes_include_model_input": False,
        "denoise_wall_ms": max(item["denoise_wall_ms"] for item in all_metrics),
        "forward_total_ms": sum(row["forward_ms"] for item in all_metrics for row in item["steps"]),
        "comm_wait_rank_sum_ms": sum(item["comm_wait_ms"] for item in all_metrics),
        "activation_bytes_sent": sum(item["activation_bytes_sent"] for item in all_metrics),
        "sample_bytes_sent": sum(item["sample_bytes_sent"] for item in all_metrics),
        "feedback_bytes_sent": sum(item["feedback_bytes_sent"] for item in all_metrics),
        "total_payload_bytes_sent": sum(item["total_payload_bytes_sent"] for item in all_metrics),
        "latent_gather_ms": (time.perf_counter() - gather_start) * 1000,
    }
    if kv:
        metrics.update(
            conditioning="per_layer_clean_kv" if schedule == "serial" else "per_layer_latest_kv",
            kv_history_chunks=kv_history_chunks,
            kv_version_policy=(
                "serial concatenates completed clean KV; stepwise concatenates latest completed before slot"
            ),
            kv_eviction="last_consumer_execution_slot",
            kv_resident_peak_versions=max(item["kv_resident_peak_versions"] for item in all_metrics),
            clean_kv_forward=True,
            clean_kv_forward_total_ms=sum(
                row["forward_ms"] for item in all_metrics for row in item["steps"] if row["pass_kind"] == "clean_kv"
            ),
            noise_contract="shared full-video initial sample and one re-noise tensor per step transition",
        )
    return result, metrics
