# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Noisy chunk scheduling over the existing one- or two-stage layer pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.distributed as dist
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.distributed.parallel_state import get_pp_group

ChunkStep = tuple[int, int]


def plan_chunk_pipeline(
    chunks: int, gap: int, world: int, schedule: str = "stepwise"
) -> list[tuple[ChunkStep | None, ...]]:
    """Plan stage work using only results completed before the current slot."""
    if chunks < 1 or gap not in (1, 2) or world not in (1, 2):
        raise ValueError("Chunk pipeline requires positive chunks, gap 1/2 and one/two stages")
    if schedule == "serial":
        jobs = [(chunk, step) for chunk in range(chunks) for step in range(3)]
    elif schedule == "stepwise":
        jobs = [
            (chunk, step)
            for first in range(0, chunks, 2)
            for step in range(3)
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
            if chunk > 0 and step >= gap:
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


def _tensor_bytes(payload: dict[str, torch.Tensor] | None) -> int:
    return sum(value.numel() * value.element_size() for value in payload.values()) if payload else 0


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
) -> tuple[torch.Tensor, dict]:
    """Run local layers and exchange activations/state in a common direction order.

    The callback accepts (model_input, timestep, temporal_offset, step_idx,
    intermediate_tensors=None). It returns IntermediateTensors on the first of
    two stages, and a noise-prediction Tensor on the last stage. Each chunk has
    its own RNG; only the last stage draws the two scheduler re-noising samples.
    """
    pp = get_pp_group()
    rank, world = pp.rank_in_group, pp.world_size
    slots = plan_chunk_pipeline(chunks, gap, world, schedule)
    if len(timesteps) != 3 or not 1 <= cond_frames <= shape[2]:
        raise ValueError("Chunk pipeline requires three DMD steps and a valid condition prefix")
    t_l, length = shape[2], cond_frames
    current, generators, cache, clean_chunks = {}, {}, {}, {}
    records, slot_records = [], []
    comm_ms = 0.0
    activation_bytes = sample_bytes = feedback_bytes = condition_bytes = 0
    stage_input = None

    def has_consumer(task):
        chunk, step = task
        return chunk + 1 < chunks and step + gap < 3

    def accept_feedback(task, payload):
        chunk, step = task
        current[chunk] = payload["latents"]
        if step == 2:
            clean_chunks[chunk] = current[chunk]
        if has_consumer(task):
            source = current[chunk][:, :, -length:] if gap == 1 else payload["condition"]
            cache[task] = source.to(torch.bfloat16).contiguous()

    def initialize_chunk(chunk):
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
            t = timesteps[step]
            if step == 0:
                initialize_chunk(chunk)
            has_prefix = chunk > 0 and step >= gap
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
            prediction = predict_noise(model_input, t.expand(shape[0]), offset, step, intermediate_tensors=intermediate)
            last.record(stream)
            last.synchronize()
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
            records.append(record)
            if world == 2 and rank == 0:
                forward_payload = {**prediction.tensors, "model_input": model_input}
            else:
                prediction = prediction[:, :, -t_l:]
                clean = scheduler.predict_clean(prediction, model_input[:, :, -t_l:], t)
                if step < 2:
                    noise = torch.randn(shape, generator=generators[chunk], device=device, dtype=torch.float32)
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
                    handles.extend(pp.isend_tensor_dict(feedback_payload, dst=0))
                else:
                    received_feedback, work, postprocess = pp.irecv_tensor_dict(src=1)
                    postprocessors.extend(postprocess)
                    handles.extend(work)
            if tasks[0] is not None:
                if rank == 0:
                    activation_bytes += _tensor_bytes(
                        {key: value for key, value in forward_payload.items() if key != "model_input"}
                    )
                    sample_bytes += _tensor_bytes({"model_input": forward_payload["model_input"]})
                    handles.extend(pp.isend_tensor_dict(forward_payload, dst=1))
                else:
                    received_forward, work, postprocess = pp.irecv_tensor_dict(src=0)
                    postprocessors.extend(postprocess)
                    handles.extend(work)
            # Keep all outgoing and incoming payloads alive until transfer ends.
            for handle in handles:
                handle.wait()
            for postprocess in postprocessors:
                postprocess()
            stream.synchronize()
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
        "condition_bytes_sent": condition_bytes,
        "activation_bytes_sent": activation_bytes,
        "sample_bytes_sent": sample_bytes,
        "feedback_bytes_sent": feedback_bytes,
        "total_payload_bytes_sent": activation_bytes + sample_bytes + feedback_bytes + condition_bytes,
        "denoise_peak_allocated_bytes": torch.accelerator.max_memory_allocated(device),
    }
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
    return result, metrics
