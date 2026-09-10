# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Whole-request noisy chunk pipelining for a replicated three-step DiT.

Each producer exchanges its condition at the end of its slot, including
conditions for chunks which have not started. This avoids a blocking send
waiting on a future chunk. Model layer partitioning is deliberately separate.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.parallel_state import get_pp_group


def slot_task(slot: int, rank: int, chunks: int, world: int):
    local_slot = slot - (rank if world == 2 else 0)
    if local_slot < 0:
        return None
    chunk = rank + world * (local_slot // 3)
    return (chunk, local_slot % 3) if chunk < chunks else None


@torch.inference_mode()
def run_noisy_chunk_pipeline(
    *,
    predict_noise: Callable,
    scheduler,
    timesteps: torch.Tensor,
    shape: tuple[int, ...],
    chunks: int,
    cond_frames: int,
    lag: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Return concatenated clean latents on PP rank zero and JSON metrics.

    One rank is the same-algorithm serial reference. Two ranks alternate chunks.
    Condition transport is bf16; scheduler state and final output remain fp32.
        The callable accepts (latent_input, timestep, temporal_offset, step_idx).
    """
    pp = get_pp_group()
    rank, world = pp.rank_in_group, pp.world_size
    if world not in (1, 2) or len(timesteps) != 3:
        raise ValueError("Noisy chunk pipeline requires one or two ranks and three DMD steps")
    if chunks < 1 or lag not in (1, 2) or not 1 <= cond_frames <= shape[2]:
        raise ValueError("Invalid chunk count, condition prefix length, or lag")
    group = pp.device_group
    t_l, length = shape[2], cond_frames
    cond_shape = (shape[0], shape[1], length, shape[3], shape[4])
    slots = max(r + len(range(r, chunks, world)) * 3 for r in range(world))
    cache, clean_chunks, records, slot_records = {}, {}, [], []
    comm_ms, sent_bytes = 0.0, 0

    def barrier():
        if world == 2:
            dist.barrier(group=group)

    torch.accelerator.synchronize(device)
    barrier()
    start_wall = time.perf_counter()
    for slot in range(slots):
        task = slot_task(slot, rank, chunks, world)
        peer_task = slot_task(slot, 1 - rank, chunks, world) if world == 2 else None
        send_tensor = None
        record = None
        if task is not None:
            chunk, step = task
            t = timesteps[step]
            if step == 0:
                generator = torch.Generator(device=device).manual_seed(seed + chunk * 100003)
                current = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
            cond = None
            if chunk > 0 and step >= lag:
                cond = cache.pop((chunk - 1, step - lag)).float()
                if lag == 2:
                    cond_generator = torch.Generator(device=device).manual_seed(
                        seed + 1000000007 + chunk * 100003 + step * 1009
                    )
                    noise = torch.randn(cond.shape, generator=cond_generator, device=device, dtype=torch.float32)
                    cond = scheduler.add_noise(cond, noise, t)
            model_input = torch.cat([cond, current], dim=2) if cond is not None else current
            offset = chunk * t_l - (length if cond is not None else 0)
            first, last = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            first.record()
            prediction = predict_noise(model_input, t.expand(shape[0]), offset, step)
            last.record()
            last.synchronize()
            record = {
                "chunk_idx": chunk,
                "step_idx": step,
                "slot_idx": slot,
                "rank": rank,
                "timestep": float(t.item()),
                "forward_ms": first.elapsed_time(last),
                "comm_wait_ms": 0.0,
                "input_latent_frames": model_input.shape[2],
            }
            records.append(record)
            if cond is not None:
                prediction = prediction[:, :, length:]
            clean = scheduler.predict_clean(prediction, current, t)
            if step < 2:
                noise = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
                current = scheduler.add_noise(clean, noise, timesteps[step + 1])
            else:
                current = clean
                clean_chunks[chunk] = clean
            if step + lag < 3 and chunk + 1 < chunks:
                source = current if lag == 1 else clean
                send_tensor = source[:, :, -length:].to(torch.bfloat16).contiguous()

        recv_tensor, recv_key = None, None
        if peer_task is not None:
            peer_chunk, peer_step = peer_task
            if peer_step + lag < 3 and peer_chunk + 1 < chunks:
                recv_key = peer_task
                recv_tensor = torch.empty(cond_shape, device=device, dtype=torch.bfloat16)
        exchange_ms = 0.0
        if world == 1:
            if send_tensor is not None:
                cache[task] = send_tensor
        else:
            peer = pp.ranks[1 - rank]
            ops = []
            if recv_tensor is not None:
                ops.append(dist.P2POp(dist.irecv, recv_tensor, peer, group))
            if send_tensor is not None:
                ops.append(dist.P2POp(dist.isend, send_tensor, peer, group))
                sent_bytes += send_tensor.numel() * send_tensor.element_size()
            if ops:
                exchange_start = time.perf_counter()
                work = dist.batch_isend_irecv(ops)
                for request in work:
                    request.wait()
                torch.cuda.current_stream(device).synchronize()
                exchange_ms = (time.perf_counter() - exchange_start) * 1000
                comm_ms += exchange_ms
                if recv_tensor is not None:
                    cache[recv_key] = recv_tensor
        if record is not None:
            record["comm_wait_ms"] = exchange_ms
        slot_records.append({"slot_idx": slot, "task": task, "comm_wait_ms": exchange_ms})
    if cache:
        raise RuntimeError(f"Unconsumed chunk conditions: {list(cache)}")
    torch.accelerator.synchronize(device)
    barrier()
    local_wall = (time.perf_counter() - start_wall) * 1000
    payload = {
        "rank": rank,
        "steps": records,
        "slots": slot_records,
        "denoise_wall_ms": local_wall,
        "comm_wait_ms": comm_ms,
        "condition_bytes_sent": sent_bytes,
        "denoise_peak_allocated_bytes": torch.accelerator.max_memory_allocated(device),
    }
    all_metrics = [payload]
    if world == 2:
        all_metrics = [None, None]
        dist.all_gather_object(all_metrics, payload, group=pp.cpu_group)

    gather_start = time.perf_counter()
    full_shape = (*shape[:2], t_l * chunks, *shape[3:])
    if rank == 0:
        ordered = []
        for chunk in range(chunks):
            if chunk in clean_chunks:
                ordered.append(clean_chunks[chunk])
            else:
                received = torch.empty(shape, device=device, dtype=torch.float32)
                work = dist.batch_isend_irecv([dist.P2POp(dist.irecv, received, pp.ranks[1], group)])
                for request in work:
                    request.wait()
                ordered.append(received)
        result = torch.cat(ordered, dim=2)
    else:
        for chunk in range(rank, chunks, world):
            outgoing = clean_chunks[chunk].contiguous()
            work = dist.batch_isend_irecv([dist.P2POp(dist.isend, outgoing, pp.ranks[0], group)])
            for request in work:
                request.wait()
            torch.cuda.current_stream(device).synchronize()
        # Only rank zero publishes output/decodes; preserve the existing PP
        # pipeline's tensor contract on the non-output rank.
        result = torch.zeros(full_shape, device=device, dtype=torch.float32)
    torch.accelerator.synchronize(device)
    metrics = {
        "execution": "whole_request_chunk_pipeline",
        "world_size": world,
        "chunks": chunks,
        "chunk_latent_frames": t_l,
        "cond_frames": length,
        "lag": lag,
        "seed": seed,
        "latent_shape": list(shape),
        "ranks": all_metrics,
        "denoise_wall_ms": max(item["denoise_wall_ms"] for item in all_metrics),
        "forward_total_ms": sum(row["forward_ms"] for item in all_metrics for row in item["steps"]),
        "comm_wait_rank_sum_ms": sum(item["comm_wait_ms"] for item in all_metrics),
        "condition_bytes_sent": sum(item["condition_bytes_sent"] for item in all_metrics),
        "latent_gather_ms": (time.perf_counter() - gather_start) * 1000,
    }
    return result, metrics
