#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental WaveServe Wan 1.3B serial-vs-vertical benchmark (not a recipe).

Runs the same ``chunks x denoise steps`` workload through the experimental
``WaveServeWanPipeline`` in two stage regimes:

* ``serial``: ``S = 1``, ``G = pp_world`` — layer splitting only, one chunk at a
  time (the pre-existing chunk+layer PP behaviour).
* ``vertical``: ``S = T+1``, ``G = 1`` — one denoise stage per rank, chunk
  pipelined across stages with Latest-KV.

Per-slot CPU wall time is measured on rank 0 across every stage rank, so the
number is the schedule's *rank-slot* time rather than an end-to-end request
latency (the engine admits one request at a time on this path).

Example (6 GPUs):

    torchrun --standalone --nproc-per-node=6 tools/ar_diffusion_waveserve_bench.py \
        --model /data/models/waveserve-wan2.1-1.3b-diffusers-rf-dev \
        --chunks 7 --denoise-steps 4 --warmup 1 --repeat 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "Physis-AI/waveserve-wan2.1-1.3b-diffusers-rf-dev"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--chunks", type=int, default=7)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--history", type=int, default=6, help="Latest-KV history chunks (vertical only).")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--regimes", default="serial,vertical")
    parser.add_argument(
        "--gpus-per-stage",
        type=int,
        default=1,
        help="Layer groups G inside a stage (vertical) / PP world (serial).",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    return parser.parse_args()


def _slot_times(runner) -> list[float]:
    """Read and clear the per-slot wall times recorded by the stage executor."""
    from vllm_omni.experimental.ar_diffusion import stage_executor as se

    times = list(se.SLOT_TIMES)
    se.SLOT_TIMES.clear()
    return times


def _run_regime(args: argparse.Namespace, regime: str, world: int) -> dict:
    from vllm_omni import Omni
    from vllm_omni.experimental.ar_diffusion import stage_executor as se
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    if regime == "serial":
        stages, layer_groups, history = 1, world, 0
    elif regime == "vertical":
        stages, layer_groups, history = args.denoise_steps + 1, args.gpus_per_stage, args.history
    else:
        raise ValueError(f"unknown regime {regime!r}")

    if stages * layer_groups != world:
        raise SystemExit(
            f"regime={regime} needs stages*layer_groups == world; got {stages}*{layer_groups} != {world}"
        )

    stage_config = {
        "stage_parallel_size": stages,
        "max_batch_size": 1,
        "max_history_chunks": max(1, history or 1),
    }
    engine_args = {
        "model": args.model,
        "model_config": {"ar_diffusion_stage_config": stage_config},
        "parallel_config": {"pipeline_parallel_size": world},
        "enforce_eager": bool(args.enforce_eager),
        "max_num_seqs": 1,
    }

    se.SLOT_TIMES.clear()
    omni = Omni(**engine_args)
    try:
        totals: list[float] = []
        slots: list[float] = []
        meta: dict = {}
        for index in range(args.warmup + args.repeat):
            _slot_times(None)  # clear before each measured request
            params = OmniDiffusionSamplingParams(
                extra_args={
                    "num_chunks": args.chunks,
                    "num_denoise_steps": args.denoise_steps,
                    "kv_history_chunks": history,
                    "chunk_schedule": "serial",
                    "reset": True,
                }
            )
            started = time.perf_counter()
            outputs = omni.generate("a cat walking on grass", sampling_params_list=[params])
            elapsed = time.perf_counter() - started
            recorded = _slot_times(None)
            if not outputs:
                raise RuntimeError(f"regime={regime} produced no output")
            if index < args.warmup:
                continue
            totals.append(elapsed)
            slots.extend(recorded)
            if not meta:
                result = outputs[0]
                meta = {
                    "stage_id": getattr(result, "stage_id", None),
                    "finished": bool(getattr(result, "finished", False)),
                    "metrics": getattr(result, "metrics", None) or {},
                }
    finally:
        omni.close()

    if not totals:
        raise SystemExit(f"regime={regime} collected no measurement (warmup={args.warmup})")

    per_slot = statistics.median(slots) * 1000.0 if slots else float("nan")
    schedule_slots = _expected_slots(args.chunks, args.denoise_steps, stages, layer_groups, history)
    return {
        "regime": regime,
        "stages": stages,
        "layer_groups": layer_groups,
        "history": history,
        "measured_requests": len(totals),
        "slot_samples": len(slots),
        "median_total_s": round(statistics.median(totals), 4),
        "min_total_s": round(min(totals), 4),
        "median_slot_ms": round(per_slot, 3),
        "schedule_slots": schedule_slots,
        "schedule_gpu_slots": schedule_slots * world,
        "schedule_gpu_slot_ms": round(per_slot * schedule_slots * world, 1),
        "metrics": meta.get("metrics", {}),
    }


def _expected_slots(
    chunks: int,
    denoise_steps: int,
    stages: int,
    layer_groups: int,
    history: int,
) -> int:
    from vllm_omni.experimental.ar_diffusion.stage_schedule import (
        Ordering,
        StageSchedule,
        build_stage_plan,
    )

    schedule = StageSchedule(
        chunks=chunks,
        num_denoise_steps=denoise_steps,
        stages=stages,
        layer_groups=layer_groups,
        ordering=Ordering.SERIAL,
        kv_history_chunks=max(1, history) if stages > 1 else history,
    )
    return build_stage_plan(schedule).num_slots


def main() -> None:
    args = _parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    regimes = [item.strip() for item in args.regimes.split(",") if item.strip()]

    results = []
    for regime in regimes:
        results.append(_run_regime(args, regime, world))

    if int(os.environ.get("RANK", "0")) == 0:
        print("REGIME       S  G  slots  slot_ms  gpu_slots  gpu_slot_ms  wall_s  min_wall_s")
        for row in results:
            print(
                f"{row['regime']:<10} {row['stages']:>2} {row['layer_groups']:>2} "
                f"{row['schedule_slots']:>6} {row['median_slot_ms']:>8.3f} "
                f"{row['schedule_gpu_slots']:>9} {row['schedule_gpu_slot_ms']:>11.1f} "
                f"{row['median_total_s']:>7.3f} {row['min_total_s']:>10.3f}"
            )
        if len(results) >= 2:
            base = results[0]
            for row in results[1:]:
                speedup = base["median_total_s"] / row["median_total_s"] if row["median_total_s"] else float("nan")
                slot_speedup = base["median_slot_ms"] / row["median_slot_ms"] if row["median_slot_ms"] else float("nan")
                print(
                    f"{row['regime']} vs {base['regime']}: "
                    f"end_to_end x{speedup:.2f}, per_slot x{slot_speedup:.2f}"
                )
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps({"world": world, "runs": results}, indent=2))
            print(f"WROTE_JSON={args.json_out}")


if __name__ == "__main__":
    main()
