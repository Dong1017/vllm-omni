#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental WaveServe Wan 1.3B serial-vs-latest benchmark (not a recipe).

Same topology ``S = T+1``, ``G``; only chunk schedule differs:

* ``serial``: one chunk finishes every cell before the next starts.
* ``latest``: diagonal Latest-KV pipeline.

Performance stats match the CausalWan / noisy_chunk_pp harness:

* ``request_ms`` — wall around one ``Omni.generate`` (excludes artifact I/O).
* ``dit_ms`` — ``stage_gen_time_ms`` from the request metrics (DiT/denoise wall).
* Aggregation — median over ``--repeat`` timed requests (no warmup discard).

Engine startup dummy is skipped via ``WaveServeWanPipeline.dummy_run_num_frames = 0``
(required for ``S = T+1``); that is not a measurement warmup.

Single-process only (mp diffusion executor). Do not wrap in torchrun.

    python tools/ar_diffusion_waveserve_bench.py \
        --model /data/models/waveserve-wan2.1-1.3b-diffusers-rf-dev \
        --world-size 4 --chunks 7 --denoise-steps 3 --repeat 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "Physis-AI/waveserve-wan2.1-1.3b-diffusers-rf-dev"

_DEPLOY_YAML = """\
pipeline: wan2_2_ti2v
async_chunk: false
distributed_executor_backend: mp
dtype: bfloat16

stages:
  - stage_id: 0
    max_num_seqs: 1
    enforce_eager: true
    model_class_name: WaveServeWanPipeline
    engine_backend: vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine
    parallel_config:
      pipeline_parallel_size: {world}
    model_config:
      ar_diffusion_stage_config:
        stage_parallel_size: {stages}
        max_batch_size: 1
        max_history_chunks: {history}
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Total GPUs (mp executor). Default: denoise_steps + 1.",
    )
    parser.add_argument("--chunks", type=int, default=7)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--history", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=3, help="Timed requests per regime (all counted).")
    parser.add_argument("--regimes", default="serial,latest")
    parser.add_argument("--gpus-per-stage", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _dit_ms(output) -> float | None:
    metrics = getattr(output, "metrics", None) or {}
    stage_metrics = metrics.get("stage_metrics") or {}
    if not stage_metrics:
        # Fallback: some paths expose a flat stage_gen_time_ms.
        flat = metrics.get("stage_gen_time_ms")
        if flat is not None:
            return float(flat)
        return None
    first = next(iter(stage_metrics.values()))
    ms = float((first or {}).get("stage_gen_time_ms") or 0.0)
    return ms if ms > 0 else None


def _run_regime(args: argparse.Namespace, regime: str, world: int) -> dict:
    from vllm_omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    stages = args.denoise_steps + 1
    layer_groups = args.gpus_per_stage
    history = args.history
    if regime == "serial":
        chunk_schedule = "serial"
    elif regime == "latest":
        chunk_schedule = "latest"
    else:
        raise ValueError(f"unknown regime {regime!r}; expected serial or latest")

    if stages * layer_groups != world:
        raise SystemExit(
            f"regime={regime} needs stages*layer_groups == world; got {stages}*{layer_groups} != {world}"
        )

    deploy_yaml = _DEPLOY_YAML.format(world=world, stages=stages, history=max(1, history))
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix=f"waveserve_{regime}_", delete=False) as fh:
        fh.write(deploy_yaml)
        deploy_path = fh.name

    omni = Omni(model=args.model, deploy_config=deploy_path)
    try:
        request_ms: list[float] = []
        dit_ms: list[float] = []
        for index in range(args.repeat):
            params = OmniDiffusionSamplingParams(
                extra_args={
                    "num_chunks": args.chunks,
                    "num_denoise_steps": args.denoise_steps,
                    "kv_history_chunks": history,
                    "chunk_schedule": chunk_schedule,
                    "reset": True,
                }
            )
            started = time.perf_counter()
            outputs = omni.generate("a cat walking on grass", sampling_params_list=[params])
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if not outputs:
                raise RuntimeError(f"regime={regime} produced no output")
            request_ms.append(elapsed_ms)
            gen = _dit_ms(outputs[0])
            if gen is None:
                raise RuntimeError(f"regime={regime} request {index} missing stage_gen_time_ms")
            dit_ms.append(gen)
    finally:
        omni.close()

    schedule_slots = _expected_slots(args.chunks, args.denoise_steps, stages, layer_groups, history, chunk_schedule)
    return {
        "regime": regime,
        "stages": stages,
        "layer_groups": layer_groups,
        "history": history,
        "chunk_schedule": chunk_schedule,
        "schedule_slots": schedule_slots,
        "timed_iterations": len(request_ms),
        "request_ms": round(_median(request_ms), 3),
        "dit_ms": round(_median(dit_ms), 3),
        "min_request_ms": round(min(request_ms), 3),
        "min_dit_ms": round(min(dit_ms), 3),
        "all_request_ms": [round(x, 3) for x in request_ms],
        "all_dit_ms": [round(x, 3) for x in dit_ms],
        "performance": {
            "request_ms": round(_median(request_ms), 3),
            "dit_ms": round(_median(dit_ms), 3),
        },
    }


def _expected_slots(
    chunks: int,
    denoise_steps: int,
    stages: int,
    layer_groups: int,
    history: int,
    chunk_schedule: str,
) -> int:
    from vllm_omni.experimental.ar_diffusion.stage_schedule import (
        Ordering,
        StageSchedule,
        build_stage_plan,
    )

    ordering = Ordering.SERIAL if chunk_schedule == "serial" else Ordering.INTERLEAVED
    schedule = StageSchedule(
        chunks=chunks,
        num_denoise_steps=denoise_steps,
        stages=stages,
        layer_groups=layer_groups,
        ordering=ordering,
        kv_history_chunks=max(1, history),
    )
    return build_stage_plan(schedule).num_slots


def main() -> None:
    args = _parse_args()
    world = args.world_size or (args.denoise_steps + 1)
    regimes = [item.strip() for item in args.regimes.split(",") if item.strip()]

    results = [_run_regime(args, regime, world) for regime in regimes]

    print("REGIME       S  G  slots  request_ms    dit_ms  min_req  min_dit")
    for row in results:
        print(
            f"{row['regime']:<10} {row['stages']:>2} {row['layer_groups']:>2} "
            f"{row['schedule_slots']:>6} {row['request_ms']:>11.1f} "
            f"{row['dit_ms']:>9.1f} {row['min_request_ms']:>8.1f} {row['min_dit_ms']:>8.1f}"
        )

    by_name = {row["regime"]: row for row in results}
    comparisons: dict[str, dict[str, float]] = {}
    if "serial" in by_name and "latest" in by_name:
        serial, latest = by_name["serial"], by_name["latest"]
        comparisons["latest_vs_serial"] = {
            "request_ms_speedup": round(serial["request_ms"] / latest["request_ms"], 3),
            "dit_ms_speedup": round(serial["dit_ms"] / latest["dit_ms"], 3),
        }
        print(
            "latest vs serial: "
            f"request x{comparisons['latest_vs_serial']['request_ms_speedup']:.2f}, "
            f"dit x{comparisons['latest_vs_serial']['dit_ms_speedup']:.2f}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "world": world,
            "chunks": args.chunks,
            "denoise_steps": args.denoise_steps,
            "timed_iterations": args.repeat,
            "aggregation": "median over timed requests (no warmup)",
            "runs": results,
            "comparisons": comparisons,
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"WROTE_JSON={args.json_out}")


if __name__ == "__main__":
    main()
