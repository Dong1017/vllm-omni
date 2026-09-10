# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark layer-partitioned full / serial / stepwise video generation.

An instance runs one complete video warmup, three complete video measurements,
then latent-only correctness requests. Run this program through ``gpu run``.
Only the last measured video is saved; artifact writing is outside timing.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", choices=("full", "serial", "stepwise"), required=True)
    parser.add_argument("--gap", type=int, choices=(1, 2), default=1)
    parser.add_argument("--pp-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--cond-frames", type=int, choices=range(1, 21), default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.chunks < 1:
        parser.error("--chunks must be positive")
    return args


def write_metadata(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    checkout = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(checkout))
    old_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(checkout) + (os.pathsep + old_pythonpath if old_pythonpath else "")

    import numpy as np
    import torch
    from diffusers.utils import export_to_video

    import vllm_omni
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput

    module_path = Path(vllm_omni.__file__).resolve()
    if not module_path.is_relative_to(checkout):
        raise RuntimeError(f"Expected vllm_omni from {checkout}, imported {module_path}")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"Use a fresh experiment directory; {metadata_path} already exists")

    chunk_frames, chunk_latent_frames = 77, 20
    num_frames = args.chunks * chunk_latent_frames * 4 - 3
    mode = "layer" if args.execution == "full" else "chunk"
    schedule = "serial" if args.execution == "full" else args.execution
    model = "FastVideo/FastWan2.2-TI2V-5B-Diffusers"
    prompt = "A cinematic drone shot over snowy mountains at golden hour"
    run_id = uuid.uuid4().hex
    metadata = {
        "schema_version": 2,
        "status": "initializing",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkout": str(checkout),
        "checkout_head": subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip(),
        "vllm_omni_import": str(module_path),
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model": model,
        "prompt": prompt,
        "execution": args.execution,
        "gap": 0 if args.execution == "full" else args.gap,
        "pipeline_parallel_size": args.pp_size,
        "pipeline_parallel_mode": mode,
        "enforce_eager": True,
        "chunks": args.chunks,
        "chunk_latent_frames": chunk_latent_frames,
        "cond_frames": 0 if args.execution == "full" else args.cond_frames,
        "total_latent_frames": args.chunks * chunk_latent_frames,
        "height": 480,
        "width": 832,
        "num_frames": num_frames,
        "num_inference_steps": 3,
        "seed": 1024,
        "guidance_scale": 1.0,
        "warmup_iterations": 1,
        "timed_iterations": 3,
        "iterations": [],
        "artifacts": {},
        "timing_scope": {
            "omni_init_wall_ms": "Omni constructor, including automatic engine initialization/warmup",
            "generate_request_wall_ms": "One complete Omni.generate response; excludes artifact saving",
            "performance": "Median of exactly three phase=timed video requests following one complete video warmup",
            "validation": "All latent-only requests excluded from performance statistics",
            "worker_metrics": "CHUNK_PP_METRICS matched by unique experiment_iteration",
        },
    }
    write_metadata(metadata_path, metadata)
    print("LAYER_STEP_CLIENT_CONFIG " + json.dumps(metadata), flush=True)
    omni = None

    def request(phase, index, output_type, request_schedule):
        iteration = f"{run_id}:{phase}:{index}"
        extra_args = {
            "chunk_frames": chunk_frames,
            "chunk_cond_frames": args.cond_frames,
            "chunk_gap": args.gap,
            "chunk_schedule": request_schedule,
            "collect_pp_metrics": True,
            "experiment_iteration": iteration,
        }
        # New params / extra_args per call; no generator or request state reused.
        sampling = OmniDiffusionSamplingParams(
            height=480,
            width=832,
            num_frames=num_frames,
            num_inference_steps=3,
            seed=1024,
            guidance_scale=1.0,
            output_type=output_type,
            extra_args=extra_args,
        )
        record = {
            "experiment_iteration": iteration,
            "phase": phase,
            "index": index,
            "execution": args.execution,
            "schedule": "full" if args.execution == "full" else request_schedule,
            "gap": 0 if args.execution == "full" else args.gap,
            "output_type": output_type,
            "extra_args": dict(extra_args),
            "status": "running",
        }
        metadata["iterations"].append(record)
        write_metadata(metadata_path, metadata)
        print("LAYER_STEP_ITERATION_START " + json.dumps(record), flush=True)
        start = time.perf_counter()
        outputs = omni.generate(
            {"prompt": prompt, "negative_prompt": "", "modalities": ["video"]}, sampling, use_tqdm=False
        )
        record["generate_request_wall_ms"] = (time.perf_counter() - start) * 1000
        if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], OmniRequestOutput):
            raise TypeError(f"Expected one OmniRequestOutput, received {type(outputs)}")
        result = outputs[0]
        record.update(
            request_id=result.request_id,
            final_output_type=result.final_output_type,
            stage_durations=result.stage_durations,
            worker_peak_memory_mb=result.peak_memory_mb,
        )
        payload = result.latents if output_type == "latent" and result.latents is not None else result.images
        while isinstance(payload, list) and len(payload) == 1:
            payload = payload[0]
        if output_type == "np":
            payload = np.asarray(payload)
            if payload.ndim == 5 and payload.shape[0] == 1:
                payload = payload[0]
            expected = (num_frames, 480, 832, 3)
            if payload.shape != expected:
                raise ValueError(f"Expected video {expected}, received {payload.shape}")
        else:
            if not isinstance(payload, torch.Tensor):
                raise TypeError(f"Expected latent Tensor, received {type(payload)}")
            if payload.ndim == 4:
                payload = payload.unsqueeze(0)
            expected = (1, 48, args.chunks * chunk_latent_frames, 30, 52)
            if tuple(payload.shape) != expected:
                raise ValueError(f"Expected latent {expected}, received {tuple(payload.shape)}")
        record.update(status="complete", output_shape=list(payload.shape), output_dtype=str(payload.dtype))
        write_metadata(metadata_path, metadata)
        print("LAYER_STEP_ITERATION_COMPLETE " + json.dumps(record), flush=True)
        return payload, record

    try:
        start = time.perf_counter()
        omni = Omni(model=model, pipeline_parallel_size=args.pp_size, pipeline_parallel_mode=mode, enforce_eager=True)
        metadata["omni_init_wall_ms"] = (time.perf_counter() - start) * 1000
        metadata["status"] = "running"
        write_metadata(metadata_path, metadata)
        warmup, _ = request("warmup", 0, "np", schedule)
        del warmup
        for index in range(metadata["timed_iterations"]):
            frames, record = request("timed", index, "np", schedule)
            if index == metadata["timed_iterations"] - 1:
                path = args.out / "video.mp4"
                export_to_video(list(frames), str(path), fps=16)
                metadata["artifacts"]["video"] = {
                    "path": str(path.resolve()),
                    "experiment_iteration": record["experiment_iteration"],
                    "fps": 16,
                }
                write_metadata(metadata_path, metadata)
            del frames

        validations = [("latent_validation", schedule, "latents.pt", "latents")]
        if args.execution == "stepwise" and args.gap == 2:
            validations.append(("serial_latent_validation", "serial", "latents_serial_validation.pt", "serial_latents"))
        for phase, request_schedule, filename, artifact_key in validations:
            latent, record = request(phase, 0, "latent", request_schedule)
            saved = latent.detach().to(device="cpu", dtype=torch.float32).contiguous()
            path = args.out / filename
            torch.save(saved, path)
            metadata["artifacts"][artifact_key] = {
                "path": str(path.resolve()),
                "experiment_iteration": record["experiment_iteration"],
                "source_dtype": str(latent.dtype),
                "saved_dtype": str(saved.dtype),
                "shape": list(saved.shape),
            }
            write_metadata(metadata_path, metadata)
            del latent, saved
        metadata["status"] = "complete"
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_metadata(metadata_path, metadata)
        print("LAYER_STEP_CLIENT_COMPLETE " + json.dumps(metadata), flush=True)
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        write_metadata(metadata_path, metadata)
        raise
    finally:
        if omni is not None:
            omni.close()


if __name__ == "__main__":
    main()
