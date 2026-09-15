# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark layer-partitioned full / serial / stepwise video generation.

An instance runs timed complete-video measurements, then optional latent-only
correctness requests. Run this program through ``gpu run``.
Only the last measured video is saved; artifact writing is outside timing.
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # full = one temporal chunk (whole clip); serial/stepwise = multi-chunk Wan DMD path.
    parser.add_argument(
        "--execution",
        choices=("full", "serial", "stepwise"),
        required=True,
        help="full: whole-clip path (chunks=1). serial/stepwise: multi-chunk Latest KV over layer PP.",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=2,
        help="pipeline_parallel_size: number of DiT layer stages / GPUs.",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=6,
        help="Number of temporal chunks for serial/stepwise (full forces 1).",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=77,
        help="Pixel frames per chunk; must be >=5 and 1 mod 4 (Wan VAE temporal scale).",
    )
    parser.add_argument(
        "--kv-history-chunks",
        type=int,
        default=6,
        help="serial/stepwise: max past chunks whose KV may be attended (H). Ignored for full.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Fresh experiment directory for metadata.json, videos, and frames.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSON with samples[{id,prompt,seed}]; default is a single smoke prompt.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timed complete-video measurement requests.")
    parser.add_argument(
        "--model",
        default="FastVideo/FastWan2.2-TI2V-5B-Diffusers",
        help="Wan DMD Diffusers ID or local path (chunk PP requires is_dmd).",
    )
    parser.add_argument(
        "--skip-latent-validation",
        action="store_true",
        help="Skip post-benchmark latent-only correctness requests.",
    )
    args = parser.parse_args()
    if args.chunks < 1:
        parser.error("--chunks must be positive")
    if args.pp_size < 1:
        parser.error("--pp-size must be >= 1")
    if args.kv_history_chunks < 1:
        parser.error("--kv-history-chunks must be >= 1")
    if args.chunk_frames < 5 or (args.chunk_frames - 1) % 4:
        parser.error("--chunk-frames must be 1 modulo 4 and at least 5")
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

    chunk_frames = args.chunk_frames
    chunk_latent_frames = (chunk_frames - 1) // 4 + 1
    num_frames = args.chunks * chunk_latent_frames * 4 - 3
    # full = one temporal chunk covering the whole clip; serial/stepwise cut by chunk_frames.
    schedule_chunks = 1 if args.execution == "full" else args.chunks
    request_chunk_frames = num_frames if args.execution == "full" else chunk_frames
    schedule = "serial" if args.execution == "full" else args.execution
    model = args.model
    prompt = "A cinematic drone shot over snowy mountains at golden hour"
    samples = (
        json.loads(args.manifest.read_text())["samples"]
        if args.manifest
        else [{"id": "smoke", "prompt": prompt, "seed": 1024}]
    )
    seed = 1024
    sample_id = "smoke"
    run_id = uuid.uuid4().hex
    metadata = {
        "status": "initializing",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "execution": args.execution,
        "pipeline_parallel_size": args.pp_size,
        "chunks": schedule_chunks,
        "chunk_frames": request_chunk_frames,
        "num_frames": num_frames,
        "kv_history_chunks": None if args.execution == "full" else args.kv_history_chunks,
        "repeats": args.repeats,
        "samples": samples,
        "iterations": [],
        "artifacts": {},
    }
    write_metadata(metadata_path, metadata)
    print("LAYER_STEP_CLIENT_CONFIG " + json.dumps(metadata), flush=True)
    omni = None

    def request(phase, index, output_type, request_schedule):
        iteration = f"{run_id}:{sample_id}:{phase}:{index}"
        extra_args = {
            "collect_pp_metrics": True,
            "experiment_iteration": iteration,
        }
        if args.execution != "full":
            extra_args.update(
                chunk_frames=request_chunk_frames,
                chunk_schedule=request_schedule,
                chunk_conditioning="latest_kv",
                kv_history_chunks=args.kv_history_chunks,
            )
        # New params / extra_args per call; no generator or request state reused.
        sampling = OmniDiffusionSamplingParams(
            height=480,
            width=832,
            num_frames=num_frames,
            num_inference_steps=3,
            seed=seed,
            guidance_scale=1.0,
            output_type=output_type,
            extra_args=extra_args,
        )
        record = {
            "experiment_iteration": iteration,
            "sample_id": sample_id,
            "phase": phase,
            "index": index,
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
            expected_t = args.chunks * chunk_latent_frames
            if payload.ndim != 5 or payload.shape[0] != 1 or payload.shape[2] != expected_t:
                raise ValueError(
                    f"Expected latent (1, C, {expected_t}, H, W), received {tuple(payload.shape)}"
                )
        record.update(status="complete", output_shape=list(payload.shape))
        write_metadata(metadata_path, metadata)
        print("LAYER_STEP_ITERATION_COMPLETE " + json.dumps(record), flush=True)
        return payload, record

    try:
        start = time.perf_counter()
        omni = Omni(
            model=model,
            pipeline_parallel_size=args.pp_size,
            enforce_eager=True,
        )
        metadata["omni_init_wall_ms"] = (time.perf_counter() - start) * 1000
        metadata["status"] = "running"
        write_metadata(metadata_path, metadata)
        for sample in samples:
            prompt, seed, sample_id = sample["prompt"], sample["seed"], sample["id"]
            sample_out = args.out / sample_id if args.manifest else args.out
            sample_out.mkdir(exist_ok=True)
            for index in range(args.repeats):
                frames, record = request("timed", index, "np", schedule)
                if index == args.repeats - 1:
                    path = sample_out / "video.mp4"
                    rgb = np.rint(np.clip(frames, 0, 1) * 255).astype(np.uint8)
                    np.save(sample_out / "frames.npy", rgb)
                    # NumPy frames passed to export_to_video must stay in [0, 1];
                    # the helper performs the uint8 scaling itself.
                    export_to_video(list(frames), str(path), fps=16)
                    del rgb
                    metadata["artifacts"][f"{sample_id}:video"] = {
                        "path": str(path.resolve()),
                        "experiment_iteration": record["experiment_iteration"],
                    }
                    write_metadata(metadata_path, metadata)
                del frames

            if not args.skip_latent_validation:
                latent, record = request("latent_validation", 0, "latent", schedule)
                saved = latent.detach().to(device="cpu", dtype=torch.float32).contiguous()
                path = sample_out / "latents.pt"
                torch.save(saved, path)
                metadata["artifacts"][f"{sample_id}:latents"] = {
                    "path": str(path.resolve()),
                    "experiment_iteration": record["experiment_iteration"],
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
