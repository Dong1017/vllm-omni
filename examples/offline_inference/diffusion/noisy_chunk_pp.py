# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark one whole-request noisy chunk pipeline through Omni.generate.

Run from this checkout under the GPU scheduler, for example::

    PYTHONPATH=/data/dxw/noisy_chunk_pp/vllm-omni gpu run --gpus 2 \
        --timeout 30m --note omni-chunk-pp -- python \
        examples/offline_inference/diffusion/noisy_chunk_pp.py \
        --pp-size 2 --chunks 6 --mode lag1 --out /data/dxw/noisy_chunk_pp/omni_pp2

The default saves a latent tensor for same-algorithm PP=1 versus PP=2 parity.
With --decode, it saves a video instead. Worker CHUNK_PP_METRICS log records
contain denoising/forward/communication timings; metadata.json records the
client request wall time, including response transport but excluding saving.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pp-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument(
        "--cond-frames", type=int, choices=range(1, 21), default=4, help="Condition prefix length in latent frames."
    )
    parser.add_argument("--mode", choices=("lag1", "lag2"), default="lag1")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--decode", action="store_true", help="Return decoded np video instead of latents.")
    args = parser.parse_args()
    if args.chunks < 1:
        parser.error("--chunks must be positive")
    return args


def main():
    args = parse_args()
    # Workers spawned by Omni must import this checkout as well as the client.
    checkout = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(checkout))
    existing_paths = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(checkout) + (os.pathsep + existing_paths if existing_paths else "")

    import numpy as np
    import torch

    import vllm_omni
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput

    module_path = Path(vllm_omni.__file__).resolve()
    if not module_path.is_relative_to(checkout):
        raise RuntimeError(f"Expected vllm_omni from {checkout}, imported {module_path}")

    args.out.mkdir(parents=True, exist_ok=True)
    chunk_frames, chunk_latent_frames = 77, 20
    num_frames = args.chunks * chunk_latent_frames * 4 - 3
    output_type = "np" if args.decode else "latent"
    prompt = "A cinematic drone shot over snowy mountains at golden hour"
    model = "FastVideo/FastWan2.2-TI2V-5B-Diffusers"
    extra_args = {
        "chunk_frames": chunk_frames,
        "chunk_cond_frames": args.cond_frames,
        "chunk_lag": int(args.mode[-1]),
    }
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
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkout": str(checkout),
        "checkout_head": subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip(),
        "vllm_omni_import": str(module_path),
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model": model,
        "prompt": prompt,
        "pipeline_parallel_size": args.pp_size,
        "pipeline_parallel_mode": "chunk",
        "enforce_eager": True,
        "chunks": args.chunks,
        "height": 480,
        "width": 832,
        "num_frames": num_frames,
        "num_inference_steps": 3,
        "seed": 1024,
        "guidance_scale": 1.0,
        "output_type": output_type,
        "extra_args": extra_args,
        "timing_scope": {
            "omni_init_wall_ms": "Omni constructor, including worker/model initialization",
            "generate_request_wall_ms": "One Omni.generate call through complete response; excludes artifact saving",
            "worker_metrics": "CHUNK_PP_METRICS JSON in worker log",
        },
    }
    print("CHUNK_PP_CLIENT_CONFIG " + json.dumps(metadata), flush=True)
    init_start = time.perf_counter()
    omni = Omni(
        model=model,
        pipeline_parallel_size=args.pp_size,
        pipeline_parallel_mode="chunk",
        enforce_eager=True,
    )
    metadata["omni_init_wall_ms"] = (time.perf_counter() - init_start) * 1000
    try:
        request_start = time.perf_counter()
        outputs = omni.generate(
            {"prompt": prompt, "negative_prompt": "", "modalities": ["video"]},
            sampling,
            use_tqdm=False,
        )
        metadata["generate_request_wall_ms"] = (time.perf_counter() - request_start) * 1000
        if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], OmniRequestOutput):
            raise TypeError(f"Expected one OmniRequestOutput, received {type(outputs)}")
        result = outputs[0]
        metadata["request_id"] = result.request_id
        metadata["final_output_type"] = result.final_output_type
        metadata["stage_durations"] = result.stage_durations
        metadata["worker_peak_memory_mb"] = result.peak_memory_mb
        # Wan's latent postprocessor can carry its tensor in images; .latents
        # is also used for diffusion trajectory data and may remain None.
        payload = result.images
        if not args.decode and result.latents is not None:
            payload = result.latents
        while isinstance(payload, list) and len(payload) == 1:
            payload = payload[0]

        if args.decode:
            from diffusers.utils import export_to_video

            frames = np.asarray(payload)
            if frames.ndim == 5 and frames.shape[0] == 1:
                frames = frames[0]
            expected_shape = (num_frames, 480, 832, 3)
            if frames.shape != expected_shape:
                raise ValueError(f"Expected decoded video {expected_shape}, received {frames.shape}")
            artifact_path = args.out / "video.mp4"
            export_to_video(list(frames), str(artifact_path), fps=16)
            metadata.update(artifact_shape=list(frames.shape), artifact_dtype=str(frames.dtype), fps=16)
        else:
            if not isinstance(payload, torch.Tensor):
                raise TypeError(f"Expected latent Tensor, received {type(payload)}")
            if payload.ndim == 4:
                payload = payload.unsqueeze(0)
            expected_shape = (1, 48, args.chunks * chunk_latent_frames, 30, 52)
            if tuple(payload.shape) != expected_shape:
                raise ValueError(f"Expected full latent {expected_shape}, received {tuple(payload.shape)}")
            latents = payload.detach().to(device="cpu", dtype=torch.float32).contiguous()
            artifact_path = args.out / "latents.pt"
            torch.save(latents, artifact_path)
            metadata.update(artifact_shape=list(latents.shape), artifact_dtype=str(latents.dtype))
        metadata["artifact"] = str(artifact_path.resolve())
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print("CHUNK_PP_CLIENT_METRICS " + json.dumps(metadata), flush=True)
    finally:
        omni.close()


if __name__ == "__main__":
    main()
