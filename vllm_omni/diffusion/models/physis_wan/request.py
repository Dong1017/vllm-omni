# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def validate_physis_request(
    params: OmniDiffusionSamplingParams,
    *,
    num_frames: int,
    frame_rate: float | None = None,
) -> None:
    expected = {
        "height": 480,
        "width": 832,
        "num_frames": num_frames,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "guidance_scale_2": 1.0,
        "num_outputs_per_prompt": 1,
    }
    if frame_rate is not None:
        expected["frame_rate"] = frame_rate
    for name, required in expected.items():
        value = getattr(params, name)
        if value is not None and value != required:
            raise ValueError(f"Physis requires {name}={required}; received {value}")
    if params.do_classifier_free_guidance:
        raise ValueError("Physis checkpoints are guidance distilled and do not support CFG")
    if params.timesteps is not None or params.sigmas is not None:
        raise ValueError("Physis uses its trained four-step timestep and sigma schedule")
