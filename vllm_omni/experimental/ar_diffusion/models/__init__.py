# SPDX-License-Identifier: Apache-2.0

__all__ = ["WaveServeWanPipeline", "register_experimental_diffusion_models"]


def register_experimental_diffusion_models() -> None:
    """Register WaveServe without editing production ``registry.py``."""
    try:
        from vllm_omni.diffusion.registry import register_diffusion_model
    except Exception:
        return
    module = "vllm_omni.experimental.ar_diffusion.models.waveserve_wan.pipeline_waveserve_wan"
    for arch in ("WaveServeWanPipeline", "WaveServeWanRFPipeline"):
        register_diffusion_model(arch, module, "WaveServeWanPipeline")


register_experimental_diffusion_models()


def __getattr__(name: str):
    if name == "WaveServeWanPipeline":
        from vllm_omni.experimental.ar_diffusion.models.waveserve_wan.pipeline_waveserve_wan import (
            WaveServeWanPipeline,
        )

        return WaveServeWanPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
