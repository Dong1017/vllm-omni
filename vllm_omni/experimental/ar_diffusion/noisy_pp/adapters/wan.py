# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wan-family adapter: CausalWan DMD sample-update contract + tower routing.

Migrated from the noisy-chunk-pp branch's ``causal_dmd.py`` and
``pipeline_wan2_2._dit_router``. The sample-update math mirrors FastVideo's
``CausalDMDDenosingStage`` and ``SelfForcingFlowMatchScheduler``
(hao-ai-lab/FastVideo); the public euler scheduler keeps its upstream
behavior -- these are model-adapter-side functions over its sigma table.
"""

from __future__ import annotations

import torch

__all__ = [
    "sigma_for_timestep",
    "predict_clean",
    "predict_boundary_state",
    "renoise_low",
    "renoise_high",
    "DmdUpdateKind",
    "classify_update",
    "update_sample",
    "WanNoisyPPAdapter",
]

# -- Sigma math (reference-aligned) -------------------------------------------


def sigma_for_timestep(
    scheduler,
    timestep: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the sigma table entry nearest to ``timestep`` (0-dim tensor)."""
    timesteps = scheduler.timesteps.detach().to(device=device or scheduler.timesteps.device, dtype=dtype)
    sigmas = scheduler.sigmas.detach().to(device=device or scheduler.timesteps.device, dtype=dtype)
    index = torch.argmin(torch.abs(timesteps - float(timestep)))
    return sigmas[index]


def predict_clean(
    scheduler, prediction: torch.Tensor, sample: torch.Tensor, timestep: float
) -> torch.Tensor:
    """x_0 = x_t - sigma_t * v, computed in float64 like the reference."""
    sigma = sigma_for_timestep(scheduler, timestep, device=sample.device)
    return (sample.to(torch.float64) - sigma * prediction.to(torch.float64)).to(prediction.dtype)


def predict_boundary_state(
    scheduler, prediction: torch.Tensor, sample: torch.Tensor, timestep: float, boundary_timestep: float
) -> torch.Tensor:
    """x_b = x_t - (sigma_t - sigma_b) * v: the ODE trajectory at the
    boundary timestep. Skipping re-noising does not turn x_0 into x_b."""
    sigma_t = sigma_for_timestep(scheduler, timestep, device=sample.device)
    sigma_b = sigma_for_timestep(scheduler, boundary_timestep, device=sample.device)
    return (sample.to(torch.float64) - (sigma_t - sigma_b) * prediction.to(torch.float64)).to(prediction.dtype)


def renoise_low(scheduler, clean: torch.Tensor, noise: torch.Tensor, timestep: float) -> torch.Tensor:
    """x' = (1 - sigma) * clean + sigma * noise at a low-noise timestep."""
    sigma = sigma_for_timestep(scheduler, timestep, device=noise.device)
    return ((1.0 - sigma) * clean.to(torch.float64) + sigma * noise.to(torch.float64)).to(noise.dtype)


def renoise_high(
    scheduler, clean: torch.Tensor, noise: torch.Tensor, timestep: float, boundary_timestep: float
) -> torch.Tensor:
    """x' = alpha * clean + beta * noise with alpha=(1-sigma)/(1-sigma_b),
    beta=sqrt(sigma^2 - (alpha*sigma_b)^2): the boundary-referenced
    covariance the high expert was distilled for."""
    sigma = sigma_for_timestep(scheduler, timestep, device=noise.device)
    sigma_b = sigma_for_timestep(scheduler, boundary_timestep, device=noise.device)
    alpha = (1.0 - sigma) / (1.0 - sigma_b)
    beta = torch.sqrt(sigma * sigma - (alpha * sigma_b) ** 2)
    return (alpha * clean.to(torch.float64) + beta * noise.to(torch.float64)).to(noise.dtype)


class DmdUpdateKind:
    """Which sample update a step uses, per the reference denoising loop."""

    HIGH_MIDDLE = "high_middle"  # high expert, next step still high: renoise_high
    HIGH_LAST = "high_last"  # high expert's final step: hand the boundary state through unchanged
    LOW_MIDDLE = "low_middle"  # low expert, next step still low: renoise_low
    LOW_LAST = "low_last"  # low expert's final step: emit the clean prediction


def classify_update(*, timestep: float, next_timestep: float | None, boundary_timestep: float | None) -> str:
    """Classify a step's update kind from its timestep and the next one.

    ``boundary_timestep=None`` means a single-expert checkpoint: every step
    is low. The handoff step is the last high step when a boundary exists.
    """
    if next_timestep is None:
        return DmdUpdateKind.LOW_LAST
    is_high = boundary_timestep is not None and timestep >= boundary_timestep
    next_is_high = boundary_timestep is not None and next_timestep >= boundary_timestep
    if is_high:
        return DmdUpdateKind.HIGH_MIDDLE if next_is_high else DmdUpdateKind.HIGH_LAST
    return DmdUpdateKind.LOW_MIDDLE


def update_sample(
    scheduler,
    *,
    prediction: torch.Tensor,
    sample: torch.Tensor,
    timestep: float,
    next_timestep: float | None,
    noise: torch.Tensor | None,
    boundary_timestep: float | None,
) -> torch.Tensor:
    """The minimal model-side sample-update interface shared by native and
    chunk paths. ``noise`` is required exactly when the kind re-noises."""
    kind = classify_update(timestep=timestep, next_timestep=next_timestep, boundary_timestep=boundary_timestep)
    if kind == DmdUpdateKind.HIGH_LAST:
        if boundary_timestep is None:
            raise RuntimeError("Boundary handoff requires a boundary timestep")
        return predict_boundary_state(scheduler, prediction, sample, timestep, boundary_timestep)
    if kind == DmdUpdateKind.LOW_LAST:
        return predict_clean(scheduler, prediction, sample, timestep)
    if noise is None:
        raise RuntimeError(f"{kind} requires a re-noising tensor")
    clean = predict_clean(scheduler, prediction, sample, timestep)
    if kind == DmdUpdateKind.HIGH_MIDDLE:
        if boundary_timestep is None:
            raise RuntimeError(f"{kind} requires a boundary timestep")
        return renoise_high(scheduler, clean, noise, next_timestep, boundary_timestep)
    return renoise_low(scheduler, clean, noise, next_timestep)


# -- Adapter ------------------------------------------------------------------


class WanNoisyPPAdapter:
    """Declares the wan family's noisy-PP shape to the three layers.

    Producers: the dual-expert checkpoints route denoise steps through a
    high-noise tower and a low-noise tower (boundary 0.875 by default);
    each tower publishes and consumes its own KV. Single-tower checkpoints
    (FastWan) collapse to one producer.
    """

    HIGH = "high"
    LOW = "low"
    SINGLE = "single"

    def __init__(self, *, dual_tower: bool, boundary_timestep: float | None) -> None:
        self.dual_tower = dual_tower
        self.boundary_timestep = boundary_timestep
        if dual_tower and boundary_timestep is None:
            raise ValueError("dual-tower wan checkpoints require a boundary timestep")

    @property
    def producers(self) -> tuple[str, ...]:
        return (self.HIGH, self.LOW) if self.dual_tower else (self.SINGLE,)

    def reader_producer(self, step: int, timestep: float) -> str:
        """Which producer's KV a denoise step at ``timestep`` consumes.

        Mirrors the reference ``_get_kv_cache(t)``: t >= boundary reads the
        high tower's cache, otherwise the low tower's.
        """
        if not self.dual_tower:
            return self.SINGLE
        return self.HIGH if timestep >= self.boundary_timestep else self.LOW  # type: ignore[operator]

    def publisher_producers(self, kind: str) -> tuple[str, ...]:
        """Who publishes one version of ``kind``.

        Denoise steps publish on their own tower only; the clean/context
        pass runs on every tower (the reference refreshes both caches).
        """
        if kind == "clean":
            return self.producers
        return self.producers  # runner selects the reading tower's producer

    # Storage binding points (land with runtime.py):
    #
    # * forward hooks: ARDiffusionKVState.prepare_paged_context(branch) /
    #   commit_paged_context(branch) with branch = manager.branch_for(...)
    # * layout: wan's per-layer post-RoPE K/V mapped onto the paged pool via
    #   kv_cache.paged.pool_write_chunk (reuse; do not hand-roll)
    # * attention: ARDiffusionPagedForwardContext supplies the windowed
    #   K/V to wan attention's kv_context entry point.
