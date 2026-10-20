# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.models.physis.common import build_physis_cell_plan, validate_physis_request
from vllm_omni.experimental.ar_diffusion.cell_schedule import CellSchedule, Ordering, build_cell_plan
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize("frames", [81, 121, 162])
def test_fixed_recipe_accepts_verified_frame_counts_and_optional_defaults(frames):
    params = OmniDiffusionSamplingParams(num_frames=frames, seed=17)
    validate_physis_request(params, num_frames=frames)
    assert params.seed == 17


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("height", 512),
        ("width", 1024),
        ("num_frames", 1),
        ("num_inference_steps", 50),
        ("guidance_scale", 7.0),
        ("guidance_scale_2", 3.0),
        ("num_outputs_per_prompt", 2),
        ("do_classifier_free_guidance", True),
        ("sigmas", [1.0, 0.0]),
    ],
)
def test_rejects_explicit_unsupported_sampling_settings(name, value):
    params = OmniDiffusionSamplingParams(num_frames=81)
    setattr(params, name, value)
    with pytest.raises(ValueError):
        validate_physis_request(params, num_frames=81)


def test_ltx_model_frame_rate_is_separate_from_output_playback_rate():
    params = OmniDiffusionSamplingParams(num_frames=121, fps=16.0, frame_rate=24.0)
    validate_physis_request(params, num_frames=121, frame_rate=24.0)
    params.frame_rate = 16.0
    with pytest.raises(ValueError, match="frame_rate=24"):
        validate_physis_request(params, num_frames=121, frame_rate=24.0)


def test_optional_fixed_settings_can_be_unspecified():
    params = OmniDiffusionSamplingParams(num_frames=None)
    validate_physis_request(params, num_frames=81)


def test_explicit_recipe_is_accepted():
    params = OmniDiffusionSamplingParams(
        height=480,
        width=832,
        num_frames=81,
        num_inference_steps=4,
        guidance_scale=1.0,
        num_outputs_per_prompt=1,
        seed=0,
    )
    validate_physis_request(params, num_frames=81)


@pytest.mark.parametrize("chunks", [19, 29, 14, 48])
@pytest.mark.parametrize(("stages", "groups"), [(1, 1), (1, 2), (4, 1)])
def test_shared_plan_preserves_model_schedule_and_task_order(chunks, stages, groups):
    plan = build_physis_cell_plan(
        chunks=chunks, stages=stages, layer_groups=groups, blocks=8, blocks_per_rank=8 // groups
    )
    expected = build_cell_plan(
        CellSchedule(
            chunks=chunks - 1,
            num_denoise_steps=4,
            stages=stages,
            layer_groups=groups,
            ordering=Ordering.SAME_STEP,
            kv_history_chunks=18,
            blocks=8,
            blocks_per_rank=8 // groups,
            clean_pass=False,
        )
    )
    assert plan.schedule == expected.schedule
    assert plan.num_slots == expected.num_slots
    assert all(
        plan.task(slot, rank) == expected.task(slot, rank)
        for slot in range(plan.num_slots)
        for rank in range(plan.world)
    )
