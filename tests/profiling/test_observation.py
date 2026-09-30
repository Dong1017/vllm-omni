# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Observation 层测试（M3，ckpt-3 修订 + MVP gate Fix 1/2）：
# 确定性提取、metric 级 provenance（P1-1）、跨 metric 比值的同源机械校验、
# no_evidence / no_salient 拆分（P1-1b）、通信命名纪律（P1-2）、与 diagnosis 解耦（P1-3）。

import pytest

from vllm_omni.profiling.observation import build_observations
from vllm_omni.profiling.schema import OptimizationEvidence

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _refs(*keys):
    """按 metric key 造 metric_evidence；每个 key 一条独立记录。"""
    return {k: [f"run1:ev_{i + 1:06d}"] for i, k in enumerate(keys)}


def test_exposed_gap_observation_binds_exact_metric_provenance():
    # gap 与 wall 同 source（同一 union 记录）时才允许比值
    ev = OptimizationEvidence()
    ev.workload.wall_ms = 100.0
    ev.timeline.exposed_non_device_busy_ms = 30.0
    refs = _refs("workload.wall_ms")
    refs["timeline.exposed_non_device_busy_ms"] = refs["workload.wall_ms"]
    ev.metric_evidence = refs
    obs = build_observations(ev)
    gap = next(o for o in obs if o.kind == "exposed_gap_present")
    assert gap.metric_key == "timeline.exposed_non_device_busy_ms"
    assert gap.value == pytest.approx(0.30)
    assert gap.evidence_ids == ["run1:ev_000001"]


def test_exposed_gap_cross_source_suppressed():
    # MVP gate Fix 1：gap 与 wall 不同 source -> 不计算比值，观察被抑制
    ev = OptimizationEvidence()
    ev.workload.wall_ms = 100.0
    ev.timeline.exposed_non_device_busy_ms = 30.0
    ev.metric_evidence = _refs("workload.wall_ms", "timeline.exposed_non_device_busy_ms")
    assert all(o.kind != "exposed_gap_present" for o in build_observations(ev))


def test_device_busy_dominant_cross_source_suppressed():
    # MVP gate Fix 1：busy（kernel_details）与 wall（step_trace）不同 source ->
    # 不得产出 device_busy_dominant（真实 Ascend 输出的修正点）
    ev = OptimizationEvidence()
    ev.run.backend = "ascend"
    ev.workload.wall_ms = 100.0
    ev.timeline.device_busy_ms = 90.0
    ev.metric_evidence = _refs("workload.wall_ms", "timeline.device_busy_ms")
    assert all(o.kind != "device_busy_dominant" for o in build_observations(ev))


def test_device_busy_dominant_same_source_fires():
    ev = OptimizationEvidence()
    ev.workload.wall_ms = 100.0
    ev.timeline.device_busy_ms = 90.0
    refs = _refs("workload.wall_ms")
    refs["timeline.device_busy_ms"] = refs["workload.wall_ms"]
    ev.metric_evidence = refs
    assert any(o.kind == "device_busy_dominant" for o in build_observations(ev))


def test_runtime_launch_heavy_observation_replaces_bound_claim():
    # P0-2：launch 占比高 -> 观察层 runtime_launch_heavy，不再产出 host_dispatch_bound
    ev = OptimizationEvidence()
    ev.workload.wall_ms = 100.0
    ev.runtime.api_summed_ms = 50.0
    ev.runtime.launch_summed_ms = 30.0
    ev.runtime.other_summed_ms = 20.0
    ev.metric_evidence = _refs("runtime.api_summed_ms", "runtime.launch_summed_ms", "runtime.other_summed_ms")
    obs = build_observations(ev)
    heavy = [o for o in obs if o.kind == "runtime_launch_heavy"]
    assert len(heavy) == 1
    assert heavy[0].value == pytest.approx(0.6)
    assert "temporal correlation" in heavy[0].note  # 指明升级需要 M4 时间相关性
    assert any(o.kind == "runtime_other_heavy" for o in obs)


def test_compute_activity_dominant_requires_same_source():
    # P0-3：compute 活动主导是观察；compute_bound 保留给 M4 硬件证据。
    # busy/compute/wall 全部同 source（真实 CUDA adapter 的绑定方式）
    ev = OptimizationEvidence()
    ev.workload.wall_ms = 100.0
    ev.timeline.device_busy_ms = 90.0
    ev.timeline.compute_ms = 85.0
    refs = _refs("workload.wall_ms")
    refs["timeline.device_busy_ms"] = refs["workload.wall_ms"]
    refs["timeline.compute_ms"] = refs["workload.wall_ms"]
    ev.metric_evidence = refs
    obs = build_observations(ev)
    comp = next(o for o in obs if o.kind == "compute_activity_dominant")
    assert comp.value == pytest.approx(85.0 / 90.0)
    assert "M4 hardware evidence" in comp.note  # 指明升级路径
    # compute 跨 source -> 观察降级为 device_busy_dominant（busy/wall 仍同源）
    ev.metric_evidence["timeline.compute_ms"] = ["run1:ev_000099"]
    obs2 = build_observations(ev)
    assert all(o.kind != "compute_activity_dominant" for o in obs2)
    assert any(o.kind == "device_busy_dominant" for o in obs2)


def test_communication_exposed_cites_all_inputs():
    # P0-4.1：derived 未重叠通信必须引用 total + overlap + wall 全部输入；
    # 三个输入同 source（真实 step_trace 绑定方式）
    ev = OptimizationEvidence()
    ev.workload.wall_ms = 100.0
    ev.communication.total_ms = 35.0
    ev.communication.overlap_ms = 5.0
    refs = _refs("workload.wall_ms")
    for k in ("communication.total_ms", "communication.overlap_ms"):
        refs[k] = refs["workload.wall_ms"]
    ev.metric_evidence = refs
    obs = build_observations(ev)
    comm = next(o for o in obs if o.kind == "communication_exposed")
    assert comm.value == pytest.approx(0.30)
    assert set(comm.evidence_ids) == {"run1:ev_000001"}


def test_communication_exposed_cross_source_suppressed():
    # MVP gate Fix 2：overlap 与 wall/total 不同 source -> 不得计算 exposed 占比
    ev = OptimizationEvidence()
    ev.workload.wall_ms = 100.0
    ev.communication.total_ms = 35.0
    ev.communication.overlap_ms = 5.0
    ev.metric_evidence = _refs(
        "communication.total_ms", "communication.overlap_ms", "workload.wall_ms"
    )  # 三个输入各一条记录 -> 同源校验失败
    obs = build_observations(ev)
    assert all(o.kind != "communication_exposed" for o in obs)
    breakdown = next(o for o in obs if o.kind == "communication_present_no_overlap_breakdown")
    assert "not same-source" in breakdown.note


def test_communication_without_breakdown_is_not_called_exposed():
    # P1-2：分解不可得 -> 不得称 communication_exposed
    ev = OptimizationEvidence()
    ev.workload.wall_ms = 100.0
    ev.communication.total_ms = 40.0
    ev.metric_evidence = _refs("communication.total_ms", "workload.wall_ms")
    obs = build_observations(ev)
    comm = next(o for o in obs if o.kind == "communication_present_no_overlap_breakdown")
    assert comm.value is None
    assert all(o.kind != "communication_exposed" for o in obs)


def test_no_evidence_vs_no_salient_split():
    # P1-1b：两种"不足"是不同的观察
    empty = build_observations(OptimizationEvidence())
    assert [o.kind for o in empty] == ["no_evidence_available"]
    ev = OptimizationEvidence()
    ev.workload.wall_ms = 100.0
    ev.metric_evidence = _refs("workload.wall_ms")
    salient = build_observations(ev)
    assert [o.kind for o in salient] == ["no_salient_observation"]


def test_no_diagnosis_vocabulary_in_observations():
    ev = OptimizationEvidence()
    ev.workload.wall_ms = 100.0
    ev.timeline.exposed_non_device_busy_ms = 50.0
    ev.runtime.api_summed_ms = 50.0
    ev.runtime.launch_summed_ms = 50.0
    ev.timeline.device_busy_ms = 90.0
    ev.timeline.compute_ms = 88.0
    refs = _refs("workload.wall_ms", "runtime.api_summed_ms", "runtime.launch_summed_ms")
    refs["timeline.exposed_non_device_busy_ms"] = refs["workload.wall_ms"]
    refs["timeline.device_busy_ms"] = refs["workload.wall_ms"]
    refs["timeline.compute_ms"] = refs["workload.wall_ms"]
    ev.metric_evidence = refs
    for o in build_observations(ev):
        assert "_bound" not in o.kind
        assert "_bound" not in o.note


def test_observations_deterministic():
    ev = OptimizationEvidence()
    ev.workload.wall_ms = 100.0
    ev.timeline.exposed_non_device_busy_ms = 30.0
    refs = _refs("workload.wall_ms")
    refs["timeline.exposed_non_device_busy_ms"] = refs["workload.wall_ms"]
    ev.metric_evidence = refs
    assert build_observations(ev) == build_observations(ev)
