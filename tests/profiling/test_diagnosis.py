# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# 诊断测试（M3，ckpt-3 修订版）。修订后验收：
#   - 唯一支持的 *_bound 候选是 communication_bound（有 overlap 分解 + 同 source + 双阈值）；
#   - host/sync/allocation/compute 的"重"只到观察层，不产出 *_bound；
#   - memory_bound / load_imbalance 永不伪造；
#   - 证据不足 -> unknown(low) 是合法输出。
# 质量指标是"避免错误诊断"，不是覆盖的诊断类别数。

import pytest

from vllm_omni.profiling.analysis import enrich, rank_hotspots
from vllm_omni.profiling.diagnosis import diagnose
from vllm_omni.profiling.observation import build_observations
from vllm_omni.profiling.schema import OperatorEvidence, OptimizationEvidence

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _ev(backend: str, refs_by_key: dict[str, list[str] | str]) -> OptimizationEvidence:
    """值可以是完整全局引用字符串或其列表；统一规范为 list[str]。"""
    ev = OptimizationEvidence()
    ev.run.backend = backend
    ev.metric_evidence = {k: (v if isinstance(v, list) else [v]) for k, v in refs_by_key.items()}
    return ev


def _refs(*keys):
    return {k: [f"run1:ev_{i + 1:06d}"] for i, k in enumerate(keys)}


def test_case1_runtime_heavy_stops_at_observation():
    """案例 1（修订）：runtime-heavy 产生带 exact provenance 的观察，但不再产出 *_bound 候选。"""
    ev = _ev(
        "cuda",
        _refs(
            "workload.wall_ms",
            "timeline.exposed_non_device_busy_ms",
            "runtime.api_summed_ms",
            "runtime.launch_summed_ms",
        ),
    )
    ev.workload.wall_ms = 100.0
    ev.timeline.exposed_non_device_busy_ms = 40.0
    ev.runtime.api_summed_ms = 60.0
    ev.runtime.launch_summed_ms = 50.0
    obs = build_observations(ev)
    launch = next(o for o in obs if o.kind == "runtime_launch_heavy")
    assert launch.value == pytest.approx(50.0 / 60.0)
    assert launch.evidence_ids  # exact metric provenance
    candidates = diagnose(ev)
    assert all(c.class_ not in ("host_dispatch_bound", "synchronization_bound", "allocation_bound") for c in candidates)


def test_case2_communication_heavy_without_overlap_decomposition_does_not_fire():
    """案例 2：communication-heavy 不自动等于 communication_bound。"""
    ev = _ev("cuda", _refs("workload.wall_ms", "timeline.exposed_non_device_busy_ms", "communication.total_ms"))
    ev.workload.wall_ms = 100.0
    ev.timeline.exposed_non_device_busy_ms = 40.0
    ev.communication.total_ms = 35.0  # 无 overlap 分解
    candidates = diagnose(ev)
    assert all(c.class_ != "communication_bound" for c in candidates)
    # 观察层如实记录分解不可得
    assert any(o.kind == "communication_present_no_overlap_breakdown" for o in build_observations(ev))


def test_case2b_communication_bound_fires_with_decomposition_and_same_source():
    """唯一支持的诊断：有 overlap 分解 + 同 source + 双阈值 -> communication_bound。"""
    # 同 source：四个输入都绑定同一条 step_trace 记录（adapter 的真实绑定方式）
    refs = _refs("workload.wall_ms")
    refs["timeline.exposed_non_device_busy_ms"] = refs["workload.wall_ms"]
    refs["communication.total_ms"] = refs["workload.wall_ms"]
    refs["communication.overlap_ms"] = refs["workload.wall_ms"]
    ev = _ev("ascend", refs)
    ev.workload.wall_ms = 100.0
    ev.timeline.exposed_non_device_busy_ms = 30.0
    ev.communication.total_ms = 35.0
    ev.communication.overlap_ms = 5.0
    candidates = diagnose(ev)
    cand = next(c for c in candidates if c.class_ == "communication_bound")
    assert cand.confidence in ("medium", "high")
    # 候选引用观察携带的全部输入（total+overlap+wall，P0-4.1）
    assert set(cand.evidence_ids) == {r for v in refs.values() for r in v}


def test_case2c_communication_rule_enforces_same_source_mechanically():
    """P0-4.2：输入来自不同 source -> 机械拒绝，不依赖测试纪律。"""
    refs = _refs("workload.wall_ms", "timeline.exposed_non_device_busy_ms", "communication.total_ms")
    refs["communication.overlap_ms"] = ["run1:ev_000099"]  # overlap 来自不同 source 记录
    ev = _ev("ascend", refs)
    ev.workload.wall_ms = 100.0
    ev.timeline.exposed_non_device_busy_ms = 30.0
    ev.communication.total_ms = 35.0
    ev.communication.overlap_ms = 5.0
    assert all(c.class_ != "communication_bound" for c in diagnose(ev))


def test_case3_insufficient_evidence_stays_unknown():
    """案例 3：证据不足 -> 单个 unknown 候选，confidence=low，不编造。"""
    candidates = diagnose(_ev("cuda", {}))
    assert len(candidates) == 1
    assert candidates[0].class_ == "unknown"
    assert candidates[0].confidence == "low"


def test_case4_inclusive_time_never_enters_hotspot_fraction():
    """案例 4：framework inclusive（397s > wall）不得进入 hotspot fraction。"""
    ev = _ev("ascend", _refs("workload.wall_ms"))
    ev.operators.append(
        OperatorEvidence(
            id="op_000001",
            name="diffusion_forward",
            layer="framework",
            op_type="aten_op",
            device_self_ms=10.0,
            device_inclusive_ms=397411.092,  # inclusive 远超 wall
            evidence_ids=["run1:ev_000004"],
        )
    )
    ev.operators.append(
        OperatorEvidence(
            id="op_000002",
            name="aten::relu",
            layer="framework",
            op_type="aten_op",
            device_self_ms=10.0,
            device_inclusive_ms=5.0,
            evidence_ids=["run1:ev_000004"],
        )
    )
    hotspots = rank_hotspots(ev)
    ops = [h for h in hotspots if h.kind == "operator"]
    assert all(h.measured_fraction == pytest.approx(0.5) for h in ops)
    assert all(h.fraction_denominator == "summed_device_self_ms_of_all_framework_ops" for h in ops)
    assert all(h.measured_fraction <= 1.0 for h in hotspots)


def test_case5_same_observation_rule_both_backends_cite_own_evidence():
    """案例 5（修订）：同一观察规则跑 CUDA/Ascend，各自引用各自后端 evidence。"""
    results = {}
    for backend in ("cuda", "ascend"):
        ev = _ev(backend, {})
        ev.metric_evidence = _refs("workload.wall_ms", "runtime.api_summed_ms", "runtime.launch_summed_ms")
        ev.workload.wall_ms = 100.0
        ev.runtime.api_summed_ms = 60.0
        ev.runtime.launch_summed_ms = 50.0
        results[backend] = build_observations(ev)
    for backend, obs in results.items():
        launch = next(o for o in obs if o.kind == "runtime_launch_heavy")
        assert launch.value == pytest.approx(50.0 / 60.0)
        assert launch.evidence_ids == ["run1:ev_000002", "run1:ev_000003"]
        # 诊断层不得把该观察升格为 *_bound（P0-2）
        ev = _ev(backend, {})
        ev.metric_evidence = _refs("workload.wall_ms", "runtime.api_summed_ms", "runtime.launch_summed_ms")
        ev.workload.wall_ms = 100.0
        ev.runtime.api_summed_ms = 60.0
        ev.runtime.launch_summed_ms = 50.0
        assert all(c.class_ != "host_dispatch_bound" for c in diagnose(ev))


def test_compute_never_becomes_compute_bound_in_m3():
    """P0-3：compute 活动主导只产生观察；compute_bound 保留给 M4 硬件证据。"""
    refs = _refs("workload.wall_ms")
    refs["timeline.device_busy_ms"] = refs["workload.wall_ms"]  # 同 source
    refs["timeline.compute_ms"] = refs["workload.wall_ms"]  # 同 source
    ev = _ev("cuda", refs)
    ev.workload.wall_ms = 100.0
    ev.timeline.device_busy_ms = 95.0
    ev.timeline.compute_ms = 92.0
    candidates = diagnose(ev)
    assert all(c.class_ != "compute_bound" for c in candidates)
    assert any(o.kind == "compute_activity_dominant" for o in build_observations(ev))


def test_no_fabricated_classes():
    """全部过度声称类都在 M3 禁止出现；unknown 是唯一回退。"""
    ev = _ev(
        "cuda",
        _refs(
            "workload.wall_ms",
            "timeline.exposed_non_device_busy_ms",
            "runtime.api_summed_ms",
            "runtime.launch_summed_ms",
            "timeline.device_busy_ms",
            "timeline.compute_ms",
        ),
    )
    ev.workload.wall_ms = 100.0
    ev.timeline.exposed_non_device_busy_ms = 40.0
    ev.runtime.api_summed_ms = 60.0
    ev.runtime.launch_summed_ms = 50.0
    ev.timeline.device_busy_ms = 60.0
    ev.timeline.compute_ms = 55.0
    classes = {c.class_ for c in diagnose(ev)}
    assert classes <= {"communication_bound", "unknown"}
    assert "host_dispatch_bound" not in classes
    assert "synchronization_bound" not in classes
    assert "allocation_bound" not in classes
    assert "compute_bound" not in classes
    assert "memory_bound" not in classes
    assert "load_imbalance" not in classes


def test_below_observation_threshold_unknown():
    """低于观察阈值 -> no_salient_observation -> unknown(low)。"""
    ev = _ev(
        "cuda",
        _refs(
            "workload.wall_ms",
            "timeline.exposed_non_device_busy_ms",
            "runtime.api_summed_ms",
            "runtime.launch_summed_ms",
        ),
    )
    ev.workload.wall_ms = 100.0
    ev.timeline.exposed_non_device_busy_ms = 5.0
    ev.runtime.api_summed_ms = 60.0
    ev.runtime.launch_summed_ms = 5.0
    candidates = diagnose(ev)
    assert all(c.class_ == "unknown" for c in candidates)
    assert any(o.kind == "no_salient_observation" for o in build_observations(ev))


def test_enrich_fills_hotspots_and_diagnosis():
    refs = _refs("workload.wall_ms")
    for k in ("timeline.exposed_non_device_busy_ms", "communication.total_ms", "communication.overlap_ms"):
        refs[k] = refs["workload.wall_ms"]  # 四输入同一条 step_trace 记录（同 source）
    ev = _ev("ascend", refs)
    ev.run.run_id = "runabc123456"
    ev.workload.wall_ms = 100.0
    ev.timeline.exposed_non_device_busy_ms = 30.0
    ev.communication.total_ms = 35.0
    ev.communication.overlap_ms = 5.0
    ev.operators.append(
        OperatorEvidence(
            id="op_000001",
            name="task_x",
            layer="device",
            op_type="npu_task",
            calls=3,
            summed_device_ms=70.0,
            evidence_ids=["ev_000002"],  # 本地形式，hotspot 应规范化为全局
        )
    )
    enriched = enrich(ev)
    assert any(h.kind == "device_task" for h in enriched.hotspots)
    assert any(c.class_ == "communication_bound" for c in enriched.diagnosis.candidates)
    # P1：hotspot evidence_ids 统一为全局 run_id:ev_id 形式
    for h in enriched.hotspots:
        assert h.evidence_ids
        assert all(":" in eid for eid in h.evidence_ids)
    dev_hs = [h for h in enriched.hotspots if h.kind == "device_task"]
    assert any("runabc123456:ev_000002" in h.evidence_ids for h in dev_hs)


def test_gap_correlated_cross_source_never_upgrades():
    """MVP gate 审计：runtime 分类与 gap_overlap 绑定不同记录 -> 相关观察不产生，
    规则层同源校验兜底，绝不升格 *_bound。"""
    ev = _ev("cuda", _refs("workload.wall_ms", "runtime.api_summed_ms", "runtime.launch_summed_ms"))
    ev.workload.wall_ms = 100.0
    ev.runtime.api_summed_ms = 60.0
    ev.runtime.launch_summed_ms = 50.0
    ev.runtime.gap_overlap_ms = {"launch": 50.0}
    # gap_overlap 绑到与 runtime.launch_summed_ms 不同的记录（跨源）
    ev.metric_evidence["runtime.gap_overlap_ms.launch"] = ["run1:ev_000099"]
    obs = build_observations(ev)
    assert all(o.kind != "runtime_launch_gap_correlated" for o in obs)
    assert all(c.class_ != "host_dispatch_bound" for c in diagnose(ev))


def test_gap_wall_cross_source_suppresses_correlation():
    """gap 与 wall 跨源 -> 相关观察被抑制（观察层 Fix 1 纪律同样适用于相关性）。"""
    ev = _ev("cuda", _refs("workload.wall_ms", "runtime.api_summed_ms", "runtime.launch_summed_ms"))
    ev.workload.wall_ms = 100.0
    ev.runtime.api_summed_ms = 60.0
    ev.runtime.launch_summed_ms = 50.0
    ev.runtime.gap_overlap_ms = {"launch": 50.0}
    ev.metric_evidence["runtime.gap_overlap_ms.launch"] = ev.metric_evidence["runtime.launch_summed_ms"]
    # exposed 与 wall 绑不同记录 -> gap/wall 跨源
    ev.timeline.exposed_non_device_busy_ms = 40.0
    ev.metric_evidence["timeline.exposed_non_device_busy_ms"] = ["run1:ev_000098"]
    obs = build_observations(ev)
    assert all(o.kind != "runtime_launch_gap_correlated" for o in obs)
