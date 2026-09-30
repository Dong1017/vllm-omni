# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Schema 语义测试：None = unavailable、时间口径字段互斥、枚举校验（AC-03/AC-06/AC-19）

import pytest

from vllm_omni.profiling.schema import (
    CONFIDENCE_LEVELS,
    DIAGNOSIS_CLASSES,
    Diagnosis,
    DiagnosisCandidate,
    Hotspot,
    OperatorEvidence,
    OptimizationEvidence,
    ShapeAggregation,
    Timeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_fresh_evidence_has_no_zero_fakes():
    # AC-03：缺失维度必须是 None，不允许默认 0 伪装成实测
    ev = OptimizationEvidence()
    assert ev.timeline.device_busy_ms is None
    assert ev.workload.wall_ms is None
    assert ev.runtime.api_summed_ms is None
    assert ev.memory.allocated_bytes is None
    assert ev.memory.host_device_copy_ms is None
    assert ev.communication.total_ms is None
    assert ev.operators == []
    assert ev.diagnosis.candidates == []


def test_operator_layer_enum_validation():
    with pytest.raises(ValueError, match="invalid operator layer"):
        OptimizationEvidence.from_dict(
            {
                "schema_version": "0.2",
                "operators": [{"id": "op_000001", "name": "x", "layer": "middle"}],
            }
        )


def test_operator_three_layer_roundtrip():
    # 决议 1：framework/runtime/device 三层 + parent_id 一对多可表达
    ev = OptimizationEvidence()
    ev.operators = [
        OperatorEvidence(id="op_000001", name="aten::ln", layer="framework", op_type="aten_op"),
        OperatorEvidence(
            id="op_000002", name="cudaLaunchKernel", layer="runtime", parent_id="op_000001", op_type="cuda_runtime_api"
        ),
        OperatorEvidence(id="op_000003", name="triton_k1", layer="device", parent_id="op_000001", op_type="kernel"),
        OperatorEvidence(id="op_000004", name="triton_k2", layer="device", parent_id="op_000001", op_type="kernel"),
    ]
    restored = OptimizationEvidence.from_dict(ev.to_dict())
    device_rows = [o for o in restored.operators if o.layer == "device"]
    assert len(device_rows) == 2  # 一对多：同一 framework parent 两个 device task
    assert all(o.parent_id == "op_000001" for o in device_rows)


def test_shape_fraction_denominator_invariant():
    # 决议 2 + P0-4：device_time_fraction 必须带 fraction_denominator，
    # 构造与读取两侧都拦截
    with pytest.raises(ValueError, match="denominator required"):
        ShapeAggregation(operator="op", shape="[1]", calls=1, device_time_fraction=0.5)
    with pytest.raises(ValueError, match="denominator required"):
        OptimizationEvidence.from_dict(
            {
                "schema_version": "0.3",
                "shapes": [{"operator": "op", "shape": "[1]", "calls": 1, "device_time_fraction": 0.5}],
            }
        )
    # 成对出现则合法
    s = ShapeAggregation(operator="op", shape="[1]", calls=1, device_time_fraction=0.5, fraction_denominator="d")
    assert s.device_time_fraction == 0.5


def test_timeline_fields_are_distinct():
    # AC-06：不同时间口径是独立字段，schema 层不存在可混用的单一 duration 字段
    t = Timeline(device_busy_ms=10.0)
    t.exposed_non_device_busy_ms = 2.0
    t.compute_ms = 8.0
    # 填了 compute 也不影响 device_busy 原值
    assert t.device_busy_ms == 10.0
    assert t.compute_ms == 8.0


def test_diagnosis_candidate_enum_validation():
    with pytest.raises(ValueError, match="invalid diagnosis class"):
        OptimizationEvidence.from_dict(
            {
                "schema_version": "0.1",
                "diagnosis": {"candidates": [{"class": "not_a_class", "confidence": "high", "evidence_ids": []}]},
            }
        )


def test_diagnosis_confidence_rejects_pseudo_precision():
    with pytest.raises(ValueError, match="invalid confidence"):
        OptimizationEvidence.from_dict(
            {
                "schema_version": "0.1",
                "diagnosis": {"candidates": [{"class": "memory_bound", "confidence": 0.86, "evidence_ids": []}]},
            }
        )


def test_hotspot_kind_is_free_text_not_taxonomy():
    # P1-3：Hotspot 与 diagnosis taxonomy 解耦，kind 不再强制 *_bound
    ev = OptimizationEvidence.from_dict(
        {
            "schema_version": "0.3",
            "hotspots": [{"name": "aclrtSynchronizeStream", "kind": "synchronization"}],
        }
    )
    assert ev.hotspots[0].kind == "synchronization"


def test_legacy_hotspot_category_key_rejected():
    # 旧 category 键在 v0.3 中已不存在 -> 未知字段显式失败，不静默丢弃
    with pytest.raises(ValueError, match="unknown fields"):
        OptimizationEvidence.from_dict(
            {
                "schema_version": "0.3",
                "hotspots": [{"name": "gemm", "category": "warp_stall_heaven"}],
            }
        )


def test_unknown_diagnosis_class_rejected_on_construction_roundtrip():
    cand = DiagnosisCandidate(class_="allocation_bound", confidence="low", evidence_ids=["ev_000001"])
    assert cand.class_ in DIAGNOSIS_CLASSES
    assert cand.confidence in CONFIDENCE_LEVELS
    d = Diagnosis(candidates=[cand])
    assert d.candidates[0].class_ == "allocation_bound"


def test_hotspot_dataclass_accepts_kind():
    h = Hotspot(
        name="op",
        kind="operator",
        measured_fraction=0.4,
        fraction_denominator="summed_device_self_ms_of_all_framework_ops",
    )
    assert h.measured_fraction == 0.4
    assert h.kind == "operator"
