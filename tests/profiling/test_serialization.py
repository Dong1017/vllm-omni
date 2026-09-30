# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# 序列化测试：round-trip、严格版本/未知键失败、确定性输出、0.1/0.2 显式迁移、
# writer 侧 validation（P0-4：非法内存对象无法序列化）

import json

import pytest

from vllm_omni.profiling.schema import (
    SCHEMA_VERSION,
    DiagnosisCandidate,
    OperatorEvidence,
    OptimizationEvidence,
    RunInfo,
    ShapeAggregation,
)
from vllm_omni.profiling.units import TimeUnit, convert_time

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _full_evidence() -> OptimizationEvidence:
    ev = OptimizationEvidence()
    ev.run = RunInfo(
        backend="cuda",
        profiler="torch_profiler",
        profiler_version="2.4.0",
        rank=0,
        device="cuda:0",
        device_id="0",
        source_files=["trace_rank0.json"],
        workload_id="qwen_image_t2i",
        run_id="a1b2c3d4e5f6",
    )
    ev.workload.wall_ms = 1200.5
    ev.timeline.device_busy_ms = 900.0
    ev.timeline.exposed_non_device_busy_ms = 300.5
    ev.operators.append(
        OperatorEvidence(
            id="op_000001",
            name="aten::matmul",
            layer="framework",
            parent_id=None,
            op_type="aten_op",
            calls=120,
            device_self_ms=450.25,
            host_inclusive_ms=480.0,
            avg_device_us=convert_time(450.25, TimeUnit.MS, TimeUnit.US) / 120,
            median_device_us=3750.0,
            shapes=["[2, 4096, 3584]"],
            evidence_ids=["ev_000001"],
        )
    )
    ev.operators.append(
        OperatorEvidence(
            id="op_000002",
            name="gemm_kernel",
            layer="device",
            parent_id="op_000001",
            op_type="kernel",
            calls=120,
            summed_device_ms=450.25,
            evidence_ids=["ev_000001"],
        )
    )
    ev.diagnosis.candidates.append(
        DiagnosisCandidate(class_="launch_bound", confidence="medium", evidence_ids=["ev_000002"])
    )
    return ev


def _v01_evidence_dict() -> dict:
    """手工构造 0.1 版 evidence（字段名与 v0.1 schema 一致）。"""
    return {
        "schema_version": "0.1",
        "run": {"backend": "cuda", "profiler": "torch_profiler", "rank": 0, "source_files": ["t.json"]},
        "workload": {"wall_ms": 100.0},
        "timeline": {
            "device_busy_ms": 80.0,
            "exposed_non_device_busy_ms": 20.0,
            "compute_ms": 60.0,
            "communication_ms": 10.0,
            "memory_ms": 5.0,
            "unknown_ms": 0.0,
        },
        "operators": [
            {"id": "op_000001", "name": "aten::mm", "op_type": "aten_op", "calls": 2, "total_host_ms": 0.09},
            {"id": "op_000002", "name": "gemm_kernel", "op_type": "kernel", "calls": 2, "total_device_ms": 0.05},
        ],
        "runtime": {"launch_count": 2, "dispatch_ms": 0.02, "copy_ms": 0.01, "synchronization_ms": 0.005},
        "memory": {"allocation_count": 3},
        "shapes": [
            {"operator": "aten::mm", "shape": "[2,4096]", "calls": 2, "total_ms": 0.05, "workload_fraction": 0.5}
        ],
        "provenance": [
            {
                "id": "ev_000001",
                "source_file": "t.json",
                "source_type": "torch_profiler_trace",
                "parser": "p",
                "unit": "us",
            }
        ],
    }


def _v02_evidence_dict() -> dict:
    """0.2 版：含 dispatch_ms 与 framework 层 total_* 字段。"""
    return {
        "schema_version": "0.2",
        "run": {"backend": "ascend", "profiler": "torch_npu", "rank": 8, "source_files": ["t.json"]},
        "runtime": {"launch_count": 4, "dispatch_ms": 15.0, "synchronization_ms": 1.0},
        "operators": [
            {
                "id": "op_000001",
                "name": "aten::mm",
                "layer": "framework",
                "op_type": "aten_op",
                "total_host_ms": 0.09,
                "total_device_ms": 0.05,
            },
            {"id": "op_000002", "name": "aclnnX", "layer": "runtime", "op_type": "cann_api", "total_host_ms": 0.02},
        ],
    }


def test_round_trip_preserves_all_sections():
    ev = _full_evidence()
    restored = OptimizationEvidence.from_dict(ev.to_dict())
    assert restored == ev


def test_json_round_trip_string_identical():
    ev = _full_evidence()
    text1 = ev.to_json()
    restored = OptimizationEvidence.from_json(text1)
    assert restored.to_json() == text1


def test_none_stays_none_never_zero():
    ev = OptimizationEvidence()
    restored = OptimizationEvidence.from_dict(ev.to_dict())
    assert restored.timeline.compute_ms is None
    assert restored.memory.allocation_count is None
    assert restored.memory.host_device_copy_ms is None
    assert json.loads(ev.to_json())["timeline"]["compute_ms"] is None


def test_diagnosis_class_json_key():
    ev = _full_evidence()
    raw = json.loads(ev.to_json())
    cand = raw["diagnosis"]["candidates"][0]
    assert "class" in cand
    assert "class_" not in cand
    assert cand["class"] == "launch_bound"


def test_unsupported_schema_version_fails_explicitly():
    data = json.loads(_full_evidence().to_json())
    data["schema_version"] = "9.9"
    with pytest.raises(ValueError, match="unsupported evidence schema_version"):
        OptimizationEvidence.from_dict(data)


def test_unknown_fields_fail_explicitly():
    data = json.loads(_full_evidence().to_json())
    data["timeline"]["made_up_metric"] = 1.0
    with pytest.raises(ValueError, match="unknown fields"):
        OptimizationEvidence.from_dict(data)


def test_deterministic_output_ac08():
    e1, e2 = _full_evidence(), _full_evidence()
    assert e1.to_json() == e2.to_json()


def test_schema_version_constant():
    assert SCHEMA_VERSION == "0.4"
    assert OptimizationEvidence().schema_version == "0.4"


# ---- writer 侧 validation（P0-4）：非法内存对象无法序列化 ----


def test_writer_rejects_invalid_diagnosis_class():
    with pytest.raises(ValueError, match="invalid diagnosis class"):
        DiagnosisCandidate(class_="made_up_class", confidence="high")


def test_writer_rejects_pseudo_precision_confidence():
    with pytest.raises(ValueError, match="invalid confidence"):
        DiagnosisCandidate(class_="memory_bound", confidence="0.82")


def test_writer_rejects_invalid_layer():
    with pytest.raises(ValueError, match="invalid operator layer"):
        OperatorEvidence(id="op_000001", name="x", layer="middle")


def test_writer_rejects_fraction_without_denominator():
    with pytest.raises(ValueError, match="denominator required"):
        ShapeAggregation(operator="op", shape="[1]", device_time_fraction=0.5)


def test_invalid_object_cannot_enter_evidence_json():
    # 模拟 M3 rule engine 直接构造非法 candidate 并挂到 evidence 上
    ev = OptimizationEvidence()
    with pytest.raises(ValueError):
        ev.diagnosis.candidates.append(DiagnosisCandidate(class_="not_a_class", confidence="high"))


def test_mutated_object_rejected_at_serialization():
    # P0-5（ckpt-3）：构造后 mutate 的非法对象，to_json 必须拒绝
    ev = OptimizationEvidence()
    cand = DiagnosisCandidate(class_="memory_bound", confidence="medium")
    cand.confidence = "0.82"  # 构造后突变
    ev.diagnosis.candidates.append(cand)
    with pytest.raises(ValueError, match="invalid confidence"):
        ev.to_json()
    with pytest.raises(ValueError, match="invalid confidence"):
        ev.to_dict()


# ---- 0.1 -> 0.2 -> 0.3 显式迁移（不 silent-drop）----


def test_migrate_01_operators_gain_layer():
    ev = OptimizationEvidence.from_dict(_v01_evidence_dict())
    assert ev.schema_version == "0.4"
    fw = next(o for o in ev.operators if o.name == "aten::mm")
    dev = next(o for o in ev.operators if o.name == "gemm_kernel")
    assert fw.layer == "framework"
    assert dev.layer == "device"
    assert fw.parent_id is None
    # 0.1 CUDA host 是 chrome inclusive dur；device 关联求和是 self 语义（P0-3 迁移规则）
    assert fw.host_inclusive_ms == 0.09
    assert fw.device_self_ms is None  # 0.1 该行未记录 device 时间
    assert dev.summed_device_ms == 0.05


def test_migrate_01_memory_union_preserved_not_renamed():
    ev = OptimizationEvidence.from_dict(_v01_evidence_dict())
    assert not hasattr(ev.timeline, "memory_ms")
    assert ev.backend_metrics.cuda["legacy_memory_union_ms"] == 5.0
    assert ev.memory.host_device_copy_ms == 0.01


def test_migrate_01_runtime_dispatch_becomes_api_summed():
    # P0-2：0.1/0.2 的 dispatch_ms 实际是 runtime API 总和，映射到 api_summed_ms
    ev = OptimizationEvidence.from_dict(_v01_evidence_dict())
    assert ev.runtime.api_summed_ms == 0.02
    assert not hasattr(ev.runtime, "dispatch_ms")


def test_migrate_02_framework_semantics_backend_aware():
    ev = OptimizationEvidence.from_dict(_v02_evidence_dict())
    fw = next(o for o in ev.operators if o.layer == "framework")
    # ascend 的 Device Total 是 inclusive
    assert fw.device_inclusive_ms == 0.05
    assert fw.host_inclusive_ms == 0.09
    assert fw.device_self_ms is None
    rt = next(o for o in ev.operators if o.layer == "runtime")
    assert rt.summed_host_ms == 0.02


def test_migrate_02_dispatch_becomes_api_summed():
    ev = OptimizationEvidence.from_dict(_v02_evidence_dict())
    assert ev.runtime.api_summed_ms == 15.0
    # 0.2 的 synchronization 与 dispatch 重复计数，互斥分类无法恢复 -> 不迁移
    assert ev.runtime.synchronization_summed_ms is None


def test_migrate_01_does_not_mutate_input():
    data = _v01_evidence_dict()
    OptimizationEvidence.from_dict(data)
    assert data["timeline"]["memory_ms"] == 5.0
    assert data["shapes"][0]["workload_fraction"] == 0.5


def test_migrate_01_keeps_provenance():
    ev = OptimizationEvidence.from_dict(_v01_evidence_dict())
    assert ev.provenance[0].id == "ev_000001"
    assert ev.provenance[0].run_id is None
    assert ev.provenance[0].source_sha256 is None
