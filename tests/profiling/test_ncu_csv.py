# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# NCU CSV 硬件 evidence 测试（M4.2a）：unit 保留与归一、n/a 不可用纪律、
# 千分位解析、salient 硬件观察（observation-only，无 *_bound）。
# 真实 NCU capture 需 GPU 权限 -> prepared_not_run；本层用 fixture 验证契约。

from pathlib import Path

import pytest

from vllm_omni.profiling.backends.ncu_csv import parse_ncu_csv
from vllm_omni.profiling.diagnosis import diagnose
from vllm_omni.profiling.observation import build_observations
from vllm_omni.profiling.schema import OptimizationEvidence

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HEADER = (
    '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
    '"Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"'
)


def _csv(rows: list[str]) -> str:
    return _HEADER + "\n" + "\n".join(rows) + "\n"


def _row(kid: int, kernel: str, metric: str, unit: str, value: str) -> str:
    return f'"{kid}","1234","python","host","{kernel}","0","7","(128, 1, 1)","(1, 1, 1)","0","8.0","LaunchStats","{metric}","{unit}","{value}"'


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "ncu_export.csv"
    p.write_text(
        _csv(
            [
                _row(0, "ampere_sgemm", "dram__bytes_read.sum", "byte", "1,073,741,824"),
                _row(0, "ampere_sgemm", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "%", "85.5"),
                _row(0, "ampere_sgemm", "gpu__time_duration.sum", "ns", "1,500"),
                _row(0, "ampere_sgemm", "smsp__average_threads_executed_per_instruction.ratio", "", "24.5"),
                _row(0, "ampere_sgemm", "smsp__warp_cycles_per_issued_instruction.ratio", "cycle", "n/a"),
                _row(0, "ampere_sgemm", "dram__bytes_read.sum", "Kbyte", "1073741.824"),
                _row(0, "ampere_sgemm", "lts__t_hit_rate.pct", "%", "90"),
                _row(0, "ampere_sgemm", "unknown__metric.custom", "foo", "7"),
            ]
        ),
        encoding="utf-8",
    )
    return p


def test_parse_preserves_ncu_triple(csv_file):
    # P0 unit 纪律：原样 (metric_name, metric_unit, metric_value) 三元组保留
    ev = parse_ncu_csv(csv_file)
    assert ev.backend == "cuda"
    assert ev.source == "ncu_export.csv"
    dram = next(e for e in ev.entries if e.metric_name == "dram__bytes_read.sum")
    assert dram.metric_unit == "byte"
    # 原样数值（千分位已解析为 int）；unit 原样保留
    assert dram.metric_value == 1_073_741_824
    assert dram.canonical_unit == "byte"
    assert dram.canonical_value == pytest.approx(1_073_741_824.0)  # 千分位解析


def test_scope_id_composed_from_source_fields(tmp_path):
    # P0-1：scope_id 由源字段组合（Process ID:Device:Context:Stream:ID），
    # 不用数组位置；同 kernel 名的两次 invocation 得到不同 scope_id
    p = tmp_path / "two_id.csv"
    p.write_text(
        _csv(
            [
                _row(0, "ampere_sgemm", "dram__bytes_read.sum", "byte", "1000"),
                _row(1, "ampere_sgemm", "dram__bytes_read.sum", "byte", "2000"),
            ]
        ),
        encoding="utf-8",
    )
    ev = parse_ncu_csv(p)
    ids = {e.scope_id for e in ev.entries}
    assert ids == {"1234:0:0:7:0", "1234:0:0:7:1"}


def test_metric_unit_empty_preserved_as_empty(csv_file):
    # P0-2：空 unit = source 明确报告的 unitless，保留 ""（不得变 None）
    ev = parse_ncu_csv(csv_file)
    ratio = next(e for e in ev.entries if e.metric_name.endswith(".ratio") and e.metric_unit == "")
    assert ratio.metric_unit == ""


def test_two_invocations_distinct_keys_only_a_salient(tmp_path):
    # gate 回归：同 kernel 名两次 invocation（A=85%、B=40%）——
    # 两个 entry 都保留、scope_id/metric key 互异、只有 A（85% ≥ 80）产
    # salient，且 salient 的 evidence 解析到 A 自己的 entry
    p = tmp_path / "two_inv.csv"
    p.write_text(
        _csv(
            [
                _row(0, "ampere_sgemm", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "%", "85"),
                _row(1, "ampere_sgemm", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "%", "40"),
            ]
        ),
        encoding="utf-8",
    )
    hw = parse_ncu_csv(p)
    ev = OptimizationEvidence()
    ev.run.backend = "cuda"
    ev.hardware = hw
    for e in hw.entries:
        key = f"hardware:{e.scope_id}:{e.metric_name}"
        ev.metric_evidence[key] = ["hwruntime:a" if e.canonical_value == 85.0 else "hwruntime:b"]
    obs = build_observations(ev)
    salient = [o for o in obs if o.kind == "hardware_resource_saturation"]
    # 只有 A（85% ≥ 80）产 salient；B（40%）不产
    assert len(salient) == 1
    assert salient[0].metric_key == "hardware:1234:0:0:7:0:sm__throughput.avg.pct_of_peak_sustained_elapsed"
    assert salient[0].evidence_ids == ["hwruntime:a"]
    # 两个 entry 都保留（invocation 不合并）
    entries = [e for e in hw.entries if e.metric_name == "sm__throughput.avg.pct_of_peak_sustained_elapsed"]
    assert len(entries) == 2
    assert {e.canonical_value for e in entries} == {85.0, 40.0}


def test_byte_si_normalization(csv_file):
    # P0：byte 族按 SI 1000 阶（Kbyte = 1000 byte，非 1024）
    ev = parse_ncu_csv(csv_file)
    kb = next(e for e in ev.entries if e.metric_unit == "Kbyte")
    assert kb.canonical_unit == "byte"
    assert kb.canonical_value == pytest.approx(1_073_741.824 * 1000)


def test_pct_and_time_normalization(csv_file):
    ev = parse_ncu_csv(csv_file)
    sm = next(e for e in ev.entries if e.metric_name.startswith("sm__throughput"))
    assert sm.canonical_unit == "pct"
    assert sm.canonical_value == pytest.approx(85.5)
    dur = next(e for e in ev.entries if e.metric_name == "gpu__time_duration.sum")
    assert dur.canonical_unit == "ns"
    assert dur.canonical_value == pytest.approx(1500.0)


def test_ratio_unitless_and_na_discipline(csv_file):
    ev = parse_ncu_csv(csv_file)
    ratio = next(e for e in ev.entries if e.metric_name.endswith(".ratio") and e.canonical_unit == "ratio")
    assert ratio.canonical_value == pytest.approx(24.5)
    # n/a -> unavailable（metric_value=None，canonical=None）：不填 0，不编造
    na = next(e for e in ev.entries if e.metric_name == "smsp__warp_cycles_per_issued_instruction.ratio")
    assert na.metric_value is None
    assert na.canonical_value is None


def test_unknown_unit_stays_unresolved(csv_file):
    # 未知 unit foo -> canonical None（不猜测换算）
    ev = parse_ncu_csv(csv_file)
    unknown = next(e for e in ev.entries if e.metric_unit == "foo")
    assert unknown.canonical_value is None
    assert unknown.canonical_unit is None
    assert unknown.metric_value == 7  # 原样数值保留


def test_hardware_saturation_observation_and_no_bound(csv_file):
    # M4.2a checkpoint：sm_throughput 85.5% >= 80 saturation 观察线 ->
    # hardware_resource_saturation 观察；不产 compute_bound/memory_bound
    ev = OptimizationEvidence()
    ev.hardware = parse_ncu_csv(csv_file)
    ev.run.backend = "cuda"
    obs = build_observations(ev)
    sat = [o for o in obs if o.kind == "hardware_resource_saturation"]
    assert len(sat) == 1
    # scope_id 键格式（P0-1）：hardware:{Process ID}:{Device}:{Context}:{Stream}:{ID}
    assert sat[0].metric_key == "hardware:1234:0:0:7:0:sm__throughput.avg.pct_of_peak_sustained_elapsed"
    assert sat[0].value == pytest.approx(85.5)
    candidates = diagnose(ev)
    assert all(c.class_ not in ("compute_bound", "memory_bound") for c in candidates)


def test_hit_rate_high_is_not_salient(csv_file):
    # P1：高命中率不是问题信号——lts hit rate 90% 不产任何观察
    ev = OptimizationEvidence()
    ev.hardware = parse_ncu_csv(csv_file)
    ev.run.backend = "cuda"
    obs = build_observations(ev)
    assert all("hit_rate" not in o.metric_key for o in obs)


def test_salient_observation_traces_to_hardware_evidence(csv_file):
    # P1-4：观察 evidence_ids 指向 hardware provenance 引用
    ev = OptimizationEvidence()
    ev.hardware = parse_ncu_csv(csv_file)
    key = "hardware:1234:0:0:7:0:sm__throughput.avg.pct_of_peak_sustained_elapsed"
    ev.metric_evidence[key] = ["run1:ev_000001"]
    obs = build_observations(ev)
    salient = next(o for o in obs if o.kind == "hardware_resource_saturation")
    assert salient.evidence_ids == ["run1:ev_000001"]


def test_round_trip_with_hardware(csv_file):
    ev = OptimizationEvidence()
    ev.hardware = parse_ncu_csv(csv_file)
    restored = OptimizationEvidence.from_dict(ev.to_dict())
    assert restored == ev
    assert restored.hardware.entries == ev.hardware.entries
    assert restored.hardware.provenance_ref is None


def test_missing_columns_fail_explicitly(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text('"Kernel Name","Metric Name"\n"x","y"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        parse_ncu_csv(p)


def test_empty_csv_fails_explicitly(tmp_path):
    p = tmp_path / "ncu.csv"
    p.write_text(_HEADER + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no metric rows"):
        parse_ncu_csv(p)


def test_schema_version_current():
    # M4.1c gate P0（2026-10-08）批准 v0.6 -> v0.7：Timeline.window_ms；
    # hardware 契约（M4.2a）不变
    ev = OptimizationEvidence()
    assert ev.schema_version == "0.7"
    assert ev.hardware is None


def test_duplicate_invocation_rows_preserved(tmp_path):
    # scope identity：同一 kernel 的两次 invocation 各成一行，均保留（不合并、不覆盖）；
    # 已知限制：metric_evidence 绑定键 hardware:{scope}:{metric_name} 在
    # invocation 粒度会碰撞——M4.2b 引入 launch_id 归属时修复
    p = tmp_path / "ncu.csv"
    p.write_text(
        _csv(
            [
                _row(0, "ampere_sgemm", "dram__bytes_read.sum", "byte", "1000"),
                _row(1, "ampere_sgemm", "dram__bytes_read.sum", "byte", "2000"),
            ]
        ),
        encoding="utf-8",
    )
    ev = parse_ncu_csv(p)
    rows = [e for e in ev.entries if e.metric_name == "dram__bytes_read.sum"]
    assert len(rows) == 2
    assert [e.metric_value for e in rows] == [1000, 2000]
