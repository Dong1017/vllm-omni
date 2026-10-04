# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# NCU CSV 硬件计数器解析（M4.2a，final-review 修订）。
# P0 unit 纪律：canonical 硬件计数器 source 是原始 NCU 的
# (metric_name, metric_unit, metric_value) 三元组——原样保留三元组，unit 归一
# 只写 canonical_value/canonical_unit，不覆盖原值。
#
# 单位口径（M4.2a gate 审裁，引 NVIDIA Nsight Compute CLI 文档）：
#   - ncu 自动缩放使用 SI factors：byte 族 K/M/G = 1000/1e6/1e9（非 1024）；
#   - ncu --csv 默认隐含 --print-units base（M4.2b collector contract 显式加
#     --print-units base，双保险）；
#   - 因此 byte 族按 1000 归一，1024 写法是错误语义。
#
# unitless 推断（final-review P1）：空 unit 仅对 METRIC_REGISTRY 内的已知
# counter 按注册表 canonical unit 解释；未知 metric + 空 unit -> canonical
# =None（不猜）。registry 同时承载观察语义：仅 saturation 族的 pct counter
# 越过观察线才产 hardware_resource_saturation 观察（hit-rate/计数类不产）。

from __future__ import annotations

import csv
from pathlib import Path

from vllm_omni.profiling.schema import HardwareEvidence, HardwareMetricEntry

# 单位归一表：ncu 原样 unit -> (canonical_unit, factor)。全部按 SI factors。
_UNIT_TABLE: dict[str, tuple[str, float]] = {
    "byte": ("byte", 1.0),
    "Kbyte": ("byte", 1e3),
    "Mbyte": ("byte", 1e6),
    "Gbyte": ("byte", 1e9),
    "nsecond": ("ns", 1.0),
    "usecond": ("ns", 1e3),
    "msecond": ("ns", 1e6),
    "second": ("ns", 1e9),
    "ns": ("ns", 1.0),
    "us": ("ns", 1e3),
    "ms": ("ns", 1e6),
    "s": ("ns", 1e9),
    "%": ("pct", 1.0),
    "cycle": ("cycle", 1.0),
    "Kcycle": ("cycle", 1e3),
    "Mcycle": ("cycle", 1e6),
    "Gcycle": ("cycle", 1e9),
    "instance": ("count", 1.0),
    "invocation": ("count", 1.0),
    "warp": ("count", 1.0),
}

_REQUIRED_COLUMNS = ("Kernel Name", "Metric Name", "Metric Unit", "Metric Value")

# M4.2 固定 counter registry（种子 = Yotta feasibility 20-counter 清单 +
# 架构 fallback 名；canonical unit 为 ncu --print-units base 下的语义单位）。
# semantic: "saturation" = 资源吞吐/占用族，pct 高值可产
# hardware_resource_saturation 观察；None = 描述性 counter，不产观察
# （hit-rate 高是好事；计数/寄存器是描述）。
_METRIC_REGISTRY: dict[str, dict[str, str | None]] = {
    "gpu__time_duration.sum": {"canonical_unit": "ns", "semantic": None},
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": {"canonical_unit": "pct", "semantic": "saturation"},
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": {"canonical_unit": "pct", "semantic": "saturation"},
    "l1tex__throughput.avg.pct_of_peak_sustained_active": {"canonical_unit": "pct", "semantic": "saturation"},
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": {"canonical_unit": "pct", "semantic": "saturation"},
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": {"canonical_unit": "pct", "semantic": "saturation"},
    "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": {
        "canonical_unit": "pct",
        "semantic": "saturation",
    },
    "smsp__issue_active.avg.pct_of_peak_sustained_active": {"canonical_unit": "pct", "semantic": "saturation"},
    "sm__warps_active.avg.pct_of_peak_sustained_active": {"canonical_unit": "pct", "semantic": None},
    "l1tex__t_hit_rate.pct": {"canonical_unit": "pct", "semantic": None},
    "l1tex__t_sector_hit_rate.pct": {"canonical_unit": "pct", "semantic": None},
    "lts__t_hit_rate.pct": {"canonical_unit": "pct", "semantic": None},
    "lts__t_request_hit_rate.pct": {"canonical_unit": "pct", "semantic": None},
    "lts__t_sector_hit_rate.pct": {"canonical_unit": "pct", "semantic": None},
    "dram__bytes_read.sum": {"canonical_unit": "byte", "semantic": None},
    "dram__bytes_op_read.sum": {"canonical_unit": "byte", "semantic": None},
    "fbpa__dram_read_bytes.sum": {"canonical_unit": "byte", "semantic": None},
    "dram__bytes_written.sum": {"canonical_unit": "byte", "semantic": None},
    "dram__bytes_op_write.sum": {"canonical_unit": "byte", "semantic": None},
    "fbpa__dram_write_bytes.sum": {"canonical_unit": "byte", "semantic": None},
    "sm__active_cycles_elapsed.sum": {"canonical_unit": "cycle", "semantic": None},
    "sm__cycles_active.sum": {"canonical_unit": "cycle", "semantic": None},
    "smsp__inst_executed.sum": {"canonical_unit": "count", "semantic": None},
    "smsp__inst_executed.avg.per_cycle_active": {"canonical_unit": "ratio", "semantic": None},
    "smsp__average_threads_executed_per_instruction.ratio": {"canonical_unit": "ratio", "semantic": None},
    "smsp__average_thread_inst_executed_per_inst_executed.ratio": {"canonical_unit": "ratio", "semantic": None},
    "smsp__warp_cycles_per_issued_instruction.ratio": {"canonical_unit": "ratio", "semantic": None},
    "launch__registers_per_thread": {"canonical_unit": "count", "semantic": None},
    "l1tex__data_pipe_lsu_wavefronts_mem_local.sum": {"canonical_unit": "count", "semantic": None},
    "l1tex__t_output_wavefronts_pipe_lsu_mem_local.sum": {"canonical_unit": "count", "semantic": None},
    "launch__shared_mem_per_block_dynamic": {"canonical_unit": "byte", "semantic": None},
}

# saturation 语义族：pct 类吞吐/占用 counter，高值可产
# hardware_resource_saturation 观察（hit-rate/计数类不在此列——
# 高命中率高 occupancy 是好事或仅描述性）。
SATURATION_PCT_METRICS = frozenset(
    name for name, policy in _METRIC_REGISTRY.items() if policy.get("semantic") == "saturation"
)


def _parse_value(raw: str) -> float | int | str | None:
    """ncu CSV Metric Value 原样解析：数值（含千分位逗号）/文本（如 n/a）。"""
    v = raw.strip()
    if not v or v.lower() in ("n/a", "na"):
        return None
    numeric = v.replace(",", "")
    try:
        f = float(numeric)
        is_int = f.is_integer() and "." not in numeric and "e" not in numeric.lower()
        return int(f) if is_int else f
    except ValueError:
        return v  # 非数值文本原样保留（unavailable 纪律：不猜数值）


def _canonical(metric_name: str, unit: str | None, value: float | int) -> tuple[float, str] | None:
    """单位归一（final-review P0 修订）：

    - 已知 unit：按 SI 单位表换算（byte 族 1000 阶，非 1024）；
    - 空 unit：仅当 metric 在 METRIC_REGISTRY 内才按注册表 canonical unit
      解释（registry-gated unitless 推断）；未知 metric + 空 unit -> None；
    - 其他未知 unit -> None（不猜测换算）。
    """
    if unit is None:
        return None
    entry = _UNIT_TABLE.get(unit)
    if entry is not None:
        canonical_unit, factor = entry
        return (float(value) * factor, canonical_unit)
    if unit == "":
        reg = _METRIC_REGISTRY.get(metric_name)
        if reg is not None and reg.get("canonical_unit"):
            return (float(value), str(reg["canonical_unit"]))
        return None  # 未知 metric + 空 unit：不猜语义单位
    return None


def parse_ncu_csv(csv_path: Path) -> HardwareEvidence:
    """解析 ncu --csv long-format 导出为 HardwareEvidence。

    契约（M4.2a gate 修订）：Metric Unit 为必需列（缺失显式报错，不得按
    unitless 解释）；行内空 unit = source 明确报告的 unitless（经 registry
    门控解释）；entries 逐条保留 (metric_name, metric_unit, metric_value)
    原样三元组 + scope_id 调用身份；canonical_* 仅做单位换算。
    """
    csv_path = Path(csv_path)
    entries: list[HardwareMetricEntry] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        missing = [c for c in _REQUIRED_COLUMNS if c not in cols]
        if missing:
            raise ValueError(f"ncu csv: missing columns {missing} in {csv_path.name}; unsupported export format")
        for row in reader:
            kernel = (row.get("Kernel Name") or "").strip()
            metric = (row.get("Metric Name") or "").strip()
            if not kernel or not metric:
                continue
            # P0-1：scope_id 由源字段组合（Process ID:Device:Context:Stream:ID），
            # 不用数组位置；真实 capture 的全局唯一性由 M4.2b 实测验证
            # P0-2：Metric Unit 为必需列；空 unit 保留 ""（source 明确报告的
            # unitless），None 仅用于"该列不存在"的迁移数据，两者不得混淆
            unit_raw = (row.get("Metric Unit") or "").strip()
            value_raw = (row.get("Metric Value") or "").strip()
            value = _parse_value(value_raw)
            canonical = _canonical(metric, unit_raw, value) if isinstance(value, (int, float)) else None
            scope_id = ":".join(
                (row.get(col) or "").strip() for col in ("Process ID", "Device", "Context", "Stream", "ID")
            )
            entries.append(
                HardwareMetricEntry(
                    scope=kernel,
                    scope_id=scope_id,
                    metric_name=metric,
                    metric_unit=unit_raw,  # "" = source 明确报告的 unitless，保留不吞（P0-2）
                    metric_value=value,
                    canonical_value=canonical[0] if canonical else None,
                    canonical_unit=canonical[1] if canonical else None,
                )
            )
    if not entries:
        raise ValueError(f"ncu csv: no metric rows parsed from {csv_path.name}; unsupported export format")
    return HardwareEvidence(backend="cuda", source=csv_path.name, entries=entries)
