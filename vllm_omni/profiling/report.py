# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# 报告生成（T14）：evidence.json / summary.md / operators.csv / shapes.csv / provenance.json。
# 状态口径（AC-20）：measured / derived / estimated / unavailable 四档，禁止混用。
# 数值格式：时间输出 ms（内部存储口径见 schema 字段名）。

from __future__ import annotations

import csv
from pathlib import Path

from vllm_omni.profiling.schema import OperatorEvidence, OptimizationEvidence

STATUS_MEASURED = "measured"
STATUS_DERIVED = "derived"
STATUS_ESTIMATED = "estimated"
STATUS_UNAVAILABLE = "unavailable"


def _fmt(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "unavailable"


def _status(value: float | None, derived: bool = False) -> str:
    if value is None:
        return STATUS_UNAVAILABLE
    return STATUS_DERIVED if derived else STATUS_MEASURED


def _csv_path(out_dir: Path, name: str) -> Path:
    return out_dir / name


def write_evidence_json(ev: OptimizationEvidence, out_dir: Path) -> Path:
    p = out_dir / "evidence.json"
    p.write_text(ev.to_json(), encoding="utf-8")
    return p


def write_provenance_json(ev: OptimizationEvidence, out_dir: Path) -> Path:
    import json

    p = out_dir / "provenance.json"
    payload = {
        "schema_version": ev.schema_version,
        "evidence": [
            {
                "id": r.id,
                "source_file": r.source_file,
                "source_type": r.source_type,
                "rank": r.rank,
                "query": r.query,
                "parser": r.parser,
                "aggregation": r.aggregation,
                "unit": r.unit,
            }
            for r in ev.provenance
        ],
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def write_operators_csv(ev: OptimizationEvidence, out_dir: Path, sort_key: str = "device_self") -> Path:
    p = _csv_path(out_dir, "operators.csv")
    rows = list(ev.operators)

    # P0-3：排序只允许同语义比较。framework 层用 device_self_ms；
    # runtime/device 层用 summed_device_ms；跨层不混合排序，按 layer 分组内排序。
    def key(op: OperatorEvidence):
        if op.layer == "framework":
            return ("framework", op.device_self_ms is None, -(op.device_self_ms or 0.0))
        return (op.layer or "", op.summed_device_ms is None, -(op.summed_device_ms or 0.0))

    rows.sort(key=key)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "name",
                "layer",
                "op_type",
                "calls",
                "device_self_ms",
                "device_inclusive_ms",
                "summed_device_ms",
                "host_self_ms",
                "host_inclusive_ms",
                "summed_host_ms",
                "avg_device_us",
                "median_device_us",
                "shapes",
            ]
        )
        for op in rows:
            w.writerow(
                [
                    op.id,
                    op.name,
                    op.layer or "",
                    op.op_type or "",
                    op.calls if op.calls is not None else "",
                    _fmt(op.device_self_ms),
                    _fmt(op.device_inclusive_ms),
                    _fmt(op.summed_device_ms),
                    _fmt(op.host_self_ms),
                    _fmt(op.host_inclusive_ms),
                    _fmt(op.summed_host_ms),
                    _fmt(op.avg_device_us),
                    _fmt(op.median_device_us),
                    "; ".join(op.shapes),
                ]
            )
    return p


def write_shapes_csv(ev: OptimizationEvidence, out_dir: Path) -> Path:
    p = _csv_path(out_dir, "shapes.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["operator", "shape", "calls", "total_ms", "device_time_fraction", "fraction_denominator", "profiler_ratio"]
        )
        for s in ev.shapes:
            w.writerow(
                [
                    s.operator,
                    s.shape,
                    s.calls if s.calls is not None else "",
                    _fmt(s.total_ms),
                    _fmt(s.device_time_fraction, 6),
                    s.fraction_denominator or "",
                    _fmt(s.profiler_ratio, 6),
                ]
            )
    return p


def _metric_source(ev: OptimizationEvidence, key: str) -> str:
    """P1：metric 来源按 metric_evidence 渲染（引用 -> provenance 记录的 source_file+unit）。"""
    refs = ev.metric_evidence.get(key, [])
    if not refs:
        return "unbound"
    by_id = {r.id: r for r in ev.provenance}
    parts = []
    for ref in refs:
        eid = ref.split(":", 1)[1] if ":" in ref else ref
        rec = by_id.get(eid)
        parts.append(f"{rec.source_file} [{rec.unit}]" if rec else ref)
    return ", ".join(sorted(set(parts)))


def render_summary_md(ev: OptimizationEvidence) -> str:
    lines: list[str] = []
    lines.append("# Profiling evidence summary")
    lines.append("")
    lines.append(f"- backend: `{ev.run.backend or 'unavailable'}`")
    lines.append(f"- profiler: `{ev.run.profiler or 'unavailable'}`")
    lines.append(f"- rank: `{ev.run.rank if ev.run.rank is not None else 'unavailable'}`")
    lines.append(f"- device: `{ev.run.device or 'unavailable'}`")
    lines.append(f"- source: `{', '.join(ev.run.source_files) or 'unavailable'}`")
    lines.append("")

    lines.append("## Measured: workload path")
    lines.append("")
    lines.append("| metric | value (ms) | status | source |")
    lines.append("|---|---:|---|---|")
    wall = ev.workload.wall_ms
    busy = ev.timeline.device_busy_ms
    gap = ev.timeline.exposed_non_device_busy_ms
    # P1：每个 metric 的 source 列按 metric_evidence 绑定渲染，不再笼统列全部文件
    lines.append(f"| wall | {_fmt(wall)} | {_status(wall)} | {_metric_source(ev, 'workload.wall_ms')} |")
    lines.append(
        f"| device busy (union) | {_fmt(busy)} | {_status(busy)} | {_metric_source(ev, 'timeline.device_busy_ms')} |"
    )
    # CUDA 的 gap 来自 wall-busy 推导（derived）；Ascend 来自 CANN Free 列（实测）
    is_cuda = ev.run.backend == "cuda"
    gap_src = _metric_source(ev, "timeline.exposed_non_device_busy_ms")
    gap_note = " (derived: wall - busy)" if is_cuda else ""
    lines.append(
        f"| exposed non-device-busy gap | {_fmt(gap)} | {_status(gap, derived=is_cuda)} | {gap_src}{gap_note} |"
    )
    # compute/communication 时间口径按后端标注（AC-06）：CUDA = busy 的 partition
    # union（subtract 得非重叠分解）；Ascend = step-trace/overlap type 列求和（summed）
    time_basis = "union" if is_cuda else "summed"
    compute = ev.timeline.compute_ms
    compute_src = _metric_source(ev, "timeline.compute_ms")
    lines.append(f"| compute ({time_basis}) | {_fmt(compute)} | {_status(compute)} | {compute_src} |")
    comm = ev.timeline.communication_ms
    comm_src = _metric_source(ev, "timeline.communication_ms")
    lines.append(f"| communication ({time_basis}) | {_fmt(comm)} | {_status(comm)} | {comm_src} |")
    alloc = ev.runtime.allocation_summed_ms
    alloc_src = _metric_source(ev, "runtime.allocation_summed_ms")
    lines.append(f"| allocation | {_fmt(alloc)} | {_status(alloc)} | {alloc_src} |")
    lines.append("")

    lines.append("## Top operators")
    lines.append("")
    # P0-3：framework 层 TopK 只用 device_self_ms（inclusive 不得进入排序/fraction）
    aten = [o for o in ev.operators if o.layer == "framework"]
    aten.sort(key=lambda o: (o.device_self_ms is None, -(o.device_self_ms or 0.0), -(o.host_inclusive_ms or 0.0)))
    lines.append("| operator | calls | device self (ms) | host inclusive (ms) | common shapes |")
    lines.append("|---|---:|---:|---:|---|")
    for op in aten[:10]:
        lines.append(
            f"| {op.name} | {op.calls if op.calls is not None else 'unavailable'} "
            f"| {_fmt(op.device_self_ms)} | {_fmt(op.host_inclusive_ms)} | {op.shapes[0] if op.shapes else ''} |"
        )
    lines.append("")

    lines.append("## Runtime characteristics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(
        f"| launch count | {ev.runtime.launch_count if ev.runtime.launch_count is not None else 'unavailable'} |"
    )
    # P0-2：runtime API 互斥分类，api_summed = 各分类之和；禁止与 wall 相除当 fraction
    lines.append(f"| runtime API total, summed (ms) | {_fmt(ev.runtime.api_summed_ms)} |")
    lines.append(f"| launch total (ms) | {_fmt(ev.runtime.launch_summed_ms)} |")
    lines.append(f"| synchronization total (ms) | {_fmt(ev.runtime.synchronization_summed_ms)} |")
    lines.append(f"| allocation total (ms) | {_fmt(ev.runtime.allocation_summed_ms)} |")
    lines.append(f"| other runtime total (ms) | {_fmt(ev.runtime.other_summed_ms)} |")
    lines.append(f"| host-device copy total (ms) | {_fmt(ev.memory.host_device_copy_ms)} |")
    lines.append("")

    lines.append("## Optimization opportunities")
    lines.append("")
    lines.append("| candidate | evidence | estimated scope | confidence |")
    lines.append("|---|---|---:|---:|")
    if ev.diagnosis.candidates:
        for c in ev.diagnosis.candidates:
            lines.append(f"| {c.class_} | {', '.join(c.evidence_ids)} | unavailable | {c.confidence} |")
    else:
        lines.append("| none yet (deterministic diagnosis lands at M3) | | | |")
    lines.append("")
    return "\n".join(lines)


def write_summary_md(ev: OptimizationEvidence, out_dir: Path) -> Path:
    p = out_dir / "summary.md"
    p.write_text(render_summary_md(ev), encoding="utf-8")
    return p


def write_outputs(ev: OptimizationEvidence, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "evidence.json": write_evidence_json(ev, out_dir),
        "summary.md": write_summary_md(ev, out_dir),
        "operators.csv": write_operators_csv(ev, out_dir),
        "provenance.json": write_provenance_json(ev, out_dir),
    }
    if ev.shapes:
        outputs["shapes.csv"] = write_shapes_csv(ev, out_dir)
    return outputs
