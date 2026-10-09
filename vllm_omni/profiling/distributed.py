# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Distributed timeline analysis（M4.3）：多 rank 空泡分析（#8228 语义）。
# 语义来源（T0 审计，updates/20261008-1400-m43-t0-semantic-audit-profiling-agent-01.md）：
#   DeviceTaskActive_raw(rank) = union(TASK 表全部行)——#8228 原始定义，
#     semantics = raw_TASK_rows_including_sentinel，compatibility = issue_8228_legacy_metric；
#     不得泛化为 canonical physical device-idle（sentinel 语义见 M4.1c 调查）。
#   ComputeActive(rank) = union(TASK ∧ globalTaskId ∈ COMPUTE_TASK_INFO)。
#   NoDeviceTask/NoCompute = W − Active；Common* = ∩_rank No*(rank, W)。
#   不变量：CommonNoDeviceTask ⊆ CommonNoCompute（compute ⊆ raw_task ⇒ gaps 单调）。
# 两层校准（gate 裁定 2026-10-08）：
#   1) 内部层显式拆分 raw/sentinel/attributable/compute；T8 先复现 published
#      metric，再报排除 sentinel 的 delta，禁止静默改定义；
#   2) shared clock 不由 "epoch ns" 自动成立：有独立 metadata/evidence 才
#      validated；用户/实验显式保证则 explicit_assertion；两者皆无 fail fast；
#      不自动估计 rank clock offset。
# 纪律：integer ns 全程（跨 rank 对齐禁 float）；窗口显式（默认窗口命名为
# common_coverage_window，非 workload wall）；不改 M4.1c 单 rank 冻结语义；
# 分布式结果走独立 artifact，不进 OptimizationEvidence。

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from vllm_omni.profiling.backends.ascend_trace_db import _open_trace_db
from vllm_omni.profiling.intervals import (
    Interval,
    complement,
    intersect,
    subtract,
    union,
)
from vllm_omni.profiling.intervals import (
    total as intervals_total,
)
from vllm_omni.profiling.provenance import sha256_file

DISTRIBUTED_SCHEMA_VERSION = "0.1"
RAW_TASK_SEMANTICS = "raw_TASK_rows_including_sentinel"
ISSUE_8228_COMPATIBILITY = "issue_8228_legacy_metric"
TOP_INTERVALS = 20
SENTINEL_CONNECTION_ID = -1  # M4.1c 调查：conn=-1 TASK = per-stream capture-session 哨兵


class DistributedClockError(ValueError):
    """多 rank 输入未通过 clock/window 可比性 gate（T2）。"""


class DistributedWindowError(ValueError):
    """named window 越界或格式非法（T3，默认 fail fast，不 silent clip）。"""


@dataclass
class RankTimelineIntervals:
    """单 rank 的 backend-neutral interval 层（T1）。全部 integer ns，已 merge。"""

    rank_id: int
    source: str
    source_sha256: str
    unit: str  # 恒 "ns"；跨 rank 对齐前由 gate 校验一致
    coverage_intervals: list[Interval] = field(default_factory=list)  # SESSION_TIME_INFO（可空）
    raw_task_intervals: list[Interval] = field(default_factory=list)  # 全部 TASK 行 union（#8228 语义）
    sentinel_task_intervals: list[Interval] = field(default_factory=list)  # conn=-1 哨兵 union
    attributable_task_intervals: list[Interval] = field(default_factory=list)  # conn!=-1 union
    compute_intervals: list[Interval] = field(default_factory=list)  # gid ∈ COMPUTE_TASK_INFO
    communication_intervals: list[Interval] = field(default_factory=list)  # gid ∈ COMMUNICATION_TASK_INFO
    unknown_intervals: list[Interval] = field(default_factory=list)  # attributable − compute − communication
    task_row_count: int = 0
    sentinel_row_count: int = 0
    communication_source: str = "unavailable"  # "COMMUNICATION_TASK_INFO" / "unavailable"

    def to_dict(self) -> dict:
        return {
            "rank_id": self.rank_id,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "unit": self.unit,
            "coverage_intervals": [[a, b] for a, b in self.coverage_intervals],
            "raw_task_intervals": [[a, b] for a, b in self.raw_task_intervals],
            "sentinel_task_intervals": [[a, b] for a, b in self.sentinel_task_intervals],
            "attributable_task_intervals": [[a, b] for a, b in self.attributable_task_intervals],
            "compute_intervals": [[a, b] for a, b in self.compute_intervals],
            "communication_intervals": [[a, b] for a, b in self.communication_intervals],
            "unknown_intervals": [[a, b] for a, b in self.unknown_intervals],
            "task_row_count": self.task_row_count,
            "sentinel_row_count": self.sentinel_row_count,
            "communication_source": self.communication_source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RankTimelineIntervals:
        layers: dict[str, list[Interval]] = {
            name: [(int(a), int(b)) for a, b in data[name]]
            for name in (
                "coverage_intervals",
                "raw_task_intervals",
                "sentinel_task_intervals",
                "attributable_task_intervals",
                "compute_intervals",
                "communication_intervals",
                "unknown_intervals",
            )
        }
        return cls(
            rank_id=int(data["rank_id"]),
            source=data["source"],
            source_sha256=data["source_sha256"],
            unit=data["unit"],
            coverage_intervals=layers["coverage_intervals"],
            raw_task_intervals=layers["raw_task_intervals"],
            sentinel_task_intervals=layers["sentinel_task_intervals"],
            attributable_task_intervals=layers["attributable_task_intervals"],
            compute_intervals=layers["compute_intervals"],
            communication_intervals=layers["communication_intervals"],
            unknown_intervals=layers["unknown_intervals"],
            task_row_count=int(data["task_row_count"]),
            sentinel_row_count=int(data["sentinel_row_count"]),
            communication_source=data["communication_source"],
        )


def build_rank_timeline_from_ascend_db(db_path: Path) -> RankTimelineIntervals:
    """从单个 Ascend 大库提取 rank timeline 层（T1）。

    复用 M4.1b/M4.1c 的 `_open_trace_db`（schema 校验一致）；TASK 流式扫描
    一次性构建全部层；不改动单 rank analyze 的冻结语义。"""
    db_path = Path(db_path)
    con = _open_trace_db(db_path)
    try:
        sha = sha256_file(db_path)
        rank_rows = con.execute("SELECT rankId, deviceId FROM RANK_DEVICE_MAP").fetchall()
        rank_ids = {r[0] for r in rank_rows}
        if len(rank_ids) != 1:
            raise DistributedClockError(
                f"rank timeline: {db_path.name} RANK_DEVICE_MAP has {len(rank_ids)} rankIds; "
                "one DB must carry exactly one rank (filter first)"
            )
        rank_id = int(next(iter(rank_ids)))
        session = con.execute("SELECT startTimeNs, endTimeNs FROM SESSION_TIME_INFO").fetchone()
        coverage: list[Interval] = [(int(session[0]), int(session[1]))] if session and None not in session else []
        compute_ids = {
            int(g) for (g,) in con.execute("SELECT globalTaskId FROM COMPUTE_TASK_INFO WHERE globalTaskId IS NOT NULL")
        }
        existing = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "COMMUNICATION_TASK_INFO" in existing:
            comm_ids = {
                int(g)
                for (g,) in con.execute(
                    "SELECT globalTaskId FROM COMMUNICATION_TASK_INFO WHERE globalTaskId IS NOT NULL"
                )
            }
            communication_source = "COMMUNICATION_TASK_INFO"
        else:
            comm_ids = set()
            communication_source = "unavailable"
        # 流式单遍：raw/sentinel/attributable/compute/communication 同步 merge
        # （复用 intervals.union 的 merge 语义：touching 合并、零长跳过）。
        collected: dict[str, list[Interval]] = {
            k: [] for k in ("raw", "sentinel", "attributable", "compute", "communication")
        }
        task_rows = 0
        sentinel_rows = 0
        for conn_id, start, end, gid in con.execute("SELECT connectionId, startNs, endNs, globalTaskId FROM TASK"):
            if start is None or end is None or end <= start:
                continue  # 零长/非法行跳过（与既有 union 纪律一致）
            task_rows += 1
            iv: Interval = (int(start), int(end))
            collected["raw"].append(iv)
            if conn_id == SENTINEL_CONNECTION_ID:
                sentinel_rows += 1
                collected["sentinel"].append(iv)
            else:
                collected["attributable"].append(iv)
            if gid is not None and int(gid) in compute_ids:
                collected["compute"].append(iv)
            if gid is not None and int(gid) in comm_ids:
                collected["communication"].append(iv)
        raw_iv = union(collected["raw"])
        attributable_iv = union(collected["attributable"])
        compute_iv = union(collected["compute"])
        comm_iv = union(collected["communication"])
        # unknown = attributable − (compute ∪ communication)：不自动当 idle 或 compute
        unknown_iv = subtract(attributable_iv, union(collected["compute"] + collected["communication"]))
        return RankTimelineIntervals(
            rank_id=rank_id,
            source=db_path.name,
            source_sha256=sha,
            unit="ns",
            coverage_intervals=coverage,
            raw_task_intervals=raw_iv,
            sentinel_task_intervals=union(collected["sentinel"]),
            attributable_task_intervals=attributable_iv,
            compute_intervals=compute_iv,
            communication_intervals=comm_iv,
            unknown_intervals=unknown_iv,
            task_row_count=task_rows,
            sentinel_row_count=sentinel_rows,
            communication_source=communication_source,
        )
    finally:
        con.close()


def _base_gate_checks(rank_timelines: list[RankTimelineIntervals]) -> None:
    """T2 基础可比性检查：rank 唯一、单位一致、timeline 非空、coverage 存在。"""
    if not rank_timelines:
        raise DistributedClockError("distributed: no rank timelines provided")
    ids = [t.rank_id for t in rank_timelines]
    if len(set(ids)) != len(ids):
        raise DistributedClockError(f"distributed: duplicate rank ids {sorted(ids)}")
    for t in rank_timelines:
        if t.unit != "ns":
            raise DistributedClockError(f"distributed: rank{t.rank_id} unit {t.unit!r} != 'ns'")
        if not t.raw_task_intervals:
            raise DistributedClockError(f"distributed: rank{t.rank_id} has empty raw_task timeline")
        if not t.coverage_intervals:
            raise DistributedClockError(
                f"distributed: rank{t.rank_id} has no SESSION_TIME_INFO coverage; cannot support cross-rank windows"
            )


def resolve_clock_validation(
    rank_timelines: list[RankTimelineIntervals],
    shared_clock: bool,
    clock_metadata: dict | None,
) -> tuple[str, dict | None]:
    """T2 clock gate：validated（独立 metadata 有 evidence）> explicit_assertion
    （调用者显式保证）> fail fast。不做 rank clock offset 估计。"""
    _base_gate_checks(rank_timelines)
    if clock_metadata is not None:
        evidence = clock_metadata.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            raise DistributedClockError(
                "distributed: clock metadata must carry a non-empty 'evidence' string "
                "to justify clock_validation='validated'"
            )
        return "validated", clock_metadata
    if shared_clock:
        return "explicit_assertion", None
    raise DistributedClockError(
        "distributed: shared clock between ranks is neither evidenced (clock metadata) "
        "nor explicitly asserted (--shared-clock); refusing to compare timelines"
    )


def _common_coverage(rank_timelines: list[RankTimelineIntervals]) -> Interval:
    lo = max(t.coverage_intervals[0][0] for t in rank_timelines)
    hi = min(t.coverage_intervals[-1][1] for t in rank_timelines)
    if hi <= lo:
        raise DistributedClockError(
            "distributed: rank coverage windows have empty intersection; no common analysis window exists"
        )
    return (lo, hi)


def resolve_windows(
    rank_timelines: list[RankTimelineIntervals],
    windows_path: Path | None,
) -> list[tuple[str, Interval]]:
    """T3 窗口契约：默认 common_coverage_window = ∩ rank coverage；named windows
    来自外部 JSON（{"name": [start_ns, end_ns]}），越界/倒置直接 fail fast。"""
    common = _common_coverage(rank_timelines)
    windows: list[tuple[str, Interval]] = [("common_coverage_window", common)]
    if windows_path is None:
        return windows
    raw = json.loads(Path(windows_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise DistributedWindowError(f"distributed: windows file must be a non-empty JSON object: {windows_path}")
    for name, bounds in raw.items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise DistributedWindowError(f"distributed: window {name!r} must be [start_ns, end_ns]")
        lo, hi = int(bounds[0]), int(bounds[1])
        if hi <= lo:
            raise DistributedWindowError(f"distributed: window {name!r} has start >= end ({lo} >= {hi})")
        for t in rank_timelines:
            cov_lo, cov_hi = t.coverage_intervals[0][0], t.coverage_intervals[-1][1]
            if lo < cov_lo or hi > cov_hi:
                raise DistributedWindowError(
                    f"distributed: window {name!r} [{lo}, {hi}] not inside rank{t.rank_id} "
                    f"coverage [{cov_lo}, {cov_hi}]; out-of-range windows are rejected, not clipped"
                )
        windows.append((str(name), (lo, hi)))
    return windows


def _no_task_within(active: list[Interval], window: Interval) -> list[Interval]:
    """No*(rank, W) = W − Active；W 内的补区间（integer ns）。"""
    return complement(window, active)


def _clip_total(intervals: list[Interval], window: Interval) -> float:
    return sum(b - a for a, b in intervals if a < window[1] and b > window[0])


def _interval_block(intervals: list[Interval]) -> dict:
    """审计块：count/total/top-N（duration 降序、start 升序，与原分析一致）。
    传入区间已由构造保证位于 window 内（complement/交集均不越窗）。"""
    return {
        "interval_count": len(intervals),
        "duration_ns": sum(b - a for a, b in intervals),
        "top_intervals": [
            {"start_ns": a, "end_ns": b, "duration_ns": b - a}
            for a, b in sorted(intervals, key=lambda x: (-(x[1] - x[0]), x[0]))[:TOP_INTERVALS]
        ],
    }


def _fold_intersection(rank_lists: list[list[Interval]]) -> list[Interval]:
    acc: list[Interval] | None = None
    for intervals in rank_lists:
        acc = intervals if acc is None else intersect(acc, intervals)
    return acc if acc is not None else []


def reduce_window(rank_timelines: list[RankTimelineIntervals], window_name: str, window: Interval) -> dict:
    """T4 核心 reducer：per-rank + distributed（含 interval 审计结果）。

    禁止：sum/average 各 rank idle、用最慢/最快 rank 代替 common、把 rank 时间相加当 wall。"""
    lo, hi = int(window[0]), int(window[1])
    width = hi - lo
    per_rank = {}
    no_device_lists, no_compute_lists, no_device_attr_lists = [], [], []
    for t in rank_timelines:
        raw_active = _clip_total(t.raw_task_intervals, window)
        attr_active = _clip_total(t.attributable_task_intervals, window)
        compute_active = _clip_total(t.compute_intervals, window)
        comm_active = (
            _clip_total(t.communication_intervals, window) if t.communication_source != "unavailable" else None
        )
        no_task = _no_task_within(t.raw_task_intervals, window)
        no_task_attr = _no_task_within(t.attributable_task_intervals, window)
        no_compute = _no_task_within(t.compute_intervals, window)
        no_device_lists.append(no_task)
        no_device_attr_lists.append(no_task_attr)
        no_compute_lists.append(no_compute)
        per_rank[str(t.rank_id)] = {
            "device_task_active_ms": raw_active / 1e6,
            "no_device_task_ms": (width - raw_active) / 1e6,
            "attributable_task_active_ms": attr_active / 1e6,
            "no_device_task_attributable_ms": (width - attr_active) / 1e6,
            "compute_ms": compute_active / 1e6,
            "no_compute_ms": (width - compute_active) / 1e6,
            "communication_ms": comm_active / 1e6 if comm_active is not None else None,
        }
    common_no_device = _fold_intersection(no_device_lists)
    common_no_device_attr = _fold_intersection(no_device_attr_lists)
    common_no_compute = _fold_intersection(no_compute_lists)
    # 不变量（代码即纪律）：CommonNoDeviceTask ⊆ CommonNoCompute；数值夹在 [0, |W|]
    assert not intersect(common_no_device, complement(window, common_no_compute)), (
        "distributed: CommonNoDeviceTask must be a subset of CommonNoCompute"
    )
    common_no_device_ms = intervals_total(common_no_device)
    common_no_compute_ms = intervals_total(common_no_compute)
    assert 0 <= common_no_device_ms <= width and 0 <= common_no_compute_ms <= width
    return {
        "window": {"name": window_name, "start_ns": lo, "end_ns": hi, "duration_ms": width / 1e6},
        "per_rank": per_rank,
        "distributed": {
            "common_no_device_task": {
                "semantics": RAW_TASK_SEMANTICS,
                "compatibility": ISSUE_8228_COMPATIBILITY,
                "duration_ms": common_no_device_ms / 1e6,
                **_interval_block(common_no_device),
                "intervals_ns": [[a, b] for a, b in common_no_device],
            },
            "common_no_device_task_attributable": {
                "semantics": "attributable_TASK_rows_sentinel_excluded",
                "compatibility": None,
                "duration_ms": intervals_total(common_no_device_attr) / 1e6,
                **_interval_block(common_no_device_attr),
                "intervals_ns": [[a, b] for a, b in common_no_device_attr],
            },
            "common_no_compute": {
                "semantics": "complement_of_COMPUTE_TASK_INFO_backed_compute_union",
                "compatibility": ISSUE_8228_COMPATIBILITY,
                "duration_ms": common_no_compute_ms / 1e6,
                **_interval_block(common_no_compute),
                "intervals_ns": [[a, b] for a, b in common_no_compute],
            },
        },
    }


def build_distributed_summary(
    rank_timelines: list[RankTimelineIntervals],
    clock_validation: str,
    clock_metadata: dict | None,
    windows_path: Path | None,
) -> dict:
    """T5 artifact 组装：独立 schema（非 OptimizationEvidence），每个 window 独立结果。"""
    windows = resolve_windows(rank_timelines, windows_path)
    ordered = sorted(rank_timelines, key=lambda t: t.rank_id)
    summary = {
        "schema_version": DISTRIBUTED_SCHEMA_VERSION,
        "backend": "ascend",
        "rank_ids": [t.rank_id for t in ordered],
        "input_sources": [
            {
                "rank_id": t.rank_id,
                "source": t.source,
                "source_sha256": t.source_sha256,
                "task_row_count": t.task_row_count,
                "sentinel_row_count": t.sentinel_row_count,
                "communication_source": t.communication_source,
            }
            for t in ordered
        ],
        "clock_validation": {
            "status": clock_validation,
            "metadata": clock_metadata,
            "offset_estimation": "not_performed",
        },
        "analysis_windows": [name for name, _ in windows],
        "windows": [reduce_window(ordered, name, window) for name, window in windows],
        "rank_timelines": [t.to_dict() for t in ordered],
        "provenance": {
            "parser": "ascend.trace_db.rank_timeline",
            "rank_timeline_layers": (
                "raw_task(all TASK rows) / sentinel_task(conn=-1) / "
                "attributable_task(conn!=-1) / compute(COMPUTE_TASK_INFO) / "
                "communication(COMMUNICATION_TASK_INFO) / unknown(attributable-compute-communication)"
            ),
            "source_mapping": {str(t.rank_id): t.source for t in ordered},
        },
    }
    return summary


def compare_distributed(base: dict, candidate: dict) -> dict:
    """T9 薄比较：两份 distributed_summary 的逐 window 指标 delta。"""
    if base.get("schema_version") != candidate.get("schema_version"):
        raise ValueError(
            f"compare: schema_version mismatch {base.get('schema_version')!r} vs {candidate.get('schema_version')!r}"
        )
    base_windows = {w["window"]["name"]: w for w in base["windows"]}
    cand_windows = {w["window"]["name"]: w for w in candidate["windows"]}
    if set(base_windows) != set(cand_windows):
        raise ValueError(f"compare: window sets differ: {sorted(base_windows)} vs {sorted(cand_windows)}")
    rows = []
    for name in sorted(base_windows):
        b, c = base_windows[name]["distributed"], cand_windows[name]["distributed"]
        for metric in ("common_no_device_task", "common_no_compute"):
            b_ms, c_ms = b[metric]["duration_ms"], c[metric]["duration_ms"]
            rows.append(
                {
                    "window": name,
                    "metric": metric,
                    "baseline_ms": b_ms,
                    "candidate_ms": c_ms,
                    "delta_ms": c_ms - b_ms,
                    "delta_pct": (c_ms - b_ms) / b_ms * 100 if b_ms > 0 else None,
                }
            )
    return {"comparisons": rows}


def render_markdown(summary: dict) -> str:
    """T5 人工阅读视图：window / all-rank no-task / no-compute / 差值 / 最长区间 / rank spread。"""
    lines = ["# Distributed timeline summary", ""]
    cv = summary["clock_validation"]
    lines.append(f"- backend: {summary['backend']}; ranks: {summary['rank_ids']}")
    lines.append(
        f"- clock_validation: **{cv['status']}**"
        + (" (caller-asserted, not independently verified)" if cv["status"] == "explicit_assertion" else "")
        + ("; offset estimation: not performed" if cv.get("offset_estimation") == "not_performed" else "")
    )
    lines.append("")
    lines.append(
        "Metric semantics: `common no device task` = all-rank intersection of the complement of "
        f"raw TASK-row unions (`{RAW_TASK_SEMANTICS}`, `{ISSUE_8228_COMPATIBILITY}`). "
        "`common no compute` contains it; the difference is common no-compute time that still "
        "contains some non-compute device activity (not labeled communication without interval attribution)."
    )
    lines.append("")
    for window in summary["windows"]:
        w = window["window"]
        dist = window["distributed"]
        no_task = dist["common_no_device_task"]["duration_ms"]
        no_compute = dist["common_no_compute"]["duration_ms"]
        lines.append(f"## window `{w['name']}`")
        lines.append("")
        lines.append(f"- window: [{w['start_ns']}, {w['end_ns']}] ns, {w['duration_ms']:.3f} ms")
        lines.append(f"- all-rank no device task (raw TASK semantics): **{no_task:.3f} ms**")
        lines.append(f"- all-rank no compute task: **{no_compute:.3f} ms**")
        lines.append(
            f"- difference (common no-compute containing non-compute device activity): {no_compute - no_task:.3f} ms"
        )
        ranks = window["per_rank"]
        no_task_spread = [v["no_device_task_ms"] for v in ranks.values()]
        lines.append(
            f"- per-rank no-device-task spread: min {min(no_task_spread):.3f} ms, max {max(no_task_spread):.3f} ms"
        )
        lines.append("")
        lines.append(
            "| rank | device_task_active_ms | no_device_task_ms | no_device_task_attributable_ms "
            "| compute_ms | no_compute_ms | communication_ms |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for rank_id in sorted(ranks, key=int):
            v = ranks[rank_id]
            comm = "unavailable" if v["communication_ms"] is None else f"{v['communication_ms']:.3f}"
            lines.append(
                f"| {rank_id} | {v['device_task_active_ms']:.3f} | {v['no_device_task_ms']:.3f} | "
                f"{v['no_device_task_attributable_ms']:.3f} | {v['compute_ms']:.3f} | "
                f"{v['no_compute_ms']:.3f} | {comm} |"
            )
        top = dist["common_no_device_task"]["top_intervals"][:5]
        if top:
            lines.append("")
            lines.append("Longest common no-device-task intervals (top 5):")
            lines.append("")
            for item in top:
                rel = item["start_ns"] - w["start_ns"]
                lines.append(
                    f"- start_rel {rel / 1e9:.3f} s, duration {item['duration_ns'] / 1e9:.3f} s "
                    f"[{item['start_ns']}, {item['end_ns']}]"
                )
        lines.append("")
    return "\n".join(lines)


def write_distributed_outputs(summary: dict, output_dir: Path, intervals_csv: bool = False) -> dict:
    """T5 落盘：distributed_summary.json + .md；可选 common 层 intervals CSV。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    json_path = output_dir / "distributed_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written["distributed_summary.json"] = json_path
    md_path = output_dir / "distributed_summary.md"
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    written["distributed_summary.md"] = md_path
    if intervals_csv:
        import csv

        csv_path = output_dir / "distributed_intervals.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["window", "layer", "start_ns", "end_ns", "duration_ns"])
            for window in summary["windows"]:
                w = window["window"]
                for metric in ("common_no_device_task", "common_no_device_task_attributable", "common_no_compute"):
                    for a, b in window["distributed"][metric]["intervals_ns"]:
                        writer.writerow([w["name"], metric, a, b, b - a])
        written["distributed_intervals.csv"] = csv_path
    return written
