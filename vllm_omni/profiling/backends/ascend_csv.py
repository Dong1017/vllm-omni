# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Ascend CSV adapter（M2-T1）：解析 torch_npu ASCEND_PROFILER_OUTPUT 家族。
# 输入：step_trace_time.csv / op_statistic.csv / kernel_details.csv /
#       operator_details.csv / api_statistic.csv（真实样本见 docs/ascend-artifact-mapping.md）。
# 规则（全部确定性、写入 provenance）：
#   - 三层模型（决议 1）：framework(operator_details) -> runtime(api_statistic Level=acl)
#     -> device(kernel_details)；CSV 之间无关联键，parent_id 置 None，不做跨文件猜测；
#   - op_statistic 的 Ratio(%) 分母不明 -> 原样放 backend_metrics.ascend（决议 2）；
#   - timeline：compute/communication/exposed 来自 step_trace（CANN 官方归类，逐列存在才填），
#     device_busy 来自 kernel_details 区间 union；memory 不进入 common timeline（决议 3）；
#   - 大 CSV 全部流式逐行解析（AC-30），数值字段 strip 处理真实样本中的尾部 \t。

from __future__ import annotations

import csv
import statistics
from collections import Counter
from pathlib import Path

from vllm_omni.profiling.discovery import check_size
from vllm_omni.profiling.intervals import Interval, union
from vllm_omni.profiling.intervals import total as intervals_total
from vllm_omni.profiling.provenance import EvidenceRecord, ProvenanceStore, compute_run_id, sha256_file
from vllm_omni.profiling.schema import (
    BackendMetrics,
    CommunicationStats,
    OperatorEvidence,
    OptimizationEvidence,
    RunInfo,
    RuntimeStats,
    ShapeAggregation,
    StageTime,
    Timeline,
    Workload,
)

_PARSER = "ascend.csv"
_LAYER_FRAMEWORK = "framework"
_LAYER_RUNTIME = "runtime"
_LAYER_DEVICE = "device"
_FRACTION_DENOMINATOR = "summed_duration_all_device_tasks"


def _us_to_ms(value_us: float) -> float:
    return value_us / 1000.0


def _rows(path: Path):
    check_size(path)
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # strip 每个字段值（真实样本 Start Time 含尾部 \t）
        for row in reader:
            yield {k: v.strip() if isinstance(v, str) else v for k, v in row.items() if k is not None}


def _to_f(value: str | None) -> float | None:
    if value is None or value == "" or value == "N/A":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int_or_none(value: float | None) -> int | None:
    return int(value) if value is not None else None


class _DeviceAgg:
    """device task 聚合：时长 + 按 shape 的时长归集（决议 2）。"""

    __slots__ = ("calls", "total_us", "durs", "shape_counts", "shape_us", "core_types")

    def __init__(self) -> None:
        self.calls = 0
        self.total_us = 0.0
        self.durs: list[float] = []
        self.shape_counts: Counter[str] = Counter()
        self.shape_us: dict[str, float] = {}
        self.core_types: Counter[str] = Counter()


class _FrameworkAgg:
    """framework op 聚合：operator_details.csv 的 Self/Total 四列（P0-3）。"""

    __slots__ = (
        "rows",
        "host_us",
        "host_seen",
        "device_us",
        "device_seen",
        "host_self_us",
        "host_self_seen",
        "device_self_us",
        "device_self_seen",
        "shapes",
    )

    def __init__(self) -> None:
        self.rows = 0
        self.host_us = 0.0
        self.host_seen = False
        self.device_us = 0.0
        self.device_seen = False
        self.host_self_us = 0.0
        self.host_self_seen = False
        self.device_self_us = 0.0
        self.device_self_seen = False
        self.shapes: Counter[str] = Counter()


class _RuntimeAgg:
    __slots__ = ("calls", "total_us")

    def __init__(self) -> None:
        self.calls = 0
        self.total_us = 0.0


def analyze_ascend_csv(prof_dir: Path) -> tuple[OptimizationEvidence, ProvenanceStore]:
    """解析 ASCEND_PROFILER_OUTPUT 目录，返回 evidence 与 provenance store。"""
    prof_dir = Path(prof_dir)
    files = {
        "step_trace": prof_dir / "step_trace_time.csv",
        "op_statistic": prof_dir / "op_statistic.csv",
        "kernel_details": prof_dir / "kernel_details.csv",
        "operator_details": prof_dir / "operator_details.csv",
        "api_statistic": prof_dir / "api_statistic.csv",
    }
    present = {k: p for k, p in files.items() if p.exists()}
    if not present:
        raise ValueError(f"ascend adapter: no known torch_npu CSV artifacts under {prof_dir}")
    if "kernel_details" not in present and "operator_details" not in present:
        raise ValueError(f"ascend adapter: need at least kernel_details.csv or operator_details.csv under {prof_dir}")

    # run_id 覆盖全部参与分析的 source（决议 4）
    hashes = {k: sha256_file(p) for k, p in sorted(present.items())}
    run_id = compute_run_id(list(hashes.values()), backend="ascend", parser=_PARSER)
    store = ProvenanceStore(run_id=run_id)
    rec_by_key: dict[str, EvidenceRecord] = {}
    for key in sorted(present):
        path = present[key]
        rec_by_key[key] = store.add(
            source_file=path.name,
            source_type="ascend_csv",
            parser=_PARSER,
            unit="us" if key != "op_statistic" else "ratio",
            query=f"csv:{path.name}",
            aggregation=None,
            source_sha256=hashes[key],
        )
    file_evidences = {key: rec.id for key, rec in rec_by_key.items()}
    aten_inputs = [
        file_evidences[k] for k in ("operator_details", "kernel_details", "api_statistic") if k in rec_by_key
    ]
    ev_aten = store.add(
        source_file=",".join(sorted(p.name for p in present.values())),
        source_type="ascend_csv",
        parser=_PARSER,
        unit="us",
        query="operator_details + kernel_details + api_statistic(Level=acl) group by name",
        aggregation="group_by(name); sum; median; union_intervals",
        depends_on=aten_inputs,
    )

    # P0-1：CSV 家族只有 Device_id（设备号），没有分布式 rank 证据。
    # 真实样本经大库 RANK_DEVICE_MAP 核实 rankId=0, deviceId=8 —— Device_id 不是 rank。
    device_id: str | None = None
    timeline = Timeline()
    workload = Workload()
    comm = CommunicationStats()

    # ---- step_trace_time.csv：逐列独立聚合，列存在才填（0.0 也是实测值）----
    step_seen = False
    stage_times: list[StageTime] = []
    col_sums: dict[str, float] = {}
    _STEP_COLS = (
        "Stage",
        "Computing",
        "Communication",
        "Communication(Not Overlapped)",
        "Overlapped",
        "Free",
        "Bubble",
        "Preparing",
    )
    if "step_trace" in present:
        for row in _rows(present["step_trace"]):
            if device_id is None:
                device_id = row.get("Device_id") or None
            step_seen = True
            step = row.get("Step") or ""
            stage = _to_f(row.get("Stage"))
            if stage is not None:
                stage_times.append(StageTime(name=f"step_{step}", wall_ms=_us_to_ms(stage)))
            for col in _STEP_COLS:
                v = _to_f(row.get(col))
                if v is not None:
                    col_sums[col] = col_sums.get(col, 0.0) + v
    if step_seen:
        if stage_times:
            workload.wall_ms = (
                stage_times[-1].wall_ms if len(stage_times) == 1 else _us_to_ms(col_sums.get("Stage", 0.0))
            )
            workload.stage_times = stage_times
        if "Computing" in col_sums:
            timeline.compute_ms = _us_to_ms(col_sums["Computing"])
        if "Communication" in col_sums:
            timeline.communication_ms = _us_to_ms(col_sums["Communication"])
        if "Free" in col_sums:
            timeline.exposed_non_device_busy_ms = _us_to_ms(col_sums["Free"])
        if "Communication" in col_sums:
            comm.total_ms = _us_to_ms(col_sums["Communication"])
        if "Overlapped" in col_sums:
            comm.overlap_ms = _us_to_ms(col_sums["Overlapped"])
        # overlap_ratio 不填：CANN 未文档化 Overlapped 的分母语义（mapping doc）

    # ---- kernel_details.csv：device 层 + device-busy union ----
    kernel_aggs: dict[str, _DeviceAgg] = {}
    kern_ints: list[Interval] = []
    task_sum_us = 0.0
    core_type_time: Counter[str] = Counter()
    device_event_seen = False
    if "kernel_details" in present:
        for row in _rows(present["kernel_details"]):
            name = row.get("Name") or ""
            if not name:
                continue
            start = _to_f(row.get("Start Time(us)"))
            dur = _to_f(row.get("Duration(us)"))
            agg = kernel_aggs.setdefault(name, _DeviceAgg())
            agg.calls += 1
            if dur is not None:
                agg.total_us += dur
                agg.durs.append(dur)
                task_sum_us += dur
                core = row.get("Accelerator Core") or "unknown"
                agg.core_types[core] += 1
                core_type_time[core] += dur
            if start is not None and dur is not None and dur > 0:
                device_event_seen = True
                kern_ints.append((start, start + dur))
            shape = row.get("Input Shapes") or ""
            if shape and shape != "N/A":
                agg.shape_counts[shape] += 1
                if dur is not None:
                    agg.shape_us[shape] = agg.shape_us.get(shape, 0.0) + dur
            if device_id is None:
                device_id = row.get("Device_id") or None
    if device_event_seen:
        timeline.device_busy_ms = _us_to_ms(intervals_total(union(kern_ints)))

    # ---- operator_details.csv：framework 层（P0-3：Self/Total 四列全保留）----
    framework_aggs: dict[str, _FrameworkAgg] = {}
    if "operator_details" in present:
        for row in _rows(present["operator_details"]):
            name = row.get("Name") or ""
            if not name:
                continue
            agg = framework_aggs.setdefault(name, _FrameworkAgg())
            agg.rows += 1
            host_total = _to_f(row.get("Host Total Duration(us)"))
            device_total = _to_f(row.get("Device Total Duration(us)"))
            host_self = _to_f(row.get("Host Self Duration(us)"))
            device_self = _to_f(row.get("Device Self Duration(us)"))
            if host_total is not None:
                agg.host_us += host_total
                agg.host_seen = True
            if device_total is not None:
                agg.device_us += device_total
                agg.device_seen = True
            if host_self is not None:
                agg.host_self_us += host_self
                agg.host_self_seen = True
            if device_self is not None:
                agg.device_self_us += device_self
                agg.device_self_seen = True
            shape = row.get("Input Shapes") or ""
            if shape:
                agg.shapes[shape] += 1

    # ---- api_statistic.csv：runtime 层（Level=acl，P0-2 互斥分类）----
    runtime_aggs: dict[str, _RuntimeAgg] = {}
    other_api_levels: dict[str, int] = {}
    api_time_seen = False
    api_sum_us = 0.0
    launch_us = 0.0
    sync_us = 0.0
    alloc_us = 0.0
    other_us = 0.0
    launch_count = 0
    if "api_statistic" in present:
        for row in _rows(present["api_statistic"]):
            name = row.get("API Name") or ""
            if not name:
                continue
            level = row.get("Level") or ""
            if level != "acl":
                other_api_levels[level] = other_api_levels.get(level, 0) + 1
                continue
            agg = runtime_aggs.setdefault(name, _RuntimeAgg())
            count = _to_f(row.get("Count"))
            agg.calls += int(count) if count is not None else 1
            t = _to_f(row.get("Time(us)"))
            if t is not None:
                agg.total_us += t
                api_time_seen = True
                api_sum_us += t
                # P0-2：acl API 名关键字互斥分类（与 CUDA 规则同构，文档见 mapping doc）
                lname = name.lower()
                if "launch" in lname:
                    launch_us += t
                    launch_count += int(count) if count is not None else 1
                elif "synchronize" in lname:
                    sync_us += t
                elif "malloc" in lname or "free" in lname:
                    alloc_us += t
                else:
                    other_us += t

    # ---- operators 组装（framework -> runtime -> device，排序确定性）----
    operators: list[OperatorEvidence] = []
    for name in sorted(framework_aggs):
        agg = framework_aggs[name]
        operators.append(
            OperatorEvidence(
                id=f"op_{len(operators) + 1:06d}",
                name=name,
                layer=_LAYER_FRAMEWORK,
                parent_id=None,
                op_type="aten_op",
                # operator_details.csv 无 call count 列（mapping doc）；rows 是文件行数，不冒充 calls
                calls=None,
                # P0-3：Self/Total 四列各自成字段，语义不混
                host_self_ms=_us_to_ms(agg.host_self_us) if agg.host_self_seen else None,
                host_inclusive_ms=_us_to_ms(agg.host_us) if agg.host_seen else None,
                device_self_ms=_us_to_ms(agg.device_self_us) if agg.device_self_seen else None,
                device_inclusive_ms=_us_to_ms(agg.device_us) if agg.device_seen else None,
                shapes=[s for s, _ in agg.shapes.most_common()],
                evidence_ids=[file_evidences["operator_details"], ev_aten.id],
            )
        )
    for name in sorted(runtime_aggs):
        agg = runtime_aggs[name]
        operators.append(
            OperatorEvidence(
                id=f"op_{len(operators) + 1:06d}",
                name=name,
                layer=_LAYER_RUNTIME,
                parent_id=None,  # CSV 之间无关联键，不猜测
                op_type="cann_api",
                calls=agg.calls,
                summed_host_ms=_us_to_ms(agg.total_us) if agg.total_us else None,
                evidence_ids=[file_evidences["api_statistic"], ev_aten.id],
            )
        )
    for name in sorted(kernel_aggs):
        agg = kernel_aggs[name]
        operators.append(
            OperatorEvidence(
                id=f"op_{len(operators) + 1:06d}",
                name=name,
                layer=_LAYER_DEVICE,
                parent_id=None,
                op_type="npu_task",
                calls=agg.calls,
                summed_device_ms=_us_to_ms(agg.total_us) if agg.total_us else None,
                avg_device_us=(agg.total_us / agg.calls) if agg.calls else None,
                median_device_us=statistics.median(agg.durs) if agg.durs else None,
                shapes=[s for s, _ in agg.shape_counts.most_common()],
                evidence_ids=[file_evidences["kernel_details"], ev_aten.id],
            )
        )

    # ---- shapes（决议 2：fraction 仅在分母明确时给出）----
    shapes: list[ShapeAggregation] = []
    if task_sum_us > 0:
        for name in sorted(kernel_aggs):
            agg = kernel_aggs[name]
            for shape in sorted(agg.shape_counts):
                shape_total = agg.shape_us.get(shape, 0.0)
                has_time = shape in agg.shape_us
                shapes.append(
                    ShapeAggregation(
                        operator=name,
                        shape=shape,
                        calls=agg.shape_counts[shape],
                        total_ms=_us_to_ms(shape_total) if has_time else None,
                        device_time_fraction=(_us_to_ms(shape_total) / _us_to_ms(task_sum_us)) if has_time else None,
                        fraction_denominator=_FRACTION_DENOMINATOR if has_time else None,
                        profiler_ratio=None,
                        evidence_ids=[file_evidences["kernel_details"], ev_aten.id],
                    )
                )

    # ---- backend_metrics.ascend（原样保留，不翻译成 CUDA counter）----
    backend: dict = {
        "core_type_time_us": {k: core_type_time[k] for k in sorted(core_type_time)},
        "api_statistic_other_levels": other_api_levels,
    }
    if "step_trace" in present and step_seen:
        # P1-4A：Bubble/Preparing 落入 backend_metrics（mapping doc 承诺兑现）
        if "Bubble" in col_sums:
            backend["step_trace_bubble_us"] = col_sums["Bubble"]
        if "Preparing" in col_sums:
            backend["step_trace_preparing_us"] = col_sums["Preparing"]
    if "op_statistic" in present:
        backend["op_statistic"] = [row for row in _rows(present["op_statistic"]) if any(row.values())]

    ev = OptimizationEvidence(
        run=RunInfo(
            backend="ascend",
            profiler="torch_npu",
            rank=None,  # P0-1：CSV 无 rank 证据（真实样本 RANK_DEVICE_MAP: rankId=0, deviceId=8）
            device=None,  # 设备型号来自大 DB NPU_INFO，CSV 家族不携带
            device_id=device_id,  # 设备号，与 rank 严格分离
            source_files=sorted(p.name for p in present.values()),
            run_id=run_id,
        ),
        workload=workload,
        timeline=timeline,
        operators=operators,
        # P0-2：runtime API 互斥分类（launch/synchronization/allocation/other）
        runtime=RuntimeStats(
            api_summed_ms=_us_to_ms(api_sum_us) if api_time_seen else None,
            launch_summed_ms=_us_to_ms(launch_us) if api_time_seen else None,
            synchronization_summed_ms=_us_to_ms(sync_us) if api_time_seen else None,
            allocation_summed_ms=_us_to_ms(alloc_us) if api_time_seen else None,
            other_summed_ms=_us_to_ms(other_us) if api_time_seen else None,
            launch_count=launch_count if api_time_seen else None,
        ),
        communication=comm,
        shapes=shapes,
        backend_metrics=BackendMetrics(ascend=backend),
        provenance=store.records,
        metric_evidence=_metric_evidence(store, rec_by_key, device_event_seen, api_time_seen, step_seen),
    )
    return ev, store


def _metric_evidence(
    store: ProvenanceStore,
    rec_by_key: dict[str, object],
    device_event_seen: bool,
    api_time_seen: bool,
    step_seen: bool,
) -> dict[str, list[str]]:
    """P1-1：核心标量指标 -> 全局 evidence 引用；未观测的 metric 不绑定。"""
    ref = store.ref
    me: dict[str, list[str]] = {}
    if "step_trace" in rec_by_key and step_seen:
        step_ref = ref(rec_by_key["step_trace"].id)
        me["workload.wall_ms"] = [step_ref]
        me["timeline.compute_ms"] = [step_ref]
        me["timeline.communication_ms"] = [step_ref]
        me["timeline.exposed_non_device_busy_ms"] = [step_ref]
        me["communication.total_ms"] = [step_ref]
        me["communication.overlap_ms"] = [step_ref]
    if "kernel_details" in rec_by_key and device_event_seen:
        me["timeline.device_busy_ms"] = [ref(rec_by_key["kernel_details"].id)]
    if "api_statistic" in rec_by_key and api_time_seen:
        api_ref = ref(rec_by_key["api_statistic"].id)
        for cat in (
            "api_summed_ms",
            "launch_summed_ms",
            "synchronization_summed_ms",
            "allocation_summed_ms",
            "other_summed_ms",
            "launch_count",
        ):
            me[f"runtime.{cat}"] = [api_ref]
    return me
