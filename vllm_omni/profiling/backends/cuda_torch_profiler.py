# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# CUDA PyTorch profiler adapter（T10/T13）。
# 输入：OmniTorchProfilerWrapper 导出的 trace_rank*.json[.gz]（Chrome trace 格式）。
# 规则（全部确定性、写入 provenance）：
#   - 三层模型（决议 1）：framework(cpu_op) -> runtime(cuda_runtime) -> device(kernel/memcpy)；
#     关联键为 args["External id"]；聚合行仅在归属唯一时填 parent_id；
#   - communication kernel 识别规则：kernel 名包含 "nccl"（大小写不敏感）；
#   - timeline 全部为区间 union 语义；memory 相关进 memory.host_device_copy_ms（决议 3）；
#   - run_id 由 source 内容 hash 派生（决议 4，确定性）。

from __future__ import annotations

import gzip
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from vllm_omni.profiling.discovery import check_size
from vllm_omni.profiling.intervals import Interval, subtract, total, union
from vllm_omni.profiling.provenance import ProvenanceStore, compute_run_id, sha256_file
from vllm_omni.profiling.schema import (
    MemoryStats,
    OperatorEvidence,
    OptimizationEvidence,
    RunInfo,
    RuntimeStats,
    ShapeAggregation,
    Timeline,
    Workload,
)

# Chrome trace 事件类别
_CAT_CPU_OP = "cpu_op"
_CAT_KERNEL = "kernel"
_CAT_MEMCPY = "gpu_memcpy"
_CAT_MEMSET = "gpu_memset"
_CAT_RUNTIME = "cuda_runtime"

_LAYER_FRAMEWORK = "framework"
_LAYER_RUNTIME = "runtime"
_LAYER_DEVICE = "device"

# runtime API 互斥分类规则（显式清单，不猜测；P0-2）
_LAUNCH_KEYWORD = "launch"
_SYNC_NAMES = (
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cudaStreamWaitEvent",
)
_ALLOC_PREFIXES = ("cudaMalloc", "cudaFree", "cudaHostAlloc", "cudaHostRegister")
_COMM_KEYWORD = "nccl"

_RANK_RE = re.compile(r"rank(\d+)")

# shape fraction 的唯一合法分母（决议 2）：全部 device task 的 summed 时长
_FRACTION_DENOMINATOR = "summed_device_time_all_device_tasks"


def _open_trace(path: Path) -> dict:
    check_size(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _format_dims(dims: object) -> str | None:
    # torch profiler "Input Dims" 形如 [[2,4096],[4096,3584]]
    if not isinstance(dims, list) or not dims:
        return None
    parts = []
    for tensor in dims:
        if isinstance(tensor, (list, tuple)):
            parts.append("[" + ",".join(str(d) for d in tensor) + "]")
        else:
            parts.append(str(tensor))
    return "x".join(parts) if parts else None


def _is_comm_kernel(name: str) -> bool:
    return _COMM_KEYWORD in name.lower()


class _Agg:
    __slots__ = ("name", "calls", "total_us", "durs", "shape_counts", "external_ids")

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0
        self.total_us = 0.0
        self.durs: list[float] = []
        self.shape_counts: Counter[str] = Counter()
        self.external_ids: set[str] = set()


def analyze_cuda_torch_profiler(trace_path: Path) -> tuple[OptimizationEvidence, ProvenanceStore]:
    """解析单 rank trace 文件，返回 evidence 与 provenance store。"""
    trace_path = Path(trace_path)
    source_sha = sha256_file(trace_path)
    run_id = compute_run_id([source_sha], backend="cuda", parser="cuda.torch_profiler")
    store = ProvenanceStore(run_id=run_id)

    rec_file = store.add(
        source_file=trace_path.name,
        source_type="torch_profiler_trace",
        parser="cuda.torch_profiler",
        unit="us",
        query="traceEvents[ph=='X']",
        aggregation=None,
        source_sha256=source_sha,
    )
    ev_aten = store.add(
        source_file=trace_path.name,
        source_type="torch_profiler_trace",
        parser="cuda.torch_profiler",
        unit="us",
        query="cpu_op + kernel + cuda_runtime by External id",
        aggregation="group_by(name); median; sum",
        source_sha256=source_sha,
        depends_on=[rec_file.id],
    )
    rec_tl = store.add(
        source_file=trace_path.name,
        source_type="torch_profiler_trace",
        parser="cuda.torch_profiler",
        unit="us",
        query="kernel/gpu_memcpy/gpu_memset intervals",
        aggregation="union_intervals",
        source_sha256=source_sha,
        depends_on=[rec_file.id],
    )
    rec_wall = store.add(
        source_file=trace_path.name,
        source_type="torch_profiler_trace",
        parser="cuda.torch_profiler",
        unit="us",
        query="min(ts)..max(ts+dur)",
        aggregation=None,
        source_sha256=source_sha,
        depends_on=[rec_file.id],
    )
    rec_rt = store.add(
        source_file=trace_path.name,
        source_type="torch_profiler_trace",
        parser="cuda.torch_profiler",
        unit="us",
        query="cuda_runtime events",
        aggregation="sum; classify(launch/synchronization/allocation/other)",
        source_sha256=source_sha,
        depends_on=[rec_file.id],
    )

    # ---- 单遍收集 ----
    cpu_aggs: dict[str, _Agg] = {}
    kernel_aggs: dict[str, _Agg] = {}
    runtime_aggs: dict[tuple[str | None, str], _Agg] = {}
    ext_to_op: dict[str, str] = {}
    ext_to_shape: dict[str, str] = {}
    # kernel 时长按 external id 归集，用于关联到 aten op
    kern_durs_by_ext: dict[str, list[float]] = defaultdict(list)
    kern_ints: list[Interval] = []
    comm_ints: list[Interval] = []
    copy_ints: list[Interval] = []
    all_ints: list[Interval] = []
    wall_min: float | None = None
    wall_max: float | None = None
    launch_count = 0
    api_sum_us = 0.0
    launch_us = 0.0
    sync_us = 0.0
    alloc_us = 0.0
    other_us = 0.0
    copy_sum_us = 0.0
    kern_sum_us = 0.0
    device_event_seen = False

    data = _open_trace(trace_path)
    for e in json_events(data):
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        ts, dur = e.get("ts"), e.get("dur")
        if not isinstance(ts, (int, float)) or not isinstance(dur, (int, float)) or dur < 0:
            continue
        name = e.get("name")
        if not isinstance(name, str):
            continue
        cat = e.get("cat")
        end = ts + dur
        wall_min = ts if wall_min is None else min(wall_min, ts)
        wall_max = end if wall_max is None else max(wall_max, end)

        if cat == _CAT_CPU_OP:
            agg = cpu_aggs.setdefault(name, _Agg(name))
            agg.calls += 1
            agg.total_us += dur
            agg.durs.append(dur)
            args = e.get("args") or {}
            ext = args.get("External id")
            if ext is not None:
                key = str(ext)
                agg.external_ids.add(key)
                ext_to_op.setdefault(key, name)
            shape = _format_dims(args.get("Input Dims"))
            if shape:
                agg.shape_counts[shape] += 1
                if ext is not None:
                    ext_to_shape.setdefault(key, shape)
        elif cat in (_CAT_KERNEL, _CAT_MEMCPY, _CAT_MEMSET):
            device_event_seen = True
            iv: Interval = (float(ts), float(end))
            all_ints.append(iv)
            if cat == _CAT_KERNEL:
                kern_ints.append(iv)
                if _is_comm_kernel(name):
                    comm_ints.append(iv)
                agg = kernel_aggs.setdefault(name, _Agg(name))
                agg.calls += 1
                agg.total_us += dur
                agg.durs.append(dur)
                ext = (e.get("args") or {}).get("External id")
                if ext is not None:
                    kern_durs_by_ext[str(ext)].append(dur)
                    agg.external_ids.add(str(ext))
                kern_sum_us += dur
            else:
                copy_ints.append(iv)
                copy_sum_us += dur
        elif cat == _CAT_RUNTIME:
            # P0-2：每个 runtime 事件归入恰好一个互斥分类（if/elif 保证）
            api_sum_us += dur
            lname = name.lower()
            if _LAUNCH_KEYWORD in lname:
                launch_us += dur
                launch_count += 1
            elif name in _SYNC_NAMES:
                sync_us += dur
            elif name.startswith(_ALLOC_PREFIXES):
                alloc_us += dur
            else:
                other_us += dur
            ext = (e.get("args") or {}).get("External id")
            parent_name = ext_to_op.get(str(ext)) if ext is not None else None
            agg = runtime_aggs.setdefault((parent_name, name), _Agg(name))
            agg.calls += 1
            agg.total_us += dur
            agg.durs.append(dur)

    # ---- timeline（union 语义，决议 3：无 memory 字段）----
    timeline = Timeline()
    workload = Workload()
    if wall_min is not None and wall_max is not None:
        workload.wall_ms = (wall_max - wall_min) / 1000.0
    memory = MemoryStats()
    if device_event_seen:
        busy_iv = union(all_ints)
        comm_iv = union(comm_ints)
        compute_iv = subtract(union(kern_ints), comm_iv)
        unknown_iv = subtract(busy_iv, union(compute_iv + comm_ints + copy_ints))
        timeline.device_busy_ms = total(busy_iv) / 1000.0
        timeline.compute_ms = total(compute_iv) / 1000.0
        timeline.communication_ms = total(comm_iv) / 1000.0
        timeline.unknown_ms = total(unknown_iv) / 1000.0
        if workload.wall_ms is not None:
            timeline.exposed_non_device_busy_ms = workload.wall_ms - timeline.device_busy_ms
        memory.host_device_copy_ms = copy_sum_us / 1000.0

    # ---- operator 聚合：framework 层（aten，device 经 External id 关联）----
    operators: list[OperatorEvidence] = []
    framework_id_by_name: dict[str, str] = {}
    for name in sorted(cpu_aggs):
        agg = cpu_aggs[name]
        dev_durs: list[float] = []
        for ext in sorted(agg.external_ids):
            dev_durs.extend(kern_durs_by_ext.get(ext, []))
        common_shapes = [s for s, _ in agg.shape_counts.most_common()]
        op_id = f"op_{len(operators) + 1:06d}"
        framework_id_by_name[name] = op_id
        operators.append(
            OperatorEvidence(
                id=op_id,
                name=name,
                layer=_LAYER_FRAMEWORK,
                parent_id=None,
                op_type="aten_op",
                calls=agg.calls,
                # P0-3 语义：chrome cpu_op dur 含子 op = inclusive；
                # External id 关联的 kernel 求和是 leaf 归属 = self 语义
                host_inclusive_ms=agg.total_us / 1000.0,
                device_self_ms=(sum(dev_durs) / 1000.0) if dev_durs else None,
                avg_device_us=(sum(dev_durs) / len(dev_durs)) if dev_durs else None,
                median_device_us=statistics.median(dev_durs) if dev_durs else None,
                exposed_ms=None,
                shapes=common_shapes,
                evidence_ids=[store.records[0].id, ev_aten.id],
            )
        )

    # ---- runtime 层（cuda_runtime API，按 (parent framework op, name) 聚合）----
    for parent_name, name in sorted(runtime_aggs, key=lambda k: (k[0] or "", k[1])):
        agg = runtime_aggs[(parent_name, name)]
        operators.append(
            OperatorEvidence(
                id=f"op_{len(operators) + 1:06d}",
                name=name,
                layer=_LAYER_RUNTIME,
                parent_id=framework_id_by_name.get(parent_name) if parent_name else None,
                op_type="cuda_runtime_api",
                calls=agg.calls,
                summed_host_ms=agg.total_us / 1000.0,
                avg_device_us=None,
                median_device_us=None,
                exposed_ms=None,
                shapes=[],
                evidence_ids=[store.records[0].id, ev_aten.id],
            )
        )

    # ---- device 层（kernel 名聚合；parent 仅在归属唯一时填写，一对多由多行表达）----
    for name in sorted(kernel_aggs):
        agg = kernel_aggs[name]
        parent_names = {ext_to_op[ext] for ext in agg.external_ids if ext in ext_to_op}
        parent_id = framework_id_by_name.get(next(iter(parent_names))) if len(parent_names) == 1 else None
        operators.append(
            OperatorEvidence(
                id=f"op_{len(operators) + 1:06d}",
                name=name,
                layer=_LAYER_DEVICE,
                parent_id=parent_id,
                op_type="kernel",
                calls=agg.calls,
                summed_device_ms=agg.total_us / 1000.0,
                avg_device_us=agg.total_us / agg.calls if agg.calls else None,
                median_device_us=statistics.median(agg.durs) if agg.durs else None,
                exposed_ms=None,
                shapes=[],
                evidence_ids=[store.records[0].id, ev_aten.id],
            )
        )

    # ---- shape 聚合（决议 2）：fraction 分母显式记录，kernel 时长按 (op, shape) 归集 ----
    shape_acc: dict[tuple[str, str], list] = {}
    for ext, durs in kern_durs_by_ext.items():
        op_name = ext_to_op.get(ext)
        shape = ext_to_shape.get(ext)
        if op_name is None or shape is None:
            continue
        cell = shape_acc.setdefault((op_name, shape), [0, 0.0])
        cell[0] += len(durs)
        cell[1] += sum(durs)
    shapes = [
        ShapeAggregation(
            operator=op_name,
            shape=shape,
            calls=cell[0],
            total_ms=cell[1] / 1000.0,
            device_time_fraction=(cell[1] / kern_sum_us) if kern_sum_us > 0 else None,
            fraction_denominator=_FRACTION_DENOMINATOR if kern_sum_us > 0 else None,
            profiler_ratio=None,
            evidence_ids=[store.records[0].id, ev_aten.id],
        )
        for (op_name, shape), cell in sorted(shape_acc.items())
    ]

    ev = OptimizationEvidence(
        run=RunInfo(
            backend="cuda",
            profiler="torch_profiler",
            rank=_rank_from_name(trace_path.name),
            device=_device_from_metadata(data),
            source_files=[trace_path.name],
            run_id=run_id,
        ),
        workload=workload,
        timeline=timeline,
        operators=operators,
        runtime=RuntimeStats(
            launch_count=launch_count,
            api_summed_ms=api_sum_us / 1000.0,
            launch_summed_ms=launch_us / 1000.0,
            synchronization_summed_ms=sync_us / 1000.0,
            allocation_summed_ms=alloc_us / 1000.0,
            other_summed_ms=other_us / 1000.0,
        ),
        memory=memory,
        shapes=shapes,
        provenance=store.records,
        metric_evidence={
            "workload.wall_ms": [store.ref(rec_wall.id)],
            "timeline.device_busy_ms": [store.ref(rec_tl.id)],
            "timeline.compute_ms": [store.ref(rec_tl.id)],
            "timeline.communication_ms": [store.ref(rec_tl.id)],
            "timeline.unknown_ms": [store.ref(rec_tl.id)],
            "timeline.exposed_non_device_busy_ms": [store.ref(rec_tl.id), store.ref(rec_wall.id)],
            "runtime.api_summed_ms": [store.ref(rec_rt.id)],
            "runtime.launch_summed_ms": [store.ref(rec_rt.id)],
            "runtime.synchronization_summed_ms": [store.ref(rec_rt.id)],
            "runtime.allocation_summed_ms": [store.ref(rec_rt.id)],
            "runtime.other_summed_ms": [store.ref(rec_rt.id)],
            "runtime.launch_count": [store.ref(rec_rt.id)],
            "memory.host_device_copy_ms": [store.ref(rec_tl.id)],
        },
    )
    return ev, store


def json_events(data: dict) -> list:
    events = data.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("cuda adapter: trace file has no traceEvents list; unsupported format")
    return events


def _rank_from_name(name: str) -> int | None:
    m = _RANK_RE.search(name)
    return int(m.group(1)) if m else None


def _device_from_metadata(data: dict) -> str | None:
    props = data.get("deviceProperties")
    if isinstance(props, list) and props and isinstance(props[0], dict):
        name = props[0].get("name")
        if isinstance(name, str):
            return name
    return None
