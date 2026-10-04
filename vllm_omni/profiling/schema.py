# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# OptimizationEvidence 数据契约。schema 版本化；None 表示 unavailable，
# 禁止把未知填 0（AC-03）。时间口径字段名互不兼容（AC-06）。
#
# v0.2 变更（schema_decisions.md 决议 1-4）：
#   - operator 三层模型 layer=framework/runtime/device + parent_id（一对多）；
#   - ShapeAggregation 拆分 workload_fraction -> device_time_fraction + fraction_denominator
#     + profiler_ratio（不同 denominator 的 ratio 不得混用）；
#   - timeline 移除 memory_ms；host-device copy 进 memory.host_device_copy_ms；
#   - EvidenceRecord/RunInfo 增加 run_id 与 source_sha256（跨 run 引用）。
#
# v0.3 变更（ckpt-2 code-level audit P0-1/2/3/4）：
#   - RunInfo.device_id 与 run.rank 分离（Device_id 不是 rank；P0-1）；
#   - RuntimeStats.dispatch_ms（实为全部 runtime API 求和）改为互斥分类：
#     api_summed_ms = launch + synchronization + allocation + other（P0-2）；
#   - OperatorEvidence 时间语义按层拆分（P0-3）：framework 层
#     host_self/host_inclusive/device_self/device_inclusive；runtime/device 层
#     summed_host/summed_device。不同 semantics 禁止互相排序或相除；
#   - writer 侧 validation：构造（__post_init__）与序列化（to_dict）双向校验，
#     非法内存对象无法产出 JSON（P0-4）。
#
# v0.4 变更（M3 Observation layer，P1-1/P1-2）：
#   - OptimizationEvidence.metric_evidence: dict[str, list[str]] —— 核心标量指标的
#     field-level provenance binding（JSON-path 风格 metric key -> 全局 evidence 引用）；
#   - EvidenceRecord.depends_on: list[str] —— derived 记录显式声明依赖的 input evidence。
#
# v0.5 变更（M4.1 timeline correlation）：
#   - RuntimeStats.gap_overlap_ms: dict[str, float] —— 各 runtime API 分类落在
#     device exposed-gap 区间内的时长（ms，与 api_summed 同 source）。
#     这是把 runtime_*_heavy 升格为 *_bound 所需的时间相关性证据；
#     无时间轴数据的 source（如 Ascend CSV）保持 None，不伪造。
# 0.1..0.4 输入按显式 migration 规则读取，不 silent-drop（见 _migrate）。

from __future__ import annotations

import json
from dataclasses import dataclass, field

from vllm_omni.profiling.provenance import EvidenceRecord

SCHEMA_VERSION = "0.6"
READ_VERSIONS = ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6")

# operator 三层模型（决议 1）：framework_op -> runtime_op -> device_task(s)
OPERATOR_LAYERS = ("framework", "runtime", "device")

# 诊断候选的固定类别（SPEC 3.2/12），规则引擎只能输出其中的值
DIAGNOSIS_CLASSES = (
    "compute_bound",
    "memory_bound",
    "launch_bound",
    "host_dispatch_bound",
    "allocation_bound",
    "communication_bound",
    "synchronization_bound",
    "load_imbalance",
    "graph_or_compile_bound",
    "mixed",
    "unknown",
)

# confidence 用三档语义等级，不用伪精确小数（2026-09-29 裁定）
CONFIDENCE_LEVELS = ("high", "medium", "low")


@dataclass
class RunInfo:
    backend: str | None = None  # "cuda" / "ascend" / ...
    profiler: str | None = None  # "torch_profiler" / "torch_npu" / ...
    profiler_version: str | None = None
    rank: int | None = None  # 分布式 rank；无独立证据时必须为 None（P0-1）
    device: str | None = None  # 设备型号/名称（如 "Ascend910B"）
    device_id: str | None = None  # 设备编号（如 "8"）；与 rank 严格分离（P0-1）
    source_files: list[str] = field(default_factory=list)
    workload_id: str | None = None
    run_id: str | None = None  # analysis identity（内容寻址，决议 4）


@dataclass
class StageTime:
    name: str
    wall_ms: float | None = None


@dataclass
class Workload:
    wall_ms: float | None = None
    stage_times: list[StageTime] = field(default_factory=list)


@dataclass
class Timeline:
    # 时间口径分离（AC-06）：以下字段语义互斥，禁止互相推导后覆盖。
    # 决议 3：common timeline 只保留硬件无关的高层概念，memory 相关进 memory/backend_metrics
    device_busy_ms: float | None = None
    exposed_non_device_busy_ms: float | None = None
    compute_ms: float | None = None
    communication_ms: float | None = None
    unknown_ms: float | None = None


@dataclass
class OperatorEvidence:
    id: str
    name: str
    layer: str | None = None  # OPERATOR_LAYERS 之一
    parent_id: str | None = None  # 上层行 id；一对多 = 多个下层行指向同一 parent
    op_type: str | None = None
    calls: int | None = None
    # runtime/device 层：事件求和口径（summed）
    summed_host_ms: float | None = None
    summed_device_ms: float | None = None
    # framework 层：profiler 提供的 self/inclusive 口径（P0-3）
    host_self_ms: float | None = None
    host_inclusive_ms: float | None = None
    device_self_ms: float | None = None
    device_inclusive_ms: float | None = None
    avg_device_us: float | None = None
    median_device_us: float | None = None
    exposed_ms: float | None = None
    shapes: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_tree(self)


@dataclass
class RuntimeStats:
    # P0-2：runtime API 时间按互斥分类（同一 source/time domain，各分类之和 = api_summed）
    api_summed_ms: float | None = None
    launch_summed_ms: float | None = None
    synchronization_summed_ms: float | None = None
    allocation_summed_ms: float | None = None
    other_summed_ms: float | None = None
    launch_count: int | None = None
    # M4.1 时间相关性：各分类 API 区间与 device exposed-gap 区间的交集时长（ms）。
    # key: "launch"/"synchronization"/"allocation"/"other"。
    # 仅当 source 自带时间轴（CUDA trace / Ascend 大库）时由 adapter 计算；
    # 无时间轴的 source 保持 None，不得用 summed 值冒充（ckpt-3 审计 P0-2 的升格证据）。
    gap_overlap_ms: dict[str, float] | None = None


@dataclass
class MemoryStats:
    allocation_count: int | None = None
    free_count: int | None = None
    allocated_bytes: int | None = None
    size_histogram: dict[str, int] | None = None
    churn_ratio: float | None = None  # 0..1
    host_device_copy_ms: float | None = None  # host<->device 拷贝（summed）；决议 3


@dataclass
class CommunicationStats:
    calls: int | None = None
    total_ms: float | None = None
    overlap_ms: float | None = None
    overlap_ratio: float | None = None  # 0..1
    collectives: dict[str, int] | None = None  # collective type -> call count


@dataclass
class ShapeAggregation:
    # 决议 2：不同 denominator 的 ratio 不得映射成同一指标
    operator: str
    shape: str
    calls: int | None = None  # occurrences
    total_ms: float | None = None
    device_time_fraction: float | None = None  # 0..1，仅 denominator 明确时由框架计算
    fraction_denominator: str | None = None  # device_time_fraction 的分母语义描述
    profiler_ratio: float | None = None  # profiler 原样给出的比例，不解释
    evidence_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_tree(self)


@dataclass
class Hotspot:
    name: str
    kind: str | None = None  # "operator" / "runtime_api" / "device_task" / "communication" / ...
    measured_fraction: float | None = None  # 0..1，实测占比
    fraction_denominator: str | None = None  # measured_fraction 的分母语义（与 D2 同纪律）
    addressable_fraction: float | None = None  # 0..1，估计可优化占比（estimated）
    evidence_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_tree(self)


@dataclass
class DiagnosisCandidate:
    class_: str  # DIAGNOSIS_CLASSES 之一，JSON 键为 "class"
    confidence: str  # CONFIDENCE_LEVELS 之一
    evidence_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_tree(self)


@dataclass
class Diagnosis:
    candidates: list[DiagnosisCandidate] = field(default_factory=list)


@dataclass
class HardwareMetricEntry:
    """单条硬件计数器证据：(metric_name, metric_unit, metric_value) 原样三元组
    + canonical_value/canonical_unit 规范化结果（P0 unit 纪律：无 unit 的数值
    不可解释；规范化只做单位换算，不改语义）。
    scope 为 kernel 名；scope_id 为稳定的 per-invocation 身份（由 NCU 源字段
    Process ID/Device/Context/Stream/ID 组合，M4.2a gate P0-1），metric_evidence
    与观察的 metric_key 均以 scope_id 为准，不用数组位置。"""

    scope: str
    metric_name: str
    metric_unit: str | None = None
    metric_value: str | float | int | None = None
    canonical_value: float | None = None
    canonical_unit: str | None = None
    scope_id: str | None = None


@dataclass
class HardwareEvidence:
    """一个硬件计数器 capture 的 evidence（当前：NCU CSV；后续：Ascend 硬件）。

    entries 逐条原样保留 NCU 三元组；规范化只在 canonical_* 字段。
    未识别 unit/value（如 "n/a"）保持 canonical=None —— unavailable 纪律。"""

    backend: str | None = None
    source: str | None = None  # NCU CSV 文件名（sha256 进 provenance）
    provenance_ref: str | None = None  # 指向 provenance 记录的全局引用（P1-4 纪律）
    entries: list[HardwareMetricEntry] = field(default_factory=list)


@dataclass
class BackendMetrics:
    cuda: dict = field(default_factory=dict)
    ascend: dict = field(default_factory=dict)


@dataclass
class OptimizationEvidence:
    schema_version: str = SCHEMA_VERSION
    run: RunInfo = field(default_factory=RunInfo)
    workload: Workload = field(default_factory=Workload)
    timeline: Timeline = field(default_factory=Timeline)
    operators: list[OperatorEvidence] = field(default_factory=list)
    runtime: RuntimeStats = field(default_factory=RuntimeStats)
    memory: MemoryStats = field(default_factory=MemoryStats)
    communication: CommunicationStats = field(default_factory=CommunicationStats)
    shapes: list[ShapeAggregation] = field(default_factory=list)
    hotspots: list[Hotspot] = field(default_factory=list)
    diagnosis: Diagnosis = field(default_factory=Diagnosis)
    backend_metrics: BackendMetrics = field(default_factory=BackendMetrics)
    hardware: HardwareEvidence | None = None  # M4.2a：硬件计数器 evidence（None = 未采集）
    provenance: list[EvidenceRecord] = field(default_factory=list)
    # P1-1：核心标量指标 -> 全局 evidence 引用（"run_id:ev_id"）。
    # key 为 schema 内的 metric 路径，如 "timeline.device_busy_ms" / "runtime.api_summed_ms"
    metric_evidence: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # writer 侧 defense-in-depth：构造时递归校验（P0-4）
        _validate_tree(self)

    def to_dict(self) -> dict:
        # P0-5（ckpt-3）：序列化路径同样校验——构造后 mutate 出的非法对象必须在此被拒绝
        _validate_tree(self)
        return _dump(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> OptimizationEvidence:
        return _build(OptimizationEvidence, data)

    @classmethod
    def from_json(cls, text: str) -> OptimizationEvidence:
        return cls.from_dict(json.loads(text))


# ---- 校验（P0-4）：构造、序列化、反序列化三处共用 ----


# 按类型注册的校验器；新增带约束的类型只需注册，不需要扩 if 列表（P0-5）
_VALIDATORS: dict[type, object] = {}


def _register(cls: type) -> object:
    def deco(fn: object) -> object:
        _VALIDATORS[cls] = fn
        return fn

    return deco


@_register(OperatorEvidence)
def _v_operator(obj: OperatorEvidence) -> None:
    if obj.layer is not None and obj.layer not in OPERATOR_LAYERS:
        raise ValueError(f"schema: invalid operator layer {obj.layer!r}, expected {OPERATOR_LAYERS}")


@_register(ShapeAggregation)
def _v_shape(obj: ShapeAggregation) -> None:
    # 决议 2：fraction 与分母必须成对出现，禁止无分母的 device_time_fraction
    if (obj.device_time_fraction is None) != (obj.fraction_denominator is None):
        raise ValueError(
            "schema: device_time_fraction and fraction_denominator must be set together (denominator required)"
        )


@_register(Hotspot)
def _v_hotspot(obj: Hotspot) -> None:
    # D2 同纪律适用于 hotspot fraction
    if (obj.measured_fraction is None) != (obj.fraction_denominator is None):
        raise ValueError(
            "schema: measured_fraction and fraction_denominator must be set together (denominator required)"
        )


@_register(DiagnosisCandidate)
def _v_candidate(obj: DiagnosisCandidate) -> None:
    if obj.class_ not in DIAGNOSIS_CLASSES:
        raise ValueError(f"schema: invalid diagnosis class {obj.class_!r}")
    if obj.confidence not in CONFIDENCE_LEVELS:
        raise ValueError(f"schema: invalid confidence {obj.confidence!r}, expected {CONFIDENCE_LEVELS}")


def _validate_tree(obj: object) -> None:
    """泛化递归校验（P0-5）：遍历任意 dataclass 树，按注册表执行类型校验器。"""
    stack = [obj]
    while stack:
        node = stack.pop()
        validator = _VALIDATORS.get(type(node))
        if validator is not None:
            validator(node)
        if hasattr(node, "__dataclass_fields__"):
            for f in node.__dataclass_fields__.values():
                value = getattr(node, f.name)
                if hasattr(value, "__dataclass_fields__"):
                    stack.append(value)
                elif isinstance(value, list):
                    stack.extend(v for v in value if hasattr(v, "__dataclass_fields__"))
                elif isinstance(value, dict):
                    stack.extend(v for v in value.values() if hasattr(v, "__dataclass_fields__"))


# ---- 序列化实现：手写映射保持键名一致，未知键显式失败 ----


def _json_key(field_name: str) -> str:
    # DiagnosisCandidate.class_ 在 JSON 中命名为 "class"
    return "class" if field_name == "class_" else field_name


def _dump(obj: object) -> object:
    if isinstance(obj, list):
        return [_dump(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {_json_key(f.name): _dump(getattr(obj, f.name)) for f in obj.__dataclass_fields__.values()}
    return obj


# 反序列化时按宿主类区分的子结构类型；未列出的字段按标量处理。
_CHILD_LIST_TYPES: dict[type, dict[str, type]] = {
    OptimizationEvidence: {
        "operators": OperatorEvidence,
        "shapes": ShapeAggregation,
        "hotspots": Hotspot,
        "provenance": EvidenceRecord,
    },
    Workload: {"stage_times": StageTime},
    Diagnosis: {"candidates": DiagnosisCandidate},
    HardwareEvidence: {"entries": HardwareMetricEntry},
}
_SECTION_TYPES: dict[str, type] = {
    "run": RunInfo,
    "workload": Workload,
    "timeline": Timeline,
    "runtime": RuntimeStats,
    "memory": MemoryStats,
    "communication": CommunicationStats,
    "diagnosis": Diagnosis,
    "backend_metrics": BackendMetrics,
    "hardware": HardwareEvidence,
}


def _migrate(data: dict, from_version: str) -> dict:
    """0.1..0.4 -> 0.5 显式迁移；保留所有原值，不 silent-drop。"""
    if from_version == "0.1":
        data = _migrate_0_1(data)
        from_version = "0.2"
    if from_version == "0.2":
        data = _migrate_0_2(data)
        from_version = "0.3"
    if from_version in ("0.3", "0.4", "0.5"):
        # 0.4 metric_evidence/depends_on、0.5 gap_overlap_ms、0.6 hardware
        # 均为空默认（历史文件无法补充绑定，不编造）
        data["schema_version"] = SCHEMA_VERSION
    return data


def _migrate_0_1(data: dict) -> dict:
    """0.1 -> 0.2：layer 推断、memory/copy 迁移、shape fraction 补分母。"""
    migrated = dict(data)
    migrated["operators"] = [dict(op) for op in migrated.get("operators", [])]
    migrated["shapes"] = [dict(s) for s in migrated.get("shapes", [])]
    migrated["timeline"] = dict(migrated.get("timeline", {}))
    migrated["runtime"] = dict(migrated.get("runtime", {}))
    migrated["memory"] = dict(migrated.get("memory", {}))
    for op in migrated["operators"]:
        op["layer"] = {"aten_op": "framework", "kernel": "device"}.get(op.get("op_type"))
    legacy_mem = migrated["timeline"].pop("memory_ms", None)
    legacy_copy = migrated["runtime"].pop("copy_ms", None)
    if legacy_copy is not None:
        migrated["memory"]["host_device_copy_ms"] = legacy_copy
    if legacy_mem is not None:
        # union 口径禁止冒名迁入 summed 字段；原值标记 legacy 保留
        migrated.setdefault("backend_metrics", {}).setdefault("cuda", {})["legacy_memory_union_ms"] = legacy_mem
    for shape in migrated["shapes"]:
        wf = shape.pop("workload_fraction", None)
        shape["device_time_fraction"] = wf
        shape["fraction_denominator"] = "summed_device_time_all_device_tasks" if wf is not None else None
    migrated["schema_version"] = "0.2"
    return migrated


def _migrate_0_2(data: dict) -> dict:
    """0.2 -> 0.3：RuntimeStats 互斥分类 + framework 时间语义拆分 + device_id（P0-1/2/3）。"""
    migrated = dict(data)
    migrated["operators"] = [dict(op) for op in migrated.get("operators", [])]
    runtime = dict(migrated.get("runtime", {}))
    legacy_dispatch = runtime.pop("dispatch_ms", None)
    if legacy_dispatch is not None:
        # 0.2 的 dispatch_ms 实际是全部 runtime API 求和（P0-2 审计确认），映射到 api_summed
        runtime["api_summed_ms"] = legacy_dispatch
    runtime.pop("synchronization_ms", None)  # 0.2 中与 dispatch 重复计数，无法恢复互斥分类 -> 不迁移
    migrated["runtime"] = runtime
    backend = migrated.get("run", {}).get("backend")
    for op in migrated["operators"]:
        if op.get("layer") in ("runtime", "device"):
            op["summed_host_ms"] = op.pop("total_host_ms", None)
            op["summed_device_ms"] = op.pop("total_device_ms", None)
        elif op.get("layer") == "framework":
            # host Total 在两后端都是 inclusive（CUDA chrome dur / Ascend Host Total）
            op["host_inclusive_ms"] = op.pop("total_host_ms", None)
            dev = op.pop("total_device_ms", None)
            # CUDA 的关联 kernel 求和是 leaf 归属（self 语义）；
            # Ascend operator_details 的 Device Total 是 inclusive（P0-3 审计确认）
            if backend == "ascend":
                op["device_inclusive_ms"] = dev
            else:
                op["device_self_ms"] = dev
        else:
            op.pop("total_host_ms", None)
            op.pop("total_device_ms", None)
    migrated["schema_version"] = SCHEMA_VERSION
    return migrated


def _build(cls, data: dict):
    if cls is OptimizationEvidence:
        version = data.get("schema_version")
        if version not in READ_VERSIONS:
            raise ValueError(
                f"schema: unsupported evidence schema_version {version!r}, "
                f"supported read versions: {list(READ_VERSIONS)}"
            )
        if version != SCHEMA_VERSION:
            data = _migrate(data, version)
    known = {_json_key(f.name) for f in cls.__dataclass_fields__.values()}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"schema: unknown fields {sorted(unknown)} for {cls.__name__} "
            f"(schema_version {SCHEMA_VERSION}); refusing silent drop"
        )
    child_lists = _CHILD_LIST_TYPES.get(cls, {})
    kwargs = {}
    for f in cls.__dataclass_fields__.values():
        key = _json_key(f.name)
        if key not in data:
            continue
        value = data[key]
        if f.name in child_lists and value is not None:
            item_cls = child_lists[f.name]
            kwargs[f.name] = [_build(item_cls, item) for item in value]
        elif f.name in _SECTION_TYPES and value is not None:
            kwargs[f.name] = _build(_SECTION_TYPES[f.name], value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)
