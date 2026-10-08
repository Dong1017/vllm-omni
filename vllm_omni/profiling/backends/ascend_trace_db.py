# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Ascend 大库 timeline correlation adapter（M4.1b，final-review 修订）。
# 输入：ascend_pytorch_profiler_0.db（ascend_pytorch_profiler 主库，459MB 级）。
# 关联链（真实样本勘察 2026-10-02/03，connectionId 直连，无名字/时间戳猜测）：
#   PYTORCH_API(type=50001, name→STRING_IDS=aten::) --connectionId--> CANN_API
#     --connectionId--> TASK(globalTaskId→COMPUTE_TASK_INFO.name→STRING_IDS)
# final-review P0 修订：
#   P0-1 parent 归属：聚合 device 行的全部实例都必须 resolve 到唯一 framework op，
#        任一 unresolved 实例 -> parent_id=None（不再"部分未知也归唯一已知"）；
#   P0-2 TASK 的 parent 归因必须先过 CANN_API 中间链：TASK.conn ∈ cann_conn_ids
#        才允许经同 conn 的 PYTORCH(type=50001) 解析 framework 名（真实样本
#        TASK.conn ⊆ CANN.conn = 297330/297332，2 个未链接 conn 如实排除）；
#   P0-3 framework 层时间落 host_inclusive_ms（operator_details.csv 自带
#        Host Self/Host Total 双列 -> 行时长为 inclusive 语义，有据）。
# P1 修订：时间戳保持 int ns（1.787e18 量级禁止 float 有损转换，IEEE double
#   相邻间隔 ~256ns）；COMPUTE_TASK_INFO 进 schema 必需表；NPU_INFO 按
#   deviceId 查；provenance 拆 rec_session/rec_cann/rec_task/rec_pytorch
#   四条（D11：同 DB ≠ 自动同 timeline，metric 按输入显式绑定）。
# 哨兵发现：TASK.connectionId=-1 共 267,873 行（未关联任务），不参与 parent
#   链（conn 不在 CANN_API）；calls/时长如实进未解析聚合行。
# M4.1c（Ascend authoritative timeline semantics，2026-10-07 gate 裁定）：
#   conn=-1 调查（m41b-conn-neg1-investigation.md）证明 raw TASK union 不是
#   canonical device activity——26.7 万行 conn=-1 是 per-stream capture-session
#   哨兵，PROFILER 自己的 OVERLAP_ANALYSIS 预分类与 step_trace 精确一致：
#   type 0=compute / 1=communication / 2=communication not overlapped /
#   3=free（官方语义，勿按 0..3 顺序猜）。裁定五条：
#   1) conn=-1 永不 parent 归因、永不进 named operator 的 summed_device_ms；
#   2) device_busy = OVERLAP_ANALYSIS type0∪type1 的 interval union
#      （compute/communication 允许重叠，禁止 summed 相加冒充 busy）；
#   3) type3=Free 是 profiler 定义的 device idle 证据（exposed gap 同源）；
#   4) type2=Communication(Not Overlapped) 是 communication exposure 证据：
#      total(type1) = not_overlapped(type2) + overlapped，三者不是 wall partition；
#   5) SESSION_TIME_INFO 是 capture session 窗口，不自动等同 workload wall；
#      overlap-analysis 分类窗口与 session 窗口的差值如实记录（真实样本
#      111.049s vs 124.550s，差 13.5s 为 capture 初始化/未覆盖区间）。
#   gate P0（2026-10-08，semantic）：DB 无独立 step/workload boundary，不能证明
#      分类窗口是用户 workload wall——span/coverage 写 timeline.window_ms
#      （schema v0.7），workload.wall_ms 保持 None；观察层比值优先 window_ms，
#      unavailable 才回退 workload.wall_ms（CUDA/MVP 兼容）。措辞纪律：type0/1
#      可 overlap、type2 ⊂ type1，不得称四类"互斥 partition"；OVERLAP_ANALYSIS
#      provides the profiler timeline; in the validated sample, known busy union
#      (compute∪communication) and Free partition the observed window.
#   OVERLAP_ANALYSIS 表存在 -> authoritative path；不存在 -> task_union
#   fallback（排除 conn=-1 后 union），timeline_source 显式标记，不 silent。

from __future__ import annotations

import sqlite3
import statistics
from pathlib import Path

from vllm_omni.profiling.intervals import (
    Interval,
    complement,
    total_event_overlap,
    union,
)
from vllm_omni.profiling.intervals import total as intervals_total
from vllm_omni.profiling.provenance import EvidenceRecord, ProvenanceStore, compute_run_id, sha256_file
from vllm_omni.profiling.schema import (
    BackendMetrics,
    CommunicationStats,
    OperatorEvidence,
    OptimizationEvidence,
    RunInfo,
    RuntimeStats,
    Timeline,
    Workload,
)

_PARSER = "ascend.trace_db"

_TRACE_DB_REQUIRED: dict[str, set[str]] = {
    "PYTORCH_API": {"startNs", "endNs", "connectionId", "name", "type"},
    "CANN_API": {"startNs", "endNs", "connectionId", "name"},
    "TASK": {"startNs", "endNs", "deviceId", "connectionId", "globalTaskId", "taskType"},
    "COMPUTE_TASK_INFO": {"globalTaskId", "name"},
    "RANK_DEVICE_MAP": {"rankId", "deviceId"},
    "NPU_INFO": {"id", "name"},
    "SESSION_TIME_INFO": {"startTimeNs", "endTimeNs"},
    "STRING_IDS": {"id", "value"},
}

_TRACE_CATEGORIES = ("launch", "synchronization", "allocation", "other")

_PYTORCH_TYPE_ATEN = 50001  # 实测：type 50001=aten 调用、50002=Enqueue@、50003=profiler 事件

# OVERLAP_ANALYSIS（M4.1c authoritative timeline source，真实 459MB 库勘察）：
#   列 id/deviceId/startNs/endNs/type；type 语义来自 profiler 官方
#   Overlap Analysis 定义（与 step_trace_time.csv 精确对照核实，勿增删）。
_OVERLAP_REQUIRED_COLS = {"startNs", "endNs", "type"}
_OVERLAP_COMPUTE = 0
_OVERLAP_COMMUNICATION = 1
_OVERLAP_COMM_NOT_OVERLAPPED = 2
_OVERLAP_FREE = 3
_TASK_SENTINEL_CONN = -1  # conn=-1 TASK = per-stream capture-session 哨兵（非 device work）


class UnsupportedTraceDBError(ValueError):
    pass


def _open_trace_db(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise UnsupportedTraceDBError(f"trace DB not found: {path}")
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    existing = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = set(_TRACE_DB_REQUIRED) - existing
    if missing:
        con.close()
        raise UnsupportedTraceDBError(f"unsupported trace DB schema: missing tables {sorted(missing)} in {path.name}")
    for table, required_cols in _TRACE_DB_REQUIRED.items():
        cols = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
        miss = required_cols - cols
        if miss:
            con.close()
            raise UnsupportedTraceDBError(f"unsupported trace DB schema: table {table} missing columns {sorted(miss)}")
    return con


def _overlap_table_ready(con: sqlite3.Connection) -> bool:
    """OVERLAP_ANALYSIS 探测（T26 纪律）：表缺失 -> fallback；表在但缺列 -> 显式
    unsupported（存在即承诺了 schema，半 schema 不允许静默跳过）。"""
    existing = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "OVERLAP_ANALYSIS" not in existing:
        return False
    cols = {r[1] for r in con.execute('PRAGMA table_info("OVERLAP_ANALYSIS")')}
    miss = _OVERLAP_REQUIRED_COLS - cols
    if miss:
        raise UnsupportedTraceDBError(
            f"unsupported trace DB schema: table OVERLAP_ANALYSIS missing columns {sorted(miss)}"
        )
    return True


def _classify(name: str) -> str:
    """CANN API 名关键字互斥分类（与 M4.1 CSV 适配器同规则）。"""
    n = name.lower()
    if "launch" in n:
        return "launch"
    if "synchronize" in n:
        return "synchronization"
    if "malloc" in n or "free" in n:
        return "allocation"
    return "other"


class _DeviceAgg:
    """设备名聚合：parent 仅在全部实例 resolve 且指向同一 framework op 时给出。"""

    __slots__ = ("calls", "total_us", "durs", "fw_names", "resolved", "unresolved")

    def __init__(self) -> None:
        self.calls = 0
        self.total_us = 0.0
        self.durs: list[float] = []
        self.fw_names: set[str] = set()
        self.resolved = 0
        self.unresolved = 0


class _RuntimeAgg:
    __slots__ = ("calls", "total_us")

    def __init__(self) -> None:
        self.calls = 0
        self.total_us = 0.0


def analyze_ascend_trace_db(db_path: Path) -> tuple[OptimizationEvidence, ProvenanceStore]:
    """M4.1b：大库时间轴 -> CANN 分类区间 / TASK busy union / exposed-gap /
    runtime 分类相关性（gap_overlap_ms），证据链三层 connectionId 直连。"""
    db_path = Path(db_path)
    con = _open_trace_db(db_path)
    try:
        sha = sha256_file(db_path)
        run_id = compute_run_id([sha], backend="ascend", parser=_PARSER)
        store = ProvenanceStore(run_id=run_id)

        # P1-4：provenance 按输入源拆分（同 DB ≠ 自动同 timeline）
        rec_session = store.add(
            source_file=db_path.name,
            source_type="profiler_db",
            parser=_PARSER,
            unit="ns",
            query="sqlite:SESSION_TIME_INFO",
            aggregation=None,
            source_sha256=sha,
        )
        rec_pytorch = store.add(
            source_file=db_path.name,
            source_type="profiler_db",
            parser=_PARSER,
            unit="ns",
            query="sqlite:PYTORCH_API (type=50001)",
            aggregation="group_by(name); sum",
            source_sha256=sha,
        )
        rec_cann = store.add(
            source_file=db_path.name,
            source_type="profiler_db",
            parser=_PARSER,
            unit="ns",
            query="sqlite:CANN_API",
            aggregation="group_by(name); sum; per-category intervals",
            source_sha256=sha,
        )
        rec_task = store.add(
            source_file=db_path.name,
            source_type="profiler_db",
            parser=_PARSER,
            unit="ns",
            query="sqlite:TASK (LEFT JOIN COMPUTE_TASK_INFO)",
            aggregation="group_by(resolved name); sum; union_intervals",
            source_sha256=sha,
        )

        # rank/device（P0-1）：rankId 来自 RANK_DEVICE_MAP；多 rank 显式拒绝
        rank_rows = con.execute("SELECT rankId, deviceId FROM RANK_DEVICE_MAP").fetchall()
        rank_ids = {r[0] for r in rank_rows}
        if len(rank_ids) > 1:
            raise UnsupportedTraceDBError(
                f"multi-rank trace DB ({len(rank_ids)} rankIds) is out of M4.1b scope; filter to a single rank first"
            )
        rank = next(iter(rank_ids)) if rank_ids else None
        device_id = str(rank_rows[0][1]) if rank_rows else None
        device_name = con.execute(
            "SELECT name FROM NPU_INFO WHERE id = ?", (rank_rows[0][1] if rank_rows else None,)
        ).fetchone()

        session = con.execute("SELECT startTimeNs, endTimeNs FROM SESSION_TIME_INFO").fetchone()
        wall_ns = session[1] - session[0] if session and None not in session else None

        # ---- M4.1c：OVERLAP_ANALYSIS authoritative timeline 探测与读取 ----
        has_overlap = _overlap_table_ready(con)
        if has_overlap:
            rec_overlap = store.add(
                source_file=db_path.name,
                source_type="profiler_db",
                parser=_PARSER,
                unit="ns",
                query="sqlite:OVERLAP_ANALYSIS",
                aggregation="type 0/1 union_intervals; type sums; window span",
                source_sha256=sha,
            )
        ov_intervals: dict[int, list[Interval]] = {}
        ov_all_ints: list[Interval] = []
        ov_unknown_ints: list[Interval] = []
        ov_sum_ns: dict[int, int] = {}
        ov_calls: dict[int, int] = {}
        ov_unknown: dict[int, int] = {}
        ov_span: Interval | None = None
        overlap_rows = 0
        if has_overlap:
            for start, end, otype in con.execute("SELECT startNs, endNs, type FROM OVERLAP_ANALYSIS"):
                if start is None or end is None or end <= start:
                    continue
                overlap_rows += 1
                ov_all_ints.append((start, end))  # int ns（P1-1）
                ov_span = (start, end) if ov_span is None else (min(ov_span[0], start), max(ov_span[1], end))
                known = otype in (
                    _OVERLAP_COMPUTE,
                    _OVERLAP_COMMUNICATION,
                    _OVERLAP_COMM_NOT_OVERLAPPED,
                    _OVERLAP_FREE,
                )
                if otype in (_OVERLAP_COMPUTE, _OVERLAP_COMMUNICATION, _OVERLAP_FREE):
                    ov_intervals.setdefault(otype, []).append((start, end))  # int ns（P1-1）
                if otype in (_OVERLAP_COMPUTE, _OVERLAP_COMMUNICATION, _OVERLAP_COMM_NOT_OVERLAPPED):
                    ov_sum_ns[otype] = ov_sum_ns.get(otype, 0) + (end - start)
                    ov_calls[otype] = ov_calls.get(otype, 0) + 1
                if not known:
                    # 未收录 type：如实记录计数，不猜测语义、不进 busy/idle（真实库仅 0-3）
                    ov_unknown[otype] = ov_unknown.get(otype, 0) + 1
                    ov_unknown_ints.append((start, end))

        string_ids = {i: v for i, v in con.execute("SELECT id, value FROM STRING_IDS")}

        # ---- P0-2：CANN 中间链门控。TASK parent 归因必须先证明
        # TASK.conn ∈ CANN.conn，再经同 conn 的 PYTORCH(type=50001) 解析 framework 名。
        cann_conn_ids = {
            c for (c,) in con.execute("SELECT DISTINCT connectionId FROM CANN_API WHERE connectionId IS NOT NULL")
        }
        conn_to_fw: dict[int, set[str]] = {}
        for conn_id, name_id in con.execute(
            "SELECT connectionId, name FROM PYTORCH_API WHERE connectionId IS NOT NULL AND type = 50001"
        ):
            if conn_id not in cann_conn_ids:
                continue  # 无 CANN 中间链：不建立 framework 关联（保守）
            resolved = string_ids.get(name_id)
            if resolved:
                conn_to_fw.setdefault(conn_id, set()).add(resolved)

        # ---- CANN_API：分类区间/求和/调用数（流式遍历 1.08M 行，int ns 不转 float）----
        cat_ints: dict[str, list[Interval]] = {c: [] for c in _TRACE_CATEGORIES}
        cat_sum_ns: dict[str, float] = {c: 0.0 for c in _TRACE_CATEGORIES}
        cat_calls: dict[str, int] = {c: 0 for c in _TRACE_CATEGORIES}
        runtime_names: dict[str, _RuntimeAgg] = {}
        api_seen = False
        for name_id, start, end in con.execute("SELECT name, startNs, endNs FROM CANN_API"):
            if start is None or end is None:
                continue
            name = string_ids.get(name_id, f"string_id_{name_id}")
            cat = _classify(name)
            dur = end - start
            api_seen = True
            cat_sum_ns[cat] += dur
            cat_calls[cat] += 1
            if dur > 0:
                cat_ints[cat].append((start, end))  # int ns，不做 float 有损转换（P1-1）
            agg = runtime_names.setdefault(name, _RuntimeAgg())
            agg.calls += 1
            agg.total_us += dur / 1000.0  # ns -> us

        # ---- TASK：busy union + 设备行 + parent 归属计数（P0-1/P0-2）----
        task_ints: list[Interval] = []
        device_names: dict[str, _DeviceAgg] = {}
        task_seen = False
        for conn_id, start, end, task_type, info_name_id in con.execute(
            "SELECT t.connectionId, t.startNs, t.endNs, t.taskType, c.name "
            "FROM TASK t LEFT JOIN COMPUTE_TASK_INFO c ON t.globalTaskId = c.globalTaskId"
        ):
            if start is None or end is None:
                continue
            task_seen = True
            # M4.1c 裁定 1：conn=-1 哨兵是 per-stream capture-session marker，
            # 不是 device work —— 永不进 busy union（authoritative path 本就
            # 不用 TASK 做 busy；fallback path 也必须排除）。
            if end > start and conn_id != _TASK_SENTINEL_CONN:
                task_ints.append((start, end))  # int ns（P1-1）
            # M4.1c 裁定 1：conn=-1 永不解析 kernel 名、永不进 named operator 的
            # summed_device_ms / parent 链——强制类型名兜底聚合行（契约化，不只靠
            # "COMPUTE_TASK_INFO 零命中"的经验观察）。
            if conn_id == _TASK_SENTINEL_CONN:
                resolved = None
            else:
                resolved = string_ids.get(info_name_id)
            name = resolved if resolved else f"npu_task_type_{task_type}"
            dev_agg = device_names.setdefault(name, _DeviceAgg())
            dev_agg.calls += 1
            if end > start:
                dev_agg.total_us += (end - start) / 1000.0  # ns -> us
                dev_agg.durs.append((end - start) / 1000.0)
            # P0-2：parent 链必须 TASK.conn ∈ CANN.conn（经 cann_conn_ids 门控），
            # 且同 conn 有 PYTORCH(type=50001) 行；否则计入 unresolved
            fw = conn_to_fw.get(conn_id) if conn_id in cann_conn_ids else None
            if fw:
                dev_agg.fw_names |= fw
                dev_agg.resolved += 1
            else:
                dev_agg.unresolved += 1

        # ---- PYTORCH_API：framework 行（type=50001 aten，SQL 聚合）----
        # P0-3：每行 (start,end) 为该次调用的 host inclusive 时长
        # （operator_details.csv 自带 Host Self/Host Total 双列佐证 Total=inclusive），
        # 同名求和 = inclusive 口径；落 host_inclusive_ms，不借用 summed_host_ms。
        framework_aggs = [
            (name, calls, host_ns)
            for name, calls, host_ns in con.execute(
                "SELECT s.value, COUNT(*), SUM(p.endNs - p.startNs) "
                "FROM PYTORCH_API p JOIN STRING_IDS s ON p.name = s.id "
                "WHERE p.type = 50001 GROUP BY s.value ORDER BY s.value"
            )
        ]

        # ---- timeline（M4.1c 双路径）----
        # authoritative：OVERLAP_ANALYSIS 存在且有行 -> busy = type0∪type1 interval
        #   union（裁定 2：compute/communication 允许重叠，禁止 summed 相加冒充
        #   busy）；exposed/gap = type3 Free（profiler 定义的 device idle，裁定 3）。
        #   compute_ms/communication_ms 与 CSV 路径同口径（summed），与 busy 的
        #   union 口径互斥（AC-06）——summed 相加 > busy 正是 overlap 存在的证据。
        # fallback：OVERLAP_ANALYSIS 缺失 -> busy = TASK union（conn=-1 已排除），
        #   exposed/gap 仍用 session 窗口；timeline_source 显式标记，不 silent。
        # 裁定 5 + gate P0（2026-10-08）：SESSION_TIME_INFO 是 capture session 窗口，
        #   不自动等同 workload wall。authoritative path 的 timeline 覆盖窗口写
        #   timeline.window_ms（v0.7），workload.wall_ms 保持 None——DB 无独立
        #   step/workload boundary，不能证明该窗口是用户 workload wall。措辞纪律：
        #   type0/1 可 overlap、type2 ⊂ type1，不得称四类"互斥 partition"；
        #   OVERLAP_ANALYSIS provides the profiler timeline; in the validated
        #   sample, known busy union (compute∪communication) and Free partition
        #   the observed window.
        # final-review P0（fallback）：gap 窗口必须用绝对 session 起止 ns——误用
        #   相对时长 wall_ns 当右端点会与绝对 ns 区间完全错位。
        timeline = Timeline()
        workload = Workload()
        session_iv: Interval | None = (session[0], session[1]) if session and None not in session else None
        gap_known = False
        if has_overlap and overlap_rows > 0:
            workload.wall_ms = None  # gate P0：span/coverage ≠ workload wall，不冒名
            busy_iv = union(ov_intervals.get(_OVERLAP_COMPUTE, []) + ov_intervals.get(_OVERLAP_COMMUNICATION, []))
            free_iv = union(ov_intervals.get(_OVERLAP_FREE, []))
            timeline.device_busy_ms = intervals_total(busy_iv) / 1e6
            timeline.exposed_non_device_busy_ms = intervals_total(free_iv) / 1e6
            timeline.compute_ms = ov_sum_ns.get(_OVERLAP_COMPUTE, 0) / 1e6
            timeline.communication_ms = ov_sum_ns.get(_OVERLAP_COMMUNICATION, 0) / 1e6
            # window = 全部 OVERLAP 行（含 unknown type）的 coverage union 总长——
            # timeline 比值的 denominator，与 busy/exposed 同 provenance（同源校验）；
            # unknown coverage 另记 backend_metrics，known partition 不声称完整
            timeline.window_ms = intervals_total(union(ov_all_ints)) / 1e6
            gap_iv = free_iv
            gap_known = True
        elif has_overlap:
            # OVERLAP_ANALYSIS 在但零行：timeline 不可用——不得回退 TASK union，
            # 也不得填 0 冒充（AC-03）；gap_unknown -> gap_overlap 保持 None
            if wall_ns is not None:
                workload.wall_ms = wall_ns / 1e6  # session 是唯一窗口证据
            gap_iv = []
        else:
            if wall_ns is not None:
                workload.wall_ms = wall_ns / 1e6  # session 是唯一窗口证据
            busy_iv = union(task_ints) if task_seen else []
            busy_ns = intervals_total(busy_iv)
            if task_seen:
                timeline.device_busy_ms = busy_ns / 1e6
                if wall_ns is not None:
                    timeline.exposed_non_device_busy_ms = (wall_ns - busy_ns) / 1e6
            # gap 区间：session 绝对窗口 − busy union（同 ns int 域，complement 一次扫描）
            gap_iv = complement(session_iv, task_ints) if (task_seen and session_iv) else []
            gap_known = bool(task_seen and session_iv)

        # ---- runtime 分类 + gap overlap（M4.1 机制复用：同 ns int 域逐事件求交，ns→ms）----
        gap_overlap: dict[str, float] | None = None
        if gap_known and api_seen:
            gap_overlap = {
                "launch": total_event_overlap(cat_ints["launch"], gap_iv) / 1e6,
                "synchronization": total_event_overlap(cat_ints["synchronization"], gap_iv) / 1e6,
                "allocation": total_event_overlap(cat_ints["allocation"], gap_iv) / 1e6,
                "other": total_event_overlap(cat_ints["other"], gap_iv) / 1e6,
            }
        runtime_stats = RuntimeStats(
            api_summed_ms=(sum(cat_sum_ns.values()) / 1e6) if api_seen else None,
            launch_summed_ms=(cat_sum_ns["launch"] / 1e6) if api_seen else None,
            synchronization_summed_ms=(cat_sum_ns["synchronization"] / 1e6) if api_seen else None,
            allocation_summed_ms=(cat_sum_ns["allocation"] / 1e6) if api_seen else None,
            other_summed_ms=(cat_sum_ns["other"] / 1e6) if api_seen else None,
            launch_count=cat_calls["launch"] if api_seen else None,
            gap_overlap_ms=gap_overlap,
        )

        # ---- communication exposure（M4.1c 裁定 4，仅 authoritative path）----
        # 官方语义：Communication(Not Overlapped) = Communication − Overlapped，
        # 即 total(type1) = not_overlapped(type2) + overlapped(与 compute 重叠)。
        # 三者是 exposure 分解，不是 wall partition——禁止与 busy/free 求和对照。
        # section 对象恒存在（schema 契约），unavailable 用字段 None 表达（AC-03）。
        comm = CommunicationStats()
        if has_overlap and overlap_rows > 0:
            total_comm_ms = ov_sum_ns.get(_OVERLAP_COMMUNICATION, 0) / 1e6
            not_overlapped_ms = ov_sum_ns.get(_OVERLAP_COMM_NOT_OVERLAPPED, 0) / 1e6
            overlap_ms = total_comm_ms - not_overlapped_ms
            comm = CommunicationStats(
                calls=ov_calls.get(_OVERLAP_COMMUNICATION),
                total_ms=total_comm_ms,
                overlap_ms=overlap_ms,
                overlap_ratio=(overlap_ms / total_comm_ms) if total_comm_ms > 0 else None,
            )

        # ---- operators 组装（framework -> runtime -> device；parent 走 connectionId 链）----
        operators: list[OperatorEvidence] = []
        framework_id_by_name: dict[str, str] = {}
        for name, calls, host_ns in framework_aggs:
            op_id = f"op_{len(operators) + 1:06d}"
            framework_id_by_name[name] = op_id
            operators.append(
                OperatorEvidence(
                    id=op_id,
                    name=name,
                    layer="framework",
                    parent_id=None,
                    op_type="aten_op",
                    calls=calls,
                    # P0-3：PYTORCH_API 行时长为 inclusive host 语义，落专用字段
                    host_inclusive_ms=host_ns / 1e6 if host_ns else None,
                    evidence_ids=[store.ref(rec_pytorch.id)],
                )
            )
        for name in sorted(runtime_names):
            agg = runtime_names[name]
            operators.append(
                OperatorEvidence(
                    id=f"op_{len(operators) + 1:06d}",
                    name=name,
                    layer="runtime",
                    parent_id=None,  # per-instance 链路留 backend evidence；聚合行不猜
                    op_type="cann_api",
                    calls=agg.calls,
                    summed_host_ms=agg.total_us / 1000.0 if agg.total_us else None,
                    evidence_ids=[store.ref(rec_cann.id)],
                )
            )
        for name in sorted(device_names):
            dev_agg = device_names[name]
            # P0-1：全部实例 resolve 且指向唯一 framework op 才填 parent；
            # 任一 unresolved 实例 -> None（"部分未知归属"不得归给唯一已知 parent）
            parent_id = None
            if dev_agg.unresolved == 0 and len(dev_agg.fw_names) == 1:
                parent_id = framework_id_by_name[next(iter(dev_agg.fw_names))]
            operators.append(
                OperatorEvidence(
                    id=f"op_{len(operators) + 1:06d}",
                    name=name,
                    layer="device",
                    parent_id=parent_id,
                    op_type="npu_task",
                    calls=dev_agg.calls,
                    summed_device_ms=dev_agg.total_us / 1000.0 if dev_agg.total_us else None,
                    avg_device_us=(dev_agg.total_us / dev_agg.calls) if dev_agg.calls else None,
                    median_device_us=statistics.median(dev_agg.durs) if dev_agg.durs else None,
                    evidence_ids=[store.ref(rec_task.id)],
                )
            )

        # ---- backend_metrics：窗口语义与 timeline source 显式化（M4.1c 裁定 5）----
        backend_meta: dict = {
            "db_tables_used": sorted(_TRACE_DB_REQUIRED),
            "rank_device_map": [f"rankId={r} deviceId={d}" for r, d in rank_rows],
            # M4.1c：哨兵行不进 busy union、不进 named operator（契约化，非经验观察）
            "unlinked_task_sentinel": "connectionId=-1 rows stay unattributed, never in busy union",
            "timeline_source": "overlap_analysis" if has_overlap else "task_union_fallback",
        }
        if has_overlap:
            backend_meta["overlap_analysis_rows"] = overlap_rows
            if wall_ns is not None:
                # 裁定 5：session 窗口单独记录，workload.wall_ms 在 authoritative
                # path 用 overlap 分类窗口（两者不是同一窗口，禁止混用）
                backend_meta["session_window_ms"] = wall_ns / 1e6
            if ov_span is not None:
                backend_meta["overlap_window_ms"] = (ov_span[1] - ov_span[0]) / 1e6
                if wall_ns is not None:
                    # 裁定 5：差值是 capture 初始化/未覆盖区间，不是 device idle，
                    # 不计入 exposed gap（真实样本 13.5s）
                    backend_meta["outside_overlap_window_ms"] = (wall_ns - (ov_span[1] - ov_span[0])) / 1e6
            if ov_unknown:
                backend_meta["overlap_unknown_types"] = {str(k): v for k, v in sorted(ov_unknown.items())}
                # gate P0：unknown coverage 显式记录——known partition（busy∪free）
                # 不声称覆盖完整 window
                backend_meta["overlap_unknown_window_ms"] = intervals_total(union(ov_unknown_ints)) / 1e6
            if overlap_rows > 0:
                backend_meta["communication_not_overlapped_ms"] = ov_sum_ns.get(_OVERLAP_COMM_NOT_OVERLAPPED, 0) / 1e6

        ev = OptimizationEvidence(
            run=RunInfo(
                backend="ascend",
                profiler="torch_npu",
                rank=rank,  # RANK_DEVICE_MAP.rankId（真实 rank，非 deviceId）
                device=device_name[0] if device_name else None,  # NPU_INFO 型号
                device_id=device_id,
                source_files=[db_path.name],
                run_id=run_id,
            ),
            workload=workload,
            timeline=timeline,
            operators=operators,
            runtime=runtime_stats,
            communication=comm,
            provenance=store.records,
            # P1-4：metric 按输入显式绑定（同 DB ≠ 自动同 timeline；
            # 全部 metric 同一 ns clock domain 的证据见 SESSION_TIME_INFO 包络勘察）
            metric_evidence=_trace_metric_evidence(
                store,
                rec_session,
                rec_pytorch,
                rec_cann,
                rec_task,
                rec_overlap if has_overlap else None,
                wall_ns is not None,
                task_seen,
                api_seen,
                gap_known,
            ),
            backend_metrics=BackendMetrics(ascend=backend_meta),
        )
        return ev, store
    finally:
        con.close()


def _trace_metric_evidence(
    store: ProvenanceStore,
    rec_session: EvidenceRecord,
    rec_pytorch: EvidenceRecord,
    rec_cann: EvidenceRecord,
    rec_task: EvidenceRecord,
    rec_overlap: EvidenceRecord | None,
    has_wall: bool,
    has_task: bool,
    has_api: bool,
    gap_known: bool,
) -> dict[str, list[str]]:
    """P1-4：metric 按输入显式绑定到对应 provenance 记录（同 DB ≠ 自动同 timeline）。
    M4.1c：authoritative path 的 timeline/communication 指标绑定 rec_overlap；
    fallback path 保持 rec_task 绑定；busy source 不同的两条路径不得共用绑定。"""
    s, _pt, cn, tk = (
        store.ref(rec_session.id),
        store.ref(rec_pytorch.id),
        store.ref(rec_cann.id),
        store.ref(rec_task.id),
    )
    ov = store.ref(rec_overlap.id) if rec_overlap is not None else None
    me: dict[str, list[str]] = {}
    if ov is not None:
        # gate P0：denominator 是 timeline.window_ms（与 busy/exposed/comm 同一条
        # OVERLAP 记录）；workload.wall_ms 在 authoritative path 为 None，不绑定
        me["timeline.window_ms"] = [ov]
        me["timeline.device_busy_ms"] = [ov]
        me["timeline.exposed_non_device_busy_ms"] = [ov]
        me["timeline.compute_ms"] = [ov]
        me["timeline.communication_ms"] = [ov]
        me["communication.total_ms"] = [ov]
        me["communication.overlap_ms"] = [ov]
    else:
        if has_wall:
            me["workload.wall_ms"] = [s]
        if has_task:
            me["timeline.device_busy_ms"] = [tk]
            me["timeline.exposed_non_device_busy_ms"] = [s, tk]  # derived: wall(session) - busy(task)
    if has_api:
        for cat in (
            "api_summed_ms",
            "launch_summed_ms",
            "synchronization_summed_ms",
            "allocation_summed_ms",
            "other_summed_ms",
            "launch_count",
        ):
            me[f"runtime.{cat}"] = [cn]
        # gap overlap：derived，依赖 CANN 区间 + busy/Free 区间来源的全部输入
        if gap_known:
            gap_refs = [cn, ov] if ov is not None else [cn, tk, s]
            for cat in ("launch", "synchronization", "allocation", "other"):
                me[f"runtime.gap_overlap_ms.{cat}"] = gap_refs
    return me
