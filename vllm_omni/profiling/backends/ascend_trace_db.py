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
from vllm_omni.profiling.provenance import ProvenanceStore, compute_run_id, sha256_file
from vllm_omni.profiling.schema import (
    BackendMetrics,
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
            if end > start:
                task_ints.append((start, end))  # int ns（P1-1）
            resolved = string_ids.get(info_name_id)
            name = resolved if resolved else f"npu_task_type_{task_type}"
            agg = device_names.setdefault(name, _DeviceAgg())
            agg.calls += 1
            if end > start:
                agg.total_us += (end - start) / 1000.0  # ns -> us
                agg.durs.append((end - start) / 1000.0)
            # P0-2：parent 链必须 TASK.conn ∈ CANN.conn（经 cann_conn_ids 门控），
            # 且同 conn 有 PYTORCH(type=50001) 行；否则计入 unresolved
            fw = conn_to_fw.get(conn_id) if conn_id in cann_conn_ids else None
            if fw:
                agg.fw_names |= fw
                agg.resolved += 1
            else:
                agg.unresolved += 1

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

        # ---- timeline：wall=SESSION_TIME_INFO；busy=TASK union；gap=session 窗口−busy ----
        # final-review P0：gap 窗口必须用绝对 session 起止 ns——误用相对时长 wall_ns
        # 当右端点会与绝对 ns 区间完全错位（overlap 恒 0，checkpoint 审查实测发现）。
        timeline = Timeline()
        workload = Workload()
        session_iv: Interval = (session[0], session[1]) if session and None not in session else None
        if wall_ns is not None:
            workload.wall_ms = wall_ns / 1e6
        busy_iv = union(task_ints) if task_seen else []
        busy_ns = intervals_total(busy_iv)
        if task_seen:
            timeline.device_busy_ms = busy_ns / 1e6
            if wall_ns is not None:
                timeline.exposed_non_device_busy_ms = (wall_ns - busy_ns) / 1e6
        # gap 区间：session 绝对窗口 − busy union（同 ns int 域，complement 一次扫描）
        gap_iv = complement(session_iv, task_ints) if (task_seen and session_iv) else []

        # ---- runtime 分类 + gap overlap（M4.1 机制复用：同 ns int 域逐事件求交，ns→ms）----
        gap_overlap: dict[str, float] | None = None
        if task_seen and wall_ns and api_seen:
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
            agg = device_names[name]
            # P0-1：全部实例 resolve 且指向唯一 framework op 才填 parent；
            # 任一 unresolved 实例 -> None（"部分未知归属"不得归给唯一已知 parent）
            parent_id = None
            if agg.unresolved == 0 and len(agg.fw_names) == 1:
                parent_id = framework_id_by_name[next(iter(agg.fw_names))]
            operators.append(
                OperatorEvidence(
                    id=f"op_{len(operators) + 1:06d}",
                    name=name,
                    layer="device",
                    parent_id=parent_id,
                    op_type="npu_task",
                    calls=agg.calls,
                    summed_device_ms=agg.total_us / 1000.0 if agg.total_us else None,
                    avg_device_us=(agg.total_us / agg.calls) if agg.calls else None,
                    median_device_us=statistics.median(agg.durs) if agg.durs else None,
                    evidence_ids=[store.ref(rec_task.id)],
                )
            )

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
            provenance=store.records,
            # P1-4：metric 按输入显式绑定（同 DB ≠ 自动同 timeline；
            # 全部 metric 同一 ns clock domain 的证据见 SESSION_TIME_INFO 包络勘察）
            metric_evidence=_trace_metric_evidence(
                store,
                rec_session,
                rec_pytorch,
                rec_cann,
                rec_task,
                wall_ns is not None,
                task_seen,
                api_seen,
            ),
            backend_metrics=BackendMetrics(
                ascend={
                    "db_tables_used": sorted(_TRACE_DB_REQUIRED),
                    "rank_device_map": [f"rankId={r} deviceId={d}" for r, d in rank_rows],
                    "unlinked_task_sentinel": "connectionId=-1 rows stay unattributed",
                }
            ),
        )
        return ev, store
    finally:
        con.close()


def _trace_metric_evidence(
    store: ProvenanceStore,
    rec_session: object,
    rec_pytorch: object,
    rec_cann: object,
    rec_task: object,
    has_wall: bool,
    has_task: bool,
    has_api: bool,
) -> dict[str, list[str]]:
    """P1-4：metric 按输入显式绑定到对应 provenance 记录（同 DB ≠ 自动同 timeline）。"""
    s, _pt, cn, tk = (
        store.ref(rec_session.id),
        store.ref(rec_pytorch.id),
        store.ref(rec_cann.id),
        store.ref(rec_task.id),
    )
    me: dict[str, list[str]] = {}
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
        # gap overlap：derived，依赖 CANN 区间 + TASK busy 区间 + session 窗口全部输入
        if has_task:
            for cat in ("launch", "synchronization", "allocation", "other"):
                me[f"runtime.gap_overlap_ms.{cat}"] = [cn, tk, s]
    return me
