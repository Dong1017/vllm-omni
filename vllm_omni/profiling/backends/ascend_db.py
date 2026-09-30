# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Ascend profiler DB adapter（M2-T2）：只读打开 analysis.db。
# 支持的 schema 按真实样本（2026-08 CANN）检测：StepTraceTime + CommAnalyzer{Time,Bandwidth,Matrix}。
# schema 不符 -> 显式 unsupported-version error，禁止静默解析（T26/AC-31）。
# dispatch 链路（PYTORCH_API -> CANN_API -> TASK）在 ascend_pytorch_profiler_0.db 主库，
# connectionId 关联已勘察（docs/ascend-artifact-mapping.md），实现为后续增量。

from __future__ import annotations

import sqlite3
from pathlib import Path

from vllm_omni.profiling.provenance import EvidenceRecord, ProvenanceStore, compute_run_id, sha256_file
from vllm_omni.profiling.schema import (
    BackendMetrics,
    CommunicationStats,
    OptimizationEvidence,
    RunInfo,
    Timeline,
    Workload,
)

_PARSER = "ascend.profiler_db"

# 真实样本 schema（docs/ascend-artifact-mapping.md）；新版本缺表/缺列必须显式报错
_REQUIRED_TABLES: dict[str, set[str]] = {
    "StepTraceTime": {"deviceId", "step", "computing", "communication", "free", "stage"},
    "CommAnalyzerTime": {"hccl_op_name", "group_name", "elapse_time", "step", "type"},
}


class UnsupportedProfilerDBError(ValueError):
    pass


def _open_db(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise UnsupportedProfilerDBError(f"profiler DB not found: {path}")
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    existing = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing_tables = set(_REQUIRED_TABLES) - existing
    if missing_tables:
        con.close()
        raise UnsupportedProfilerDBError(
            f"unsupported profiler DB schema: missing tables {sorted(missing_tables)} in {path.name}; "
            "CANN/profiler version may differ — refusing to guess column semantics"
        )
    for table, required_cols in _REQUIRED_TABLES.items():
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        missing_cols = required_cols - cols
        if missing_cols:
            con.close()
            raise UnsupportedProfilerDBError(
                f"unsupported profiler DB schema: table {table} missing columns {sorted(missing_cols)}"
            )
    return con


def _sha(path: Path) -> str:
    return sha256_file(path)


def analyze_ascend_db(db_path: Path) -> tuple[OptimizationEvidence, ProvenanceStore]:
    """分析 analysis.db：step trace + communication 视图。"""
    db_path = Path(db_path)
    con = _open_db(db_path)
    try:
        sha = _sha(db_path)
        run_id = compute_run_id([sha], backend="ascend", parser=_PARSER)
        store = ProvenanceStore(run_id=run_id)
        store.add(
            source_file=db_path.name,
            source_type="profiler_db",
            parser=_PARSER,
            unit="us",
            query="sqlite:analysis.db (read-only)",
            aggregation=None,
            source_sha256=sha,
        )
        rec = store.records[0]

        timeline = Timeline()
        workload = Workload()
        device_id: str | None = None
        for db_device_id, _step, computing, communication, free, stage in con.execute(
            "SELECT deviceId, step, computing, communication, free, stage FROM StepTraceTime"
        ):
            if device_id is None and db_device_id is not None:
                # P0-1：analysis.db 只有 deviceId（设备号）；rank 需大库 RANK_DEVICE_MAP
                device_id = str(db_device_id)
            if stage is not None:
                workload.wall_ms = (workload.wall_ms or 0.0) + stage / 1000.0
            if computing is not None:
                timeline.compute_ms = (timeline.compute_ms or 0.0) + computing / 1000.0
            if communication is not None:
                timeline.communication_ms = (timeline.communication_ms or 0.0) + communication / 1000.0
            if free is not None:
                timeline.exposed_non_device_busy_ms = (timeline.exposed_non_device_busy_ms or 0.0) + free / 1000.0

        comm = CommunicationStats()
        collectives: dict[str, int] = {}
        total_elapse_us = 0.0
        comm_detail: dict[str, float] = {"wait_us": 0.0, "synchronization_us": 0.0, "idle_us": 0.0, "transit_us": 0.0}
        for hccl_name, elapse, wait, sync_t, idle, transit in con.execute(
            "SELECT hccl_op_name, elapse_time, wait_time, synchronization_time, idle_time, transit_time "
            "FROM CommAnalyzerTime"
        ):
            collectives[hccl_name] = collectives.get(hccl_name, 0) + 1
            if elapse is not None:
                total_elapse_us += elapse
            if wait is not None:
                comm_detail["wait_us"] += wait
            if sync_t is not None:
                comm_detail["synchronization_us"] += sync_t
            if idle is not None:
                comm_detail["idle_us"] += idle
            if transit is not None:
                comm_detail["transit_us"] += transit
        if total_elapse_us or collectives:
            comm.calls = sum(collectives.values())
            comm.total_ms = total_elapse_us / 1000.0
        comm.collectives = collectives or None

        ev = OptimizationEvidence(
            run=RunInfo(
                backend="ascend",
                profiler="torch_npu",
                rank=None,  # P0-1：deviceId 不是 rank；rank 来自大库 RANK_DEVICE_MAP（后续增量）
                device_id=device_id,
                source_files=[db_path.name],
                run_id=run_id,
            ),
            workload=workload,
            timeline=timeline,
            communication=comm,
            backend_metrics=BackendMetrics(
                ascend={
                    "comm_detail_us": {k: comm_detail[k] for k in sorted(comm_detail)},
                    "db_tables_used": ["StepTraceTime", "CommAnalyzerTime"],
                }
            ),
            provenance=store.records,
            metric_evidence=_metric_evidence(store, rec, workload, timeline, comm),
        )
        return ev, store
    finally:
        con.close()


def _metric_evidence(
    store: ProvenanceStore,
    rec: EvidenceRecord,
    workload: Workload,
    timeline: Timeline,
    comm: CommunicationStats,
) -> dict[str, list[str]]:
    """P1-1：DB 单一 source，全部核心指标绑定同一条记录。"""
    ref = store.ref(rec.id)
    me: dict[str, list[str]] = {}
    if workload.wall_ms is not None:
        me["workload.wall_ms"] = [ref]
    for key in ("device_busy_ms", "exposed_non_device_busy_ms", "compute_ms", "communication_ms", "unknown_ms"):
        if getattr(timeline, key) is not None:
            me[f"timeline.{key}"] = [ref]
    if comm.total_ms is not None:
        me["communication.total_ms"] = [ref]
    return me
