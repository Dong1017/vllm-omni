# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Ascend profiler DB adapter 测试（M2-T2）：fixture 复刻真实 analysis.db schema；
# 真实 DB 存在时执行真实验证（CI 自动跳过）。

import sqlite3
from pathlib import Path

import pytest

from vllm_omni.profiling.backends import UnsupportedProfilerDBError, analyze_ascend_db
from vllm_omni.profiling.schema import OptimizationEvidence

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REAL_DB = Path(
    "D:/world_model/outputs/h3_16die_validation_20260922/ref2va_15s_280ts/"
    "ref2va_15s_280ts/ASCEND_PROFILER_OUTPUT/analysis.db"
)


def _create_fixture_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE StepTraceTime (deviceId INTEGER, step INTEGER, computing REAL, "
        "communication_not_overlapped REAL, overlapped REAL, communication REAL, free REAL, "
        "stage REAL, bubble REAL, "
        "communication_not_overlapped_and_exclude_receive REAL, preparing REAL)"
    )
    con.execute(
        "CREATE TABLE CommAnalyzerTime (hccl_op_name TEXT, group_name TEXT, start_timestamp REAL, "
        "elapse_time REAL, transit_time REAL, wait_time REAL, synchronization_time REAL, "
        "idle_time REAL, step INTEGER, type TEXT)"
    )
    con.execute(
        "CREATE TABLE CommAnalyzerBandwidth (hccl_op_name TEXT, group_name TEXT, transport_type TEXT, "
        "transit_size REAL, transit_time REAL, bandwidth REAL, large_packet_ratio REAL, "
        "package_size REAL, count INTEGER, total_duration REAL, step INTEGER, type TEXT)"
    )
    con.execute("INSERT INTO StepTraceTime VALUES (0, 1, 8570.0, 387.0, 63.0, 450.0, 2147.0, 11104.0, 0, 387.0, 0.6)")
    con.executemany(
        "INSERT INTO CommAnalyzerTime VALUES (?, 'g0', 0, ?, 0, 0, 0, 0, 1, 'allgather')",
        [("HcclAllGather", 100.0), ("HcclAllGather", 150.0), ("HcclAllToAll", 200.0)],
    )
    con.commit()
    con.close()


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "analysis.db"
    _create_fixture_db(p)
    return p


def test_step_trace_and_comm(db_path):
    ev, store = analyze_ascend_db(db_path)
    assert ev.run.backend == "ascend"
    assert ev.run.profiler == "torch_npu"
    # P0-1：analysis.db 只有 deviceId（设备号），rank 必须为 None
    assert ev.run.rank is None
    assert ev.run.device_id == "0"
    assert ev.run.run_id is not None
    assert all(r.run_id == ev.run.run_id for r in store.records)
    assert ev.workload.wall_ms == pytest.approx(11.104)
    assert ev.timeline.compute_ms == pytest.approx(8.570)
    assert ev.timeline.communication_ms == pytest.approx(0.450)
    assert ev.timeline.exposed_non_device_busy_ms == pytest.approx(2.147)
    assert ev.communication.calls == 3
    assert ev.communication.total_ms == pytest.approx(0.450)
    assert ev.communication.collectives == {"HcclAllGather": 2, "HcclAllToAll": 1}
    detail = ev.backend_metrics.ascend["comm_detail_us"]
    assert detail["wait_us"] == 0.0


def test_round_trip(db_path):
    ev, _ = analyze_ascend_db(db_path)
    assert OptimizationEvidence.from_dict(ev.to_dict()) == ev


def test_unsupported_schema_missing_table(tmp_path):
    p = tmp_path / "analysis.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE something_else (a INTEGER)")
    con.commit()
    con.close()
    with pytest.raises(UnsupportedProfilerDBError, match="missing tables"):
        analyze_ascend_db(p)


def test_unsupported_schema_missing_column(tmp_path):
    p = tmp_path / "analysis.db"
    con = sqlite3.connect(p)
    con.execute(
        "CREATE TABLE CommAnalyzerTime (hccl_op_name TEXT, group_name TEXT, start_timestamp REAL, "
        "elapse_time REAL, transit_time REAL, wait_time REAL, synchronization_time REAL, "
        "idle_time REAL, step INTEGER, type TEXT)"
    )
    # StepTraceTime 缺 communication/free/stage 列 -> 必须显式报错而非静默解析
    con.execute("CREATE TABLE StepTraceTime (deviceId INTEGER, step INTEGER, computing REAL)")
    con.commit()
    con.close()
    with pytest.raises(UnsupportedProfilerDBError, match="missing columns"):
        analyze_ascend_db(p)


def test_missing_db_fails(tmp_path):
    with pytest.raises(UnsupportedProfilerDBError, match="not found"):
        analyze_ascend_db(tmp_path / "nope.db")


@pytest.mark.skipif(not REAL_DB.exists(), reason="real analysis.db not available (stays prepared_not_run in CI)")
def test_real_analysis_db_validation():
    ev, _ = analyze_ascend_db(REAL_DB)
    # 真实 DB 自身数值（2026-09-29 直查记录，见 docs/ascend-artifact-mapping.md 数据注记）：
    # DB 的 stage/free 与 CSV 的 Stage/Free 存在约 2% 口径差异（导出路径不同），
    # framework 如实分别记录，不做跨源合并或折算。
    assert ev.run.device_id == "8"  # 真实 DB：deviceId=8（RANK_DEVICE_MAP 核实 rankId=0）
    assert ev.workload.wall_ms == pytest.approx(108815.501753)
    assert ev.timeline.compute_ms == pytest.approx(85704.020577)
    assert ev.timeline.communication_ms == pytest.approx(4504.063418)  # 与 CSV 完全一致
    assert ev.timeline.exposed_non_device_busy_ms == pytest.approx(19237.962204)
    assert ev.communication.calls == 501
