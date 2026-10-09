# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Distributed timeline analysis 测试（M4.3 T7）：reducer 语义不变量 + clock/window
# gate fail-fast + Ascend rank-timeline adapter + CLI e2e + artifact round trip。

import json
import sqlite3
from pathlib import Path

import pytest

from vllm_omni.profiling.cli import main as cli_main
from vllm_omni.profiling.distributed import (
    DistributedClockError,
    DistributedWindowError,
    RankTimelineIntervals,
    build_distributed_summary,
    build_rank_timeline_from_ascend_db,
    compare_distributed,
    reduce_window,
    resolve_clock_validation,
    resolve_windows,
    write_distributed_outputs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def mk_timeline(rank: int, raw: list[tuple[int, int]], **layers) -> RankTimelineIntervals:
    """构造已 merge 的 rank timeline；未指定的层默认由 raw 派生的最小一致形态。"""
    defaults = dict(sentinel=[], attributable=list(raw), compute=[], communication=[], unknown=[])
    defaults.update(layers)
    return RankTimelineIntervals(
        rank_id=rank,
        source=f"rank{rank}.db",
        source_sha256=f"sha-{rank}",
        unit="ns",
        coverage_intervals=[(0, 100)],
        raw_task_intervals=list(raw),
        sentinel_task_intervals=list(defaults["sentinel"]),
        attributable_task_intervals=list(defaults["attributable"]),
        compute_intervals=list(defaults["compute"]),
        communication_intervals=list(defaults["communication"]),
        unknown_intervals=list(defaults["unknown"]),
        task_row_count=len(raw),
        sentinel_row_count=len(defaults["sentinel"]),
        communication_source="COMMUNICATION_TASK_INFO" if defaults["communication"] else "unavailable",
    )


def run_common(timelines, window=None):
    name, bounds = window or ("common_coverage_window", (0, 100))
    return reduce_window(timelines, name, bounds)


# ---- 1. 基础两 rank：idle 交集 ----


def test_two_rank_common_idle_intersection():
    # 用户样例（ns）：rank0 idle [10,20],[30,40]；rank1 idle [15,25],[35,45]
    r0 = mk_timeline(0, raw=[(0, 10), (20, 30), (40, 100)])
    r1 = mk_timeline(1, raw=[(0, 15), (25, 35), (45, 100)])
    result = run_common([r0, r1])
    common = result["distributed"]["common_no_device_task"]
    assert common["intervals_ns"] == [[15, 20], [35, 40]]
    assert common["duration_ms"] == pytest.approx(10 / 1e6)  # 10 ns -> ms
    assert common["semantics"] == "raw_TASK_rows_including_sentinel"
    assert common["compatibility"] == "issue_8228_legacy_metric"


# ---- 2. no-task vs no-compute：通信期 no_compute=true / no_device_task=false ----


def test_common_no_device_task_subset_of_common_no_compute():
    # 两 rank 同构：compute [0,10]，comm TASK [10,30]（device task 仍在跑），raw 覆盖两者
    layers = dict(
        raw=[(0, 30), (40, 100)],
        attributable=[(0, 30), (40, 100)],
        compute=[(0, 10)],
        communication=[(10, 30)],
    )
    r0 = mk_timeline(0, **layers)
    r1 = mk_timeline(1, **layers)
    result = run_common([r0, r1])
    no_task = result["distributed"]["common_no_device_task"]["intervals_ns"]
    no_compute = result["distributed"]["common_no_compute"]["intervals_ns"]
    assert no_task == [[30, 40]]
    assert no_compute == [[10, 100]]
    # 子集不变量：no_task ⊆ no_compute
    for a, b in no_task:
        assert any(x <= a and b <= y for x, y in no_compute)
    # 差值 [10,30] 是含非 compute device activity 的 common no-compute 时间


# ---- 3. 三个以上 rank（reducer 不是 pairwise）----


def test_three_rank_fold_not_pairwise():
    r0 = mk_timeline(0, raw=[(50, 100)])  # idle [0,50]
    r1 = mk_timeline(1, raw=[(0, 25), (75, 100)])  # idle [25,75]
    r2 = mk_timeline(2, raw=[(0, 40), (60, 100)])  # idle [40,100]
    result = run_common([r0, r1, r2])
    common = result["distributed"]["common_no_device_task"]["intervals_ns"]
    assert common == [[40, 50]]  # pairwise 只看前两个会给出 [25,50]


# ---- 4. rank 顺序不影响结果 ----


def test_rank_order_permutation_invariant():
    r0 = mk_timeline(0, raw=[(0, 10), (20, 30), (40, 100)])
    r1 = mk_timeline(1, raw=[(0, 15), (25, 35), (45, 100)])
    r2 = mk_timeline(2, raw=[(0, 12), (22, 32), (42, 100)])
    summary_a = build_distributed_summary([r0, r1, r2], "explicit_assertion", None, None)
    summary_b = build_distributed_summary([r2, r0, r1], "explicit_assertion", None, None)
    assert summary_a["windows"] == summary_b["windows"]
    assert summary_a["rank_ids"] == summary_b["rank_ids"] == [0, 1, 2]


# ---- 5. touching / zero interval：沿用现有 interval semantics ----


def test_touching_and_zero_duration_intervals():
    # touching actives merge 成一段 -> r0 idle = [0,10]+[30,100]；零长行 (20,20) 无影响；
    # r1 idle = [5,100]（raw 仅 [0,5]）-> 交集 [5,10]+[30,100]
    r0 = mk_timeline(0, raw=[(10, 20), (20, 30), (20, 20)])
    r1 = mk_timeline(1, raw=[(0, 5)])
    result = run_common([r0, r1])
    assert result["distributed"]["common_no_device_task"]["intervals_ns"] == [[5, 10], [30, 100]]


# ---- 6. clock gate：无证据无断言 fail fast；显式断言/metadata 两档 ----


def test_clock_gate_fail_fast_and_levels():
    r0 = mk_timeline(0, raw=[(0, 50)])
    r1 = mk_timeline(1, raw=[(0, 50)])
    with pytest.raises(DistributedClockError, match="neither evidenced"):
        resolve_clock_validation([r0, r1], shared_clock=False, clock_metadata=None)
    status, meta = resolve_clock_validation([r0, r1], shared_clock=True, clock_metadata=None)
    assert status == "explicit_assertion" and meta is None
    status, meta = resolve_clock_validation(
        [r0, r1], shared_clock=False, clock_metadata={"evidence": "host NTP sync log"}
    )
    assert status == "validated" and meta["evidence"] == "host NTP sync log"
    with pytest.raises(DistributedClockError, match="non-empty 'evidence'"):
        resolve_clock_validation([r0, r1], shared_clock=False, clock_metadata={"evidence": ""})


def test_clock_gate_duplicate_and_empty_timeline_fail_fast():
    r0 = mk_timeline(0, raw=[(0, 50)])
    r0_dup = mk_timeline(0, raw=[(0, 40)])
    with pytest.raises(DistributedClockError, match="duplicate rank ids"):
        resolve_clock_validation([r0, r0_dup], shared_clock=True, clock_metadata=None)
    empty = RankTimelineIntervals(rank_id=3, source="x.db", source_sha256="s", unit="ns")
    with pytest.raises(DistributedClockError, match="empty raw_task"):
        resolve_clock_validation([r0, empty], shared_clock=True, clock_metadata=None)


# ---- 7. coverage/window gate：越界 fail fast，不 silent clip ----


def test_window_outside_coverage_fail_fast(tmp_path: Path):
    r0 = mk_timeline(0, raw=[(0, 40)])
    r1 = mk_timeline(1, raw=[(0, 40)])
    r1.coverage_intervals = [(0, 50)]
    windows = tmp_path / "windows.json"
    windows.write_text(json.dumps({"late": [60, 80]}), encoding="utf-8")
    with pytest.raises(DistributedWindowError, match="not inside rank1 coverage"):
        resolve_windows([r0, r1], windows)
    inverted = tmp_path / "inverted.json"
    inverted.write_text(json.dumps({"bad": [80, 60]}), encoding="utf-8")
    with pytest.raises(DistributedWindowError, match="start >= end"):
        resolve_windows([r0, r1], inverted)


def test_default_common_coverage_window():
    r0 = mk_timeline(0, raw=[(0, 40)])
    r1 = mk_timeline(1, raw=[(0, 40)])
    r1.coverage_intervals = [(10, 80)]
    windows = resolve_windows([r0, r1], None)
    assert windows[0][0] == "common_coverage_window"
    assert windows[0][1] == (10, 80)
    r2 = mk_timeline(2, raw=[(0, 40)])
    r2.coverage_intervals = [(200, 300)]
    with pytest.raises(DistributedClockError, match="empty intersection"):
        resolve_windows([r0, r1, r2], None)


# ---- 8. phase slicing：named windows 各自独立结果 ----


def test_named_phase_slicing():
    r0 = mk_timeline(0, raw=[(0, 10), (20, 30), (40, 100)])
    r1 = mk_timeline(1, raw=[(0, 15), (25, 35), (45, 100)])
    summary = build_distributed_summary([r0, r1], "explicit_assertion", None, None)
    assert [w["window"]["name"] for w in summary["windows"]] == ["common_coverage_window"]
    # 手动切片：preparation [0,30] / decode [30,100]
    prep = run_common([r0, r1], window=("preparation", (0, 30)))
    decode = run_common([r0, r1], window=("decode", (30, 100)))
    assert prep["distributed"]["common_no_device_task"]["intervals_ns"] == [[15, 20]]
    assert decode["distributed"]["common_no_device_task"]["intervals_ns"] == [[35, 40]]
    assert prep["distributed"]["common_no_device_task"]["duration_ms"] == pytest.approx(5 / 1e6)  # 5 ns
    assert decode["distributed"]["common_no_device_task"]["duration_ms"] == pytest.approx(5 / 1e6)  # 5 ns


# ---- 9. unknown activity：不自动当 idle 或 compute ----


def test_unknown_activity_layer_neither_idle_nor_compute():
    # attributable [30,40] 既非 compute 也非 communication -> unknown 层；
    # raw 语义下该段仍算 device task（不是 idle），也绝非 compute
    r0 = mk_timeline(
        0,
        raw=[(0, 30), (30, 40), (50, 100)],
        attributable=[(0, 40), (50, 100)],
        compute=[(0, 30)],
        unknown=[(30, 40)],
    )
    r1 = mk_timeline(1, raw=[(0, 100)])
    result = run_common([r0, r1])
    # unknown TASK 段 [30,40]：raw 活动 -> 不进 common no-task；
    # 但它不是 compute -> 仍落在 common no-compute 内（含非 compute device activity 的语义）
    no_task = result["distributed"]["common_no_device_task"]["intervals_ns"]
    no_compute = result["distributed"]["common_no_compute"]["intervals_ns"]
    assert all(not (a <= 30 and 40 <= b) for a, b in no_task)  # unknown 段不是 all-rank idle
    assert any(a <= 30 and 40 <= b for a, b in no_compute)  # 但仍是 all-rank no-compute


# ---- 10. round trip：artifact serialize/deserialize 稳定 ----


def test_rank_timeline_round_trip():
    r0 = mk_timeline(
        0, raw=[(0, 10), (20, 30)], sentinel=[(0, 10)], attributable=[(20, 30)], compute=[(20, 30)], unknown=[]
    )
    restored = RankTimelineIntervals.from_dict(r0.to_dict())
    assert restored == r0


def test_summary_round_trip(tmp_path: Path):
    r0 = mk_timeline(0, raw=[(0, 10), (20, 30), (40, 100)])
    r1 = mk_timeline(1, raw=[(0, 15), (25, 35), (45, 100)])
    summary = build_distributed_summary([r0, r1], "explicit_assertion", None, None)
    written = write_distributed_outputs(summary, tmp_path)
    loaded = json.loads(written["distributed_summary.json"].read_text(encoding="utf-8"))
    assert loaded == summary
    assert (tmp_path / "distributed_summary.md").exists()
    assert "common no-compute time that still contains" in (tmp_path / "distributed_summary.md").read_text(
        encoding="utf-8"
    )


# ---- invariants + compare（T9）----


def test_bounds_and_subset_invariants():
    r0 = mk_timeline(0, raw=[(5, 90)])
    r1 = mk_timeline(1, raw=[(0, 95)])
    result = run_common([r0, r1])
    w = result["window"]["duration_ms"]
    for metric in ("common_no_device_task", "common_no_compute"):
        ms = result["distributed"][metric]["duration_ms"]
        assert 0 <= ms <= w
    no_task = result["distributed"]["common_no_device_task"]["intervals_ns"]
    no_compute = result["distributed"]["common_no_compute"]["intervals_ns"]
    for a, b in no_task:
        assert any(x <= a and b <= y for x, y in no_compute)


def test_compare_distributed_delta():
    r0 = mk_timeline(0, raw=[(0, 10), (20, 30), (40, 100)])
    r1 = mk_timeline(1, raw=[(0, 15), (25, 35), (45, 100)])
    base = build_distributed_summary([r0, r1], "explicit_assertion", None, None)
    r1c = mk_timeline(1, raw=[(0, 18), (28, 38), (48, 100)])
    cand = build_distributed_summary([r0, r1c], "explicit_assertion", None, None)
    comparison = compare_distributed(base, cand)
    rows = {c["metric"]: c for c in comparison["comparisons"]}
    assert rows["common_no_device_task"]["baseline_ms"] > rows["common_no_device_task"]["candidate_ms"]
    assert rows["common_no_device_task"]["delta_ms"] < 0
    with pytest.raises(ValueError, match="window sets differ"):
        cand_bad = json.loads(json.dumps(cand))
        cand_bad["windows"] = cand_bad["windows"][:0]
        compare_distributed(base, cand_bad)


# ---- Ascend rank-timeline adapter（T1 抽取层）----


def _create_rank_db(path: Path, rank_id: int) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE PYTORCH_API (startNs INTEGER, endNs INTEGER, globalTid INTEGER,
            connectionId INTEGER, name INTEGER, sequenceNumber INTEGER, type INTEGER);
        CREATE TABLE CANN_API (startNs INTEGER, endNs INTEGER, type INTEGER,
            globalTid INTEGER, connectionId INTEGER, name INTEGER, depth INTEGER);
        CREATE TABLE TASK (startNs INTEGER, endNs INTEGER, deviceId INTEGER,
            connectionId INTEGER, globalTaskId INTEGER, globalPid INTEGER,
            taskType INTEGER, contextId INTEGER, streamId INTEGER, taskId INTEGER,
            modelId INTEGER, depth INTEGER);
        CREATE TABLE RANK_DEVICE_MAP (rankId INTEGER, deviceId INTEGER);
        CREATE TABLE NPU_INFO (id INTEGER, name TEXT);
        CREATE TABLE SESSION_TIME_INFO (startTimeNs INTEGER, endTimeNs INTEGER);
        CREATE TABLE STRING_IDS (id INTEGER, value TEXT);
        CREATE TABLE COMPUTE_TASK_INFO (globalTaskId INTEGER, name INTEGER, taskType INTEGER);
        CREATE TABLE COMMUNICATION_TASK_INFO (globalTaskId INTEGER, name INTEGER, taskType INTEGER);
        """
    )
    con.execute("INSERT INTO RANK_DEVICE_MAP VALUES (?, 8)", (rank_id,))
    con.execute("INSERT INTO NPU_INFO VALUES (8, 'Ascend910B')")
    con.execute("INSERT INTO SESSION_TIME_INFO VALUES (0, 1000)")
    # TASK：compute gid=7 / comm gid=8 / unknown gid=9 / sentinel conn=-1（gid 也会解析）
    con.executemany(
        "INSERT INTO TASK (startNs, endNs, deviceId, connectionId, globalTaskId, taskType) VALUES (?, ?, 8, ?, ?, 23)",
        [
            (100, 200, 100, 7),  # compute
            (300, 400, 101, 8),  # communication
            (500, 600, 102, 9),  # unknown（ attributable，非 compute/comm）
            (0, 1000, -1, None),  # sentinel：覆盖整个 session
        ],
    )
    con.execute("INSERT INTO COMPUTE_TASK_INFO VALUES (7, 1, 23)")
    con.execute("INSERT INTO COMMUNICATION_TASK_INFO VALUES (8, 2, 23)")
    con.commit()
    con.close()


def test_build_rank_timeline_layers(tmp_path: Path):
    db = tmp_path / "rank3.db"
    _create_rank_db(db, rank_id=3)
    timeline = build_rank_timeline_from_ascend_db(db)
    assert timeline.rank_id == 3
    assert timeline.unit == "ns"
    assert timeline.coverage_intervals == [(0, 1000)]
    assert timeline.task_row_count == 4 and timeline.sentinel_row_count == 1
    assert timeline.raw_task_intervals == [(0, 1000)]  # 哨兵覆盖全窗，raw union 一段
    assert timeline.sentinel_task_intervals == [(0, 1000)]
    assert timeline.attributable_task_intervals == [(100, 200), (300, 400), (500, 600)]
    assert timeline.compute_intervals == [(100, 200)]
    assert timeline.communication_intervals == [(300, 400)]
    assert timeline.communication_source == "COMMUNICATION_TASK_INFO"
    # unknown = attributable − compute − communication
    assert timeline.unknown_intervals == [(500, 600)]
    restored = RankTimelineIntervals.from_dict(timeline.to_dict())
    assert restored == timeline


def test_build_rank_timeline_multi_rankid_fail_fast(tmp_path: Path):
    db = tmp_path / "rank_bad.db"
    _create_rank_db(db, rank_id=0)
    con = sqlite3.connect(db)
    con.execute("INSERT INTO RANK_DEVICE_MAP VALUES (1, 9)")
    con.commit()
    con.close()
    with pytest.raises(DistributedClockError, match="exactly one rank"):
        build_rank_timeline_from_ascend_db(db)


def test_build_rank_timeline_comm_table_optional(tmp_path: Path):
    db = tmp_path / "rank0.db"
    _create_rank_db(db, rank_id=0)
    con = sqlite3.connect(db)
    con.execute("DROP TABLE COMMUNICATION_TASK_INFO")
    con.commit()
    con.close()
    timeline = build_rank_timeline_from_ascend_db(db)
    assert timeline.communication_source == "unavailable"
    assert timeline.communication_intervals == []
    # comm 表缺失不改变 raw/compute 语义
    assert timeline.compute_intervals == [(100, 200)]


# ---- CLI e2e（T6）----


def test_cli_analyze_distributed_end_to_end(tmp_path: Path):
    db0 = tmp_path / "db" / "rank0.db"
    db1 = tmp_path / "db" / "rank1.db"
    db0.parent.mkdir()
    _create_rank_db(db0, rank_id=0)
    _create_rank_db(db1, rank_id=1)
    out = tmp_path / "out"
    rc = cli_main(
        [
            "analyze-distributed",
            "--backend",
            "ascend",
            "--rank-input",
            f"0={db0}",
            "--rank-input",
            f"1={db1}",
            "--shared-clock",
            "--output",
            str(out),
            "--intervals-csv",
        ]
    )
    assert rc == 0
    summary = json.loads((out / "distributed_summary.json").read_text(encoding="utf-8"))
    assert summary["rank_ids"] == [0, 1]
    assert summary["clock_validation"]["status"] == "explicit_assertion"
    # 哨兵覆盖 [0,1000] -> raw 语义下两 rank 全程 device task：common no-task = 0
    assert summary["windows"][0]["distributed"]["common_no_device_task"]["duration_ms"] == 0.0
    # compute 只有 [100,200] -> common no-compute = 全窗减去该段
    no_compute = summary["windows"][0]["distributed"]["common_no_compute"]
    assert no_compute["intervals_ns"] == [[0, 100], [200, 1000]]
    assert (out / "distributed_summary.md").exists()
    assert (out / "distributed_intervals.csv").exists()


def test_cli_distributed_requires_clock_evidence_or_assertion(tmp_path: Path):
    db0 = tmp_path / "rank0.db"
    _create_rank_db(db0, rank_id=0)
    rc = cli_main(
        ["analyze-distributed", "--backend", "ascend", "--rank-input", f"0={db0}", "--output", str(tmp_path / "out")]
    )
    assert rc == 2  # fail fast：无 evidence 也无 --shared-clock


def test_cli_distributed_rank_mismatch_fail_fast(tmp_path: Path):
    db0 = tmp_path / "rank0.db"
    _create_rank_db(db0, rank_id=0)
    rc = cli_main(
        [
            "analyze-distributed",
            "--backend",
            "ascend",
            "--rank-input",
            f"7={db0}",
            "--shared-clock",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
