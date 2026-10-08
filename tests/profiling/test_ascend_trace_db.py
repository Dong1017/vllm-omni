# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Ascend 大库 timeline correlation 测试（M4.1b/M4.1c）：fixture 复刻真实
# ascend_pytorch_profiler_0.db schema；真实 DB 存在时执行真实验证（CI 跳过）。

import sqlite3
from pathlib import Path

import pytest

from vllm_omni.profiling.backends import UnsupportedTraceDBError, analyze_ascend_trace_db
from vllm_omni.profiling.diagnosis import diagnose
from vllm_omni.profiling.schema import OptimizationEvidence

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REAL_DB = Path(
    "D:/world_model/outputs/h3_16die_validation_20260922/ref2va_15s_280ts/"
    "ref2va_15s_280ts/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db"
)

# fixture 时间轴设计（ns）：
#   session [0, 1_000_000] -> wall 1.0ms
#   TASK busy [100_000, 400_000] -> busy 0.3ms；gap [0,100k]+[400k,1M] = 0.7ms
#   CANN launch   conn=100 [450_000, 500_000] 全在 gap（50us）
#   CANN sync     conn=102 [600_000, 650_000] 全在 gap（50us）
#   CANN alloc    conn=103 [50_000, 60_000]   全在 gap（10us）
#   CANN other    conn=104 [100_000, 150_000] 全在 busy（对 gap 零贡献）
#   TASK: busy 任务 + conn=100/102 对应任务 + 无 COMPUTE_TASK_INFO 的任务
#         + conn=-1 哨兵（M4.1c：永不进 busy union / 永不解析 kernel 名）
#   PYTORCH: type=50001 aten::mm(conn=100) x2 / aten::relu(conn=102)；
#            type=50002 Enqueue@aclnnMul(conn=101)
#   OVERLAP_ANALYSIS（overlap=True，M4.1c authoritative path，设计值）：
#     type0 compute [100k,400k]=300us；type1 comm [350k,450k]=100us（与 compute
#     重叠 50us）；type2 not-overlapped [400k,450k]=50us；type3 free
#     [0,100k]+[450k,600k]=250us；type9 未知 [900k,950k]（只记录不进 busy/idle）
#     -> busy=union(0∪1)=[100k,450k]=350us；compute+comm summed=400us>busy
#     （overlap 证据）；exposed=Free=250us；window=coverage union（含 type9）
#     =650us；span=[0,950k]=950us 仅 informational


def _create_fixture_db(path: Path, *, overlap: bool = False, overlap_rows: bool = True) -> None:
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
        """
    )
    con.execute("INSERT INTO RANK_DEVICE_MAP VALUES (0, 8)")
    con.execute("INSERT INTO NPU_INFO VALUES (8, 'Ascend910B')")
    con.execute("INSERT INTO SESSION_TIME_INFO VALUES (0, 1000000)")
    # STRING_IDS：1..8 手工分配
    for sid, value in [
        (1, "aten::mm"),
        (2, "aten::relu"),
        (3, "aclnnLaunchKernelWithHostArgs"),
        (4, "aclrtSynchronizeStream"),
        (5, "aclrtMalloc"),
        (6, "aclnnMul"),
        (7, "npu_kernel_x"),
        (8, "Enqueue@aclnnMul"),
        (9, "npu_kernel_ambig"),
    ]:
        con.execute("INSERT INTO STRING_IDS VALUES (?, ?)", (sid, value))
    # PYTORCH：type=50001 aten 调用（带 conn），type=50002 Enqueue（带 conn）
    con.executemany(
        "INSERT INTO PYTORCH_API (startNs, endNs, globalTid, connectionId, name, type) VALUES (?, ?, 1, ?, ?, 50001)",
        [(10_000, 20_000, 100, 1), (30_000, 40_000, 101, 1), (50_000, 55_000, 102, 2)],
    )
    con.execute(
        "INSERT INTO PYTORCH_API (startNs, endNs, globalTid, connectionId, name, type)"
        " VALUES (60_000, 61_000, 1, 101, 8, 50002)"
    )
    # CANN：conn=100 launch(110us, 全在 gap) / conn=101 other(150us, busy 内) /
    #       conn=102 sync(50us, gap 尾部) / conn=103 alloc(10us, gap)
    con.executemany(
        "INSERT INTO CANN_API (startNs, endNs, connectionId, name) VALUES (?, ?, ?, ?)",
        [(450_000, 560_000, 100, 3), (100_000, 250_000, 101, 6), (700_000, 750_000, 102, 4), (50_000, 60_000, 103, 5)],
    )
    # TASK：busy 任务 [100k,400k]（conn=999 无 PYTORCH/无 name），conn=100 任务
    #       [200k,250k]（COMPUTE_TASK_INFO 有 kernel 名，落在 busy 区间内），
    #       conn=102 任务 [600k,650k]（类型名兜底）
    con.executemany(
        "INSERT INTO TASK (startNs, endNs, deviceId, connectionId, globalTaskId, taskType) VALUES (?, ?, 8, ?, ?, 23)",
        [
            (100_000, 400_000, 999, 0),
            (200_000, 250_000, 100, 7),
            (600_000, 650_000, 102, 0),
            # 歧义行：同设备名两实例分别归 aten::mm / aten::relu -> parent None
            (300_000, 350_000, 100, 8),
            (350_000, 400_000, 102, 8),
        ],
    )
    # M4.1c：conn=-1 哨兵两行（per-stream capture marker）。第一行 globalTaskId=7
    # 与真实任务 npu_kernel_x 同 gid —— 门控失效就会错误解析进 named operator；
    # 第二行 gid=NULL 连 COMPUTE_TASK_INFO 都无。两行都覆盖/穿越 busy 区间，
    # 若进了 busy union 会污染所有 timeline 数字。
    con.executemany(
        "INSERT INTO TASK (startNs, endNs, deviceId, connectionId, globalTaskId, taskType) VALUES (?, ?, 8, -1, ?, 37)",
        [(0, 1_000_000, 7), (10_000, 20_000, None)],
    )
    # COMPUTE_TASK_INFO：globalTaskId=7 -> npu_kernel_x；gid=8 -> npu_kernel_ambig
    con.execute("INSERT INTO COMPUTE_TASK_INFO VALUES (7, 7, 23)")
    con.execute("INSERT INTO COMPUTE_TASK_INFO VALUES (8, 9, 23)")
    if overlap:
        con.executescript(
            """
            CREATE TABLE OVERLAP_ANALYSIS (id INTEGER, deviceId INTEGER,
                startNs INTEGER, endNs INTEGER, type INTEGER);
            """
        )
        if overlap_rows:
            con.executemany(
                "INSERT INTO OVERLAP_ANALYSIS (startNs, endNs, type) VALUES (?, ?, ?)",
                [
                    (100_000, 400_000, 0),
                    (350_000, 450_000, 1),
                    (400_000, 450_000, 2),
                    (0, 100_000, 3),
                    (450_000, 600_000, 3),
                    (900_000, 950_000, 9),
                ],
            )
    con.commit()
    con.close()


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "ascend_pytorch_profiler_0.db"
    _create_fixture_db(p)
    return p


@pytest.fixture()
def db_overlap_path(tmp_path: Path) -> Path:
    p = tmp_path / "ascend_pytorch_profiler_0.db"
    _create_fixture_db(p, overlap=True)
    return p


def test_rank_from_rank_device_map(db_path):
    # P0-1：rank 来自 RANK_DEVICE_MAP.rankId=0；deviceId=8 单独存放，不得混用
    ev, _ = analyze_ascend_trace_db(db_path)
    assert ev.run.rank == 0
    assert ev.run.device_id == "8"
    assert ev.run.device == "Ascend910B"
    assert ev.run.run_id is not None


def test_wall_session_and_busy_union(db_path):
    ev, _ = analyze_ascend_trace_db(db_path)
    # wall 来自 SESSION_TIME_INFO = 1.0ms（fallback path：session 是唯一窗口证据）
    assert ev.workload.wall_ms == pytest.approx(1.0)
    assert ev.timeline.window_ms is None  # 无 OVERLAP_ANALYSIS -> 无 timeline 窗口
    # busy = TASK union：[100k,400k]∪[200k,250k]（子集）∪[600k,650k] = 350us
    assert ev.timeline.device_busy_ms == pytest.approx(0.35)
    assert ev.timeline.exposed_non_device_busy_ms == pytest.approx(0.65)


def test_cann_categories_and_gap_overlap(db_path):
    # CANN 分类：launch conn=100(110us) / other conn=101(150us) /
    # sync conn=102(50us) / alloc conn=103(10us)
    ev, _ = analyze_ascend_trace_db(db_path)
    r = ev.runtime
    assert r.api_summed_ms == pytest.approx(0.32)  # 110+150+50+10 us
    assert r.launch_summed_ms == pytest.approx(0.11)
    assert r.synchronization_summed_ms == pytest.approx(0.05)
    assert r.allocation_summed_ms == pytest.approx(0.01)
    assert r.other_summed_ms == pytest.approx(0.15)
    # gap overlap：gap = [0,100k]∪[400k,600k]∪[650k,1M]
    # launch [450k,560k] 全在 gap -> 110us；other [100k,250k] 与 busy 重叠 -> 0
    assert r.gap_overlap_ms["launch"] == pytest.approx(0.11)
    assert r.gap_overlap_ms["other"] == pytest.approx(0.0)


def test_sentinel_never_attributed(db_path):
    # M4.1c 裁定 1（fallback path）：conn=-1 哨兵 globalTaskId=7 与 npu_kernel_x
    # 同 gid —— 若未门控会错误解析进 named operator；必须落类型名兜底行
    # 且不进 busy union（busy union 数字由 test_wall_session_and_busy_union 守住）
    ev, _ = analyze_ascend_trace_db(db_path)
    dev = {o.name: o for o in ev.operators if o.layer == "device"}
    x = dev["npu_kernel_x"]
    assert x.calls == 1  # 只有 conn=100 的真实任务行
    assert x.summed_device_ms == pytest.approx(0.05)  # [200k,250k] = 50us
    sentinel = dev["npu_task_type_37"]
    assert sentinel.parent_id is None
    assert sentinel.calls == 2  # 两行哨兵（gid=7 / gid=None）


def test_overlap_authoritative_timeline(db_overlap_path):
    # M4.1c 裁定 2：busy = OVERLAP type0∪type1 interval union = [100k,450k]=350us；
    # compute+comm summed = 400us > busy —— summed 相加不得冒充 busy
    ev, _ = analyze_ascend_trace_db(db_overlap_path)
    bm = ev.backend_metrics.ascend
    assert bm["timeline_source"] == "overlap_analysis"
    assert ev.timeline.device_busy_ms == pytest.approx(0.35)
    assert ev.timeline.compute_ms == pytest.approx(0.30)
    assert ev.timeline.communication_ms == pytest.approx(0.10)
    # 裁定 3：exposed = Free union（profiler 定义的 device idle），不再是 session−busy
    assert ev.timeline.exposed_non_device_busy_ms == pytest.approx(0.25)
    # gate P0（2026-10-08）：timeline 覆盖窗口写 timeline.window_ms（全部 OVERLAP 行
    # 的 coverage union = [0,600k]∪[900k,950k] = 650us），workload.wall_ms 保持
    # None——DB 无独立 step/workload boundary，不得冒名
    assert ev.workload.wall_ms is None
    assert ev.timeline.window_ms == pytest.approx(0.65)
    # session 窗口与 span 差值如实另记（informational）；unknown coverage 显式记录
    assert bm["session_window_ms"] == pytest.approx(1.0)
    assert bm["overlap_window_ms"] == pytest.approx(0.95)
    assert bm["outside_overlap_window_ms"] == pytest.approx(0.05)
    assert bm["overlap_unknown_types"] == {"9": 1}
    assert bm["overlap_unknown_window_ms"] == pytest.approx(0.05)


def test_overlap_communication_exposure(db_overlap_path):
    # 裁定 4：total(type1)=0.10ms = not_overlapped(type2)=0.05 + overlapped=0.05；
    # exposure 分解，不是 wall partition
    ev, _ = analyze_ascend_trace_db(db_overlap_path)
    c = ev.communication
    assert c.calls == 1
    assert c.total_ms == pytest.approx(0.10)
    assert c.overlap_ms == pytest.approx(0.05)
    assert c.overlap_ratio == pytest.approx(0.5)
    assert ev.backend_metrics.ascend["communication_not_overlapped_ms"] == pytest.approx(0.05)


def test_overlap_gap_correlation_uses_free(db_overlap_path):
    # gap = Free 区间：launch [450k,560k] ∩ [450k,600k] = 110us 全落 Free
    ev, _ = analyze_ascend_trace_db(db_overlap_path)
    assert ev.runtime.gap_overlap_ms["launch"] == pytest.approx(0.11)
    assert ev.runtime.gap_overlap_ms["other"] == pytest.approx(0.0)


def test_overlap_empty_table_is_unavailable(tmp_path):
    # OVERLAP_ANALYSIS 在但零行：timeline 不可用（None）——不回退 TASK、不填 0 冒充
    p = tmp_path / "ascend_pytorch_profiler_0.db"
    _create_fixture_db(p, overlap=True, overlap_rows=False)
    ev, _ = analyze_ascend_trace_db(p)
    bm = ev.backend_metrics.ascend
    assert bm["timeline_source"] == "overlap_analysis"
    assert bm["overlap_analysis_rows"] == 0
    assert ev.timeline.device_busy_ms is None
    assert ev.timeline.exposed_non_device_busy_ms is None
    assert ev.timeline.compute_ms is None
    assert ev.timeline.communication_ms is None
    assert ev.timeline.window_ms is None
    # session 是唯一窗口证据：wall 仍如实来自 SESSION_TIME_INFO
    assert ev.workload.wall_ms == pytest.approx(1.0)
    assert ev.communication.total_ms is None
    assert ev.runtime.gap_overlap_ms is None


def test_overlap_window_partition_invariants(db_overlap_path):
    # gate P0 item 4：interval-level invariants——busy = union(type0,type1)，
    # busy ∩ Free = []；busy ∪ Free ∪ unknown coverage 铺满已分类 window
    import sqlite3

    from vllm_omni.profiling.intervals import intersect, union
    from vllm_omni.profiling.intervals import total as iv_total

    ev, _ = analyze_ascend_trace_db(db_overlap_path)
    con = sqlite3.connect(f"file:{db_overlap_path.as_posix()}?mode=ro", uri=True)
    rows = con.execute("SELECT startNs, endNs, type FROM OVERLAP_ANALYSIS").fetchall()
    con.close()
    busy_iv = union([r[:2] for r in rows if r[2] in (0, 1)])
    free_iv = union([r[:2] for r in rows if r[2] == 3])
    unknown_iv = union([r[:2] for r in rows if r[2] not in (0, 1, 2, 3)])
    all_iv = union([r[:2] for r in rows])
    # busy 与 Free 不相交（端点相接不计）
    assert iv_total(intersect(busy_iv, free_iv)) == 0
    # busy ∪ Free ∪ unknown = 全部分类 coverage（unknown 显式记录，known 不声称完整）
    assert iv_total(union(busy_iv + free_iv + unknown_iv)) == iv_total(all_iv)
    # window_ms 与 coverage union 一致（ms <- ns）
    assert ev.timeline.window_ms == pytest.approx(iv_total(all_iv) / 1e6)
    # known busy∪Free 不覆盖完整 window（unknown type9 在 [900k,950k]）
    assert iv_total(union(busy_iv + free_iv)) == pytest.approx(600_000)
    assert ev.backend_metrics.ascend["overlap_unknown_window_ms"] == pytest.approx(0.05)


def test_overlap_schema_missing_column_fails(tmp_path):
    # 表在但缺列 -> 显式 unsupported（存在即承诺 schema，不允许静默降级到 fallback）
    p = tmp_path / "ascend_pytorch_profiler_0.db"
    _create_fixture_db(p)
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE OVERLAP_ANALYSIS (startNs INTEGER, endNs INTEGER)")
    con.commit()
    con.close()
    with pytest.raises(UnsupportedTraceDBError, match="OVERLAP_ANALYSIS missing columns"):
        analyze_ascend_trace_db(p)


def test_three_layers_with_parent_linkage(db_path):
    # M4.1b 链路：TASK(conn=100) -> CANN(conn=100) -> PYTORCH(aten::mm)
    # -> device 行 npu_kernel_x 的 parent 唯一归属 framework aten::mm
    ev, _ = analyze_ascend_trace_db(db_path)
    fw = {o.name: o for o in ev.operators if o.layer == "framework"}
    dev = {o.name: o for o in ev.operators if o.layer == "device"}
    assert set(fw) == {"aten::mm", "aten::relu"}
    mm = fw["aten::mm"]
    assert mm.calls == 2
    x = dev["npu_kernel_x"]
    assert x.parent_id == mm.id
    # 无 COMPUTE_TASK_INFO 名的任务 -> 类型名兜底（如实，不猜测）；
    # 该聚合行的实例来自 conn=999（unresolved）与 conn=102（aten::relu），
    # P0-1：存在 unresolved 实例 -> parent=None（"部分未知"不得归唯一已知）
    fallback = next(o for o in dev.values() if o.name.startswith("npu_task_type_"))
    assert fallback.parent_id is None
    # P0-1 歧义回归：同设备名实例分别 resolve 到两个 framework op
    # （conn=100→aten::mm / conn=102→aten::relu）-> fw_names 集合大小 2 -> parent=None
    ambig = dev["npu_kernel_ambig"]
    assert ambig.parent_id is None


def test_diagnosis_correlated_bound_fires(db_path):
    # M4.1 证据链在 Ascend DB 路径成立：launch 100% 落 gap 且 coverage 50/700 >= 0.10
    ev, _ = analyze_ascend_trace_db(db_path)
    candidates = {c.class_: c.confidence for c in diagnose(ev)}
    assert candidates.get("host_dispatch_bound") in ("medium", "high")


def test_round_trip(db_path):
    ev, _ = analyze_ascend_trace_db(db_path)
    assert OptimizationEvidence.from_dict(ev.to_dict()) == ev


def test_round_trip_overlap_path(db_overlap_path):
    # authoritative path：communication/backend_metrics（含 str 化 unknown types）也要稳定 round-trip
    ev, _ = analyze_ascend_trace_db(db_overlap_path)
    assert OptimizationEvidence.from_dict(ev.to_dict()) == ev


def test_unsupported_schema_missing_table(tmp_path):
    p = tmp_path / "ascend_pytorch_profiler_0.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE something_else (a INTEGER)")
    con.commit()
    con.close()
    with pytest.raises(UnsupportedTraceDBError, match="missing tables"):
        analyze_ascend_trace_db(p)


def test_unsupported_schema_missing_column(tmp_path):
    p = tmp_path / "ascend_pytorch_profiler_0.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE CANN_API (startNs INTEGER, endNs INTEGER, connectionId INTEGER, name INTEGER)")
    # P1-2：COMPUTE_TASK_INFO 已进必需表；此处建全列表，改从 TASK 缺列触发列级错误
    con.execute("CREATE TABLE COMPUTE_TASK_INFO (globalTaskId INTEGER, name INTEGER)")
    con.execute("CREATE TABLE TASK (startNs INTEGER, endNs INTEGER, deviceId INTEGER, connectionId INTEGER)")
    con.execute("CREATE TABLE RANK_DEVICE_MAP (rankId INTEGER, deviceId INTEGER)")
    con.execute("CREATE TABLE NPU_INFO (id INTEGER, name TEXT)")
    con.execute("CREATE TABLE SESSION_TIME_INFO (startTimeNs INTEGER, endTimeNs INTEGER)")
    con.execute("CREATE TABLE STRING_IDS (id INTEGER, value TEXT)")
    con.execute(
        "CREATE TABLE PYTORCH_API (startNs INTEGER, endNs INTEGER, connectionId INTEGER, name INTEGER, type INTEGER)"
    )
    con.commit()
    con.close()
    with pytest.raises(UnsupportedTraceDBError, match="missing columns"):
        analyze_ascend_trace_db(p)


def test_missing_db_fails(tmp_path):
    with pytest.raises(UnsupportedTraceDBError, match="not found"):
        analyze_ascend_trace_db(tmp_path / "nope.db")


@pytest.mark.skipif(not REAL_DB.exists(), reason="real trace DB not available (stays prepared_not_run in CI)")
def test_real_trace_db_validation():
    ev, store = analyze_ascend_trace_db(REAL_DB)
    # 真实样本（勘察记录）：rankId=0, deviceId=8, Ascend910B；
    # gate P0：workload.wall_ms = None（无 step boundary 证据），
    # timeline 覆盖窗口 111.049s 落 timeline.window_ms，session 124.550s 另记
    assert ev.run.rank == 0
    assert ev.run.device_id == "8"
    assert ev.run.device == "Ascend910B"
    assert ev.workload.wall_ms is None
    assert ev.timeline.window_ms == pytest.approx(111049.0, rel=1e-3)
    assert ev.backend_metrics.ascend["session_window_ms"] == pytest.approx(124550.0, rel=1e-3)
    assert ev.timeline.device_busy_ms > 0
    r = ev.runtime
    # 互斥分类恒等式
    total = r.launch_summed_ms + r.synchronization_summed_ms + r.allocation_summed_ms + r.other_summed_ms
    assert total == pytest.approx(r.api_summed_ms)
    # 三层行都存在
    layers = {o.layer for o in ev.operators}
    assert layers == {"framework", "runtime", "device"}
    # P0-3：framework 行必须落 host_inclusive_ms（inclusive 语义专用字段）
    fw_rows = [o for o in ev.operators if o.layer == "framework"]
    assert fw_rows and all(o.host_inclusive_ms is not None for o in fw_rows)
    # gap_overlap 保守可用：与 exposed gap 同源
    assert ev.runtime.gap_overlap_ms is not None
    assert all(v >= 0.0 for v in ev.runtime.gap_overlap_ms.values())
    # M4.1c authoritative（真实样本勘察值，m41b-conn-neg1-investigation.md）：
    # compute 85.704s / comm 4.504s / not-overlapped 3.874s / free 21.471s /
    # overlap 窗口 span 111.049s / session 124.550s
    bm = ev.backend_metrics.ascend
    assert bm["timeline_source"] == "overlap_analysis"
    assert ev.timeline.compute_ms == pytest.approx(85704.0, rel=1e-3)
    assert ev.timeline.communication_ms == pytest.approx(4504.0, rel=1e-3)
    # busy = type0∪type1 union ≈ 85704 + 4504 − 630(overlapped) = 89578ms
    assert ev.timeline.device_busy_ms == pytest.approx(89578.0, rel=0.01)
    assert ev.timeline.exposed_non_device_busy_ms == pytest.approx(21471.0, rel=1e-3)
    c = ev.communication
    assert c.total_ms == pytest.approx(4504.0, rel=1e-3)
    assert c.overlap_ms == pytest.approx(630.0, rel=0.05)
    assert bm["communication_not_overlapped_ms"] == pytest.approx(3874.0, rel=1e-3)
    assert bm["overlap_window_ms"] == pytest.approx(111049.0, rel=1e-3)
    # 裁定 5：session − overlap 窗口 ≈ 13.5s，是 capture 初始化/未覆盖区间，不是 idle
    assert bm["outside_overlap_window_ms"] == pytest.approx(13501.0, rel=0.02)
    # gate P0 item 4（真实库）：interval-level invariants——独立 SQL 重读
    # OVERLAP_ANALYSIS，busy∩Free = []，busy∪Free 覆盖完整已分类 coverage
    import sqlite3

    from vllm_omni.profiling.intervals import intersect, union
    from vllm_omni.profiling.intervals import total as iv_total

    con = sqlite3.connect(f"file:{REAL_DB.as_posix()}?mode=ro", uri=True)
    rows = con.execute("SELECT startNs, endNs, type FROM OVERLAP_ANALYSIS").fetchall()
    con.close()
    assert {r[2] for r in rows} == {0, 1, 2, 3}  # 无 unknown type：known partition 可声称完整
    busy_iv = union([r[:2] for r in rows if r[2] in (0, 1)])
    free_iv = union([r[:2] for r in rows if r[2] == 3])
    assert iv_total(intersect(busy_iv, free_iv)) == 0
    window_ns = ev.timeline.window_ms * 1e6
    assert iv_total(union(busy_iv + free_iv)) == pytest.approx(window_ns, rel=1e-6)
    # 实测样本上 coverage union 与 span 一致（无缝铺满，2026-10-08 只读核查）
    assert iv_total(union([r[:2] for r in rows])) == pytest.approx(window_ns, rel=1e-6)


def test_multi_rank_fail_fast(tmp_path):
    # final-review：真实双 rankId fixture -> 显式 fail fast（M4.1b 不做分布式）
    p = tmp_path / "ascend_pytorch_profiler_0.db"
    _create_fixture_db(p)
    con = sqlite3.connect(p)
    con.execute("INSERT INTO RANK_DEVICE_MAP VALUES (1, 9)")
    con.commit()
    con.close()
    with pytest.raises(UnsupportedTraceDBError, match="multi-rank"):
        analyze_ascend_trace_db(p)
