# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Ascend 大库 timeline correlation 测试（M4.1b）：fixture 复刻真实
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
#   PYTORCH: type=50001 aten::mm(conn=100) x2 / aten::relu(conn=102)；
#            type=50002 Enqueue@aclnnMul(conn=101)


def _create_fixture_db(path: Path) -> None:
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
    # COMPUTE_TASK_INFO：globalTaskId=7 -> npu_kernel_x；gid=8 -> npu_kernel_ambig
    con.execute("INSERT INTO COMPUTE_TASK_INFO VALUES (7, 7, 23)")
    con.execute("INSERT INTO COMPUTE_TASK_INFO VALUES (8, 9, 23)")
    con.commit()
    con.close()


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "ascend_pytorch_profiler_0.db"
    _create_fixture_db(p)
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
    # wall 来自 SESSION_TIME_INFO = 1.0ms
    assert ev.workload.wall_ms == pytest.approx(1.0)
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
    # 真实样本（勘察记录）：rankId=0, deviceId=8, Ascend910B，session 124.55s
    assert ev.run.rank == 0
    assert ev.run.device_id == "8"
    assert ev.run.device == "Ascend910B"
    assert ev.workload.wall_ms == pytest.approx(124550.0)
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
