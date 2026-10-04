# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Ascend CSV adapter 测试（M2-T1）：fixture 按真实 torch_npu 列结构构造，
# 保留真实样本中的 \t 尾字符 quirk。AC-24：不需要 NPU。

import json
from pathlib import Path

import pytest

from vllm_omni.profiling.backends import analyze_ascend_csv
from vllm_omni.profiling.diagnosis import diagnose
from vllm_omni.profiling.schema import OptimizationEvidence

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_STEP_TRACE = """Device_id,Step,Computing,Communication(Not Overlapped),Overlapped,Communication,Free,Stage,Bubble,Communication(Not Overlapped and Exclude Receive),Preparing
0,1,85704.02,3873.52,630.54,4504.06,21471.45,111048.99,0,3873.52,0.67
"""

_KERNEL_DETAILS = '''Device_id,Model ID,Task ID,Stream ID,Name,Type,OP State,Accelerator Core,Start Time(us),Duration(us),Wait Time(us),Block Num,Mix Block Num,HF32 Eligible,Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Context ID
0,4294967295,1,1,aclnnLaserAttention,LaserAttention,dynamic,MIX_AIC,100.0\t,30.0,0,1,0,NO,"""[2,4096]""",FP16,ND,"""[2,4096]""",FP16,ND,N/A
0,4294967295,2,1,aclnnLaserAttention,LaserAttention,dynamic,MIX_AIC,200.0\t,20.0,0,1,0,NO,"""[2,4096]""",FP16,ND,"""[2,4096]""",FP16,ND,N/A
0,4294967295,3,2,aclnnConv3DV2,Conv3DV2,dynamic,AI_CORE,500.0\t,10.0,0,2,0,NO,"""[1,3,16]""",FP16,ND,"""[1,4,16]""",FP16,ND,N/A
0,4294967295,4,2,aclnnInplaceZero_ZerosLikeAiCore_ZerosLike,ZerosLike,dynamic,AI_VECTOR_CORE,700.0\t,5.0,0,1,0,NO,"""8""",INT64,ND,"""8""",INT64,ND,N/A
'''

_OPERATOR_DETAILS = """Name,Input Shapes,Call Stack,Host Self Duration(us),Host Total Duration(us),Device Self Duration(us),Device Total Duration(us),Device Self Duration With AICore(us),Device Total Duration With AICore(us)
aten::linear,,<stack>,100.0,120.0,80.0,50.0,80.0,50.0
aten::linear,,<stack>,10.0,15.0,8.0,5.0,8.0,5.0
aten::relu,,<stack>,2.0,2.0,0,0,0,0
"""

_API_STATISTIC = """Device_id,Level,API Name,Time(us),Count,Avg(us),Min(us),Max(us),Variance
0,acl,aclnnLaserAttention,50.0,2,25.0,20.0,30.0,25.0
0,acl,aclnnConv3DV2,5.0,1,5.0,5.0,5.0,0.0
0,host,BroadcastTo_Tiling,89.53,6,14.922,4.17,33.36,156.118
0,communication,hcclAllGather,10.0,1,10.0,10.0,10.0,0.0
"""

_OP_STATISTIC = """Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
0,LaserAttention,MIX_AIC,2,50.0,20.0,25.0,30.0,66.667
0,Conv3DV2,AI_CORE,1,10.0,10.0,10.0,10.0,13.333
"""


def write_ascend_fixtures(base: Path) -> Path:
    d = base / "ASCEND_PROFILER_OUTPUT"
    d.mkdir(parents=True, exist_ok=True)
    (d / "step_trace_time.csv").write_text(_STEP_TRACE, encoding="utf-8", newline="")
    (d / "kernel_details.csv").write_text(_KERNEL_DETAILS, encoding="utf-8", newline="")
    (d / "operator_details.csv").write_text(_OPERATOR_DETAILS, encoding="utf-8", newline="")
    (d / "api_statistic.csv").write_text(_API_STATISTIC, encoding="utf-8", newline="")
    (d / "op_statistic.csv").write_text(_OP_STATISTIC, encoding="utf-8", newline="")
    return d


@pytest.fixture()
def ascend_dir(tmp_path: Path) -> Path:
    return write_ascend_fixtures(tmp_path)


def test_run_metadata(ascend_dir):
    ev, _ = analyze_ascend_csv(ascend_dir)
    assert ev.run.backend == "ascend"
    assert ev.run.profiler == "torch_npu"
    # P0-1：CSV 只有 Device_id（设备号），rank 必须为 None
    assert ev.run.rank is None
    assert ev.run.device_id == "0"
    assert sorted(ev.run.source_files) == [
        "api_statistic.csv",
        "kernel_details.csv",
        "op_statistic.csv",
        "operator_details.csv",
        "step_trace_time.csv",
    ]


def test_run_id_stable_and_stamped(ascend_dir):
    ev1, store1 = analyze_ascend_csv(ascend_dir)
    ev2, store2 = analyze_ascend_csv(ascend_dir)
    assert ev1 == ev2
    assert len(ev1.run.run_id) == 12
    assert all(r.run_id == ev1.run.run_id for r in store1.records)
    assert all(r.run_id == ev2.run.run_id for r in store2.records)


def test_timeline_from_step_trace(ascend_dir):
    ev, _ = analyze_ascend_csv(ascend_dir)
    tl = ev.timeline
    assert tl.compute_ms == pytest.approx(85.70402)
    assert tl.communication_ms == pytest.approx(4.50406)
    # CANN Free = 暴露空闲
    assert tl.exposed_non_device_busy_ms == pytest.approx(21.47145)
    # device_busy 来自 kernel_details 区间 union（4 个不重叠区间）
    assert tl.device_busy_ms == pytest.approx(0.065)
    assert not hasattr(tl, "memory_ms")  # 决议 3


def test_workload_and_stage(ascend_dir):
    ev, _ = analyze_ascend_csv(ascend_dir)
    assert ev.workload.wall_ms == pytest.approx(111.04899)
    assert len(ev.workload.stage_times) == 1
    assert ev.workload.stage_times[0].name == "step_1"


def test_three_layers(ascend_dir):
    ev, _ = analyze_ascend_csv(ascend_dir)
    fw = [o for o in ev.operators if o.layer == "framework"]
    rt = [o for o in ev.operators if o.layer == "runtime"]
    dev = [o for o in ev.operators if o.layer == "device"]
    assert {o.name for o in fw} == {"aten::linear", "aten::relu"}
    # runtime 仅收 Level=acl；host/communication level 不混入
    assert {o.name for o in rt} == {"aclnnLaserAttention", "aclnnConv3DV2"}
    assert {o.name for o in dev} == {
        "aclnnLaserAttention",
        "aclnnConv3DV2",
        "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike",
    }
    # CSV 之间无关联键，parent_id 必须为 None（不猜测）
    assert all(o.parent_id is None for o in ev.operators)


def test_framework_row_semantics(ascend_dir):
    ev, _ = analyze_ascend_csv(ascend_dir)
    linear = next(o for o in ev.operators if o.name == "aten::linear")
    # operator_details 无 call count 列 -> calls 不可用
    assert linear.calls is None
    # P0-3：Self/Total 四列独立（fixture: Host Self 100+10, Host Total 120+15,
    # Device Self 80+8, Device Total 50+5）
    assert linear.host_self_ms == pytest.approx(110.0 / 1000.0)
    assert linear.host_inclusive_ms == pytest.approx(135.0 / 1000.0)
    assert linear.device_self_ms == pytest.approx(88.0 / 1000.0)
    assert linear.device_inclusive_ms == pytest.approx(55.0 / 1000.0)
    relu = next(o for o in ev.operators if o.name == "aten::relu")
    # Device Self/Total=0 是实测值，不得吞成 unavailable（AC-03 禁止反向造假）
    assert relu.device_self_ms == pytest.approx(0.0)
    assert relu.device_inclusive_ms == pytest.approx(0.0)


def test_runtime_rows_only_acl_level(ascend_dir):
    ev, _ = analyze_ascend_csv(ascend_dir)
    rt = {o.name: o for o in ev.operators if o.layer == "runtime"}
    assert rt["aclnnLaserAttention"].calls == 2
    assert rt["aclnnLaserAttention"].summed_host_ms == pytest.approx(50.0 / 1000.0)
    assert rt["aclnnConv3DV2"].summed_host_ms == pytest.approx(5.0 / 1000.0)
    assert "BroadcastTo_Tiling" not in rt
    assert "hcclAllGather" not in rt


def test_device_rows_and_core_types(ascend_dir):
    ev, _ = analyze_ascend_csv(ascend_dir)
    dev = {o.name: o for o in ev.operators if o.layer == "device"}
    assert dev["aclnnLaserAttention"].calls == 2
    assert dev["aclnnLaserAttention"].summed_device_ms == pytest.approx(50.0 / 1000.0)
    assert dev["aclnnLaserAttention"].median_device_us == pytest.approx(25.0)
    bm = ev.backend_metrics.ascend
    # Accelerator Core 时长按原语义保留，不翻译成 CUDA counter（AC-15/决议 3）
    assert bm["core_type_time_us"]["MIX_AIC"] == pytest.approx(50.0)
    assert bm["core_type_time_us"]["AI_CORE"] == pytest.approx(10.0)
    assert bm["core_type_time_us"]["AI_VECTOR_CORE"] == pytest.approx(5.0)


def test_op_statistic_verbatim_and_ratio_isolated(ascend_dir):
    # 决议 2：Ratio(%) 分母不明 -> 原样进 backend_metrics，绝不进 device_time_fraction
    ev, _ = analyze_ascend_csv(ascend_dir)
    rows = ev.backend_metrics.ascend["op_statistic"]
    assert len(rows) == 2
    assert rows[0]["OP Type"] == "LaserAttention"
    assert rows[0]["Ratio(%)"] == "66.667"
    assert all(s.profiler_ratio is None for s in ev.shapes)


def test_shape_fraction_denominator(ascend_dir):
    ev, _ = analyze_ascend_csv(ascend_dir)
    # task_sum = 30+20+10+5 = 65us
    by_shape = {(s.operator, s.shape): s for s in ev.shapes}
    s = by_shape[("aclnnLaserAttention", '"[2,4096]"')]
    assert s.calls == 2
    assert s.total_ms == pytest.approx(50.0 / 1000.0)
    assert s.device_time_fraction == pytest.approx(50.0 / 65.0)
    assert s.fraction_denominator == "summed_duration_all_device_tasks"


def test_runtime_api_mutually_exclusive(ascend_dir):
    # P0-2：acl API 关键字互斥分类。aclnnLaserAttention 含 launch(50us)，
    # aclnnConv3DV2 -> other(5us)；api_summed = 各分类之和
    ev, _ = analyze_ascend_csv(ascend_dir)
    r = ev.runtime
    # fixture 的两个 acl API 名都不含 launch/synchronize/malloc 关键字 -> 全部 other
    assert r.api_summed_ms == pytest.approx(55.0 / 1000.0)
    assert r.launch_summed_ms == 0.0
    assert r.launch_count == 0
    assert r.synchronization_summed_ms == 0.0
    assert r.allocation_summed_ms == 0.0
    assert r.other_summed_ms == pytest.approx(55.0 / 1000.0)
    total = r.launch_summed_ms + r.synchronization_summed_ms + r.allocation_summed_ms + r.other_summed_ms
    assert total == pytest.approx(r.api_summed_ms)


def test_bubble_preparing_in_backend_metrics(ascend_dir):
    # P1-4A：mapping doc 承诺的 Bubble/Preparing 落入 backend_metrics
    ev, _ = analyze_ascend_csv(ascend_dir)
    bm = ev.backend_metrics.ascend
    assert bm["step_trace_bubble_us"] == pytest.approx(0.0)
    assert bm["step_trace_preparing_us"] == pytest.approx(0.67)


def test_communication_stats(ascend_dir):
    ev, _ = analyze_ascend_csv(ascend_dir)
    assert ev.communication.total_ms == pytest.approx(4.50406)
    assert ev.communication.overlap_ms == pytest.approx(0.63054)
    assert ev.communication.overlap_ratio is None  # 分母语义不明，不猜测


def test_round_trip(ascend_dir):
    ev, _ = analyze_ascend_csv(ascend_dir)
    assert OptimizationEvidence.from_dict(ev.to_dict()) == ev


def test_tab_character_quirk_handled(ascend_dir):
    # 真实样本 Start Time 带尾部 \t；解析不得失败或改变数值
    ev1, _ = analyze_ascend_csv(ascend_dir)
    assert ev1.timeline.device_busy_ms == pytest.approx(0.065)


def test_empty_dir_fails_explicitly(tmp_path):
    with pytest.raises(ValueError, match="no known torch_npu CSV"):
        analyze_ascend_csv(tmp_path)


def test_db_only_dir_fails_explicitly(tmp_path):
    (tmp_path / "analysis.db").write_bytes(b"x")
    with pytest.raises(ValueError, match="no known torch_npu CSV"):
        analyze_ascend_csv(tmp_path)


def test_json_dumpable(ascend_dir):
    ev, _ = analyze_ascend_csv(ascend_dir)
    raw = json.loads(ev.to_json())
    assert raw["schema_version"] == "0.6"
    assert raw["run"]["backend"] == "ascend"


def test_no_api_timeline_no_gap_overlap_no_bound(ascend_dir):
    # MVP gate 审计：CSV 无 API 时间轴 -> gap_overlap_ms 必须为 None（不伪造），
    # 且 runtime 类 *_bound 不得输出（相关性证据缺席）
    ev, _ = analyze_ascend_csv(ascend_dir)
    assert ev.runtime.gap_overlap_ms is None
    classes = {c.class_ for c in diagnose(ev)}
    assert not classes & {"host_dispatch_bound", "synchronization_bound", "allocation_bound"}
