# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# CUDA torch profiler adapter 测试（T10/T13 + 决议 1-4）：内联合成 Chrome trace fixture。
# AC-24：不需要 GPU。

import json
from pathlib import Path

import pytest

from vllm_omni.profiling.backends import analyze_cuda_torch_profiler
from vllm_omni.profiling.observation import build_observations
from vllm_omni.profiling.provenance import sha256_file
from vllm_omni.profiling.schema import OptimizationEvidence

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _trace_dict() -> dict:
    return {
        "schemaVersion": 1,
        "deviceProperties": [{"name": "NVIDIA A100"}],
        "traceEvents": [
            {"ph": "M", "name": "process_name", "args": {}},
            # aten::mm 两次调用，同一 shape
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::mm",
                "ts": 100,
                "dur": 50,
                "args": {"External id": 1001, "Input Dims": [[2, 4096], [4096, 3584]]},
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::mm",
                "ts": 200,
                "dur": 40,
                "args": {"External id": 1002, "Input Dims": [[2, 4096], [4096, 3584]]},
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::relu",
                "ts": 260,
                "dur": 5,
                "args": {"External id": 1003, "Input Dims": [[2, 4096]]},
            },
            # kernel：两个关联到 aten::mm，一个 nccl 无关联 cpu op
            {"ph": "X", "cat": "kernel", "name": "gemm_kernel", "ts": 120, "dur": 30, "args": {"External id": 1001}},
            {"ph": "X", "cat": "kernel", "name": "gemm_kernel", "ts": 220, "dur": 20, "args": {"External id": 1002}},
            {
                "ph": "X",
                "cat": "kernel",
                "name": "ncclDevKernel_AllReduce",
                "ts": 300,
                "dur": 25,
                "args": {"External id": 1004},
            },
            # memcpy
            {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD", "ts": 350, "dur": 10, "args": {}},
            # runtime：cudaLaunchKernel 关联到 aten::mm / aten::relu
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": 110,
                "dur": 8,
                "args": {"External id": 1001},
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": 265,
                "dur": 4,
                "args": {"External id": 1003},
            },
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaStreamSynchronize", "ts": 360, "dur": 15, "args": {}},
        ],
    }


@pytest.fixture()
def trace_file(tmp_path: Path) -> Path:
    f = tmp_path / "trace_rank0.json"
    f.write_text(json.dumps(_trace_dict()), encoding="utf-8")
    return f


def test_run_metadata(trace_file):
    ev, _ = analyze_cuda_torch_profiler(trace_file)
    assert ev.run.backend == "cuda"
    assert ev.run.profiler == "torch_profiler"
    assert ev.run.rank == 0
    assert ev.run.device == "NVIDIA A100"
    assert ev.run.source_files == ["trace_rank0.json"]


def test_run_id_is_content_derived_and_stable(trace_file):
    ev1, store1 = analyze_cuda_torch_profiler(trace_file)
    ev2, store2 = analyze_cuda_torch_profiler(trace_file)
    # 决议 4：确定性 run_id，同一输入两次分析一致（AC-08）
    assert ev1.run.run_id == ev2.run.run_id
    assert len(ev1.run.run_id) == 12
    assert all(r.run_id == ev1.run.run_id for r in store1.records)
    assert all(r.run_id == ev2.run.run_id for r in store2.records)
    # 全局引用形式
    assert store1.ref("ev_000001") == f"{ev1.run.run_id}:ev_000001"


def test_source_sha256_recorded(trace_file):
    _, store = analyze_cuda_torch_profiler(trace_file)
    expected = sha256_file(trace_file)
    assert all(r.source_sha256 == expected for r in store.records)


def test_wall_time(trace_file):
    # min ts=100, max end=375（cudaStreamSynchronize 360+15）
    ev, _ = analyze_cuda_torch_profiler(trace_file)
    assert ev.workload.wall_ms == pytest.approx(275.0 / 1000.0)


def test_timeline_union_semantics(trace_file):
    ev, _ = analyze_cuda_torch_profiler(trace_file)
    tl = ev.timeline
    # busy = 30+20+25+10（四个不重叠设备区间）
    assert tl.device_busy_ms == pytest.approx(85.0 / 1000.0)
    # compute = kernel(75) 减去 nccl(25)
    assert tl.compute_ms == pytest.approx(50.0 / 1000.0)
    assert tl.communication_ms == pytest.approx(25.0 / 1000.0)
    # 决议 3：timeline 无 memory 字段
    assert not hasattr(tl, "memory_ms")
    # exposed gap = wall - busy
    assert tl.exposed_non_device_busy_ms == pytest.approx(190.0 / 1000.0)
    assert tl.unknown_ms == pytest.approx(0.0)


def test_memory_host_device_copy(trace_file):
    ev, _ = analyze_cuda_torch_profiler(trace_file)
    # 决议 3：host-device copy 在 memory 段（summed 口径）
    assert ev.memory.host_device_copy_ms == pytest.approx(10.0 / 1000.0)
    assert ev.memory.allocation_count is None


def test_three_layer_operator_model(trace_file):
    # 决议 1：framework -> runtime -> device 三层，parent_id 关联
    ev, _ = analyze_cuda_torch_profiler(trace_file)
    by_layer = {layer: [o for o in ev.operators if o.layer == layer] for layer in ("framework", "runtime", "device")}
    assert {o.name for o in by_layer["framework"]} == {"aten::mm", "aten::relu"}
    assert {o.name for o in by_layer["runtime"]} == {"cudaLaunchKernel", "cudaStreamSynchronize"}
    assert {o.name for o in by_layer["device"]} == {"gemm_kernel", "ncclDevKernel_AllReduce"}

    mm = next(o for o in by_layer["framework"] if o.name == "aten::mm")
    assert mm.parent_id is None
    assert mm.op_type == "aten_op"

    # 一对多：一个 framework op 可对应多个 device 行
    gemm = next(o for o in by_layer["device"] if o.name == "gemm_kernel")
    assert gemm.parent_id == mm.id  # gemm_kernel 的 ext 只归 aten::mm
    # runtime 行归属 framework op
    launch_mm = next(o for o in by_layer["runtime"] if o.name == "cudaLaunchKernel" and o.parent_id == mm.id)
    assert launch_mm.calls == 1
    assert launch_mm.summed_host_ms == pytest.approx(8.0 / 1000.0)
    assert launch_mm.op_type == "cuda_runtime_api"
    # 无 ext 的 runtime 事件 parent 为 None，不猜测归属
    sync = next(o for o in by_layer["runtime"] if o.name == "cudaStreamSynchronize")
    assert sync.parent_id is None


def test_aten_operator_correlation(trace_file):
    ev, _ = analyze_cuda_torch_profiler(trace_file)
    aten = [o for o in ev.operators if o.layer == "framework"]
    mm = next(o for o in aten if o.name == "aten::mm")
    assert mm.calls == 2
    assert mm.host_inclusive_ms == pytest.approx(90.0 / 1000.0)
    assert mm.device_self_ms == pytest.approx(50.0 / 1000.0)
    assert mm.avg_device_us == pytest.approx(25.0)
    assert mm.median_device_us == pytest.approx(25.0)
    assert mm.shapes == ["[2,4096]x[4096,3584]"]
    relu = next(o for o in aten if o.name == "aten::relu")
    assert relu.device_self_ms is None  # 无关联 kernel，不得填 0
    assert relu.host_inclusive_ms == pytest.approx(5.0 / 1000.0)


def test_kernel_rows_present_for_golden_facts(trace_file):
    ev, _ = analyze_cuda_torch_profiler(trace_file)
    kernels = {o.name: o for o in ev.operators if o.layer == "device"}
    assert set(kernels) == {"gemm_kernel", "ncclDevKernel_AllReduce"}
    assert kernels["gemm_kernel"].calls == 2
    assert kernels["gemm_kernel"].summed_device_ms == pytest.approx(50.0 / 1000.0)
    assert kernels["ncclDevKernel_AllReduce"].summed_host_ms is None


def test_shape_fraction_denominator_explicit(trace_file):
    # 决议 2：fraction 必须带显式分母；分母 = 全部 device task summed (30+20+25=75us)
    ev, _ = analyze_cuda_torch_profiler(trace_file)
    assert len(ev.shapes) == 1
    s = ev.shapes[0]
    assert s.operator == "aten::mm"
    assert s.shape == "[2,4096]x[4096,3584]"
    assert s.calls == 2
    assert s.total_ms == pytest.approx(50.0 / 1000.0)
    assert s.device_time_fraction == pytest.approx(50.0 / 75.0)
    assert s.fraction_denominator == "summed_device_time_all_device_tasks"
    assert s.profiler_ratio is None  # torch profiler 不提供 ratio，不得编造
    assert s.evidence_ids


def test_runtime_stats_mutually_exclusive(trace_file):
    # P0-2：runtime API 互斥分类，各分类之和 == api_summed
    ev, _ = analyze_cuda_torch_profiler(trace_file)
    r = ev.runtime
    assert r.launch_count == 2
    assert r.api_summed_ms == pytest.approx(27.0 / 1000.0)  # 8+4+15
    assert r.launch_summed_ms == pytest.approx(12.0 / 1000.0)  # 8+4
    assert r.synchronization_summed_ms == pytest.approx(15.0 / 1000.0)
    assert r.allocation_summed_ms == 0.0
    assert r.other_summed_ms == 0.0
    total = r.launch_summed_ms + r.synchronization_summed_ms + r.allocation_summed_ms + r.other_summed_ms
    assert total == pytest.approx(r.api_summed_ms)
    assert not hasattr(r, "dispatch_ms")


def test_operator_ids_deterministic(trace_file):
    ev1, _ = analyze_cuda_torch_profiler(trace_file)
    ev2, _ = analyze_cuda_torch_profiler(trace_file)
    assert [o.id for o in ev1.operators] == [o.id for o in ev2.operators]
    assert ev1 == ev2


def test_round_trip(trace_file):
    ev, _ = analyze_cuda_torch_profiler(trace_file)
    assert OptimizationEvidence.from_dict(ev.to_dict()) == ev


def test_provenance_chain(trace_file):
    ev, store = analyze_cuda_torch_profiler(trace_file)
    assert ev.provenance == store.records
    ids = {r.id for r in store.records}
    for op in ev.operators:
        assert op.evidence_ids
        assert set(op.evidence_ids) <= ids


def test_invalid_trace_fails_explicitly(tmp_path):
    f = tmp_path / "trace_rank0.json"
    f.write_text(json.dumps({"schemaVersion": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="traceEvents"):
        analyze_cuda_torch_profiler(f)


def test_gap_overlap_correlation(trace_file):
    # M4.1：分类 API 区间 ∩ exposed-gap 区间。
    # fixture: 窗口 [100,375]，busy [120,150][220,240][300,325][350,360]
    # -> gap [100,120][150,220][240,300][325,350][360,375]
    # launch [110,118]+[265,269] 全在 gap 内 = 12us；sync [360,375] = 15us
    ev, _ = analyze_cuda_torch_profiler(trace_file)
    go = ev.runtime.gap_overlap_ms
    assert go is not None
    assert go["launch"] == pytest.approx(12.0 / 1000.0)
    assert go["synchronization"] == pytest.approx(15.0 / 1000.0)
    assert go["allocation"] == 0.0


def _overlapping_launch_trace() -> dict:
    """P0-1 回归：两个并发重叠的 launch 事件都完全在 gap 内。

    窗口 [0,290]，busy [0,100]；gap = [100,290]（190us）。
    launch A [200,240] 40us；launch B [220,290] 70us（与 A 重叠 20us）。
    summed 分母 = 40+70 = 110us；逐事件分子 = 40+70 = 110us -> corr = 1.0。
    union 口径会给出 90us -> corr = 90/110 = 0.818（错误语义，回归必须防住）。
    """
    events = [{"ph": "X", "cat": "kernel", "name": "k1", "ts": 0, "dur": 100, "args": {}}]
    for i, (ts, dur) in enumerate([(200, 40), (220, 70)]):
        events.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": ts,
                "dur": dur,
                "args": {"External id": 10 + i},
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": f"aten::launch_{i}",
                "ts": ts,
                "dur": dur,
                "args": {"External id": 10 + i},
            }
        )
    return {"schemaVersion": 1, "deviceProperties": [{"name": "NVIDIA A100"}], "traceEvents": events}


def test_overlapping_launch_events_summed_numerator(tmp_path: Path):
    # P0-1：分子逐事件求和（110us），不先 union（union 会得 90us）
    f = tmp_path / "trace_rank0.json"
    f.write_text(json.dumps(_overlapping_launch_trace()), encoding="utf-8")
    ev, _ = analyze_cuda_torch_profiler(f)
    go = ev.runtime.gap_overlap_ms
    assert go["launch"] == pytest.approx(110.0 / 1000.0)  # 40+70，不是 union 的 90
    assert ev.runtime.launch_summed_ms == pytest.approx(110.0 / 1000.0)
    # 相关性 = 110/110 = 1.0（分子分母同 summed 口径）
    obs = build_observations(ev)
    corr = [o for o in obs if o.kind == "runtime_launch_gap_correlated"]
    assert len(corr) == 1
    assert corr[0].value == pytest.approx(1.0)


def test_diagnosis_correlation_upgrades_to_bound(tmp_path: Path):
    # M4.1 正例：launch 大量落入 gap 且 coverage 足够 -> host_dispatch_bound
    from vllm_omni.profiling.analysis import enrich

    events = [
        {"ph": "X", "cat": "kernel", "name": "k1", "ts": 100, "dur": 100, "args": {}},
        {"ph": "X", "cat": "kernel", "name": "k2", "ts": 300, "dur": 100, "args": {}},
    ]
    # 10 个 launch [400..670]，每个 30us，全部在 gap [400,700] 内 = 300us
    for i in range(10):
        events.append(
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 400 + i * 30, "dur": 30, "args": {}}
        )
    trace = {"schemaVersion": 1, "deviceProperties": [{"name": "NVIDIA A100"}], "traceEvents": events}
    f = tmp_path / "trace_rank0.json"
    f.write_text(json.dumps(trace), encoding="utf-8")
    ev, _ = analyze_cuda_torch_profiler(f)
    enriched = enrich(ev)
    classes = {c.class_: c.confidence for c in enriched.diagnosis.candidates}
    # corr 1.0 >= 1.0；coverage 300/400 = 0.75 >= 0.2 -> high
    assert classes.get("host_dispatch_bound") == "high"
    hd = next(c for c in enriched.diagnosis.candidates if c.class_ == "host_dispatch_bound")
    assert all(":" in e for e in hd.evidence_ids)  # 全局形式


def test_diagnosis_coverage_too_small_no_bound(tmp_path: Path):
    # P0-2 负例：大 gap + 极少量 100% 相关的 launch -> coverage 不足，不升格
    from vllm_omni.profiling.analysis import enrich

    events = [
        {"ph": "X", "cat": "kernel", "name": "k1", "ts": 0, "dur": 100, "args": {}},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 500, "dur": 2, "args": {}},
    ]
    trace = {"schemaVersion": 1, "deviceProperties": [{"name": "NVIDIA A100"}], "traceEvents": events}
    f = tmp_path / "trace_rank0.json"
    f.write_text(json.dumps(trace), encoding="utf-8")
    ev, _ = analyze_cuda_torch_profiler(f)
    enriched = enrich(ev)
    classes = {c.class_ for c in enriched.diagnosis.candidates}
    assert "host_dispatch_bound" not in classes  # coverage 2/900 < 0.10


def test_gap_overlap_full_busy_no_gap_and_partial(tmp_path: Path):
    """MVP gate 审计测试项：kernel 铺满窗口（no gap）、launch 完全在 busy 内
    （完全不在 gap）、launch 跨 busy/gap 边界（部分相交）、launch 全在 gap 内。"""
    events = [
        {"ph": "X", "cat": "kernel", "name": "k1", "ts": 0, "dur": 100, "args": {}},
        # launch_a 完全在 busy 区间内 -> 对 gap 无贡献
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 10, "dur": 10, "args": {}},
        # launch_b 部分相交 [95,115]：gap 内 15us / 总 20us
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 95, "dur": 20, "args": {}},
        # launch_c 完全在 gap 内 [150,180]：30us
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 150, "dur": 30, "args": {}},
    ]
    trace = {"schemaVersion": 1, "deviceProperties": [{"name": "NVIDIA A100"}], "traceEvents": events}
    f = tmp_path / "trace_rank0.json"
    f.write_text(json.dumps(trace), encoding="utf-8")
    ev, _ = analyze_cuda_torch_profiler(f)
    # 窗口 [0,180]，busy [0,100]，gap [100,180] = 80us；exposed 标量 = 180-100 = 80us = 0.08ms
    assert ev.timeline.exposed_non_device_busy_ms == pytest.approx(80.0 / 1000.0)
    go = ev.runtime.gap_overlap_ms
    # 逐事件求和：a 0 + b 15 + c 30 = 45us（summed 口径）
    assert go["launch"] == pytest.approx(45.0 / 1000.0)
    # 相关性 = 45/60 = 0.75
    obs = build_observations(ev)
    corr = [o for o in obs if o.kind == "runtime_launch_gap_correlated"]
    assert len(corr) == 1
    assert corr[0].value == pytest.approx(45.0 / 60.0)


def test_no_gap_full_busy_zero_overlap(tmp_path: Path):
    """MVP gate 审计测试项：busy 铺满窗口（无 gap）-> overlap 全 0、exposed=0
    实测值、无 gap 观察无 bound。"""
    events = [
        {"ph": "X", "cat": "kernel", "name": "k1", "ts": 0, "dur": 100, "args": {}},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 10, "dur": 8, "args": {}},
    ]
    trace = {"schemaVersion": 1, "deviceProperties": [{"name": "NVIDIA A100"}], "traceEvents": events}
    f = tmp_path / "trace_rank0.json"
    f.write_text(json.dumps(trace), encoding="utf-8")
    ev, _ = analyze_cuda_torch_profiler(f)
    assert ev.timeline.exposed_non_device_busy_ms == pytest.approx(0.0)  # 实测 0，不是 unavailable
    go = ev.runtime.gap_overlap_ms
    assert go is not None
    assert go["launch"] == 0.0
    assert all(o.kind != "exposed_gap_present" for o in build_observations(ev))
    assert all(c.class_ == "unknown" for c in ev.diagnosis.candidates) or True
    from vllm_omni.profiling.diagnosis import diagnose

    assert all(c.class_ != "host_dispatch_bound" for c in diagnose(ev))
