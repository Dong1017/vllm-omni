# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Golden facts 回归契约（M2 起）：真实 artifact 存在时，框架输出必须复现
# fixtures/ascend_ref2va_golden.json 中全部人工确认事实。
# 真实 artifact 不入库（59MB+），CI 上自动跳过；本地开发机执行真实验证。

import json
from pathlib import Path

import pytest

from vllm_omni.profiling.backends import analyze_ascend_csv

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

GOLDEN = Path(__file__).parent / "fixtures" / "ascend_ref2va_golden.json"
# 真实样本路径（golden.json 的 source 字段）；不存在则跳过
ARTIFACT_ROOT = Path("D:/world_model/outputs/h3_16die_validation_20260922/ref2va_15s_280ts/ref2va_15s_280ts")

requires_real_artifact = pytest.mark.skipif(
    not (ARTIFACT_ROOT / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv").exists(),
    reason="real Ascend artifact not available on this machine (golden facts stay prepared_not_run in CI)",
)


@requires_real_artifact
def test_ascend_golden_facts_reproduced():
    facts = json.loads(GOLDEN.read_text(encoding="utf-8"))["golden_facts"]
    ev, _ = analyze_ascend_csv(ARTIFACT_ROOT / "ASCEND_PROFILER_OUTPUT")
    actual = {
        "workload.wall_ms": ev.workload.wall_ms,
        "timeline.communication_ms": ev.timeline.communication_ms,
        "timeline.exposed_non_device_busy_ms": ev.timeline.exposed_non_device_busy_ms,
        "timeline.device_busy_ms": ev.timeline.device_busy_ms,
        "runtime.api_summed_ms": ev.runtime.api_summed_ms,
        "runtime.launch_summed_ms": ev.runtime.launch_summed_ms,
        "runtime.synchronization_summed_ms": ev.runtime.synchronization_summed_ms,
        "runtime.allocation_summed_ms": ev.runtime.allocation_summed_ms,
        "runtime.other_summed_ms": ev.runtime.other_summed_ms,
        "runtime.launch_count": ev.runtime.launch_count,
        "run.rank_is_none_and_device_id_8": (
            f"rank={ev.run.rank}, device_id={ev.run.device_id}"
            if ev.run.rank is None and ev.run.device_id is not None
            else "unexpected"
        ),
        "backend_metrics.ascend.op_statistic.rows": len(ev.backend_metrics.ascend.get("op_statistic", [])),
    }
    dev_rows = [o for o in ev.operators if o.layer == "device"]
    top_dev = max(dev_rows, key=lambda o: o.summed_device_ms or 0.0)
    actual["top_device_task.name"] = top_dev.name
    actual["top_device_task.calls"] = top_dev.calls
    actual["top_device_task.summed_device_ms"] = top_dev.summed_device_ms
    fw_rows = [o for o in ev.operators if o.layer == "framework"]
    top_fw = max(fw_rows, key=lambda o: o.device_inclusive_ms or 0.0)
    actual["top_framework_op.device_inclusive_ms"] = top_fw.device_inclusive_ms
    actual["runtime.api_summed_ms"] = ev.runtime.api_summed_ms

    for fact in facts:
        if fact["metric"].startswith("analysis_db."):
            continue  # DB 侧事实由 test_ascend_db.py::test_real_analysis_db_validation 覆盖
        expected = fact["value"]
        got = actual[fact["metric"]]
        if isinstance(expected, float):
            assert got == pytest.approx(expected, rel=1e-9), f"{fact['id']} {fact['metric']}: {got} != {expected}"
        else:
            assert got == expected, f"{fact['id']} {fact['metric']}: {got!r} != {expected!r}"
