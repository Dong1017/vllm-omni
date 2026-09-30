# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# 分析编排（M3）：hotspot 排序（T32）+ Observation + diagnosis 回填 evidence。
# 机械规则（P0-3/D6）：hotspot 排序与 fraction 只允许同语义比较——
# framework 层只用 device_self_ms；runtime/device 层用 summed_*；
# inclusive 值只作展示，永不进入排序或 fraction。

from __future__ import annotations

from vllm_omni.profiling.diagnosis import diagnose
from vllm_omni.profiling.observation import build_observations
from vllm_omni.profiling.schema import Hotspot, OptimizationEvidence

_HOTSPOT_TOP = 5

_DENOM_FRAMEWORK_SELF = "summed_device_self_ms_of_all_framework_ops"
_DENOM_RUNTIME_HOST = "summed_host_ms_of_all_runtime_rows"
_DENOM_DEVICE = "summed_device_ms_of_all_device_rows"
_DENOM_WALL = "workload.wall_ms_same_source"


def _global_refs(ev: OptimizationEvidence, ids: list[str]) -> list[str]:
    """行级本地 evidence id 统一为全局形式 run_id:ev_id（P1，ckpt-MVP gate）。"""
    if not ev.run.run_id:
        return sorted(set(ids))
    return sorted({eid if ":" in eid else f"{ev.run.run_id}:{eid}" for eid in ids})


def rank_hotspots(ev: OptimizationEvidence) -> list[Hotspot]:
    """按层做同语义 TopK；不同层不混合排序。"""
    hotspots: list[Hotspot] = []

    fw = [o for o in ev.operators if o.layer == "framework" and o.device_self_ms is not None]
    fw_total = sum(o.device_self_ms for o in fw)
    fw.sort(key=lambda o: -o.device_self_ms)
    for op in fw[:_HOTSPOT_TOP]:
        if fw_total > 0:
            hotspots.append(
                Hotspot(
                    name=op.name,
                    kind="operator",
                    measured_fraction=op.device_self_ms / fw_total,
                    fraction_denominator=_DENOM_FRAMEWORK_SELF,
                    addressable_fraction=None,
                    evidence_ids=_global_refs(ev, op.evidence_ids),
                )
            )

    rt = [o for o in ev.operators if o.layer == "runtime" and o.summed_host_ms is not None]
    rt_total = sum(o.summed_host_ms for o in rt)
    rt.sort(key=lambda o: -o.summed_host_ms)
    for op in rt[:_HOTSPOT_TOP]:
        if rt_total > 0:
            hotspots.append(
                Hotspot(
                    name=op.name,
                    kind="runtime_api",
                    measured_fraction=op.summed_host_ms / rt_total,
                    fraction_denominator=_DENOM_RUNTIME_HOST,
                    addressable_fraction=None,
                    evidence_ids=_global_refs(ev, op.evidence_ids),
                )
            )

    dev = [o for o in ev.operators if o.layer == "device" and o.summed_device_ms is not None]
    dev_total = sum(o.summed_device_ms for o in dev)
    dev.sort(key=lambda o: -o.summed_device_ms)
    for op in dev[:_HOTSPOT_TOP]:
        if dev_total > 0:
            hotspots.append(
                Hotspot(
                    name=op.name,
                    kind="device_task",
                    measured_fraction=op.summed_device_ms / dev_total,
                    fraction_denominator=_DENOM_DEVICE,
                    addressable_fraction=None,
                    evidence_ids=_global_refs(ev, op.evidence_ids),
                )
            )

    # communication hotspot：communication 与 wall 同 source 才计算（P1-1 交集判定）
    comm = ev.communication.total_ms
    wall = ev.workload.wall_ms
    if comm is not None and wall is not None and wall > 0:
        comm_refs = set(ev.metric_evidence.get("communication.total_ms", []))
        wall_refs = set(ev.metric_evidence.get("workload.wall_ms", []))
        if comm_refs & wall_refs:
            hotspots.append(
                Hotspot(
                    name="communication_total",
                    kind="communication",
                    measured_fraction=comm / wall,
                    fraction_denominator=_DENOM_WALL,
                    addressable_fraction=None,
                    evidence_ids=_global_refs(ev, sorted(comm_refs | wall_refs)),
                )
            )
    return hotspots


def enrich(ev: OptimizationEvidence) -> OptimizationEvidence:
    """在 adapter 产出的 evidence 上回填 hotspots 与 diagnosis candidates（原地更新）。"""
    ev.hotspots = rank_hotspots(ev)
    ev.diagnosis.candidates = diagnose(ev)
    return ev


def build_all_observations(ev: OptimizationEvidence) -> list:
    """暴露 Observation 列表（query/debug 用；observations 不入 evidence schema）。"""
    return build_observations(ev)
