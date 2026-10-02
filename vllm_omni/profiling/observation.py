# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Observation 层（M3）：从 evidence 确定性提取"发生了什么"。
# 每个 Observation 绑定 exact metric key + 全局 evidence 引用（P1-1）。
# 只描述、不解释；时间相关性证据落地（M4）之前，
# launch/synchronization/allocation/compute 的"重"只允许停留在观察层（ckpt-3 审计 P0-2/P0-3）。

from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.profiling.schema import OptimizationEvidence

# Observation 的固定类别（what is happening 的词汇表，与 diagnosis taxonomy 分离，P1-3）
OBSERVATION_KINDS = (
    "exposed_gap_present",  # 设备暴露空闲占比可观
    "runtime_launch_heavy",  # launch API 占 runtime 时间可观（观察，非 *_bound）
    "runtime_sync_heavy",  # synchronize API 占比可观
    "runtime_allocation_heavy",  # malloc/free API 占比可观
    "runtime_other_heavy",  # 其余 API 占比可观（含无法归类的 bucket）
    "runtime_launch_gap_correlated",  # M4.1：launch API 时间与 exposed gap 时间相关
    "runtime_sync_gap_correlated",
    "runtime_allocation_gap_correlated",
    "communication_exposed",  # 未重叠通信占比可观（derived，引用全部输入）
    "communication_present_no_overlap_breakdown",  # 通信总量存在但分解不可得（P1-2：不得称 exposed）
    "compute_activity_dominant",  # 设备上 compute 类活动占主导（观察，非 compute_bound）
    "no_evidence_available",  # 核心指标全缺（P1-1 拆分）
    "no_salient_observation",  # 有指标但无一超过观察阈值（P1-1 拆分）
)

# 观察阈值（宽于诊断阈值：观察负责"看见"，诊断负责"定罪"）
OBSERVE_GAP_RATIO = 0.10
OBSERVE_CATEGORY_SHARE = 0.20
OBSERVE_BUSY_RATIO = 0.75
OBSERVE_COMPUTE_SHARE = 0.75
OBSERVE_GAP_CORRELATION = 0.30  # 分类 API 时间落在 exposed gap 内的占比（M4.1）

_RUNTIME_CATEGORIES = (
    ("launch_summed_ms", "runtime_launch_heavy", "runtime_launch_gap_correlated"),
    ("synchronization_summed_ms", "runtime_sync_heavy", "runtime_sync_gap_correlated"),
    ("allocation_summed_ms", "runtime_allocation_heavy", "runtime_allocation_gap_correlated"),
    ("other_summed_ms", "runtime_other_heavy", None),
)


@dataclass
class Observation:
    id: str
    kind: str  # OBSERVATION_KINDS 之一
    metric_key: str  # schema 内 metric 路径（P1-1 约定）
    value: float | str | None
    note: str
    evidence_ids: list[str]  # 全局引用形式 "run_id:ev_id"（derived 值必须引用全部输入，P0-4.1）


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return num / den


def _same_source(ev: OptimizationEvidence, *keys: str) -> bool:
    """所有 metric 共享至少一个 evidence 记录 -> 同 source/time domain。"""
    sets = [set(ev.metric_evidence.get(k, [])) for k in keys]
    if any(not s for s in sets):
        return False
    return bool(set.intersection(*sets))


def build_observations(ev: OptimizationEvidence) -> list[Observation]:
    """确定性 Observation 提取；顺序固定，id 连续（AC-08）。"""
    observations: list[Observation] = []

    def add(kind: str, metric_key: str, value: float | str | None, note: str, refs: list[str]) -> None:
        observations.append(
            Observation(
                id=f"obs_{len(observations) + 1:03d}",
                kind=kind,
                metric_key=metric_key,
                value=value,
                note=note,
                evidence_ids=sorted(set(refs)),
            )
        )

    core_ref_count = len(ev.metric_evidence)
    # 同 source/time domain 机械校验（MVP gate 审计 Fix 1/2）：
    # 任何跨 metric 的比值（gap/wall、busy/wall、(total-overlap)/wall），
    # 输入不同源时一律不计算，宁可缺少观察也不伪造口径。
    gap_wall_same = _same_source(ev, "timeline.exposed_non_device_busy_ms", "workload.wall_ms")
    busy_wall_same = _same_source(ev, "timeline.device_busy_ms", "workload.wall_ms")
    comm_same = _same_source(ev, "communication.total_ms", "communication.overlap_ms", "workload.wall_ms")
    gap_ratio = _ratio(ev.timeline.exposed_non_device_busy_ms, ev.workload.wall_ms) if gap_wall_same else None
    api = ev.runtime.api_summed_ms
    comm_total = ev.communication.total_ms
    busy_ratio = _ratio(ev.timeline.device_busy_ms, ev.workload.wall_ms) if busy_wall_same else None

    if gap_ratio is not None and gap_ratio >= OBSERVE_GAP_RATIO:
        add(
            "exposed_gap_present",
            "timeline.exposed_non_device_busy_ms",
            gap_ratio,
            f"exposed non-device-busy gap is {gap_ratio:.1%} of wall",
            ev.metric_evidence.get("timeline.exposed_non_device_busy_ms", [])
            + ev.metric_evidence.get("workload.wall_ms", []),
        )

    if api is not None and api > 0:
        for cat, kind, corr_kind in _RUNTIME_CATEGORIES:
            share = _ratio(getattr(ev.runtime, cat), api)
            if share is not None and share >= OBSERVE_CATEGORY_SHARE:
                add(
                    kind,
                    f"runtime.{cat}",
                    share,
                    f"{cat} is {share:.1%} of runtime API summed time "
                    "(observation only; causal link to the exposed gap requires "
                    "temporal correlation evidence, e.g. M4 timeline tools)",
                    ev.metric_evidence.get(f"runtime.{cat}", []) + ev.metric_evidence.get("runtime.api_summed_ms", []),
                )
            # M4.1 时间相关性：该分类 API 时间有多大比例真的落在 exposed gap 区间内。
            # P1-1：发出观察前机械校验同源——分类时长与 gap_overlap 必须可追溯到
            # 同一记录，且 gap/wall 同源；校验不过则观察不出（代码即纪律）。
            cat_short = cat.replace("_summed_ms", "")
            corr_key = f"runtime.gap_overlap_ms.{cat_short}"
            in_gap = (ev.runtime.gap_overlap_ms or {}).get(cat_short)
            corr_ratio = _ratio(in_gap, getattr(ev.runtime, cat))
            if (
                corr_kind is not None
                and corr_ratio is not None
                and share is not None
                and share >= OBSERVE_CATEGORY_SHARE
                and corr_ratio >= OBSERVE_GAP_CORRELATION
                and _same_source(ev, f"runtime.{cat}", corr_key)
                and _same_source(ev, "timeline.exposed_non_device_busy_ms", "workload.wall_ms")
            ):
                add(
                    corr_kind,
                    corr_key,
                    corr_ratio,
                    f"{corr_ratio:.1%} of {cat} falls inside the device exposed-gap intervals "
                    "(temporal correlation from same-source timeline; "
                    "evidence to upgrade the heavy observation)",
                    ev.metric_evidence.get(corr_key, []) + ev.metric_evidence.get(f"runtime.{cat}", []),
                )

    if comm_total is not None:
        # P0-4.2 + Fix 2：total-overlap 与 /wall 必须先过同源校验；
        # 分解缺失或来源不兼容都不得称 exposed
        if ev.communication.overlap_ms is not None and comm_same:
            not_overlapped = comm_total - ev.communication.overlap_ms
            no_ratio = _ratio(not_overlapped, ev.workload.wall_ms)
            if no_ratio is not None and no_ratio >= OBSERVE_GAP_RATIO:
                # P0-4.1：derived 值必须引用全部输入（total + overlap + wall）
                add(
                    "communication_exposed",
                    "communication.total_ms",
                    no_ratio,
                    f"not-overlapped communication is {no_ratio:.1%} of wall (derived: total - overlap, same source)",
                    ev.metric_evidence.get("communication.total_ms", [])
                    + ev.metric_evidence.get("communication.overlap_ms", [])
                    + ev.metric_evidence.get("workload.wall_ms", []),
                )
        else:
            reason = (
                "overlap decomposition unavailable"
                if ev.communication.overlap_ms is None
                else "overlap present but not same-source with total/wall"
            )
            add(
                "communication_present_no_overlap_breakdown",
                "communication.total_ms",
                None,
                f"communication total present but {reason}; "
                "exposed-communication share cannot be computed without guessing",
                ev.metric_evidence.get("communication.total_ms", []),
            )

    if busy_ratio is not None and busy_ratio >= OBSERVE_BUSY_RATIO:
        compute_ratio = _ratio(ev.timeline.compute_ms, ev.timeline.device_busy_ms)
        # compute 活动占主导的观察：device_busy 与 compute 必须同 source（机械判定）
        if (
            compute_ratio is not None
            and compute_ratio >= OBSERVE_COMPUTE_SHARE
            and _same_source(ev, "timeline.device_busy_ms", "timeline.compute_ms")
        ):
            add(
                "compute_activity_dominant",
                "timeline.compute_ms",
                compute_ratio,
                f"compute-classified device activity is {compute_ratio:.1%} of device busy "
                "(observation only; hardware-throughput bound classification requires "
                "M4 hardware evidence, e.g. NCU/DRAM/SM or AIC/AIV/MTE metrics)",
                ev.metric_evidence.get("timeline.compute_ms", [])
                + ev.metric_evidence.get("timeline.device_busy_ms", []),
            )
        else:
            add(
                "device_busy_dominant",
                "timeline.device_busy_ms",
                busy_ratio,
                f"device busy (union) is {busy_ratio:.1%} of wall",
                ev.metric_evidence.get("timeline.device_busy_ms", []) + ev.metric_evidence.get("workload.wall_ms", []),
            )

    if not observations:
        # P1-1：区分"无证据"与"有证据但无显著信号"
        if core_ref_count == 0:
            add(
                "no_evidence_available",
                "workload.wall_ms",
                ev.workload.wall_ms,
                "no core metric binding available (wall/busy/runtime/communication all absent); "
                "no diagnosis is possible without guessing",
                [],
            )
        else:
            add(
                "no_salient_observation",
                "workload.wall_ms",
                ev.workload.wall_ms,
                "core metrics exist but none crosses observation thresholds; no diagnosis candidate is justified",
                sorted({r for refs in ev.metric_evidence.values() for r in refs}),
            )
    return observations
