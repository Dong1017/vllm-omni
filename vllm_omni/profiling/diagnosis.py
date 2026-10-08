# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# 确定性诊断（M3/M4.1，ckpt-3 修订 + MVP gate）：
# Evidence -> Observation -> DiagnosisCandidate，诊断只消费 Observation（P0-1）。
#
# 支持的 *_bound 判定（每条都必须"证据链完整"，缺一不判）：
#   1. communication_bound（M3）：
#      communication_exposed 观察（引用含 total+overlap+wall）
#      + 四输入机械同源 + 未重叠通信双阈值。
#   2. host_dispatch_bound / synchronization_bound / allocation_bound（M4.1 新增）：
#      exposed_gap_present 观察 + runtime_*_heavy 观察
#      + runtime_*_gap_correlated 观察（M4.1 时间相关性：
#        该分类 API 区间 ∩ exposed-gap 区间占其自身时长的比例 >= 50%）。
#      这满足了 ckpt-3 审计 P0-2 要求的"时间相关性证据"——
#      只有 summed 占比、没有 gap 区间交集时，仍然只输出观察。
#
# compute_bound 仍保留给 M4.2 硬件证据（NCU/DRAM/SM、AIC/AIV/MTE）。
# memory_bound / load_imbalance 无证据来源，规则缺席。
# 无候选时输出 unknown(low)——合法输出，不是失败。

from __future__ import annotations

from vllm_omni.profiling.observation import Observation, build_observations
from vllm_omni.profiling.schema import CONFIDENCE_LEVELS, DiagnosisCandidate, OptimizationEvidence

# 诊断阈值（观察阈值宽于诊断阈值；全部显式常量）
DIAGNOSE_COMM_EXPOSED_RATIO = 0.15  # 未重叠通信 >= 15% wall
DIAGNOSE_COMM_GAP_DOMINANCE = 0.50  # 未重叠通信 >= 50% exposed gap
DIAGNOSE_GAP_CORRELATION = 0.50  # 分类 API 时间 >= 50% 落在 exposed gap 内（相关性）
DIAGNOSE_GAP_COVERAGE = 0.10  # 该分类实际解释 >= 10% 的 exposed gap（影响条件，P0-2 显式定义）
STRONG_FACTOR = 2.0  # 相关性与覆盖同时达到阈值 2 倍 -> confidence high


def _same_source(ev: OptimizationEvidence, *keys: str) -> bool:
    """所有 metric 共享至少一个 evidence 记录 -> 同 source/time domain（机械校验）。"""
    sets = [set(ev.metric_evidence.get(k, [])) for k in keys]
    if any(not s for s in sets):
        return False
    return bool(set.intersection(*sets))


def _num(value: object) -> float:
    return value if isinstance(value, (int, float)) else 0.0


def _rule_communication(ev: OptimizationEvidence, obs: list[Observation]) -> list[DiagnosisCandidate]:
    # P0-1：只从 Observation 出发；没有 communication_exposed 观察就没有候选
    exposed = next((o for o in obs if o.kind == "communication_exposed"), None)
    if exposed is None or exposed.value is None:
        return []
    # M4.1c gate P0：denominator 优先 timeline.window_ms，unavailable 回退
    # workload.wall_ms；same-source 校验绑定实际使用的 denominator
    use_window = ev.timeline.window_ms is not None
    window_ms = ev.timeline.window_ms if use_window else ev.workload.wall_ms
    window_key = "timeline.window_ms" if use_window else "workload.wall_ms"
    # P0-4.2：同 source/time domain 机械校验（不依赖测试），四个输入缺一不判
    if not _same_source(
        ev,
        "communication.total_ms",
        "communication.overlap_ms",
        window_key,
        "timeline.exposed_non_device_busy_ms",
    ):
        return []
    total = ev.communication.total_ms
    overlap = ev.communication.overlap_ms
    wall = window_ms
    gap = ev.timeline.exposed_non_device_busy_ms
    # 显式逐项判 None（mypy 可 narrow；行为与 `None in (...)` 等价）
    if total is None or overlap is None or wall is None or gap is None or wall <= 0:
        return []
    not_overlapped = total - overlap
    exposed_ratio = not_overlapped / wall if wall > 0 else 0.0
    gap_dominance = not_overlapped / gap if gap > 0 else 0.0
    if exposed_ratio < DIAGNOSE_COMM_EXPOSED_RATIO or gap_dominance < DIAGNOSE_COMM_GAP_DOMINANCE:
        return []
    confidence = (
        "high"
        if min(exposed_ratio / DIAGNOSE_COMM_EXPOSED_RATIO, gap_dominance / DIAGNOSE_COMM_GAP_DOMINANCE)
        >= STRONG_FACTOR
        else "medium"
    )
    return [
        DiagnosisCandidate(
            class_="communication_bound",
            confidence=confidence,
            # 引用观察携带的全部输入引用（total+overlap+wall，P0-4.1）
            evidence_ids=sorted(set(exposed.evidence_ids)),
        )
    ]


def _rule_gap_correlated(
    ev: OptimizationEvidence,
    obs: list[Observation],
    bound_class: str,
    heavy_kind: str,
    corr_kind: str,
    cat: str,
) -> list[DiagnosisCandidate]:
    """M4.1：runtime 分类"重" + "时间上落入 exposed gap" + "实际解释可观比例的 gap"
    三条观察/条件齐备才升格 *_bound（P0-2：相关性与影响是两个独立条件）。"""
    gap = next((o for o in obs if o.kind == "exposed_gap_present"), None)
    heavy = next((o for o in obs if o.kind == heavy_kind), None)
    corr = next((o for o in obs if o.kind == corr_kind), None)
    # 证据链：三个观察缺一不判（corr 观察缺失 = 无时间轴或相关性不足）
    if gap is None or heavy is None or corr is None or corr.value is None:
        return []
    cat_short = cat.replace("_summed_ms", "")
    corr_key = f"runtime.gap_overlap_ms.{cat_short}"
    # 机械同源：分类时长与 gap_overlap 同 source；gap_overlap 与 exposed 同 source；
    # gap 与 wall 同 source
    if not _same_source(ev, f"runtime.{cat}", corr_key):
        return []
    if not _same_source(ev, corr_key, "timeline.exposed_non_device_busy_ms"):
        return []
    if not _same_source(ev, "timeline.exposed_non_device_busy_ms", "workload.wall_ms"):
        return []
    corr_ratio = _num(corr.value)
    if corr_ratio < DIAGNOSE_GAP_CORRELATION:
        return []
    # P0-2 影响条件：该分类落在 gap 内的时间必须实际解释可观比例的 exposed gap
    # （大 gap + 极少量 100% 相关的 API 时间 -> 不构成 *_bound）
    in_gap = (ev.runtime.gap_overlap_ms or {}).get(cat_short)
    gap_time = ev.timeline.exposed_non_device_busy_ms
    coverage = in_gap / gap_time if in_gap is not None and gap_time else 0.0
    if coverage < DIAGNOSE_GAP_COVERAGE:
        return []
    strong = (
        corr_ratio >= STRONG_FACTOR * DIAGNOSE_GAP_CORRELATION and coverage >= STRONG_FACTOR * DIAGNOSE_GAP_COVERAGE
    )
    confidence = "high" if strong else "medium"
    refs = sorted(set(gap.evidence_ids + heavy.evidence_ids + corr.evidence_ids))
    return [DiagnosisCandidate(class_=bound_class, confidence=confidence, evidence_ids=refs)]


def _rule_host_dispatch(ev: OptimizationEvidence, obs: list[Observation]) -> list[DiagnosisCandidate]:
    return _rule_gap_correlated(
        ev,
        obs,
        "host_dispatch_bound",
        "runtime_launch_heavy",
        "runtime_launch_gap_correlated",
        "launch_summed_ms",
    )


def _rule_synchronization(ev: OptimizationEvidence, obs: list[Observation]) -> list[DiagnosisCandidate]:
    return _rule_gap_correlated(
        ev,
        obs,
        "synchronization_bound",
        "runtime_sync_heavy",
        "runtime_sync_gap_correlated",
        "synchronization_summed_ms",
    )


def _rule_allocation(ev: OptimizationEvidence, obs: list[Observation]) -> list[DiagnosisCandidate]:
    return _rule_gap_correlated(
        ev,
        obs,
        "allocation_bound",
        "runtime_allocation_heavy",
        "runtime_allocation_gap_correlated",
        "allocation_summed_ms",
    )


_RULES = (_rule_communication, _rule_host_dispatch, _rule_synchronization, _rule_allocation)


def diagnose(ev: OptimizationEvidence) -> list[DiagnosisCandidate]:
    """消费 Observation 运行规则；无候选时输出 unknown(low)（合法输出）。"""
    obs = build_observations(ev)
    candidates: list[DiagnosisCandidate] = []
    seen: set[str] = set()
    for rule in _RULES:
        for cand in rule(ev, obs):
            if cand.class_ not in seen:
                candidates.append(cand)
                seen.add(cand.class_)
    if not candidates:
        fallback = next(
            (o for o in obs if o.kind in ("no_evidence_available", "no_salient_observation")),
            None,
        )
        refs = fallback.evidence_ids if fallback else sorted({r for refs in ev.metric_evidence.values() for r in refs})
        candidates.append(DiagnosisCandidate(class_="unknown", confidence="low", evidence_ids=refs))
    assert all(c.confidence in CONFIDENCE_LEVELS for c in candidates)
    return candidates
