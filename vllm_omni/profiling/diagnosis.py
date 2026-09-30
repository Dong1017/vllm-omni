# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# 确定性诊断（M3，ckpt-3 审计修订版）：
# Evidence -> Observation -> DiagnosisCandidate，诊断只消费 Observation（P0-1）。
#
# 当前唯一支持的 *_bound 判定是 communication_bound（P0-4）：
#   输入 = communication_exposed 观察（其 evidence 引用已含 total+overlap+wall 全部输入），
#   再机械校验 total/overlap/wall/exposed_gap 同 source/time domain，
#   且未重叠通信同时跨过 wall 占比与 gap 主导两条线。
#
# host/synchronization/allocation/compute 的"重"停留在 Observation 层
# （runtime_launch_heavy / runtime_sync_heavy / runtime_allocation_heavy /
#  compute_activity_dominant）；升格为 *_bound 需要 M4 时间相关性或硬件证据。
# memory_bound / load_imbalance 无证据来源，规则缺席。
# 无候选时输出 unknown(low)——这是合法输出，不是失败。

from __future__ import annotations

from vllm_omni.profiling.observation import Observation, build_observations
from vllm_omni.profiling.schema import CONFIDENCE_LEVELS, DiagnosisCandidate, OptimizationEvidence

# 诊断阈值（观察阈值宽于诊断阈值；全部显式常量）
DIAGNOSE_COMM_EXPOSED_RATIO = 0.15  # 未重叠通信 >= 15% wall
DIAGNOSE_COMM_GAP_DOMINANCE = 0.50  # 未重叠通信 >= 50% exposed gap
STRONG_FACTOR = 2.0  # 达到阈值 2 倍 -> confidence high


def _same_source(ev: OptimizationEvidence, *keys: str) -> bool:
    """所有 metric 共享至少一个 evidence 记录 -> 同 source/time domain（P0-4.2 机械校验）。"""
    sets = [set(ev.metric_evidence.get(k, [])) for k in keys]
    if any(not s for s in sets):
        return False
    return bool(set.intersection(*sets))


def _rule_communication(ev: OptimizationEvidence, obs: list[Observation]) -> list[DiagnosisCandidate]:
    # P0-1：只从 Observation 出发；没有 communication_exposed 观察就没有候选
    exposed = next((o for o in obs if o.kind == "communication_exposed"), None)
    if exposed is None or exposed.value is None:
        return []
    # P0-4.2：同 source/time domain 机械校验（不依赖测试），四个输入缺一不判
    if not _same_source(
        ev,
        "communication.total_ms",
        "communication.overlap_ms",
        "workload.wall_ms",
        "timeline.exposed_non_device_busy_ms",
    ):
        return []
    total = ev.communication.total_ms
    overlap = ev.communication.overlap_ms
    wall = ev.workload.wall_ms
    gap = ev.timeline.exposed_non_device_busy_ms
    if None in (total, overlap, wall, gap) or wall <= 0:
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


_RULES = (_rule_communication,)


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
