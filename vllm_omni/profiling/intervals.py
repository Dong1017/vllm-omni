# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# 区间运算：device busy 必须用区间 union（重叠只计一次），
# 与 summed duration 严格区分（AC-06 / 禁止把 summed kernel time 当 wall）。

from __future__ import annotations

# 区间一律 (start_us, end_us)，end > start，零时长事件跳过
Interval = tuple[float, float]


def union(intervals: list[Interval]) -> list[Interval]:
    """合并重叠/相接区间，返回按 start 排序的不相交区间列表。"""
    if not intervals:
        return []
    ordered = sorted((i for i in intervals if i[1] > i[0]), key=lambda x: (x[0], x[1]))
    merged: list[Interval] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            last = merged[-1]
            merged[-1] = (last[0], max(last[1], end))
        else:
            merged.append((start, end))
    return merged


def total(intervals: list[Interval]) -> float:
    """union 后的总时长（us），不是 summed duration。"""
    return sum(end - start for start, end in union(intervals))


def subtract(base: list[Interval], removal: list[Interval]) -> list[Interval]:
    """从 base 区间集合中减去 removal 区间集合（集合语义，先各自 union）。"""
    result = union(base)
    for start, end in union(removal):
        nxt: list[Interval] = []
        for b_start, b_end in result:
            if end <= b_start or start >= b_end:
                nxt.append((b_start, b_end))
                continue
            if start > b_start:
                nxt.append((b_start, min(start, b_end)))
            if end < b_end:
                nxt.append((max(end, b_start), b_end))
        result = [iv for iv in nxt if iv[1] > iv[0]]
    return result


def intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    """两个区间集合的交集（M4.1：API 区间 ∩ exposed-gap 区间）。"""
    ua, ub = union(a), union(b)
    out: list[Interval] = []
    i = j = 0
    while i < len(ua) and j < len(ub):
        start = max(ua[i][0], ub[j][0])
        end = min(ua[i][1], ub[j][1])
        if end > start:
            out.append((start, end))
        if ua[i][1] < ub[j][1]:
            i += 1
        else:
            j += 1
    return out


def total_event_overlap(events: list[Interval], against: list[Interval]) -> float:
    """逐事件求与 against（将 union）的交集时长再求和（summed 口径，P0-1）。

    与 total(intersect(events, against)) 的区别：events 不先 union——
    并发重叠的 runtime 事件各自与 gap 求交后相加，与分母
    `*_summed_ms`（逐事件时长求和）保持同一时间语义。
    """
    ag = union(against)
    return sum(total(intersect([ev], ag)) for ev in events)
