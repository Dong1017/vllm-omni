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
