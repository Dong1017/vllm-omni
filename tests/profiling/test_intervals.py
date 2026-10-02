# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# 区间运算测试：union 语义与 summed 严格区分（AC-06）

import pytest

from vllm_omni.profiling.intervals import intersect, subtract, total, union

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_union_overlapping():
    assert union([(0, 10), (5, 15)]) == [(0, 15)]


def test_union_adjacent_merges():
    assert union([(0, 10), (10, 20)]) == [(0, 20)]


def test_union_disjoint_keeps_separate():
    assert union([(0, 5), (7, 9)]) == [(0, 5), (7, 9)]


def test_union_nested():
    assert union([(0, 100), (10, 20), (30, 40)]) == [(0, 100)]


def test_union_unsorted_input():
    assert union([(20, 30), (0, 10)]) == [(0, 10), (20, 30)]


def test_union_empty_and_zero_duration():
    assert union([]) == []
    assert union([(5, 5)]) == []  # 零时长事件不占用设备


def test_total_is_union_not_sum():
    # 重叠 5us 只计一次；summed 会给出 15，union 语义应为 10
    assert total([(0, 5), (0, 10)]) == pytest.approx(10.0)


def test_subtract_basic():
    assert subtract([(0, 10)], [(3, 5)]) == [(0, 3), (5, 10)]


def test_subtract_full_coverage():
    assert subtract([(0, 10)], [(0, 10)]) == []


def test_subtract_disjoint_removal():
    assert subtract([(0, 10), (20, 30)], [(20, 25)]) == [(0, 10), (25, 30)]


def test_subtract_removal_outside_base():
    assert subtract([(0, 10)], [(15, 20)]) == [(0, 10)]


def test_intersect_partial_overlap():
    assert intersect([(0, 10)], [(5, 15)]) == [(5, 10)]


def test_intersect_multiple_pairs():
    a = [(0, 10), (20, 30)]
    b = [(5, 25)]
    assert intersect(a, b) == [(5, 10), (20, 25)]


def test_intersect_disjoint_empty():
    assert intersect([(0, 5)], [(10, 20)]) == []


def test_intersect_nested():
    assert intersect([(0, 100)], [(10, 20), (30, 40)]) == [(10, 20), (30, 40)]
