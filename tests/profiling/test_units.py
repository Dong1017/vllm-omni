# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# 单位换算测试（AC-04）：ns <-> us <-> ms 往返一致，None/NaN 显式拒绝

import math

import pytest

from vllm_omni.profiling.units import TimeUnit, convert_time, ms_to_us, us_to_ms

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("value", "src", "dst", "expected"),
    [
        (1.0, TimeUnit.MS, TimeUnit.US, 1000.0),
        (1.0, TimeUnit.US, TimeUnit.MS, 0.001),
        (1.0, TimeUnit.US, TimeUnit.NS, 1000.0),
        (1.0, TimeUnit.NS, TimeUnit.US, 0.001),
        (1.0, TimeUnit.MS, TimeUnit.NS, 1000000.0),
        (1234.0, TimeUnit.NS, TimeUnit.MS, 0.001234),
    ],
)
def test_convert_time_table(value, src, dst, expected):
    assert convert_time(value, src, dst) == pytest.approx(expected)


@pytest.mark.parametrize("unit", list(TimeUnit))
def test_round_trip_all_pairs(unit):
    value = 42.5
    for other in TimeUnit:
        converted = convert_time(value, unit, other)
        assert convert_time(converted, other, unit) == pytest.approx(value)


def test_us_to_ms_and_back():
    assert us_to_ms(1500.0) == pytest.approx(1.5)
    assert ms_to_us(1.5) == pytest.approx(1500.0)


def test_convert_time_rejects_none():
    with pytest.raises(ValueError, match="unavailable"):
        convert_time(None, TimeUnit.US, TimeUnit.MS)


def test_convert_time_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        convert_time(math.nan, TimeUnit.US, TimeUnit.MS)
