# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# 时间单位规范化（T03）。内部口径统一为 us，报告层输出 ms。
# 禁止在 units 之外做裸数字时间换算（AC-04：禁止无单位数字）。

from __future__ import annotations

from enum import Enum


class TimeUnit(Enum):
    NS = 1
    US = 1000
    MS = 1000 * 1000


def convert_time(value: float, src: TimeUnit, dst: TimeUnit) -> float:
    # value 不接受 None：调用方必须先处理 unavailable，再换算
    if value is None:
        raise ValueError("convert_time: value is None (unavailable must not be converted)")
    # None 也可能被传入 float 注解掩盖不了的场景，例如 NaN
    if value != value:
        raise ValueError("convert_time: value is NaN")
    return value * (src.value / dst.value)


def us_to_ms(value_us: float) -> float:
    return convert_time(value_us, TimeUnit.US, TimeUnit.MS)


def ms_to_us(value_ms: float) -> float:
    return convert_time(value_ms, TimeUnit.MS, TimeUnit.US)
