"""Exact rational time arithmetic.

设计约束（见 README "设计决策"）：

* 内核内部从不使用 ``float`` 累加时间，所有真实时间都是
  :class:`fractions.Fraction` 秒，帧率 30000/1001 也不会漂移；
* 换到输出时间基准（整数 tick）时才发生第一次量化，量化规则与
  FFmpeg ``av_rescale_q`` 的默认舍入（四舍五入、等距远离 0）一致，
  这样内核产出的 tick 与真正的 PyAV/FFmpeg 封装器逐值对齐；
* 量化误差以精确 :class:`~fractions.Fraction` 秒随每个包公开，
  绝不把无法精确表示的时间伪装成原值；
* 每次换算都只依赖该包自己的输入，多段接续不存在"累加误差"，
  可公开检查的单包残差之和即是总误差上界。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from fractions import Fraction
from typing import Union

#: 所有时间基准参数都接受 Fraction 或 (num, den) 二元组。
TimeBaseLike = Union[Fraction, "tuple[int, int]"]


def as_fraction(value: TimeBaseLike) -> Fraction:
    """Normalize a time base to :class:`~fractions.Fraction`."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return Fraction(value[0], value[1])
    raise TypeError(f"time base must be Fraction or (num, den), got {value!r}")


class Rounding(enum.Enum):
    """Rounding modes for mapping a rational time to integer output ticks.

    The default :attr:`NEAREST_AWAY_FROM_ZERO` matches FFmpeg's
    ``AV_ROUND_NEAR_INF`` (the default of ``av_rescale_q``): round to the
    nearest integer; ties go away from zero.
    """

    NEAREST_AWAY_FROM_ZERO = enum.auto()
    DOWN = enum.auto()   # towards negative infinity
    UP = enum.auto()     # towards positive infinity
    TOWARD_ZERO = enum.auto()


def _round(x: Fraction, rounding: Rounding) -> int:
    if rounding is Rounding.NEAREST_AWAY_FROM_ZERO:
        # floor(x + 1/2) for x >= 0, ceil(x - 1/2) for x < 0
        if x >= 0:
            return (x.numerator * 2 + x.denominator) // (2 * x.denominator)
        return -(((-x.numerator) * 2 + x.denominator) // (2 * x.denominator))
    if rounding is Rounding.DOWN:
        return x.numerator // x.denominator
    if rounding is Rounding.UP:
        return -((-x.numerator) // x.denominator)
    if rounding is Rounding.TOWARD_ZERO:
        # 向 0 截断；不能用 divmod（Python 对负数是向负无穷取整）
        q, r = divmod(abs(x.numerator), x.denominator)
        return q if x.numerator >= 0 else -q
    raise ValueError(f"unknown rounding mode: {rounding!r}")  # pragma: no cover


@dataclass(frozen=True)
class TickResult:
    """Result of converting an exact time to integer output ticks.

    Attributes:
        ticks: 量化后的整数 tick（可以是负数，B 帧 DTS 平移前就是负的）。
        exact: 量化结果对应的精确秒数。
        residual: ``ticks * output_time_base - 原精确秒数``，
            即表示误差，精确有理值，单位秒。
    """

    ticks: int
    exact: Fraction
    residual: Fraction


def rescale_ticks(
    seconds: Fraction,
    output_time_base: TimeBaseLike,
    rounding: Rounding = Rounding.NEAREST_AWAY_FROM_ZERO,
) -> TickResult:
    """Convert an exact time in seconds to integer ticks in *output_time_base*.

    >>> tr = rescale_ticks(Fraction(-2, 25), Fraction(1, 12800))
    >>> tr.ticks
    -1024
    >>> tr.residual
    0
    """
    tb = as_fraction(output_time_base)
    exact_index = seconds / tb
    ticks = _round(exact_index, rounding)
    quantized = ticks * tb
    return TickResult(ticks=ticks, exact=quantized, residual=quantized - seconds)


@dataclass(frozen=True)
class AffineMapping:
    """One field's full timing relationship.

    保留原始值与修正值之间每一步可检查的关系：

    ``output_ticks * output_time_base``
        = 修正后的精确显示/解码时间；
    ``original_seconds`` = 原始整数 tick * 原始时间基准；
    ``residual`` = 修正精确时间 - (原始精确时间 + 输出起点)；
        只可能来自输出时间基准的量化误差，多段接续不会产生额外漂移。
    """

    original_ticks: int | None
    original_time_base: Fraction | None
    original_seconds: Fraction | None
    mapped_seconds: Fraction | None
    """原始精确时间 + 分段输出起点（若有）。"""
    output_ticks: int | None
    output_time_base: Fraction
    residual: Fraction | None
    convertible: bool


def map_field(
    original_ticks: int | None,
    original_time_base: TimeBaseLike | None,
    output_start: Fraction,
    output_time_base: TimeBaseLike,
    rounding: Rounding = Rounding.NEAREST_AWAY_FROM_ZERO,
) -> AffineMapping:
    """Map one PTS/DTS-like field through the segment affine transform.

    ``修正秒 = output_start + 原始 tick * 原始时间基准``。
    任一侧为 None（时间字段缺失 / 无法换算）时，输出 ticks 为 None，
    对应 :class:`~media_clock_core.events.TimeStatus` 中的不可换算状态，
    但不影响载荷交付。
    """
    out_tb = as_fraction(output_time_base)
    if original_ticks is None or original_time_base is None:
        return AffineMapping(
            original_ticks=original_ticks,
            original_time_base=(
                as_fraction(original_time_base)
                if original_time_base is not None
                else None
            ),
            original_seconds=None,
            mapped_seconds=None,
            output_ticks=None,
            output_time_base=out_tb,
            residual=None,
            convertible=False,
        )
    in_tb = as_fraction(original_time_base)
    original_seconds = Fraction(original_ticks) * in_tb
    mapped = output_start + original_seconds
    tr = rescale_ticks(mapped, out_tb, rounding)
    return AffineMapping(
        original_ticks=original_ticks,
        original_time_base=in_tb,
        original_seconds=original_seconds,
        mapped_seconds=mapped,
        output_ticks=tr.ticks,
        output_time_base=out_tb,
        residual=tr.residual,
        convertible=True,
    )
