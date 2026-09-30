"""Exact rational arithmetic tests — no float drift, residuals must be honest."""

from fractions import Fraction

import pytest

from media_clock_core import (
    Rounding,
    rescale_ticks,
    as_fraction,
)
from media_clock_core.timebase import map_field


def test_negative_dts_maps_to_mp4_ticks_exactly():
    # 用户场景：1/25 的 -2 在 1/12800 下必须精确得到 -1024
    r = rescale_ticks(Fraction(-2, 25), Fraction(1, 12800))
    assert r.ticks == -1024
    assert r.residual == 0
    r0 = rescale_ticks(Fraction(0, 25), Fraction(1, 12800))
    assert r0.ticks == 0


def test_25fps_12800_series():
    # 0,1,2,... 帧的精确映射：512 tick 间隔
    ticks = [
        rescale_ticks(Fraction(i, 25), Fraction(1, 12800)).ticks
        for i in range(-2, 10)
    ]
    assert ticks == [-1024 + 512 * i for i in range(12)]


def test_30000_1001_frame_rate_has_zero_residual_per_packet():
    # 30000/1001 fps 帧间隔 = 1001/30000 秒。输出 tb 1/90000（MPEG-TS 常见）。
    frame_dt = Fraction(1001, 30000)
    tb = Fraction(1, 90000)
    # 逐包独立换算；误差不会因为段很长而累加
    worst = Fraction(0)
    prev = None
    for i in range(100_000):
        t = i * frame_dt
        tr = rescale_ticks(t, tb)
        worst = max(worst, abs(tr.residual))
        if prev is not None:
            assert tr.ticks - prev in (3002, 3003)  # 3003/1001 的两种间隔
        prev = tr.ticks
    # 单包误差严格小于一个输出 tick，且不随 i 增大
    assert worst < tb
    # 浮点做法会在十万帧后漂移；这里第一帧与最后一帧的残差同样有界
    first = rescale_ticks(0 * frame_dt, tb)
    last = rescale_ticks(99_999 * frame_dt, tb)
    assert abs(first.residual) < tb and abs(last.residual) < tb


def test_residual_is_reported_not_hidden():
    # 无法精确表示的时间：tick 被四舍五入，残差精确公开
    tr = rescale_ticks(Fraction(1, 3), Fraction(1, 10))
    assert tr.ticks == 3  # 3.333... -> 3
    assert tr.residual == Fraction(3, 10) - Fraction(1, 3)
    assert tr.residual != 0


def test_rounding_modes():
    assert rescale_ticks(Fraction(5, 10), Fraction(1), Rounding.NEAREST_AWAY_FROM_ZERO).ticks == 1
    assert rescale_ticks(Fraction(4, 10), Fraction(1), Rounding.NEAREST_AWAY_FROM_ZERO).ticks == 0
    assert rescale_ticks(Fraction(-5, 10), Fraction(1)).ticks == -1
    assert rescale_ticks(Fraction(14, 10), Fraction(1), Rounding.DOWN).ticks == 1
    assert rescale_ticks(Fraction(14, 10), Fraction(1), Rounding.UP).ticks == 2
    assert rescale_ticks(Fraction(-14, 10), Fraction(1), Rounding.TOWARD_ZERO).ticks == -1


def test_affine_segment_mapping_preserves_relative_times():
    # 段起点 12.5 秒；原始 0/1/2 帧应映射到 12.5 + i/25
    start = Fraction(25, 2)
    for i in range(5):
        m = map_field(i, Fraction(1, 25), start, Fraction(1, 12800))
        assert m.original_seconds == Fraction(i, 25)
        assert m.mapped_seconds == start + Fraction(i, 25)
        assert m.residual == 0


def test_multi_segment_continuity_is_exact_not_accumulated():
    # 两段接续：第二段起点 = 第一段起点 + 内容时长。
    # 每段独立做仿射映射，拼接处必须精确相接，无累加误差。
    tb = Fraction(1, 12800)
    seg1_start = Fraction(0)
    seg2_start = Fraction(100, 25)  # 第一段 100 帧
    last_seg1 = map_field(99, Fraction(1, 25), seg1_start, tb)
    first_seg2 = map_field(0, Fraction(1, 25), seg2_start, tb)
    assert first_seg2.output_ticks - last_seg1.output_ticks == 512


def test_missing_field_is_unconvertible_but_maps_none():
    m = map_field(None, None, Fraction(0), Fraction(1, 12800))
    assert m.convertible is False
    assert m.output_ticks is None


def test_as_fraction_accepts_tuple():
    assert as_fraction((1, 25)) == Fraction(1, 25)
    with pytest.raises(TypeError):
        as_fraction(0.04)  # float 明确拒绝，防止意外浮点路径
