"""精确有理时间换算测试：无浮点、无漂移、误差可检查。"""
from fractions import Fraction

import pytest

from media_clock_core.timing import TickConverter
from media_clock_core import TimingStatus, TrackSpec, MediaKind


def conv(output_tb=Fraction(1, 12800), start=0, offset=0, tb=Fraction(1, 25), rounding="half_even"):
    spec = TrackSpec("v", MediaKind.VIDEO, tb, output_tb, rounding=rounding)
    return TickConverter(spec, Fraction(start), Fraction(offset))


def test_exact_scale_25fps_to_12800():
    c = conv()
    t, warn = c.convert(pts=0, dts=-2, duration=1, packet_tb=None)
    assert warn is None
    assert t.dts.ticks == -1024
    assert t.dts.exact_seconds == Fraction(-2, 25)
    assert t.dts_status is TimingStatus.EXACT
    assert t.pts.ticks == 0
    assert t.duration.ticks == 512
    assert t.duration.exact_seconds == Fraction(1, 25)
    assert t.worst_residual_seconds == 0


def test_bframe_timestamps_preserved_with_negative_dts():
    # 题目给的形状：pts 0,3,1,2；dts -2,-1,0,1
    c = conv()
    rows = [
        (0, -2), (3, -1), (1, 0), (2, 1),
        (7, 2), (5, 3), (4, 4), (6, 5),
    ]
    out = []
    for pts, dts in rows:
        t, warn = c.convert(pts, dts, 1, None)
        assert warn is None
        out.append((t.pts.ticks, t.dts.ticks))
    # 显示顺序不变（只是换了基准），负 dts 原样保留
    assert [p for p, _ in out] == [0, 1536, 512, 1024, 3584, 2560, 2048, 3072]
    assert [d for _, d in out] == [-1024, -512, 0, 512, 1024, 1536, 2048, 2560]


def test_30000_1001_no_float_drift_across_many_packets():
    tb = Fraction(1001, 30000)
    c = conv(output_tb=Fraction(1, 90000), tb=tb)
    prev = None
    residual_sum = Fraction(0)
    for i in range(100000):
        t, warn = c.convert(pts=i, dts=i, duration=1, packet_tb=None)
        assert warn is None
        residual_sum += t.pts.residual_seconds
        if prev is not None:
            assert t.pts.ticks > prev
        prev = t.pts.ticks
    # 精确间隔恒为 1001/30000 s，tick 间隔恒为 3003
    t0, _ = TickConverter(
        TrackSpec("v", MediaKind.VIDEO, tb, Fraction(1, 90000)), Fraction(0), Fraction(0)
    ).convert(0, 0, 1, None)
    t1, _ = TickConverter(
        TrackSpec("v", MediaKind.VIDEO, tb, Fraction(1, 90000)), Fraction(0), Fraction(0)
    ).convert(1, 1, 1, None)
    assert (t1.pts.exact_seconds - t0.pts.exact_seconds) == Fraction(1001, 30000)
    assert t1.pts.ticks - t0.pts.ticks == 3003
    # 残差有界（每包 < 1 tick），绝不伪装成精确值
    assert abs(residual_sum) < 100000 * Fraction(1, 90000)


def test_inexact_value_is_marked_rounded_not_exact():
    # 1/25 = 512/12800 恰好精确；选一个不能整除的基准制造残差
    c = conv(output_tb=Fraction(1, 1000), tb=Fraction(1, 25))
    t, _ = c.convert(pts=1, dts=1, duration=1, packet_tb=None)
    # 0.04 s = 40/1000 精确；pts=3 -> 0.12 -> 120 精确。换输出 tb=1/1001:
    c2 = conv(output_tb=Fraction(1, 1001), tb=Fraction(1, 25))
    t2, _ = c2.convert(pts=1, dts=1, duration=1, packet_tb=None)
    # 0.04*1001 = 40.04 -> 40 ticks，有残差
    assert t2.pts.ticks == 40
    assert t2.pts_status is TimingStatus.ROUNDED
    assert t2.pts.residual_seconds != 0
    assert t2.pts.represented_seconds != t2.pts.exact_seconds


def test_segment_offset_and_multisegment_continuity():
    tb = Fraction(1001, 30000)
    spec = TrackSpec("v", MediaKind.VIDEO, tb, Fraction(1, 90000))
    # 段 A 从 0 开始，放 5 帧；段 B 紧接第 5 帧的显示时刻开始
    cA = TickConverter(spec, Fraction(0), Fraction(0))
    lastA, _ = cA.convert(4, 4, 1, None)
    startB = lastA.pts.exact_seconds + lastA.duration.exact_seconds
    cB = TickConverter(spec, startB, Fraction(0))
    firstB, warn = cB.convert(0, 0, 1, None)
    assert warn is None
    # B 的第 0 帧精确接在 A 第 4 帧时长之后，无浮点缝隙
    assert firstB.pts.exact_seconds == Fraction(5 * 1001, 30000)
    assert firstB.pts.ticks == 5 * 3003


def test_rounding_collision_is_bumped_with_recorded_adjustment():
    # 输出基准粗到两个精确不同的 DTS 会四舍五入到同一 tick
    c = conv(output_tb=Fraction(1, 10), tb=Fraction(1, 25))
    t0, w0 = c.convert(0, 0, 1, None)
    t1, w1 = c.convert(1, 1, 1, None)
    # 0.0 和 0.04 都落到 tick 0 -> 第二个被钳到 1
    assert t0.dts.ticks == 0
    assert t1.dts.ticks == 1
    assert t1.monotonic_adjustment_seconds == Fraction(1, 10)


def test_exact_non_monotonic_dts_reports_warning_not_silent_fix():
    c = conv()
    c.convert(0, 0, 1, None)
    _, warn = c.convert(100, -1, 1, None)  # 精确 dts 倒退
    assert warn is not None and "non-monotonic" in warn


def test_none_fields_are_unmapped():
    c = conv()
    t, _ = c.convert(None, None, None, None)
    assert t.pts is None and t.dts is None and t.duration is None
    assert t.pts_status is TimingStatus.UNMAPPED
