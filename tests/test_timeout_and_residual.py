"""等待超时冲刷与累计表示误差的可观察性。"""
from fractions import Fraction

from media_clock_core import (
    InputPacket,
    MediaClockKernel,
    MediaKind,
    PacketDelivered,
    RangeSkipped,
    SegmentSpec,
    TimingStatus,
    TrackSpec,
    WaitingBudget,
)


def _pkt(seq, track="v"):
    tb = Fraction(1, 25)
    return InputPacket("s", track, seq, seq, seq - 2, 1, tb, seq == 0, b"x", 1)


def test_timeout_flushes_gap_as_skipped_not_silent():
    # 用假时钟精确控制，不依赖真实睡眠
    t = [1000.0]
    k = MediaClockKernel(
        WaitingBudget(default_timeout=2.0),
        clock=lambda: t[0],
    )
    k.register_track(TrackSpec("v", MediaKind.VIDEO, Fraction(1, 25), Fraction(1, 12800)))
    k.open_segment(SegmentSpec("s", Fraction(0)))
    k.feed(_pkt(0))
    k.drain()
    k.feed(_pkt(3))  # 1,2 缺失，3 占住，界定缺口 [1,3)
    # 时间推进但未超时时，仍是 waiting
    t[0] = 1001.0
    assert k.sweep_timeouts() == 0
    st = k.status()
    assert st.tracks[0].waiting_ranges[0].start_seq == 1
    # 超时后：缺口 [1,3) 冲刷为确认跳过，3 随后可交付
    t[0] = 1003.0
    n = k.sweep_timeouts()
    assert n == 1
    evs = k.drain()
    skips = [e for e in evs if isinstance(e, RangeSkipped)]
    assert len(skips) == 1
    assert skips[0].reason == "timeout"
    assert (skips[0].start_seq, skips[0].end_seq) == (1, 3)


def test_timeout_uses_highest_observed_when_out_of_order():
    t = [0.0]
    k = MediaClockKernel(WaitingBudget(default_timeout=1.0), clock=lambda: t[0])
    k.register_track(TrackSpec("v", MediaKind.VIDEO, Fraction(1, 25), Fraction(1, 12800)))
    k.open_segment(SegmentSpec("s", Fraction(0)))
    k.feed(_pkt(0)); k.drain()
    k.feed(_pkt(4))  # 1,2,3 缺失，4 占住
    t[0] = 5.0
    k.sweep_timeouts()
    evs = k.drain()  # skip 与随后连带交付的包 4 在同一批
    skip = [e for e in evs if isinstance(e, RangeSkipped)][0]
    # 超时只把“已确定边界”的缺口 [1,4) 冲刷，占住的 4 随后交付
    assert (skip.start_seq, skip.end_seq) == (1, 4)
    deliv = [e for e in evs if isinstance(e, PacketDelivered)]
    assert [d.seq for d in deliv] == [4]

def test_cumulative_residual_exposed_for_drift_audit():
    # 30000/1001 -> 1/12800：大部分帧无法精确表示
    cap_tb = Fraction(1001, 30000)
    k = MediaClockKernel()
    k.register_track(TrackSpec("v", MediaKind.VIDEO, cap_tb, Fraction(1, 12800)))
    k.open_segment(SegmentSpec("s", Fraction(0)))
    for i in range(20):
        k.feed(InputPacket("s", "v", i, i, i, 1, cap_tb, i == 0, b"x", 1))
    k.finish_track("s", "v")
    evs = k.drain()
    delivered = [e for e in evs if isinstance(e, PacketDelivered)]
    rounded = [e for e in delivered if e.timing.pts_status is TimingStatus.ROUNDED]
    assert len(rounded) > 0
    st = k.status().tracks[0]
    assert st.rounded_packets == len(rounded)
    # 累计残差非零且有界（每包严格小于一个输出 tick）
    assert st.cumulative_pts_residual_seconds != 0
    one_tick = Fraction(1, 12800)
    assert abs(st.cumulative_pts_residual_seconds) < 20 * one_tick
    # 逐包关系可检查：exact 与 represented 之差就是残差
    for e in rounded:
        tick = e.timing.pts
        assert tick.represented_seconds - tick.exact_seconds == tick.residual_seconds
