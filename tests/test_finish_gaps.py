"""段结束时多个交错缺口的冲刷语义。"""
from fractions import Fraction

from media_clock_core import (
    InputPacket,
    MediaClockKernel,
    MediaKind,
    PacketDelivered,
    RangeSkipped,
    SegmentSpec,
    TrackSpec,
)

TB = Fraction(1, 25)


def _kernel():
    k = MediaClockKernel()
    k.register_track(TrackSpec("v", MediaKind.VIDEO, TB, Fraction(1, 12800)))
    k.open_segment(SegmentSpec("s", Fraction(0)))
    return k


def _pkt(seq, key=False):
    return InputPacket("s", "v", seq, seq, seq - 2, 1, TB, key, b"x", 1)


def test_finish_flushes_multiple_interleaved_gaps():
    k = _kernel()
    # 到达 0,2,4；缺 1 和 3（两个交错缺口）
    k.feed(_pkt(0, key=True))
    k.feed(_pkt(2))
    k.feed(_pkt(4))
    k.finish_track("s", "v")
    evs = k.drain()
    skips = [(e.start_seq, e.end_seq) for e in evs if isinstance(e, RangeSkipped)]
    seqs = [e.seq for e in evs if isinstance(e, PacketDelivered)]
    # 两个缺口都被显式确认为跳过，事件按序：0,skip[1,2),2,skip[3,4),4
    assert skips == [(1, 2), (3, 4)]
    assert seqs == [0, 2, 4]
    # 不留下任何占住资源，状态为已结束
    st = k.status()
    assert st.resources.held_packets == 0
    assert all(not t.can_input for t in st.tracks if t.track_id == "v")


def test_finish_with_only_keyframe_then_gap():
    k = _kernel()
    k.feed(_pkt(0, key=True))
    k.feed(_pkt(3))  # 缺 1,2
    k.finish_track("s", "v")
    evs = k.drain()
    skip = next(e for e in evs if isinstance(e, RangeSkipped))
    assert (skip.start_seq, skip.end_seq) == (1, 3)
    assert [e.seq for e in evs if isinstance(e, PacketDelivered)] == [0, 3]
