"""整理次序、缺口、阴影、预算、段结束冲刷与状态可见性测试。"""
from fractions import Fraction

import pytest

from media_clock_core import (
    FeedOutcome,
    InputPacket,
    MediaClockKernel,
    MediaKind,
    PacketDelivered,
    RangeSkipped,
    SegmentSpec,
    TrackSpec,
    Usability,
    WaitingBudget,
)
from media_clock_core.buffer import SkipOverlapError

V_TB = Fraction(1, 25)
OUT_TB = Fraction(1, 12800)


def make_kernel(budget=None):
    k = MediaClockKernel(budget or WaitingBudget())
    k.register_track(TrackSpec("v", MediaKind.VIDEO, V_TB, OUT_TB))
    k.register_track(TrackSpec("a", MediaKind.AUDIO, Fraction(1, 48000), Fraction(1, 48000)))
    k.open_segment(SegmentSpec("seg0", Fraction(0)))
    return k


def pkt(seq, pts=None, dts=None, key=False, track="v", size=10, dur=1, seg="seg0"):
    if pts is None:
        pts = seq
    if dts is None:
        dts = seq
    tb = V_TB if track == "v" else Fraction(1, 48000)
    if track != "v" and pts == seq and dts == seq:
        pts, dts, dur = seq * 1024, seq * 1024, 1024
    return InputPacket(
        segment_id=seg, track_id=track, seq=seq, pts=pts, dts=dts,
        duration=dur, time_base=tb, is_keyframe=key,
        payload=f"{track}:{seq}".encode(), payload_bytes=size,
    )


def drain_delivered(k):
    evs = k.drain()
    return [e for e in evs if isinstance(e, PacketDelivered)]


def test_in_order_delivers_immediately_in_capture_order_not_pts_order():
    k = make_kernel()
    # B 帧风格：交付顺序 0,1,2,3，但 PTS 是 0,3,1,2
    for seq, pts in [(0, 0), (1, 3), (2, 1), (3, 2)]:
        r = k.feed(pkt(seq, pts=pts, dts=seq - 2, key=(seq == 0)))
        assert r.outcome is FeedOutcome.ACCEPTED
    deliv = drain_delivered(k)
    assert [d.seq for d in deliv] == [0, 1, 2, 3]
    assert [d.timing.pts.ticks for d in deliv] == [0, 1536, 512, 1024]
    assert [d.timing.dts.ticks for d in deliv] == [-1024, -512, 0, 512]


def test_out_of_order_network_arrival_reorders_within_known_range():
    k = make_kernel()
    # 读取器按乱序调用：3,1,0,2
    assert k.feed(pkt(3)).outcome is FeedOutcome.ACCEPTED
    assert k.feed(pkt(1)).outcome is FeedOutcome.ACCEPTED
    assert drain_delivered(k) == []  # 前缘是 0，什么都不能交付
    st = k.status()
    v = next(t for t in st.tracks if t.track_id == "v")
    assert v.holding == 2
    assert v.waiting_ranges[0].start_seq == 0
    assert v.waiting_ranges[0].end_seq == 1  # 缺口是 [0,1)，不是猜边界
    assert k.feed(pkt(0, key=True)).outcome is FeedOutcome.ACCEPTED
    # 0 到达后，已占住且连续的 1 立即交付；2 未到所以停在那
    assert [d.seq for d in drain_delivered(k)] == [0, 1]
    k.feed(pkt(2))
    assert [d.seq for d in drain_delivered(k)] == [2, 3]


def test_waiting_status_names_segment_and_track():
    k = make_kernel()
    k.feed(pkt(5, track="v"))
    k.feed(pkt(2, track="a"))
    st = k.status()
    by = {(t.segment_id, t.track_id): t for t in st.tracks}
    assert by[("seg0", "v")].waiting_ranges[0].kind == "waiting"
    assert by[("seg0", "a")].waiting_ranges[0].kind == "waiting"
    # 音频视频缺口互相独立
    assert by[("seg0", "v")].next_contiguous_seq == 0
    assert by[("seg0", "a")].next_contiguous_seq == 0


def test_confirm_skip_emits_range_with_decode_shadow():
    k = make_kernel()
    # 0 是关键帧；1,2,3 缺失；下游关键帧 8 先乱序到达并占住
    k.feed(pkt(0, key=True))
    k.feed(pkt(8, key=True))
    k.confirm_skip("seg0", "v", 1, 4, reason="confirmed")
    for s in range(4, 8):
        k.feed(pkt(s))
    evs = k.drain()
    skips = [e for e in evs if isinstance(e, RangeSkipped)]
    deliv = [e for e in evs if isinstance(e, PacketDelivered)]
    assert len(skips) == 1
    assert skips[0].start_seq == 1 and skips[0].end_seq == 4
    # 确认跳过时关键帧 8 已在占住中可见，阴影上界可精确报告
    assert skips[0].shadow_until_seq == 8
    # 4..7 标记为解码阴影，8 恢复
    assert {d.seq: d.usability for d in deliv} == {
        0: Usability.INTACT,
        4: Usability.DEPENDENCY_BROKEN,
        5: Usability.DEPENDENCY_BROKEN,
        6: Usability.DEPENDENCY_BROKEN,
        7: Usability.DEPENDENCY_BROKEN,
        8: Usability.INTACT,
    }
    # 影响随数据流向下游，不只是计数
    assert [d for d in deliv if d.seq == 4][0].shadowed_by_gap is True


def test_shadow_impact_carried_per_packet_when_keyframe_unknown_at_skip():
    k = make_kernel()
    k.feed(pkt(0, key=True))
    # 确认缺口时下一个关键帧尚未出现（没有任何后续包占住）：上界未知 None
    k.confirm_skip("seg0", "v", 1, 4)
    skip = next(e for e in k.drain() if isinstance(e, RangeSkipped))
    assert skip.shadow_until_seq is None
    # 之后包陆续到达：4..7 在阴影里，8 是新关键帧，阴影解除
    for s in range(4, 9):
        k.feed(pkt(s, key=(s == 8)))
    deliv = drain_delivered(k)
    flags = {d.seq: d.usability for d in deliv}
    assert flags[4] is Usability.DEPENDENCY_BROKEN
    assert flags[7] is Usability.DEPENDENCY_BROKEN
    assert flags[8] is Usability.INTACT


def test_skip_refuses_to_silently_delete_arrived_data():
    k = make_kernel()
    k.feed(pkt(2))
    # 想跳过 [0,3)，但 2 已经到达并占住
    with pytest.raises(SkipOverlapError):
        k.confirm_skip("seg0", "v", 0, 3)


def test_audio_gap_does_not_poison_later_packets():
    k = make_kernel()
    k.feed(pkt(0, track="a"))
    k.confirm_skip("seg0", "a", 1, 3)
    k.feed(pkt(3, track="a"))
    evs = k.drain()
    skip = next(e for e in evs if isinstance(e, RangeSkipped))
    deliv = [e for e in evs if isinstance(e, PacketDelivered)]
    assert skip.shadow_until_seq is None  # 音频无帧间预测，不存在解码阴影
    assert all(d.usability is Usability.INTACT for d in deliv)


def test_finish_track_flushes_waiting_into_results_not_eternal_read():
    k = make_kernel()
    k.feed(pkt(0, key=True))
    k.feed(pkt(2))  # 1 永远不会来了
    k.finish_track("seg0", "v")
    evs = k.drain()
    skip = [e for e in evs if isinstance(e, RangeSkipped)]
    deliv = [e for e in evs if isinstance(e, PacketDelivered)]
    assert skip[0].reason == "unfinished-at-end"
    assert [d.seq for d in deliv] == [0, 2]
    st = k.status()
    v = next(t for t in st.tracks if t.track_id == "v")
    assert not v.can_input
    assert v.holding == 0
    # 结束后再喂被拒绝
    assert k.feed(pkt(5)).outcome is FeedOutcome.FINISHED


def test_budget_caps_held_packets_not_inorder():
    # 预算只约束“因乱序占住未交付”的包；in-order 包立即交付、不占预算。
    k = make_kernel(WaitingBudget(max_packets=1, max_bytes=10_000_000))
    assert k.feed(pkt(1)).outcome is FeedOutcome.ACCEPTED  # 占住 1
    assert k.feed(pkt(2)).outcome is FeedOutcome.BUDGET_EXCEEDED
    # 前缘包即使预算已满也立即交付，并连带释放占住的 1
    assert k.feed(pkt(0, key=True)).outcome is FeedOutcome.ACCEPTED
    assert [d.seq for d in drain_delivered(k)] == [0, 1]
    assert k.status().resources.held_packets == 0
    # 重复喂包幂等，不产生新事件
    assert k.feed(pkt(1)).outcome is FeedOutcome.ACCEPTED
    assert drain_delivered(k) == []
    # 确认跳过真正缺失的缺口后前缘推进
    k.confirm_skip("seg0", "v", 2, 3)
    assert k.feed(pkt(3)).outcome is FeedOutcome.ACCEPTED


def test_budget_bytes_cap():
    k = make_kernel(WaitingBudget(max_packets=100, max_bytes=15))
    k.feed(pkt(1, size=10))
    assert k.feed(pkt(2, size=10)).outcome is FeedOutcome.BUDGET_EXCEEDED


def test_payload_is_never_modified_and_origin_link_retained():
    k = make_kernel()
    p = pkt(0, key=True)
    k.feed(p)
    d = drain_delivered(k)[0]
    assert d.origin is p
    assert d.payload == b"v:0"
    # 修正时间与原始时间的可检查关系
    assert d.timing.output_time_base == OUT_TB
    assert d.timing.dts.exact_seconds == p.dts * V_TB


def test_unknown_target_and_cancel_release_resources():
    k = make_kernel()
    bad = InputPacket("nope", "v", 0, 0, 0, 1, V_TB, True, b"x", 1)
    assert k.feed(bad).outcome is FeedOutcome.UNKNOWN_TARGET
    k.feed(pkt(5))  # 占住
    k.cancel()
    st = k.status()
    assert st.resources.held_packets == 0
    assert st.resources.cancelled is True
    assert k.feed(pkt(6)).outcome is FeedOutcome.CANCELLED


def test_duplicate_and_late_packets_ignored_idempotently():
    k = make_kernel()
    k.feed(pkt(0, key=True))
    k.feed(pkt(0))
    k.feed(pkt(1))
    deliv = drain_delivered(k)
    assert [d.seq for d in deliv] == [0, 1]


def test_multisegment_identity_and_anchor():
    k = MediaClockKernel()
    k.register_track(TrackSpec("v", MediaKind.VIDEO, V_TB, OUT_TB))
    k.open_segment(SegmentSpec("A", Fraction(0)))
    k.open_segment(SegmentSpec("B", Fraction(1)))  # B 从公共时间线 1s 开始
    k.feed(InputPacket("A", "v", 0, 0, 0, 1, V_TB, True, b"a0", 1))
    k.feed(InputPacket("B", "v", 0, 0, 0, 1, V_TB, True, b"b0", 1))
    deliv = {d.segment_id: d for d in drain_delivered(k)}
    assert deliv["A"].timing.pts.ticks == 0
    assert deliv["B"].timing.pts.ticks == 12800


def test_segment_boundary_is_never_inferred_from_time_decrease():
    k = make_kernel()
    # PTS 回跳来自 B 帧，序号继续前进——不能产生新段，不能当丢包
    k.feed(pkt(0, pts=0, dts=-2, key=True))
    k.feed(pkt(1, pts=3, dts=-1))
    k.feed(pkt(2, pts=1, dts=0))
    deliv = drain_delivered(k)
    assert [d.seq for d in deliv] == [0, 1, 2]
    assert all(not d.shadowed_by_gap for d in deliv)
    assert len(k.status().tracks) == 2  # 仍然只有 v,a 两条轨、一个段
