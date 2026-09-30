"""Kernel contract tests without PyAV: ordering, gaps, budgets, identity, lifecycle."""

from fractions import Fraction

import pytest

from media_clock_core import (
    BudgetExceeded,
    Cancelled,
    DecodeAvailability,
    EndOfTimeline,
    InputPacket,
    InputTrackSpec,
    OutputTrackSpec,
    PayloadUsability,
    Rounding,
    SkipReason,
    SubmitOutcome,
    Timeline,
    TimeStatus,
)
from media_clock_core.events import (
    DeliveredPacket,
    RecoveryObserved,
    SegmentClosed,
    SkippedRange,
    WaitingRange,
)

VID = "v"
AUD = "a"


def make_timeline(queue_cap=256):
    tl = Timeline(
        default_wait_budget_packets=16,
        output_queue_capacity=queue_cap,
    )
    tl.register_output_track(OutputTrackSpec(VID, Fraction(1, 12800)))
    tl.register_output_track(OutputTrackSpec(AUD, Fraction(1, 48000)))
    return tl


def open_seg(tl, sid="s1", start=0, tracks=("v", "a")):
    tl.open_segment(
        sid,
        Fraction(start),
        [InputTrackSpec(t, t) for t in tracks],
    )


def vpkt(seq, pts, dts, *, key=False, sid="s1", track="v",
         duration=1, usability=PayloadUsability.USABLE, payload=None):
    return InputPacket(
        segment_id=sid, track_id=track, capture_seq=seq,
        pts=pts, dts=dts, duration=duration,
        time_base=Fraction(1, 25),
        payload=payload if payload is not None else bytes([seq & 0xFF]),
        keyframe=key, payload_usability=usability,
    )


def drain(tl, kinds=None):
    events = []
    while True:
        ev = tl.pull(block=False)
        if ev is None:
            return events
        if kinds is None or getattr(ev, "kind", None) in kinds:
            events.append(ev)


# --------------------------------------------------------------------------- #
# B 帧：按采集序号（DTS 顺序）交付，PTS 回跳原样保留
# --------------------------------------------------------------------------- #


def test_bframe_pts_rewind_is_delivered_not_treated_as_late_packet():
    tl = make_timeline()
    open_seg(tl, tracks=("v",))
    # 用户给的形态：显示序 0,3,1,2；解码序 dts -2,-1,0,1
    pkts = [
        vpkt(0, pts=0, dts=-2, key=True),
        vpkt(1, pts=3, dts=-1),
        vpkt(2, pts=1, dts=0),
        vpkt(3, pts=2, dts=1),
    ]
    assert all(tl.submit(p) is SubmitOutcome.ACCEPTED for p in pkts)
    tl.close_segment("s1")
    tl.close()
    delivered = [e for e in tl.events() if e.kind == "packet"]
    seqs = [e.capture_seq for e in delivered]
    assert seqs == [0, 1, 2, 3]                     # 解码顺序
    assert [e.output_pts for e in delivered] == [0, 1536, 512, 1024]
    assert [e.output_dts for e in delivered] == [-1024, -512, 0, 512]
    # 没有任何 waiting/skipped 事件：PTS 回跳不是缺包
    assert not any(e.kind in ("waiting", "skipped") for e in delivered)
    assert all(e.decode_availability is DecodeAvailability.INTACT for e in delivered)


def test_payload_is_untouched_and_linked_to_original():
    tl = make_timeline()
    open_seg(tl, tracks=("v",))
    raw = bytes(range(10))
    tl.submit(vpkt(0, 0, -2, key=True, payload=raw))
    tl.close_segment("s1"); tl.close()
    ev = next(e for e in tl.events() if e.kind == "packet")
    assert ev.payload is raw  # 同一份字节，零拷贝
    assert (ev.segment_id, ev.track_id, ev.capture_seq) == ("s1", "v", 0)
    assert ev.correction.pts.original_ticks == 0
    assert ev.correction.dts.original_ticks == -2


# --------------------------------------------------------------------------- #
# 乱序、空洞、等待、迟到
# --------------------------------------------------------------------------- #


def test_out_of_order_arrivals_are_reordered_and_waiting_is_reported():
    tl = make_timeline(queue_cap=64)
    open_seg(tl, tracks=("v",))
    tl.submit(vpkt(0, 0, -2, key=True))
    tl.submit(vpkt(2, 1, 0))
    tl.submit(vpkt(3, 2, 1))
    evs = drain(tl)
    # 只有 seq0 能交付；seq1 缺失，seq2/3 在缓存里
    packets = [e for e in evs if e.kind == "packet"]
    assert [e.capture_seq for e in packets] == [0]
    waiting = [e for e in evs if e.kind == "waiting"]
    assert len(waiting) == 1  # 同一个空洞只报一次，不随积压数刷屏
    assert waiting[0].first_seq == 1
    assert waiting[0].buffered_beyond == 1  # 首报时只有 seq2 在缓存
    assert waiting[0].highest_observed_seq == 2

    # 当前积压规模通过 status() 查询
    st = tl.status()
    vstat = next(t for t in st.tracks if t.track_id == "v")
    assert vstat.buffered_count == 2

    # 空洞补上后，积压的包按序全部交付
    tl.submit(vpkt(1, 3, -1))
    evs2 = drain(tl)
    packets2 = [e for e in evs2 if e.kind == "packet"]
    assert [e.capture_seq for e in packets2] == [1, 2, 3]


def test_duplicate_and_late_after_skip_outcomes():
    tl = make_timeline()
    open_seg(tl, tracks=("v",))
    tl.submit(vpkt(0, 0, -2, key=True))
    tl.submit(vpkt(0, 0, -2, key=True, payload=b"zz"))
    assert tl.submit(vpkt(0, 0, -2, key=True)) is SubmitOutcome.DUPLICATE
    # seq1 确认跳过，seq2/3 已在缓存里
    tl.submit(vpkt(2, 1, 0, key=False))
    tl.submit(vpkt(3, 2, 1, key=False))
    sk = tl.skip("s1", "v", 1)
    assert isinstance(sk, SkippedRange) and sk.count == 1
    assert sk.reason is SkipReason.EXPLICIT
    # 迟到的 seq1 被拒绝，但不会当作乱序缓存
    assert tl.submit(vpkt(1, 3, -1)) is SubmitOutcome.LATE_AFTER_SKIP
    tl.close_segment("s1"); tl.close()
    evs = list(tl.events())
    assert [e.capture_seq for e in evs if e.kind == "packet"] == [0, 2, 3]
    assert [e.first_seq for e in evs if e.kind == "skipped"] == [1]


def test_close_segment_bounds_remaining_wait_as_skipped():
    tl = make_timeline()
    open_seg(tl, tracks=("v",))
    tl.submit(vpkt(0, 0, -2, key=True))
    tl.submit(vpkt(1, 3, -1))
    tl.submit(vpkt(4, 5, 2))  # 2,3 缺失
    tl.close_segment("s1")
    tl.close()
    evs = list(tl.events())
    skipped = [e for e in evs if e.kind == "skipped"]
    assert [(e.first_seq, e.last_seq, e.reason) for e in skipped] == [
        (2, 3, SkipReason.END_OF_SEGMENT)
    ]
    # seq4 仍然被交付（只是降级）
    assert any(e.kind == "packet" and e.capture_seq == 4 for e in evs)
    assert any(e.kind == "segment_closed" for e in evs)


def test_segment_close_is_not_inferred_from_timestamp_decrease():
    # 时间变小绝不能产生新分段；分段只能显式 open
    tl = make_timeline()
    open_seg(tl, tracks=("v",))
    tl.submit(vpkt(0, 3, 0))   # pts 3
    tl.submit(vpkt(1, 0, 1))   # pts 回退到 0
    st = tl.status()
    assert len(st.segments) == 1
    assert st.segments[0].segment_id == "s1"


# --------------------------------------------------------------------------- #
# 解码可用性契约
# --------------------------------------------------------------------------- #


def test_skip_publishes_decodability_impact_until_keyframe_recovery():
    tl = make_timeline(queue_cap=64)
    open_seg(tl, tracks=("v",))
    tl.submit(vpkt(0, 0, 0, key=True))
    # seq1 缺失，seq2/3 是非关键，seq4 是下一个关键帧
    tl.submit(vpkt(2, 2, 2, key=False))
    tl.submit(vpkt(3, 3, 3, key=False))
    tl.submit(vpkt(4, 4, 4, key=True))
    sk = tl.skip("s1", "v", 1)
    assert sk.next_keyframe_seq == 4
    assert sk.next_recovery_time_seconds == Fraction(4, 25)

    tl.close_segment("s1"); tl.close()
    evs = list(tl.events())
    p2 = next(e for e in evs if e.kind == "packet" and e.capture_seq == 2)
    p3 = next(e for e in evs if e.kind == "packet" and e.capture_seq == 3)
    p4 = next(e for e in evs if e.kind == "packet" and e.capture_seq == 4)
    assert p2.decode_availability is DecodeAvailability.DEGRADED_AFTER_GAP
    assert p3.decode_availability is DecodeAvailability.DEGRADED_AFTER_GAP
    assert p4.decode_availability is DecodeAvailability.RECOVERY_POINT
    rec = [e for e in evs if e.kind == "recovery"]
    assert len(rec) == 1
    assert rec[0].recovery_seq == 4


def test_skip_with_unknown_recovery_says_so():
    tl = make_timeline()
    open_seg(tl, tracks=("v",))
    tl.submit(vpkt(0, 0, 0, key=True))
    tl.submit(vpkt(2, 2, 2, key=False))
    sk = tl.skip("s1", "v", 1)
    assert sk.next_keyframe_seq is None
    assert sk.next_recovery_time_seconds is None


def test_payload_corruption_and_time_unconvertibility_are_orthogonal():
    tl = make_timeline(queue_cap=64)
    open_seg(tl, tracks=("v",))
    tl.submit(vpkt(0, 0, 0, key=True, usability=PayloadUsability.CORRUPT))
    bad_time = InputPacket(
        segment_id="s1", track_id="v", capture_seq=1,
        pts=10, dts=10, duration=1, time_base=None,
        payload=b"x", keyframe=False,
    )
    tl.submit(bad_time)
    tl.close_segment("s1"); tl.close()
    evs = [e for e in tl.events() if e.kind == "packet"]
    assert evs[0].decode_availability is DecodeAvailability.CORRUPTED
    assert evs[1].decode_availability is DecodeAvailability.TIME_UNKNOWN
    assert evs[1].correction.status is TimeStatus.UNCONVERTIBLE
    assert evs[1].output_pts is None
    # 两个包都交付了：可用性和时间是正交的两件事
    assert [e.capture_seq for e in evs] == [0, 1]


# --------------------------------------------------------------------------- #
# 预算与背压
# --------------------------------------------------------------------------- #


def test_wait_budget_caps_undelivered_packets():
    tl = Timeline(
        default_wait_budget_packets=2, output_queue_capacity=64,
    )
    tl.register_output_track(OutputTrackSpec(VID, Fraction(1, 12800)))
    tl.open_segment("s1", 0, [InputTrackSpec("v", VID)])
    tl.submit(vpkt(0, 0, 0, key=True))
    # seq1 永不补齐：seq2、seq3 正好占满 2 个缓存位
    tl.submit(vpkt(2, 2, 2))
    tl.submit(vpkt(3, 3, 3))
    with pytest.raises(BudgetExceeded):
        tl.submit(vpkt(4, 4, 4))
    # 调用方显式确认缺失后，缓存压力解除，seq4 可以正常接收
    tl.skip("s1", "v", 1)
    tl.submit(vpkt(4, 4, 4))


def test_wait_budget_also_caps_buffered_bytes():
    tl = Timeline(
        default_wait_budget_packets=None,
        default_wait_budget_bytes=1000,
        output_queue_capacity=64,
    )
    tl.register_output_track(OutputTrackSpec(VID, Fraction(1, 12800)))
    tl.open_segment("s1", 0, [InputTrackSpec("v", VID)])
    tl.submit(InputPacket(
        "s1", "v", 0, 0, 0, 1, Fraction(1, 25), b"x" * 100, keyframe=True))
    tl.submit(InputPacket(
        "s1", "v", 2, 2, 2, 1, Fraction(1, 25), b"x" * 600))
    # seq1 缺失，再来一个 600 字节会把未交付缓存推到 1200 > 1000
    with pytest.raises(BudgetExceeded):
        tl.submit(InputPacket(
            "s1", "v", 3, 3, 3, 1, Fraction(1, 25), b"x" * 600))


def test_status_shows_held_resources_and_their_release_after_finish():
    tl = make_timeline(queue_cap=64)
    open_seg(tl, tracks=("v",))
    tl.submit(vpkt(0, 0, 0, key=True))
    tl.submit(vpkt(2, 2, 2))  # seq1 缺失，seq2 占着未交付缓存
    st = tl.status()
    assert not st.finished
    assert st.resources.buffered_packets >= 1
    assert st.resources.buffered_bytes >= 1

    tl.skip("s1", "v", 1)
    tl.close_segment("s1"); tl.close()
    # 排空前，已产出事件还在有界队列里（未释放给下游）
    for _ in tl.events():
        pass
    st2 = tl.status()
    # 结束并被消费后：没有任何未释放资源
    assert st2.finished
    assert st2.resources.buffered_packets == 0
    assert st2.resources.buffered_bytes == 0
    assert st2.resources.in_flight_events == 0
    assert st2.resources.open_segments == 0
    assert st2.resources.open_tracks == 0


def test_status_reports_resources_and_input_liveness():
    tl = make_timeline(queue_cap=64)
    open_seg(tl, tracks=("v", "a"))
    tl.submit(vpkt(0, 0, 0, key=True))
    tl.submit(vpkt(2, 2, 2))
    st = tl.status()
    vstat = next(t for t in st.tracks if t.track_id == "v")
    assert vstat.waiting is True
    assert vstat.can_input is True
    assert vstat.buffered_count == 1
    assert st.resources.buffered_packets >= 1
    tl.close_segment("s1"); tl.close()
    list(tl.events())
    st2 = tl.status()
    assert st2.finished is True
    assert st2.resources.buffered_packets == 0
    assert all(not t.can_input for t in st2.tracks)


def test_backpressure_blocks_producer_until_consumer_pulls():
    import queue as queue_mod
    import threading

    tl = make_timeline(queue_cap=3)
    open_seg(tl, tracks=("v",))
    tl.submit(vpkt(0, 0, 0, key=True))
    tl.submit(vpkt(1, 1, 1))
    tl.submit(vpkt(2, 2, 2))
    # 第 4 个包必然被阻塞（队列容量 3，已产生 3 个 packet 事件）
    entered = threading.Event()
    done = threading.Event()

    def producer():
        entered.set()
        tl.submit(vpkt(3, 3, 3))
        done.set()

    th = threading.Thread(target=producer)
    th.start()
    entered.wait(2)
    assert not done.wait(0.3)  # 慢消费者没取，生产者被背压
    for _ in range(4):
        tl.pull(block=True, timeout=1)
    assert done.wait(2)
    th.join()


def test_cancel_unblocks_producer_and_consumer():
    import threading

    # 1) 取消唤醒被背压阻塞的生产者
    tl = make_timeline(queue_cap=3)
    open_seg(tl, tracks=("v",))
    for i in range(3):
        tl.submit(vpkt(i, i, i, key=(i == 0)))

    producer_started = threading.Event()
    producer_waiting = threading.Event()
    producer_err = []

    def producer():
        producer_started.set()
        try:
            tl.submit(vpkt(99, 99, 99))
        except Cancelled:
            producer_err.append("cancelled")

    tl.on_backpressure_wait = producer_waiting.set
    th = threading.Thread(target=producer)
    th.start()
    assert producer_started.wait(2)
    assert producer_waiting.wait(2)  # 生产者确实已经被慢消费者顶住
    tl.cancel()
    th.join(2)
    assert not th.is_alive()
    assert producer_err == ["cancelled"]

    # 2) 取消唤醒阻塞的消费者：已排队的事件先被取完，随后立即 Cancelled，
    #    不会留下永远阻塞的 pull
    drained = 0
    try:
        while True:
            tl.pull(block=True, timeout=1)
            drained += 1
    except Cancelled:
        pass
    assert drained == 3  # 取消前已产出的三个包事件仍然可被消费


# --------------------------------------------------------------------------- #
# 多分段 / 多轨道
# --------------------------------------------------------------------------- #


def test_audio_video_share_segment_origin_with_different_time_bases():
    tl = make_timeline(queue_cap=64)
    # 同一段，视频 1/25，音频 1/48000，起点都是 0
    tl.open_segment("s1", Fraction(0), [
        InputTrackSpec("v", "v"), InputTrackSpec("a", "a"),
    ])
    tl.submit(vpkt(0, 0, 0, key=True, track="v"))
    tl.submit(InputPacket(
        segment_id="s1", track_id="a", capture_seq=0,
        pts=0, dts=0, duration=1024, time_base=Fraction(1, 48000),
        payload=b"a0", keyframe=True,
    ))
    tl.close_segment("s1"); tl.close()
    evs = list(tl.events())
    v = next(e for e in evs if e.output_track_id == "v" and e.kind == "packet")
    a = next(e for e in evs if e.output_track_id == "a" and e.kind == "packet")
    assert v.output_dts == 0 and a.output_dts == 0
    assert v.correction.output_time_base == Fraction(1, 12800)
    assert a.correction.output_time_base == Fraction(1, 48000)
    assert v.correction.dts.mapped_seconds == a.correction.dts.mapped_seconds == 0


def test_multi_segment_concatenation_on_common_timeline():
    tl = make_timeline(queue_cap=64)
    # 两段各 8 帧，第二段起点 8/25
    tl.open_segment("s1", Fraction(0), [InputTrackSpec("v", "v")])
    tl.open_segment("s2", Fraction(8, 25), [InputTrackSpec("v", "v")])
    for sid, base in (("s1", -2), ("s2", 6)):
        pass
    for i in range(8):
        tl.submit(vpkt(i, i, -2 + i, key=(i == 0), sid="s1"))
    # 第二段采集器从自己的 0 重新计时（dts 也重新从 -2 开始）
    for i in range(8):
        tl.submit(vpkt(i, i, -2 + i, key=(i == 0), sid="s2"))
    tl.close_segment("s1")
    tl.close_segment("s2")
    tl.close()
    evs = [e for e in tl.events() if e.kind == "packet"]
    assert len(evs) == 16
    # 公共时间线上 dts 严格单调，步长 512，第二段负值已被起点抬升
    dts = [e.output_dts for e in evs]
    assert dts == [-1024 + 512 * i for i in range(16)]
    assert evs[8].segment_id == "s2"


def test_second_segment_packets_do_not_overtake_first_segment():
    tl = make_timeline(queue_cap=64)
    tl.open_segment("s1", 0, [InputTrackSpec("v", "v")])
    tl.open_segment("s2", Fraction(100), [InputTrackSpec("v", "v")])
    # 第二段数据全到了，但第一段还缺一个包
    for i in range(4):
        tl.submit(vpkt(i, i, i, sid="s2", key=(i == 0)))
    tl.submit(vpkt(0, 0, 0, sid="s1", key=True))
    tl.submit(vpkt(2, 2, 2, sid="s1"))
    evs = drain(tl)
    # s1/0 已经交付；s2 的包不能越过 s1 的空洞
    early = [(e.segment_id, e.capture_seq)
             for e in evs if e.kind == "packet"]
    assert early == [("s1", 0)]
    tl.skip("s1", "v", 1)
    tl.close_segment("s1"); tl.close_segment("s2"); tl.close()
    evs2 = [e for e in tl.events() if e.kind == "packet"]
    ids = [(e.segment_id, e.capture_seq) for e in evs2]
    # 门控保证空洞补齐后 s1/2 一定先于全部 s2
    assert ids == [("s1", 2),
                   ("s2", 0), ("s2", 1), ("s2", 2), ("s2", 3)]


def test_pull_nonblocking_and_end_of_timeline():
    tl = make_timeline()
    open_seg(tl, tracks=("v",))
    assert tl.pull(block=False) is None
    tl.close_segment("s1"); tl.close()
    # 排空已产出的全部事件；finished 后非阻塞 pull 直接抛 EndOfTimeline
    try:
        while True:
            tl.pull(block=False)
    except EndOfTimeline:
        pass
    with pytest.raises(EndOfTimeline):
        tl.pull(block=True)


def test_quantization_residual_is_visible_on_packet():
    tl = Timeline(output_queue_capacity=16)
    tl.register_output_track(OutputTrackSpec(
        VID, Fraction(1, 10), rounding=Rounding.NEAREST_AWAY_FROM_ZERO))
    tl.open_segment("s1", 0, [InputTrackSpec("v", VID)])
    # 1 个 tick 的 1/3 秒 = 1/3 秒，在输出 1/10 下不可精确表示
    tl.submit(InputPacket(
        segment_id="s1", track_id="v", capture_seq=0,
        pts=1, dts=1, duration=1,
        time_base=Fraction(1, 3), payload=b"x", keyframe=True))
    tl.close_segment("s1"); tl.close()
    ev = next(e for e in tl.events() if e.kind == "packet")
    assert ev.correction.status is TimeStatus.QUANTIZED
    # 残差 = 3/10 - 1/3，每个字段精确公开；residual_seconds 是上界
    assert ev.correction.pts.residual == Fraction(-1, 30)
    assert ev.residual_seconds == Fraction(1, 30)
    assert ev.output_pts == 3
