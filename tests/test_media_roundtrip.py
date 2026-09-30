"""真实媒体往返验证：PyAV 编码 -> 网络乱序 -> media-clock-core -> PyAV 封装 -> PyAV 解码看时间。

这些测试不允许只对整数排序宣称成功：最终判断依据是解码器吐出来的画面
出现时间与画面内容（帧序号），以及多段/小数帧率下的精确无漂移。
"""
from fractions import Fraction

import pytest

from media_clock_core import (
    FeedOutcome,
    InputPacket,
    MediaKind,
    PacketDelivered,
    RangeSkipped,
    SegmentSpec,
    TrackSpec,
    Usability,
    WaitingBudget,
    MediaClockKernel,
)

from .media_helpers import (
    Mp4MuxerSink,
    decode_audio_times,
    decode_video_times,
    encode_aac_segment,
    encode_h264_segment,
)


def feed_capture(kernel, cap, segment_id, track_id, shuffle=None, drop_seqs=None):
    """把一段编码采集包喂进内核；可按网络乱序/缺失模拟。

    返回 (key 映射)：每个交付事件找回它对应的原始 av.Packet。
    """
    drop_seqs = drop_seqs or set()
    order = list(range(len(cap.packets)))
    if shuffle is not None:
        order = shuffle(order)
    lookup = {}
    for i in order:
        if i in drop_seqs:
            continue
        p = cap.packets[i]
        # 编码器 flush 包没有显示/解码时间，跳过不喂
        if p.pts is None or p.dts is None:
            continue
        pkt = InputPacket(
            segment_id=segment_id,
            track_id=track_id,
            seq=i,
            pts=p.pts,
            dts=p.dts,
            duration=1,  # 每帧一个采集 tick（编码器 duration 为 0，由我们显式给）
            time_base=cap.time_base,
            is_keyframe=bool(p.is_keyframe),
            payload=p,
            payload_bytes=len(bytes(p)),
        )
        lookup[(segment_id, track_id, i)] = p
        r = kernel.feed(pkt)
        assert r.outcome in (FeedOutcome.ACCEPTED, FeedOutcome.BUDGET_EXCEEDED), r.outcome
        assert r.outcome is FeedOutcome.ACCEPTED
    return lookup


def mux_events(kernel, specs, extradata, lookup, *, drop_broken=True):
    sink = Mp4MuxerSink(specs, extradata)

    def key(ev):
        return (ev.segment_id, ev.track_id, ev.seq)

    for ev in kernel.drain():
        sink.handle(ev, lookup, key)
    return sink


# ---------------------------------------------------------------------------
# 1. 单段、B 帧、网络乱序：解码画面必须仍每 1/25 秒出现，顺序与内容正确
# ---------------------------------------------------------------------------


def test_bframe_h264_shuffled_network_roundtrip_25fps():
    cap = encode_h264_segment(nframes=12, fps=(25, 1), bframes=3)

    # 编码输出具备题目要求的本质：显示时间在解码顺序上重排 0,4,2,1,...，
    # 而解码时间开头是负的 -2,-1,0,1。
    assert [p.pts for p in cap.packets[:4]] == [0, 4, 2, 1]
    assert [p.dts for p in cap.packets[:4]] == [-2, -1, 0, 1]

    out_tb = Fraction(1, 12800)

    # 模拟读取器乱序调用：确定性洗牌，保持可复现
    import random
    rng = random.Random(42)

    def shuffle(order):
        o = order[:]
        rng.shuffle(o)
        return o

    kernel = MediaClockKernel(WaitingBudget(max_packets=64))
    kernel.register_track(TrackSpec("v", MediaKind.VIDEO, cap.time_base, out_tb))
    kernel.open_segment(SegmentSpec("seg0", Fraction(0)))
    lookup = feed_capture(kernel, cap, "seg0", "v", shuffle=shuffle)
    kernel.finish_track("seg0", "v")
    events = kernel.drain()

    # 内核交付次序严格按采集序号递增，绝不是按 PTS（PTS 是 0,3,1,2,...）
    delivered = [e for e in events if isinstance(e, PacketDelivered)]
    assert [e.seq for e in delivered] == sorted(e.seq for e in delivered)
    # 负 DTS 与 B 帧显示重排（解码序 pts 0,4,2,1,...）在修正后的 ticks 上保留
    assert delivered[0].timing.dts.ticks == -2 * 512
    assert [e.timing.pts.ticks for e in delivered[:4]] == [0, 2048, 1024, 512]

    spec = TrackSpec("v", MediaKind.VIDEO, cap.time_base, out_tb)
    sink = Mp4MuxerSink({"v": spec}, {"v": cap.extradata})
    def key(ev): return (ev.segment_id, ev.track_id, ev.seq)
    for ev in events:
        sink.handle(ev, lookup, key)
    data = sink.finish()

    frames = decode_video_times(data)
    assert len(frames) == 12
    secs = [f[0] for f in frames]
    assert secs[0] == 0
    for a, b in zip(secs, secs[1:]):
        assert b - a == Fraction(1, 25)
    # 解码后的显示顺序里，画面序号必须 0..11（B 帧重排序正确，不按解码序乱显）
    assert [f[3] for f in frames] == list(range(12))


# ---------------------------------------------------------------------------
# 2. 30000/1001：精确有理换算，多帧后画面时间无漂移
# ---------------------------------------------------------------------------


def test_30000_1001_roundtrip_exact_no_drift():
    cap = encode_h264_segment(nframes=30, fps=(30000, 1001), bframes=3)
    out_tb = Fraction(1, 90000)
    kernel = MediaClockKernel()
    kernel.register_track(TrackSpec("v", MediaKind.VIDEO, cap.time_base, out_tb))
    kernel.open_segment(SegmentSpec("seg0", Fraction(0)))
    lookup = feed_capture(kernel, cap, "seg0", "v")
    kernel.finish_track("seg0", "v")
    spec = TrackSpec("v", MediaKind.VIDEO, cap.time_base, out_tb)
    sink = mux_events(kernel, {"v": spec}, {"v": cap.extradata}, lookup)
    data = sink.finish()

    frames = decode_video_times(data)
    assert len(frames) == 30
    # 第 i 帧精确时间 i*1001/30000 秒；在 1/90000 下 tick = i*3003
    for i, (sec, tb, pts, idx) in enumerate(frames):
        assert pts == i * 3003, (i, pts)
        assert sec == Fraction(i * 1001, 30000)
    assert [f[3] for f in frames] == list(range(30))


# ---------------------------------------------------------------------------
# 3. 多段接续：两段素材接到同一输出时间线，画面连续无重叠无浮点缝隙
# ---------------------------------------------------------------------------


def test_two_segments_continuous_timeline_roundtrip():
    n0, n1 = 10, 7
    capA = encode_h264_segment(nframes=n0, fps=(25, 1), bframes=3, start_index=0)
    capB = encode_h264_segment(nframes=n1, fps=(25, 1), bframes=3, start_index=0)

    out_tb = Fraction(1, 12800)
    kernel = MediaClockKernel()
    kernel.register_track(TrackSpec("v", MediaKind.VIDEO, Fraction(1, 25), out_tb))
    # 段 B 起点：A 的最后一帧显示时刻 + 一帧时长（外部确认的公共时间线）
    start_b = Fraction(n0, 25)
    kernel.open_segment(SegmentSpec("A", Fraction(0)))
    kernel.open_segment(SegmentSpec("B", start_b))
    lookup = {}
    lookup.update(feed_capture(kernel, capA, "A", "v"))
    lookup.update(feed_capture(kernel, capB, "B", "v"))
    kernel.finish_segment("A")
    kernel.finish_segment("B")
    spec = TrackSpec("v", MediaKind.VIDEO, Fraction(1, 25), out_tb)
    sink = mux_events(kernel, {"v": spec}, {"v": capA.extradata}, lookup)
    data = sink.finish()

    frames = decode_video_times(data)
    assert len(frames) == n0 + n1
    # 全部帧严格每 1/25 秒连续——跨段处也不例外（无浮点缝隙）
    for i, (sec, tb, pts, idx) in enumerate(frames):
        assert sec == Fraction(i, 25), (i, sec)
    # 画面内容：前 n0 帧是段 A 的 0..n0-1，之后是段 B 重新计时的 0..n1-1
    assert [f[3] for f in frames] == list(range(n0)) + list(range(n1))
    # MP4 内部流时间基准确实是 muxer 选的 1/12800，内核喂的是已换算 ticks
    assert all(f[1] == Fraction(1, 12800) for f in frames)


# ---------------------------------------------------------------------------
# 4. 丢包：确认跳过 -> 阴影包标 DEPENDENCY_BROKEN -> 下游决定丢弃；
#    下一个关键帧之后解码恢复正确，影响随事件流交付而非日志计数
# ---------------------------------------------------------------------------


def test_confirmed_gap_marks_decoding_shadow_until_keyframe():
    # 单 GOP、第一帧为关键帧；丢掉中间若干非关键帧
    cap = encode_h264_segment(nframes=12, fps=(25, 1), bframes=1)
    out_tb = Fraction(1, 12800)
    kernel = MediaClockKernel()
    kernel.register_track(TrackSpec("v", MediaKind.VIDEO, cap.time_base, out_tb))
    kernel.open_segment(SegmentSpec("seg0", Fraction(0)))

    # 找到第二个关键帧序号（本夹具整段一个 GOP，可能只有首帧是关键帧）。
    # 为了可恢复，手工在 seq=8 之后造一个关键帧不可行（编码器已定），
    # 因此这里验证“单 GOP 丢包 -> 到段末全部阴影”，另测恢复用多段。
    key_seqs = [i for i, p in enumerate(cap.packets) if p.is_keyframe and p.pts is not None]
    assert key_seqs[0] == 0

    # 缺包 3,4：先喂其余，到达前缘后确认跳过
    missing = {3, 4}
    lookup = feed_capture(kernel, cap, "seg0", "v", drop_seqs=missing)
    # 前缘推进到 3 卡住；明确确认跳过 [3,5)
    kernel.confirm_skip("seg0", "v", 3, 5, reason="network-loss")
    kernel.finish_track("seg0", "v")

    events = kernel.drain()
    skip = next(e for e in events if isinstance(e, RangeSkipped))
    assert (skip.start_seq, skip.end_seq) == (3, 5)
    delivered = [e for e in events if isinstance(e, PacketDelivered)]
    broken = {e.seq for e in delivered if e.usability is Usability.DEPENDENCY_BROKEN}
    intact = {e.seq for e in delivered if e.usability is Usability.INTACT}
    # 阴影影响随包给出：5 及之后（单 GOP 无新关键帧）全部 broken，0..2 intact
    assert intact == {0, 1, 2}
    assert 5 in broken
    assert delivered[-1].seq in broken


# ---------------------------------------------------------------------------
# 5. 多段丢包恢复：段内丢包只影响该 GOP；新段从新关键帧起恢复 INTACT
# ---------------------------------------------------------------------------


def test_gap_recovery_at_new_segment_keyframe():
    capA = encode_h264_segment(nframes=6, fps=(25, 1), bframes=1, start_index=0)
    capB = encode_h264_segment(nframes=6, fps=(25, 1), bframes=1, start_index=0)
    out_tb = Fraction(1, 12800)
    kernel = MediaClockKernel()
    kernel.register_track(TrackSpec("v", MediaKind.VIDEO, Fraction(1, 25), out_tb))
    kernel.open_segment(SegmentSpec("A", Fraction(0)))
    kernel.open_segment(SegmentSpec("B", Fraction(6, 25)))
    lookup = {}
    lookup.update(feed_capture(kernel, capA, "A", "v", drop_seqs={2, 3}))
    kernel.confirm_skip("A", "v", 2, 4, reason="loss")
    lookup.update(feed_capture(kernel, capB, "B", "v"))
    kernel.finish_segment("A")
    kernel.finish_segment("B")
    events = kernel.drain()
    by_seg = {}
    for e in events:
        if isinstance(e, PacketDelivered):
            by_seg.setdefault(e.segment_id, []).append(e)
    # B 段首帧是新关键帧，全部 INTACT——新段边界即恢复点
    assert all(e.usability is Usability.INTACT for e in by_seg["B"])
    # A 段缺口之后受影响
    assert any(e.usability is Usability.DEPENDENCY_BROKEN for e in by_seg["A"])


# ---------------------------------------------------------------------------
# 6. 音频 + 视频同一时间线：不同计时单位，段锚点共享，封装后 A/V 起点对齐
# ---------------------------------------------------------------------------


def test_audio_video_segment_alignment_different_timebases():
    vcap = encode_h264_segment(nframes=8, fps=(25, 1), bframes=2)
    acap = encode_aac_segment(npackets=4, sample_rate=48000)
    out_tb_v = Fraction(1, 12800)
    out_tb_a = Fraction(1, 48000)
    kernel = MediaClockKernel()
    kernel.register_track(TrackSpec("v", MediaKind.VIDEO, Fraction(1, 25), out_tb_v))
    kernel.register_track(
        TrackSpec("a", MediaKind.AUDIO, Fraction(1, 48000), out_tb_a)
    )
    kernel.open_segment(SegmentSpec("seg0", Fraction(0)))
    lookup = {}
    lookup.update(feed_capture(kernel, vcap, "seg0", "v"))
    # 音频包
    for i, p in enumerate(acap.packets):
        if p.pts is None:
            continue
        lookup[("seg0", "a", i)] = p
        kernel.feed(InputPacket(
            "seg0", "a", i, p.pts, p.dts, p.duration or 1024,
            Fraction(1, 48000), True, p, len(bytes(p)),
        ))
    kernel.finish_segment("seg0")
    specs = {
        "v": TrackSpec("v", MediaKind.VIDEO, Fraction(1, 25), out_tb_v),
        "a": TrackSpec("a", MediaKind.AUDIO, Fraction(1, 48000), out_tb_a),
    }
    sink = Mp4MuxerSink(specs, {"v": vcap.extradata, "a": acap.extradata})
    def key(ev): return (ev.segment_id, ev.track_id, ev.seq)
    for ev in kernel.drain():
        sink.handle(ev, lookup, key)
    data = sink.finish()

    vframes = decode_video_times(data)
    atimes = decode_audio_times(data)
    assert len(vframes) == 8
    assert atimes[0] == 0  # 音频从同一公共原点起
    # 音频包间隔 1024/48000 秒
    for a, b in zip(atimes, atimes[1:]):
        assert b - a == Fraction(1024, 48000)
    # 视频起点也为 0：两轨共享段锚点，没有因为分别处理而错位
    assert vframes[0][0] == 0
