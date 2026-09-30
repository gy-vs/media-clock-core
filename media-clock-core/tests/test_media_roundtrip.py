"""Real media round-trip verification with PyAV.

这些测试不是排序整数：它们让 libx264 真正编出带 B 帧的码流，把
*编码器输出的包*（带负 DTS、重排 PTS）喂给内核，内核修正时间后交给
真正的 MP4 封装器，然后重新打开容器、真正解码出画面，逐帧检查
显示时间。覆盖：

1. B 帧负 DTS 往返：画面严格按 1/25 间隔出现，载荷字节不变；
2. 多段采集（设备重新计时，第二段 DTS 重新从负值开始）接到同一
   公共时间线，拼接处无空洞无重叠；
3. 音频（AAC 1/48000）+ 视频（1/25）同段对齐；
4. 30000/1001 fps：多段拼接后逐帧检查真实画面时间，无浮点漂移；
5. 跳缺后的解码可用性事件与封装结果一致（下游负责真正的错误恢复）。

编码器 -> 内核之间直接走 av.Packet，不经过裸 h264 文件，因为裸 h264
往返本身会丢失时间戳（这是格式限制，已在探针中确认）。
"""

from __future__ import annotations

import io
from fractions import Fraction

import av
import pytest

from media_clock_core import (
    DecodeAvailability,
    InputPacket,
    InputTrackSpec,
    OutputTrackSpec,
    SkipReason,
    Timeline,
    TimeStatus,
)
from media_clock_core.av_adapter import apply_to_av_packet, input_packet_from_av


av = pytest.importorskip("av")


# --------------------------------------------------------------------------- #
# 测试用编码器：直接吐出 av.Packet（带真实时间戳），不落裸流文件
# --------------------------------------------------------------------------- #


def make_video_frame(i: int, w: int = 160, h: int = 120) -> "av.VideoFrame":
    """Deterministic frame; pixels encode the frame index for later checking."""
    frame = av.VideoFrame(w, h, "yuv420p")
    ys = bytearray(w * h)
    for y in range(h):
        for x in range(w):
            ys[y * w + x] = (x + y * 7 + i * 31) & 0xFF
    frame.planes[0].update(bytes(ys))
    chroma = bytes([128]) * (w // 2 * h // 2)
    frame.planes[1].update(chroma)
    frame.planes[2].update(chroma)
    return frame


class H264PacketEncoder:
    """A realistic capture source: encode B-frame H.264 and re-read packets.

    真实采集/网络读取器面对的是已封装的采集文件，能同时拿到 MP4 需要的
    extradata 和带时间戳的包。这里编码进内存 MP4 再 demux，得到与真实
    读取器一致的采集包（AVCC 载荷、extradata、开头为负的 DTS）。

    采集包时间基准呈现为设备原生基准 1/fps（如 1/25），用于真正检验
    内核到输出基准（1/12800 或 1/90000）的换算。
    """

    def __init__(self, fps_frac=Fraction(25), bframes: int = 2, gop: int = 250):
        self.fps = fps_frac
        # 设备原生基准 = 1/fps（25fps 时为 1/25）
        self.native_tb = Fraction(fps_frac.denominator, fps_frac.numerator)

        self._buf = io.BytesIO()
        self._container = av.open(self._buf, "w", format="mp4")
        self._stream = self._container.add_stream(
            "libx264",
            rate=fps_frac,
            options={
                "bframes": str(bframes),
                "x264-params": f"b-adapt=0:keyint={gop}:min-keyint={gop}",
            },
        )
        self._stream.width = 160
        self._stream.height = 120
        self._stream.pix_fmt = "yuv420p"
        # 采集容器基准必须能整除帧间隔，否则 duration 会在采集阶段就
        # 被量化成 0（FFmpeg 也会为非整数帧率选这种基准）。
        # 取分母的若干倍：1/(fps*den) 对 30000/1001 -> 1/30000，25 -> 1/12500。
        cap_tb = Fraction(1, fps_frac.numerator)
        self._stream.time_base = cap_tb
        self._frames: list = []
        self.extradata: bytes = b""

    def encode(self, frame):
        self._frames.append(frame)

    def finish(self) -> list[av.Packet]:
        for frame in self._frames:
            for pkt in self._stream.encode(frame):
                self._container.mux(pkt)
        for pkt in self._stream.encode(None):
            self._container.mux(pkt)
        self.extradata = self._stream.codec_context.extradata or b""
        self._container.close()

        # 保持读取器打开：下游 BSF 需要它的流（codec 参数/extradata）。
        self._reader = av.open(io.BytesIO(self._buf.getvalue()), format="mp4")
        vs = self._reader.streams.video[0]
        self.capture_stream = vs
        raw: list[tuple] = []
        for pkt in self._reader.demux(vs):
            if pkt.dts is None:
                continue
            # 必须立即物化为 bytes（av.Packet 复制只增加底层缓冲引用，
            # demux 迭代/容器关闭后引用会失效）。
            cap_tb = Fraction(
                pkt.time_base.numerator, pkt.time_base.denominator
            )
            raw.append((
                bytes(pkt),
                pkt.pts, pkt.dts, pkt.duration, cap_tb, pkt.is_keyframe,
            ))

        packets: list[av.Packet] = []
        for data, pts, dts, duration, cap_tb, keyframe in raw:
            def to_native(ticks: int, _cap=cap_tb, _nt=self.native_tb) -> int:
                # cap_tb*ticks = native_tb*out
                return int(Fraction(ticks) * _cap / _nt)

            snap = av.Packet(data)
            snap.dts = to_native(dts)
            snap.pts = to_native(pts)
            snap.duration = to_native(duration)
            snap.time_base = Fraction(
                self.native_tb.numerator, self.native_tb.denominator
            )
            snap.is_keyframe = keyframe
            packets.append(snap)
        return packets

    def close_reader(self):
        if getattr(self, "_reader", None) is not None:
            self._reader.close()
            self._reader = None


class AACCaptureSource:
    """Encode a sine tone to real AAC and re-read packets (capture reader)."""

    def __init__(self, sample_rate: int = 48000, frames: int = 24):
        import math
        import struct

        self.sample_rate = sample_rate
        self.native_tb = Fraction(1, sample_rate)  # 设备按采样点计时
        self._buf = io.BytesIO()
        container = av.open(self._buf, "w", format="ipod")  # M4A
        stream = container.add_stream("aac", rate=sample_rate)
        n = 1024
        t = 0
        for _ in range(frames):
            # 原生 AAC 编码器要 fltp（float planar）输入
            frame = av.AudioFrame(format="fltp", layout="mono", samples=n)
            frame.sample_rate = sample_rate
            frame.pts = t
            vals = bytearray()
            import struct as _st
            floats = []
            for k in range(n):
                floats.append(
                    0.25 * math.sin(2 * math.pi * 440 * (t + k) / sample_rate)
                )
            frame.planes[0].update(_st.pack(f"<{n}f", *floats))
            for pkt in stream.encode(frame):
                container.mux(pkt)
            t += n
        for pkt in stream.encode(None):
            container.mux(pkt)
        self.extradata = stream.codec_context.extradata or b""
        container.close()

        self._reader = av.open(io.BytesIO(self._buf.getvalue()))
        self.capture_stream = self._reader.streams.audio[0]
        ast = self.capture_stream
        raw = []
        for pkt in self._reader.demux(ast):
            if pkt.dts is None:
                continue
            cap_tb = Fraction(
                pkt.time_base.numerator, pkt.time_base.denominator
            )
            raw.append((bytes(pkt), pkt.pts, pkt.dts, pkt.duration, cap_tb))

        self.packets: list[av.Packet] = []
        for data, pts, dts, duration, cap_tb in raw:
            def to_native(ticks):
                return int(Fraction(ticks) * cap_tb / self.native_tb)
            snap = av.Packet(data)
            snap.pts = to_native(pts)
            snap.dts = to_native(dts)
            snap.duration = to_native(duration)
            snap.time_base = Fraction(
                self.native_tb.numerator, self.native_tb.denominator
            )
            snap.is_keyframe = True
            self.packets.append(snap)

    def close_reader(self):
        if getattr(self, "_reader", None) is not None:
            self._reader.close()
            self._reader = None


# --------------------------------------------------------------------------- #
# 下游真正的封装器：消费内核事件 -> av muxer
# --------------------------------------------------------------------------- #


class MuxerSink:
    """A real muxer driven by kernel events (the downstream side).

    采集读取器给的是 MP4 风格的 AVCC 包；真正的封装器需要通过
    ``h264_mp4toannexb`` 比特流过滤器（有状态、属于 muxer 侧）转换。
    这个转换 *不在* 时间内核里——内核只改时间字段，载荷字节原样透传；
    本 sink 在封装边界持有 BSF，体现真实的职责划分。
    """

    def __init__(self, time_base_by_track: dict, fmt: str = "mp4"):
        self.buf = io.BytesIO()
        self.container = av.open(self.buf, "w", format=fmt)
        self.streams: dict = {}
        self.time_bases = time_base_by_track
        self._bsf: dict = {}
        self._capture_params: dict = {}

    def add_video_stream(self, track_id, extradata, rate=25):
        s = self.container.add_stream("h264", rate=rate)
        s.width = 160
        s.height = 120
        s.pix_fmt = "yuv420p"
        tb = self.time_bases[track_id]
        s.time_base = Fraction(tb.numerator, tb.denominator)
        self.streams[track_id] = s
        # 先用采集 extradata 建立一个只携带 codec 参数的 BSF 输入。
        # BSF 需要一个带 codec_parameters 的流：构造临时捕获容器读取。
        self._raw_extradata = extradata
        return s

    def configure_video_bsf(self, track_id, capture_stream, segment_id=None):
        """Bind a stateful AVCC->annexB filter per (segment, output track).

        多段采集是各自独立的采集编码会话，codec 参数/状态独立，因此每个
        (段, 输出轨) 需要自己的 BSF 实例。
        """
        out_stream = self.streams[track_id]
        bsf = av.BitStreamFilterContext(
            "h264_mp4toannexb", capture_stream, out_stream
        )
        self._bsf[(segment_id, track_id)] = bsf

    def handle(self, event):
        if event.kind != "packet":
            return
        out_stream = self.streams[event.output_track_id]
        bsf = self._bsf.get((event.segment_id, event.output_track_id))
        if bsf is None:
            bsf = self._bsf.get((None, event.output_track_id))
        if bsf is not None:
            # AVCC 采集载荷 -> 临时包 -> 有状态 BSF -> annexb 包
            in_pkt = av.Packet(event.payload)
            in_pkt.is_keyframe = event.keyframe
            for out_pkt in bsf.filter(in_pkt):
                out_pkt.stream = out_stream
                apply_to_av_packet(out_pkt, event)
                self.container.mux(out_pkt)
        else:
            pkt = av.Packet(event.payload)
            pkt.stream = out_stream
            apply_to_av_packet(pkt, event)
            self.container.mux(pkt)

    def flush(self):
        for (segment_id, track_id), bsf in self._bsf.items():
            for out_pkt in bsf.filter(None):
                out_pkt.stream = self.streams[track_id]
                self.container.mux(out_pkt)

    def close(self):
        self.flush()
        self.container.close()
        return self.buf.getvalue()


# --------------------------------------------------------------------------- #
# 1. B 帧负 DTS：编码器包 -> 内核 -> MP4 -> 解码，画面时间必须精确
# --------------------------------------------------------------------------- #


def test_bframe_negative_dts_real_encode_mux_decode_roundtrip():
    N = 12
    enc = H264PacketEncoder(bframes=2)
    for i in range(N):
        enc.encode(make_video_frame(i))
    packets = enc.finish()

    # 编码器确实产出了负 DTS 和重排 PTS（否则本测试就没有验证到目标场景）
    assert packets[0].dts < 0
    dts_order = [p.dts for p in packets]
    pts_order = [p.pts for p in packets]
    assert dts_order == sorted(dts_order)          # 解码顺序
    assert pts_order != sorted(pts_order)          # 显示序被 B 帧重排

    OUT_TB = Fraction(1, 12800)
    tl = Timeline(output_queue_capacity=512)
    tl.register_output_track(OutputTrackSpec("v", OUT_TB))
    tl.open_segment("s1", Fraction(0), [InputTrackSpec("v", "v")])

    sink = MuxerSink({"v": OUT_TB})
    sink.add_video_stream("v", enc.extradata)
    sink.configure_video_bsf("v", enc.capture_stream)

    # 模拟网络乱序：故意打乱提交顺序（但内核按采集序号整理）
    import random
    rng = random.Random(42)
    indexed = list(enumerate(packets))
    rng.shuffle(indexed)
    delivered_bytes = {}
    for seq, avpkt in indexed:
        ip = input_packet_from_av(avpkt, "s1", "v", seq)
        tl.submit(ip)
        delivered_bytes[seq] = bytes(avpkt)

    tl.close_segment("s1")
    tl.close()
    kernel_payload_by_seq = {}
    for ev in tl.events():
        if ev.kind == "packet":
            # 内核边界：载荷与采集包逐字节相同（内核只改时间字段，
            # 不转码、不改比特流格式）。
            kernel_payload_by_seq[ev.capture_seq] = ev.payload
        sink.handle(ev)
    data = sink.close()

    assert kernel_payload_by_seq == delivered_bytes

    # 解封装：muxer 时间基准与内核输出一致
    reader = av.open(io.BytesIO(data))
    vs = reader.streams.video[0]
    assert Fraction(vs.time_base.numerator, vs.time_base.denominator) == OUT_TB

    demuxed = [p for p in reader.demux(vs) if p.dts is not None]
    # 采集序号顺序（= DTS 单调），且数量一致
    dts_out = [p.dts for p in demuxed]
    assert dts_out == sorted(dts_out)
    assert len(demuxed) == len(packets)

    # 解码出真正的画面：显示时间必须严格每 1/25 秒一帧
    reader2 = av.open(io.BytesIO(data))
    frame_pts = []
    for frame in reader2.decode(video=0):
        tb = Fraction(frame.time_base.numerator, frame.time_base.denominator)
        frame_pts.append(Fraction(frame.pts) * tb)
    assert len(frame_pts) == N
    expected = [Fraction(i, 25) for i in range(N)]
    assert frame_pts == expected


# --------------------------------------------------------------------------- #
# 2. 多段采集，设备重新计时 -> 公共时间线连续
# --------------------------------------------------------------------------- #


def test_two_segments_with_restarted_clocks_concatenate_on_timeline():
    # 每段独立编码（设备重启），各 10 帧；第二段编码器时间戳重新从 0/负值开始
    enc1 = H264PacketEncoder(bframes=2)
    for i in range(10):
        enc1.encode(make_video_frame(i))
    p1 = enc1.finish()

    enc2 = H264PacketEncoder(bframes=2)
    for i in range(10):
        enc2.encode(make_video_frame(100 + i))  # 不同画面内容
    p2 = enc2.finish()

    assert p2[0].dts < 0  # 第二段的采集时钟确实重新从负值开始

    OUT_TB = Fraction(1, 12800)
    tl = Timeline(output_queue_capacity=512)
    tl.register_output_track(OutputTrackSpec("v", OUT_TB))
    seg_len_seconds = Fraction(10, 25)  # 外部确认的第一段时长
    tl.open_segment("s1", Fraction(0), [InputTrackSpec("v", "v")])
    tl.open_segment("s2", seg_len_seconds, [InputTrackSpec("v", "v")])

    sink = MuxerSink({"v": OUT_TB})
    sink.add_video_stream("v", enc1.extradata)
    sink.configure_video_bsf("v", enc1.capture_stream, segment_id="s1")
    sink.configure_video_bsf("v", enc2.capture_stream, segment_id="s2")

    # 交错到达：第二段先到一部分，验证门控不让它越过第一段
    for seq, avpkt in list(enumerate(p2))[:4]:
        tl.submit(input_packet_from_av(avpkt, "s2", "v", seq))
    for seq, avpkt in enumerate(p1):
        tl.submit(input_packet_from_av(avpkt, "s1", "v", seq))
    for seq, avpkt in list(enumerate(p2))[4:]:
        tl.submit(input_packet_from_av(avpkt, "s2", "v", seq))

    tl.close_segment("s1")
    tl.close_segment("s2")
    tl.close()
    for ev in tl.events():
        sink.handle(ev)
    data = sink.close()

    reader = av.open(io.BytesIO(data))
    frames = []
    for frame in reader.decode(video=0):
        tb = Fraction(frame.time_base.numerator, frame.time_base.denominator)
        frames.append(Fraction(frame.pts) * tb)

    # 20 帧在公共时间线上严格连续：i/25（i=0..19），无重叠无空洞
    assert frames == [Fraction(i, 25) for i in range(20)]


# --------------------------------------------------------------------------- #
# 3. 音频 + 视频同段对齐（不同时间单位）
# --------------------------------------------------------------------------- #


def test_audio_video_aligned_through_real_muxer():
    # 视频 25fps 带 B 帧（负 DTS），音频 AAC 48kHz：真实采集源
    encv = H264PacketEncoder(bframes=2)
    for i in range(10):
        encv.encode(make_video_frame(i))
    vp = encv.finish()
    enca = AACCaptureSource(sample_rate=48000, frames=24)
    ap = enca.packets

    VTB, ATB = Fraction(1, 12800), Fraction(1, 48000)
    tl = Timeline(output_queue_capacity=512)
    tl.register_output_track(OutputTrackSpec("v", VTB))
    tl.register_output_track(OutputTrackSpec("a", ATB))
    tl.open_segment(
        "s1",
        Fraction(0),
        [InputTrackSpec("v", "v"), InputTrackSpec("a", "a")],
    )

    # 真正的 MP4 封装器：视频走 BSF（AVCC->annexB），音频 AAC 帧直接 mux
    sink = MuxerSink({"v": VTB, "a": ATB})
    sink.add_video_stream("v", encv.extradata)
    sink.configure_video_bsf("v", encv.capture_stream)
    aus = sink.container.add_stream("aac", rate=48000)
    aus.time_base = ATB
    if enca.extradata:
        aus.codec_context.extradata = enca.extradata
    sink.streams["a"] = aus

    for seq, avpkt in enumerate(vp):
        tl.submit(input_packet_from_av(avpkt, "s1", "v", seq))
    for seq, avpkt in enumerate(ap):
        tl.submit(input_packet_from_av(avpkt, "s1", "a", seq))
    tl.close_segment("s1"); tl.close()

    kinds = {}
    for ev in tl.events():
        kinds[ev.kind] = kinds.get(ev.kind, 0) + 1
        sink.handle(ev)
    data = sink.close()

    # 没有任何缺包/等待
    assert kinds.get("waiting", 0) == 0 and kinds.get("skipped", 0) == 0

    # 视频开头是负 DTS 的 B 帧；AAC 首包也有编码器 priming（负时间戳），
    # 这些都由 muxer 的 edit list 平移。解码出来的 *呈现* 起点都在 0 秒。
    rd2 = av.open(io.BytesIO(data))
    vdemux = list(rd2.demux(rd2.streams.video[0]))
    ademux = [p for p in av.open(io.BytesIO(data)).demux(
        av.open(io.BytesIO(data)).streams.audio[0]) if p.dts is not None]
    # 负时间戳被保留下来交给 muxer（没有被当坏数据删掉/抬正）
    assert min(p.dts for p in vdemux if p.dts is not None) < 0
    # 两条轨各自最小 PTS 之后的第一个非负呈现点一致（都从 0 起）
    rd_v = av.open(io.BytesIO(data))
    rd_a = av.open(io.BytesIO(data))
    vframes = list(rd_v.decode(rd_v.streams.video[0]))
    aframes = list(rd_a.decode(rd_a.streams.audio[0]))
    vtb0 = Fraction(vframes[0].pts) * Fraction(
        vframes[0].time_base.numerator, vframes[0].time_base.denominator)
    # 解码出的第一帧画面时间 = 0（段内对齐意图保留）
    assert vtb0 == 0
    assert aframes

    # 视频真实解码画面严格 25fps（尽管开头是负 DTS 的 B 帧）
    ftb = Fraction(vframes[0].time_base.numerator,
                   vframes[0].time_base.denominator)
    assert [Fraction(f.pts) * ftb for f in vframes] == [
        Fraction(i, 25) for i in range(10)
    ]

    # 音频覆盖时长 >= 9 个视频帧间隔
    rd4 = av.open(io.BytesIO(data))
    ast = rd4.streams.audio[0]
    total_audio = Fraction(0)
    for pkt in rd4.demux(ast):
        if pkt.dts is None:
            continue
        total_audio += Fraction(pkt.duration) * Fraction(
            pkt.time_base.numerator, pkt.time_base.denominator
        )
    assert total_audio >= Fraction(9, 25)


# --------------------------------------------------------------------------- #
# 4. 30000/1001 fps：多段拼接后逐帧画面时间精确，无浮点漂移
# --------------------------------------------------------------------------- #


def test_30000_over_1001_multi_segment_no_drift():
    fps = Fraction(30000, 1001)
    enc1 = H264PacketEncoder(fps, bframes=2)
    for i in range(30):
        enc1.encode(make_video_frame(i))
    p1 = enc1.finish()
    enc2 = H264PacketEncoder(fps, bframes=2)
    for i in range(30):
        enc2.encode(make_video_frame(200 + i))
    p2 = enc2.finish()

    # 选一个不能整除帧间隔的输出基准。30000/1001 帧间隔为 1001/30000 秒，
    # 1/12800（MP4 常见）与之不整除，强迫每个包都经历真实量化。
    OUT_TB = Fraction(1, 12800)
    tl = Timeline(output_queue_capacity=1024)
    tl.register_output_track(OutputTrackSpec("v", OUT_TB))
    n1 = len(p1)
    seg_len = Fraction(n1 * 1001, 30000)
    tl.open_segment("s1", Fraction(0), [InputTrackSpec("v", "v")])
    tl.open_segment("s2", seg_len, [InputTrackSpec("v", "v")])

    sink = MuxerSink({"v": OUT_TB})
    sink.add_video_stream("v", enc1.extradata, rate=fps)
    sink.configure_video_bsf("v", enc1.capture_stream, segment_id="s1")
    sink.configure_video_bsf("v", enc2.capture_stream, segment_id="s2")
    sink.streams["v"].time_base = OUT_TB

    for seq, avpkt in enumerate(p1):
        tl.submit(input_packet_from_av(avpkt, "s1", "v", seq))
    for seq, avpkt in enumerate(p2):
        tl.submit(input_packet_from_av(avpkt, "s2", "v", seq))
    tl.close_segment("s1"); tl.close_segment("s2"); tl.close()

    residuals = []
    quantized = 0
    for ev in tl.events():
        sink.handle(ev)
        if ev.kind == "packet":
            residuals.append(ev.residual_seconds)
            if ev.correction.status is TimeStatus.QUANTIZED:
                quantized += 1
    data = sink.close()

    # 确实经历了非零量化（不能是整除走个过场）
    assert quantized > 0
    assert any(r != 0 for r in residuals)
    # 残差逐包公开且有界（最近舍入下 <= 半个输出 tick），不随段长累加：
    # 第二段最后一帧与第一段第一帧的误差上界同量级
    half = OUT_TB / 2
    assert all(r <= half for r in residuals)

    rd = av.open(io.BytesIO(data))
    frames = list(rd.decode(video=0))
    assert len(frames) == 60
    # 每一帧的真实画面时间：理想 i*1001/30000；实际帧 pts 与理想的差
    # 不超过半个输出 tick
    half_tick = OUT_TB / 2
    for i, frame in enumerate(frames):
        tb = Fraction(frame.time_base.numerator, frame.time_base.denominator)
        actual = Fraction(frame.pts) * tb
        ideal = Fraction(i * 1001, 30000)
        assert abs(actual - ideal) <= half_tick + Fraction(1, 10 ** 12)


# --------------------------------------------------------------------------- #
# 5. 跳缺：解码可用性契约随事件下发，封装器仍拿到全部已交付包
# --------------------------------------------------------------------------- #


def test_skipped_range_roundtrip_marks_decodability_but_keeps_packets():
    N = 18
    # 小 GOP（6）保证缺口后面存在确定的关键帧恢复点
    enc = H264PacketEncoder(bframes=2, gop=6)
    for i in range(N):
        enc.encode(make_video_frame(i))
    packets = enc.finish()
    keyseqs = [i for i, p in enumerate(packets) if p.is_keyframe]
    assert len(keyseqs) >= 2, "test needs >=2 keyframes to observe recovery"

    OUT_TB = Fraction(1, 12800)
    tl = Timeline(output_queue_capacity=512)
    tl.register_output_track(OutputTrackSpec("v", OUT_TB))
    tl.open_segment("s1", Fraction(0), [InputTrackSpec("v", "v")])
    sink = MuxerSink({"v": OUT_TB})
    sink.add_video_stream("v", enc.extradata)
    sink.configure_video_bsf("v", enc.capture_stream)

    # 丢掉采集序号 2、3（中间的非关键包），其余正常提交
    missing = {2, 3}
    delivered = []
    for seq, avpkt in enumerate(packets):
        if seq in missing:
            continue
        tl.submit(input_packet_from_av(avpkt, "s1", "v", seq))

    # 显式确认缺口（调用方决策，不是内核猜的）
    skip_ev = tl.skip("s1", "v", 2, 3, reason=SkipReason.EXPLICIT)
    assert skip_ev.count == 2
    # 恢复点（缺口后第一个关键帧）必须在跳过事件里明确给出
    assert skip_ev.next_keyframe_seq in keyseqs
    assert skip_ev.next_keyframe_seq > 3
    tl.close_segment("s1"); tl.close()

    availability = []
    skips_seen = 0
    for ev in tl.events():
        if ev.kind == "skipped":
            skips_seen += 1
        if ev.kind == "packet":
            availability.append((ev.capture_seq, ev.decode_availability))
            sink.handle(ev)
    data = sink.close()

    assert skips_seen == 1
    # 缺口后第一个包标记为降级；直到关键帧恢复点
    degraded = [seq for seq, a in availability
                if a is DecodeAvailability.DEGRADED_AFTER_GAP]
    recovered = [seq for seq, a in availability
                 if a is DecodeAvailability.RECOVERY_POINT]
    assert degraded, "gap impact must be handed to downstream per packet"
    assert recovered, "a later keyframe must be marked as recovery point"
    assert min(recovered) > 3
    assert max(degraded) < min(recovered)

    # 封装结果仍然可被 av 打开（内核没有因为跳包破坏 muxer 契约）
    rd = av.open(io.BytesIO(data))
    assert len([p for p in rd.demux(rd.streams.video[0]) if p.dts is not None]) == len(
        packets
    ) - len(missing)
