"""真实 PyAV 编解码夹具：供媒体往返测试使用。

负责：用 libx264 编出带 B 帧的 H264（采集时间基准 1/25 或 1001/30000）、
用 AAC 编音频；解码后取出每帧的“画面时间”和画面里写入的帧序号。

本模块是测试设施，不属于 media_clock_core 运行时代码。
"""
from __future__ import annotations

import fractions
import io
from dataclasses import dataclass
from typing import Optional

import av
import numpy as np

W, H = 160, 120


@dataclass
class RawCapture:
    """一段采集素材的原始编码包（按采集/解码顺序）与编码器信息。"""

    packets: list  # av.Packet，时间戳为采集时间基准 ticks
    time_base: fractions.Fraction
    framerate: fractions.Fraction
    extradata: bytes
    frame_indices: list[int]  # 每个编码包对应的显示帧序（按解码顺序）


_CODE_LEVELS = (0, 85, 170, 255)  # 间距 >=85，CRF23 量化不会跨级


def _paint(index: int) -> np.ndarray:
    """生成一帧：用 3 个通道各 2 位（4 个相距很远的灰阶）编码 6 位序号。

    H264 有损，1 位级差异会被量化抹掉；4 个等级间隔 85，解码后最近邻量化
    可以在 CRF23 下无歧义恢复 0..63 的序号。
    """
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for ch in range(3):
        img[:, :, ch] = _CODE_LEVELS[(index >> (2 * ch)) & 0b11]
    return img


def encode_h264_segment(
    nframes: int = 10,
    fps=(25, 1),
    bframes: int = 3,
    start_index: int = 0,
) -> RawCapture:
    framerate = fractions.Fraction(*fps) if isinstance(fps, tuple) else fractions.Fraction(fps)
    tb = fractions.Fraction(framerate.denominator, framerate.numerator)
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height = W, H
    enc.framerate = framerate
    enc.time_base = tb
    enc.options = {
        "preset": "veryfast",
        "crf": "18",
        "bframes": str(bframes),
        # b-adapt=0：固定的 B 帧放置，图案内容变化不会改变重排结构，
        # 让“时间正确性”测试不依赖编码器的自适应决策。
        "x264-params": "b-adapt=0:keyint=1000:min-keyint=1000:scenecut=0",
    }
    enc.pix_fmt = "yuv420p"
    packets = []
    indices = []
    for i in range(nframes):
        idx = start_index + i
        frame = av.VideoFrame.from_ndarray(_paint(idx), format="rgb24").reformat(
            format="yuv420p"
        )
        frame.time_base = tb
        frame.pts = i  # 采集计时从 0 开始（每段设备重新计时）
        for p in enc.encode(frame):
            packets.append(p)
            indices.append(idx)
    for p in enc.encode(None):
        packets.append(p)
        indices.append(-1)  # flush 包（无显示帧）
    # 去掉编码器 flush 的空包尾（PyAV 有时给 None/None 的尾包）
    while packets and packets[-1].pts is None and packets[-1].dts is None:
        packets.pop()
        indices.pop()
    return RawCapture(packets, tb, framerate, bytes(enc.extradata or b""), indices)


def encode_aac_segment(npackets: int = 8, sample_rate: int = 48000) -> "RawCapture":
    tb = fractions.Fraction(1, sample_rate)
    enc = av.CodecContext.create("aac", "w")
    enc.sample_rate = sample_rate
    enc.layout = av.AudioLayout("mono")
    enc.format = av.AudioFormat("fltp")
    enc.time_base = tb
    enc.open()
    silence = np.zeros((1, 1024), dtype=np.float32)
    packets = []
    for i in range(npackets):
        # from_ndarray 明确让帧持有该数组，避免 to_ndarray() 临时缓冲写空
        frame = av.AudioFrame.from_ndarray(silence.copy(), format="fltp", layout="mono")
        frame.sample_rate = sample_rate
        frame.time_base = tb
        frame.pts = i * 1024
        for p in enc.encode(frame):
            packets.append(p)
    for p in enc.encode(None):
        packets.append(p)
    return RawCapture(packets, tb, fractions.Fraction(sample_rate, 1),
                      bytes(enc.extradata or b""), list(range(npackets)))


# ---------------------------------------------------------------------------
# 封装：消费内核交付的事件，真正写进 MP4；不改载荷
# ---------------------------------------------------------------------------


class Mp4MuxerSink:
    """把 PacketDelivered 事件按轨 mux 进 MP4。

    封装策略（这是下游策略，不是内核的一部分）：
    - 时间用内核已经换算好的整数 ticks + 内核输出时间基准；
    - 载荷按原字节重建 av.Packet，DTS/PTS/关键帧标记原样；
    - DEPENDENCY_BROKEN 的包默认丢弃（下游决定），返回被丢弃清单以便核对。
    """

    def __init__(self, track_specs: dict, extradata: dict, fps_hint=None):
        self._buf = io.BytesIO()
        self._container = av.open(self._buf, "w", format="mp4")
        self._streams = {}
        for tid, spec in track_specs.items():
            if spec.kind.value == "video":
                st = self._container.add_stream("h264")
                st.width, st.height, st.pix_fmt = W, H, "yuv420p"
                st.time_base = spec.output_time_base
                if extradata.get(tid):
                    st.extradata = extradata[tid]
            else:
                st = self._container.add_stream(
                    "aac", rate=48000,
                    layout=av.AudioLayout("mono"),
                    format=av.AudioFormat("fltp"),
                )
                st.time_base = spec.output_time_base
                if extradata.get(tid):
                    st.extradata = extradata[tid]
            self._streams[tid] = st
        self.dropped_broken = []
        self.muxed = []

    @property
    def streams(self):
        return self._streams

    def mux_delivered(self, event, src_packet) -> bool:
        """重建并封装一个已交付事件；DEPENDENCY_BROKEN 的包按下游策略丢弃。

        返回是否真正写入容器。时间用内核换算好的整数 ticks，载荷按原字节
        重建（不改一个字节），PTS/DTS/关键帧标记按修正值/原标记。
        """
        from media_clock_core.types import Usability

        if event.usability is Usability.DEPENDENCY_BROKEN:
            self.dropped_broken.append((event.track_id, event.seq))
            return False
        st = self._streams[event.track_id]
        out = av.Packet(bytes(src_packet))
        out.stream = st
        t = event.timing
        out.pts = t.pts.ticks if t.pts else None
        out.dts = t.dts.ticks if t.dts else None
        out.duration = t.duration.ticks if t.duration else src_packet.duration
        out.is_keyframe = src_packet.is_keyframe
        out.time_base = event.timing.output_time_base
        self._container.mux(out)
        self.muxed.append((event.track_id, event.seq, out.pts, out.dts))
        return True

    def handle(self, event, raw_packet_by_key, key):
        from media_clock_core.types import PacketDelivered

        if not isinstance(event, PacketDelivered):
            return
        self.mux_delivered(event, raw_packet_by_key[key(event)])

    def finish(self) -> bytes:
        self._container.close()
        return self._buf.getvalue()


def decode_video_times(data: bytes, stream_index: int = 0):
    """解码 MP4，返回 [(显示秒 Fraction, 帧时间基准, frame_pts, 识别出的画面序号)]。"""
    container = av.open(io.BytesIO(data))
    stream = container.streams.video[stream_index]
    result = []
    for frame in container.decode(stream):
        rgb = frame.to_ndarray(format="rgb24")
        idx = _read_index(rgb)
        result.append((
            fractions.Fraction(frame.pts, 1) * frame.time_base,
            frame.time_base,
            frame.pts,
            idx,
        ))
    container.close()
    return result


def _read_index(rgb: np.ndarray) -> int:
    """从三个通道的中值灰阶最近邻反解 6 位序号（见 :func:`_paint`）。"""
    idx = 0
    for ch in range(3):
        median = int(np.median(rgb[:, :, ch]))
        level = min(range(4), key=lambda j: abs(median - _CODE_LEVELS[j]))
        idx |= level << (2 * ch)
    return idx


def decode_audio_times(data: bytes):
    container = av.open(io.BytesIO(data))
    stream = container.streams.audio[0]
    times = []
    for frame in container.decode(stream):
        times.append(fractions.Fraction(frame.pts, 1) * frame.time_base)
    container.close()
    return times
