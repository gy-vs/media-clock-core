"""Optional PyAV integration.

内核本身不依赖 PyAV；这里只做两件机械转换，让真实的读取器和封装器
能直接接上：

* :func:`input_packet_from_av` —— 把读取器解封装/编码得到的
  ``av.Packet`` 快照成内核的 :class:`~media_clock_core.InputPacket`
  （载荷拷贝一次成 ``bytes``，之后内核与读取器生命周期解耦）；
* :func:`apply_to_av_packet` —— 把内核输出事件里的修正时间写回
  一个要交给 muxer 的 ``av.Packet``。**只改时间字段，不动载荷**：
  解码依赖（B 帧重排关系）完全保留。

封装时的比特流格式（如 H.264 Annex-B ↔ MP4 length-prefixed）由 PyAV /
FFmpeg muxer 按其 ``codec_tag`` 自行处理，不属于时间内核的职责。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Optional

from .events import (
    DeliveredPacket,
    InputPacket,
    PayloadUsability,
)


def packet_time_base(packet: Any) -> Optional[Fraction]:
    """Return a packet's time base as an exact :class:`Fraction`."""
    tb = packet.time_base
    if tb is None:
        return None
    return Fraction(tb.numerator, tb.denominator)


def input_packet_from_av(
    packet: Any,
    segment_id: Any,
    track_id: Any,
    capture_seq: int,
) -> InputPacket:
    """Snapshot an ``av.Packet`` coming from a reader/encoder.

    Args:
        capture_seq: 读取器提供的采集序号。注意 B 帧场景下必须使用
            *解码顺序* 的序号（即 demux 顺序），不是 PTS 顺序。
    """
    payload = bytes(packet)
    is_corrupt = bool(getattr(packet, "is_corrupt", False))
    usability = (
        PayloadUsability.CORRUPT if is_corrupt else PayloadUsability.USABLE
    )
    return InputPacket(
        segment_id=segment_id,
        track_id=track_id,
        capture_seq=capture_seq,
        pts=packet.pts,
        dts=packet.dts,
        duration=packet.duration if packet.duration else None,
        time_base=packet_time_base(packet),
        payload=payload,
        keyframe=bool(packet.is_keyframe),
        payload_usability=usability,
    )


def apply_to_av_packet(av_packet: Any, event: DeliveredPacket) -> Any:
    """Write corrected timing onto *av_packet*; payload is never touched.

    用法（下游真正的封装器）::

        out_packet = av.Packet(reader_packet)   # 载荷随 av.Packet 复制
        out_packet.stream = output_stream
        apply_to_av_packet(out_packet, delivered_event)
        container.mux(out_packet)

    负的 ``output_dts`` 会原样保留——FFmpeg 的 MP4 muxer 会整体平移并写
    edit list，删掉负值就会破坏开头 B 帧的解码关系。
    """
    av_packet.pts = event.output_pts
    av_packet.dts = event.output_dts
    if event.output_duration is not None:
        av_packet.duration = event.output_duration
    tb = event.correction.output_time_base
    av_packet.time_base = Fraction(tb.numerator, tb.denominator)
    av_packet.is_keyframe = event.keyframe
    return av_packet
