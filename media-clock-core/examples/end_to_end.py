"""端到端示例：两段真实采集 -> media-clock-core -> 一个 MP4。

运行：python examples/end_to_end.py
依赖：pip install av

演示：
* libx264 真实编出带 B 帧（负 DTS / PTS 重排）的两段视频；
* 模拟网络乱序与一段中的丢包（调用方显式 skip）；
* 内核按采集序号整理、平移到公共时间线、公开残差与解码可用性；
* 真正的 PyAV 封装器（AVCC->AnnexB BSF 在封装边界）写出 MP4；
* 重新解码，打印逐帧画面时间。
"""

from __future__ import annotations

import io
import random
from fractions import Fraction

import av

from media_clock_core import (
    InputPacket,
    InputTrackSpec,
    OutputTrackSpec,
    Timeline,
)
from media_clock_core.av_adapter import apply_to_av_packet, input_packet_from_av


def frame(i: int, w=160, h=120) -> "av.VideoFrame":
    f = av.VideoFrame(w, h, "yuv420p")
    f.planes[0].update(bytes((x + y * 3 + i * 17) & 0xFF
                             for y in range(h) for x in range(w)))
    u = bytes([128]) * (w // 2 * h // 2)
    f.planes[1].update(u)
    f.planes[2].update(u)
    return f


def capture_segment(n: int, offset: int):
    """真实采集设备一段：编码进 MP4，再作为读取器 demux 出采集包。"""
    buf = io.BytesIO()
    c = av.open(buf, "w", format="mp4")
    s = c.add_stream("libx264", rate=Fraction(25),
                     options={"bframes": "2", "x264-params": "b-adapt=0"})
    s.width, s.height, s.pix_fmt = 160, 120, "yuv420p"
    s.time_base = Fraction(1, 12800)
    for i in range(n):
        for p in s.encode(frame(offset + i)):
            c.mux(p)
    for p in s.encode(None):
        c.mux(p)
    extradata = s.codec_context.extradata
    c.close()

    reader = av.open(io.BytesIO(buf.getvalue()))
    st = reader.streams.video[0]
    raw = [(bytes(p), p.pts, p.dts, p.duration,
            Fraction(p.time_base.numerator, p.time_base.denominator),
            p.is_keyframe)
           for p in reader.demux(st) if p.dts is not None]
    return reader, st, raw, extradata


def main() -> None:
    n1, n2 = 10, 10
    r1, st1, raw1, extra1 = capture_segment(n1, 0)
    r2, st2, raw2, _ = capture_segment(n2, 100)

    def to_input(raw, sid):
        out = []
        for seq, (data, pts, dts, dur, cap_tb, key) in enumerate(raw):
            # 设备原生计时基准 1/25
            p = av.Packet(data)
            p.pts = int(Fraction(pts) * cap_tb * 25)
            p.dts = int(Fraction(dts) * cap_tb * 25)
            p.duration = int(Fraction(dur) * cap_tb * 25)
            p.time_base = Fraction(1, 25)
            p.is_keyframe = key
            out.append(input_packet_from_av(p, sid, "v", seq))
        return out

    seg1 = to_input(raw1, "s1")
    seg2 = to_input(raw2, "s2")

    OUT_TB = Fraction(1, 12800)
    tl = Timeline(default_wait_budget_packets=64, output_queue_capacity=256)
    tl.register_output_track(OutputTrackSpec("v", OUT_TB))
    tl.open_segment("s1", Fraction(0), [InputTrackSpec("v", "v")])
    tl.open_segment("s2", Fraction(n1, 25), [InputTrackSpec("v", "v")])

    # 输出封装器
    obuf = io.BytesIO()
    out = av.open(obuf, "w", format="mp4")
    vs = out.add_stream("h264", rate=Fraction(25))
    vs.width, vs.height, vs.pix_fmt = 160, 120, "yuv420p"
    vs.time_base = OUT_TB
    bsf1 = av.BitStreamFilterContext("h264_mp4toannexb", st1, vs)
    bsf2 = av.BitStreamFilterContext("h264_mp4toannexb", st2, vs)
    bsfs = {"s1": bsf1, "s2": bsf2}

    def feed(packets, sid, drop=frozenset()):
        indexed = [p for p in packets if p.capture_seq not in drop]
        random.Random(1 if sid == "s1" else 2).shuffle(indexed)
        first_missing = min(drop) if drop else None
        for ip in indexed:
            tl.submit(ip)
        if drop:  # 调用方确认缺口
            tl.skip(sid, "v", first_missing, max(drop))

    feed(seg1, "s1", drop={5})          # 第一段丢 seq5 并确认
    feed(seg2, "s2")                    # 第二段乱序但不丢
    tl.close_segment("s1")
    tl.close_segment("s2")
    tl.close()

    delivered = degraded = recovery = skipped = 0
    max_residual = Fraction(0)
    for ev in tl.events():
        if ev.kind == "skipped":
            skipped += 1
            print(f"[skip] {ev.segment_id} seq "
                  f"{ev.first_seq}..{ev.last_seq} "
                  f"recovery_seq={ev.next_keyframe_seq}")
        elif ev.kind == "packet":
            delivered += 1
            if ev.decode_availability.name == "DEGRADED_AFTER_GAP":
                degraded += 1
            if ev.decode_availability.name == "RECOVERY_POINT":
                recovery += 1
            max_residual = max(max_residual, ev.residual_seconds)
            in_pkt = av.Packet(ev.payload)
            in_pkt.is_keyframe = ev.keyframe
            for q in bsfs[ev.segment_id].filter(in_pkt):
                q.stream = vs
                apply_to_av_packet(q, ev)
                out.mux(q)
    for bsf in bsfs.values():
        for q in bsf.filter(None):
            q.stream = vs
            out.mux(q)
    out.close()
    r1.close(); r2.close()

    print(f"delivered={delivered} skipped_ranges={skipped} "
          f"degraded_packets={degraded} recovery_points={recovery}")
    print(f"max timing residual = {float(max_residual):.3e} s")

    rd = av.open(io.BytesIO(obuf.getvalue()))
    times = []
    for f in rd.decode(video=0):
        tb = Fraction(f.time_base.numerator, f.time_base.denominator)
        times.append(float(Fraction(f.pts) * tb))
    print(f"decoded {len(times)} frames; first={times[0]:.3f}s "
          f"last={times[-1]:.3f}s")


if __name__ == "__main__":
    main()
