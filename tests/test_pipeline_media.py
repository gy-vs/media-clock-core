"""异步管道 + 真实 PyAV 往返：增量乱序喂入 -> 慢封装器 -> 解码验证。

证明内核不是“先缓存整文件再排序”：事件是边喂边产出的，且慢消费者/预算
背压与取消都在真实编码载荷链路上工作。
"""
import asyncio
import random
from fractions import Fraction

from media_clock_core import (
    ClockPipeline,
    InputPacket,
    MediaKind,
    PacketDelivered,
    RangeSkipped,
    SegmentSpec,
    TrackSpec,
    WaitingBudget,
)

from .media_helpers import (
    Mp4MuxerSink,
    decode_video_times,
    encode_h264_segment,
)


def test_pipeline_incremental_real_media_roundtrip():
    async def main():
        cap = encode_h264_segment(nframes=12, fps=(25, 1), bframes=3)
        out_tb = Fraction(1, 12800)
        pipe = ClockPipeline(
            WaitingBudget(max_packets=8, max_bytes=8 * 1024 * 1024),
            max_pending_events=4,  # 小队列：强迫慢封装器产生背压
        )
        spec = TrackSpec("v", MediaKind.VIDEO, cap.time_base, out_tb)
        pipe.register_track(spec)
        pipe.open_segment(SegmentSpec("seg0", Fraction(0)))

        sink = Mp4MuxerSink({"v": spec}, {"v": cap.extradata})
        raw_lookup = {}
        delivered_order = []

        async def muxer_consumer():
            # 模拟慢封装器：边接收边 mux，每包刻意让出事件循环产生背压。
            # 不吞异常——否则消费者静默死掉会让测试表现为“假死”。
            async for ev in pipe.events():
                if isinstance(ev, PacketDelivered):
                    raw = raw_lookup[(ev.segment_id, ev.track_id, ev.seq)]
                    sink.mux_delivered(ev, raw)
                    delivered_order.append(ev.seq)
                await asyncio.sleep(0)  # 慢封装器：每包让出一次

        consumer = asyncio.create_task(muxer_consumer())

        # 读取器乱序喂入（确定性洗牌）
        rng = random.Random(7)
        order = list(range(len(cap.packets)))
        rng.shuffle(order)
        for i in order:
            p = cap.packets[i]
            if p.pts is None or p.dts is None:
                continue
            raw_lookup[("seg0", "v", i)] = p
            await pipe.feed(InputPacket(
                "seg0", "v", i, p.pts, p.dts, 1, cap.time_base,
                bool(p.is_keyframe), p, len(bytes(p)),
            ))
        await pipe.aclose()
        await asyncio.wait_for(consumer, timeout=5)

        # 增量交付也必须严格按采集序号（即使封装器边收边写）
        assert delivered_order == sorted(delivered_order)
        data = sink.finish()
        frames = decode_video_times(data)
        assert len(frames) == 12
        for i, (sec, _tb, _pts, idx) in enumerate(frames):
            assert sec == Fraction(i, 25)
        assert [f[3] for f in frames] == list(range(12))

    asyncio.run(main())


def test_pipeline_cancel_during_gap_frees_real_payloads():
    async def main():
        cap = encode_h264_segment(nframes=8, fps=(25, 1), bframes=2)
        out_tb = Fraction(1, 12800)
        pipe = ClockPipeline(WaitingBudget(max_packets=64), max_pending_events=2)
        spec = TrackSpec("v", MediaKind.VIDEO, cap.time_base, out_tb)
        pipe.register_track(spec)
        pipe.open_segment(SegmentSpec("seg0", Fraction(0)))
        # 只喂乱序包（缺前缘），占住真实编码载荷
        total_bytes = 0
        for i in range(1, 6):
            p = cap.packets[i]
            total_bytes += len(bytes(p))
            await pipe.feed(InputPacket(
                "seg0", "v", i, p.pts, p.dts, 1, cap.time_base,
                bool(p.is_keyframe), p, len(bytes(p)),
            ))
        assert pipe.status().resources.held_packets == 5
        await pipe.cancel()
        st = pipe.status()
        assert st.resources.held_packets == 0
        assert st.resources.cancelled is True

    asyncio.run(main())
