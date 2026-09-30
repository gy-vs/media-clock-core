"""异步背压、慢消费者、取消传播与优雅结束测试。"""
import asyncio
from fractions import Fraction

import pytest

from media_clock_core import (
    ClockPipeline,
    FeedOutcome,
    InputPacket,
    MediaKind,
    PacketDelivered,
    PipelineCancelled,
    RangeSkipped,
    SegmentSpec,
    TrackSpec,
    WaitingBudget,
)

V_TB = Fraction(1, 25)
OUT_TB = Fraction(1, 12800)


def pkt(seq, track="v", seg="s0"):
    return InputPacket(seg, track, seq, seq, seq - 2, 1, V_TB, seq == 0,
                       f"{seq}".encode(), 100)


async def build(budget=None, qsize=4):
    p = ClockPipeline(budget or WaitingBudget(max_packets=100),
                      max_pending_events=qsize)
    p.register_track(TrackSpec("v", MediaKind.VIDEO, V_TB, OUT_TB))
    p.open_segment(SegmentSpec("s0", Fraction(0)))
    return p


def test_reordered_events_reach_slow_consumer():
    async def main():
        p = await build()
        got = []

        async def consume():
            async for ev in p.events():
                if isinstance(ev, PacketDelivered):
                    got.append(ev.seq)

        task = asyncio.create_task(consume())
        # 乱序到达
        for s in (3, 1, 0, 2):
            await p.feed(pkt(s))
        await p.aclose()
        await asyncio.wait_for(task, timeout=2)
        assert got == [0, 1, 2, 3]

    asyncio.run(main())


def test_budget_backpressure_blocks_reader_then_releases():
    async def main():
        # 预算只允许占住 1 个包；先占住 seq=1，再喂 seq=2 必须挂起。
        # 用足够大的事件队列，把“预算背压”和“慢消费者队列背压”隔离开。
        p = await build(WaitingBudget(max_packets=1), qsize=64)
        drained = []

        async def consume():
            async for ev in p.events():
                if isinstance(ev, PacketDelivered):
                    drained.append(ev.seq)

        consumer = asyncio.create_task(consume())
        await p.feed(pkt(1))
        fed = asyncio.Event()

        async def blocked_feed():
            await p.feed(pkt(2))
            fed.set()

        t = asyncio.create_task(blocked_feed())
        await asyncio.sleep(0.05)
        assert not fed.is_set()  # 读取侧被预算背压
        await p.feed(pkt(0))     # 前缘到达，1 交付，预算释放，2 可被接收
        await asyncio.wait_for(t, timeout=2)
        assert fed.is_set()
        await p.aclose()
        await asyncio.wait_for(consumer, timeout=2)
        assert drained == [0, 1, 2]

    asyncio.run(main())


def test_slow_muxer_backpressures_event_queue():
    async def main():
        p = await build(WaitingBudget(max_packets=1000), qsize=2)
        consumed = []

        async def slow_consume():
            async for ev in p.events():
                if isinstance(ev, PacketDelivered):
                    consumed.append(ev.seq)
                    await asyncio.sleep(0.02)  # 慢封装器

        task = asyncio.create_task(slow_consume())
        # 连续 in-order 喂入；队列只有 2，feed 必须等待而非无限缓存
        for s in range(10):
            await asyncio.wait_for(p.feed(pkt(s)), timeout=2)
        await p.aclose()
        await asyncio.wait_for(task, timeout=5)
        assert consumed == list(range(10))

    asyncio.run(main())


def test_cancel_unblocks_reader_and_frees_resources():
    async def main():
        p = await build(WaitingBudget(max_packets=1))
        await p.feed(pkt(1))

        async def blocked():
            with pytest.raises(PipelineCancelled):
                await p.feed(pkt(2))

        t = asyncio.create_task(blocked())
        await asyncio.sleep(0.05)
        await p.cancel()
        await asyncio.wait_for(t, timeout=2)
        st = p.status()
        assert st.resources.held_packets == 0
        assert st.resources.cancelled is True

    asyncio.run(main())


def test_aclose_flushes_waiting_range():
    async def main():
        p = await build()
        got_skips = []

        async def consume():
            async for ev in p.events():
                if isinstance(ev, RangeSkipped):
                    got_skips.append((ev.start_seq, ev.end_seq, ev.reason))

        task = asyncio.create_task(consume())
        await p.feed(pkt(0))
        await p.feed(pkt(2))  # 1 缺失
        await p.aclose()      # 冲刷等待
        await asyncio.wait_for(task, timeout=2)
        assert (1, 2, "unfinished-at-end") in got_skips

    asyncio.run(main())
