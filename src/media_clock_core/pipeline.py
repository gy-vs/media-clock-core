"""异步增量管道：把同步内核包在 asyncio 的背压边界里。

锁模型（关键：不存在持锁等待对方的环）：

- 一把 ``_serial`` 串行锁包住“推进内核 -> drain -> 入队”整个临界区，因此
  drain 出来的顺序与入队顺序严格一致，多个写入协程并发也不会把交付顺序
  打乱（这是真实压测出的并发顺序 bug）。
- 只有“等待预算”发生在串行锁之外：缺口未补、预算占满时释放锁，去等
  ``_progress``；填补缺口的写入者因此能进入临界区推进并唤醒等待者。
- 事件队列是有界 ``asyncio.Queue``：满时 ``await put`` 由消费者 ``get``
  自动唤醒，慢封装器背压沿链路传到读取侧。消费者只依赖队列，从不获取
  串行锁，所以写入者在满队列上等待时消费者一定能取件，不会死锁。
- 取消同时解除预算等待与入队等待。无后台线程/定时器；超时冲刷由调用方
  显式驱动（可自行包一个 task）。
"""
from __future__ import annotations

import asyncio
from typing import Optional

from .kernel import MediaClockKernel
from .types import (
    FeedOutcome,
    InputPacket,
    SegmentSpec,
    WaitingBudget,
)

_END = object()
_CANCELLED = object()


class PipelineCancelled(Exception):
    """管道已取消后仍试图输入。"""


class ClockPipeline:
    def __init__(
        self,
        budget: Optional[WaitingBudget] = None,
        *,
        max_pending_events: int = 256,
        kernel: Optional[MediaClockKernel] = None,
    ):
        self._kernel = kernel or MediaClockKernel(budget)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_pending_events)
        self._cancelled = asyncio.Event()
        self._serial = asyncio.Lock()
        # 预算/状态推进信号：每次内核推进都 set；预算等待者 clear 后在锁外等。
        self._progress = asyncio.Event()
        self._progress.set()
        self._closed = False

    @property
    def kernel(self) -> MediaClockKernel:
        return self._kernel

    # -- 配置（同步、立即） -------------------------------------------------

    def register_track(self, spec) -> None:
        self._kernel.register_track(spec)

    def open_segment(self, spec: SegmentSpec) -> None:
        self._kernel.open_segment(spec)

    # -- 输入（异步、带背压） ----------------------------------------------

    async def feed(self, packet: InputPacket) -> FeedOutcome:
        await self._serial.acquire()
        held = True
        try:
            while True:
                if self._cancelled.is_set():
                    raise PipelineCancelled()
                result = self._kernel.feed(packet)
                if result.outcome is FeedOutcome.BUDGET_EXCEEDED:
                    # 在锁内 clear（此时没有别的写入者能推进），随后释放锁
                    # 去等；填补缺口/确认跳过的写入者推进时会 set 它。
                    self._progress.clear()
                    self._serial.release()
                    held = False
                    await self._wait_progress()
                    await self._serial.acquire()
                    held = True
                    continue
                self._progress.set()
                await self._enqueue(self._kernel.drain())
                return result.outcome
        finally:
            if held:
                self._serial.release()

    async def feed_all(self, packets) -> list[FeedOutcome]:
        return [await self.feed(p) for p in packets]

    async def _advance(self, fn) -> None:
        """在串行锁内执行一次内核推进并把新事件入队（保持交付顺序）。"""
        async with self._serial:
            if self._cancelled.is_set():
                return
            fn()
            self._progress.set()
            await self._enqueue(self._kernel.drain())

    async def confirm_skip(self, segment_id, track_id, start, end, *, reason="confirmed"):
        await self._advance(
            lambda: self._kernel.confirm_skip(segment_id, track_id, start, end, reason=reason)
        )

    async def finish_track(self, segment_id, track_id):
        await self._advance(lambda: self._kernel.finish_track(segment_id, track_id))

    async def finish_segment(self, segment_id):
        await self._advance(lambda: self._kernel.finish_segment(segment_id))

    async def sweep_timeouts(self, now=None) -> int:
        n_box = [0]

        def do():
            n_box[0] = self._kernel.sweep_timeouts(now)

        await self._advance(do)
        return n_box[0]

    # -- 输出 ---------------------------------------------------------------

    async def events(self):
        """异步迭代已确定事件；aclose/cancel 放入哨兵后结束。

        只依赖队列本身，绝不获取串行锁，消费者总能解除慢队列背压。
        """
        while True:
            item = await self._queue.get()
            if item is _END or item is _CANCELLED:
                return
            yield item

    # -- 生命周期 -----------------------------------------------------------

    async def cancel(self) -> None:
        """读取取消：内核释放占住资源，解除预算等待，并终止消费循环。"""
        async with self._serial:
            if self._cancelled.is_set():
                return
            self._cancelled.set()
            self._kernel.cancel()
            self._progress.set()
        # 队列若满，等消费者取走已确定事件后放入终止哨兵（消费者不被锁阻塞）
        while True:
            try:
                self._queue.put_nowait(_CANCELLED)
                return
            except asyncio.QueueFull:
                await asyncio.sleep(0)

    async def aclose(self) -> None:
        """优雅结束：冲刷所有仍在等待的内容，消费者收完后结束。"""
        async with self._serial:
            if self._closed:
                return
            self._kernel.close()
            self._closed = True
            self._progress.set()
            events = self._kernel.drain()
            await self._enqueue(events)
            await self._enqueue([_END])

    def status(self):
        return self._kernel.status()

    # -- 内部 ---------------------------------------------------------------

    async def _wait_progress(self) -> None:
        prog = asyncio.create_task(self._progress.wait())
        canc = asyncio.create_task(self._cancelled.wait())
        try:
            await asyncio.wait({prog, canc}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            prog.cancel()
            canc.cancel()

    async def _enqueue(self, events: list) -> None:
        """把事件顺序放入有界队列；满时挂起等慢消费者，对取消敏感。

        调用方持有 ``_serial``；消费者不获取它，因此这里等待不会死锁。
        """
        for ev in events:
            if self._cancelled.is_set():
                try:
                    self._queue.put_nowait(ev)
                except asyncio.QueueFull:
                    # 放不下：按序退回内核 drain 队列头部，保持可观察不丢失
                    self._kernel._events.appendleft(ev)  # type: ignore[attr-defined]
                continue
            put_task = asyncio.create_task(self._queue.put(ev))
            cancel_task = asyncio.create_task(self._cancelled.wait())
            try:
                await asyncio.wait(
                    {put_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                if not cancel_task.done():
                    cancel_task.cancel()
            if self._cancelled.is_set() and not put_task.done():
                put_task.cancel()
                self._kernel._events.appendleft(ev)  # type: ignore[attr-defined]
