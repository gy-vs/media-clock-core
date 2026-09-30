"""同步进程内增量内核。

典型生命周期::

    k = MediaClockKernel(WaitingBudget(max_packets=512))
    k.register_track(TrackSpec("v", MediaKind.VIDEO, Fraction(1,25), Fraction(1,12800)))
    k.open_segment(SegmentSpec("seg0", Fraction(0)))
    k.feed(packet)              # 读取器乱序调用也没关系
    for ev in k.drain(): ...    # 已确定的交付 / 确认跳过事件
    k.confirm_skip(...)         # 明确放弃某个前缘缺口
    k.finish_track("seg0","v") # 结束：等待部分冲刷为确定结果
    k.close()                   # 释放全部占住资源

内核不缓存完整文件：in-order 包在 ``feed`` 内立即转为事件，只有真正等待
缺口的包才占住预算。
"""
from __future__ import annotations

import time
from collections import deque
from fractions import Fraction
from typing import Iterable, Optional

from .buffer import TrackBuffer
from .types import (
    FeedOutcome,
    FeedResult,
    InputPacket,
    KernelEvent,
    KernelStatus,
    ResourceStatus,
    SegmentId,
    SegmentSpec,
    TrackId,
    TrackSpec,
    TrackState,
    WaitingBudget,
)


class MediaClockKernel:
    def __init__(
        self,
        budget: Optional[WaitingBudget] = None,
        *,
        clock=time.monotonic,
    ):
        self._budget = budget or WaitingBudget()
        self._clock = clock
        self._tracks: dict[TrackId, TrackSpec] = {}
        self._segments: dict[SegmentId, SegmentSpec] = {}
        # (segment_id, track_id) -> TrackBuffer
        self._buffers: dict[tuple[SegmentId, TrackId], TrackBuffer] = {}
        self._events: deque[KernelEvent] = deque()
        self._cancelled = False
        self._closed = False
        # 每个仍开放 (段,轨) 当前前缘缺口的等待起点（墙钟）
        self._waiting_since: dict[tuple[SegmentId, TrackId], float] = {}

    # -- 配置 ---------------------------------------------------------------

    def register_track(self, spec: TrackSpec) -> None:
        if spec.track_id in self._tracks:
            raise ValueError(f"track already registered: {spec.track_id}")
        if spec.default_time_base <= 0 or spec.output_time_base <= 0:
            raise ValueError("time bases must be positive Fractions")
        self._tracks[spec.track_id] = spec

    def open_segment(self, spec: SegmentSpec) -> None:
        """声明段身份与输出起点。段边界只能来自这里，不能被推断出来。"""
        if self._cancelled or self._closed:
            raise RuntimeError("kernel cancelled or closed")
        if spec.segment_id in self._segments:
            raise ValueError(f"segment already open: {spec.segment_id}")
        self._segments[spec.segment_id] = spec
        for track_id, tspec in self._tracks.items():
            offset = spec.track_capture_offsets.get(track_id, Fraction(0))
            self._buffers[(spec.segment_id, track_id)] = TrackBuffer(
                spec.segment_id, tspec, spec.output_start, offset
            )
            self._waiting_since[(spec.segment_id, track_id)] = self._clock()

    # -- 输入 ---------------------------------------------------------------

    def feed(self, packet: InputPacket) -> FeedResult:
        if self._cancelled or self._closed:
            return FeedResult(FeedOutcome.CANCELLED)
        buf = self._buffers.get((packet.segment_id, packet.track_id))
        if buf is None:
            return FeedResult(FeedOutcome.UNKNOWN_TARGET)
        if buf.state is not TrackState.OPEN:
            return FeedResult(FeedOutcome.FINISHED)
        if packet.time_base is not None and packet.time_base <= 0:
            raise ValueError("packet time_base must be positive")

        # 只有“会真正占住预算”的乱序包才受预算约束；in-order 包即刻交付。
        if packet.seq > buf.next_seq:
            held_pkts = sum(b.held_count for b in self._buffers.values())
            held_bytes = sum(b.held_bytes for b in self._buffers.values())
            if (
                held_pkts + 1 > self._budget.max_packets
                or held_bytes + max(packet.payload_bytes, 0) > self._budget.max_bytes
            ):
                return FeedResult(FeedOutcome.BUDGET_EXCEEDED)

        before = len(self._events)
        events, _held = buf.feed(packet)
        self._emit(events)
        delivered = len(self._events) - before
        self._touch_wait((packet.segment_id, packet.track_id))
        return FeedResult(FeedOutcome.ACCEPTED, delivered=delivered)

    def feed_many(self, packets: Iterable[InputPacket]) -> list[FeedResult]:
        return [self.feed(p) for p in packets]

    def confirm_skip(
        self,
        segment_id: SegmentId,
        track_id: TrackId,
        start: int,
        end: int,
        *,
        reason: str = "confirmed",
    ) -> None:
        """确认跳过当前前缘缺口 ``[start, end)``。

        只能从当前确定的前缘开始；范围里若已有到达并占住的包，拒绝重叠，
        防止调用方把数据静默删掉。
        """
        buf = self._require_open(segment_id, track_id)
        events = buf.confirm_skip(start, end, reason)
        self._emit(events)
        self._touch_wait((segment_id, track_id))

    def finish_track(self, segment_id: SegmentId, track_id: TrackId) -> None:
        """调用方明确结束一条 (段,轨)：仍在等待的部分转为可处理结果。"""
        buf = self._buffers.get((segment_id, track_id))
        if buf is None:
            raise KeyError((segment_id, track_id))
        events = buf.finish()
        self._emit(events)
        self._waiting_since.pop((segment_id, track_id), None)

    def finish_segment(self, segment_id: SegmentId) -> None:
        for track_id in self._tracks:
            key = (segment_id, track_id)
            if key in self._buffers and self._buffers[key].state is TrackState.OPEN:
                self.finish_track(segment_id, track_id)

    def sweep_timeouts(self, now: Optional[float] = None) -> int:
        """扫描超过等待预算时长的缺口，将其确认为 ``reason='timeout'``。

        超时不等于悄悄删数据：超时缺口同样产出 :class:`RangeSkipped` 并附带
        解码阴影影响。返回本次冲刷的 (段,轨) 数量。
        """
        if self._budget.default_timeout is None:
            return 0
        now = self._clock() if now is None else now
        flushed = 0
        for key, buf in list(self._buffers.items()):
            if buf.state is not TrackState.OPEN:
                continue
            since = self._waiting_since.get(key)
            st = buf.status()
            if since is None or not st.waiting_ranges:
                continue
            if now - since >= self._budget.default_timeout:
                gap = st.waiting_ranges[0]
                events = buf.confirm_skip(gap.start_seq, gap.end_seq, "timeout")
                self._emit(events)
                self._touch_wait(key)
                flushed += 1
        return flushed

    # -- 输出 ---------------------------------------------------------------

    def drain(self, max_events: Optional[int] = None) -> list[KernelEvent]:
        out: list[KernelEvent] = []
        while self._events and (max_events is None or len(out) < max_events):
            out.append(self._events.popleft())
        return out

    def drain_one(self) -> Optional[KernelEvent]:
        return self._events.popleft() if self._events else None

    @property
    def events_pending(self) -> int:
        return len(self._events)

    # -- 生命周期 -----------------------------------------------------------

    def cancel(self) -> None:
        """读取取消：沿链路通知，释放所有占住包；后续 feed 返回 CANCELLED。"""
        self._cancelled = True
        for buf in self._buffers.values():
            buf.cancel()
        self._waiting_since.clear()

    def close(self) -> None:
        """关闭内核。仍开放的轨会先 finish，保证不会留下永不结束的等待。"""
        if self._closed:
            return
        for key, buf in list(self._buffers.items()):
            if buf.state is TrackState.OPEN:
                self._emit(buf.finish())
        self._closed = True
        self._waiting_since.clear()

    # -- 状态 ---------------------------------------------------------------

    def status(self) -> KernelStatus:
        track_status = tuple(b.status() for b in self._buffers.values())
        held_packets = sum(t.holding for t in track_status)
        held_bytes = sum(t.holding_bytes for t in track_status)
        open_tracks = sum(1 for t in track_status if t.can_input)
        res = ResourceStatus(
            held_packets=held_packets,
            held_bytes=held_bytes,
            pending_events=len(self._events),
            open_tracks=open_tracks,
            timeout_waiters=sum(1 for t in track_status if t.waiting_ranges),
            cancelled=self._cancelled,
        )
        return KernelStatus(tracks=track_status, resources=res)

    # -- 内部 ---------------------------------------------------------------

    def _require_open(self, segment_id, track_id) -> TrackBuffer:
        buf = self._buffers.get((segment_id, track_id))
        if buf is None:
            raise KeyError((segment_id, track_id))
        if buf.state is not TrackState.OPEN:
            raise RuntimeError(f"{segment_id}/{track_id} is {buf.state.value}")
        return buf

    def _emit(self, events: list[KernelEvent]) -> None:
        self._events.extend(events)

    def _touch_wait(self, key) -> None:
        buf = self._buffers[key]
        st = buf.status()
        if st.waiting_ranges:
            self._waiting_since.setdefault(key, self._clock())
        else:
            self._waiting_since.pop(key, None)
