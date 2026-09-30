"""The media timing kernel.

``Timeline`` 是唯一需要的运行时类。它把读取器的采集包整理到一条公共
时间线上，对外只承诺四件事：时间、身份、交付顺序、可用性契约。

线程模型：所有输入输出都经过同一条进程内有界管线，``submit`` 是生产者，
``pull`` / ``events`` 是消费者。输出队列满时生产者阻塞（慢封装器形成
背压），``cancel`` 可以同时唤醒两端。内核不缓存完整媒体——未交付的包
受每个输入轨道的等待预算硬限制。
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Iterator, Optional

from .errors import BudgetExceeded, Cancelled, IdentityError, StateError
from .events import (
    DecodeAvailability,
    DeliveredPacket,
    InputPacket,
    PayloadUsability,
    RecoveryObserved,
    SegmentClosed,
    SegmentStatus,
    ResourceStatus,
    SkipReason,
    SkippedRange,
    StatusSnapshot,
    SubmitOutcome,
    TimeCorrection,
    TimeStatus,
    TrackInputState,
    TrackStatus,
    WaitingRange,
)
from .timebase import Rounding, TimeBaseLike, as_fraction, map_field


class EndOfTimeline(StateError):
    """Raised by a blocking :meth:`Timeline.pull` after every event is done."""


@dataclass(frozen=True)
class OutputTrackSpec:
    """An output (muxer) track.

    Args:
        output_track_id: 不透明身份，下游用它把事件路由到具体 muxer 流。
        time_base: 调用方指定的输出时间基准（如 MP4 视频的 1/12800）。
        rounding: 量化舍入模式，默认与 FFmpeg ``av_rescale_q`` 一致。
    """

    output_track_id: Any
    time_base: TimeBaseLike
    rounding: Rounding = Rounding.NEAREST_AWAY_FROM_ZERO


@dataclass(frozen=True)
class InputTrackSpec:
    """One input track's registration inside a segment.

    Args:
        track_id: 上游采集轨道身份（分段内唯一）。
        output_track_id: 映射到的输出轨道；同一段的音频轨和视频轨各自
            注册、各自有时间基准，但通过同一个 ``output_start`` 对齐，
            分别处理不会丢掉段内对齐意图。
        first_capture_seq: 该分段该轨道 *预期的第一个* 采集序号。
            真实读取器在分段开始时就知道序列计数器起点（RTP seq、
            容器 frame 序号等），因此网络乱序时一个晚到的高序号包不会
            把交付前沿错误地抬高，使更早的包被当成重复/迟到。默认 0。
        wait_budget_packets / wait_budget_bytes: 该轨道允许缓存的未交付
            包上限；None 表示沿用 :class:`Timeline` 级默认值。
    """

    track_id: Any
    output_track_id: Any
    first_capture_seq: int = 0
    wait_budget_packets: Optional[int] = None
    wait_budget_bytes: Optional[int] = None


# --------------------------------------------------------------------------- #
# 内部状态
# --------------------------------------------------------------------------- #


@dataclass
class _OutputTrack:
    spec: OutputTrackSpec
    time_base: Fraction
    reg_index: int
    drained_segments: set[Any] = field(default_factory=set)


@dataclass
class _Segment:
    segment_id: Any
    output_start: Fraction
    order: int
    track_keys: list[tuple[Any, Any]] = field(default_factory=list)
    output_track_ids: set[Any] = field(default_factory=set)
    closed: bool = False
    drained: bool = False


@dataclass
class _InputTrack:
    key: tuple[Any, Any]
    segment_id: Any
    track_id: Any
    output_track_id: Any
    reg_index: int
    budget_packets: Optional[int]
    budget_bytes: Optional[int]
    out_tb: Fraction
    rounding: Rounding
    output_start: Fraction
    first_seq: int = 0

    state: TrackInputState = TrackInputState.OPEN
    buffer: dict[int, InputPacket] = field(default_factory=dict)
    buffered_bytes: int = 0
    frontier: Optional[int] = None
    highest: Optional[int] = None
    observed_any: bool = False

    # 解码可用性状态
    dirty: bool = False
    unrecovered: list[tuple[int, int]] = field(default_factory=list)

    # 已确认跳过的序号区间（append-only，仅用于识别迟到包）
    skip_intervals: list[tuple[int, int]] = field(default_factory=list)

    delivered_count: int = 0
    skipped_count: int = 0

    last_wait_frontier: Optional[int] = None

    @property
    def waiting(self) -> bool:
        # 只有真正观察到过包、且前沿序号缺失而后面有包时才算空洞等待。
        # 分段刚打开、一个包都没来时不算"缺包"，只是尚未开始。
        return (
            self.observed_any
            and self.frontier is not None
            and self.highest is not None
            and self.frontier not in self.buffer
            and self.frontier <= self.highest
        )


# --------------------------------------------------------------------------- #
# Timeline
# --------------------------------------------------------------------------- #


class Timeline:
    """Incremental in-process media timing kernel.

    Args:
        default_wait_budget_packets / default_wait_budget_bytes:
            每个输入轨道未交付包的默认硬预算，``open_segment`` 可按轨覆盖。
        output_queue_capacity: 已产出但未取走事件的上限；满了之后
            ``submit`` 阻塞，慢封装器由此被背压。None 表示不限（不推荐）。
    """

    def __init__(
        self,
        *,
        default_wait_budget_packets: Optional[int] = 128,
        default_wait_budget_bytes: Optional[int] = None,
        output_queue_capacity: Optional[int] = 128,
    ) -> None:
        self._default_budget_packets = default_wait_budget_packets
        self._default_budget_bytes = default_wait_budget_bytes
        self._queue_capacity = output_queue_capacity

        self._cv = threading.Condition()
        self._queue: deque[Any] = deque()

        self._output_tracks: dict[Any, _OutputTrack] = {}
        self._segments: dict[Any, _Segment] = {}
        self._tracks: dict[tuple[Any, Any], _InputTrack] = {}
        self._reg_counter = 0

        self._accepting = True
        self._cancelled = False
        self._finished = False

        #: 可选回调：生产者因为输出队列满（慢消费者）而即将阻塞时触发。
        #: 主要用于测试与装配层的流控遥测；在锁外语义上调用方自行保证线程安全。
        self.on_backpressure_wait: Optional[Callable[[], None]] = None

    # ------------------------------------------------------------------ #
    # 注册
    # ------------------------------------------------------------------ #

    def register_output_track(self, spec: OutputTrackSpec) -> None:
        """Register a muxer-side track with its fixed output time base."""
        with self._cv:
            if spec.output_track_id in self._output_tracks:
                raise IdentityError(
                    f"output track {spec.output_track_id!r} already registered"
                )
            tb = as_fraction(spec.time_base)
            self._output_tracks[spec.output_track_id] = _OutputTrack(
                spec=spec,
                time_base=tb,
                reg_index=self._reg_counter,
            )
            self._reg_counter += 1

    def open_segment(
        self,
        segment_id: Any,
        output_start_seconds: Fraction | int | tuple[int, int],
        tracks: list[InputTrackSpec],
    ) -> None:
        """Open a capture segment at an externally confirmed output start.

        ``output_start_seconds`` 必须来自外部确认（采集设备重启计时的真实
        边界），内核永远不会从时间戳回退里推断分段。同一段的所有轨道共用
        这个起点，因此音频/视频分别处理仍保留段内对齐意图。
        """
        if isinstance(output_start_seconds, tuple):
            start = Fraction(output_start_seconds[0], output_start_seconds[1])
        else:
            start = Fraction(output_start_seconds)
        with self._cv:
            self._check_live_locked()
            if segment_id in self._segments:
                raise IdentityError(f"segment {segment_id!r} already open")
            if not tracks:
                raise ValueError("a segment needs at least one input track")
            seen: set[Any] = set()
            for t in tracks:
                if t.output_track_id not in self._output_tracks:
                    raise IdentityError(
                        f"output track {t.output_track_id!r} is not registered"
                    )
                if t.track_id in seen:
                    raise IdentityError(
                        f"duplicate track {t.track_id!r} in segment {segment_id!r}"
                    )
                seen.add(t.track_id)

            seg = _Segment(
                segment_id=segment_id,
                output_start=start,
                order=len(self._segments),
            )
            self._segments[segment_id] = seg

            for t in tracks:
                ot = self._output_tracks[t.output_track_id]
                key = (segment_id, t.track_id)
                budget_p = (
                    t.wait_budget_packets
                    if t.wait_budget_packets is not None
                    else self._default_budget_packets
                )
                budget_b = (
                    t.wait_budget_bytes
                    if t.wait_budget_bytes is not None
                    else self._default_budget_bytes
                )
                self._tracks[key] = _InputTrack(
                    key=key,
                    segment_id=segment_id,
                    track_id=t.track_id,
                    output_track_id=t.output_track_id,
                    reg_index=self._reg_counter,
                    budget_packets=budget_p,
                    budget_bytes=budget_b,
                    out_tb=ot.time_base,
                    rounding=ot.spec.rounding,
                    output_start=start,
                    first_seq=t.first_capture_seq,
                    frontier=t.first_capture_seq,
                    observed_any=False,
                )
                self._reg_counter += 1
                seg.track_keys.append(key)
                seg.output_track_ids.add(t.output_track_id)

    # ------------------------------------------------------------------ #
    # 输入
    # ------------------------------------------------------------------ #

    def submit(self, packet: InputPacket) -> SubmitOutcome:
        """Hand one captured packet to the kernel (producer side).

        阻塞只可能发生在输出队列满（慢封装器背压）或被取消时；
        等待预算不足时抛 :class:`BudgetExceeded`，由调用方显式 skip，
        内核不替调用方选择丢包。
        """
        if not isinstance(packet, InputPacket):
            raise TypeError("submit() expects an InputPacket")
        with self._cv:
            while True:
                if self._cancelled:
                    raise Cancelled("timeline cancelled")
                if not self._accepting:
                    raise StateError("timeline is closed; submit() rejected")
                tr = self._tracks.get((packet.segment_id, packet.track_id))
                if tr is None:
                    raise IdentityError(
                        f"unknown segment/track: "
                        f"{packet.segment_id!r}/{packet.track_id!r}"
                    )
                outcome = self._classify_arrival_locked(tr, packet.capture_seq)
                if outcome is not SubmitOutcome.ACCEPTED:
                    return outcome

                # 预算在背压等待之前检查：队列腾出空间不会增加重排缓存。
                self._enforce_budget_locked(tr, len(packet.payload))

                # 若该包会产生事件而队列已满，先等消费者腾位置。
                if self._queue_full_locked():
                    if self.on_backpressure_wait is not None:
                        self.on_backpressure_wait()
                    self._cv.wait()
                    continue

                self._accept_locked(tr, packet)
                events = self._pump_locked()
                self._enqueue_locked(events)
                return SubmitOutcome.ACCEPTED

    def skip(
        self,
        segment_id: Any,
        track_id: Any,
        first_seq: int,
        last_seq: Optional[int] = None,
        *,
        reason: SkipReason = SkipReason.EXPLICIT,
    ) -> SkippedRange:
        """Confirm that ``[first_seq, last_seq]`` will never arrive.

        缺口必须从当前 frontier 开始、且区间内没有已缓存的包。跳过的
        解码影响通过返回值和事件流交给下游（恢复点未知时如实告知 None）。
        """
        if last_seq is None:
            last_seq = first_seq
        if first_seq > last_seq:
            raise ValueError("first_seq > last_seq")
        with self._cv:
            self._check_live_locked()
            tr = self._require_track_locked(segment_id, track_id)
            if tr.frontier is None:
                raise StateError(
                    "cannot skip before any packet has been observed on "
                    f"track {track_id!r}"
                )
            if first_seq != tr.frontier:
                raise StateError(
                    f"skip must start at the frontier seq {tr.frontier}, "
                    f"got {first_seq}"
                )
            for seq in range(first_seq, last_seq + 1):
                if seq in tr.buffer:
                    raise StateError(
                        f"seq {seq} is already buffered and cannot be skipped"
                    )
            if tr.highest is not None and last_seq > tr.highest:
                raise StateError(
                    f"cannot skip beyond observed seq {tr.highest}; "
                    "unknown tail is bounded by closing the segment"
                )
            event = self._make_skip_locked(tr, first_seq, last_seq, reason)
            tr.frontier = last_seq + 1
            events = [event]
            events.extend(self._pump_locked())
            self._enqueue_locked(events)
            return event

    def close_segment(self, segment_id: Any) -> None:
        """Caller declares the segment ended.

        仍缺失的已知范围立刻变成 END_OF_SEGMENT 跳过事件并继续排空；
        内核不会留下永远等待的读取。从未观察到任何包的轨道直接排空。
        """
        with self._cv:
            self._check_live_locked()
            seg = self._segments.get(segment_id)
            if seg is None:
                raise IdentityError(f"unknown segment {segment_id!r}")
            if seg.closed:
                raise StateError(f"segment {segment_id!r} already closed")
            seg.closed = True
            for key in seg.track_keys:
                self._tracks[key].state = TrackInputState.CLOSED
            events = self._pump_locked()
            self._enqueue_locked(events)

    def close(self) -> None:
        """Stop accepting input. All open segments are closed implicitly."""
        with self._cv:
            if not self._accepting:
                return
            for seg in self._segments.values():
                if not seg.closed:
                    seg.closed = True
                    for key in seg.track_keys:
                        self._tracks[key].state = TrackInputState.CLOSED
            self._accepting = False
            events = self._pump_locked()
            self._enqueue_locked(events)

    def cancel(self) -> None:
        """Propagate cancellation to blocked producers and consumers."""
        with self._cv:
            self._cancelled = True
            self._accepting = False
            self._cv.notify_all()

    # ------------------------------------------------------------------ #
    # 上下文管理
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "Timeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # 异常退出（含读取取消）走 cancel，正常退出走 close，
        # 保证不会留下永远阻塞的生产者/消费者。
        if exc_type is None:
            self.close()
        else:
            self.cancel()
        return None

    # ------------------------------------------------------------------ #
    # 输出
    # ------------------------------------------------------------------ #

    def pull(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        """Pull one event; non-blocking returns None if nothing is ready.

        正常排空后阻塞调用抛 :class:`EndOfTimeline`；取消后抛
        :class:`Cancelled`。事件类型见 ``events`` 模块。
        """
        with self._cv:
            if not self._queue:
                if not block:
                    self._raise_if_done_locked()
                    return None
                if not self._cv.wait_for(
                    lambda: self._queue or self._cancelled or self._finished,
                    timeout,
                ):
                    return None
            if self._queue:
                ev = self._queue.popleft()
                # 腾出一个位置，唤醒可能被慢消费者背压的生产者
                self._cv.notify_all()
                return ev
            if self._cancelled:
                raise Cancelled("timeline cancelled")
            raise EndOfTimeline("all segments drained")

    def events(self) -> Iterator[Any]:
        """Yield every output event until the timeline is finished."""
        while True:
            try:
                yield self.pull(block=True)
            except EndOfTimeline:
                return

    def status(self) -> StatusSnapshot:
        """Snapshot: which tracks still accept input, what is determined,
        and which resources are still held."""
        with self._cv:
            track_stats = []
            for tr in self._tracks.values():
                if tr.state is TrackInputState.CLOSED:
                    state = TrackInputState.CLOSED
                elif tr.waiting:
                    state = TrackInputState.WAITING
                else:
                    state = TrackInputState.OPEN
                track_stats.append(
                    TrackStatus(
                        segment_id=tr.segment_id,
                        track_id=tr.track_id,
                        output_track_id=tr.output_track_id,
                        state=state,
                        can_input=(
                            not self._cancelled
                            and self._accepting
                            and tr.state is TrackInputState.OPEN
                        ),
                        frontier_seq=tr.frontier,
                        highest_observed_seq=tr.highest,
                        buffered_count=len(tr.buffer),
                        buffered_bytes=tr.buffered_bytes,
                        budget_limit=tr.budget_packets,
                        waiting=tr.waiting,
                        confirmed_skipped=tr.skipped_count,
                        delivered=tr.delivered_count,
                    )
                )
            seg_stats = [
                SegmentStatus(
                    segment_id=seg.segment_id,
                    output_start_seconds=seg.output_start,
                    output_track_ids=tuple(sorted(
                        seg.output_track_ids,
                        key=lambda oid: self._output_tracks[oid].reg_index,
                    )),
                    closed=seg.closed,
                    drained=seg.drained,
                )
                for seg in sorted(self._segments.values(), key=lambda s: s.order)
            ]
            resources = ResourceStatus(
                buffered_packets=sum(len(t.buffer) for t in self._tracks.values()),
                buffered_bytes=sum(t.buffered_bytes for t in self._tracks.values()),
                in_flight_events=len(self._queue),
                output_queue_capacity=self._queue_capacity,
                open_segments=sum(
                    1 for s in self._segments.values() if not s.drained
                ),
                open_tracks=sum(
                    1
                    for t in self._tracks.values()
                    if t.state is not TrackInputState.CLOSED
                ),
            )
            return StatusSnapshot(
                tracks=tuple(track_stats),
                segments=tuple(seg_stats),
                resources=resources,
                cancelled=self._cancelled,
                finished=self._finished,
            )

    # ------------------------------------------------------------------ #
    # 内部：到达分类 / 预算
    # ------------------------------------------------------------------ #

    def _check_live_locked(self) -> None:
        if self._cancelled:
            raise Cancelled("timeline cancelled")
        if not self._accepting:
            raise StateError("timeline is closed")

    def _require_track_locked(
        self, segment_id: Any, track_id: Any
    ) -> _InputTrack:
        tr = self._tracks.get((segment_id, track_id))
        if tr is None:
            raise IdentityError(
                f"unknown segment/track: {segment_id!r}/{track_id!r}"
            )
        return tr

    def _classify_arrival_locked(
        self, tr: _InputTrack, seq: int
    ) -> SubmitOutcome:
        if tr.state is TrackInputState.CLOSED:
            return SubmitOutcome.LATE_AFTER_CLOSE
        if seq in tr.buffer:
            return SubmitOutcome.DUPLICATE
        if tr.frontier is not None and seq < tr.frontier:
            if any(f <= seq <= l for f, l in tr.skip_intervals):
                return SubmitOutcome.LATE_AFTER_SKIP
            return SubmitOutcome.DUPLICATE
        return SubmitOutcome.ACCEPTED

    def _enforce_budget_locked(self, tr: _InputTrack, payload_bytes: int) -> None:
        if (
            tr.budget_packets is not None
            and len(tr.buffer) >= tr.budget_packets
        ):
            raise BudgetExceeded(
                track_id=(tr.segment_id, tr.track_id),
                limit=tr.budget_packets,
                current=len(tr.buffer),
            )
        if (
            tr.budget_bytes is not None
            and tr.buffered_bytes + payload_bytes > tr.budget_bytes
        ):
            raise BudgetExceeded(
                track_id=(tr.segment_id, tr.track_id),
                limit=tr.budget_bytes,
                current=tr.buffered_bytes,
            )

    def _accept_locked(self, tr: _InputTrack, packet: InputPacket) -> None:
        tr.buffer[packet.capture_seq] = packet
        tr.buffered_bytes += len(packet.payload)
        tr.observed_any = True
        if tr.highest is None or packet.capture_seq > tr.highest:
            tr.highest = packet.capture_seq

    # ------------------------------------------------------------------ #
    # 内部：泵——重排交付、缺口状态、分段门控
    # ------------------------------------------------------------------ #

    def _queue_full_locked(self) -> bool:
        return (
            self._queue_capacity is not None
            and len(self._queue) >= self._queue_capacity
        )

    def _enqueue_locked(self, events: list[Any]) -> None:
        """把泵产出的事件放进有界队列；满了就等消费者（同锁，不死锁）。"""
        for ev in events:
            while self._queue_full_locked() and not self._cancelled:
                self._cv.wait()
            if self._cancelled:
                raise Cancelled("timeline cancelled")
            self._queue.append(ev)
        if events:
            self._cv.notify_all()

    def _pump_locked(self) -> list[Any]:
        out: list[Any] = []
        progressed = True
        while progressed:
            progressed = False

            # 全局分段门控：只有"最前面尚未排空"的分段允许交付。
            # 后面分段的包即使全部到达也只进入缓存，避免后段的包
            # 在 muxer 流上越过前段（包括前段有空洞的情况——否则后段
            # 会持续占用输出事件，前段反而永远得不到推进机会）。
            head = self._head_undrained_segment_locked()
            if head is not None:
                for key in sorted(
                    head.track_keys,
                    key=lambda k: self._tracks[k].reg_index,
                ):
                    tr = self._tracks[key]
                    before = len(out)
                    self._drain_track_locked(tr, out)
                    if len(out) > before:
                        progressed = True

            # 分段排空判定：所有轨道都已结束、无缓存、无空洞
            for seg in sorted(self._segments.values(), key=lambda s: s.order):
                if seg.drained:
                    continue
                if not seg.closed:
                    continue
                if all(
                    self._tracks[k].state is TrackInputState.CLOSED
                    and not self._tracks[k].buffer
                    and not self._tracks[k].waiting
                    for k in seg.track_keys
                ):
                    seg.drained = True
                    for oid in seg.output_track_ids:
                        self._output_tracks[oid].drained_segments.add(
                            seg.segment_id
                        )
                    out.append(
                        SegmentClosed(
                            segment_id=seg.segment_id,
                            output_track_ids=tuple(sorted(
                                seg.output_track_ids,
                                key=lambda o: self._output_tracks[o].reg_index,
                            )),
                        )
                    )
                    progressed = True

            # 调用方已宣告结束（close）且每个分段都已排空时，不会再有事件。
            if (
                not self._finished
                and not self._accepting
                and self._segments
                and all(s.drained for s in self._segments.values())
            ):
                self._finished = True

        if self._finished and self._segments:
            self._cv.notify_all()
        return out

    def _head_undrained_segment_locked(self) -> Optional[_Segment]:
        for seg in sorted(self._segments.values(), key=lambda s: s.order):
            if not seg.drained:
                return seg
        return None

    def _drain_track_locked(self, tr: _InputTrack, out: list[Any]) -> None:
        # 从未收到任何包的已关闭轨道：该分段这条轨道本就没有数据，
        # 不产生缺口（没有"已知存在却缺失"的范围），直接排空。
        if not tr.observed_any and tr.state is TrackInputState.CLOSED:
            tr.frontier = None
            return
        # 已关闭轨道：把已知窗口内的缺失段先确认成跳过，再尝试排空。
        while True:
            if tr.frontier is None:
                return
            if tr.frontier in tr.buffer:
                break
            if tr.highest is not None and tr.frontier <= tr.highest:
                if tr.state is TrackInputState.CLOSED:
                    # 缺口只延伸到下一个 *已缓存* 序号之前；
                    # 不能一口气推到 highest，否则会跨过缓存包造成死循环。
                    buffered_ahead = [s for s in tr.buffer if s >= tr.frontier]
                    end = min(buffered_ahead) - 1
                    out.append(
                        self._make_skip_locked(
                            tr,
                            tr.frontier,
                            end,
                            SkipReason.END_OF_SEGMENT,
                        )
                    )
                    tr.frontier = end + 1
                    continue
                # 开放分段：空洞必须等数据或等显式 skip
                self._emit_waiting_if_changed_locked(tr, out)
                return
            return

        while tr.frontier is not None and tr.frontier in tr.buffer:
            packet = tr.buffer.pop(tr.frontier)
            tr.buffered_bytes -= len(packet.payload)
            event = self._make_delivered_locked(tr, packet, out)
            out.append(event)
            tr.delivered_count += 1
            tr.frontier += 1

        # frontier 窗口耗尽（highest 已交付）：重置等待签名；
        # 否则根据空洞情况刷新 WaitingRange。
        self._emit_waiting_if_changed_locked(tr, out)

    def _emit_waiting_if_changed_locked(
        self, tr: _InputTrack, out: list[Any]
    ) -> None:
        # 每次"开始等待一个新空洞"只发一份 WaitingRange；空洞期间积压包
        # 数量的变化属于资源信息，走 status()，避免事件流被重复状态刷屏。
        if tr.waiting:
            if tr.last_wait_frontier != tr.frontier:
                tr.last_wait_frontier = tr.frontier
                out.append(
                    WaitingRange(
                        segment_id=tr.segment_id,
                        track_id=tr.track_id,
                        output_track_id=tr.output_track_id,
                        first_seq=tr.frontier,
                        buffered_beyond=len(tr.buffer),
                        highest_observed_seq=tr.highest,
                    )
                )
        else:
            tr.last_wait_frontier = None

    # ------------------------------------------------------------------ #
    # 内部：事件构造（时间映射 / 解码可用性）
    # ------------------------------------------------------------------ #

    def _make_skip_locked(
        self,
        tr: _InputTrack,
        first: int,
        last: int,
        reason: SkipReason,
    ) -> SkippedRange:
        tr.skip_intervals.append((first, last))
        tr.skipped_count += last - first + 1
        tr.dirty = True

        next_keyframe_seq: Optional[int] = None
        next_recovery_time: Optional[Fraction] = None
        for seq in sorted(tr.buffer):
            if seq <= last:
                continue
            pkt = tr.buffer[seq]
            if pkt.keyframe and pkt.payload_usability is not PayloadUsability.CORRUPT:
                next_keyframe_seq = seq
                next_recovery_time = self._exact_pts_seconds_locked(tr, pkt)
                break
        tr.unrecovered.append((first, last))
        return SkippedRange(
            segment_id=tr.segment_id,
            track_id=tr.track_id,
            output_track_id=tr.output_track_id,
            first_seq=first,
            last_seq=last,
            count=last - first + 1,
            reason=reason,
            next_keyframe_seq=next_keyframe_seq,
            next_recovery_time_seconds=next_recovery_time,
        )

    def _exact_pts_seconds_locked(
        self, tr: _InputTrack, packet: InputPacket
    ) -> Optional[Fraction]:
        if packet.pts is None or packet.time_base is None:
            return None
        return tr.output_start + Fraction(packet.pts) * as_fraction(
            packet.time_base
        )

    def _make_delivered_locked(
        self, tr: _InputTrack, packet: InputPacket, out: list[Any]
    ) -> DeliveredPacket:
        pts_m = map_field(
            packet.pts,
            packet.time_base,
            tr.output_start,
            tr.out_tb,
            tr.rounding,
        )
        dts_m = map_field(
            packet.dts,
            packet.time_base,
            tr.output_start,
            tr.out_tb,
            tr.rounding,
        )
        dur_m = map_field(
            packet.duration,
            packet.time_base,
            Fraction(0),  # 持续时间是相对量，不平移
            tr.out_tb,
            tr.rounding,
        )
        present_fields = [
            m
            for m in (pts_m, dts_m, dur_m)
            if m.original_ticks is not None
        ]
        if packet.time_base is None and present_fields:
            status = TimeStatus.UNCONVERTIBLE
        elif any(m.residual not in (None, Fraction(0)) for m in present_fields):
            status = TimeStatus.QUANTIZED
        else:
            status = TimeStatus.EXACT

        correction = TimeCorrection(
            pts=pts_m,
            dts=dts_m,
            duration=dur_m,
            output_time_base=tr.out_tb,
            status=status,
        )

        # 解码可用性：脏链路上的第一个 *完好关键帧* 恢复可用性。
        availability: DecodeAvailability
        recovers = (
            packet.keyframe
            and packet.payload_usability is not PayloadUsability.CORRUPT
            and tr.dirty
        )
        if recovers:
            availability = DecodeAvailability.RECOVERY_POINT
            recovery_time = pts_m.mapped_seconds
            for first, last in tr.unrecovered:
                out.append(
                    RecoveryObserved(
                        segment_id=tr.segment_id,
                        track_id=tr.track_id,
                        output_track_id=tr.output_track_id,
                        after_skipped_first_seq=first,
                        after_skipped_last_seq=last,
                        recovery_seq=packet.capture_seq,
                        recovery_time_seconds=recovery_time,
                    )
                )
            tr.unrecovered.clear()
            tr.dirty = False
        elif packet.payload_usability is PayloadUsability.CORRUPT:
            availability = DecodeAvailability.CORRUPTED
        elif tr.dirty:
            availability = DecodeAvailability.DEGRADED_AFTER_GAP
        elif status is TimeStatus.UNCONVERTIBLE:
            availability = DecodeAvailability.TIME_UNKNOWN
        else:
            availability = DecodeAvailability.INTACT

        return DeliveredPacket(
            segment_id=packet.segment_id,
            track_id=packet.track_id,
            output_track_id=tr.output_track_id,
            capture_seq=packet.capture_seq,
            keyframe=packet.keyframe,
            payload=packet.payload,
            payload_usability=packet.payload_usability,
            correction=correction,
            decode_availability=availability,
            output_pts=pts_m.output_ticks,
            output_dts=dts_m.output_ticks,
            output_duration=dur_m.output_ticks,
        )

    # ------------------------------------------------------------------ #
    # 小工具
    # ------------------------------------------------------------------ #

    def _raise_if_done_locked(self) -> None:
        if self._cancelled:
            raise Cancelled("timeline cancelled")
        if self._finished:
            raise EndOfTimeline("all segments drained")

    # 让取消时等待的生产者也能从队列事件中回收内存：取消后仍可 pull。
    @property
    def cancelled(self) -> bool:
        with self._cv:
            return self._cancelled
