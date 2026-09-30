"""单条 (段, 轨) 的序号整理与解码阴影状态机。

整理的是 *采集序号* 的交付次序，不是 PTS。显示时间回跳（B 帧重排序）在
这里完全不可见——序号小的包永远先交付，无论它的 PTS 是大是小。
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

from .timing import TickConverter
from .types import (
    InputPacket,
    KernelEvent,
    MediaKind,
    PacketDelivered,
    RangeSkipped,
    RangeStatus,
    SeqNum,
    TimingStatus,
    TrackSpec,
    TrackState,
    TrackStatus,
    Usability,
)


class SkipOverlapError(ValueError):
    """试图确认跳过的范围与已经到达、仍占住的包重叠。"""


class TrackBuffer:
    def __init__(
        self,
        segment_id: str,
        spec: TrackSpec,
        output_start,
        capture_offset,
    ):
        self.segment_id = segment_id
        self.spec = spec
        self._converter = TickConverter(spec, output_start, capture_offset)
        self.state = TrackState.OPEN
        self._next: SeqNum = 0
        self._held: dict[SeqNum, InputPacket] = {}
        # 已确认跳过的区间（有序、不重叠）
        self._skipped: list[tuple[SeqNum, SeqNum]] = []
        self._max_observed: SeqNum = -1  # 见过的最大序号
        self._delivered_count = 0
        self._skipped_count = 0
        self._held_bytes = 0
        # 解码阴影：跳过后置位，视频上到下一个交付的关键帧解除
        self._shadow_active = False
        # 跨包表示误差累计（Fraction 秒）
        self._sum_pts_residual = Fraction(0)
        self._sum_dts_residual = Fraction(0)
        self._sum_adjust = Fraction(0)
        self._rounded = 0

    # -- 查询 ---------------------------------------------------------------

    @property
    def next_seq(self) -> SeqNum:
        return self._next

    @property
    def held_count(self) -> int:
        return len(self._held)

    @property
    def held_bytes(self) -> int:
        return self._held_bytes

    def can_accept(self, seq: SeqNum) -> bool:
        return self.state == TrackState.OPEN

    def status(self) -> TrackStatus:
        waiting: list[RangeStatus] = []
        if self.state == TrackState.OPEN and self._next not in self._held:
            # 唯一的前缘缺口：[next, 第一个占住包)；无占住包则上界是已观察最大值+1
            upper = min(self._held) if self._held else self._max_observed + 1
            if upper > self._next:
                waiting.append(RangeStatus("waiting", self._next, upper))
        skipped_ranges = tuple(RangeStatus("skipped", a, b) for a, b in self._skipped)
        return TrackStatus(
            segment_id=self.segment_id,
            track_id=self.spec.track_id,
            state=self.state,
            next_contiguous_seq=self._next,
            delivered=self._delivered_count,
            skipped=self._skipped_count,
            holding=len(self._held),
            holding_bytes=self._held_bytes,
            waiting_ranges=tuple(waiting),
            skipped_ranges=skipped_ranges,
            can_input=self.state == TrackState.OPEN,
            determined_through=self._next - 1,
            cumulative_pts_residual_seconds=self._sum_pts_residual,
            cumulative_dts_residual_seconds=self._sum_dts_residual,
            cumulative_monotonic_adjustment_seconds=self._sum_adjust,
            rounded_packets=self._rounded,
        )

    # -- 写入 ---------------------------------------------------------------

    def feed(self, packet: InputPacket) -> tuple[list[KernelEvent], bool]:
        """喂入一个包，返回 (新产生事件, 是否占住预算)。

        已交付/已跳过序号的重复包被幂等忽略。
        """
        if self.state != TrackState.OPEN:
            return [], False
        seq = packet.seq
        if seq > self._max_observed:
            self._max_observed = seq
        if seq < self._next:
            return [], False  # 重复/迟到于已确定前缘：忽略而不是报错
        if seq in self._held:
            return [], False
        if self._is_skipped(seq):
            return [], False

        held_it = seq > self._next
        if held_it:
            self._held[seq] = packet
            self._held_bytes += packet.payload_bytes
            return [], True

        # seq == next：直接交付并尽可能推进
        self._held[seq] = packet
        self._held_bytes += packet.payload_bytes
        events = self._advance()
        return events, False

    def confirm_skip(self, start: SeqNum, end: SeqNum, reason: str) -> list[KernelEvent]:
        if start >= end:
            raise ValueError("skip range must be non-empty [start, end)")
        if start != self._next:
            raise ValueError(
                f"can only confirm skip at current frontier {self._next}, got {start}"
            )
        overlap = sorted(s for s in self._held if start <= s < end)
        if overlap:
            raise SkipOverlapError(
                f"seqs already arrived and held: {overlap[:8]}{'...' if len(overlap) > 8 else ''}"
            )
        return self._finalize_gap(start, end, reason)

    def finish(self) -> list[KernelEvent]:
        """调用方明确结束本段本轨：把等待部分变成确定结果，冲刷全部占住包。"""
        if self.state != TrackState.OPEN:
            return []
        events: list[KernelEvent] = []
        # 可能有多个交错缺口（如到达 0,2,4，缺 1,3）：逐个把前缘到下一个
        # 已到达包之间的缺口确认为跳过，再连带交付连续占住包，直到清空。
        while self._held:
            upper = min(self._held)
            if upper > self._next:
                events.extend(
                    self._finalize_gap(self._next, upper, "unfinished-at-end")
                )
            else:
                events.extend(self._advance())
        self.state = TrackState.FINISHED
        return events

    def cancel(self) -> None:
        self.state = TrackState.CANCELLED
        self._held.clear()
        self._held_bytes = 0

    # -- 内部 ---------------------------------------------------------------

    def _is_skipped(self, seq: SeqNum) -> bool:
        for a, b in self._skipped:
            if a <= seq < b:
                return True
            if a > seq:
                break
        return False

    def _first_held_keyframe(self, start: SeqNum) -> Optional[SeqNum]:
        """缺口之后（>=start）占住中的第一个关键帧；缺口前的旧关键帧不算。"""
        for s in sorted(self._held):
            if s < start:
                continue
            if self._held[s].is_keyframe:
                return s
        return None

    def _finalize_gap(self, start: SeqNum, end: SeqNum, reason: str) -> list[KernelEvent]:
        events: list[KernelEvent] = []
        # 缺口内是否曾观察到关键帧？到达即占住，不可能——缺口内没有任何到达包。
        # contained_keyframe 恒为 False，但保留字段以支持未来“到达后丢弃”的跳过。
        shadow_until: Optional[SeqNum] = None
        if self.spec.kind is MediaKind.VIDEO:
            kf = self._first_held_keyframe(end)
            if kf is not None:
                shadow_until = kf  # 关键帧本身可解码，阴影到它之前
            self._shadow_active = True
        self._skipped.append((start, end))
        self._skipped.sort()
        self._skipped_count += end - start
        events.append(
            RangeSkipped(
                segment_id=self.segment_id,
                track_id=self.spec.track_id,
                start_seq=start,
                end_seq=end,
                reason=reason,
                shadow_until_seq=shadow_until,
                contained_keyframe=False,
            )
        )
        self._next = end
        events.extend(self._advance())
        return events

    def _advance(self) -> list[KernelEvent]:
        events: list[KernelEvent] = []
        while True:
            # 跳过已确认跳过区间（理论上 confirm_skip 已推进，这里兜底）
            while self._is_skipped(self._next):
                for a, b in self._skipped:
                    if a == self._next:
                        self._next = b
            packet = self._held.pop(self._next, None)
            if packet is None:
                break
            self._held_bytes -= packet.payload_bytes
            shadowed = self._shadow_active
            if self.spec.kind is MediaKind.VIDEO and packet.is_keyframe:
                self._shadow_active = False
                shadowed = False
            timing, warning = self._converter.convert(
                packet.pts, packet.dts, packet.duration, packet.time_base
            )
            if timing.pts is not None:
                self._sum_pts_residual += timing.pts.residual_seconds
                if timing.pts_status is TimingStatus.ROUNDED:
                    self._rounded += 1
            if timing.dts is not None:
                self._sum_dts_residual += timing.dts.residual_seconds
            self._sum_adjust += timing.monotonic_adjustment_seconds
            events.append(
                PacketDelivered(
                    segment_id=self.segment_id,
                    track_id=self.spec.track_id,
                    seq=packet.seq,
                    origin=packet,
                    timing=timing,
                    usability=(
                        Usability.DEPENDENCY_BROKEN if shadowed else Usability.INTACT
                    ),
                    shadowed_by_gap=shadowed,
                    warnings=((warning,) if warning else ()),
                )
            )
            self._delivered_count += 1
            self._next += 1
        return events
