"""精确有理时间换算。

规则：
  输出秒 = 段输出起点 + 轨内采集偏移 + 采集 tick * 采集时间基准

从采集 tick 到输出秒只用 :class:`fractions.Fraction`，30000/1001 这类帧率
连续换算、多段接续时不存在浮点累加。换算到输出时间基准的整数 tick 时
记录残差，并在“精确顺序保持单调”的前提下消除取整造成的 +0/+0/+1 抖动。
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

from .types import CorrectedTiming, Tick, TimingStatus, TrackSpec


def _round(value: Fraction, mode: str) -> int:
    if mode == "floor":
        return value.numerator // value.denominator
    if mode == "ceil":
        return -((-value.numerator) // value.denominator)
    if mode != "half_even":
        raise ValueError(f"unknown rounding mode: {mode!r}")
    # round-half-even，纯整数实现
    q, r = divmod(value.numerator, value.denominator)
    twice = 2 * r
    if twice > value.denominator or (twice == value.denominator and q & 1):
        q += 1
    return q


class TickConverter:
    """单条 (段, 轨) 的时间换算器，持有“上一精确/输出 DTS”以处理取整单调。"""

    def __init__(self, spec: TrackSpec, output_start: Fraction, capture_offset: Fraction):
        self._spec = spec
        self._origin = output_start + capture_offset  # Fraction 秒
        self._last_exact_dts: Optional[Fraction] = None
        self._last_out_dts: Optional[int] = None
        self._adjust_total = Fraction(0)

    @property
    def output_time_base(self) -> Fraction:
        return self._spec.output_time_base

    def _to_tick(self, raw: Optional[int], raw_tb: Fraction) -> Optional[Tick]:
        if raw is None:
            return None
        exact = self._origin + raw * raw_tb
        ticks = _round(exact / self._spec.output_time_base, self._spec.rounding)
        residual = ticks * self._spec.output_time_base - exact
        return Tick(ticks=ticks, exact_seconds=exact, residual_seconds=residual)

    @staticmethod
    def _status(tick: Optional[Tick]) -> TimingStatus:
        if tick is None:
            return TimingStatus.UNMAPPED
        return TimingStatus.EXACT if tick.residual_seconds == 0 else TimingStatus.ROUNDED

    def convert(
        self,
        pts: Optional[int],
        dts: Optional[int],
        duration: Optional[int],
        packet_tb: Optional[Fraction],
    ) -> tuple[CorrectedTiming, Optional[str]]:
        raw_tb = packet_tb if packet_tb is not None else self._spec.default_time_base
        pt = self._to_tick(pts, raw_tb)
        dt = self._to_tick(dts, raw_tb)
        du = self._to_tick(duration, raw_tb)
        # duration 是相对量，不应加段原点：重做
        if duration is not None:
            exact_dur = duration * raw_tb
            ticks_dur = _round(exact_dur / self._spec.output_time_base, self._spec.rounding)
            du = Tick(
                ticks=ticks_dur,
                exact_seconds=exact_dur,
                residual_seconds=ticks_dur * self._spec.output_time_base - exact_dur,
            )

        warning: Optional[str] = None
        adjustment = Fraction(0)

        if dt is not None:
            exact_dts = dt.exact_seconds
            if self._last_exact_dts is not None and exact_dts < self._last_exact_dts:
                # 精确时间已经逆序——不是取整能解决的问题。典型原因：段锚点配错
                # 或上游把不同段的包喂进了同一段。绝不静默改时间。
                warning = "non-monotonic exact dts across packets"
            elif self._last_out_dts is not None and dt.ticks <= self._last_out_dts:
                if self._last_exact_dts is not None and exact_dts >= self._last_exact_dts:
                    # 精确值仍单调（或相等），只是取整后挤到同一个 tick：钳 +1。
                    bumped = self._last_out_dts + 1
                    delta = bumped - dt.ticks
                    if delta > 0:
                        adjustment = delta * self._spec.output_time_base
                        dt = Tick(
                            ticks=bumped,
                            exact_seconds=dt.exact_seconds,
                            residual_seconds=bumped * self._spec.output_time_base - dt.exact_seconds,
                        )
                        self._adjust_total += adjustment
            self._last_exact_dts = exact_dts
            self._last_out_dts = dt.ticks

        timing = CorrectedTiming(
            output_time_base=self._spec.output_time_base,
            pts=pt,
            dts=dt,
            duration=du,
            pts_status=self._status(pt),
            dts_status=self._status(dt),
            duration_status=self._status(du),
            monotonic_adjustment_seconds=adjustment,
        )
        return timing, warning
