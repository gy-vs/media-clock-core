"""media-clock-core — packet timing, identity, delivery order and
decodability contracts between capture readers and muxers.

公开 API 保持很小：一个内核（:class:`Timeline`）、输入输出契约、
精确时间运算，以及可选的 PyAV 适配层（:mod:`media_clock_core.av_adapter`）。
"""

from .errors import (
    BudgetExceeded,
    Cancelled,
    IdentityError,
    MediaClockError,
    StateError,
)
from .events import (
    DecodeAvailability,
    DeliveredPacket,
    InputPacket,
    PayloadUsability,
    RecoveryObserved,
    SegmentClosed,
    SkipReason,
    SkippedRange,
    StatusSnapshot,
    SubmitOutcome,
    TimeCorrection,
    TimeStatus,
    WaitingRange,
)
from .kernel import EndOfTimeline, InputTrackSpec, OutputTrackSpec, Timeline
from .timebase import Rounding, TickResult, as_fraction, rescale_ticks

__version__ = "0.1.0"

__all__ = [
    "Timeline",
    "InputTrackSpec",
    "OutputTrackSpec",
    "EndOfTimeline",
    "InputPacket",
    "DeliveredPacket",
    "SkippedRange",
    "RecoveryObserved",
    "WaitingRange",
    "SegmentClosed",
    "TimeCorrection",
    "TimeStatus",
    "DecodeAvailability",
    "PayloadUsability",
    "SkipReason",
    "SubmitOutcome",
    "StatusSnapshot",
    "Rounding",
    "TickResult",
    "rescale_ticks",
    "as_fraction",
    "MediaClockError",
    "IdentityError",
    "StateError",
    "BudgetExceeded",
    "Cancelled",
]
