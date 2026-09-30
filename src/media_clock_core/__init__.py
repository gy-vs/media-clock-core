"""media-clock-core：媒体时间、身份、交付顺序与可用性契约内核。

不做播放器、不做转码 CLI、不改载荷。上游是读取器，下游是真正的封装器。
"""
from .kernel import MediaClockKernel
from .pipeline import ClockPipeline, PipelineCancelled
from .types import (
    CorrectedTiming,
    FeedOutcome,
    FeedResult,
    InputPacket,
    KernelEvent,
    KernelStatus,
    MediaKind,
    PacketDelivered,
    RangeSkipped,
    RangeStatus,
    ResourceStatus,
    SegmentId,
    SegmentSpec,
    Tick,
    TimingStatus,
    TrackId,
    TrackSpec,
    TrackState,
    TrackStatus,
    Usability,
    WaitingBudget,
)

__version__ = "0.1.0"

__all__ = [
    "MediaClockKernel",
    "ClockPipeline",
    "PipelineCancelled",
    "CorrectedTiming",
    "FeedOutcome",
    "FeedResult",
    "InputPacket",
    "KernelEvent",
    "KernelStatus",
    "MediaKind",
    "PacketDelivered",
    "RangeSkipped",
    "RangeStatus",
    "ResourceStatus",
    "SegmentId",
    "SegmentSpec",
    "Tick",
    "TimingStatus",
    "TrackId",
    "TrackSpec",
    "TrackState",
    "TrackStatus",
    "Usability",
    "WaitingBudget",
    "__version__",
]
