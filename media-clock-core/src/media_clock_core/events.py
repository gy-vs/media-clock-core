"""Input and output contracts.

内核不认识任何具体编解码器，载荷是不透明的 ``bytes``；内核只处理四件事：

1. **时间** —— 原始时间字段到公共时间线的精确有理映射；
2. **身份** —— 分段 / 轨道 / 采集序号；
3. **交付次序** —— 按采集序号（= 解码顺序）整理网络乱序；
4. **可用性契约** —— 交付、确认跳过、等待三种状态，
   以及跳过对后续解码可用性的影响如何交给下游，而不是写进日志。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Optional

from .timebase import AffineMapping, TimeBaseLike


# --------------------------------------------------------------------------- #
# 输入
# --------------------------------------------------------------------------- #


class PayloadUsability(enum.Enum):
    """读取器对 *载荷本身* 能否继续解码的判断。

    这与时间字段是否能换算互相独立：损坏的包可能带着完美的时间戳，
    时间戳不可换算的包也可能载荷完好。
    """

    USABLE = enum.auto()
    """载荷可供解码（内核不验证，只转述读取器的判断）。"""

    CORRUPT = enum.auto()
    """读取器确认载荷已损坏，交付后由下游决定是否喂给解码器。"""

    UNKNOWN = enum.auto()
    """读取器无法判断（例如只收到部分数据）。"""


@dataclass(frozen=True)
class InputPacket:
    """One captured packet handed to the kernel.

    所有字段由上游读取器提供。``segment_id`` / ``track_id`` 是不透明的
    可哈希身份；内核绝不依据"时间变小了"自行猜测分段边界。

    Attributes:
        capture_seq: 采集设备的包序号，按解码顺序单调递增（B 帧场景下
            这是 DTS 顺序，**不是** PTS 顺序）。
        pts / dts / duration: 原始整数时间戳，允许为负（B 帧开头的 DTS），
            也允许为 None（字段缺失）。
        time_base: 该轨道采集时使用的时间基准，如 Fraction(1, 25)。
            为 None 表示时间字段不可换算，时间状态会相应标记，但不影响交付。
        keyframe: 采集包是否为关键帧/随机访问点。跳缺之后，内核据此
            判断后续非关键包在哪个包上恢复解码可用性。
        payload: 不透明载荷字节，内核永不修改、永不重新编码。
    """

    segment_id: Any
    track_id: Any
    capture_seq: int
    pts: Optional[int]
    dts: Optional[int]
    duration: Optional[int]
    time_base: Optional[TimeBaseLike]
    payload: bytes
    keyframe: bool = False
    payload_usability: PayloadUsability = PayloadUsability.USABLE

    def __post_init__(self) -> None:
        if not isinstance(self.capture_seq, int) or isinstance(
            self.capture_seq, bool
        ):
            raise TypeError("capture_seq must be an int")
        if not isinstance(self.payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        if not isinstance(self.keyframe, bool):
            raise TypeError("keyframe must be bool")


# --------------------------------------------------------------------------- #
# 输出：时间状态与解码可用性
# --------------------------------------------------------------------------- #


class TimeStatus(enum.Enum):
    """时间字段在公共时间线上的可换算状态（与载荷可用性正交）。"""

    EXACT = enum.auto()
    """所有已提供字段均精确映射，无量化误差。"""

    QUANTIZED = enum.auto()
    """映射成功，但输出时间基准无法精确表示，残差非零（已公开数值）。"""

    UNCONVERTIBLE = enum.auto()
    """缺少时间基准或时间字段，无法换算；载荷照常交付。"""


class DecodeAvailability(enum.Enum):
    """下游拿到这个/这批数据时，对"能否继续解码"应有的判断。

    内核不做编解码决策，只把缺口对解码链的影响结构化为契约：
    具体怎么处理仍然是下游（解码器/封装器）的事。
    """

    INTACT = enum.auto()
    """本包及其依赖链上没有确认缺口。"""

    DEGRADED_AFTER_GAP = enum.auto()
    """本包之前存在确认跳过的范围，且本包不是随机访问点；
    在到达下一个关键帧之前，解码器可能花屏/报错/需要错误恢复。"""

    RECOVERY_POINT = enum.auto()
    """本包是缺口之后的第一个关键帧；从这里开始解码链重新可用。"""

    CORRUPTED = enum.auto()
    """读取器已确认载荷本身损坏（与时间缺口无关）。"""

    TIME_UNKNOWN = enum.auto()
    """载荷可能完好，但时间字段无法换算，不能据此安排时间线。"""


@dataclass(frozen=True)
class TimeCorrection:
    """一个包的完整时间修正结果，保留全部可检查关系。"""

    pts: AffineMapping
    dts: AffineMapping
    duration: AffineMapping
    output_time_base: Fraction
    status: TimeStatus
    """三个字段的综合状态（取最差者）。"""

    def residual_seconds_bound(self) -> Fraction:
        """该包时间表示误差的上界（秒，精确有理）。

        取 PTS/DTS/duration 三个字段残差绝对值的最大值——这是"这个包的
        时间在输出基准下最坏偏了多少"的保守上界。多段接续时逐包独立换算，
        该上界不随段长增长，可直接逐包公开/校验。
        """
        bound = Fraction(0)
        for m in (self.pts, self.dts, self.duration):
            if m.residual is not None:
                bound = max(bound, abs(m.residual))
        return bound


@dataclass(frozen=True)
class DeliveredPacket:
    """已整理次序、完成时间修正、可以交给封装器的包。

    与原始 :class:`InputPacket` 一一关联（``capture_seq`` + 双身份），
    ``payload`` 是同一份字节（零拷贝），解码依赖关系没有被改动：
    交付顺序严格按采集序号，B 帧的 PTS 回跳原样保留。
    """

    kind: str = field(default="packet", init=False)
    segment_id: Any
    track_id: Any
    output_track_id: Any
    capture_seq: int
    keyframe: bool
    payload: bytes
    payload_usability: PayloadUsability
    correction: TimeCorrection
    decode_availability: DecodeAvailability
    output_pts: Optional[int]
    output_dts: Optional[int]
    output_duration: Optional[int]

    @property
    def residual_seconds(self) -> Fraction:
        """精确表示误差上界（秒），供调用方公开/校验，绝不伪装成 0。"""
        return self.correction.residual_seconds_bound()


# --------------------------------------------------------------------------- #
# 输出：缺口契约
# --------------------------------------------------------------------------- #


class SkipReason(enum.Enum):
    EXPLICIT = enum.auto()
    """调用方显式确认跳过（例如读取器报告 NACK / 永久缺失）。"""

    END_OF_SEGMENT = enum.auto()
    """分段被显式结束，结束时仍未到达的已知缺口被一次性确认。"""


@dataclass(frozen=True)
class SkippedRange:
    """一段被 *确认* 不会到达的采集序号区间 ``[first_seq, last_seq]``。

    内核不会静默丢任何东西；跳过必须来自调用方显式决定或分段结束。
    对后续解码可用性的影响在这里结构化交代，而不是记一个丢帧计数：
    下游应把 ``next_keyframe_seq`` 之前的非关键包按降级状态处理。
    """

    kind: str = field(default="skipped", init=False)
    segment_id: Any
    track_id: Any
    output_track_id: Any
    first_seq: int
    last_seq: int
    count: int
    reason: SkipReason
    next_keyframe_seq: Optional[int]
    """缺口之后第一个已到达关键帧的序号；None 表示目前还没观察到，
    恢复点未知——此时不能假定解码链在何时恢复。"""

    next_recovery_time_seconds: Optional[Fraction]
    """若恢复点已知，其在公共时间线上的精确 PTS（秒）。"""


@dataclass(frozen=True)
class RecoveryObserved:
    """之前一个跳过范围的恢复点终于到达时发出。

    与 :class:`SkippedRange` 配对：让"跳过影响到哪里为止"在恢复包
    实际到来时再次得到确认，而不是要求下游一直猜。
    """

    kind: str = field(default="recovery", init=False)
    segment_id: Any
    track_id: Any
    output_track_id: Any
    after_skipped_first_seq: int
    after_skipped_last_seq: int
    recovery_seq: int
    recovery_time_seconds: Fraction


@dataclass(frozen=True)
class WaitingRange:
    """轨道当前 *已知存在但尚未交付* 的序号区间状态。

    注意它描述的是序号空洞（解码顺序），不是 PTS 回跳：
    B 帧造成的显示时间回跳不会产生 WaitingRange。
    """

    kind: str = field(default="waiting", init=False)
    segment_id: Any
    track_id: Any
    output_track_id: Any
    first_seq: int
    """第一个缺失序号（frontier），交付从这里开始重新推进。"""

    buffered_beyond: int
    """已到达但因空洞而压在后面的包数量（它们占着等待预算）。"""

    highest_observed_seq: Optional[int]
    """目前观察到的最大序号，None 表示一个包都还没到。"""


@dataclass(frozen=True)
class SegmentClosed:
    """一个分段在全部轨道上都已排空（没有任何等待范围）后发出。

    收到它意味着该分段不会再有任何事件；分段结束时仍缺失的部分
    已经作为 reason=END_OF_SEGMENT 的 :class:`SkippedRange` 交代过。
    """

    kind: str = field(default="segment_closed", init=False)
    segment_id: Any
    output_track_ids: tuple[Any, ...]


#: pull/events() 可能返回的事件联合类型。
Event = Any


# --------------------------------------------------------------------------- #
# submit 的即时回执
# --------------------------------------------------------------------------- #


class SubmitOutcome(enum.Enum):
    ACCEPTED = enum.auto()
    """包已被内核接收（可能已交付，也可能进入等待缓存，查 status()）。"""

    DUPLICATE = enum.auto()
    """该序号已交付/缓存，重复提交被忽略，载荷未被采用。"""

    LATE_AFTER_SKIP = enum.auto()
    """该序号落在已经确认跳过的区间内，包不再参与时间线，载荷被拒绝。"""

    LATE_AFTER_CLOSE = enum.auto()
    """该序号属于分段结束时确认缺失的尾部，分段已经封口。"""


# --------------------------------------------------------------------------- #
# 状态快照
# --------------------------------------------------------------------------- #


class TrackInputState(enum.Enum):
    OPEN = enum.auto()
    """可以继续 submit。"""

    WAITING = enum.auto()
    """可以提交，但有序号空洞，交付被卡住。"""

    CLOSED = enum.auto()
    """调用方已结束该分段（此轨道），不再接受包。"""


@dataclass(frozen=True)
class TrackStatus:
    segment_id: Any
    track_id: Any
    output_track_id: Any
    state: TrackInputState
    can_input: bool
    frontier_seq: Optional[int]
    """下一个将被交付的序号。"""

    highest_observed_seq: Optional[int]
    buffered_count: int
    buffered_bytes: int
    budget_limit: Optional[int]
    waiting: bool
    confirmed_skipped: int
    delivered: int


@dataclass(frozen=True)
class SegmentStatus:
    segment_id: Any
    output_start_seconds: Fraction
    output_track_ids: tuple[Any, ...]
    closed: bool
    drained: bool
    """True 表示全部轨道排空，SegmentClosed 已发出或可发出。"""


@dataclass(frozen=True)
class ResourceStatus:
    buffered_packets: int
    """因空洞等待、未交付的包总数（受等待预算限制）。"""

    buffered_bytes: int
    in_flight_events: int
    """已产出但调用方尚未取走的事件数（受输出队列容量限制，慢封装器
    会在这里形成背压）。"""

    output_queue_capacity: Optional[int]
    open_segments: int
    open_tracks: int


@dataclass(frozen=True)
class StatusSnapshot:
    tracks: tuple[TrackStatus, ...]
    segments: tuple[SegmentStatus, ...]
    resources: ResourceStatus
    cancelled: bool
    finished: bool
