"""媒体时间内核的公开数据类型与契约。

本模块只描述 *包的时间、身份、交付顺序和可用性*，不涉及编解码细节。
所有时间量都是整数 ticks 或 :class:`fractions.Fraction` 秒；本包内
任何公开路径都不允许用 float 累加时间。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Generic, Optional, TypeVar

__all__ = [
    # 身份
    "TrackId",
    "SegmentId",
    "SeqNum",
    "MediaKind",
    "TrackSpec",
    "SegmentSpec",
    # 输入
    "InputPacket",
    # 时间
    "Tick",
    "TimingStatus",
    "CorrectedTiming",
    # 可用性
    "Usability",
    # 输出事件
    "KernelEvent",
    "PacketDelivered",
    "RangeSkipped",
    # 反馈
    "FeedResult",
    "FeedOutcome",
    # 状态
    "TrackState",
    "TrackStatus",
    "RangeStatus",
    "ResourceStatus",
    "KernelStatus",
]

TrackId = str
SegmentId = str
# 采集序号：采集设备每重新计时一次（一个新段）就从 0 开始单调递增。
# 段边界由调用方显式声明（SegmentSpec），内核绝不靠“时间变小”猜测边界。
SeqNum = int

PayloadT = TypeVar("PayloadT")


class MediaKind(enum.Enum):
    """轨道媒体类别，只用于决定“跳过后的解码可用性”传播模型。"""

    VIDEO = "video"
    AUDIO = "audio"


class TimingStatus(enum.Enum):
    """时间字段换算结果。与载荷能否解码完全无关。"""

    EXACT = "exact"        # 原始值在输出时间基准下可精确表示
    ROUNDED = "rounded"    # 无法精确表示，已按既定舍入规则取整，误差已记录
    UNMAPPED = "unmapped"  # 该时间字段缺失（如 None），无时间可换算


class Usability(enum.Enum):
    """载荷对下游解码器的可用性判断（只描述 *时间内核掌握的* 依赖事实）。

    时间字段能不能换算（:class:`TimingStatus`）与这个枚举是两件独立的事：
    时间被四舍五入的包仍可解码；处在丢失关键帧阴影里的包即使时间完全精确
    也解码不出正确画面。最终丢弃/降级策略由下游决定。
    """

    INTACT = "intact"      # 就内核已知依赖关系而言，载荷可正常送入解码器
    DEPENDENCY_BROKEN = "dependency_broken"
    # 该包之前有被确认跳过的包，且按本轨道依赖模型它解码结果不可信。
    # 视频：阴影从前一个未被跳过的关键帧之后的丢包起，到下一个关键帧止。
    # 音频：包之间无帧间预测，永远不会是 DEPENDENCY_BROKEN。


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackSpec:
    """注册一条输出轨道。

    ``default_time_base`` 是该轨道采集时间戳的默认时间基准（秒/tick）；
    单个包仍可用自己的 time_base 覆盖，用于音频/视频单位不同的情况。
    """

    track_id: TrackId
    kind: MediaKind
    default_time_base: Fraction
    output_time_base: Fraction
    # 采集时间基准 -> 输出时间基准换算时的舍入方式。
    # 默认 round-half-even（银行家舍入），误差写入 residual_seconds。
    # 可选 "floor" / "ceil"。
    rounding: str = "half_even"


@dataclass(frozen=True)
class SegmentSpec:
    """声明一段采集素材在公共输出时间线上的位置。

    ``output_start`` 是外部确认过的输出起点（秒，Fraction）。音频轨和视频轨
    共享同一个段锚点，因此两条轨道分别喂包也不会丢失同段对齐意图；
    ``track_capture_offsets`` 仅用于调用方明确提供的额外逐轨采集偏移
    （默认 0），内核不发明偏移。
    """

    segment_id: SegmentId
    output_start: Fraction
    track_capture_offsets: dict[TrackId, Fraction] = field(default_factory=dict)


@dataclass(frozen=True)
class WaitingBudget:
    """等待预算：硬性限制“因乱序而占住未交付”的资源。

    - ``max_packets`` / ``max_bytes`` 只统计已接收、但因前面序号缺失而无法
      交付的包（in-order 到达的包会立刻被下游取走，不长期占内存）。
    - ``default_timeout``：某段某轨的缺口等待墙钟超时；None 表示可无限等，
      直到调用方显式 skip 或 finish。
    """

    max_packets: int = 4096
    max_bytes: int = 256 * 1024 * 1024
    default_timeout: Optional[float] = None


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputPacket(Generic[PayloadT]):
    """上游（读取器）交给内核的一个采集包。

    序号是交付排序依据；pts/dts 是显示/解码时间，绝不用来排序。
    pts/dts 允许为负数（编码器重排序延迟，如 -2），也允许为 None。
    ``payload`` 对内核不透明，内核只读不写不拷贝语义上不改内容。
    """

    segment_id: SegmentId
    track_id: TrackId
    seq: SeqNum
    pts: Optional[int]
    dts: Optional[int]
    duration: Optional[int]
    time_base: Optional[Fraction] = None  # None -> 用 TrackSpec.default_time_base
    is_keyframe: bool = False
    payload: Optional[PayloadT] = None
    payload_bytes: int = 0  # 用于预算统计；载荷可能不是字节串，由调用方报告大小


# ---------------------------------------------------------------------------
# 修正后的时间
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tick:
    """某个时间字段在输出时间基准下的换算结果，保留可检查的误差。

    - ``ticks``：写入容器的整数 tick（可为负，DTS 就是这样）。
    - ``exact_seconds``：未经取整的精确秒值（相对公共时间线原点）。
    - ``residual_seconds`` = 取整表示值 - 精确值（Fraction）。永远不把无法
      精确表示的时间伪装成原值；调用方可以累加它检查长期误差。
    """

    ticks: int
    exact_seconds: Fraction
    residual_seconds: Fraction

    @property
    def represented_seconds(self) -> Fraction:
        return self.exact_seconds + self.residual_seconds


@dataclass(frozen=True)
class CorrectedTiming:
    """一个包在公共输出时间线上的完整时间关系。"""

    output_time_base: Fraction
    pts: Optional[Tick]
    dts: Optional[Tick]
    duration: Optional[Tick]
    pts_status: TimingStatus
    dts_status: TimingStatus
    duration_status: TimingStatus
    # 同一 (段, 轨) 内为了消除取整抖动而做的单调性调整量（Fraction 秒，>=0）。
    # 不为 0 时表示纯按比例换算会得到非递增 DTS，已在“精确顺序仍然单调”的
    # 前提下钳到 +1 tick；若精确顺序本身已逆序，会改为报错事件而非调整。
    monotonic_adjustment_seconds: Fraction = Fraction(0)

    @property
    def worst_residual_seconds(self) -> Fraction:
        """本包表示误差的最大绝对值（PTS/DTS/时长三者）。"""
        worst = Fraction(0)
        for t in (self.pts, self.dts, self.duration):
            if t is not None and abs(t.residual_seconds) > abs(worst):
                worst = t.residual_seconds
        return worst


# ---------------------------------------------------------------------------
# 输出事件
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelEvent:
    segment_id: SegmentId
    track_id: TrackId


@dataclass(frozen=True)
class PacketDelivered(KernelEvent, Generic[PayloadT]):
    """已经确定交付次序、完成时间修正的包。

    载荷原样透传；原始字段全部保留在 ``origin`` 中，修正值在 ``timing`` 中，
    二者之间的关系可逐条检查。
    """

    seq: SeqNum
    origin: InputPacket[PayloadT]
    timing: CorrectedTiming
    usability: Usability
    # 该包是否正处在某个被确认跳过范围造成的解码阴影里（与 usability 冗余，
    # 但显式给出，方便下游不理解依赖模型时直接判断）。
    shadowed_by_gap: bool = False
    # 精确（未经取整的）时间关系出现的异常，例如 DTS 在精确尺度上逆序。
    # 内核不擅自修复这类问题，只在这里暴露给调用方核对段锚点/输入归属。
    warnings: tuple[str, ...] = ()

    @property
    def payload(self) -> Optional[PayloadT]:
        return self.origin.payload


@dataclass(frozen=True)
class RangeSkipped(KernelEvent):
    """一个被“确认跳过”的序号范围（不是等待，也不是悄悄丢帧）。

    由调用方 :meth:`MediaClockKernel.confirm_skip` 或段结束冲刷产生。
    ``decode_impact`` 说明对后续解码可用性的影响，随数据流交给下游，
    而不是只写一条丢帧日志。
    """

    start_seq: SeqNum  # inclusive
    end_seq: SeqNum    # exclusive
    reason: str        # "confirmed" / "unfinished-at-end" / "timeout"
    # 阴影结束序号（exclusive）：在 [end_seq, shadow_until_seq) 内交付的
    # 视频包 usability=DEPENDENCY_BROKEN；下一个关键帧处阴影解除。
    shadow_until_seq: Optional[SeqNum]
    contained_keyframe: bool  # 被跳过范围中是否出现过关键帧（影响阴影判定的说明）


# ---------------------------------------------------------------------------
# feed 反馈
# ---------------------------------------------------------------------------


class FeedOutcome(enum.Enum):
    ACCEPTED = "accepted"          # 已接收；可能已产生可 drain 的事件
    BUDGET_EXCEEDED = "budget"     # 等待预算已满，调用方需先 drain 或 skip
    CANCELLED = "cancelled"        # 管道已取消
    FINISHED = "finished"          # 该 (段,轨) 已结束，不再接受输入
    UNKNOWN_TARGET = "unknown_target"  # 段未声明或轨道未注册


@dataclass(frozen=True)
class FeedResult:
    outcome: FeedOutcome
    delivered: int = 0  # 本次调用同步产生了多少个可交付事件（提示用，精确事件走 drain）


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------


class TrackState(enum.Enum):
    OPEN = "open"        # 仍可输入
    FINISHED = "finished"  # 调用方已结束本 (段,轨)，等待内容已冲刷
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class RangeStatus:
    """一个序号范围的交付状态说明。"""

    kind: str  # "delivered" / "waiting" / "skipped"
    start_seq: SeqNum
    end_seq: SeqNum  # exclusive；waiting 的 end 表示当前已确认的连续前缘+1


@dataclass(frozen=True)
class TrackStatus:
    segment_id: SegmentId
    track_id: TrackId
    state: TrackState
    # 下一个等待接收的采集序号（重复推送等于它的包会被拒绝/忽略）。
    next_contiguous_seq: SeqNum
    delivered: int
    skipped: int
    holding: int                 # 因乱序占住未交付的包数
    holding_bytes: int
    waiting_ranges: tuple[RangeStatus, ...]   # 仍在等待更多输入的缺口
    skipped_ranges: tuple[RangeStatus, ...]
    can_input: bool
    # 该轨已经完全确定的时间关系上界：<= 此序号的包要么交付要么确认跳过。
    determined_through: SeqNum
    # 跨包累计表示误差（Fraction 秒）：所有已交付包的取整残差之和。
    # 用来核对“连续换算/多段接续后有没有逐渐偏移”，而不是逐包自欺。
    cumulative_pts_residual_seconds: Fraction = Fraction(0)
    cumulative_dts_residual_seconds: Fraction = Fraction(0)
    cumulative_monotonic_adjustment_seconds: Fraction = Fraction(0)
    rounded_packets: int = 0


@dataclass(frozen=True)
class ResourceStatus:
    """操作结束/进行中仍未释放的资源，供调用方判断泄漏。"""

    held_packets: int
    held_bytes: int
    pending_events: int
    open_tracks: int
    timeout_waiters: int
    cancelled: bool


@dataclass(frozen=True)
class KernelStatus:
    tracks: tuple[TrackStatus, ...]
    resources: ResourceStatus

    def tracks_accepting_input(self) -> tuple[tuple[SegmentId, TrackId], ...]:
        return tuple((t.segment_id, t.track_id) for t in self.tracks if t.can_input)
