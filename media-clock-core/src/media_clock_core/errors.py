"""Exception hierarchy for media-clock-core.

所有异常都属于编程/协议错误（调用方传错身份、超预算、状态不对）；
"缺包"、"迟到"、"时间戳不可换算" 都不是异常，而是通过事件契约上报。
"""

from __future__ import annotations


class MediaClockError(Exception):
    """Base class for every error raised by the kernel."""


class IdentityError(MediaClockError):
    """Unknown segment/track identity or an identity reused in a wrong state."""


class StateError(MediaClockError):
    """The operation is not valid for the current lifecycle state.

    例如：对已经结束的分段提交包、在 close 之后继续 submit。
    """


class BudgetExceeded(MediaClockError):
    """The configurable wait budget would be exceeded by accepting a packet.

    内核不替调用方决定丢哪些包，因此超预算时拒绝接收；调用方应当
    显式调用 :meth:`Timeline.skip` 划定缺口，或提高预算 / 先消费输出。
    """

    def __init__(self, *, track_id: object, limit: int, current: int):
        self.track_id = track_id
        self.limit = limit
        self.current = current
        super().__init__(
            f"wait budget for track {track_id!r} exhausted: "
            f"{current} buffered packets, limit is {limit}"
        )


class Cancelled(MediaClockError):
    """A blocking operation returned because the kernel was cancelled."""
