# media-clock-core

进程内、增量的**媒体时间内核**。它只负责四件事：

- **时间**：采集时间戳 → 公共输出时间线的精确有理换算；
- **身份**：多段素材、多轨道在同一输出时间线上的归属与对齐；
- **交付顺序**：按采集序号（而非 PTS）整理网络乱序；
- **可用性契约**：哪些包确定交付、哪些被确认跳过、哪些仍在等待，以及跳过对后续解码的影响。

它**不是**播放器，**不是**命令行转码工具，也不改写任何编码载荷。上游是读取器，下游是真正的封装器（如 PyAV muxer）。

## 为什么需要它

带 B 帧的 H264，编码包的显示时间（PTS）在解码顺序上会回跳，例如：

```
PTS:  0, 2, 1, ...     （显示顺序重排）
DTS: -2,-1, 0, ...     （解码时间可以是负数）
time_base: 1/25 s
```

封装进 MP4 后流时间基准变成 `1/12800`，但解码得到的画面仍然每 `1/25` 秒一个。
由此得到几条不能违背的原则：

- **不能按 PTS 排序后再交给封装器**——封装/解码要的是解码顺序的包；
- **不能把负 DTS 当坏数据删掉**——它是编码器重排序延迟的正常结果；
- **显示时间回跳不等于迟到/丢帧**——B 帧依赖导致的回跳与网络缺包是两回事；
- **段边界不能靠“看到时间变小”猜**——它是采集设备重新计时的真实边界，必须由调用方显式声明。

本内核把这些做成显式契约，而不是埋在排序逻辑里。

## 精确有理时间，不做浮点累加

所有时间运算都用整数 ticks 与 `fractions.Fraction`（秒）。对 `30000/1001` 这类帧率，无论连续换算多少包、跨多少段，都不会产生浮点漂移。

无法在输出时间基准下精确表示的值会被取整，但**绝不伪装成原值**：每个时间字段返回

- `ticks`：写入容器的整数；
- `exact_seconds`：未经取整的精确秒；
- `residual_seconds`：取整表示值 − 精确值。

每条轨道的 `TrackStatus` 还汇总累计 PTS/DTS 残差与累计单调性修正量，供长期漂移审计。

## 核心概念

| 概念 | 说明 |
| --- | --- |
| `SegmentSpec` | 段身份 + 外部确认的输出起点（公共时间线上的位置）。边界只能显式声明。 |
| `TrackSpec` | 轨道身份、媒体类别（video/audio）、采集时间基准、输出时间基准、舍入方式。 |
| `InputPacket` | 段/轨/采集序号 + 原始 PTS/DTS/duration/time_base + 不透明载荷。 |
| `PacketDelivered` | 已确定交付：`origin`（原始包）+ `timing`（修正时间）+ `usability`（解码可用性）。 |
| `RangeSkipped` | 被确认跳过的序号区间，带“对后续解码的影响范围”，随数据流交下游。 |

音频轨与视频轨计时单位可以不同，但它们共享同一个段锚点，因此两条轨道分别喂包也不会丢失同段对齐意图。

## 交付顺序与三种状态

内核按**采集序号**整理交付（网络读取器乱序调用时，在已知范围内重排），并区分：

- **已交付**：序号前缘连续、时间已修正的 `PacketDelivered`；
- **被确认跳过**：调用方 `confirm_skip(...)` 或段结束/超时冲刷产生的 `RangeSkipped`；
- **等待更多输入**：前缘缺口，经 `status().tracks[*].waiting_ranges` 可见，标注属于哪段哪轨。

`status()` 返回每条 (段,轨) 的：能否继续输入、已确定到哪个序号、占住了多少包/字节、等待与跳过区间，以及未释放资源。

## 跳过数据时，把解码影响交给下游

跳过一段数据不是加一行丢帧日志。视频存在帧间预测：缺口之后到下一个关键帧之前的包都无法正确解码。内核在这些包上标

```
usability = DEPENDENCY_BROKEN   （shadowed_by_gap=True）
```

关键帧处自动恢复 `INTACT`；音频无帧间预测，永远不会被标为 broken。`RangeSkipped.shadow_until_seq` 在确认跳过时若关键帧已可见会给出阴影上界。**是否真的丢弃这些包由下游封装器决定**（参考测试里的 `Mp4MuxerSink` 默认丢弃 broken 包并记录清单）。

## 增量、背压、取消、结束

- in-order 包在 `feed` 内立即转为事件，不缓存整文件；只有因乱序而等待缺口的包才占住资源。
- `WaitingBudget(max_packets, max_bytes, default_timeout)` 硬性限制占住资源；超预算返回 `BUDGET_EXCEEDED`，`sweep_timeouts()` 可把超时缺口冲刷为 `RangeSkipped(reason='timeout')`。
- `ClockPipeline`（asyncio）在预算满或事件队列满时把背压沿链路传到读取侧；慢封装器取件自动驱动。
- `cancel()` 解除读取与消费等待、释放占住载荷；`aclose()`/`finish_track()` 把仍在等待的部分冲刷为确定结果，不会留下永不结束的读取。

## 用法

同步内核：

```python
from fractions import Fraction
from media_clock_core import (MediaClockKernel, TrackSpec, SegmentSpec,
                              MediaKind, InputPacket, WaitingBudget,
                              PacketDelivered, RangeSkipped)

k = MediaClockKernel(WaitingBudget(max_packets=512))
k.register_track(TrackSpec("v", MediaKind.VIDEO,
                           default_time_base=Fraction(1, 25),
                           output_time_base=Fraction(1, 12800)))
k.open_segment(SegmentSpec("cam0/run1", Fraction(0)))

k.feed(InputPacket(segment_id="cam0/run1", track_id="v", seq=3,
                   pts=1, dts=1, duration=1,
                   time_base=Fraction(1, 25), is_keyframe=False,
                   payload=raw_bytes, payload_bytes=len(raw_bytes)))
# 乱序到达没关系；前缘缺口未补时 seq=3 只占住、不交付

for ev in k.drain():
    if isinstance(ev, PacketDelivered):
        t = ev.timing
        mux(ev.payload, pts=t.pts.ticks, dts=t.dts.ticks,
            time_base=t.output_time_base)   # 负 dts 原样保留
    elif isinstance(ev, RangeSkipped):
        ...                                   # 下游决定如何处理解码阴影

k.finish_track("cam0/run1", "v")             # 等待部分冲刷为确定结果
```

异步管道（背压 + 慢封装器）：

```python
pipe = ClockPipeline(WaitingBudget(max_packets=512), max_pending_events=128)
pipe.register_track(spec); pipe.open_segment(seg)

async for ev in pipe.events():      # 慢消费者在这里自然产生背压
    await muxer.handle(ev)

await pipe.aclose()                 # 或 pipe.cancel()
```

## 真实媒体往返验证

测试不停留在整数排序：`tests/test_media_roundtrip.py` 与 `tests/test_pipeline_media.py` 走完整链路——

**PyAV 编码（带 B 帧 H264 / AAC）→ 模拟网络乱序喂入内核 → 内核修正时间 → PyAV 封装进 MP4 → PyAV 解码 → 直接检查画面出现时间与画面内容**。

覆盖：

- 25fps B 帧、负 DTS、乱序到达：解码画面仍每 `1/25` 秒一个、显示顺序正确；
- `30000/1001`：每帧 PTS 精确为 `i*3003`（tb=1/90000），时间 `i*1001/30000` 秒，无漂移；
- 多段接续到同一时间线：跨段处帧间隔仍精确为 `1/25`，无浮点缝隙；
- 确认跳过 → 解码阴影逐包标注，新段新关键帧恢复 `INTACT`；
- 音频（1/48000）与视频（1/25→1/12800）共享段锚点，起点对齐；
- 异步管道 + 小队列 + 慢封装器的增量往返，以及取消时真实编码载荷被释放；
- 并发交付顺序压测（2000 次交错唤醒，顺序恒为采集序号）。

```
pip install av pytest pytest-asyncio numpy
PYTHONPATH=src pytest tests/
```

## 范围与非目标

- 不改载荷、不重排编码字节、不做编解码策略；
- 不做播放器、不做转码 CLI；
- 编解码/封装由成熟依赖（PyAV）完成，本项目只产出时间、身份、交付顺序与可用性契约。
