# media-clock-core

一个位于**读取器与封装器之间**的进程内媒体时间内核。它不做播放器，也不做
命令行转码工具；只负责四件事：

1. **时间** —— 采集包的原始 PTS/DTS/duration 到公共时间线的**精确有理**映射；
2. **身份** —— 分段 / 轨道 / 采集序号，分段边界来自外部确认而非时间戳猜测；
3. **交付顺序** —— 按采集序号（解码顺序）整理网络乱序，区分已交付 / 已确认跳过 / 等待中；
4. **解码可用性契约** —— 跳过数据对后续解码的影响结构化地交给下游，而不是记一个丢帧日志。

内核本体零第三方依赖（只用标准库）；PyAV 仅用于可选适配层和真实媒体往返测试。

---

## 为什么需要它

带 B 帧的 H.264，编码器吐出的包顺序是**解码顺序**，显示时间会回跳：

```
display(PTS):  0, 3, 1, 2, ...      # 显示顺序重排
decode(DTS):   -2, -1, 0, 1, ...    # 开头 DTS 为负
time_base:     1/25
```

封装进 MP4 后流时间基准变成 `1/12800`，FFmpeg 整体平移负 DTS 并写 edit list，
但真正解码出的画面仍然每隔 `1/25` 秒出现。由此得到两条硬约束：

- **不能**把包按显示时间（PTS）排好再交给封装器——那会破坏解码依赖；
- **不能**把负 DTS 当坏数据删掉——它是 B 帧的正常形态。

本内核按采集序号（= DTS 顺序）交付，保留 PTS 回跳和负 DTS，只做时间基准换算
与分段平移。本仓库的测试用真实 libx264 编码 + PyAV 封装 + 真实解码逐帧验证
（见 `tests/test_media_roundtrip.py`），不是对几组整数排序后宣称完成。

---

## 快速上手

```python
from fractions import Fraction
from media_clock_core import (
    Timeline, OutputTrackSpec, InputTrackSpec, InputPacket,
    PayloadUsability,
)

tl = Timeline(
    default_wait_budget_packets=128,   # 每条输入轨的等待预算硬上限
    output_queue_capacity=128,         # 慢封装器背压
)
# 下游封装器的输出轨（时间基准由调用方指定）
tl.register_output_track(OutputTrackSpec("video", Fraction(1, 12800)))
tl.register_output_track(OutputTrackSpec("audio", Fraction(1, 48000)))

# 分段：外部确认的输出起点；音视频同段对齐，各自有时间基准
tl.open_segment("seg-1", Fraction(0), [
    InputTrackSpec("cam-v", "video", first_capture_seq=0),
    InputTrackSpec("mic-a", "audio", first_capture_seq=0),
])

tl.submit(InputPacket(
    segment_id="seg-1", track_id="cam-v", capture_seq=0,
    pts=0, dts=-2, duration=1, time_base=Fraction(1, 25),
    payload=raw_bytes, keyframe=True,
))

for event in tl.events():
    if event.kind == "packet":
        mux(event)            # -> av muxer，时间字段见 event.output_*
    elif event.kind == "skipped":
        handle_gap(event)      # 缺口 + 恢复点信息，交给下游
    elif event.kind == "waiting":
        report_wait(event)     # 已知空洞，在等数据
```

### PyAV 适配层（可选）

```python
from media_clock_core.av_adapter import input_packet_from_av, apply_to_av_packet

ip = input_packet_from_av(reader_packet, segment_id, track_id, capture_seq)
tl.submit(ip)
...
out_packet = av.Packet(event.payload)
out_packet.stream = output_stream
apply_to_av_packet(out_packet, event)   # 只改时间字段，载荷字节不动
container.mux(out_packet)
```

> 比特流格式转换（如 H.264 AVCC↔Annex-B）是**有状态的封装器侧职责**，
> 应在封装边界用 `h264_mp4toannexb` 之类的 bitstream filter 完成；
> 时间内核不碰载荷、不做这种转换。测试里的 `MuxerSink` 示范了这一分工。

---

## 输出事件契约

| `kind` | 含义 |
|---|---|
| `packet` (`DeliveredPacket`) | 已整理次序、完成时间修正、可交给封装器。携带 `output_pts/dts/duration`、与原始包的完整映射 `correction`、解码可用性 `decode_availability`。 |
| `skipped` (`SkippedRange`) | 调用方显式确认、或分段结束时确认不会到达的序号区间。携带原因和**恢复点**（缺口后第一个关键帧序号与时间，未知则为 `None`）。 |
| `recovery` (`RecoveryObserved`) | 之前跳过区间的恢复关键帧实际到达，缺口影响到此为止。 |
| `waiting` (`WaitingRange`) | 当前已知但未交付的序号空洞（按解码序号，**不是** PTS 回跳）。同一空洞只报一次。 |
| `segment_closed` (`SegmentClosed`) | 该分段所有轨道排空，不会再有事件。 |

### 解码可用性（与时间字段正交）

`DecodeAvailability`：`INTACT` / `DEGRADED_AFTER_GAP` / `RECOVERY_POINT` /
`CORRUPTED` / `TIME_UNKNOWN`。载荷损坏与时间字段不可换算是两件独立的事，
内核分别标记，但两种包都照常交付——**是否喂给解码器由下游决定**。

### 精确时间与误差公开

- 内核内部全部使用 `fractions.Fraction`，**没有 float 累加**；
  `30000/1001` 连续换算、多段接续不会漂移。
- 只有映射到整数输出 tick 时才量化，默认舍入与 FFmpeg `av_rescale_q` 一致。
- 每个包的 `correction.pts/dts/duration` 都是 `AffineMapping`，保留
  原始 ticks、原始时间基准、原始秒数、修正秒数、输出 ticks、**残差**。
- `event.residual_seconds` 是该包时间表示误差的精确有理上界。无法精确
  表示时状态为 `QUANTIZED`，残差非零并如实公开；字段缺失则是
  `UNCONVERTIBLE`，绝不伪装成 0。

---

## 资源、背压与生命周期

- **增量接口**：`submit` / `pull`（或 `events()`）都是进程内逐包调用，
  内核不缓存完整媒体。
- **等待预算**：每条输入轨的未交付缓存受 `wait_budget_packets` /
  `wait_budget_bytes` 硬限制。超预算时 `submit` 抛 `BudgetExceeded`——
  内核不替调用方决定丢哪些包；调用方应当显式 `skip(...)` 划定缺口。
- **慢封装器背压**：输出事件队列有界（`output_queue_capacity`），
  消费者不及时取走时生产者 `submit` 阻塞。
- **取消**：`cancel()` 同时唤醒阻塞的生产者与消费者（抛 `Cancelled`）。
  `Timeline` 支持 `with`，异常退出自动 cancel。
- **结束**：`close_segment(seg)` 把该段仍等待的已知范围变成
  `END_OF_SEGMENT` 跳过事件并继续排空，不会留下永不结束的读取；
  `close()` 隐式关闭所有分段。全部排空后阻塞式 `pull` 抛 `EndOfTimeline`。
- **状态查询**：`status()` 返回每条轨能否继续输入、frontier、缓存包数/字节数、
  已交付/已跳过计数，以及分段是否排空、整体是否 finished、还有哪些资源占用。

### 关键设计决策

- **分段边界不猜测**：只在 `open_segment` 显式声明。PTS 回退可能是 B 帧依赖，
  绝不用"时间变小"来推断新分段。
- **采集序号起点显式声明**：`InputTrackSpec.first_capture_seq`
  （默认 0）。真实读取器在分段开始时就知道序列计数器起点，因此一个晚到的
  高序号包不会把交付前沿抬高、让更早的包被误判为重复/迟到。
- **分段门控**：同一输出轨上，只有最前面未排空分段允许交付；后段包即使全部
  到达也只缓存，保证 DTS/序号不跨段倒挂。

---

## 开发与测试

```bash
pip install -e '.[test]'
python -m pytest
```

测试分两层：

- `tests/test_timebase.py`、`tests/test_kernel.py`：精确有理运算、B 帧次序、
  网络乱序、缺口契约、等待预算、背压、取消、多段/多轨对齐（无需 PyAV）。
- `tests/test_media_roundtrip.py`：**真实媒体往返**——libx264 编 B 帧
  （含负 DTS / PTS 重排）、内核修正时间、真正的 PyAV MP4 封装、重新解码
  逐帧核对画面时间；覆盖多段采集设备重启计时、AAC+视频同段对齐、
  `30000/1001` 量化残差、跳缺后的解码可用性。
