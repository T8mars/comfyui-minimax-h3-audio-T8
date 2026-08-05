# MiniMax H3 Audio T8

面向当前 ComfyUI 原生 MiniMax H3 的独立音频节点包。它不修改旧的
`comfyui-vrgamedevgirl`，节点统一位于 `T8/MiniMax H3/Audio`。

本包不是把源音频简单塞进 latent：它按 ComfyUI 当前 H3 实现维护媒体展示顺序、
`<Picture N>` / `<Video N>` / `<Audio N>` 标签、联合 AV latent、首尾关键帧、参考媒体和
噪声掩码之间的契约。

## 节点

| 节点 | 用途 |
|---|---|
| MiniMax H3 Audio Conditioning (T8) | T2VA、I2VA、FL2VA、L2VA、Ref2VA 和关键帧+参考媒体 Hybrid 的统一条件节点 |
| MiniMax H3 Audio Latent Control (T8) | 对已有 H3 AV latent 锁定或重绘源音频，并保留已有视频 mask |
| MiniMax H3 Duration Planner (T8) | 把场景时间换算成 24fps、`17n+5` 的渲染窗口和最终裁切参数 |
| MiniMax H3 Audio Window (T8) | 直接切取/补零 AUDIO，短场景可自动扩展到 124 帧训练下限 |
| MiniMax H3 Prompt Tags (T8) | 把 `Image 1`、`Audio1` 等写法规范为官方标签并严格校验编号 |
| MiniMax H3 AV Decode (T8) | 用视频/音频 VAE 分别解码联合 AV latent |
| MiniMax H3 Audio Mix (T8) | 源音轨与模型生成音轨重采样、增益、ducking、峰值限制后混合 |
| MiniMax H3 Output Trim (T8) | 把 Planner 的时间窗口同时应用到解码帧和音频 |
| MiniMax H3 Preflight (T8) | 在采样前检查模型、尺寸、帧数、音频、参考数量和参考视频时长 |

## 四种音频模式

| 模式 | 目标音频 latent | 源音频是否作为参考 | 适用场景 |
|---|---|---|---|
| `lock_source` | 源音频，denoise mask=0 | 默认是 | 画面严格跟随音频，最终保留原音轨 |
| `remix_source` | 源音频，按 strength 重绘 | 默认是 | 保留节奏/语音结构，同时让模型改造声音 |
| `reference_only` | 空白、完整生成 | 是 | 源音频只提供语义/节奏参考，输出使用模型音频 |
| `native` | 空白、完整生成 | 否 | 纯 H3 原生音画联合生成，无需输入音频 |

`drive_audio` 是给模型的驱动轨，`final_audio` 是最终 mux 的干净轨。二者分开可以让你把
外部人声分离器得到的 vocal stem 用作驱动，同时把原混音或另一条 stem 送到最终输出；
本包不会假装内置了一个未经验证的分离模型。

## 推荐连接

锁定原音频生成画面：

1. `Load Audio -> MiniMax H3 Audio Window (T8)`。
2. `context_audio`、视频 VAE、音频 VAE、CLIP 接入统一 Conditioning，选择 `lock_source`。
3. Conditioning 的 `positive` 和 `av_latent` 进入原生 H3 sampler。
4. sampler 输出进入 `MiniMax H3 AV Decode (T8)`。
5. 解码 frames、Conditioning 的 `mux_audio`、Audio Window 的两个 trim 输出进入
   `MiniMax H3 Output Trim (T8)`。
6. 将裁切后的 frames/audio 交给 VideoHelperSuite 或你现有的保存节点。

短场景开启 `ensure_minimum_context` 时，节点会添加上下文，但不会再让动作时间轴悄悄漂移：
`prompt_timing_note` 给出主场景在渲染窗口中的真实开始/结束时间，最终 trim 参数再恢复用户请求时长。

## 媒体编号

H3 的展示顺序是：所有 Picture；然后每个参考视频（其声轨 Audio 标签位于对应 Video
标签前）；最后是独立 Audio。因而两个参考视频都带声轨时，主驱动音频会是 `<Audio 3>`，
而不是 `<Audio 1>`。统一 Conditioning 会输出完整 `media_map_json`，并把 prompt 中配置的
`prompt_primary_audio_ordinal` 自动映射到主驱动音频的真实编号。设为 0 可关闭重映射。

严格模式会拒绝引用未连接媒体的标签，避免模型收到看似合法、实际无对应条件的 prompt。

## H3 边界

- 固定 24fps，帧数向上对齐到 `17n+5`。
- 当前模型近似训练区间为 124–362 帧；区间外允许规划但 Preflight 会警告。
- 生成画布像素面积不能超过 `768*1344`，宽高必须是 32 的倍数。
- 原生 H3 目前只支持 batch size 1。
- 引用上限：9 张 Picture、3 个 Video、3 个独立 Audio；参考视频官方建议 2–15 秒。
- `Hybrid` 同时使用精确首/尾帧和参考媒体。节点包含针对当前 ComfyUI `PackedLayout`
  行为的运行时契约检查；上游若改变结构会明确停止，而不是生成错位条件。

## 示例与测试

API 示例见 `examples/audio_lock_api.json`。替换里面的模型、VAE、CLIP 和音频文件名后即可使用；
保存节点使用已安装的 VideoHelperSuite。

从 ComfyUI 根目录、使用启动 ComfyUI 的同一 Python 环境运行：

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -q .\custom_nodes\minimax-h3-audio-T8
```

本包没有额外 pip 依赖，使用 ComfyUI 自带的 PyTorch、torchaudio 和原生 MiniMax H3 实现。
