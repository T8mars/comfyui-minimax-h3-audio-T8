# MiniMax H3 Audio T8

面向当前 ComfyUI 原生 MiniMax H3 的独立 T8 节点扩展。当前版本为 `1.3.3`，共注册
14 个节点，覆盖原生音画条件、音频控制与后处理、稳定双时钟采样、实验性多速率采样，
以及 Ref2VA 单图/多图参考的静态语义编辑。

节点按稳定性与用途分为三个菜单：

| 菜单 | 状态 | 内容 |
|---|---|---|
| `T8/MiniMax H3/Audio` | 稳定 | 音画条件、音频处理、预检、双时钟采样与 AV 解码 |
| `T8/MiniMax H3/Audio/Experimental` | 实验 | 视频宏步/音频微步的多速率联合采样 |
| `T8/MiniMax H3/Still/Experimental` | 实验 | Ref2VA 静态图像条件、预检与候选帧解码 |

本包不是把源音频简单塞进 latent：它按 ComfyUI 当前 H3 实现维护媒体展示顺序、
`<Picture N>` / `<Video N>` / `<Audio N>` 标签、联合 AV latent、首尾关键帧、参考媒体和
噪声掩码之间的契约。

## 安装与兼容性

将项目目录放入 ComfyUI 的 `custom_nodes/minimax-h3-audio-T8`，重启 ComfyUI 后即可在上述
菜单中找到节点。本项目没有额外 pip 依赖，复用 ComfyUI 自带的 PyTorch、torchaudio 和
MiniMax H3 实现；当前记录的验证基线为 ComfyUI `0.30.0`、提交 `a464ac335`、Python
3.10+。模型、VAE、CLIP 和可选 LoRA 仍需按具体任务自行安装。

`1.3.3` 在稳定双时钟节点末尾追加可选的采样器与调度器下拉框；原有三个控件的顺序、
默认双时钟 Euler、原生 flow sigma 和旧 API 缺省行为均保持不变。`1.3.2` 保留 `1.3.1`
对两代 H3 采样协议的兼容：旧版 ComfyUI 的 slope-scaled 音频速度，以及当前
`FLOW_AV` / `ModelSamplingAV` 的原始音频速度。兼容性由实际 H3 基模能力检测，不依赖用户
手动选择，也不会对新版 ComfyUI 再次应用音频 carry/scale。本版本还兼容
VideoHelperSuite 的延迟 `AUDIO Mapping`，用 H3 latent 契约识别视频/音频 VAE，并把画布
像素面积上限放宽到 `1920×1088 = 2,088,960`；超过旧 0.98M 档只提示显存风险，不再阻止执行。

## 项目目录

| 路径 | 内容 |
|---|---|
| `tools/` | MiniMax H3 Turbo LoRA 转换工具 |
| `docs/` | LoRA 使用说明与验证记录 |
| `examples/` | API 与 ComfyUI 前端工作流 |
| `artifacts/` | 历史发布包和代码迁移归档；已由 `.gitignore` 排除 |

项目源码、文档、工具和本地交付资产均以当前项目目录为唯一事实源，不依赖其他盘符
中的工程副本。模型权重不存放在本项目中，应继续使用 ComfyUI 的标准模型目录。

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
| MiniMax H3 Dual-Clock Sampler (T8) | 默认配置 12/3 shift、原生 flow sigma 与双时钟 Euler，也可选择当前 ComfyUI 的原生采样器和调度器 |
| MiniMax H3 Multi-Rate Sampler (EXP/T8) | 实验性视频宏步/音频微步采样；独立实现，不替换稳定双时钟节点 |
| MiniMax H3 Reference Image Edit (EXP/T8) | 用 Ref2VA 对单张主图进行语义编辑，并支持最多 8 张附加参考图 |
| MiniMax H3 Still Preflight (EXP/T8) | 检查单帧 OOD、画布、参考数量、模型和 VAE 契约 |
| MiniMax H3 Still Decode (EXP/T8) | 只解码视频 latent，并从 1/5/22/124 帧候选中选出一张图 |

`MiniMax H3 Audio Conditioning (T8)` 的 `task_type` 下拉框会显示中英双语说明：

| 选项 | 中文含义 |
|---|---|
| `auto` | 自动判断（按已连接输入） |
| `T2VA` | 文生音视频 |
| `I2VA` | 图生音视频（首帧） |
| `FL2VA` | 首尾帧生音视频 |
| `L2VA` | 尾帧生音视频 |
| `Ref2VA` | 参考生音视频 |
| `Hybrid` | 关键帧与参考媒体混合生成 |

中文仅用于前端显示，后端和 API 仍提交原有英文枚举，因此旧工作流与 API JSON 无需修改。

## EXP：参考图像编辑

`MiniMax H3 Reference Image Edit (EXP/T8)` 位于
`T8/MiniMax H3/Still/Experimental`，复用 H3 Ref2VA 的 Picture 条件生成静态候选。
`edit_image` 始终是 `<Picture 1>`；附加参考图依次成为 `<Picture 2>` 至 `<Picture 9>`。
Prompt 应明确每张图的职责，例如主体身份、服装、背景或光照。

目标模式：

- `direct_1_frame`：直接创建 `video latent_t=1`，成本最低，但严重偏离训练帧数；
- `micro_video_5_frames`：生成 H3 最短 5 帧，再在 Still Decode 中选帧；
- `short_video_22_frames`：生成下一档原生 `17n+5` 网格的22帧，视频 latent T=7，
  音频 latent T=37；比124帧便宜很多，但仍低于约124帧的训练下限；
- `trained_124_frames`：按近似训练下限生成 124 帧，作为质量基准，成本最高。

默认 `reference_strength=0.999` 与 H3 参考条件的原始噪声增强接近；降低该值会向参考
latent 注入更多噪声，可能增强重绘幅度，也可能损坏身份与构图。`generate_and_discard`
让联合模型正常生成短音频但最终不解码；`lock_silence` 锁定零音频，仅用于对照。

推荐链路：

1. 加载 H3 Ref2VA 模型、H3 Qwen3-VL CLIP 和视频 VAE；
2. 将主图和附加参考图接入 Reference Image Edit；
3. 同一个 `av_latent` 同时连接到双时钟采样设置与 `SamplerCustomAdvanced.latent_image`；
4. 采样输出接 Still Decode，再接 `SaveImage`。

本机现有 Ref2VA 是 pruned INT8，不能完整应用本项目转换的 Turbo LoRA；示例因此不加载
LoRA，并以 20 步作为结构基线。若以后安装非裁剪 Ref2VA，再单独进行 Turbo LoRA 对照。
这项能力是参考引导的语义重绘，不是 mask/inpainting，也不保证未编辑区域像素不变。
API 示例见 `examples/still_image_edit_api.json`；可直接拖入画布的完整示例见
`examples/workflows/H3_Still_Edit_22Frames_EXP.json`。两者默认使用512×512、22帧、20步，
并连接 Still Preflight；在 Reference Image Edit 节点上点击“＋”可追加最多8张参考图。

本机真实模型验证中，pruned Ref2VA INT8 在 512×512、20 步、`direct_1_frame` 下成功
保留手袋主体并把黑色皮革改成深红色；相同任务在 128×128 下结构明显崩坏。因此默认推荐
`canvas_mode=from_edit_image`，自定义画布短边不要低于 512。该结果只是单个可用案例，
不能代替多图、不同主体、不同编辑类型和多种 seed 的系统质量评估。

## H3 Turbo 四步双时钟采样

H3 的视频流默认使用 shift 12，音频流使用 shift 3。旧版 ComfyUI 的 H3 DiT 会把音频
速度乘上 `d(sigma_audio)/d(sigma_video)`；当前 ComfyUI 已改为 `FLOW_AV`，模型返回原始
音频速度，并由原生 `ModelSamplingAV` 支持音频 carry/scale。T8 双时钟节点自己维护两个
时钟，因此会检测实际基模协议：旧版移除 schedule slope，当前版直接按音频 sigma 差积分，
同时把自定义 sampling 的 `audio_scale` 固定为 `1.0`，避免重复缩放。

`MiniMax H3 Dual-Clock Sampler (T8)` 每步仍只做一次联合 AV 模型前向，不拆开模型，
但更新 latent 时执行：

- 视频：`delta_video * velocity_video`；
- 音频：旧协议先除去 schedule slope，当前协议直接使用原始速度，再乘 `delta_audio`；
- mask=0 的锁定区域保留 ComfyUI 原有的 inpaint 时钟，完整生成区域使用音频时钟。

四步 Turbo 推荐连接：

1. `UNET/Diffusion Model Loader -> LoraLoaderBypassModelOnly -> Dual-Clock Sampler.model`；
   当前 INT8/量化模型不要改用普通 LoRA 合并链并假设结果等价。
2. Conditioning/Empty H3 AV Latent 的同一个 `av_latent` 同时连接到
   `Dual-Clock Sampler.av_latent` 和 `SamplerCustomAdvanced.latent_image`。
3. Dual-Clock 的 `model` 接 `BasicGuider.model`，`sampler` 和 `sigmas` 分别接
   `SamplerCustomAdvanced` 的同名输入。
4. `steps=4`、`shift_video=12`、`shift_audio=3`、`sampler=dual_clock_euler`、
   `scheduler=native_flow`。LoRA 强度使用作者建议值。

节点内部现在可选择采样器和调度器：

| 控件 | 默认值 | 行为与兼容范围 |
|---|---|---|
| `sampler / 采样器` | `dual_clock_euler` | 原有 T8 显式双时钟 Euler，数值路径不变；兼容旧版与当前 ComfyUI |
| 其他采样器 | 无 | 使用当前 ComfyUI 自带的 sampler，并切换到原生 `ModelSamplingAV` carry/scale；旧版 ComfyUI 不提供这些选项 |
| `scheduler / 调度器` | `native_flow` | 原有 shifted-uniform H3 flow sigma，数值路径不变 |
| 其他调度器 | 无 | 调用当前 ComfyUI 的同名 scheduler；改变 sigma 时间网格，不承诺一定改善 Turbo 画质或音质 |

`dual_clock_euler` 配其他调度器时，仍由 T8 显式维护视频/音频两个时钟；其他采样器则由
当前 ComfyUI 原生 `FLOW_AV` 协议把联合 latent 映射为单一求解时钟。两条路径不能混用
carry/scale。标准采样器只在新版原生协议存在时开放，因为旧版 H3 没有可证明等价的通用
多阶求解适配。

这个节点已经代替 `MiniMax H3 Sigma Shift`、`KSamplerSelect` 和 scheduler 三个节点。
不要再串联一次 Sigma Shift，也不要外接 `KSamplerSelect` 或 `BasicScheduler`；需要更换时
直接使用本节点新增的两个下拉框。
`SamplerCustomAdvanced`、`RandomNoise` 和 `BasicGuider` 仍照常使用。

可导入的 API 结构示例见 `examples/dual_clock_4step_api.json`。其中模型文件名是占位符，
请替换为本机的 H3 基模、两个 VAE、Qwen3-VL CLIP 和已转换 LoRA 文件名。旧 API JSON
可以不提供 `sampler_name` 与 `scheduler`，后端会使用上述两个默认值。

## EXP：视频 4 步、音频更多步

`MiniMax H3 Multi-Rate Sampler (EXP/T8)` 位于独立的 `Experimental` 分类，代码也在独立
模块中，并使用与稳定版相同的新旧 ComfyUI 音频速度协议检测。EXP 节点把视频
Euler 更新保持为 `video_steps` 个宏步，同时在每个宏步内部为音频安排更多微步。例如：

- `video_steps=4, audio_steps=8`：每个视频区间 2 个音频微步；
- `video_steps=4, audio_steps=10`：四个区间均衡分配为 2、3、2、3 个音频微步；
- 四个视频宏时间边界与稳定 4 步网格完全一致。

H3 是联合音画 Transformer，无法只计算音频分支。因此 `audio_steps` 也是实际的完整 H3
DiT 前向次数：4/8 约是稳定 4/4 的 2 倍计算量，4/10 约是 2.5 倍，并会同时受到显存和
耗时影响。视频 latent 只在四个宏边界提交更新，但每个音频微步仍需联合模型前向。

建议先用相同 seed、prompt 和输入做 4/4 稳定版与 EXP 4/8 对照；若音频仍明显不够，再试
4/10。更多步不保证一定更好，因为 Turbo LoRA 的训练设计点仍是四步，额外中间时间点可能
改善音频数值积分，也可能产生分布外误差。EXP 不应直接替代已验证的生产工作流。

连接方法与稳定版相同，只把三个输出接入 `BasicGuider` / `SamplerCustomAdvanced`；不要再
叠加 Sigma Shift 或外部 scheduler。示例见 `examples/multirate_exp_api.json`。

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
- 生成画布像素面积不能超过 `1920×1088 = 2,088,960`，宽高必须是 32 的倍数。
- 超过 `1344×768 = 1,032,192` 像素不再报错，但 Preflight 会提示显存需求显著增加；
  模型支持该画布不代表所有帧数、参考数量和显卡都能在相同显存内运行。
- 原生 H3 目前只支持 batch size 1。
- 引用上限：9 张 Picture、3 个 Video、3 个独立 Audio；参考视频官方建议 2–15 秒。
- `Hybrid` 同时使用精确首/尾帧和参考媒体。节点包含针对当前 ComfyUI `PackedLayout`
  行为的运行时契约检查；上游若改变结构会明确停止，而不是生成错位条件。

官方建议的 16:9、32 倍数尺寸可直接使用：

| 约百万像素 | 输出尺寸 |
|---:|---:|
| 0.2 | 608×352 |
| 0.3 | 736×416 |
| 0.4 | 864×480 |
| 0.5 | 960×544 |
| 0.6 | 1056×608 |
| 0.7 | 1152×640 |
| 0.8 | 1216×672 |
| 0.9 | 1280×736 |
| 0.98 | 1344×768 |
| 1.0 | 1376×768 |
| 1.2 | 1504×832 |
| 1.5 | 1664×928 |
| 1.8 | 1824×1024 |
| 2.0 | 1920×1088 |

## 示例与测试

可直接拖入画布的稳定 4/4、EXP 4/8、EXP 4/10 和 Ref2VA 22帧静态候选编辑示例位于
`examples/workflows/`。API 示例见 `examples/audio_lock_api.json`、
`examples/dual_clock_4step_api.json`、`examples/multirate_exp_api.json` 和
`examples/still_image_edit_api.json`。替换 API 示例里的模型、VAE、CLIP、可选 LoRA、
输入图像和音频文件名后即可使用；
保存节点使用已安装的 VideoHelperSuite。

从 ComfyUI 根目录、使用启动 ComfyUI 的同一 Python 环境运行：

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -q .\custom_nodes\minimax-h3-audio-T8
```

自动化测试用于验证节点注册、条件与 latent 契约、sigma 数学、mask/callback、工作流结构
和静态图像路径；它不等同于对所有模型、提示词、种子和画布的感知质量保证。

## 显存与 DynamicVRAM 验证

项目提供独立诊断工具 `tools/validate_h3_vram.py`，用于排查 H3 工作流在
DynamicVRAM/VBAR、`LoraLoaderBypassModelOnly` 和双时钟采样组合下的 OOM。工具不修改
采样数学或模型权重，可完成 API 工作流静态检查、生成 stock Euler/双时钟严格 A/B、按节点
和采样进度记录显存曲线，以及比较两次运行的控制变量与峰值增量。

第一轮稳定 Turbo 对照必须统一为 4 步、相同模型/LoRA/Prompt/seed/尺寸/帧数，并建议关闭
预览。完整命令、判定规则和限制见 [显存验证方法](docs/VRAM_VALIDATION.md)。在取得真实 OOM
traceback 和有效 A/B 前，不应把高显存直接归因于双时钟节点，也不应盲目替换 INT8 旁路
LoRA 或关闭 VBAR。

2026-08-07 的本机暖缓存实测中，`0.6M`、362 帧、4 步的 stock Euler 与双时钟设备峰值
分别为 16,213.5 MiB 和 16,182.2 MiB，PyTorch 峰值均为 14,573.5 MiB；未发现双时钟路径
存在实质峰值增加。两条路径都已非常接近 16 GiB 上限，这个单机结果不能替代反馈用户的
精确工作流、OOM traceback 和冷启动换序复测。
