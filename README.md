# MiniMax H3 Audio T8

面向 ComfyUI 的 MiniMax H3 视频与音频节点包。它保留原生 H3 的工作流接口，并在此基础上提供双时钟采样、音频控制、长视频、关键帧、脸部修复和显存诊断等能力。

当前版本：**1.40.0** · 节点 148 个 · GPL-3.0-or-later

## 能做什么

- 文生音视频（T2VA）、图生视频、首尾帧、参考图/参考视频/参考音频和 Hybrid 工作流
- 双时钟视频/音频采样，以及音频锁定、重混、参考和最终混音
- 学习型双采样在非1.0起点把音频重建到自己的sigma时钟，再由第二阶段完成联合AV；显式锁源审计仍可选
- `13-latent-upscale` 中标准、native语音、lock_source、remix_source 0.20 和 reference_only 五份学习型二采工作流已统一到作者对齐的低清4步+高清4步，共8次联合AV前向
- 同一清晰语音素材上的 native、lock_source、remix_source 0.20 与 reference_only 双采成片已由用户完整审核通过；结论限定于该素材
- 32 倍数对齐的 3D latent 放大，避免官方放大节点造成宽高偏移
- 长视频分段、断点续跑、时间线报告和音视频同步检查
- 单人/多人脸部修复、SAM3.1 追踪、多帧关键帧和实验性动态细节增强
- 语音、对白、演绎、ADR 和生产辅助节点（实验功能会明确标注）
- 预检、显存/缓存报告、模型兼容检查和异常路径清理

稳定节点与 `Advanced`/`Experimental` 节点分开。高级节点采用追加式设计：未连接时不改变旧工作流、旧节点 ID、输入顺序和默认采样数学。

## 安装

### Comfy Registry（推荐）

在 ComfyUI Manager 中搜索 `MiniMax H3 Audio T8`，或在终端执行：

```powershell
comfy node install minimax-h3-audio-t8
```

### 手动安装

将仓库放到：

```text
ComfyUI/custom_nodes/minimax-h3-audio-T8
```

本插件不自动下载 H3 权重，也没有强制的额外 pip 依赖。请先准备与任务匹配的模型、CLIP、VAE 和 LoRA，并重启 ComfyUI。

## 第一次使用

1. 打开 `examples/workflows/`，先从 `01-basic-generation` 中的基础工作流开始。
2. 将工作流中的模型、图片、视频和音频替换为自己的文件。
3. 初次运行建议使用 22 帧、较小画布和 stock sampler，确认能正常解码后再增加时长、分辨率或参考条件。
4. 使用工作流中的 NOTE 节点作为参数说明；报告输出可用于排查标签、显存、音频和兼容性问题。

## 工作流目录

| 目录 | 用途 |
| --- | --- |
| `01-basic-generation` | 基础 T2VA、I2VA、FL2VA 和 Ref2VA |
| `02-audio-control` | 音频驱动、音频锁定、重混、参考音频和双时钟 |
| `03-image-video-edit` | 图像编辑、视频编辑和音视频参考 |
| `04-long-video` | 分段生成、长视频、断点恢复和拼接 |
| `05-speech-dialogue` | 单人语音、对白、演绎和 ADR（实验） |
| `06-face-refine` | 单人/多人脸部修复与 SAM3.1 追踪（实验） |
| `07-motion-detail` | 动态细节、尾段采样、FETA 时序注意力增强和质量对照（实验） |
| `08-multi-keyframe` | 中间关键帧和关键帧计划（实验） |
| `09-hybrid-model` | FL2VA/Ref2VA 混合模型研究（实验） |
| `10-speed` | SPEED 多分辨率采样研究（实验） |
| `11-studio-production` | 生产、字幕、时间线和批量辅助 |
| `12-system-memory` | 预检、显存、缓存、VBAR 和释放策略 |
| `13-latent-upscale` | 32 倍数对齐的 latent 放大 |
| `14-prompt-relay` | 全局提示词常驻、局部事件按时间接力（实验） |

每个目录都有自己的 `README.md`，说明适用场景、参数建议和已知限制。完整索引见 [`examples/workflows/README.md`](examples/workflows/README.md)。

## 常用稳定节点

| 节点 | 作用 |
| --- | --- |
| `MiniMaxH3AudioConditioningT8` | 统一处理任务类型、提示词、图像/视频/音频参考和 AV 条件 |
| `MiniMaxH3DualClockSamplerT8` | 使用独立的视频/音频时钟采样 |
| `MiniMaxH3AVDecodeT8` | 解码 AV latent 并保持音视频时间线 |
| `MiniMaxH3AudioLatentControlT8` | 音频锁定、重混和参考策略 |
| `MiniMaxH3LatentUpscaleBy32T8` | 按 32 像素网格放大 3D latent |
| `MiniMaxH3PreflightT8` | 在生成前检查模型、标签、尺寸、帧数和显存风险 |

## 提示词媒体标签

显式引用建议使用官方标签：`<Picture 1>`、`<Video 1>`、`<Audio 1>`。标签编号必须与实际输入对应。节点会规范化大小写和空白，并在报告中列出最终绑定结果。

- 只有一个明确的旧式参考输入时，可兼容历史工作流的零编号/单参考写法。
- 多个参考输入出现歧义、缺失或重复标签时，严格模式会直接报错，不会静默猜测。
- 看到 `prompt media tag validation failed` 时，请先检查任务类型、标签编号和对应输入是否真的接入。

## 模型与尺寸

常见目录：

```text
models/diffusion_models   H3 DiT 模型
models/text_encoders      Qwen/MiniMax CLIP
models/vae                视频 VAE、音频 VAE
models/loras              Turbo、质量或任务 LoRA
models/latent_upscale_models
models/face_detection/checkpoints
```

不要只按文件名判断兼容性；任务类型、模型家族、LoRA、VAE 和 ComfyUI 版本必须匹配。工作流中的 NOTE 和预检报告会提示推荐组合。

重要约束：

- H3 是联合 AV Transformer，视频和音频共享部分计算；参考图、参考视频、参考音频越多，显存和序列长度越高。
- 宽高应为 32 的倍数；`1920×1088`作为官方高分辨率参考面积，而不是学习型放大节点的执行禁令。该节点允许用户选择更大画布并报告高显存风险，实际能否完成取决于显卡、帧数和参考数量。
- 常用帧数遵循 H3 的网格约束，例如 22、124、362；不要只按“秒数”猜可用帧数。
- 16GB 显卡不要默认视为所有任务安全。先用较小画布/短片段通过预检，再逐步增加规模。
- SPEED、Prompt Relay、动态细节、Hybrid、多帧关键帧、多人脸修复和语音节点目前按实验功能使用；未通过本机验证的组合会 fail-closed。
- Enhance-A-Video / FETA 节点按论文公式从目标视频 Q/K 计算跨帧 CFI，只直接增强目标视频 attention 输出。原节点继续支持 Stock20 的 T2VA / I2VA / FL2VA / L2VA，以及严格限定的修正 Alpha8 Turbo8 T2VA；独立 Reference Composer 只开放原生 Stock20 Ref2VA 和任务型 Hybrid，不改变旧节点合同。上述七类路线现均至少完成一组 0.7MP 同输入、同 seed 的 disabled/apply 机械与媒体对照；Ref2VA 增强端最低显存余量仅约 417MiB，低于项目 512MiB 门槛。自动指标只能证明轨迹和联合音频发生变化，不能证明画质、参考遵循或声音更好，因此所有路线仍不宣称稳定提质、音频非劣或通用16GB安全。
- `Enhance-A-Video + Strict Sage Advanced EXP`由一个组合节点同时拥有 FETA 路由和本机 SageAttention HND 后端，避免第三方整块 Attention patch 绕过 FETA。它不会静默回退到 PyTorch attention；本机一条 1152×640×124、Stock20 实测完成 1000 次 FETA 测量和 1000 次 Sage 调用，失败/回退为 0，并通过三轮严格音视频解码。该单条机械验证不代表画质更好、声音非劣、速度更快或通用 16GB 安全；使用时不要再叠加 KJ Sage、BlockCache、STG 或其他全局 attention patch。
- `Enhance-A-Video + Prompt Relay Composer Advanced EXP`解决两个独立节点争用同一Attention入口的问题：它验证现有Relay绑定后，在一次路由中先执行局部事件Relay，再只对目标视频输出行应用FETA；关闭FETA时保留原Relay MODEL。当前仅开放Stock20 T2VA，一组736×416×124、20步基础机械对照及严格媒体检查已通过，但余量低于512MiB；不作为稳定提质、音频非劣或16GB安全路线宣传。
- `Enhance-A-Video + BlockCache / STG / Long Video`是三个隔离追加组合器：BlockCache只接受已核对合同的CPU缓存并按full=50/hit=1审计实际执行块；STG把同一FETA规则用于主分支和弱分支，默认审计50/49块及额外联合AV前向；Long Video保留原上下文/layout拥有者，并为每个`segment_index/context_frames`创建独立Stock20审计。三者当前只通过低负载确定性合同、注册和工作流导入检查；按要求未做压力测试，不宣称提质、音频非劣、提速、省显存或通用16GB安全。
- Prompt Relay 与 Turbo8 组合时，连接顺序必须是 `UNET → Prompt Relay Conditioning → 修正 Alpha8 Bypass LoRA → DualClock Sampler`；把 LoRA 接在 Relay 前面会被主动拒绝。
- Prompt Relay 需要保留输入原声时使用 `lock_source`，并把节点的 `mux_audio` 接到最终保存节点；`native/remix/reference_only`仍可能因 H3 联合 AV Transformer 而改变声音。
- Prompt Relay 默认 `video_only_paper` 不改变旧工作流；只有显式插入 `Query Route`并选择`joint_av_exp`，才会把局部事件时间扩展到目标音频。该模式是H3实验扩展，不是论文已验证能力。
- `14-prompt-relay` 已提供 T2VA、I2VA、FL2VA、L2VA、Ref2VA、Hybrid 六种视觉任务模板，以及参考视频+同编号音轨、独立参考音频、联合AV和Turbo8模板；各任务的图片/视频/音频接线、模型顺序和已验证边界写在画布 NOTE 中。
- 六种视觉任务以及“参考视频+同音轨”“独立参考音频”现均至少有一组同输入、同seed、同NFE的baseline/Relay机械对照：16条成片均为736×416×124、24fps、32kHz双声道，严格音视频解码通过且无黑帧、冻结或削波；I2VA/FL2VA/L2VA/Hybrid的首尾帧锚点代理没有出现明显回退。该结果只关闭媒体与锚点合同，不代表Relay必然改善动作、身份、画质或声音；参考音轨不是原波形复制，感知优劣仍需观看和试听。
- 局部动作可用多个 `Prompt Relay Event Advanced` 节点逐项复制串联，并连接普通 Plan 或 Studio Packet桥接；普通Plan未连接时继续读取旧`local_prompts/time_ranges`，Packet桥接未连接时继续读取旧`events_json`。Studio 的 `Unified Cast + Sound Canvas + Prompt Compiler`编译结果仍是全局画面/声音事实源，节点不会自动拆剧情或改写事件。
- Prompt Relay显式时间范围必须按开始时间排列；`frames`只接受整数帧，`percent`只接受`0..100`。零个事件（包括整条Event链全部关闭）会只保留全局提示并无补丁直通；只有一个事件时也不安装补丁，通常应直接并入`global_prompt`。
- `Prompt Relay Preview Advanced`可以在不加载UNET、CLIP、VAE且不执行采样的情况下检查事件帧段、秒数、覆盖关系和Plan哈希；`Prompt Relay Resource Estimate Advanced`还能按画布、参考素材数量和query chunk估算H3 packed行数及Relay显式bias峰值，并在同一报告中列出736×416、1152×640、1920×1088三档矩阵。后者只是一项内存规划代理，不包含模型权重、完整attention激活、VAE/CLIP、VBAR或碎片，绝不等同于“16GB安全”。`14-prompt-relay`里的11份生成模板已内联Preview，另附一份纯时间线/资源预检工作流。
- Long Video需要分段事件时，使用独立的`Prompt Relay Long Video Window/Conditioning Advanced`；全局Plan只创建一次，本地渲染起点按`accepted timeline start - context overlap`计算。旧Long Video节点与旧工作流未改。本机已完成一条736×416、Turbo8、22帧上下文的segment 0→1真实链：输出124+102帧、事件没有从头重启、视频/音频各3轮严格解码通过，整卡峰值约15478/14984MiB。该单条结果仍只证明机械与时间线可用；最终音频接缝需试听，不能外推为普遍画质或16GB安全。

## 故障排查

遇到 OOM、NaN、花屏、音频丢失或人脸跳动时，请保留完整 ComfyUI Error Report，并记录：任务类型、模型文件、宽高、帧数、采样器/调度器、是否双时钟、参考输入、音频采样率和所有显式媒体标签。常见原因是：输入尺寸未对齐、重复连接 Sigma/采样器、参考标签错位、把实验节点接到不兼容的旧 core，或显存不足后继续复用缓存。

## 开发与回归

```powershell
$env:PYTHONPATH = 'F:\AI-T8-video-onekey\ComfyUI'
& 'F:\AI-T8-video-onekey\python\python.exe' -m pytest -q
```

修改节点前请先阅读本地 `SKILL.md`、`features.json`、`meta.json` 和对应工作流目录说明。新增高级功能应保持旧工作流可加载；检测到未知 ComfyUI core 或第三方全局 patch 时，应明确报错而不是静默降级。

## 链接与许可

- [GitHub](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)
- [Comfy Registry](https://registry.comfy.org/)
- [MiniMax H3 官方仓库](https://github.com/MiniMax-AI/MiniMax-H3)

## 社区与服务

- [B站：T8star](https://space.bilibili.com/385085361)
- [YouTube：T8star AI](https://www.youtube.com/@T8star-Aix/)
- [API 服务注册](https://api.seedance.nz/sign-up?aff=5f4w)
- [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd)
- [模型网盘](https://pan.quark.cn/s/c9c267081fbf)
- [Hugging Face](https://huggingface.co/t8star)

插件代码采用 GPL-3.0-or-later。H3 权重、Qwen、ASR、SAM3.1 和第三方 LoRA 各自遵循其上游许可；本仓库不随插件分发这些权重。使用人声、人物参考图或克隆功能时，请自行确认素材权利并遵守适用法律和模型许可。
