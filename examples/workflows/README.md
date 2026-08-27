# MiniMax H3 T8 工作流目录

这里仅保存可直接拖入或通过 ComfyUI“工作流”菜单打开的前端 JSON，以及每个功能目录的一份说明文件。文件名前的日期是该工作流的发布日期，不代表功能稳定等级；带 `EXP` 或 `Advanced` 的路线应先阅读所在目录说明和画布 NOTE。

| 目录 | 主要用途 |
|---|---|
| `01-basic-generation` | 稳定双时钟与不同音频步数组合的基础生成 |
| `02-audio-control` | 音频锁定、重混、只参考及计划式音频注入 |
| `03-image-video-edit` | 单帧语义编辑、源视频重绘、参考强度实验与 LanPaint 局部AV修复 |
| `04-long-video` | 分段长视频、accepted manifest、后台续跑、完成态latent检查点、双时钟Euler逐NFE恢复，以及全局Prompt Relay事件时间线 |
| `05-speech-dialogue` | 单人语音、参考音色、对白、长文本和音色库实验 |
| `06-face-refine` | 单人/动漫/多人脸部五官修复与追踪回贴 |
| `07-motion-detail` | 动态引导、尾段细化、Restart、STG与组合采样 |
| `08-multi-keyframe` | 首尾帧之外的中间关键帧时间线 |
| `09-hybrid-model` | FL2VA/Ref2VA混合权重补丁、兼容审计和显存策略 |
| `10-speed` | SPEED空间渐进采样、频谱标定与多任务研究路线 |
| `11-studio-production` | 时间线、上下文、选择性修复、解码安全和交付工具 |
| `12-system-memory` | 环境审计、激活分块、前缀缓存、轨迹诊断和外部 BlockSwap 桥接 |
| `13-latent-upscale` | 普通32整除放大、学习型3D latent放大与二阶段H3生成 |
| `14-prompt-relay` | 全局提示词常驻、局部事件按时间接力、可选联合AV路由与8B提示词重写 |
| `15-sla-attention` | LightX2V Turbo-SLA LoRA、动态块稀疏 Sage2、KJ Sage单入口组合器与强制运行审计（实验） |
| `16-raven-streaming` | 外部RAVEN因果分块T2VA、统一参数、加载前资源保护与请求合同审计（实验） |
| `17-skin-finish` | 最终解码后的肤色/油光候选、专用Oil Control低内存文件流、Studio镜头内参数关键帧、候选低频与来源高频解耦、单轨及SAM3.1逐镜多人五点ParseNet语义皮肤MASK、可续跑状态、YuNet代理两遍流和ParseNet语义Quality Stream，以及源片相对的曝光/纹理/裁切P2硬门（实验） |
| `18-audio-refine` | Turbo低步数生成音频的精确无缓存双时钟尾段精修、原始/候选试听、逐值视频latent回填和默认回退原结果的人工质量门（实验） |
| `19-pdd-acceleration` | Alibaba PAI MiniMax-H3 PDD 8步蒸馏，分别用于完整FL2VA与Ref2VA基模的动态LoRA和输出头（实验） |

使用顺序建议：先从稳定基础/音频工作流确认模型链可运行，再按具体目的进入 Advanced/EXP 目录。不要把不同高级采样器直接串联；组合能力应使用专门的 Mixer 工作流或遵循画布 NOTE。
