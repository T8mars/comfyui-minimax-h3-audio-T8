# RF Restart：显式联合时钟初始化（EXP）

本地新增节点 `MiniMaxH3RFRestartJointClockSetupEXPT8`，显示名
**H3 RF · Restart Joint Clock (No Double Audio Rebase) (T8 EXP)**。
它只设置第二段 RF 重启，不在内部重跑第一遍。旧一体节点、旧分离节点、默认值及已保存工作流不改。

## 如何使用

在 RF Standalone / Detail Mixer 的分离图中，仅将
`MiniMaxH3RFRestartStageSetupEXPT8` 替换为这个新节点，保持同名输入、输出和参数连接。
Base Setup、RF Handoff、NOISE、Guider、Stage Sampler、EAV、Prompt Relay 与显式 Stage Save/Load
继续独立使用。没有重启的普通采样不需要这个节点；不能直接把旧一体采样器连到此处。
原模板/已完成端点的身份、线性重启 SIGMAS、二值遮罩、音频全参与、单 batch 与采样完整性检查保留。
零步或零 sigma 仍显式 no-op，无隐藏补采样。

## 修正了什么

RF 已先按视频与音频各自的 sigma 对完成端点重噪。
旧兼容路径随后又执行通用 dual-clock Euler 的音频初始化 rebase，重复缩放已初始化的音频。
当 video/audio shift 为 12/3、重启 video sigma 为 0.15 时，audio sigma 为约 0.04225；
空原始音频模板下第二次处理会再缩放约 0.28169 倍。

新节点显式标记状态已按双时钟初始化，通过只委托调用的边界避免第二次通用初始化。
实际 Core inpaint 对象仍持有原模板、NOISE、遮罩和原模型调用；不改全局采样公式，
不替换端点、不冻结音频、不混入 BASE 音轨、不加后期滤波/降噪。
新策略进入独立 StageContext/缓存身份，不能混用旧策略的采样器和 context。
旧路径仍保留，包括它与原一体实现的数值兼容性。

## 验证边界

针对用户反馈，已从原生画布各重跑一次 C1 Standalone、C2 Detail Mixer：
加载同源已保存 BASE，仅执行原 3 步 RF，外置效果、种子及重启参数不改。
两份完整音画均为 124 帧、448×256、24 fps、5.167 秒，完整解码通过。
音频 latent 标准差从旧候选约 0.10 回到约 0.36，接近同源 BASE；
这证明重复缩放得到纠正，**不等于听感通过或所有噪声均消失**。

独立 Long Relay 音频精修不使用这个 RF 修正。
B8 仅以同源、同种子、原 4 步将精修强度 0.50 降至 0.35 做新候选；
不是通用质量修复，原始音轨继续保留，Quality Gate 不自动接受。

2026-10-02 用户分别复审通过 B8 / C1 / C2 三份指定新候选；这是独立听看意见，
不是从音频统计量推导出的质量结论，也不泛化到所有素材或任意配方。
已保存原生实跑图，另追加 [完整/冷恢复配对模板](../examples/workflows/64-reviewed-audio-followup/README.md)，
模板保留原通用素材设置和明确检查点占位值，不能冒充审核实跑的逐值副本。
全部分离路线的保存目录见 [总索引](../examples/workflows/SEPARATED_WORKFLOWS.md)。
修正后的九个完整定向 CPU 文件共 239 项通过，包含旧 RF 数值兼容、新策略外置效果、
保存恢复和独立进程仅重启采样；未跳过或排除用例。
实际 Core 确认只追加第 582 个节点，旧 581 个完整接口和顺序不变。
此修正自 GitHub v1.88.1 提供；Registry 审核和实际安装可见性独立。
不将局部 CPU/代表素材验收称为全项目或任意模型资格。
