# S06／S07 旧 Dual MODEL 分离式二采（EXP）

这里新增六种旧 Dual MODEL 单段配方各一对图，共 12 张前端 JSON：S06 LOW4→HIGH3/4/5，以及 S07 完整 LOW20→HIGH3/4/5。每对 `Full_Stages` 显式运行并保存 LOW/HIGH；`Cold_HIGH` 读取同配方已完成 LOW 的真实 manifest 路径和 SHA，只运行 HIGH。原有一体节点、旧长视频工作流及其默认设置没有迁移。

LOW4 是旧 simple8 的前四个区间，音频按旧联合策略继续；LOW20 是完整 native-flow 首采，保留其已完成音频。HIGH3/4/5 使用各自发布的 LBH 细化表，并非首采剩余步数；这也不是 FastH3 V2 的绝对窗口。LOW 的 `denoised_output` 经原学习型 3D 潜空间放大与 Native Dual Handoff 到 HIGH，保留旧音频前缀和锁定策略。两阶段有独立 MODEL／LoRA、条件、NOISE、Prompt Relay Plan 与 Stage EAV；EAV 默认 `report_only`，不会增强画质。

先核对本机底模、EMA B、Qwen、双 VAE、learned3D 和任务画布。运行完整图后，从 LOW 保存输出抄取 `artifact_path` 与 SHA；冷图的占位值不可直接排队。更改了 LOW 模型、LoRA、提示词、条件或噪声，应重新运行 LOW，不要把旧冻结结果当作新设置生成。

已有六组合的真实权重测试仅覆盖固定小画布22帧的机械执行与跨 Core 仅 HIGH 恢复，不代表本示例默认尺寸、其它后端／素材／LoRA、长片接缝、`apply_exp` 画质或人工画音验收。旧工作流继续可用。
