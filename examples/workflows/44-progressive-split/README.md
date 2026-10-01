# S10 Progressive 分离式二采（EXP）

本目录是现有 Progressive single 的 T2VA／I2VA 分离式实验图。旧 `28-progressive-sampling` 一体工作流与默认采样入口均保留。160 张图分为两组配方：文件名含 `Legacy6plus2` 的80张对应旧公开样例的总8步、LOW6＋HIGH2、两阶段EMA B LoRA、1024×512×124；另外80张为现有分离候选的Stock20 LOW10＋HIGH10，**不是旧6＋2的替代**。每组再覆盖两种任务 × 10 种已实现效果配方（无效果，或 EAV／Prompt Relay／两者组合各作用于 LOW、HIGH、两段）× 四种操作：

- `Full_NoSave`：显式运行 LOW→原 learned3D→HIGH，不保存阶段。
- `Full_Save`：同样运行并保存 LOW 边界和完成 HIGH；记录真实 path 与 SHA。
- `Cold_HIGH`：读取对应已完成 LOW 的 path 与 SHA，只运行 HIGH；图中没有 LOW 模型、条件、效果或采样链。
- `Load_Completed_HIGH`：读取已完成 HIGH，直接解码，不运行模型、放大或采样。

LOW 边界包含干净视频和**仍在演化的**音频 `audio_next`；外置 learned3D 只放大视频，HIGH 按冻结计划和原 mask／音频锚继续。它不是普通“输出视频再采一次”，也不是 FastH3 V2。两阶段 MODEL／LoRA、提示词、条件和噪声独立；HIGH 专属编辑不需改 LOW。Relay 和 EAV 在相应阶段外置，EAV 默认 `report_only`，不增强画质。旧6＋2的配置已按公开图的步数、LOW分界、LoRA、种子和几何复制到分离图；**尚未据此声称真实权重成片逐位等价**。

I2VA 请替换 `SELECT_YOUR_FIRST_FRAME.png`；冷图及已完成 HIGH 读取图里的路径和 SHA 只是占位，不能直接排队。更改 LOW 模型、LoRA、首帧、提示词、尺寸、计划或效果后须重新运行并保存 LOW；冷图不会自动证明旧冻结边界与今天的控件一致。`Load_Completed_HIGH` 的效果标签说明来源配方；若仅 LOW 启用效果，读取图自身自然没有可编辑的 LOW 效果链。

目前这些通用图已做结构、Core 静态与前端可编辑持久化验证；既有数值／缓存／冷恢复证据主要来自 tiny CPU 和替身放大器。不能据此宣称真实训练权重的默认尺寸、不同模型／LoRA／后端、长视频、`apply_exp` 画质或人工画音通过。请先在自己的资产与画布上逐步验证。
