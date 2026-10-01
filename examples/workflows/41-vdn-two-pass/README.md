# S05 OpenVDN 分离式二采（EXP）

这里的 18 张新增前端图覆盖 DMD8、Stage B 50 完整首采之后的独立后段：各自的 VDN 尾段 4 步、原生 EMA B HIGH 3/4/5 步，以及旧 Stage B 入口使用的 VDN 尾段 5 步。每种配方各有 `Full_Stages`（运行并保存 LOW/HIGH）和 `Cold_HIGH`（凭已保存 LOW 的 manifest 路径与 SHA 只运行 HIGH）。原 `10-speed` 四张一体式 VDN 二采图仍可照旧使用，不自动替换。

LOW/HIGH 有分开的模型及可插 LoRA 位置、条件、噪声、Prompt Relay Plan 和 Stage EAV；VDN 分支另有专门的 Relay Apply/Audit，以观察其窗口与线性注意力路径。EAV 默认 `report_only`，这不会增强画质。LOW 必须是完成的 VDN 输出，再经原学习型 3D 潜空间放大和 Reconcile 接到 HIGH；HIGH 保留第一遍音频并做独立审计。原生 HIGH 使用独立干净底模和 EMA B，不继承 VDN branch。

先从 `Full_Stages` 按本机模型与任务核对全部资产并运行；若需要只重跑后段，把 LOW Save 输出的真实 `artifact_path` 与 SHA 填入同配方的 `Cold_HIGH`，不要套用另一训练阶段／帧长／模型的结果。默认图中的占位 manifest/SHA 无法直接用于恢复。Stage B 的两份训练资产若未安装，其图也不能直接运行；工具不会自动下载模型。

已取得的真实权重证据仅限固定小画布 DMD8→原生 HIGH3/4/5 的机械执行与跨 Core 冷恢复；其余配方目前是结构／tiny CPU 候选，不是 Stage B 或 VDN 自身尾段的 GPU 成片资格。所有图仍欠默认通用设置、原尺寸、多素材／后端、`apply_exp` 质量和人工画音验收。旧一体工作流及其默认行为保持原样。
