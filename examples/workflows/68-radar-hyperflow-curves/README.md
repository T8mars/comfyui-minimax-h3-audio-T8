# HyperFlow pruned 曲线近似：独立阶段 EXP

先看 [操作说明](../../../docs/HYPERFLOW_CURVE_EXP.md)。这是近似路线，不替代完整时间 HyperFlow；需要匹配的 pruned 底模、原 HyperFlow adapter 和已经单独生成的精确 fit。

| 工作流 | 用途 |
|---|---|
| `HyperFlow_Curve_4plus4_full_save_EXP.json` | 独立 HEAD4 → Save → 独立 TAIL4 → Save → 解码；两路 EAV／Relay 外置 |
| `HyperFlow_Curve_4plus4_cold_tail_EXP.json` | 填真实 HEAD path／SHA，只续 TAIL4；无 HEAD 模型／噪声／效果链 |
| `HyperFlow_Curve_4plus4_cold_delivery_EXP.json` | 填真实 TAIL path／SHA，零采样解码；无模型／CLIP／fit |

三图中的 FullSave／ColdTAIL 使用独立 Native FP32 Curve Base Loader，先正确加载原生 pruned 底模，再分别插内容 LoRA／Curve Apply；不改普通 UNETLoader 或修补已加载 MODEL。Fit Load 读取 `models/hyperflow/curve_fits` 或明确绝对路径。Fit 与存取路径的占位值不能直接执行；文件不随图打包。不自动拟合、下载、覆盖旧图或确认画质。当前机械与成片资格见专题文档，不能把 CPU／图验证当成人审通过。
