# Face Refine 工作流

这里的工作流都是实验路线。它们可以生成脸部修复候选，但不会自动证明身份、嘴型或五官已经修好。

## 建议优先使用 Window

- `2026-09-05_H3_Face_Refine_Window_Manual_Review_Advanced_EXP.json`：一次只修一个连续时间窗。默认只预览，确认满意后再把 `decision` 改为接受并打开确认开关。
- `2026-09-05_H3_Face_Refine_Window_Studio_Serial_Advanced_EXP.json`：多个时间窗按顺序处理，并保存接受/拒绝清单；崩溃后可继续未完成窗口。
- `2026-09-05_H3_Face_Refine_Window_Studio_Compose_Advanced_EXP.json`：所有窗口完成后，或最后一次提交后在保存前崩溃时，只读取清单合成，不加载H3。

先把源片整理为单镜头、24fps，再填 0 基闭区间帧号，例如 `0-23`。替换源视频提示词、两张清晰身份参考图、模型、LoRA 和 VAE。窗口上下文只帮助 H3 生成，不会自动写回最终片；最终音频始终接完整原始音轨。

Studio 默认是 `review_only + preview_only`。只有真人看完候选后，才使用明确的接受或拒绝决定。接受后的窗口不可回退或重复执行；同一项目也不允许两个生成任务并发。

## 旧工作流

- `Face_Refine_Parity`、`Face_Refine_Advanced` 和 `Face_Refine_Anime` 保留用于兼容和诊断。
- `SAM31_2Person/3Person` 是多人实验工作流。遮挡、背脸、人物交叉或重叠 mask 时必须人工检查；目前不具备自动身份安全承诺。

窗口路线在固定 RTX 4060 Ti 16GB 上完成了 15 次串行显存矩阵和一次真实坏脸样本，但多素材真人盲测尚未结束，因此仍为 Advanced EXP，不要批量自动接受。
