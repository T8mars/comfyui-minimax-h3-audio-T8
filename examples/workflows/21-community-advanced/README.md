# 社区创作增强（Advanced / EXP）

这组工作流只追加高级节点，不修改原有生成、长视频、Studio 或低显存工作流。

- `H3_Fun_Control`：把已经预处理好的深度、姿态或边缘视频接入 H3。控制视频必须和目标宽高、帧数一致；`strength=0`为原样旁路。当前模型放在`ComfyUI/models/controlnet`。
- `H3_Long_Video_Voice_and_Seam`：按人物绑定最终`<Audio N>`、检查跨段句子和首段人工确认；另一个独立节点只审计/缓解已拼接IMAGE的亮度与色彩接缝，不处理音频。
- `H3_System_Cache_and_Diagnostics`：低显存策略、Creator分段语义缓存、Generic Loops能力和官方问题证据诊断。它们默认只报告或生成计划，不会卸载模型、删除缓存或切换草案后端。

边界：Fun Control尚需最终人工画质审核；其余节点主要用于计划、审计和安全接入，不代表普遍16GB安全或质量提升。
