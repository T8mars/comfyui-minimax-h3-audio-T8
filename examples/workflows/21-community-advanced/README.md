# 社区创作增强（Advanced / EXP）

这组工作流只追加高级节点，不修改原有生成、长视频、Studio 或低显存工作流。

- `H3_Fun_Control`：把已经预处理好的深度、姿态或边缘视频接入 H3。控制视频必须和目标宽高、帧数一致；`strength=0`为原样旁路。当前模型放在`ComfyUI/models/controlnet`。pruned / basis控制模型必须搭配8维AdaLN主模型，原始dense控制模型搭配2688维完整主模型；节点按实际张量维度提前检查，不依赖文件名、哈希或大小。
- `H3_Long_Video_Voice_and_Seam`：按人物绑定最终`<Audio N>`、检查跨段句子和首段人工确认；另一个独立节点只审计/缓解已拼接IMAGE的亮度与色彩接缝，不处理音频。
- `H3_System_Cache_and_Diagnostics`：低显存策略、Creator分段语义缓存、Generic Loops能力、官方问题证据诊断和TAEH3原生快速预览检查。它们默认只报告或生成计划，不会卸载模型、删除缓存、切换草案后端或修改ComfyUI预览设置。TAEH3模型放在`ComfyUI/models/vae_approx/taeh3.safetensors`，启动时选择TAESD预览；它只显示每步的视频第一帧，最终成片仍以H3视频VAE为准。

边界：Fun Control已完成一条736×416×22、同seed轨迹对照，控制端对上下移动的遵循更强，证明功能有效；终点仍会受`end_percent`释放尾段影响，不能据单条样本宣称普遍画质提升。其余节点主要用于计划、审计和安全接入，不代表普遍16GB安全或质量提升。
