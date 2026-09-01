# MiniMax H3 PDD 8-Step / 8步蒸馏加速

本目录提供 Alibaba PAI `MiniMax-H3-Acc-LoRAs` 的前端 ComfyUI 工作流。PDD 是 Parallel Decoding Distillation：源模型的32个时间间隔按每4个加权融合为一次模型前向，实际固定为8 NFE。它除了258个主干LoRA adapter，还带32组视频输出头和32组音频输出头；普通 `Load LoRA` 会漏掉这些动态输出头，因此不能代替专用节点。

## 工作流

- `2026-08-27_H3_PDD_FL2VA_8Step_Advanced_EXP.json`：完整FL2VA基模、首帧+尾帧、联合音视频8步PDD。
- `2026-08-27_H3_PDD_Ref2VA_8Step_Advanced_EXP.json`：完整Ref2VA基模、单张参考图、联合音视频8步PDD。
- `2026-08-27_H3_PDD_FL2VA_Learned_Latent_TwoPass_4Plus4_Advanced_EXP.json`：FL2VA学习型latent放大双采，LOW 4步 + HIGH 4步；首尾图会分别居中预对齐到同一 16:9 画布，避免两次 Conditioning 产生渐进拉宽。
- `2026-08-27_H3_PDD_Ref2VA_Learned_Latent_TwoPass_4Plus4_Stable.json`：正式Ref2VA学习型latent双采；默认864×480×22、1.5×，实际HIGH为1312×736×22。

四个文件都是可直接拖入ComfyUI的前端工作流，不是API格式。双采工作流带五个NOTE，解释4+4切分、尺寸交接、音频继续采样和显存边界。Ref2VA双采已经实测并作为正式工作流；FL2VA双采仍保留Advanced EXP，等待同等真实验证。

## PDD 双采为什么是 4+4

双采不是完整8步跑两遍。专用PDD节点仍只产生一条官方9点sigma轨迹；`SplitSigmas step=4`把前4次模型前向交给LOW阶段，学习型latent放大后，再把后4次模型前向交给HIGH阶段。总Transformer调用仍是8次，对应PDD输出头block 0–7各一次。

HIGH阶段必须重新建立与放大后latent尺寸一致的双时钟sampler，但它不会重新生成或替换PDD轨迹。PASS 2继续联合采样视频和音频，最终解码PASS 2的`output`。默认从512×288×124放大到1024×576×124；显存紧张时应先降低LOW尺寸或帧数，并保持一次只跑一个任务。

## 必需模型

把转换后的文件放在 `ComfyUI/models/loras`：

- `MiniMax-H3-FL2VA-Acc-8Step_comfyui_pdd.safetensors`
- `MiniMax-H3-Ref2VA-Acc-8Step_comfyui_pdd.safetensors`

对应完整、非pruned基模：

- FL2VA：`minimax_h3_fl2va_int8_convrot.safetensors`
- Ref2VA：`minimax_h3_ref2va_int8_convrot.safetensors`

PDD文件与基模变体不能互换。当前节点会验证PDD metadata、四个动态head、258个adapter和AdaLN输入宽度2688；但是ComfyUI的原生MODEL对象不保留源文件名，因此最终是否选择了匹配的FL2VA/Ref2VA基模仍由工作流和用户负责。带`adaln_t_table`且AdaLN输入宽度为8的pruned基模不兼容。

## 固定官方参数

专用 `MiniMax H3 PDD 8-Step Setup (T8 Advanced EXP)` 节点同时输出MODEL、SAMPLER和SIGMAS：

- strength：`1.0`
- sampler：`euler`
- scheduler：`simple`
- NFE：`8`
- video shift：`12.0`
- audio shift：`3.0`
- CFG：`1.0`（BasicGuider）

新版ComfyUI（已包含官方PR #15908语义）会优先使用原生PDD FinalLayer：258个主干adapter走ComfyUI LoRA路径，四个绝对head bank由专用节点转换为原生首head+offset格式，并由ModelPatcher负责加载和恢复。旧版ComfyUI继续使用已验证的动态bypass与T8 final-head回退。选择依据是运行时能力，不是版本号、模型hash或文件大小。

单采工作流不要再串另一个sampler或第二个PDD节点。双采示例中的HIGH双时钟sampler只负责匹配放大后的latent几何，sigma仍来自同一个PDD节点。两种路线都不要再叠加普通/其他Turbo LoRA或SLA LoRA。低于1.0的strength虽然允许做研究性插值，但不属于上游质量配置。

## 实现与当前验证

当前节点有两条等价路径：新版核心使用官方原生32-head FinalLayer，旧核心把32组输出头预融合成8组运行头。两条路径都只允许官方8点sigma网格并逐点选择block 0到7；单元测试已证明视频shift 12、音频shift 3下八个block的head结果一致。当前转换文件仍必须使用专用节点，因为普通LoRA节点不会理解四个`pdd.final_layer.*`绝对head bank。

2026-08-29在官方ComfyUI `e7051b0`上，两份文件均以原生路径完成736×416×22串行真实渲染：258个主干adapter、4个原生head补丁、0个回退hook，8 NFE与block 0–7全部正确；H.264、32kHz双声道AAC、帧数及数值有限性均严格通过。FL2VA/Ref2VA峰值使用约15628/15477MiB，最低余量约482/633MiB，因此前者仍低于项目512MiB舒适余量，不宣传普遍16GB安全。之前旧回退路径的736×416×124、1152×640×124和Ref2VA 4+4双采结果继续有效。继续使用时不要并发排队。

上游：<https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs>，本次固定复核revision `78db175437ee05df7ec492ee366f01b68b8d20e6`，参考实现许可证Apache-2.0。
