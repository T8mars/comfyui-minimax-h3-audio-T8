# MiniMax H3 PDD 8-Step / 8步蒸馏加速

本目录提供 Alibaba PAI `MiniMax-H3-Acc-LoRAs` 的两份前端 ComfyUI 工作流。PDD 是 Parallel Decoding Distillation：源模型的32个时间间隔按每4个加权融合为一次模型前向，实际固定为8 NFE。它除了258个主干LoRA adapter，还带32组视频输出头和32组音频输出头；普通 `Load LoRA` 会漏掉这些动态输出头，因此不能代替专用节点。

## 工作流

- `2026-08-27_H3_PDD_FL2VA_8Step_Advanced_EXP.json`：完整FL2VA基模、首帧+尾帧、联合音视频8步PDD。
- `2026-08-27_H3_PDD_Ref2VA_8Step_Advanced_EXP.json`：完整Ref2VA基模、单张参考图、联合音视频8步PDD。

两个文件都是可直接拖入ComfyUI的前端工作流，不是API格式；画布中各有三个NOTE，解释接线、固定参数和验证边界。

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

PDD主干adapter采用动态bypass，不会把残差合并进INT8底模再量化。模型正常eject或注入中途异常时，节点会恢复hook并把adapter tensor移回MODEL的offload设备；这修复的是adapter生命周期，不代表整个工作流已经通过16GB峰值显存验收。

不要再串Dual-Clock、普通/其他Turbo LoRA、SLA LoRA或第二个PDD节点。低于1.0的strength虽然允许做研究性插值，但不属于上游质量配置。

## 实现与当前验证

主干LoRA通过ComfyUI动态bypass residual运行，不把BF16 LoRA先合并再重新量化到INT8基模。32组输出头在加载时预融合成8组运行头；采样时只允许官方8点sigma网格并逐点选择block 0到7。

两份转换文件已完成SHA-256复核、778 tensor结构检查、258/258当前ComfyUI模块映射、动态钩子装配和8步schedule一致性检查。两条真实736×416×124联合音画链已严格串行完成：FL2VA/Ref2VA分别约147.156/139.718秒；另有一条Ref2VA 1152×640×124复核耗时414.156秒。三条均为124帧H.264与32kHz双声道AAC并通过严格解码；最低显存余量分别只有447/510/500MiB，均低于项目固定512MiB安全门。因此本节点当前能完成受测16GB链，但不能宣传普遍16GB安全。用户确认0.7MP Ref2VA画面没有问题；它仍生成了对白字幕，即便提示词明确写了不需要字幕，记录为模型/提示遵循问题而非画面硬失败或低分辨率显示错误。对白听感和相对普通8步的速度收益仍需人工/同合同对照。继续使用时不要并发排队。

上游：<https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs>，本次固定复核revision `78db175437ee05df7ec492ee366f01b68b8d20e6`，参考实现许可证Apache-2.0。
