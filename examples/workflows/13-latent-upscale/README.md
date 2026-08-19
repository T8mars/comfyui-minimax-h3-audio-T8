# 学习型 Latent 放大与二阶段生成

这一组把低分辨率 H3 首次生成的干净联合 AV latent，经独立 3D 学习型模型放大视频空间，再在高分辨率画布继续联合音画去噪。它与普通插值放大不同，也不会修改原来的 `Latent Upscale by 32` 稳定节点。

## 工作流

- `2026-08-19_H3_Learned_Latent_TwoPass_I2VA_Advanced_EXP.json`：I2VA按公开原版schedule执行低分辨率4步、学习型latent放大、高分辨率3步；二采专用Mixer默认旁路，可显式开启Tail/Bias/STG/Restart。
- `2026-08-16_H3_Latent_Upscale_By32.json`：普通插值latent放大，只负责32像素整除和画幅误差报告，不创造学习型细节。

## 模型与成果

学习型工作流读取 `ComfyUI/models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors`。当前固定文件SHA-256为 `043E5A48E161610EF6C3EA974645220354D06FA618ABCA15F76D084812EB55C2`，322个FP16张量、24通道3D输入输出。节点只处理视频latent，音频latent保持联合H3原值；默认每次完成后只定向卸载该放大模型的GPU权重。

当前已确认本地和上游网络的322个参数及固定随机输入输出逐位一致。纠偏后的真实I2VA 4+3链在
238.96秒完成，输出1120×640、124帧、24fps、32kHz双声道，视频/音频严格解码通过；但单素材不能
证明普遍画质或显存优势，因此继续保留Advanced/EXP。
同输入只开启二采Mixer的Tail +3也完成实测：高分辨率阶段由3次增至6次前向，任务302.85秒，媒体
合同与严格解码再次通过。该结果只证明节点可正确串联，不自动证明画质增益。

## 使用方法

1. 第一套 Conditioning 使用低分辨率；第一采样器必须把 `denoised_output` 接到 Learned Latent Upscale，不能使用仍处在中间噪声状态的 `output`。
2. 第二套 Conditioning 必须用相同提示词、首帧、参考媒体和时长，但宽高改为最终高分辨率。Reconcile会拒绝低分辨率旧keyframe，避免 `cond_video_rows` 错配。
3. 使用`Learned Two-Pass Parity Plan`，默认`base_steps=8 / coarse_steps=4 / refine_steps=3 / shift 6/3`；旧`Two-Pass Sigma Plan`保留老工作流兼容，但不再作为原版复现推荐。
4. Parity Plan的`coarse_sigmas`接第一采样器；`refine_sigmas`接`Two-Pass Detail Mixer.refine_sigmas`。Mixer的MODEL、SAMPLER、SIGMAS三路接第二采样器。
5. Mixer全部开关关闭时严格透传高分辨率refine schedule；需要Tail +3时只打开`enable_tail=true`并保留`extra_tail_steps=3`。Bias/STG/Restart可组合但会改变联合音画预测，必须试听审片。
6. `Temporal Detail Enhance`不能接在latent放大和第二采样器之间；它只能接在AV Decode的IMAGE输出之后，AUDIO直接旁路到保存节点。
7. Reconcile的`audio_policy=auto`在`lock_source/remix_source`时采用高分辨率模板音频，其余模式延续首次生成音频。
8. 默认 `offload_after` 只释放学习型放大模型；`clear_after`连CPU缓存一起清理，`keep_loaded`会持续占显存，只适合连续批量放大。
9. 示例固定用VHS的H.265 MP4。本机Windows上H.264曾产生一帧损坏；H.265结果已通过124帧、32kHz双声道和`-xerror -err_detect explode`严格解码。

## 边界

这是二阶段生成，不是把最终视频直接锐化。模型最大只允许4倍空间放大、输出仍受H3 2.0MP面积上限；不能据单次运行宣称普遍更清晰或16GB永不OOM。Ref2VA、Hybrid、多关键帧和Long Video需分别验证后再扩展，不应从I2VA结果直接外推。
