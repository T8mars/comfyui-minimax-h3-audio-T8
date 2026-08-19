# 学习型 Latent 放大与二阶段生成

这一组把低分辨率 H3 首次生成的干净联合 AV latent，经独立 3D 学习型模型放大视频空间，再在高分辨率画布继续联合音画去噪。它与普通插值放大不同，也不会修改原来的 `Latent Upscale by 32` 稳定节点。

## 工作流

- `2026-08-19_H3_Learned_Latent_TwoPass_I2VA_Advanced_EXP.json`：I2VA 低分辨率4步、学习型latent放大、高分辨率4步，合计8次联合AV Transformer调用。
- `2026-08-16_H3_Latent_Upscale_By32.json`：普通插值latent放大，只负责32像素整除和画幅误差报告，不创造学习型细节。

## 模型与成果

学习型工作流读取 `ComfyUI/models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors`。当前固定文件SHA-256为 `043E5A48E161610EF6C3EA974645220354D06FA618ABCA15F76D084812EB55C2`，322个FP16张量、24通道3D输入输出。节点只处理视频latent，音频latent保持联合H3原值；默认每次完成后只定向卸载该放大模型的GPU权重。

当前已完成权重结构、哈希、尺寸、音频保持、异常卸载、sigma公式和高分辨率条件校验。真实I2VA端到端画质、峰值显存和不同素材的稳定性仍需以完整生成报告为准，因此保留Advanced/EXP。

## 使用方法

1. 第一套 Conditioning 使用低分辨率；第一采样器必须把 `denoised_output` 接到 Learned Latent Upscale，不能使用仍处在中间噪声状态的 `output`。
2. 第二套 Conditioning 必须用相同提示词、首帧、参考媒体和时长，但宽高改为最终高分辨率。Reconcile会拒绝低分辨率旧keyframe，避免 `cond_video_rows` 错配。
3. `Two-Pass Sigma Plan` 从实际MODEL读取video/audio shift，默认在base-flow `q=0.5`处分成4+4步；不要再接手工Sigma节点。
4. Reconcile的`audio_policy=auto`在`lock_source/remix_source`时采用高分辨率模板音频，其余模式延续首次生成音频。
5. 默认 `offload_after` 只释放学习型放大模型；`clear_after`连CPU缓存一起清理，`keep_loaded`会持续占显存，只适合连续批量放大。
6. 示例固定用VHS的H.265 MP4并关闭`save_metadata`。本机Windows上H.264曾产生一帧损坏，开启元数据还可能在视频主文件写完`moov`前进入音频封装；H.265结果已通过124帧、32kHz双声道和`-xerror -err_detect explode`严格解码。

## 边界

这是二阶段生成，不是把最终视频直接锐化。模型最大只允许4倍空间放大、输出仍受H3 2.0MP面积上限；不能据单次运行宣称普遍更清晰或16GB永不OOM。Ref2VA、Hybrid、多关键帧和Long Video需分别验证后再扩展，不应从I2VA结果直接外推。
