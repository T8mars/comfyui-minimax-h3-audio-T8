# HyperVAE 2× 同潜空间对照（EXP）

打开 `2026-09-27_H3_HyperVAE_2x_Same_Latent_5s_Compare_EXP.json`。这张图从 ComfyUI 画布实际运行：真实 FL2VA／Qwen／Turbo 8 步只采样一次，原生 H3 视频 VAE 与 HyperVAE 分别解码同一个 `video_latent`，共用原生音频 VAE 及同一条裁剪音轨，分别保存两个 MP4。

本机对照是 512×288／1024×576、24 fps、120 帧、严格 5.000 秒；音画严格解码均通过。把权重放进 `ComfyUI/models/vae/` 后，从 HyperVAE 加载节点的 `vae_name` 下拉框选择；本图预选 `hyperVAEKrea2Minimax_v20MinimaxX2Upscale.safetensors`，`absolute_path` 留空。换机器只需选实际文件名，不需要改盘符。权重和测试媒体不入库。

这是可选 EXP，同一短片的机械实跑不证明长视频、所有双采路线或主观画质更好。旧图和原生 VAE 默认路径不变。
