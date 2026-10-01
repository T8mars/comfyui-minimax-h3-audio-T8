# HyperVAE Krea2+MiniMax v2 2× 视频 VAE（本地 EXP）

这是独立、显式选择的 MiniMax H3 视频 VAE 加载节点，不替换项目原有视频／音频 VAE 节点，也不修改旧工作流或 ComfyUI 核心文件。

## 使用

添加 `MiniMax H3 HyperVAE 2× 加载 (T8 EXP)`。将权重放进 `ComfyUI/models/vae/`，再从 `vae_name` 下拉选择；`absolute_path` 通常留空，只保留给旧工作流兼容，填写时仍优先。将 `video_vae` 输出接到原有 `MiniMax H3 AV Decode` 或其它接受视频 `VAE` 的解码入口，音频 VAE 保持原连接。权重不会自动下载或进入发行包。

该模型的编码仍是原生 H3 空间 16×；解码头输出 12 个相位打包 RGB 通道，再经 PixelShuffle 得到相对普通 H3 VAE **宽高各 2×** 的画面。因此它是输出像素放大 VAE，**不是**采样器之间的 2× latent upscaler；若把它用于已有二采工作流的最终解码，应按 2× 输出分辨率预留内存、检查保存节点尺寸，并以新 `chain_id`／缓存身份运行，不能复用旧结果。普通 ComfyUI VAE Loader 无法直接加载这个 12 通道解码头。

本地文件 SHA-256 `84DA7F476F3732D2B7F6CD51B03063979E8CDCE225032A6D4ED16542775A2DE2` 与发布版本一致。真实权重已完成单帧和 5 帧最小编码／解码，以及单帧 tiled 解码：32×32 输入分别得到 64×64 输出，形状、有限值和 16× 编码合同通过。另有[5秒同潜空间画布工作流](../examples/workflows/58-hypervae-2x/README.md)在隔离真实 ComfyUI 前端运行，两路各120帧、严格5.000秒、音画完整解码、相同解码音轨；原生512×288与HyperVAE1024×576。此项只证明这一个短片的机械兼容；原尺寸、长视频、所有双采路线、主观画质、旧工作流逐帧等价尚未验证，不作为默认或推荐 VAE。

来源：[作者模型页](https://civitai.com/models/2894736/hypervaekrea2minimax?modelVersionId=3301496)、[作者的 MiniMax 2× VAE 解码实现](https://github.com/TripleHeadedMonkey/ComfyUI-MiniMaxH3_LatentUpscaler/blob/main/vae_decode.py)。项目只使用本机用户提供的权重，没有打包第三方代码或模型。
