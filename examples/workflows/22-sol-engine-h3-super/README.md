# NVIDIA H3 Super Acceleration

本目录把 NVIDIA 发布的 H3 Super Acceleration 两阶段方案接入 ComfyUI：MiniMax H3 先以 4 步生成草稿，TAEHV Wide 编码后经官方 x2 latent upscaler 放大，LTX-2.5 再进行 3 步视频细化，最后由 TAEHV Wide 解码。总计 7 次模型更新。

## 工作流

- `2026-08-29_H3_Sol_Engine_Super_Acceleration_LTX25_Advanced_EXP.json`：读取已经生成的 H3 MP4，保留其音频，只将视频送入 LTX-2.5 Stage 2。

## 使用方法

1. 先用现有 MiniMax H3 工作流生成 24fps 草稿视频。
2. 在本工作流的 `LoadVideo` 中选择草稿，并把 H3 使用的正向提示词原样填入 LTX 文本编码器。
3. 准备工作流 NOTE 中列出的 6 个模型。默认使用 Lightricks 官方发布的 Comfy INT8 Dev Transformer/Text Encoder；LoRA 的真实文件名是 `ltx-2.5-22b-distilled-lora-450-bf16.safetensors`，没有 `-1.0` 后缀。`taeltx2_3_wide.pth` 放入 `ComfyUI/models/taehv`；完整 LTX conv VAE 只为 x2 upscaler 提供通道统计。`ComfyUI-SolAttn_triton`为可选加速依赖；缺少时自动使用稠密 attention，不阻止运行。
4. 官方对齐参数为目标 `1920×1088`、Euler、CFG 1、LoRA 0.8、3 次 Stage 2 更新。输入 243 帧时裁为 241 帧；原 H3 音频不进入 LTX，只按最终时长裁切后封装。

本节点不检查模型文件名、哈希、文件大小，也不限制输出总像素。用户可以切换为 BF16 文件复现 NVIDIA 的非量化模型政策，但磁盘与内存需求明显更高；默认 INT8 是 ComfyUI 本地可运行适配，不属于 NVIDIA 公布的固定 4×GB200 基准配置。22.2×不代表消费级显卡速度。

模型来源：[Lightricks LTX-2.5](https://huggingface.co/Lightricks/LTX-2.3/tree/main)、[TAEHV Wide](https://raw.githubusercontent.com/madebyollin/taehv/32ac0146b11007cda5a57b60a3b35653361fb8a4/taeltx2_3_wide.pth)。上游方案：[NVIDIA H3 Super Acceleration](https://nvlabs.github.io/Sana/Sol-Engine/H3-Super-Acceleration/)

## 本机机械验证

2026-08-29 使用隔离的低显存服务串行执行一次：`320×192×22` H3 草稿裁为 17 帧，经三步 Refiner 输出 `640×384×17`。H.264 视频和原 32kHz 双声道 AAC 均通过严格解码，首/中/末帧无黑屏或花屏。该结果只证明这条小样链路可运行，不代表 1080p 性能或质量结论。
