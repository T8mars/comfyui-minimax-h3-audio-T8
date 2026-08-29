# NVIDIA H3 Super Acceleration

这个工作流读取一条已经生成的 H3 视频，用完整 LTX-2.5 Video VAE 编码后执行三步细化；TAEHV 只负责最终快速解码，原音频直接保留。

工作流：`2026-08-29_H3_Sol_Engine_Super_Acceleration_LTX25_Advanced_EXP.json`

## 模型下载与路径

整包下载：[t8star/Minimax-H3-Super-Acceleration-Comfy](https://huggingface.co/t8star/Minimax-H3-Super-Acceleration-Comfy)

下载时保留目录结构，把下面这些目录复制到 `ComfyUI/models`：

```text
ComfyUI/models/
├─ diffusion_models/
│  └─ ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors
├─ text_encoders/
│  └─ gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
├─ loras/
│  └─ ltx-2.5-22b-distilled-lora-450-bf16.safetensors
├─ vae/
│  └─ ltx-2.5-video-vae-conv-bf16.safetensors
├─ latent_upscale_models/
│  └─ ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
└─ taehv/
   └─ taeltx2_3_wide.pth
```

模型总大小约 45GB。没有 `taehv` 目录时手动创建。

## 使用

1. 用 MiniMax H3 生成 24fps 草稿视频。
2. 把视频载入本工作流，并复制生成草稿时的正向提示词。
3. 确认画布中是 `Full LTX-2.5 Video VAE encode`，不要把旧 `TAEHV Encode` 接到 Refiner。
4. 首次使用保持默认：目标 `1920×1088`、Euler、CFG 1、LoRA `0.8`、LTX Refiner 3步。
5. Sol-Attn 可不安装；找不到时节点自动使用 Dense Attention。

输入 243 帧时会裁为 241 帧。音频不进入 LTX，只按最终视频时长裁切后重新封装。

2026-08-30 已用修正后的完整 LTX VAE 编码链路完成一条低负载真实运行：17帧、640×384、24fps，视频和音频严格解码通过，未出现首帧后崩坏。该测试只确认链路正确，不代表高分辨率画质结论。

上游：[NVIDIA H3 Super Acceleration](https://nvlabs.github.io/Sana/Sol-Engine/H3-Super-Acceleration/) · [Lightricks LTX-2.5](https://huggingface.co/Lightricks/LTX-2.3) · [TAEHV](https://github.com/madebyollin/taehv)
