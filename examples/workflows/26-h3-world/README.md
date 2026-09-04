# H3-World 首帧 I2VA

工作流：`2026-09-04_H3_World_I2VA_832x480_124f_50step_Advanced.json`

示例首帧在 `assets/h3_world_official_first_frame.png`。第一次使用时把它复制到 `ComfyUI/input`，或在
`Load Image` 节点中直接换成自己的图片。不是 832×480 的图片会按上游规则等比放大到覆盖画布，再从
中心裁成 832×480，不会被直接拉伸变形。

它把 H3-World 的人物和镜头按键控制接到 ComfyUI 原生 MiniMax H3 音画链。首版固定为：

- 首帧 I2VA
- 832×480、124 帧、24fps
- 50 步 Euler / native flow，video/audio shift 12/3
- CFG 1.0（`BasicGuider`）
- 37 个动作时间点

## 需要的文件

```text
ComfyUI/models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors
ComfyUI/models/loras/minimax/H3-World/step-10000.safetensors
ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors
ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors
```

H3-World LoRA 来自 [`DANNY621/H3-World`](https://huggingface.co/DANNY621/H3-World)，已经是
ComfyUI 可直接加载的 104 对 A/B 权重，不需要转换：

```powershell
hf download DANNY621/H3-World step-10000.safetensors --local-dir ComfyUI/models/loras/minimax/H3-World
```

还需要 `ffmpeg` 可在 `PATH` 中找到。安全保存节点会把画面交给隔离的单线程 libx264 进程，并在 H.264
和 AAC 严格解码都成功后才发布 MP4。这是为避免模型占满内存时 ComfyUI 核心进程内编码器偶发写出坏码流。

## 怎么控制

`Action Timeline` 的预设对应：

- `forward/back/strafe-left/strafe-right`：人物移动
- `tilt-up/tilt-down`：镜头上下倾斜
- `pan-left/pan-right`：镜头左右摇动
- `pan-left-fast/pan-right-fast`：快速摇镜
- `still`：人物不动、镜头保持

选择 `custom` 时，JSON 必须从 0 开始、到 37 结束，中间不能有空档或重叠。例如：

```json
[
  {"start_latent": 0, "end_latent": 12, "keys": ["W"]},
  {"start_latent": 12, "end_latent": 25, "keys": ["L"]},
  {"start_latent": 25, "end_latent": 37, "keys": []}
]
```

`W/S`、`A/D`、`I/K`、`J/L` 同时出现时会互相抵消；`F` 只能和 `J` 或 `L` 一起使用。

## 当前限制

这是独立 EXP 路线。不要叠加 OpenVDN、SLA、VSA、Sol-Attn、BlockCache、其他 DiT 替换或 Attention
接管节点。也不要先改分辨率和帧数；这些值与当前动作位置和注意力掩码合同绑定。16GB 显卡一次只运行
一个 H3 任务。
