# MiniMax H3 Audio T8

MiniMax H3 的 ComfyUI 节点包：生成视频和声音，也支持参考控制、长视频、修脸和加速。

当前版本：**1.48.2** · GPL-3.0-or-later

## 主要功能

- 文生、图生、首尾帧、参考图/视频/音频、混合参考
- 视频和音频双时钟采样
- 原声保留、参考音频、音轨合成
- 多关键帧、长视频续写、断点恢复、Latent 放大
- 单人/多人脸部修复、SAM3.1 追踪、Skin Finish
- Prompt Relay、SPEED、SLA、PDD、Enhance-A-Video

带 `Advanced` 或 `EXP` 的节点属于高级/实验功能，建议直接使用配套工作流。

## 安装

在 ComfyUI Manager 搜索 `MiniMax H3 Audio T8`，安装后重启 ComfyUI。

手动安装：

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git minimax-h3-audio-T8
```

## 快速开始

1. 打开 [`examples/workflows`](examples/workflows)。
2. 新用户先用 `01-basic-generation` 或 `02-audio-control`。
3. 把 JSON 拖进 ComfyUI，替换模型和素材后运行。
4. 高级工作流先看同目录的 `README.md` 和画布 NOTE。

[查看全部工作流分类](examples/workflows/README.md)

## 模型目录

| 模型 | 放置目录 |
| --- | --- |
| H3 主模型 | `models/diffusion_models` |
| 文本编码器 | `models/text_encoders` |
| 视频/音频 VAE | `models/vae` |
| Turbo、SLA、PDD 等 LoRA | `models/loras` |
| Latent 放大模型 | `models/latent_upscale_models` |

FL2VA、Ref2VA、pruned 和完整基模不要混用。

## 常用设置

- Turbo 双时钟常用 shift：视频 `12`，音频 `3`
- 保留原声：`audio_mode=lock_source`，保存节点连接 `mux_audio`
- 媒体标签：`<Picture 1>`、`<Video 1>`、`<Audio 1>`，编号必须对应输入
- 宽高使用 32 的倍数；显存不足时先降分辨率、帧数和参考数量
- 不要同时叠加多个接管 sampler、attention 或 MODEL forward 的节点

## PDD 8 步

工作流在 [`19-pdd-acceleration`](examples/workflows/19-pdd-acceleration)。PDD 必须使用专用节点，不能当普通 LoRA 加载。

转换后的 FL2VA / Ref2VA PDD 加速 LoRA 下载：[t8star/MiniMax-H3-Acc-8Step-comfy](https://huggingface.co/t8star/MiniMax-H3-Acc-8Step-comfy)。下载后放到 `ComfyUI/models/loras`，并选择与基模一致的版本。

默认：Euler/simple、8 NFE、shift `12 / 3`、CFG 1。

## 常见问题

- **参数错位或 NaN：** 完整重启 ComfyUI，再重新载入工作流。
- **媒体标签报错：** 检查素材连接和标签编号。
- **没有保留原声：** 使用 `lock_source`，并连接 `mux_audio`。
- **OOM：** 关闭并发，降低分辨率/帧数，再逐个关闭高级节点排查。

## 文档与反馈

- [工作流索引](examples/workflows/README.md)
- [完整使用说明](docs/README_ComfyUI.md)
- [验证结果与限制](docs/VERIFICATION_REPORT.md)
- [问题反馈](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues)
- [第三方项目与许可](THIRD_PARTY_NOTICES.md)

## 相关链接

- [B站](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [API](https://api.seedance.nz/sign-up?aff=5f4w)
- [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd)
- [模型网盘](https://pan.quark.cn/s/c9c267081fbf)
- [Hugging Face](https://huggingface.co/t8star)

模型和第三方组件遵循各自许可证；人物、声音和参考素材的使用权由使用者自行负责。
