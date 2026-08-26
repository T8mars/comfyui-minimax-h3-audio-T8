# MiniMax H3 Audio T8

面向 ComfyUI 的 MiniMax H3 音视频节点包。可生成视频与声音，也提供参考控制、长视频、脸部修复和加速工作流。

当前版本：**1.48.1** · GPL-3.0-or-later

## 能做什么

- 文生、图生、首尾帧、参考图/视频/音频和混合参考生音视频
- 视频与音频双时钟采样，兼容已有基础工作流
- 保留原声、音频重混、参考音频和最终音轨合成
- 多关键帧、长视频续写、断点恢复和 Latent 放大
- 单人/多人脸部修复、SAM3.1 人物追踪和 Skin Finish
- Prompt Relay、SPEED、SLA、PDD、Enhance-A-Video 等高级功能

名称带 `Advanced` 或 `EXP` 的节点属于高级或实验功能，请优先使用配套示例工作流。

## 安装

在 ComfyUI Manager 搜索 `MiniMax H3 Audio T8`，安装后重启 ComfyUI。

也可以手动安装：

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git minimax-h3-audio-T8
```

## 模型放哪里

| 模型 | ComfyUI 目录 |
| --- | --- |
| H3 主模型 | `models/diffusion_models` |
| 文本编码器 | `models/text_encoders` |
| 视频/音频 VAE | `models/vae` |
| Turbo、SLA、PDD 等 LoRA | `models/loras` |
| Latent 放大模型 | `models/latent_upscale_models` |

主模型、任务类型、LoRA 和 VAE 必须互相匹配。不要混用 FL2VA、Ref2VA、pruned 和完整基模。

## 快速开始

1. 打开 [`examples/workflows`](examples/workflows)。
2. 新用户先看 `01-basic-generation` 和 `02-audio-control`。
3. 把工作流拖进 ComfyUI，替换其中的模型和素材。
4. 宽高使用 32 的倍数，先用小分辨率确认链路正常。
5. 高级工作流先看所在目录的 `README.md` 和画布 NOTE。

完整分类见 [工作流索引](examples/workflows/README.md)。仓库中的示例均为 ComfyUI 前端工作流，可直接导入。

## 常用设置

- Turbo 双时钟常用 shift：视频 `12`，音频 `3`
- 保留输入原声：`audio_mode=lock_source`，最终保存节点连接 `mux_audio`
- 提示词媒体标签：`<Picture 1>`、`<Video 1>`、`<Audio 1>`，编号要和输入对应
- 高分辨率、长视频和多参考会增加显存，不保证所有 16GB 显卡都能运行所有组合
- 不要随意叠加多个接管 sampler、attention 或 MODEL forward 的节点

## PDD 8 步

`19-pdd-acceleration` 中提供 FL2VA 和 Ref2VA 工作流。PDD 必须使用专用节点，不能当普通 LoRA 使用。

默认合同：Euler/simple、8 NFE、shift `12 / 3`、CFG 1。模型和用法见 [PDD 工作流说明](examples/workflows/19-pdd-acceleration/README.md)。

## 常见问题

**节点参数错位或出现 NaN**

完整重启 ComfyUI，再重新载入工作流。只刷新浏览器可能仍在使用旧节点缓存。

**显存不足（OOM）**

先降低分辨率、帧数和参考素材数量，关闭并发，再逐个排查 SLA、Sage、BlockCache 或二次采样节点。

**提示词媒体标签报错**

确认素材已连接，并检查 `<Picture N>`、`<Video N>`、`<Audio N>` 的编号。

**画面有声音，但不是原声**

使用 `lock_source`，并把 `mux_audio` 接到最终保存节点。`generated_audio` 是模型生成的新音频。

## 文档

- [工作流索引](examples/workflows/README.md)
- [完整使用说明](docs/README_ComfyUI.md)
- [验证结果与已知限制](docs/VERIFICATION_REPORT.md)
- [第三方项目与许可](THIRD_PARTY_NOTICES.md)
- [问题反馈](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues)

## 相关链接

- [B站](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [API](https://api.seedance.nz/sign-up?aff=5f4w)
- [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd)
- [模型网盘](https://pan.quark.cn/s/c9c267081fbf)
- [Hugging Face](https://huggingface.co/t8star)

插件代码采用 GPL-3.0-or-later。模型权重和第三方组件遵循各自许可证；人物、声音和参考素材的使用权由使用者自行确认。
