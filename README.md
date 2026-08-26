# MiniMax H3 Audio T8 for ComfyUI

MiniMax H3 的 ComfyUI 节点包，覆盖视频生成、联合音频、长视频、参考控制和常用后处理。

当前版本：**1.48.0** · GPL-3.0-or-later

## 主要功能

- 文生音视频、图生音视频、首尾帧、尾帧、参考图/视频/音频和混合参考。
- 视频与音频独立时钟，默认兼容现有 H3 工作流。
- 输入音频锁定、重混、参考音频和最终音轨合成。
- 多关键帧、长视频分段、断点恢复和 latent 放大。
- 单人/多人脸部修复、SAM3.1 人物追踪和 Skin Finish 肤质收尾。
- Prompt Relay、SPEED、SLA、PDD、Enhance-A-Video 等实验功能。
- 运行前检查模型、尺寸、显存和节点组合，发现不兼容时明确报错。

旧节点、默认参数和旧工作流会尽量保持兼容。名称带 `Advanced` 或 `EXP` 的功能属于实验路线，建议先用示例工作流确认效果。

## 安装

### ComfyUI Manager

在 Manager 中搜索 `MiniMax H3 Audio T8`，安装后重启 ComfyUI。

### 手动安装

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git minimax-h3-audio-T8
```

然后重启 ComfyUI。基础节点没有额外 Python 依赖；部分高级功能会在各自工作流说明中列出依赖。

## 模型放置

模型不会随插件下载，请放入 ComfyUI 标准目录：

| 模型 | 目录 |
| --- | --- |
| H3 主模型 | `ComfyUI/models/diffusion_models` |
| Qwen / MiniMax 文本编码器 | `ComfyUI/models/text_encoders` |
| 视频 VAE、音频 VAE | `ComfyUI/models/vae` |
| Turbo、SLA、PDD 等 LoRA | `ComfyUI/models/loras` |
| Latent 放大模型 | `ComfyUI/models/latent_upscale_models` |
| 人脸检测/解析模型 | `ComfyUI/models/face_detection` 或工作流 NOTE 指定目录 |

主模型、任务类型、LoRA 和 VAE 必须匹配。不要把 FL2VA、Ref2VA、pruned 和完整基模混用。

## 快速开始

1. 打开 [`examples/workflows`](examples/workflows) 目录。
2. 先从 `01-basic-generation` 或 `02-audio-control` 选择工作流。
3. 替换工作流中的模型、图片、视频和音频占位文件。
4. 保持宽高为 32 的倍数，先用较小画布和 22/124 帧确认链路。
5. 需要高级功能时，再进入对应分类目录并阅读里面的 `README.md` 和画布 NOTE。

完整工作流索引见 [`examples/workflows/README.md`](examples/workflows/README.md)。工作流文件均为 ComfyUI 前端格式，可直接拖入画布。

## 常用设置

- 常规 Turbo 双时钟：视频/音频 shift 通常为 `12 / 3`。
- 保留输入原声：使用 `audio_mode=lock_source`，最终保存节点接 `mux_audio`。
- 参考素材标签：使用 `<Picture 1>`、`<Video 1>`、`<Audio 1>`，编号必须和实际输入一致。
- 高分辨率、长帧数和多参考会明显增加显存；16GB 显卡不代表所有组合都安全。
- 不要同时堆叠多个会接管 sampler、attention 或 MODEL forward 的高级节点；使用专用组合节点或示例工作流。

## PDD 8 步加速

`19-pdd-acceleration` 提供 FL2VA 和 Ref2VA 两份 PDD 工作流。需要：

- 对应的完整、非 pruned H3 基模；
- `MiniMax-H3-FL2VA-Acc-8Step_comfyui_pdd.safetensors` 或
  `MiniMax-H3-Ref2VA-Acc-8Step_comfyui_pdd.safetensors`。

PDD 不是普通 LoRA：必须使用专用节点，固定为 Euler/simple、8 NFE、shift `12 / 3`、CFG 1。不要再串普通 LoRA、SLA 或第二个采样设置节点。

本机已经跑通 FL2VA、Ref2VA 和一条约 0.7MP 的 Ref2VA 联合音视频链；但显存余量接近上限，因此不能保证所有 16GB 环境都安全。详细参数见 [`19-pdd-acceleration/README.md`](examples/workflows/19-pdd-acceleration/README.md)。

## 常见问题

### 更新后节点参数错位或出现 NaN

完整重启 ComfyUI，再重新载入原工作流。不要只刷新浏览器继续使用旧的前端节点缓存。

### OOM

先降低像素、帧数和参考素材数量；关闭并发队列；暂时移除 SLA、Sage、BlockCache、二次采样等叠加功能，再逐项恢复。

### `prompt media tag validation failed`

检查任务类型、媒体是否真的接入，以及 `<Picture N>`、`<Video N>`、`<Audio N>` 编号是否对应。

### 有画面但没有保留原声

使用 `lock_source`，并把 Conditioning 的 `mux_audio` 接到最终视频保存节点。`generated_audio` 是模型重新生成的音频。

### 实验节点拒绝执行

错误通常表示模型、ComfyUI core 或其他补丁不匹配。请保留完整 ComfyUI Error Report，到 GitHub Issue 中提交；不要只截最后一行。

## 文档

- [工作流索引](examples/workflows/README.md)
- [ComfyUI 使用说明](docs/README_ComfyUI.md)
- [验证与已知边界](docs/VERIFICATION_REPORT.md)
- [第三方项目与许可](THIRD_PARTY_NOTICES.md)

## 社区与服务

- [B站：T8star](https://space.bilibili.com/385085361)
- [YouTube：T8star AI](https://www.youtube.com/@T8star-Aix/)
- [API 服务](https://api.seedance.nz/sign-up?aff=5f4w)
- [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd)
- [模型网盘](https://pan.quark.cn/s/c9c267081fbf)
- [Hugging Face](https://huggingface.co/t8star)

## 反馈与许可

- GitHub：<https://github.com/T8mars/comfyui-minimax-h3-audio-T8>
- 问题反馈：<https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues>

插件代码采用 GPL-3.0-or-later。模型权重和第三方组件遵循各自许可证；使用人物、声音和参考素材时，请自行确认使用权。
