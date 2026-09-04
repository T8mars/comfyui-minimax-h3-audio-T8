# DLSS-NR 图像与视频超分

这里有四份互相独立的工作流：运行时检查、图片超分、短视频帧序列超分和文件视频流式超分。
执行工作流都使用已验收的 v1.3 `Standard + 2x` 作为起点。

## 使用前准备

1. 仅支持 Windows 和 NVIDIA RTX 显卡。
2. 用户自行取得完整的外部 DLSS-NR v1.3 运行时，并放到
   `ComfyUI/models/DLSS-NR/1.3/`。节点不会下载、安装或分发 EXE、DLL。
3. 阅读并接受外部运行时及 NVIDIA 的适用许可后，在工作流中打开
   `accept_external_runtime_license`。该开关默认关闭；未打开时会明确拒绝执行。
4. Runtime Audit 必须显示 `READY`。不要绕过驱动、设备、文件哈希或 512 MiB 剩余显存检查。

## 四份工作流

| 文件 | 用途 |
| --- | --- |
| `Runtime_Audit` | 只检查 v1.3 运行时、驱动、GPU 映射和真实 feature probe |
| `Image_2x_Standard` | 每张 SDR 图片独立做 2× SR+NR，并同时预览原图和候选 |
| `Video_Frames_2x_Standard` | 把 H3 短片解成帧，用一个持久进程保持时序状态，并原样传回 AUDIO |
| `Video_File_2x_Standard` | 对未裁切的 SDR 8-bit Rec.709 CFR 文件做有界流式处理，严格复制和校验源音频 |

v1.2 运行时只允许 1× NR-only，不能用于这些默认 2× 工作流。帧序列路线适合短片；长视频用
`Video File`，避免把完整视频物化成 IMAGE batch。

固定三类素材的四路盲测已经通过非退化门，但人审结论是不同方法改变皮肤和质感，没有通用赢家。
因此 Standard 是推荐起点，不是“永远最好”。超分不会修复源片已有的身份、口型或真实纹理问题。
