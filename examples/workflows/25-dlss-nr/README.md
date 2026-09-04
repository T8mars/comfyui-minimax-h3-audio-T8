# DLSS-NR 图像与视频超分

这里有四份互相独立的工作流：运行时检查、图片超分、短视频帧序列超分和文件视频流式超分。
执行工作流都使用已验收的 v1.3 `Standard + 2x` 作为起点。

## 使用前准备

DLSS-NR **不是另一个 safetensors 模型**，而是 Windows RTX 专用的外部便携程序。图片和帧序列路线
不增加新的 pip 依赖；文件视频路线还要求安装 FFmpeg，并保证 `ffprobe` 可在 `PATH` 中找到。

1. 使用 Windows 10/11、NVIDIA RTX 显卡和 616.56 或更新的 NVIDIA 驱动。
2. 从 [`video2dlssnr` v1.3 官方 Release](https://github.com/DaniilSokolyuk/video2dlssnr/releases/tag/v1.3)
   下载完整的 `video2dlssnr_release.zip`。不要使用 light 包，也不用安装它提供的 ComfyUI 节点包。
3. 保留完整 ZIP，把 ZIP 中 `out` 目录的四个文件复制到 `bin`，并把
   [`../../runtime-manifests/dlss-nr-v1.3.json`](../../runtime-manifests/dlss-nr-v1.3.json)
   复制为 `t8-runtime-manifest.json`：

   ```text
   ComfyUI/models/DLSS-NR/1.3/
   ├── t8-runtime-manifest.json
   ├── video2dlssnr_release.zip
   └── bin/
       ├── video2dlssnr.exe
       ├── nvngx_dlss.dll
       ├── nvngx_dlssnr.dll
       └── nvngx.dll_dlssnr.dll
   ```

4. 阅读并接受外部运行时及 NVIDIA 的适用许可后，在工作流中打开
   `accept_external_runtime_license`。该开关默认关闭；未打开时会明确拒绝执行。
5. Runtime Audit 必须显示 `READY`。不要绕过驱动、设备、文件哈希或 512 MiB 剩余显存检查。

节点不会下载、安装或分发这些 EXE、DLL，也不负责升级显卡驱动。运行时是便携文件，不需要系统级安装。

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
