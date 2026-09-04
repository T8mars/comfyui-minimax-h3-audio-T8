# MiniMax H3 Audio T8

简体中文 | [English](README_EN.md)

这是一个面向 MiniMax H3 的 ComfyUI 节点包。它不只做文生视频，还把图生视频、首尾帧、参考图、参考音频、长视频、口型、加速和成片修复整理成可以直接使用的工作流。

当前版本：**1.71.0** · 292 个节点 · GPL-3.0-or-later

## 先从哪里开始

如果你第一次使用，建议按这个顺序：

1. 从 [`examples/workflows/01-basic-generation`](examples/workflows/01-basic-generation) 跑一条普通 H3 视频。
2. 需要参考图、首尾帧或音频控制时，看 [`02-audio-control`](examples/workflows/02-audio-control) 和对应的参考工作流。
3. 需要 OpenVDN 8 步时，先下载 [`t8star/Vdn-Minimax-H3-Comfy`](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy)，再用 [`10-speed`](examples/workflows/10-speed) 里的 `OpenVDN_DMD8_*_Advanced.json`。
4. 需要 WASD 人物/镜头控制时，看 [`26-h3-world`](examples/workflows/26-h3-world)；首版固定为首帧 I2VA。
5. 需要长视频或 MV 时，看 [`04-long-video`](examples/workflows/04-long-video) 和 [`24-mv-lipsync`](examples/workflows/24-mv-lipsync)。
6. 需要图片或成片超分时，看 [`25-dlss-nr`](examples/workflows/25-dlss-nr)；这是 Windows RTX 专用的可选后处理。

每个高级工作流都带画布说明。先替换模型和输入素材，再运行；不要一开始就把多个 LoRA、Attention 加速器和采样器叠在一起。

## 能做什么

### H3 生成和参考控制

- 文生视频 T2VA
- 首帧图生视频 I2VA
- 尾帧生成 L2VA
- 首尾帧生成 FL2VA
- 单张或多张参考图 Ref2VA
- 参考视频、参考音频和混合参考
- 原生视频与音频联合生成、解码和保存
- H3-World 首帧 I2VA：用 WASD 和 IJKL/F 时间线控制人物与镜头

### 音频和口型

- 原声锁定、参考音色、对白和音轨混合
- Vocal Lock：用独立人声驱动画面，完整歌曲只在最终成片混入一次
- Audio Refine：可接 Turbo、PDD、EAV、Prompt Relay 和长视频路线
- 本地 ASR、说话人和 SyncNet 检查工具

项目里的 32 秒 Vocal Lock V3 样片已经完成五镜头串行生成、逐镜口型检查和真人完整观看。用户最终反馈为“32秒这个已经没问题了，完美”。这只代表该样片通过，不代表所有素材都会自动得到同样效果。

### 长视频

- 多关键帧和分段续写
- 一次排队、节点内串行生成
- 断点恢复和 accepted manifest
- Native Masked Context Plan B
- 可选 Color Match，默认开启，用于减轻分段接缝颜色跳变

### 加速和成片修复

- OpenVDN DMD 8 步 / Stage B 50 步
- PDD、SLA、SPEED、FastH3 VSA、Enhance-A-Video
- DLSS-NR 图片/短视频帧/长视频文件超分，以及 FlashVSR、RealBasicVSR、RAFT、Skin Finish
- NVIDIA H3 + LTX-2.5 两阶段超分实验路线

带 `Advanced` 或 `EXP` 的功能需要使用对应工作流。不同加速方案通常不能叠加；节点会尽量提前拒绝明显冲突。

## 安装

### ComfyUI Manager

在 Manager 中搜索 `MiniMax H3 Audio T8`，安装后完全重启 ComfyUI。

### 手动安装

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git minimax-h3-audio-T8
```

本节点依赖较新的 ComfyUI 原生 MiniMax H3 支持。如果节点全部变红、工作流提示缺少节点，先更新 ComfyUI 本体、前端和 Manager，再彻底退出并重新启动。

## 模型放哪里

常用目录如下：

| 模型 | ComfyUI 目录 |
| --- | --- |
| H3 主模型 | `models/diffusion_models` |
| Qwen3-VL 文本编码器 | `models/text_encoders` |
| 视频与音频 VAE | `models/vae` |
| Turbo、PDD、SLA 等 LoRA | `models/loras` |
| OpenVDN 完整模型包 | [`t8star/Vdn-Minimax-H3-Comfy`](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy)；仓库已按 `models` 下的正确目录整理 |
| H3-World 动作 LoRA | [`DANNY621/H3-World`](https://huggingface.co/DANNY621/H3-World) 的 `step-10000.safetensors` → `models/loras/minimax/H3-World` |
| FlashVSR / RealBasicVSR | `models/upscale_models` 或工作流注明的专用目录 |
| DLSS-NR v1.3 外部运行时（不是模型） | `models/DLSS-NR/1.3`（用户自行取得，节点不下载或分发） |
| 人脸、光流和分割模型 | 工作流或对应文档注明的目录 |

不同 H3 基模和 LoRA 不是随便混用的。文件名相近也不代表结构兼容。

## DLSS-NR：Windows RTX 可选后处理

[`examples/workflows/25-dlss-nr`](examples/workflows/25-dlss-nr) 提供运行时检查、图片、短视频帧序列和
文件视频四份独立工作流。默认使用 v1.3 `Standard + 2x`。

**DLSS-NR 不需要新的 PyTorch / safetensors 模型，但需要外部程序。** 请从
[`video2dlssnr` v1.3 官方 Release](https://github.com/DaniilSokolyuk/video2dlssnr/releases/tag/v1.3)
取得 `video2dlssnr_release.zip` 完整包；不要用 light 包，也不需要安装上游的 ComfyUI 节点包。本项目
不会自动下载、安装或分发其中的 EXE 和 NVIDIA DLL。

运行条件：

- Windows 10/11、NVIDIA RTX 显卡、NVIDIA 驱动 **616.56 或更新版本**
- 图片和帧序列路线不增加新的 pip 依赖；使用 ComfyUI 已有的 Torch、NumPy 和 PyAV 环境
- `Video File` 路线还要求 `ffprobe` 可在 `PATH` 中找到；通常安装 FFmpeg 后即可获得
- 阅读并接受外部运行时及 NVIDIA 的适用许可；工作流中的许可开关因此默认关闭

正确目录结构如下。保留完整 ZIP，把其中 `out` 目录的四个文件复制到这里的 `bin`，并把本仓库的
[`examples/runtime-manifests/dlss-nr-v1.3.json`](examples/runtime-manifests/dlss-nr-v1.3.json)
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

先运行 `Runtime_Audit`。节点会校验完整包及已解压文件的哈希、驱动、GPU 映射和真实 feature probe；
只有显示 `READY` 后才运行超分。v1.2 只允许 1× NR-only，默认 2× 工作流必须使用 v1.3。

三类固定素材的四路盲测均通过非退化门。人审结论是不同高清方法会带来不同皮肤和质感，没有一种在所有
素材上一定更好。因此这里的Standard只是推荐起点；超分不修复源片已有的身份、口型或真实纹理问题。

## H3-World：WASD 人物与镜头控制

上游项目：[`Danzer1xxxxChan/H3-World`](https://github.com/Danzer1xxxxChan/H3-World) ·
模型：[`DANNY621/H3-World`](https://huggingface.co/DANNY621/H3-World)

首版只做一件明确的事：输入一张首帧图，以 832×480、124 帧、24fps 生成一条约 5.17 秒的 I2VA，
并让 37 个潜空间时间点分别接收人物与镜头动作。预设包括前进、后退、左右移动、上下倾斜、左右摇镜和
快速摇镜；`custom` 可以用 JSON 分段组合动作。非 832×480 的首帧会等比覆盖后居中裁剪，不会直接拉伸。

下载 LoRA：

```powershell
hf download DANNY621/H3-World step-10000.safetensors --local-dir ComfyUI/models/loras/minimax/H3-World
```

这个 LoRA 已经是 ComfyUI 可直接加载的 104 对 A/B 权重，不需要另行转换。工作流还需要现有的完整
`minimax_h3_fl2va_int8_convrot.safetensors`、Qwen3-VL 编码器、视频 VAE 和音频 VAE。它不增加新的
pip 依赖，但最终安全保存要求 `ffmpeg` 可在 `PATH` 中找到。请直接使用
[`examples/workflows/26-h3-world`](examples/workflows/26-h3-world) 的工作流，不要叠加 OpenVDN、SLA、
VSA、Sol-Attn、BlockCache 或另一个模型/Attention 接管节点。

固定停车场样本的匿名盲测已通过：用户正确识别出持续前进的 H3-World 版本，确认动作稳定、两边声音
正常，画面质量持平。该结论支持这条固定合同转为正式 Advanced 功能，不代表任意人物、动作或显存配置
都能得到相同结果。

## OpenVDN：推荐的 8 步路线

完整模型包：[`t8star/Vdn-Minimax-H3-Comfy`](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy)

模型仓库已经按 ComfyUI 的 `diffusion_models`、`text_encoders` 和 `vae` 目录整理好。获得模型访问权限后，可以直接下载到 `ComfyUI/models`：

```powershell
hf auth login
hf download t8star/Vdn-Minimax-H3-Comfy --local-dir ComfyUI/models
```

然后打开 [`examples/workflows/10-speed`](examples/workflows/10-speed) 中的正式 OpenVDN 工作流。目前提供：

- T2VA
- I2VA
- L2VA
- FL2VA
- 单图 Ref2VA
- 多图 Ref2VA
- 参考视频加原音轨
- 独立参考音频
- 首帧加参考音频 Hybrid

OpenVDN 上游公开说明的是 T2VA；其他模式是本项目利用 ComfyUI 原生 H3 条件布局实现并真实跑通的扩展。

### 完整底模和 pruned 底模都可以用

正式工作流默认使用：

```text
models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors
```

这是完整、非 pruned、AdaLN 输入宽度为 2688 的底模。安装新版模型包后，也可以在同一工作流里改用：

```text
models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
```

Composer 会读取底模里的 `adaln_t_table`，按内容 SHA 自动选择配套的 curve-projected Turbo adapter；用户不需要再接一个 LoRA 节点。该适配器保留 208 个原样可用的目标，并把 51 个 AdaLN 目标转换成 8 维 LoRA 加 51 个偏置残差，共实际应用 310 个补丁。完整底模仍走原生 2688 维 adapter。

兼容是按曲线表内容识别，不是只看文件名。目前支持模型包中的 FL2VA pruned INT8/ConvRot（同曲线表的 FL2VA pruned FP8 也通过静态签名门）；未知 pruned/curve-basis 模型仍会在采样前给出明确错误，避免套错适配器后静默生成。

### 已完成的验证

使用完整底模，项目在同一台 RTX 4060 Ti 16GB 上严格串行测试了 I2VA、L2VA、FL2VA、单图、双图、视频加音频、独立音频和首帧加音频共八条路线。随后又用 pruned INT8 底模在 320×192×39 下串行复跑了 T2VA 和同样八条多模态路线。每条 pruned 测试都完成：

- 800 个 OpenVDN 分支张量
- 104 个 default adapter 目标
- 259 个逻辑 turbo adapter 目标，其中 51 个 AdaLN 目标带独立偏置残差，共 310 个实际补丁
- 8 个 Euler/native-flow 步骤，video/audio shift 为 12/3
- 原生 H.264 视频、AAC 音频和联合严格解码
- 运行日志中 `ERROR lora = 0`

这些结果证明工作流和两类底模都能正确组合，不等于所有提示词、参考图、声音或显卡都已经通过画质验收。完整底模八条短测试最低剩余显存为 535–890MiB；pruned 九条短测试中最低只有 290MiB，仅 T2VA 和 I2VA 超过本项目的 512MiB 余量门。16GB 显卡仍应一次只跑一个任务，并根据实际占用降低分辨率或帧数。

## 常见问题

### 节点全红或找不到

先更新 ComfyUI、前端和 Manager，再完全退出重启。只更新本插件通常不够。

### 一运行就显存不足

先降低分辨率、帧数和参考素材数量，关闭同时运行的其他生成任务。16GB 显卡不要并发跑两条 H3。

### 声音变成噪音或音量异常

先检查工作流指定的 sampler、scheduler、步数、video/audio shift 和 LoRA。不要把通用 EMA、Ref2VA LoRA、OpenVDN turbo 或其他加速 LoRA随意互换。

### OpenVDN 提示 AdaLN 不兼容

先确认新版模型包中存在 `stage-dmd-step-250/adapters/turbo_pruned_curve_fl2va/adapter_model.safetensors`。如果文件存在仍报错，说明所选 pruned 底模的曲线表与已支持版本不同；换用模型包中的 FL2VA pruned INT8，或暂时换回完整 `minimax_h3_fl2va_int8_convrot.safetensors`。不要靠改文件名绕过签名检查。

### 能不能叠加 SLA、VSA、Sol-Attn 或 BlockCache

OpenVDN 自己接管模型分支和 adapter。不要再叠加其他 MODEL 或 Attention 接管节点；这不是“多加一个会更快”，通常只会造成冲突或错误结果。

## 文档

- [工作流总览](examples/workflows)
- [ComfyUI 使用与模型说明](docs/README_ComfyUI.md)
- [验证记录](docs/VERIFICATION_REPORT.md)
- [功能清单](features.json)

## 许可证

节点源码使用 GPL-3.0-or-later。

模型有各自的许可证。MiniMax H3 及其衍生模型遵循 MiniMax H3 Community License Agreement；协议定义的适用地区不包括欧盟、英国、韩国和美国。下载、运行或再分发模型前，请阅读模型仓库中的完整协议和 Acceptable Use Policy。

本 GitHub 仓库不包含模型权重。OpenVDN 模型包在 Hugging Face 单独提供，并保留原始许可、NOTICE、来源和修改说明。
