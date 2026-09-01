# MiniMax H3 Audio T8

简体中文 | [English](README_EN.md)

MiniMax H3 的 ComfyUI 节点包：生成视频和声音，也支持参考控制、长视频、修脸和加速。

当前版本：**1.61.0** · GPL-3.0-or-later

## 主要功能

- 文生、图生、首尾帧、参考图/视频/音频、混合参考
- 视频和音频双时钟采样
- 原声保留、参考音频、音轨合成；Audio Refine 可选接入 Turbo4/8、双采、PDD、EAV、Prompt Relay 和长视频8步
- 多关键帧、长视频续写、节点内一键串行、断点恢复、Latent 放大；双采可用v8人物安全RGB后处理，仅在人工逐帧alpha内采用T2、其余画面与音频保留D0
- 单人/多人脸部修复、SAM3.1 追踪、Skin Finish
- Prompt Relay、SPEED、SLA、PDD、Enhance-A-Video
- FastH3 Preview：T2VA 4步，可选真实 learned-gate VSA 90% 稀疏执行
- NVIDIA H3 Super Acceleration：H3 4步草稿经完整 LTX VAE 编码后接 LTX-2.5 3步 Refiner（TAEHV仅最终解码）
- FlashVSR v1.1：成片2×/4×超分、固定LCSA、动态预算候选和低显存分块，原音频不处理
- RAFT运动审计、轨迹控制、RealBasicVSR、FreeNoise、AYS校准契约、CADS视觉参考退火

带 `Advanced` 或 `EXP` 的节点属于高级/实验功能，建议直接使用配套工作流。

## 安装

在 ComfyUI Manager 搜索 `MiniMax H3 Audio T8`，安装后重启 ComfyUI。

> **先更新 ComfyUI 本体：** 本节点包使用新版 ComfyUI 的原生 MiniMax H3、`comfy_api.latest`、模型补丁和权重适配接口。只更新节点包但保留旧版 ComfyUI，可能导致整套 T8 节点同时爆红或显示缺失。请把 ComfyUI 本体、前端和 Manager 一起更新，再完全退出并重启。

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

希望只排队一次就自动完成全部长视频分段时，使用`04-long-video`中的
`In_Node_Long_Video_Loop`工作流；需要逐段人工挑片时继续使用原有Background/Accepted路线。
需要同时使用Prompt Relay和Enhance-A-Video时，使用同目录的
`In_Node_Long_Video_Prompt_Relay_EAV_Stock20_Advanced`工作流（原生20步，不接Turbo LoRA）。
需要尾段细分或独立低 Sigma 二次采样时，使用同目录带
`Long_Video_Sampling_Plan`的工作流；断开该节点即恢复旧路径。

[查看全部工作流分类](examples/workflows/README.md)

## 模型目录

| 模型 | 放置目录 |
| --- | --- |
| H3 主模型 | `models/diffusion_models` |
| 文本编码器 | `models/text_encoders` |
| 视频/音频 VAE | `models/vae` |
| Turbo、SLA、PDD 等 LoRA | `models/loras` |
| Latent 放大模型 | `models/latent_upscale_models` |
| H3 Fun Control（新版 Model Patch / 旧版 ControlNet） | `models/model_patches` / `models/controlnet` |
| TAEH3 快速预览模型 | `models/vae_approx` |
| RAFT 光流模型 | `models/optical_flow` |
| RealBasicVSR | `models/upscale_models` |
| FlashVSR v1.1 整套目录 | `models/FlashVSR-v1.1` |

FL2VA、Ref2VA、pruned 和完整基模不要混用。

## 常用设置

- Turbo 双时钟常用 shift：视频 `12`，音频 `3`
- 保留原声：`audio_mode=lock_source`，保存节点连接 `mux_audio`
- 媒体标签：`<Picture 1>`、`<Video 1>`、`<Audio 1>`，编号必须对应输入
- 宽高使用 32 的倍数；显存不足时先降分辨率、帧数和参考数量
- `1920×1088` 只是风险参考面积，不是执行上限；更大画布会警告但不拦截，OOM 与画质风险由用户承担
- 不要同时叠加多个接管 sampler、attention 或 MODEL forward 的节点

## PDD 8 步

工作流在 [`19-pdd-acceleration`](examples/workflows/19-pdd-acceleration)。PDD 必须使用专用节点，不能当普通 LoRA 加载。

转换后的 FL2VA / Ref2VA PDD 加速 LoRA 下载：[t8star/MiniMax-H3-Acc-8Step-comfy](https://huggingface.co/t8star/MiniMax-H3-Acc-8Step-comfy)。下载后放到 `ComfyUI/models/loras`，并选择与基模一致的版本。

默认：Euler/simple、8 NFE、shift `12 / 3`、CFG 1。

新版 ComfyUI 会自动使用官方 PDD FinalLayer；旧版继续走本项目的兼容回退。判断只看运行时能力，不按 ComfyUI 版本、模型哈希或文件大小拦截。

目录中同时提供 FL2VA / Ref2VA 的学习型 latent 双采工作流：严格把同一条 PDD 轨迹分成 LOW 4 步和 HIGH 4 步，总 NFE 仍为 8。正式 Ref2VA 预设为 864×480×22、1.5×；FL2VA 双采暂保留实验标记。

## FastH3 VSA 4 步

工作流在 [`10-speed`](examples/workflows/10-speed)，文件名含 `FastH3_VSA_T2VA_4Step`。下载官方
[VSA Data-Free 适配器](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/blob/main/vsa-datafree/adapter_model.safetensors)，放到
`models/loras/FastH3-VSA/vsa-datafree/adapter_model.safetensors`。

该路线只支持普通 T2VA、4 NFE、shift `12 / 3`。真正 VSA 还需要带
`topk_ratio / block_len / coarse_gate`接口的 Comfy Kitchen；缺少内核或50层 learned gate 时会明确回退
Dense 4步，不会冒充稀疏执行。安装与源码构建说明见[`10-speed/README.md`](examples/workflows/10-speed/README.md)。

## 官方核心兼容

[`20-core-compatibility`](examples/workflows/20-core-compatibility) 提供 AV latent、H3 Attention Hook 和每步 host sync 的按需兼容节点。H3 Audio VAE 会自动关闭旧版按对齐长度裁尾，非整倍数音频不再少掉最后一个 latent step；新版核心和非 H3 VAE 不受影响。Tiled VAE 全局坐标候选仍只保留默认旁路审计。旧工作流不需要修改。

## NVIDIA H3 Super Acceleration

这是两阶段方案，不是 H3 Attention 开关：H3 先跑 4 步草稿，完整 LTX-2.5 Video VAE 编码后经官方 x2 latent upscaler 放大，LTX-2.5 再做 3 步 Refiner，最后才由 TAEHV Wide 快速解码。TAEHV Encode 不得作为 Refiner 输入；H3 音频直接旁路并在最终保存时复用。

目录内同时提供低 Sigma 保脸实验版：`0.5 → 0.412 → 0.350 → 0`，仍是3次 Euler 更新，默认 Dense Attention。它保留更多放大后的原 latent，适合官方完全降噪版导致人脸变化过大时做 A/B；不是官方 Sigma 对齐路线，也不保证所有素材都更好。

配套模型整包：[t8star/Minimax-H3-Super-Acceleration-Comfy](https://huggingface.co/t8star/Minimax-H3-Super-Acceleration-Comfy)。下载时保留仓库中的目录结构，把各目录复制到 `ComfyUI/models` 即可。完整文件名和路径见 [`22-sol-engine-h3-super`](examples/workflows/22-sol-engine-h3-super)。Sol-Attn 是可选依赖，未安装时自动使用 Dense Attention。

## FlashVSR 视频超分

工作流在 [`23-flashvsr`](examples/workflows/23-flashvsr)。下载官方 [FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1)，把整个目录放到 `ComfyUI/models/FlashVSR-v1.1`；模型仓库未包含的 `posi_prompt.pth` 从[官方 FlashVSR 仓库](https://github.com/OpenImagingLab/FlashVSR/tree/main/examples/WanVSR/prompt_tensor)补到同一目录。再从 [SpargeAttn](https://github.com/thu-ml/SpargeAttn) 或其 [Windows wheels](https://github.com/woct0rdho/SpargeAttn/releases)安装与当前 Torch/CUDA 匹配的 `spas_sage_attn`。

先用 `Quality Locked`：固定 `2.0 / 3.0 / 11` LCSA。`Balanced Dynamic` 会改低运动块预算，必须人工看成片；`Memory Safe` 用分块和卸载换显存，通常更慢。官方主要面向4×，项目也提供保守2×实验路线。节点不设模型哈希、文件大小或像素上限，音频原对象直通。

## 社区创作增强

[`21-community-advanced`](examples/workflows/21-community-advanced) 提供 Fun Control、长视频人物音色/句界、接缝漂移审计、低显存策略、Creator语义缓存、TAEH3原生快速预览检查和只读诊断。Fun Control 同时兼容新版官方 `MODEL_PATCH` 合同和旧版 ControlNet：新版模型放`models/model_patches`，旧模型放`models/controlnet`，运行时按能力选择，不按版本号或模型哈希拦截。模型可从 [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/controlnet)取得。从[madebyollin/taehv](https://github.com/madebyollin/taehv)下载`taeh3.safetensors`放入`models/vae_approx`。

需要把Qwen参考前缀缓存与外部[T8 BlockCache](https://github.com/T8mars/comfyui-minimax-h3-blockcache-T8)组合时，使用[`12-system-memory`](examples/workflows/12-system-memory)中的Ref2VA Stock20模板。它是性能优先EXP，不保证bit-exact、省显存或16GB安全。

## 论文能力实验

- [`07-motion-detail`](examples/workflows/07-motion-detail)：RAFT运动审计/MASK传播、轨迹控制、RealBasicVSR时序恢复、H3双时钟AYS校准契约。
- [`04-long-video`](examples/workflows/04-long-video)：FreeNoise视频初始噪声重排，可接普通或Prompt Relay/EAV内循环。
- [`03-image-video-edit`](examples/workflows/03-image-video-edit)：CADS视觉参考退火；只改视觉条件，不改音频条件。

AYS没有可直接套用的H3官方最优时间表；默认仍是原生flow。FreeInit和PAG目前没有可靠的H3联合音视频数学/Attention合同，因此没有做同名伪实现。

## 常见问题

- **8 月 22 日后更新，所有 T8 节点同时爆红/显示缺失：** 这是插件在启动时整体导入失败，不是模型或工作流参数问题。先把 ComfyUI 本体、前端和 Manager 一起更新，完全退出后重启。
- **根据启动终端的第一条报错判断：** 缺少 `comfy_api.latest`、`comfy.weight_adapter`、`comfy.patcher_extension` 或 `comfy.ldm.minimax`，说明 ComfyUI 本体过旧；缺少 `torch`、`torchaudio`、`numpy`、`safetensors` 或 `PIL`（安装包名是 `Pillow`），说明当前 ComfyUI 使用的 Python 基础环境不完整。
- **修复依赖：** 本项目根目录的 `requirements.txt` 为空是正常设计，基础依赖由 ComfyUI 提供。请用启动 ComfyUI 的同一个 Python，重新安装 **ComfyUI 本体**的 `requirements.txt`；整合包用户优先使用整合包更新器。不要在系统 Python 中安装，也不要为了基础节点盲装 SLA、Transformers、OpenCV 等可选依赖，以免替换 Torch/CUDA。反馈时请附第一段完整 `IMPORT FAILED` / `ModuleNotFoundError`、ComfyUI 版本和本节点版本。
- **参数错位或 NaN：** 完整重启 ComfyUI，再重新载入工作流。
- **媒体标签报错：** 检查素材连接和标签编号。
- **Prompt Relay tokenizer 报错：** 优先使用原生 `Load CLIP` 并选择 `type=minimax`；1.52.3起兼容隐藏内部 tokenizer 的CLIP包装，但实际token必须与原生H3逐项一致。
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
