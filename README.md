# MiniMax H3 Audio T8

用于 ComfyUI 的 MiniMax H3 视频与声音节点：参考图／音频、双采与长视频、人物口型、运镜编辑，以及可选高清后处理。

简体中文 | [English](README_EN.md) · 当前版本：**1.86.0** · [更新日志](CHANGELOG.md)

## 安装

需要较新的 ComfyUI 原生 H3 支持。手动安装：

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git minimax-h3-audio-T8
```

安装或更新后，完全退出并重启 ComfyUI，再刷新网页。Manager 可搜索 `MiniMax H3 Audio T8`；Registry 与 GitHub 发布进度独立，未显示最新版时使用 GitHub 安装。

## 选一个工作流开始

先替换自己的模型和素材，不要直接运行示例里的占位文件。

|你想做什么|入口|
|---|---|
|第一次跑 H3／首帧图生视频|[基础生成](examples/workflows/01-basic-generation)|
|音频参考、人物说话／唱歌|[音频控制](examples/workflows/02-audio-control) · [原生音色／情绪与 Avatar](examples/workflows/36-avatar-voice/README.md)|
|双模型 4+4、分段长视频|[长视频](examples/workflows/04-long-video)|
|可视化编排镜头、素材和声音|[曜石导演台](examples/workflows/39-director-console/README.md)（也可点 ComfyUI 左侧独立的 `T8 曜石导演台`）|
|OpenVDN、FastH3、低显存|[加速工作流](examples/workflows/10-speed) · [低显存说明](docs/H3_MEMORY_NODES_EXP.md)|
|Meridian 图片／视频运镜、源时间编辑|[四节点工作流](examples/workflows/37-meridian/README.md)|
|一采动态预览、定向取消|[TAEH3 预览](docs/TAEH3_SAMPLING_PREVIEW_EXP.md)|
|成片高清放大／插帧|[Topaz](docs/TOPAZ_EXP.md) · [DLSS-NR](examples/workflows/25-dlss-nr) · [DLSS 插帧](examples/workflows/29-dlss-fi)|
|H16-3 分块二采／精修音频|[H16-3 工作流](examples/workflows/13-latent-upscale/2026-09-20_H3_H16_3_Chunked_PASS2_I2VA_Advanced_EXP.json) · [说明](docs/H16_3_CHUNKED_PASS2_EXP.md)|
|其他功能与详细操作|[完整工作流目录](examples/workflows) · [详细使用说明](docs/README_DETAILS_ZH.md)|

## 模型下载与位置

下载前阅读对应模型仓库的 README 和许可。不要把不同版本同名权重互相覆盖。

|模型／资源|下载与 ComfyUI 目录|
|---|---|
|H3 主模型、Qwen、视频／音频 VAE|分别放 `models/diffusion_models`、`models/text_encoders`、`models/vae`；[基础安装说明](docs/README_ComfyUI.md)|
|TAEH3 时序／2D 预览模型|[Taeh3-Comfy](https://huggingface.co/t8star/Taeh3-Comfy) → `models/vae_approx/`；2D 文件另名保存|
|Meridian ConvRot INT8 ＋ Omega 1B512|[Meridian-Comfy](https://huggingface.co/t8star/Meridian-Comfy) → `models/meridian/`；Omega 在 `vggt-omega/checkpoints/`，源码／assets 和 H3 VAE 另备|
|Semantic Bridge|[Semantic-Bridge-Comfy](https://huggingface.co/t8star/Semantic-Bridge-Comfy) → `models/semantic_bridge/t8_compat/`|
|T8 动漫战斗 Semantic Bridge|[模型](https://huggingface.co/t8star/semantic_bridge_T8-comic-combat) → `models/semantic_bridge/t8_compat/`；用本仓库最新 GitHub 源码，设置见[Bridge 说明](docs/SEMANTIC_BRIDGE_EXP.md)|
|OpenVDN 完整包|[Vdn-Minimax-H3-Comfy](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy)；保持仓库目录结构|
|H3-World 动作 LoRA|[Minimax-H3-World-Comfy](https://huggingface.co/t8star/Minimax-H3-World-Comfy)；[接线说明](examples/workflows/26-h3-world)|

Meridian 的 DMD 已合并，**不要再加载同一 DMD LoRA**。Omega 是官方原始 PT 几何模型，不是 INT8 或普通 VGGT。TAEH3 只做近似预览，不替换最终 VAE，不生成声音。

## 使用注意

- `Advanced`／`EXP` 使用对应工作流；指定样片通过不代表所有素材、显卡或长窗口都能得到相同效果。
- 旧工作流不强制迁移；未验证的 LoRA／Sage／Sol 组合只提示边界，不一刀切禁止连接。见[补丁共存策略](docs/PATCH_STACK_POLICY.md)。
- 不保证精确声纹、逐字时序、无接缝或通用 16GB 性能。双采仍有轻微接缝变色的已知限制。
- Topaz／DLSS 是独立后处理；各自的程序、资源和运行条件见对应文档，不能只下载 H3 权重。

节点变红、声音异常或模型不兼容时，先看[常见问题与高级配置](docs/README_DETAILS_ZH.md)。反馈请附完整报错、工作流与 Core／节点版本：[提交 Issue](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues)，移除密钥和隐私素材。

## T8 链接

|平台|入口|
|---|---|
|B站／YouTube|[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/)|
|API／在线 AI 应用|[API（推广链接）](https://api.seedance.nz/sign-up?aff=5f4w) · [RunningHub（邀请链接）](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)|
|整合包／模型网盘|[ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [模型网盘](https://pan.quark.cn/s/c9c267081fbf)|
|模型／节点|[Hugging Face](https://huggingface.co/t8star) · [节点 GitHub](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)|

## 许可与文档

节点源码：[GPL-3.0-or-later](LICENSE)。本 GitHub 不含模型权重；模型及外部资源各自遵守原许可。Meridian 衍生权重遵守 MiniMax H3 Community License，Omega 单独遵守 FAIR 非商用研究许可；转换不改变这些边界。

[更新日志](CHANGELOG.md) · [第三方说明](THIRD_PARTY_NOTICES.md) · [功能清单](features.json) · [首页维护规则](docs/README_POLICY.md)
