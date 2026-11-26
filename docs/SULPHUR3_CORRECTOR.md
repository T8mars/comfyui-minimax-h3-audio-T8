# Sulphur-3 Corrector / T8

Sulphur-3 底模使用现有 `UNETLoader`；`s3_raw_lora` 使用现有 H3 LoRA 加载器。本次新增的是作者可选的 **独立 Corrector 网络**，不是加速 LoRA，也不代替底模。无需安装原来的 ComfyUI-H3-Corrector 插件。

## 模型与节点

|文件|目录|用途|
|---|---|---|
|`s3.safetensors` 或自选 H3 底模|`ComfyUI/models/diffusion_models/`|由原有底模加载器加载|
|`s3_corrector.safetensors`|`ComfyUI/models/loras/` 或 `ComfyUI/models/h3_correctors/`|由 `H3 Sulphur · Corrector加载 / Load (T8)` 加载|

加载节点 ID 为 `MiniMaxH3CorrectorLoaderT8`，输出 `H3_CORRECTOR` 和准备报告；应用节点 ID 为 `MiniMaxH3CorrectorApplyT8`，输出修正后的 `MODEL` 和准备报告。类型与原节点兼容；节点 ID 使用 T8 名称，避免与原插件冲突。原插件的 Load/Apply 也可跨接相同类型。旧图里的 `LoadH3Corrector`、`ApplyH3Corrector` 不会被静默替换，迁移时换成这两个节点并保留参数。

从 `loras` 读取带 `h3_corrector` 架构元数据的 Safetensors 文件。`extra_model_paths.yaml` 配置的 `h3_correctors` 和 `loras` 路径同样可用。两个类别里有相同相对文件名时，优先选用 `h3_correctors`，原工作流可以继续使用原文件。

原文件有 1,997,065,056 个参数，FP32 权重占 7,988,260,224 字节；文件总长 7,988,398,520 字节。加载保持 **FP32**，CUDA 前向按原节点使用 BF16 autocast。不能把权重直接降到 BF16 冒充原版配方；底模、文本编码器、激活还需要额外内存。Corrector 交给 ComfyUI 的 additional-model 管理加载与卸载，不常驻一个独立隐藏 CUDA 模型。

## 接线与参数

底模加载 → 自选 LoRA／注意力／内存节点 → SigmaShift → **Corrector应用** → Guider → 原采样器。Corrector加载输出接应用节点的 `corrector`。

作者普通生成基线是 Euler、`simple` 调度、20 步、CFG 1、视频 shift 12、音频 shift 3。BasicGuider 就是这里的 CFG 1。不需要 Prompt Relay，普通完整提示词可用；结构化 JSON 提示词也可用。节点本身不更改采样器、步数、sigma、底模或 LoRA。

|参数|默认|含义|
|---|---|---|
|`strength`|1|乘在软限幅后的修正量上；0 完全关闭，并不把 Corrector 加入本次 MODEL 的额外加载列表|
|`budget_fraction`|0.5|相对视频速度 RMS 的软预算；强度在此之后相乘，因此最终修正量还受 strength 影响|
|`velocity_floor`|0.001|用于预算计算的速度 RMS 下限|
|`start_step`|0|首次启用的步骤，0 起算|
|`end_step`|-1|最后启用的步骤，包含端点；-1 到最后一步|
|可选 `sigmas`|未接|仅在采样器没有提供 `transformer_options.sample_sigmas` 时使用；有原生表时优先用实际表|

没有 sigma 表时与原节点一样警告并跳过，不伪称已生效。真实缺失 caption、latent 形状不合、文件损坏、非有限数或底层计算错误仍会报出。

修正发生在 CFG 后：用当前视频 latent 与视频预测计算数据方向速度，运行 Corrector，软限制残差，再换回视频预测。**同一次回调内，音频以及视频之后的额外流直接保留原切片。**后续 H3 联合去噪仍可让音画正常相互影响，不能承诺最终生成音频与关闭 Corrector 时相同。原生音频/视频遮罩继续由采样器处理。

## 双采与已有补丁

在 LOW 和 HIGH 的 MODEL 分支上分别接应用节点，可共享一份 Corrector加载输出。每路可独立设置强度和步骤窗口；本地步号按该路实际 `sample_sigmas` 从 0 起算。接自定义 sigma 的分段续采时，以实际传给该次采样的表为准。

保留现有两路的底模、LoRA 顺序与强度、放大器、音频时钟、提示词、参考素材和遮罩。不要把这里的20步普通生成基线当作所有双采的推荐配方。原版 Corrector 训练于 S3；其它可执行 H3 底模、加速和补丁组合允许尝试，效果不自动获得认证。已有 post-CFG 钩子依次保留，后面的用户钩子仍可能再次改变预测。

普通 T8 Stage 可以采样并保存该结果。当前额外 Corrector 网络尚未获得可移植 Stage 身份，**跨进程 Stage 恢复与自动缓存复用不开放**，以免把不同 Corrector 权重当作同一模型命中；正常新采样不受影响。准备报告不会冒充运行完成报告。

## 示例与验证范围

- [UI 候选工作流](../examples/workflows/99-sulphur3/S3_Corrector_T2VA_EXP.json)
- [同配方 API 图](../examples/workflows/99-sulphur3/S3_Corrector_T2VA_EXP_API.json)

示例是 672×384、124 个原生帧、24fps，保留全部原生时长；20步 Euler、CFG1。不叠加加速 LoRA。底模／编码器文件名可替换成已下载的兼容模型。

已用实际 Core 采样检查 Euler/Heun/LCM、单帧参考、两路不同分辨率/调度、音频已知区域、用户补丁保留、0强度一致性、Stage执行与保存，以及新节点追加注册。CPU 小模型是随机权重，只验证执行合同。

另对官方完整 Corrector 权重进行了 CPU 网络前向与 CUDA post-CFG 数值对照：固定输入下，T8 实现与原节点最大绝对差均为0，CUDA保留FP32权重，音频切片逐值一致。完整 Corrector 还通过真实 Core GPU 两步采样，确认 additional-model 自动加载与两次实际修正调用；该项 H3 主网络仍是随机小模型。这不是用 S3 底模生成完整20步视频，**没有新的完整成片或真人画质验收**。高分辨率、长时长、任意第三方组合的效果需分别验证。

## 来源

依据 [SulphurAI/Sulphur-3](https://huggingface.co/SulphurAI/Sulphur-3)，固定 revision `988224b9913740d60ee54b5a53c7a892467edc76` 的 `S3_ComfyUI.zip` 与 `s3_corrector.safetensors`。本仓库独立实现其公开 tensor/metadata 合同，不打包原插件源码或模型权重。来源、文件哈希与模型许可见 [第三方说明](../THIRD_PARTY_NOTICES/SULPHUR3-SOURCES.md)。

## English summary

Load the optional FP32 corrector from `models/loras` or `models/h3_correctors` and apply it to your MODEL before the guider. The loader identifies corrector files in the LoRA folder by their `h3_corrector` metadata. If a relative filename exists in both locations, `h3_correctors` takes precedence. Both nodes use the upstream `H3_CORRECTOR` socket type.

The author's single-pass baseline is Euler/simple, 20 steps, CFG1 and video/audio shifts12/3. For two-pass graphs, apply separately to both MODEL branches, retaining each branch's existing recipe. The audio slice is untouched within each callback; subsequent joint H3 denoising can still change the final audio. A missing sampler schedule produces a warning and skips correction unless optional fallback sigmas are connected.

Full-checkpoint fixed-input CPU and CUDA differential tests match the original node, but no new complete trained-H3 media or human quality acceptance is claimed. Portable Stage cache/resume for this separately loaded network remains unavailable; fresh sampling and saving are supported.
