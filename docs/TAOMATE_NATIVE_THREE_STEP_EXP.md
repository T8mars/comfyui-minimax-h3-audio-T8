# TaoMate 三步原生 Comfy 工作流

本页说明转换 LoRA 的短片路线，不是 TaoMate 原项目的流式 KV 运行器，也不是 SelfLift 渐进双采。
三步与四步的指定短片已完成机械核验、浏览器往返与人工审片，验收范围见
[已审工作流索引](../examples/workflows/33-selflift-taomate/README.md)。
本页是旧 TaoMate 单分辨率配方；新 TaoMate-HD 的 2 LOW＋1 HIGH 另见
[HD 说明](NOV05_TAOMATE_HD.md)，两者的模型、时间表和验收不能互换。

## 模型配对

本次固定使用完整 FL2VA int8 convrot 底模、原 H3 文本编码器与音视频 VAE，以及
`minimax_h3_taomate_fl2va_3step_ema_comfyui.safetensors`，通过现有 H3 兼容 LoRA 加载器，强度1。
该文件约2.48GB，检查到208组 rank128/alpha128 适配器，实际加载208个目标，无遗漏。
文件放在 ComfyUI 的 `models/loras/`，发行包不包含权重。

[drbaph 固定版本说明](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/633e6c48cd05f7800e2cb2c007b5c9023b92c576/README.md)提供的是这份转换对应的时间表：

| 路线 | 完整视频 SIGMAS |
| --- | --- |
| 三步 | `1.0, 0.961165, 0.853333, 0.0` |
| 四步对照 | `1.0, 0.970874, 0.907249, 0.640000, 0.0` |

Kijai 的 `minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors` 是另一份转换。
本地已做结构检查，但这里的完整模型样片不使用它；不能因为文件更小就直接替换并沿用同一时间表或声称总显存更低。

模型卡许可、上游适配器和底模义务要分别核对。上述 drbaph 固定模型卡标注 Apache-2.0，并要求保留原模型义务；
所核对的 Kijai 固定模型卡没有许可字段。这里只交付调用方法及工作流，不据此重新分发任何模型文件。

## 公共节点如何连接

```text
底模 → H3兼容LoRA → DualClock.MODEL → BasicGuider.model
Conditioning.positive               → BasicGuider.conditioning
Conditioning.av_latent              → DualClock.av_latent、SamplerCustomAdvanced.latent_image
DualClock.sampler                   → SamplerCustomAdvanced.sampler
ManualSigmas                       → SamplerCustomAdvanced.sigmas
RandomNoise                        → SamplerCustomAdvanced.noise
BasicGuider                        → SamplerCustomAdvanced.guider
SamplerCustomAdvanced 第一个输出    → H3 AV Decode → Output Trim → Safe AV Save
```

保留 `euler/native_flow`、video shift12、audio shift3；BasicGuider 为 CFG1。
实际采样 SIGMAS 来自上表的 ManualSigmas，不把 DualClock 默认生成的另一张表误接回来。
不要把视频时间表直接套到音频上，DualClock 的原生音频时间换算仍须保留。

本次 T2VA 条件 `length=73`，裁切输出3秒、72帧、24fps、896×448。
这里是原生单段 Conditioning 的长度，不是 Progressive Long 的内部窗口；后者最小124帧的限制不应搬到这个节点上。
不额外接 latent upscaler，不把三步表拆成3+3，也不替换已接受的渐进4+4工作流。

## Sage 与测试节点

公共图通过 ComfyUI 启动参数 `--use-sage-attention` 使用全局 Core Sage；不要再接显式 PyTorch 或另一个后端覆盖它。
这是启动配置，不是要在当前 Core 的后端下拉中手造不存在的选项。

测试图曾使用专用计数、计时和模型检查节点。公共候选已展开为它们实际调用的
RandomNoise、BasicGuider、SamplerCustomAdvanced 和现有 H3 AV Decode，不依赖测试扩展。
模型、LoRA、条件、seed、完整时间表、裁切、音频和编码参数保持原配方。
显式测试 Sage 选择器与全局配置已在当前 Core 中核对为同一函数；该检查不等同新的 GPU 性能测试。

## 本机短片计时范围

2026-09-14，同一底模、转换 LoRA、seed 和上述3秒配方，各执行一次，图内无缓存命中：

| 路线 | 整图墙钟 | 采样器墙钟 | 视频 VAE 解码 | 音频 VAE 解码 |
| --- | ---: | ---: | ---: | ---: |
| 三步 | 46.90秒 | 28.89秒 | 12.11秒 | 0.28秒 |
| 四步 | 56.01秒 | 38.00秒 | 12.12秒 | 0.26秒 |

整图时间从提交到执行结束并取回历史，不含启动服务、事前资产核验和事后独立审计。
采样器和 VAE 是墙钟测量，包含相应的懒加载/卸载，不是纯 CUDA 内核时间；未强制同步 GPU。
模型加载没有独立、完整的分项计时，不能将 Loader 节点的短返回时间当成全部模型加载耗时。
设备显存为周期性整卡观测，不是此模型独占的精确峰值。这两次结果也不构成统计性能基准或质量等价证明。
不能与内部生成124帧再裁切的 Progressive Long 直接比较；它们的工作量、上下文和恢复开销不同。

## 已验证和未验证

- 两条完整模型短片分别实际执行3次、4次前向；Sage调用分别150次、200次，未观察到 PyTorch 回退，完整音视频与回执核验通过。
- 公共候选已通过实际 Core API 验证，各15个执行节点、21条连线；微型 H3 CPU 对照中，移除观察包装前后的音视频潜空间逐位一致，解码包装结果一致。
- CPU 对照中的 VAE 网络是明确测试替身；真实完整模型样片与该对照是两类证据，不能混写成同一次执行。
- 独立 Chrome 中两图均完成导入、保存、重载及 API 导出；四份实际导出通过 Core CPU 校验，重载前后执行参数一致。测试未排队生成，也不依赖专用诊断节点。
- 后续已完成绑定短片的人声、音乐、口型和画面人审，记录以已审索引为准；不扩大为长片、多参考、不同底模或任意分辨率通过。
- 三步只是前向次数，不能推论总耗时按3/4缩短；加载、文本编码、VAE及保存也占时间。不承诺普遍提速或画质无损。

正式工作流只保留已通过范围；失败配方不晋级，不修改旧图默认值。
