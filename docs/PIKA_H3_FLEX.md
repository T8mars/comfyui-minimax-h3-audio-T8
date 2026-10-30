# Pika H3 Flex：6步联合音视频

Pika官方模型：[Pika-Labs/Pika-H3-Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex)。
[原项目代码](https://github.com/Pika-Labs/Pika-H3-Flex)；
[Comfy官方模板](https://docs.comfy.org/tutorials/video/pika/pika-h3-flex)。

本次新增 `H3 Pika Flex · 6步采样准备 (EXP/T8)`，节点ID
`MiniMaxH3PikaFlexSetupEXPT8`。已有778个节点的ID和顺序不变，末尾追加此节点。
它提供采样配方，LoRA、提示词、首尾帧、参考素材仍用现有节点。

## 接线

1. UNET Loader选择已有H3底模。FL2VA对应文字、首帧、尾帧、首尾帧；Ref2VA对应多模态参考。
2. 用现有 `MiniMax H3 LoRA Compatibility Loader` 或原生 `LoraLoaderModelOnly`
   加载匹配的Pika LoRA，作者基线强度1.0。已有模型直接选择，不需要重新下载。
3. H3 Audio Conditioning设置本镜提示词、尺寸、时长和素材；模型生成声音选择`native`。
4. 把MODEL和条件节点的AV LATENT接入新的Pika准备节点。
   输出MODEL接BasicGuider，SAMPLER和SIGMAS接SamplerCustomAdvanced；
   条件节点的同一AV LATENT接采样器latent_image。
5. 采样后用原有AV Decode和AV Save保存。保留生成声音，不把外部原声误接成生成声音。

示例基线是独立的6步联合采样，不是原有4+4，也没有自动叠加Turbo或HyperFlow。
用户已有多LoRA、强度、顺序、attention/FFN补丁均保留，不因文件名或未知组合拒绝运行。
真实缺少接口、张量尺寸不匹配等错误正常报告。组合可执行不代表画质或加速均生效。

## 数值配方

作者原始knots为`1000,750,500,250,100,40,0`，视频shift12、音频shift3。
视频sigma仅转换一次，精确值为：

`1, 36/37, 12/13, 4/5, 4/7, 1/3, 0`

输出原生LCM采样器，使用Core的FLOW_AV音视频载体，BasicGuider对应CFG1。
有6次模型计算、5次非末尾重新加噪，末尾返回干净预测。不是等距6步Euler。
音频沿自己的时钟运行；参考素材、首尾条件、音频锁定遮罩仍交由原生Core消费。
上游显式noise_scale保留并记录，不静默重置。

新节点不改旧采样器或默认值。每次准备报告仅说明已选配置，状态仍为
`prepared_sampling_unobserved`；不会凭准备成功就声称已运行或画质通过。

## 已有模型文件

| 文件 | 对应任务 | Comfy-Org原始SHA256 |
| --- | --- | --- |
| `pika_h3_flex_fl2va_v1_bf16.safetensors` | H3 FL2VA | `cc5453aeecd6189468409003fe4cf1b57e52ff3a830a0a154b87decbe07358b7` |
| `pika_h3_flex_ref2va_v1_bf16.safetensors` | H3 Ref2VA | `de3745ceb7d8d5c20898dd8fa3a05f1c0b00089ccd55297b2a8bab95c766b0e0` |

两文件各1,956,172,304字节、416个BF16张量，普通rank128、融合QKV rank384，
scale1已折入权重，无需再猜alpha。文件SHA只用于说明实际测试资产，不作为模型白名单。
Pika原仓库`comfy/loras/`的文件名/文件SHA不同，不能把另一包装的SHA强加给用户文件。

## 验证边界

本次170项独立CPU回归全部通过，实际注册779节点，原正式778节点序列逐项对照未变。
三图完成原生浏览器加载、保存、重开、导出，全部显式参数和连线保留。
另修正四项历史采样门禁测试的旧版本计数/顺序假设；旧基线文件未重录，变异负例仍保留。

| 实际训练GPU路线 | 输出 | 原生前向 | 本机总耗时 |
| --- | --- | --- | --- |
| FL2VA LoRA文字生成 | 832×480，120帧/5秒，生成声音 | 6 | 276.62秒 |
| FL2VA LoRA首尾帧 | 832×480，完整124帧/约5.17秒，生成声音 | 6 | 290.61秒 |
| Ref2VA LoRA单张参考图 | 832×480，120帧/5秒，生成声音 | 6 | 277.44秒 |

硬件是RTX4060Ti16GB；现有INT8 pruned底模、BF16 LoRA、Qwen NVFP4 AWQ、原生VAE，
8组注意力/8块FFN。每条208个LoRA目标实际挂载、未匹配0，实际6次时间戳与作者sigma相符，
保存后视频和音频完整解码。上述耗时是该机器该样例，不作为通用或Pika服务性能承诺。
首尾帧最初的120帧裁尾样片保留为被替代的交付证据，最终示例和审片使用124帧新版。

Agent抽看了文字/参考图的首帧、中段、末帧，以及新版首尾片最后一帧与输入尾图。
选中首尾片的结尾姿势和场景接近输入；参考图片的举手有移出画面上沿的情况。
这些观察不是普适人体、指令遵循或声音内容保证。2026-10-30用户已对三份指定样片的画面
给出通过反馈，并明确授权保存工作流和推送GitHub；导出表中的声音仍为未审核，原状态保留。
三图、媒体及反馈SHA绑定见[工作流索引](../examples/workflows/94-pika-flex/index.json)。

CPU实际tiny Core回归覆盖六次原生前向、LCM重新加噪数学、两路时钟、种子复现、
锁定原声、首尾/参考图条件、已有LoRA强度和用户wrapper实际委托、真实异常传播。
这不替代训练GPU、CUDA内核或真人画质审查。当前测试/媒体结果以本次交付记录为准。

官方公开权重的原生生成范围是480p/768p。Pika托管的4K放大、五路音轨和服务加速栈
不在这两个LoRA里。与现有双采、2K细化、导演台自动路线的进一步组合没有自动宣称通过。
本次不改变导演台默认路线，不修改底模/VAE，不安装环境包，不下载模型，不自动发布。
