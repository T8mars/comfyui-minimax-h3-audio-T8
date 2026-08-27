# 系统、显存与通用工具

这一组用于运行前审计以及内存、计算与采样轨迹诊断。

## 工作流

- `Environment_Audit`：报告ComfyUI、Torch、GPU、依赖和能力探针。
- `Activation_Chunk`：实验性MLP激活分块，降低部分中间激活峰值。
- `Qwen_Prefix_Cache`：复用重复参考前缀，减少重复编码。
- `2026-08-28_H3_Qwen_Prefix_and_Block_Cache_Ref2VA_Stock20_Advanced_EXP.json`：把本项目的Qwen参考前缀缓存与另行安装的T8 BlockCache组合到完整Ref2VA Stock20链；CLIP和MODEL缓存域保持独立。
- `Trajectory_Probe`：记录采样轨迹，定位sigma/时钟/坏帧问题。
- `2026-08-22_H3_External_BlockSwap_Stock20_Advanced_EXP.json`：把 T8 的保守 BlockSwap 参数桥接到独立 MiniMax H3 流式运行时。
- `2026-08-22_H3_ClipProj_Compatibility_Audit_Advanced_EXP.json`：只读审计外部ClipProj版本、4B/8B投影维度、Qwen3-VL声明和加载模式。
- `2026-08-23_H3_ClipProj_4B_T2VA_Bridge_Advanced_EXP.json`：独立4B低负载T2VA模板；只接受含视觉塔的`qwen3vl_4b_fp8_scaled`和2560→5120的4B v3.1矩阵，默认256×256×22用于一次机械冒烟，不替换32B默认路线。
- `2026-08-22_H3_ClipProj_8B_T2VA_Bridge_Advanced_EXP.json`：把本机8B Qwen3-VL与v3.1矩阵真实接入稳定T2VA链，审计失败即停止。
- `2026-08-22_H3_ClipProj_8B_I2VA_Bridge_Advanced_EXP.json`：把首帧同时接入VAE关键帧和8B Qwen3-VL视觉塔；审计器必须保持`has_reference_images=true`。
- `2026-08-22_H3_ClipProj_8B_FL2VA_Bridge_Advanced_EXP.json`：把独立首帧和尾帧同时接入VAE关键帧与8B Qwen3-VL视觉塔，提示词分别使用`<Picture 1>`与`<Picture 2>`。
- `2026-08-22_H3_ClipProj_8B_Ref2VA_Bridge_Advanced_EXP.json`：把一张参考图送入Ref2VA的VAE参考块与8B投影视觉塔，使用Stock20避免混入四步LoRA变量。
- `2026-08-22_H3_Sol_Attn_Compatibility_Audit_Advanced_EXP.json`：只读审计外部Sol-Attn版本、CUDA/BF16硬件能力和H3补丁所有权。
- `2026-08-22_H3_Sol_Attn_T2VA_Conservative_Advanced_EXP.json`：Scheduled Sol保守参数完整T2VA模板，带tau图、strict首跑和最终补丁审计。

## 当前成果

环境审计、前缀缓存、激活分块和轨迹报告都有独立回归。普通32整除放大与学习型二阶段放大已移动到相邻的`13-latent-upscale`目录。

## 使用方法与注意事项

排错先运行Environment Audit和Trajectory Probe。Activation Chunk、Prefix Cache、VBAR和Block Cache解决的是不同内存/计算问题，不应混称。

Qwen+BlockCache组合模板需要单独安装`comfyui-minimax-h3-blockcache-T8`。Qwen缓存默认只保留1条、256MiB；BlockCache从`threshold=0.08`、CPU、最多连续2次命中开始。两者都会改变执行路径，历史输出也不是bit-exact；它是性能优先EXP，不是无损、省显存或16GB安全保证。重要成片请用同seed关闭缓存做一次对照。

外部 BlockSwap 工作流需要单独安装 `xiaolibai-sys/ComfyUI-MiniMaxH3`；本机核对 revision 为 `099aa38c122cea030ce45a51eb1d83208b16a363`。该上游当前未声明源码许可证，因此本项目不复制或再分发其源码，只输出兼容的 `MINIMAX_H3_SWAP` 参数对象。桥只能连接外部 `MiniMaxH3KSampler`，不能连接 ComfyUI 官方 `MODEL`。示例从736×416、5秒、Stock20、SDPA和上游自动显存策略开始；未做16GB压力认证，不承诺不会OOM。

ClipProj与Sol-Attn两个审计节点都是pass-through：连接的CLIP/MODEL原样输出，未知版本、重复安装、维度不匹配、补丁不完整或未审核wrapper会fail closed。它们不会下载、导入或运行外部实现，也不证明画质、速度、显存或16GB安全。ClipProj当前要求单独安装0.1.13+；Sol-Attn当前要求0.6.2+，并需要明确填写预期dense block数后再组合其他模型补丁。

4B资产现已按固定revision下载并验证：`qwen3vl_4b_fp8_scaled.safetensors`为5,242,467,968字节，SHA-256 `54BD5144DF0BBC25DD6CCADFCB826B521445A1B06AE5A42570BDD2974CA87094`，头部含Qwen3-VL视觉塔且merger输出2560维；`mmh3-4b-ClipProj-v3.1.safetensors`为26,256,128字节，SHA-256 `0184E5C8D666A131962506D21949C2D8A8C6F33445B7B5E347E9A7E0A5BAA819`，矩阵为2560→5120。原有8.04GB `qwen_3_4b.safetensors`头部没有视觉塔，是纯文本Qwen3，明确不可用于该桥。短T2VA机械链与1088×544×124近景I2VA均已真实运行；后者同图、提示、seed和8 NFE的4B/8B、4B/原生32B单人盲评均为全维度平局且无硬失败。32B本次最低余量643MiB，仅关闭这一固定运行的512MiB门。单一简单素材不能证明声音/画面普遍等价、重复稳定、其他模态或普遍16GB安全，原生32B继续默认。

本机低负载验收已固定外部ClipProj commit `c01ba8f`、Sol-Attn commit `930a4d6`和HF矩阵revision `2ebdbcd`。8B checkpoint被上游识别为Qwen3-VL 8B，矩阵SHA-256为`DF0661849D0FD51DB66B0C9AA76F2C1C3EABD81B9A4745EDD2A4617AB24C87F7`且结构为4096→5120。固定seed的256×256×22、4-NFE T2VA短探针已分别完成8B ClipProj与原生32B生成并通过音频/视频/容器严格解码；8B冷启动峰值约比32B低1201MiB，但该短样本不构成质量、速度或普遍显存结论。Sol在RTX 4060 Ti上通过SM89/BF16能力和50块补丁所有权审计；短探针先确认547-token kernel真实执行。随后1152×640×22、同seed生产门A/B保持`min_tokens=4096`并记录`Sol active (5139 tokens)`；Sol/dense耗时49.281/41.828秒、峰值16,004.9/16,008.7MiB，两路均不足512MiB余量。因先Sol后dense，时间受缓存顺序干扰；单次结果没有证明加速或省显存。两路严格解码均3/3，视频SSIM约0.558、PCM相关约0.719，只证明输出变化，画质和音频仍需盲评。`dense_percent=0.2`在4步模板中会令四次调用全部走dense，因此4步默认保持0。

I2VA另有一条独立低负载实跑：256×256×22、4 NFE、shift 12/3、seed 2608228201，8B/原生32B耗时36.125/33.718秒、峰值15,207.4/15,591.9MiB，两条均严格解码3/3。视频SSIM约0.9293、PCM相关约0.8984，但两路都出现非要求的伪中文字幕样式，极短对白尚未人工试听或ASR核验；不能据此称质量或台词等价。

FL2VA双关键帧链与原生32B同seed控制也已完成：8B/32B耗时23.953/31.078秒、峰值15,358.0/15,670.6MiB，两条均严格解码3/3。视频SSIM约0.7090，首尾锚点接近；PCM相关仅约0.5608且响度不同，必须试听，不能称音频非劣。长时插值、0.7MP和通用16GB安全仍未验证。

Ref2VA使用pruned INT8参考模型、Stock20、256×256×22和seed 2608228401完成8B/原生32B同输入机械对照。两条成片均严格解码3/3；8B/32B耗时22.907/27.812秒，峰值16,318.2/16,018.2MiB。三帧SFace均值0.319/0.302且两路都未稳定超过0.36建议线；8B只余约61.8MiB并失败512MiB安全门。本例不能支持画质等价、省显存、普遍提速或通用16GB结论。
