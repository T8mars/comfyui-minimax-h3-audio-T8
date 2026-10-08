# FreeVideo 独立视频解码 EXP

这是新增入口，不替换旧 Core VAE Decode、不迁移旧配置，不改变原8+2、真4+4或Quality四档的采样。固定FreeVideo v0.2.3 `e7eb66326a038344ba241fa31355c15b258b98cb` 与VDN `30b6b380c2482f3519469350810c2955d8847fd9`。

两个节点分别接 **Quality完整HIGH／SINGLE** 与 **原FreeVideo完整HIGH**。Light LOW仍有待做的HIGH3、真4+4的MID不是终态，不能接这里。输入是保存阶段，不是任意LATENT；可复用实际完整阶段，仅重新解码，0新增采样NFE。

## 准备

需要官方MiniMax H3 `5d9b308a59ab12e67147f191e184baf704185bd1` 原Diffusers视频VAE三分片、config和index，约10.4GB。它不是新DiT，不需重新下载FreeVideo主体，也不能将原Comfy单体VAE改名冒充分片。模型仍受MiniMax H3许可约束。

1. 使用 `tools/prepare_freevideo_decoder_assets.py --output <独立目录>` 显式下载并核官方文件身份。不会在节点运行中下载、安装软件或覆盖原文件；中断partial保留，需明确处理后再尝试。
2. 使用 `tools/prepare_freevideo_decoder_config.py --quality-runtime <已有v0.2.3配置> --assets-proof <新目录/decoder-assets.json> --home <新缓存目录> --output-config <新配置.json>`。只读借用来源与环境，不借用无关DiT权重；完整文件身份与当前适配源码绑定，配置／home必须全新。
3. 新节点填 `decoder_config` 的实际绝对路径。`decode_mode=eager`为缺省；`compile`仅显式选择。两种模式独立编译缓存，首次编译可能慢；失败不偷偷回退成eager。配置源改变需准备新配置，不能热改旧pin。

## 接线与边界

Stage Load的对应完整阶段 → 独立视频解码 → IMAGE。第二输出是**原normalized音频latent的副本**，接正常音频VAE Decode，再使用既有AV裁齐与保存节点。不向子进程传音频、文本、模型、EAV或Relay；不换原音轨、不做TTS／剪重复句。生成阶段可能是124帧，5秒成片需按原图规则裁至120帧；节点不偷偷trim或改变时间原点。

视频输入为float32 `[1,24,T,H,W]`；按官方config mean/std只反归一化一次。上游decoder原FP32权重逐层流式加载，不开启Linear FP16存储缓存；计算保留作者FP16 autocast。像素按作者ImageNet mean/std、FP32 clamp、uint8 round转为NHWC IMAGE。编译会有融合舍入差，不能称逐位相同或保证提速；不把源码有更新当成本机GPU质量已通过。

子进程与编译子树属于本次调用，取消只清理自有进程；成功标记在解码、资源清理及配置/输入/RGB身份后才写入。原Stage不修改，失败留下独立诊断目录。启动资源门只是最低余量，不是任意大画布显存保证。

## 当前验证范围

已完成一份已通过Quality Light HIGH的真实画布SaveAs、重开、Run eager／compile对照，0新增H3采样。两路完整124帧解码，再按既有规则保存为512×288、24fps、120帧、5秒；全帧RGB/PTS和有限非静音音频已核，两路导出PCM完全一致。原Stage视频／音频逐值不变，没有替换音轨。

本机RTX4060Ti的该单次eager解码17.16秒，compile首次23.80秒；这不是稳态基准，不承诺提速。完整RGB约0.3749%通道差1个8-bit级别，非逐位一致。原已接受Core/VHS成片作为独立视觉参考保留；其AAC样本数与新SafeAV导出不同，不宣称旧新PCM一致。

实际首次运行抓到adapter inference_mode与作者异步prefetch缓冲区冲突；仅新adapter改回作者no_grad，并通过实际上下文的CPU跨线程正／反例。不改上游、Core、权重精度、旧采样或音频，也不关闭prefetch／静默fallback。旧失败证据与旧独立配置保留，修复后用新配置和缓存，不原地替换。

两个短名示例在 `examples/workflows/78-freevideo-decode`。填写真实完成Stage及独立decoder_config；公开图留空私人路径、默认eager、不自动运行。可用 `tools/prepare_freevideo_decoder_workflows.py --server <已有本机ComfyUI地址> --output <新目录>` 重新生成，只读实际节点schema，不排队。

最终画质／听感人审仍待。只这一份Quality HIGH代表实际GPU解码通过；旧版入口和其他质量档不借此取得整族资格。旧路径始终可用。

## Windows planner 与解码器不是一回事

当前固定v0.2.3的Quality采样worker已明确调用作者的`policy.choose`；独立decoder节点不负责重新规划DiT attention。复用已完成HIGH的原硬件快照、原request/result和完整Stage，重新执行该pinned纯规划函数，保存policy的所有字段／数值与原结果完全一致，没有新GPU调用或采样。

该唯一512×288／124准备帧代表有5328个视频token、0参考token；原实际配置是56个128维head，每组8个、7组、window_batch=4、8个常驻块和prefetch。完成HIGH3／50块的真实global cuDNN计数1050与分组相符，未启用attention CPU输出或residual offload。Windows动态allocator限额原本已启用。本例不满足“缩组／撤第二传输槽／host暂存”触发条件，不为测这些分支把画布放大或重新采样。

这些只认证该完成Stage的实际几何、策略及执行记录，不证明其他尺寸／GPU／backend容量、WDDM全程无spill或新分支提速。整轮peak与中途变化的budget不在同一时间点，不能直接相减判定超限。旧采样及planner默认值没有改变。
