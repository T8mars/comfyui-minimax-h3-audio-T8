# H3 权重诊断边界

通用 LoRA 加载器继续委托 Core 的适配器解析，不因陌生名字、哈希、用户组合而禁止运行。
新增报告把不同层次分开：

- `input_tensor_count`、`format_hint`：读到的原始键与格式线索，不是支持证明。
- `structural_conversion`：实际布局转换；QKV 合并使原始/转换后数量不能一对一比较。
- `mapping_diagnostics.converted_unmapped_keys`：Core 解析器没有报告消费的转换后键。
- `applied_patch_count`：ModelPatcher 注册数量，不等于 GPU 真正执行，更不代表图像/声音质量。
- `payload_diagnostics.special_runtime_requirements`：HyperFlow 双时间与固定网格、PDD 动态音画头、VSA gate attention 等额外运行条件。只加载普通骨干权重不能证明这些算法完整运行。
- 强度零也会注册 patch，但报告明确没有有效权重变化证明。

完整报告随节点 JSON 输出，并附在 MODEL 的 `t8_h3_lora_compat_report`，供后续诊断读取。
本诊断功能本身不执行 HyperFlow；另有独立的 HyperFlow 专用加载/采样实验路线。
仅看见 metadata 仍不等于已通过该路线的真实推理与成片验收。

`inspect_h3_weight_file` 仅读最大 16 MiB safetensors header，不读 tensor、不下载、不全文件哈希。
缺文件、GGUF 或无法解析的头返回 unknown，由实际加载器决定格式支持。header 通过也不证明
数据完整性、量化内核、设备支持或 LoRA 的实际数值效果。

只有 `merged_loras`、`merged_adapters`、`t8_merged_adapters`、`merge_recipe` 明确声明加速合并时，
才报告 `merged_acceleration.declared`。这仍是文件作者声明（`verified=false`），不是可信来源认证。
文件名含 turbo、量化名称或标题，不足以判定已合并。额外 LoRA 的潜在重复叠加只能给可解释风险提示，
不可静默删除用户显式选择，也不能恢复未知组合硬禁令。

ClipProj 的 quantization 字段只描述 CLIP 编码器；不能据此宣布 DiT 支持 NVFP4、RotNVFP4 或任何底模格式。

## 实际实例的 Core／精度合同观察

已有环境兼容审计的 `environment.runtime_contracts` 会分别列出已加载 Core 模块路径、当前磁盘 SHA、
FinalLayer／prefetch cleanup／视频 VAE／融合 linear 的实际 callable 签名，以及已安装发行包版本。
缺少模块保持未知，不为检查自动导入 Sage、Triton 或其它内核，更不会升级／下载环境。

`precision` 将 DiT 调制／scaling 与 VAE 的 FP16 fast accumulation 分列；启动 fast 参数、FP16
累加 opt-in 和内核符号存在都只是状态，不证明当前 dtype／设备分支已执行。也不按 Turing、其它
设备或版本名单禁止用户运行。磁盘 SHA 不认证内存字节码，CPU 观察不认证 Turing 或 GPU 数值。

该静态快照不猜执行／委托次数。需要采样后的 Spark、Veda 或缓存 Audit 报告确认实际命中；
安装层数、包版本和一次成片返回都不能替代调用证据。审计不会改变底模、LoRA 顺序、dtype 或后端。

## 当前 Core 原生缓存的定向实测

2026-10-22 在 RTX 4060 Ti 上串行完成 OFF、ON、ON＋诊断三组原生20步；各交付120帧、
736×416、24fps联合音视频，完整RGB／PCM／时间戳与严格解码检查通过，画质与听感尚未人审。
OFF实际1000次attention测量；两组ON各16次完整前向＋4次自然命中，共804次测量。
这不是公平性能对比，也不把测量减少直接换算成提速比例。

第三组在原20步完成后，另外执行8次同一真实输入的原生H3诊断前向，不是28步采样。
Error／fresh／Interrupt／fresh四个外层请求均出现自然full→hit；实际异常对象正常传出，
选项恢复且缓存释放。被动观察到非空CUDA预取队列和4次T8-owned HIT清理，清理后资源释放。
这不等同用户点击取消的端到端验收，也不证明原20步输出的独立tensor A/B逐位保持。

真实采样使用Core的 `Guider_Basic`（继承 `CFGGuider`）；附加诊断使用独立 `CFGGuider`。
不能因审计脚本误把二者类名写成同一个而误报生成失败。仅校正私有证据收取，未改采样、
权重、步数、sigma、环境或用户进程。原失败记录保留。

本次范围不包含所有GGUF／FastVAE／外部解码器或任意组合认证；未知组合仍保留用户选择并委托，
真实缺少API、尺寸、文件、缓存完整性错误仍正常报告。旧工作流及默认数学路径不变。
