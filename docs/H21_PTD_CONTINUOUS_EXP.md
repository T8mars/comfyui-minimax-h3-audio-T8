# PTD 同画布连续 HEAD／TAIL（实验）

这是v1.102.0的显式实验入口，不替换 PTD 完整单采、标准 4+4、HyperFlow 或既有接缝默认。提供进程内 typed 状态，不是可移植冷缓存；不自动重启用户服务。

2026-10-23完整4步／2+2与完整6步／3+3的四份固定来源成片已获人工通过；公开保存图见[PTD目录](../examples/workflows/90-h21-reviewed/06-ptd)。这只更新对应来源的人审状态，不扩张成任意模型、冷恢复、放大或公平性能认证；下文机械验证保留原范围。

截至 2026-10-22，训练模型的完整 4 步与连续 2+2 已实际完成，最终 raw 视频／音频 latent 逐字节相同，两份完整 5 秒 AV 机械检查通过，见开发目录 `r4-fourstep-trained-verification-v1.json`。6 步完整／3+3 另有独立实测，同样得到最终 raw AV 逐字节相同及两片全部 240 帧 RGB／PCM／PTS／strict 通过，实际专家投影为 1260=630+630，见 `h22-ptd6-media-v1/actual-v1/collection.json`。这不是从 4 步结果推断 6 步；两者均未取得人工画质或公平性能资格。

## 接线

先用原 `H3 PTD 专家加载与采样 · Hop Setup` 选择已安全转换的 4 或 6 步 safetensors。它是逐 hop 专家，不是通用加速 LoRA。将选定 MODEL 接各阶段自己的 Guider；用户的 MODEL 分支、已有 wrapper、LoRA 顺序和强度保留。

- `H3 PTD 连续 HEAD`：输入 NOISE、已选择 PTD MODEL 的 GUIDER 和原始空生成 AV LATENT；输出 typed HEAD 和实际报告。4 步运行全局 hop 0、1；6 步运行 0、1、2。
- `H3 PTD 连续 TAIL`：输入该 HEAD 和 TAIL 的 GUIDER；输出最终 AV LATENT、typed completion 和报告。它续接剩余全局专家，不再生成随机噪声或重乘初始 `.999`。
- `H3 PTD 连续阶段真实审计`：接 HEAD 或 completion，验证实际登记与内容再报告。普通 latent、JSON、手造对象或被修改的完成数据不构成执行证据。

最终 AV LATENT 沿原 AV Decode／时长裁剪／Safe AV Save 路径输出。HEAD 本身不是解码成片，也不是 denoised x0；不能用普通 sampler 的 x0 输出来代替。

## 数值与坐标

两次真实 Core guider 生命周期都收到原完整 4／6 步网格，保留全局 `sample_sigmas` 和 callback 索引。HEAD 保存的是 Core 内部 float32 packed noisy AV state，位于 inverse-noise-scaling 与 latent-output 转换之前；TAIL 从该状态恢复，并使用同一配方的 video／audio 物理时钟。中间不换画布、不放大、不冻结未完成音频。

LATENT 来源的普通字典、OrderedDict 或其它 Mapping 提供器会递归快照其全部键值；已生成 HEAD 的模板不随上游随后编辑的画布／张量改变。无法认证内容的辅助对象保留原引用并明确 partial／process-local，不获 portable 身份。相同总元素数也不意味着宽高坐标可以交换。

两阶段可以接用户各自的兼容 MODEL／LoRA 分支；不同选择会保留并警告，不静默换模型，也不借此宣称与同 MODEL 完整单采逐位相同。真实配方、shape、NaN、mask、缺接口或内容损坏仍报具体错误。

## 不包含的能力

2+2 是总共 4 个发布专家，3+3 是总共 6 个；不是 4+4。当前没有低分辨率到高分辨率的噪声／放大适配、HEAD 文件保存／独立进程冷 TAIL，也没有通用模型兼容、性能或画质保证。已完成的 GPU 资格只对应上述明确配方；完整权重转换、CPU 微型逐位对照和训练模型成片验收分别记录，不能相互替代。

保留 [原 PTD 单采说明](H21_PTD_EXP.md) 的完整配方接口。涉及后续 VAE、learned3D、Relay、遮罩、缓存或接缝时，遵守 [双采接缝门禁](DUAL_MODEL_SEAM_FIX_20260913.md)，不改旧已接受样例。
