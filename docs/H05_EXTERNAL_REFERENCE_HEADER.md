# H05 社区参考包只读检查

这是独立的文件头诊断工具，**不是参考加载器或转换器**。不改变现有 T8
ReferenceSet、Route／Apply、普通 LoRA 或旧工作流。不要把社区 latent 文件直接
改名为 T8 包，也不要仅依据作者名称、路径或 metadata 认证编码器。

另有新增的[独立 Encode 图像导入 EXP](H05_EXTERNAL_IMAGE_IMPORT_EXP.md)：必须提供
实际原图及当前原生 VAE，重新编码后按明确精度逐字节匹配。它不改变本工具的
header-only 证据边界，也不解除下述视频＋音频样本的 unsupported 状态。

在插件目录使用 ComfyUI 的 Python 运行：

```powershell
python tools/inspect_h3_external_reference.py "G:\models\refmods\example.safetensors"
```

工具只读取 8 字节长度及最多 1 MiB JSON 文件头；最多 8192 个张量描述和
256 个参考成员。检查重复字段、非有限 JSON、实际 shape／dtype／连续偏移与
文件大小、v4／v5 成员顺序和 h3_hybrid v1 的权重计数。输入不被重写，张量
payload 不加载，嵌入配置和路径不执行、不跟随；未知格式明确报错，不隐式迁移。

报告中的 token 数只是文件头声明的原生网格成本，尚未计 runtime copies／Qwen
token，不能当实测显存或推理成功。合法结构也不证明 payload 数值有限、
真实 producer、归一化、来源权限、同步或画质。

当前固定社区样本的 v5 视频＋音频文件为 955416 字节，实际读取1560字节，
声明网格分别4608／540 token；17项标准库测试通过，0张量加载／0新增NFE。
缺原 RGB、原 PCM、实际编码器及明确权限证据，因此保持
`inference_ready=false`、`conversion_performed=false`，音频标为
`sound_incomplete_original_PCM_and_encoder_unverified`。

后续合格转换只从一个具备真实原RGB、writer代码身份、actual producer、
normalization和权限证据的 image 开始，create-only 接已有 Route／Apply。
合法原素材重新编码应标为重建，不能称原 latent 无损导入。RefLoRA 的普通
权重半部保持普通 loader 路线；不同时 attach 与 Apply，不把只写参考成员
误称完整保留了 LoRA。

固定格式依据：[RefMod v5](https://raw.githubusercontent.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/6131c1f62c612a074d6aac8435c26ffa19c00563/BUNDLE_FORMAT.md)、
[RefLoRA v1](https://raw.githubusercontent.com/malcolmamal/ComfyUI-MiniMaxH3RefLoRA/5c914fbb9ca091421abf3a8ff7e59e48787d8571/HYBRID_FORMAT.md)。
社区原件、私有检查回执及素材不随插件发行。
