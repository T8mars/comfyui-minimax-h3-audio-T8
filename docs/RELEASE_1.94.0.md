# v1.94.0：FreeVideo 新四档（独立增量 EXP）

原8+2、真正前4／后4、12个旧FreeVideo入口和原663个节点的完整接口／默认保持，旧配置和缓存不迁移。本版只追加8个Quality v2入口，不包含尚未完成的Qwen参考视图项目。

| 档位 | 实际路线 | 本版验收 |
| --- | --- | --- |
| Light | LOW8 → 用户外置放大 → 独立Community HIGH3 | 指定5秒原生画布音画人审通过 |
| Medium | 联合单采12 | 指定5秒原生画布音画人审通过 |
| High | 联合单采16 | CPU时钟／表／结构支持；未新增GPU人审 |
| Max | 联合单采20 | 同上；明确published FP32时钟修正 |

新Loader用专用`user/default/T8/freevideo-runtime-v2.json`，不回落旧配置。沿用独立FP8引擎，固定FreeVideo v0.2.3源码；旧rowwise主体可按真实SHA复用，新增各档／任务的AdaLN采样表。模型和许可证仍按MiniMax H3适用限制；不包含模型、独立Python、配置或Stage缓存，不运行作者安装器。

LOW／HIGH模型、条件、LoRA、EAV和Prompt Relay均可独立外置。Light HIGH3必须接真实完成的新LOW8和其同音轨放大结果，保留LOW完成音频；旧MID、SINGLE和旧Stage不能冒充。旧4+4仍继续未完成的音频，不冻结半成品。

两条最低量验证各为512×288、120帧、24fps、5秒，正常中文对白，均从真实原生画布运行、完整RGB／PCM／PTS核对并获用户通过。Light采用448×256→learned1.2×实际512×288，非作者默认2×几何parity；两路线算量不同，不作速度对照。新四档时钟、50层表身份、坏SHA、错误家族、取消、效果配对和原接口另做CPU验证；不以CPU证据代替听感或通用画质。

[四张可编辑模板](../examples/workflows/76-freevideo-quality/README.md)来自实际保存的Full／Cold图，仅清理私有配置／缓存路径和空预览。Cold读取自己完成的Full保存出口，需填写manifest路径及SHA；本版没有额外Cold GPU运行。16／20可从Single图选择，但不借Light／Medium人审扩大资格。EAV本次仅report_only；Relay仍为实验性文本种子权重机制，不宣称论文softmax等价。

详见[新四档准备与边界](FREEVIDEO_QUALITY_EXP.md)。GitHub发行状态与Comfy Registry安全审核／实际安装状态分开，不保证Registry立即可安装。
