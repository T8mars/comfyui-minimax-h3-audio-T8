# Temporal Chunk · 窗口对白作用域（EXP）

旧工作流保留。本目录是显式选择的新分支，不自动改旧Chunk／H16执行器。

| 文件 | 用途与资格 |
| --- | --- |
| Temporal_V5_Full_12s187_4plus4_EXP.json | LOW4→learned1.2→分窗联合HIGH4；固定12秒／187／34案例已人审通过；原生Saved公开模板 |
| Temporal_V5_Cold_W1_EXP.json | 字面恢复同家族W0保存结果，只跑W1；原生Saved公开模板、CPU恢复已核 |
| Temporal_H16_Full_CPU_Template_EXP.json | 完成LOW8→低sigma HIGH4；H16独立exact-prefix合同；原生Saved、仅CPU接线资格 |
| Temporal_H16_Cold_CPU_Template_EXP.json | H16 whole capsule恢复，只跑W1；原生Saved、仅CPU接线资格 |
| 上述四文件名追加 _External_EAV_Relay_CPU_Template | 每窗独立外置EAV／Relay配置、配对MODEL＋positive；另行构建的CPU接线模板，不是已人审效果样片 |

先选择本机已安装的模型／VAE／LoRA和参考图片。占位图片为`YOUR_REFERENCE_IMAGE.png`。公开Cold的artifact_path／artifact_sha256为空：先在对应Full明确开启confirm_save并运行，复制真实路径和SHA，再填Cold。两家族缓存不能混用；原模型、sampler和sigmas仍须匹配。Cold不用重跑LOW、文本／媒体编码、放大或噪声。

Dialogue Plan分开填写Global、Speech事件与Performance事件。Global保持人物／物品／音色等全局约束；Speech事件填写唯一event_id、speaker、utterance及全局秒区间。不要在Global／Performance再复制完整对白。

外置模板的EAV默认report_only，可单独改为disabled／apply_exp；Relay使用本家族新原生窗口条件，MODEL＋positive必须成对。v5 Cold的Window Save节点设置confirm_save=false，仅校验并解包实际前缀给旧EAV输入，不写新缓存、不重复采样。模板的CLIP只用于可选效果配对，不重编Bank。

一采partial4的音频必须联合续采，不能当已完成音轨冻结。新H16合同不等同旧H16最终crossfade；不能借v5人审批准H16听感。窗口筛选与有限Relay偏置不是数学硬时间隔离，编辑素材、模型或参数后仍须完整听看。

[详细功能说明](../../../docs/TEMPORAL_DIALOGUE_SCOPE_EXP.md) · [1.92.0发行说明](../../../docs/RELEASE_1.92.0.md)
