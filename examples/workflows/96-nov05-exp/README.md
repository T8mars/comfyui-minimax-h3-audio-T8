# 11 月增量实验工作流

这里是v1.105.0的实验接线。**指定画面与已分配音轨已完成复审；各图的完整配方资格分别记录，不因发布统一升级。** 文件中的说明区写明具体已验证范围；不会替换此前已通过的工作流。

| 工作流 | 用途 | 默认行为 |
| --- | --- | --- |
| [TaoMate_HD_2plus1_EXP.json](TaoMate_HD_2plus1_EXP.json) | 新 TaoMate-HD 教师耦合 2 LOW＋1 HIGH；普通提示词，5 秒 AV | 默认 CFG=1 时 3 次模型前向，LOW 比例 0.625；无 learned upscaler |
| [NAG_CFG1_EXP.json](NAG_CFG1_EXP.json) | 在已有 Pika 6 步 CFG=1 路线上比较负提示词指导 | NAG 关闭；开启后用固定 seed 对照 |
| [Stage_Reuse_EXP.json](Stage_Reuse_EXP.json) | FastH3 V2 分离 4+4 的完成阶段复用与解码重试 | `mode=disabled`；手动选 `reuse_completed` 开启复用 |
| [Paired_Noise_Independent_EXP.json](Paired_Noise_Independent_EXP.json) | 6 LOW＋学习型放大＋2 HIGH 的独立噪声基线 | 渐进首采，共 8 次前向，1024×576、5 秒 AV |
| [Paired_Noise_Haar_EXP.json](Paired_Noise_Haar_EXP.json) | 同一渐进配方的 Haar 关联噪声对照 | 与独立噪声相比，画面和声音都可能变化 |
| [Refine_Endpoints_AB_EXP.json](Refine_Endpoints_AB_EXP.json) | 同一 Pika LOW6／学习型放大后，两路 HIGH3 首尾条件开关对照 | 保存采前、关闭、开启三份视频；固定一采音频 |
| [Local_SFX_AB_EXP.json](Local_SFX_AB_EXP.json) | 原片／MMAudio／H3 局部音效三选，5 秒同画面、保护整条原声 | 默认原声；1.2–3.5 秒添加候选；使用已有真实 PCM，不在此图内生成声音 |
| [H3_SFX_Generate_EXP.json](H3_SFX_Generate_EXP.json) | 由自己的原片生成 H3 声音候选，再接 Local SFX 对照 | 40 步、video0/audio1 mask、5 秒/24fps，同源画面保存；不是人声分离 |

TaoMate-HD、独立/Haar、NAG 三组、端点对照、H3 音效生成及实际 MMAudio/H3 混合均已完成所选本机 GPU/媒体验证；指定历史反馈见[11 月 8 日审片记录](../../../docs/NOV05_REVIEW_20261108.md)。Stage Reuse这张FastH3 V2接线完成UI/API检查；真实训练模型的复用证据来自另行验证的原生单采和标准4+4，不冒充该FastH3图的GPU画质认证。现有模型文件直接使用，不重复下载。

操作说明：[TaoMate-HD](../../../docs/NOV05_TAOMATE_HD.md)、[NAG](../../../docs/NOV05_NAG.md)。阶段复用报告中的 `verified_hit` 与 `current_denoiser_evaluations=0` 才表示本次确实跳过采样；未知自定义模型/补丁组合仍可执行，但不会伪报已认证缓存命中。

新增对照说明：[渐进关联噪声](../../../docs/NOV05_PAIRED_NOISE.md)、[二采首尾条件](../../../docs/NOV05_REFINE_ENDPOINTS.md)。Haar 仍使用真实学习型 3D 放大器；首尾条件在采样前生效，不进行事后潜空间桥接。

音效素材与接线见[局部音效说明](../../../docs/NOV05_LOCAL_SFX.md)。所有图均经过独立原生前端的导入、API 导出和保存回读，媒体效果仍需看对应实际成片。

[另存11套已审UI/API配方](../97-nov09-reviewed/README.md)和[最终验收](../../../docs/NOV08_FULL_ACCEPTANCE.md)是当前交付入口；画布中旧候选备注保留其来源，当前复审状态以上述记录为准。
