# S22 Chunked v5 分离式双采（EXP）

本目录提供固定 192 帧的 2／3／4 窗配方，共 198 张可导入研究图。各配方可让无效果、EAV、Prompt Relay 或两者组合只作用 LOW、某个 PASS2 窗或全部阶段；每种都提供完整不保存、完整逐窗保存，以及从任一后窗冷恢复。旧 v5 一体工作流没有迁移或替换。

先看 `2window_none_none_full_no_save`，再看 `2window_combined_all_full_save`。LOW partial4、全片 learned3D 放大、全片 joint AV 噪声准备和各个 remaining4 PASS2 窗是显式节点；LOW/HIGH 模型加载独立，活动窗的 Relay Plan、EAV Config 可分别编辑。各 HIGH 窗共用同一全片准备噪声和原计划；不能将逐窗随机噪声随意更换。v5 最终音频来自 PASS2 的 joint AV 结果，不是 v1–v4 的首采音频透传。EAV 默认 `report_only`，不是默认画质增强。

`full_save` 的原生 partial4 和每个完成窗保存节点都默认 `confirm_save=false`。要使用 `cold_2`，先在对应完整图明确保存 LOW 与 window1，复制两份真实回执；冷图填写 partial4 的相对路径、完整外部 manifest、文件 SHA256，及 window1 的相对 manifest 路径与 SHA256。`cold_1` 用 window0 回执，`cold_3` 用 window2 回执。冷图裁掉已完成窗及 LOW 采样，但会重新建立原计划／全片放大／噪声以验证来源，不是无模型纯解码。文件名中指定在冻结阶段的效果只是来源描述，冷图不会重施，也不会自动证明当前提示词、LoRA 或模型与冻结阶段相同；更改冻结阶段配置后应重新生成和保存。

导入后先替换首／尾帧占位图，核对本机 FL2VA INT8、Turbo LoRA、Qwen、视频及音频 VAE、learned3D 权重；冷图还须填真实回执，否则不能排队。保存回执的路径和 SHA 是身份门，不应复用别的素材或参数的旧文件。旧图仍留在 `13-latent-upscale`。本目录已通过当前 Core 静态连接校验及 198/198 张真实 Chromium 原生另存、重开语义审计；小尺寸替身测试覆盖完整保存、各个冷启动点和损坏 SHA 采前拒绝。它不代表原尺寸真实权重生成、跨资产/后端、EAV `apply_exp` 画质或人工音画验收已经完成。
