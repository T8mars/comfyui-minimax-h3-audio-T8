# 原生双采分离式 EXP：S01 base-flow／S02 LBH／S03 完整首采

这里的二十份前端 JSON 都是**新增、可编辑的实验图**，不替换任何旧一体工作流。
S01 使用原 base-flow 4+4 表；S02 分别使用发布的 LBH 4+3、4+4、4+5 表。
S03 使用完整首采 8 或 20 步，再接独立 HIGH3／4／5；它不是 LOW4 的延长：
原完成音频在 HIGH 走 `first_pass` 锁定，并保留解码前音频审计。
两阶段各有独立 MODEL／LoRA／条件／噪声、外置 Prompt Relay Plan 与 Stage EAV；
LOW `denoised_output` 经原 learned 3D latent upscaler 和 Reconcile 交给 HIGH。
这不是把一体节点改名，也没有隐藏的自动循环。

每种配方有一对图：`Full_Stages` 显式运行并保存 LOW 与 HIGH；
`Cold_HIGH` 只从已保存的 LOW 运行 HIGH，图中不保留 LOW 模型、条件或采样器。
先在完整图确认并记录 LOW 的 `artifact_path`、`artifact_sha256`，再填入相应冷图。
冷图**不会**自动证明后来改动的 LOW 模型、LoRA、提示词、条件或噪声仍与冻结结果相同；
更改这些输入应重新采 LOW。错路径、错 SHA 或错阶段会拒绝恢复。

导入后先核实本机 H3／Qwen／双 VAE／learned3D、两路 LoRA 与画布设置，
再编辑两套 Relay Plan。EAV 默认 `report_only`，只能观察，不能据此称画质增强。
这些示例已使用通用骑行提示词，未携带私有样片或首帧。

固定 FL2VA INT8／对应 EMA B 配置、128×64×22 小画布的 S01、S02 与 S03 六种完整首采配方，
此前分别完成过真实 GPU LOW→HIGH 与新 Core 只恢复 HIGH 的机械精确对照。
那不是这二十份可编辑图在默认画布／任意资产下的实机资格，更不是 EAV `apply_exp`、
其它后端、长片接缝或人工画音验收。二十图已在隔离 CPU Core 的真实前端逐一打开、
另存、刷新重开并通过节点／具名边／控件值语义核对；这不代表已排队实机生成。
更广旧图回归仍需补齐。
