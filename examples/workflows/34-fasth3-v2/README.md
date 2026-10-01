# FastH3 V2

These workflows are optional FastH3 V2 examples. They require the matching
FastH3 V2 ConvRot checkpoint and the native MiniMax H3 V2 setup/runtime nodes.
They are kept as explicit experimental routes; stable H3 workflows and their
registration order are not migrated automatically.

Check the node preflight report before queueing a clip. The examples do not
promise a universal speed, VRAM, duration, or perceptual-quality improvement;
validate the exact model, GPU and input combination you use.

## 分离式 4+4 长视频 EXP（新增，旧六图不替换）

按编号导入 `FastH3_V2_Split_01`～`06`：首段独立 LOW4／learned3D／HIGH4、
已接受父片续段的独立 LOW4／HIGH4、单独 Review／显式 Accept、最后 Compose。
`02`／`04` 是可选的冷 HIGH 图：从已完成 LOW 的 manifest＋配方旁证恢复，
图中没有 LOW UNET 或 LOW 采样器；不是“重跑完整图后跳过显示”。
四张采样图的 Prompt Relay Plan 和 Stage EAV 都在阶段外，互不绑成一只大节点。

1. 把自己的首帧放入 ComfyUI `input/`，将图中的 `SELECT_YOUR_FIRST_FRAME.png`
   改为该文件；安装图中所列 V2、Qwen、视频／音频 VAE 和 learned3D 权重。
   默认是明确的 `dense_compat_exp`＋Relay，EAV 为 disabled。训练 VSA 的稀疏内核
   没有 Relay 逐查询时间偏置，不能只改 profile 就声称两者同时生效。
2. 在 `01` 设置新的唯一 `chain_id`，同步编辑 LOW／HIGH 两份全局 Plan 与模型分支，
   运行后得到 **未接受** 的首段候选及 LOW 阶段保存回执。先完整预览画音。
3. 在 `05` 填入首段 `candidate.json` 路径；默认 `accept_candidate=false` 只读预览。
   真正确认后再显式改 true，并保存返回的 chain、candidate、revision、job SHA。
4. 在 `03` 填入这些父片回执，保持相同首帧、全局 Plan、模型／LoRA、尺寸与当前
   作业配置；续段明确选 `old_fixed_124` 渲染，只交付22帧上下文之后的新增68帧。
   如改了这些配置，来源门应拒绝；请另起新链，不要伪造父回执。再用 `05` 审阅并
   显式接受续段，最后将 chain ID 填入 `06`，只在两段都已接受时拼接192帧。
5. 如需新 Core 只跑 HIGH，在对应完整图保存 LOW 后，改用 `02` 或 `04`；必须把同一
   LOW manifest 路径和 SHA 填到 Stage Load 与 Bundle Load 两处。恢复图的占位符
   不能直接运行，改配置导致来源身份不匹配时必须重新完成 LOW。

这套是可编辑、可导入的实验图，不含测试用人物／歌曲／私人首帧，也不自动接受。
现有固定密集 V2 两段配置已有真实 GPU 旧一体／分离画音逐位机械对照与冷 HIGH
阶段精确回执；这里的通用提示词和用户替换素材**没有**继承该画质资格，也未得到
人工从接缝前至片尾的画音验收。EAV `apply_exp`、其它后端、LoRA、画布和素材须分别
验证。六张 Split 图已在隔离原生前端逐一另存、刷新重开并核对节点／具名边／控件值；
四张采样图的外置 Relay 和末段 EAV 控件可单独编辑并保持，Review 仍默认不接受。
这只是前端保存验收，不是生成或人审。候选侧记只作本地完整性追踪，
不是数字签名；旧一体六图及其缺省行为不改。
