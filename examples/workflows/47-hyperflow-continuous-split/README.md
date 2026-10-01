# S13 连续 HyperFlow 分离式采样（EXP）

本目录提供 1＋7、4＋4、7＋1 三种连续 8 步分界，每种覆盖无效果，或 Enhance A Video／Prompt Relay／两者组合分别作用于 HEAD、TAIL、两段的 10 种配置，再各有四种操作，共 120 张可编辑实验图。它们与 `fresh4`、P7 先低分辨率后放大、普通二次加噪采样是不同路线；旧 HyperFlow 工作流保留，不需要迁移。

- `Full_NoSave`：HEAD 和 TAIL 连续执行，不保存阶段。
- `Full_Save`：保存 HEAD 原始 `x_sigma` 边界及已完成 TAIL；记下两份真实路径和 SHA。
- `Cold_TAIL`：从专用 HEAD Load 读取冻结边界，只执行 TAIL。图中没有 HEAD 模型、条件、初始噪声、效果或采样。
- `Load_Completed_TAIL`：只从专用 TAIL Load 读取已完成 AV 后解码／保存视频；没有模型、条件或采样。文件名中的效果配置是冻结结果的来源标签，不是本图可重新应用的效果设置。

HEAD 在绝对区间 `0:split` 产生原始模型空间 `x_sigma`；TAIL 直接接上 `split:8`，不重加噪、不重新起算音频或时间窗、不使用 learned latent 放大。两路专用 HyperFlow Loader 上游可以分别插内容 LoRA，提示词、条件和外置效果也各自独立。不要把普通 LoRA Loader 当作 HyperFlow 权重加载器；修改 CFG 时需重新提供真正的负条件。EAV 默认 `report_only`，不会增强画质；启用 `apply_exp` 后仍需单独验收效果。

`Cold_TAIL`／`Load_Completed_TAIL` 中的路径与 64 位零 SHA 是占位，不能直接排队。先运行匹配的 `Full_Save`，复制 Save 节点输出的实际文件路径与 SHA；冻结阶段的原底模及 HyperFlow adapter 身份须保持可验证。已完成结果读取不接受后来修改的 Relay/EAV 设置作为对旧输出的追溯修改。

当前资格涵盖先前 tiny CPU 阶段／专用跨进程恢复、私有 81 张效果候选的前端验证，以及本目录正式图的静态 Core 和前端另存重开测试；不等于这些通用图的真实权重原尺寸、多素材／LoRA／注意力后端、EAV `apply_exp` 画质或人工音画验收。
