# S14／S15 HyperFlow fresh-noise 分离式采样（EXP）

本目录有两条不同配方：S14 `full8plus4`（LOW完整8步）与 S15 `partial4plus4`（LOW绝对0:4）。每条配方覆盖无效果，或 EAV／Prompt Relay／两者组合分别作用于 LOW、HIGH、两段的10种配置，再各有完整运行、保存、冷 HIGH、完成 HIGH 读取四种操作，共80张可编辑实验图。旧工作流不迁移。

- `Full_NoSave`：LOW→原 learned3D→HIGH 连续执行，不保存阶段。
- `Full_Save`：保存 LOW 及已完成 HIGH；记录两个 Stage Save 的实际 manifest 路径和 SHA。
- `Cold_HIGH`：从专用 LOW Stage Load 恢复，只进行原 learned3D 和独立 HIGH；不加载或重跑 LOW 模型、条件、噪声、效果与采样。
- `Load_Completed_HIGH`：只读取已完成 HIGH 后解码／保存；无模型、条件、放大或采样。文件名效果标签仅表示冻结结果的来源，不会在读取时重新施加效果。

S14 的 LOW 用完成 AV `output` 作为放大输入；S15 的 LOW 未完成，必须取预测干净的 `denoised_output`，不能把其 `output` 当 clean x0。两条路线都在 learned3D 后让 HIGH 用独立新噪声和原 fresh 音频重基准运行绝对4:8。这与 S13 连续 `x_sigma` 交接完全不同，也不是 P7 的长视频接受链。LOW／HIGH 的 Fresh HyperFlow Loader、内容 LoRA、条件、提示词、NOISE 和外置 Relay/EAV 可分别编辑；不要用普通 LoRA Loader 加载 HyperFlow 权重。EAV 默认 `report_only`，不增强画质。

冷恢复／读取图的路径与64位零 SHA 是占位，不能直接排队。先运行相同路线与配方的 `Full_Save`，将真实路径、SHA 填入对应 Load；改 HIGH 不会追溯修改冻结 LOW 或已经完成的 HIGH。运行前核对实际模型、HyperFlow 权重、视频／音频 VAE、learned3D 权重、画布和显存。

当前资格包括既有 tiny CPU 阶段／专用跨进程恢复、私有候选前端验证，以及本目录正式图的静态 Core 和前端往返；不等于通用图原尺寸真实权重 GPU、多素材／内容 LoRA／注意力后端、EAV `apply_exp` 画质或人工音画验收。
