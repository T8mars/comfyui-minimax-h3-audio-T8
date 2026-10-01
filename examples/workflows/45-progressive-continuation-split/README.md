# S11 已接受父片续段分离式二采（EXP）

本目录的 80 张实验工作流覆盖接受父片的 22／39 帧上下文 × 10 种已实现效果配置（无效果，或 EAV／Prompt Relay／组合分别作用于 LOW、HIGH、两段）× 四种操作。它们提供独立 LOW→原 learned3D→独立 HIGH；旧一体长视频工作流保留，不需迁移。

- `Full_NoSave`：运行两阶段，不持久保存。
- `Full_Save`：运行并保存 LOW 边界与已完成 HIGH，记下各自真实路径和 SHA。
- `Cold_HIGH`：从冻结 LOW 路径和 SHA 只运行 HIGH，图中不含 LOW 模型、条件、效果或采样。
- `Load_Completed_HIGH`：验证直接父片并读取已完成 HIGH，交付单段续片；不加载采样模型、重新编码上下文或重新采样。

Source 必须填入现有已接受链的 `chain_id`、直接前段 `parent_candidate_id`、`parent_revision`、`previous_job_sha256`，并保持画布一致。节点重新验证父片实际 MP4/context 字节。这里的 `previous_job_sha256` 只认证所选父片，不认证今天新编辑的二采配方。图中的父片 ID、路径及 SHA 是占位，不能直接排队。

LOW 使用父片 MP4 末 39 帧 RGB24，经原 resize 与当前视频 VAE 形成 motion guides；HIGH 保留完成 AV 上下文及同一音频来源。原生 8 步分为 LOW4／HIGH4，保留 22／39 上下文时钟；这不是 FastH3 V2。两阶段模型／LoRA、条件、提示词、噪声独立。Relay Plan 和 EAV Config 外置并按阶段生效，EAV 默认 `report_only`，不会增强画质。请按实际已接受父片分别修改两阶段提示词；只改变 HIGH 不会证明已冻结 LOW 与新 LOW 配方相同。

图默认使用原生生成音频。若主动配置 `final_audio`，需把最终视频的音频输入改接 Delivery 的 `mux_audio`。Delivery 只交付当前续片，不会自动创建候选、接受或拼接；仍需使用既有 Candidate／Accept／Compose 流程。

现有资格涵盖 tiny CPU 阶段数值／真实父片来源身份／冷恢复，以及此通用图的静态 Core 验证和真实前端另存重开；不等于默认原尺寸、真实权重多素材 GPU、长片接缝、EAV `apply_exp` 画质或人工音画验收。
