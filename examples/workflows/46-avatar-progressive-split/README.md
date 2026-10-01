# S12 Avatar 原录音驱动分离式二采（EXP）

本目录的 120 张可导入实验图有两组配方。80 张通用图覆盖 T2VA／I2VA × 10 种已实现效果配置（无效果，或 EAV／Prompt Relay／组合分别作用于 LOW、HIGH、两段）× 四种操作；文件名含 `LegacyAvatar512x768` 的另外 40 张，仅对应[旧 Avatar I2VA 图](../36-avatar-voice/README.md)明确使用的高画布 512×768、LOW 半尺寸、总8步 LOW4＋HIGH4、两阶段 EMA B 和旧 seed 20260918。旧配方图仍使用可编辑通用提示词／素材占位，**不宣称旧图成片逐帧相等**。两组都提供独立 LOW→原 learned3D→HIGH；旧一体图和节点保持原样。

- `Full_NoSave`：运行两个独立阶段，不保存。
- `Full_Save`：运行并保存 LOW 边界与已完成 HIGH，记录实际路径和 SHA。
- `Cold_HIGH`：从已保存 LOW 只运行 HIGH；图中没有 LOW 模型、条件、效果或采样，但仍须选同一录音窗口并核对重新编码的来源。
- `Load_Completed_HIGH`：用相同录音窗口和已完成 HIGH 的路径／SHA 做交付核验，不加载模型、音频 VAE 或重新采样。

先将 `SELECT_YOUR_AUTHORIZED_RECORDING.flac` 换成有权使用的实际录音；I2VA 还须替换 `SELECT_YOUR_FIRST_FRAME.png`。AudioWindow 显式截取录音，AudioLatentControl 只在中性源把音频锁到 AV latent；Plan 必须保持 `initialized_av_exp`、音频 mask 全零。LOW/HIGH 的 MODEL、LoRA、条件、提示词与噪声可以分别编辑。修改共同录音、选窗或源几何会使冻结 LOW 的身份失效，不能复用旧文件冒充当前配方。

AvatarDeliveryAudit 返回选中原 PCM；最终视频只解码完成视频 latent，并明确接这份原录音。不要接生成音频，也不要把此路线称作声音克隆。EAV 默认 `report_only`，只有主动设置 `apply_exp` 才实际施加；各阶段 Relay Plan 和 EAV Config 外置。文件路径、SHA 和素材占位不能直接排队。

旧 Avatar 样片的验收不会自动转移到这些分离图。当前资格包含 tiny CPU 阶段数值、原录音绑定、跨新进程仅 HIGH、Core 静态检查和真实前端保存重开；旧配方的常量来自旧公开图，但尚未做同素材旧新完整真实权重逐帧对照。真实权重通用尺寸／不同人物录音、口型与声音听感、外置效果画质及人工审片仍需分别完成。
