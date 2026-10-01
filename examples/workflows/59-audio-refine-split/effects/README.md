# Audio Refine：尾采独立 EAV / Prompt Relay（EXP）

这里的26张新图不替换父目录的20张冻结／恢复图，也不修改 `18-audio-refine` 的十张旧图。

先用父目录**同配方**的 `freeze_video` 图保存最终视频 AV，再将返回的路径、完整 manifest 和 SHA 原样填入这里的 Load。Learned／PDD 4+4 必须冻结最后一遍视频，不能用低分辨率首采替代。新图仍只有一个外置 Core 音频尾采 Sampler；没有隐藏视频重采。

按文件名后缀选择：

- `_eav_TailEffects.json`：十种配方都有独立 EAV Config／Apply／Audit。
- `_relay_TailEffects.json`：原先没有 Relay 的八种配方增加独立 Plan／Query Route／Conditioning；不需要 EAV 节点。
- `_combined_TailEffects.json`：上述八种配方的 Relay＋EAV 组合，二者分别编辑。

普通 Prompt Relay 与 Long Video Prompt Relay 两种配方，父目录的原 `resume_audio` 图已经有独立尾采 Plan；本目录的 EAV 版本即其组合版，不重复叠加第二套 Relay。

新 Relay 默认保留原全局提示词、空局部事件、`report_only`、`video_only_paper`。先编辑至少两个局部事件，再显式选择 `apply_exp` 才安装时间路由。Query Route 的 `joint_av_exp` 可实验性直接影响目标音频；默认视频路由也可能通过联合网络间接改变音频。不要把 Relay 的新空 AV 输出接到采样器，保留现成的冻结 AV 连线。

EAV 默认 `report_only`，`apply_exp` 是显式实验。它仍是目标视频 attention 增益，不是已验证的音质增强；原增益硬限保留。Quality Gate 默认仍保留原片，必须人工听审再明确选择候选，图不会自动接受新音轨。

当前证据：与交付图仅有 CRLF／LF 换行差异的候选的当前 Core 静态校验、26图原生画布编辑／保存／刷新重开，以及八种新 Relay 配方 tiny Core 真实尾采与跨新进程恢复。CLIP／VAE／权重为明确测试替身，**不是预训练 GPU 成片、全效果媒体交付或画音质量认证**。长视频已有 tiny 窗口调用证据，但新增效果整图真实权重／多素材／人审仍待。S26 和整体分离式项目尚未完成。

接口及限制见 [独立尾采效果说明](../../../../docs/AUDIO_REFINE_EXTERNAL_EFFECTS_EXP.md)。
