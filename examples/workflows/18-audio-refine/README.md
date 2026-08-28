# MiniMax H3 Audio Refine 音频精修

这组工作流用于给已经完成首遍采样的 MiniMax H3 联合 AV latent 增加一段音频 partial-tail 双时钟采样。它主要面向 Turbo 低步数画面可用、但声音发闷、金属感或细节不足的情况。

## 工作流

- `2026-08-26_H3_Audio_Refine_Turbo4_Plus_Refine4_Advanced_EXP.json`：可直接导入 ComfyUI 的 T2VA 示例。默认 1056×608、124 帧、首遍 Turbo4 + Audio Refine4，共 8 次 H3 联合 AV 前向；同时保存原始候选、精修试听候选和质量门最终选择。
- `2026-08-26_H3_Audio_Refine_Phase2_Same_Turbo4_Advanced_EXP.json`：Phase 2 的显式同 Turbo 栈路线；新增 Model Route 会核对同一基座、patch UUID、补丁结构与 LoRA metadata，不能识别就旁路。
- `2026-08-26_H3_Audio_Refine_Phase2_Base_Refine4_Advanced_EXP.json`：Turbo4 首遍后改用同一无 Turbo 基座做 Refine4；用于确认去掉蒸馏 LoRA 是否真的改善声音，不能预设结果。
- `2026-08-26_H3_Audio_Refine_Phase2_Base_Ordinary8_Control_Advanced_EXP.json`：无 Turbo 基座单次普通 8 NFE 成本控制。它不执行 Audio Refine，也不能被称为训练分布匹配控制。

2026-08-29 新增的兼容工作流不会替换上面任何旧文件：

- `2026-08-29_H3_Audio_Refine_Turbo8_Plus_Refine4_Advanced_EXP.json`：Turbo8 完成后追加 Refine4，总计 12 NFE。
- `2026-08-29_H3_Audio_Refine_Learned_TwoPass_Final8_Advanced_EXP.json`：学习型 Latent 双采，只在 Pass2 最终输出后精修。
- `2026-08-29_H3_Audio_Refine_PDD_Ref2VA8_Advanced_EXP.json`：PDD 8步生成完成后，使用明确连接的基础 H3 模型做 Refine4；不会重复 PDD。
- `2026-08-29_H3_Audio_Refine_PDD_Ref2VA_4Plus4_Advanced_EXP.json`：PDD 4+4 双采完成后再精修，总计 12 NFE；Pass1 音频不提前修改。
- `2026-08-29_H3_Audio_Refine_EAV_Turbo8_Advanced_EXP.json`：EAV 仅用于原始 Turbo8 生成，Audio Refine 侧不重复 EAV。
- `2026-08-29_H3_Audio_Refine_Prompt_Relay_Turbo8_Advanced_EXP.json`：复用同一 Prompt Relay 模型、Conditioning 与时间绑定。
- `2026-08-29_H3_Audio_Refine_Long_Video_Prompt_Relay_Turbo8_Advanced_EXP.json`：长视频 Prompt Relay Turbo8。下一段只接原始 continuation；人工接受后的音频只用于本段交付。

后三个工作流必须分别运行，不要一次全部排队。四臂对照由 Turbo4 原始、普通 8 NFE、Turbo4 + 同栈 Refine4、Turbo4 + base Refine4 组成；总 NFE 相同不代表训练分布相同。

## 使用方法

1. 先替换模型、LoRA、CLIP、双 VAE 与提示词，再保持默认 `refine_steps=4`、`audio_denoise=0.50` 完成一轮。Phase 2 只接受预注册的 `0.35` 或 `0.50`。
2. 先听 `MiniMaxH3/AudioRefine/original` 与 `refined_candidate`，检查台词、音色、演绎、音乐/环境/瞬态、声道、远近声跳变和口型同步。
3. `Audio Refine Quality Gate` 默认 `accept_candidate=false`，最终结果一定回退原始 AV latent。只有人工确认候选后才改成 `true`。
4. 接受候选时，质量门只采用候选音频 latent，并把原始视频 latent 逐值回填，防止音频精修意外改画面。
5. `lock_source`、`final_audio`、受保护外部音轨和未经验证的 `remix_source` 不进入精修；审计会旁路或拒绝。
6. 新兼容路由的 `generation_profile` 必须与工作流一致。模型文件名、文件大小和 SHA 只写入报告，不会因为指纹不同阻止运行。
7. PDD/EAV 不在 Audio Refine 侧重复应用；Prompt Relay 复用现有绑定。长视频必须把 `continuation_av_latent` 接回上下文节点，不能用 `delivery_av_latent` 续写。

## 参数与成果

- 固定首版路线：CFG 1、`dual_clock_euler`、`native_flow`、video/audio shift 12/3、视频 mask 0、音频 mask 1、确定性 seed、同一 MODEL 与 conditioning。
- `audio_denoise=0.35` 是更保守的预注册试听点；`1.0` 属于完全音频重生成，不在 Phase 2 节点中开放。
- 2026-08-26 的 256×256×22 机械烟雾验证中，首遍与精修各完整执行 4 步，两条 MP4 都通过严格解码，均为 32kHz 双声道。实测还证明 ComfyUI 的零视频 mask 不足以保证采样输出视频 latent 逐位不变，因此正式工作流必须经过新增质量门精确回填原视频 latent。
- 同日只追加了一条最终质量合同：1056×608、124帧、24fps、中文对白、Turbo4+Refine4、`audio_denoise=0.50`。用时414.14秒，观测GPU峰值14,468MiB、最低空闲1,642MiB；原始/精修候选/默认回退三条都通过严格解码，默认回退的视频和音频解码哈希与原始结果分别完全一致。该单例不是压力测试，不外推通用16GB安全。
- 同一质量对的匿名审听已由一名评审者完成。初始反馈为“差不多，但是右边声音轻一点”，随后评审者明确修正为“左边的更好一点”；揭盲为A/左=精修、B/右=原始。因此本轮记为精修组轻微胜出、原始组感知响度略低，结论为 `LIMITED_HUMAN_PREFERENCE_AUDIO_REFINE_KEEP_MANUAL_GATE`。该单素材偏好不会自动把精修候选提升为默认。
- Phase 2 的三组最终盲测覆盖清晰对白、对白+雨声+音乐+杯底瞬态，以及 I2VA 参考人物对白。揭盲后 base-without-Turbo Refine4 的音频为 3 胜 0 负；整体结果为候选 1 胜、2 平。该结果支持保留这条可选路线，但 Quality Gate 继续默认回退原始结果，避免把有限样本外推成普遍提升。
- 机械通过和一名评审者的单素材结论都不证明普遍听感更好、等价或非劣；台词、音色、口型和混合音频仍需在用户自己的素材上试听。

## 资源与边界

无缓存精修的每一步仍接近一次完整 H3 Transformer 前向，不是只计算音频。审计要求整卡至少 512MiB 空闲、系统提交余量至少 16GiB；这只是最低拒绝门，不等于所有 16GB 显卡都不会 OOM。Frozen Cache、并发、压力矩阵、跨 GPU、CFG>1、动态 CFG、第三方 sampler 和未知 transformer 补丁不属于当前稳定范围。

Audio Refine 是生成式音频 latent 重采样，不是波形降噪、EQ、锐化或无损修复。它可能改字、增删音节、改变人物声音、表演、音乐、音效、环境声和口型同步；信号审计只能提示风险，不能自动证明候选更好。

当前没有把 EAV + Prompt Relay + 长视频的 Stock20 组合冒充为 Turbo8；只有上面列出的独立 8步路线进入兼容工作流。
