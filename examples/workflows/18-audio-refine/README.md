# MiniMax H3 Audio Refine 音频精修

这组工作流用于给已经完成首遍采样的 MiniMax H3 联合 AV latent 增加一段音频 partial-tail 双时钟采样。它主要面向 Turbo 低步数画面可用、但声音发闷、金属感或细节不足的情况。

## 工作流

- `2026-08-26_H3_Audio_Refine_Turbo4_Plus_Refine4_Advanced_EXP.json`：可直接导入 ComfyUI 的 T2VA 示例。默认 1056×608、124 帧、首遍 Turbo4 + Audio Refine4，共 8 次 H3 联合 AV 前向；同时保存原始候选、精修试听候选和质量门最终选择。

## 使用方法

1. 先替换模型、LoRA、CLIP、双 VAE 与提示词，再保持默认 `refine_steps=4`、`audio_denoise=0.50` 完成一轮。
2. 先听 `MiniMaxH3/AudioRefine/original` 与 `refined_candidate`，检查台词、音色、演绎、音乐/环境/瞬态、声道、远近声跳变和口型同步。
3. `Audio Refine Quality Gate` 默认 `accept_candidate=false`，最终结果一定回退原始 AV latent。只有人工确认候选后才改成 `true`。
4. 接受候选时，质量门只采用候选音频 latent，并把原始视频 latent 逐值回填，防止音频精修意外改画面。
5. `lock_source`、`final_audio`、受保护外部音轨和未经验证的 `remix_source` 不进入精修；审计会旁路或拒绝。

## 参数与成果

- 固定首版路线：CFG 1、`dual_clock_euler`、`native_flow`、video/audio shift 12/3、视频 mask 0、音频 mask 1、确定性 seed、同一 MODEL 与 conditioning。
- `audio_denoise=0.35` 可作为更保守的后续试听点；`1.0` 接近完全重生成，台词、音色和同步风险更高。
- 2026-08-26 的 256×256×22 机械烟雾验证中，首遍与精修各完整执行 4 步，两条 MP4 都通过严格解码，均为 32kHz 双声道。实测还证明 ComfyUI 的零视频 mask 不足以保证采样输出视频 latent 逐位不变，因此正式工作流必须经过新增质量门精确回填原视频 latent。
- 同日只追加了一条最终质量合同：1056×608、124帧、24fps、中文对白、Turbo4+Refine4、`audio_denoise=0.50`。用时414.14秒，观测GPU峰值14,468MiB、最低空闲1,642MiB；原始/精修候选/默认回退三条都通过严格解码，默认回退的视频和音频解码哈希与原始结果分别完全一致。该单例不是压力测试，不外推通用16GB安全。
- 同一质量对的匿名审听已由一名评审者完成：“差不多，但是右边声音轻一点”。揭盲为A=精修、B=原始，因此只观察到原始组感知响度略低，没有明确音质胜负。结论记为 `ABSTAIN_NO_CLEAR_IMPROVEMENT_KEEP_ORIGINAL_DEFAULT`，不将精修候选提升为默认。
- 机械通过和一名评审者的单素材结论都不证明普遍听感更好、等价或非劣；台词、音色、口型和混合音频仍需在用户自己的素材上试听。

## 资源与边界

无缓存精修的每一步仍接近一次完整 H3 Transformer 前向，不是只计算音频。审计要求整卡至少 512MiB 空闲、系统提交余量至少 16GiB；这只是最低拒绝门，不等于所有 16GB 显卡都不会 OOM。Frozen Cache、并发、压力矩阵、跨 GPU、CFG>1、动态 CFG、第三方 sampler 和未知 transformer 补丁不属于当前稳定范围。

Audio Refine 是生成式音频 latent 重采样，不是波形降噪、EQ、锐化或无损修复。它可能改字、增删音节、改变人物声音、表演、音乐、音效、环境声和口型同步；信号审计只能提示风险，不能自动证明候选更好。
