# B8 / C1 / C2：音频复审对应的分离式入口（EXP）

这里是可复用模板，不是带本机私有检查点的审核实跑副本。2026-10-02 用户通过了三份固定素材的新候选；不等于任意素材或这些通用模板默认画布均已人审。旧 S26 / S29 图完整保留，不覆盖、不迁移。需要 v1.88.1 新增的 `MiniMaxH3RFRestartJointClockSetupEXPT8`；v1.88.0 不含这一后续修正。

| 路线 | 先生成并保存首采 | 已有检查点，只跑后采 |
|---|---|---|
| B8 Long Relay：4步音频精修，强度 0.35 | [Freeze 视频](B8_LongRelay_035_Freeze_Video_EXP.json) | [只跑音频尾采，Relay / EAV 外置](B8_LongRelay_035_Resume_Audio_EAV_EXP.json) |
| C1 RF Standalone：双时钟只初始化一次 | [完整 BASE + RF / Save](C1_RF_JointClock_Full_Save_EXP.json) | [载入 BASE，只跑 RF](C1_RF_JointClock_Resume_EXP.json) |
| C2 RF Detail Mixer：双时钟只初始化一次 | [完整 BASE + Detail + RF / Save](C2_RF_DetailMixer_JointClock_Full_Save_EXP.json) | [载入 BASE，只跑 Detail / RF 尾段](C2_RF_DetailMixer_JointClock_Resume_EXP.json) |

把 JSON 拖进 ComfyUI 画布或通过工作流菜单打开。先选择本机模型/CLIP/VAE，按自己的素材核对帧数、提示词和 Relay 时间布局；未随图打包模型或检查点。两阶段效果、模型和条件仍独立。

- B8：审核视频后手动设置 `confirm_save=true`，复制真实 `checkpoint_path`、`manifest_json`、`file_sha256` 到第二张的 Frozen First Pass Load。帧数保护和 Long Video 段号/上下文/时间布局必须匹配；新任务改 chain_id。第二张 QualityGate 默认不接受，试听后自己决定是否用候选。原音频不覆盖。
- C1/C2：完整图的 `RF/BASE` Stage Save 输出真实 `artifact_path` / `artifact_sha256`，填写到恢复图 Stage Load。冷图没有 BASE 首采链；修改 BASE 资产或内容时重新生成对应 BASE。不要把 RF RESTART 的路径误填成 BASE，也不要混用新旧策略的 context / 重启回执。
- 恢复图中的路径、manifest、零 SHA 是明确占位，故意不能直接 Queue。完整图保留旧通用尺寸/种子/步骤，不冒充审核用 124 帧、448×256 的精确配置。

RF 说明见 [联合时钟修正](../../../docs/RF_JOINT_CLOCK_RESTART_EXP.md)。所有分离式路线见 [总索引](../SEPARATED_WORKFLOWS.md)。
