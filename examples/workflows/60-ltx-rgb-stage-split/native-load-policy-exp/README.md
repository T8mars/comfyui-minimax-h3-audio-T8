# 可选一致流式加载（EXP）

四张新增图对应普通／Identity 的 full-save 和 cold-LTX；旧图未覆盖。

`LTX · Native Load Policy` 明确选择 `consistent_streaming_exp`，可自行切回 `legacy_default`；观察节点只报告准备调用，不认证采样完成或缓存。

完整图选自己的原视频，并明确开启 Source Save。冷图填同一任务保存得到的 manifest 路径和 SHA，保持模型、LoRA、几何、提示词等一致；它只恢复输入，仍重跑原三步。EAV／Relay 在独立节点中，保留原 `report_only` 默认，实际效果需显式开启。

新模式可能改变旧混合驻留 LoRA 的舍入结果，不保证旧输出同值、任何后端同值或普遍提速。一条 Identity＋EAV／Relay apply-exp 已完成真实画布 full／fresh cold 完整画音精确一致验收，不等于四预设全部素材或质量认证。详见 [用法与验收范围](../../../../docs/LTX_NATIVE_LOAD_POLICY_EXP.md)。
