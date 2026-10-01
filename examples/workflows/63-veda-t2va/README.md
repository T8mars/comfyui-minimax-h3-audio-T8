# Veda 官方预测器 · 8 步 T2VA（本地 EXP）

两张新增图是已完成真实 Chromium 画布 Queue、原8步及完整5秒 H264/AAC 的保存图副本，只改显示标题；旧图不替换。`report_only` 使用原 Dense；`apply_exp` 显式启用 PyTorch Flex 稀疏，**不是官方 FA4**。首次／稳定提速和画音质量尚未通过，不根据机械运行自动接受。

- [Dense 对照](H3_Veda_8NFE_9a1fd3a_report_only_EXP.json)
- [显式 Sparse](H3_Veda_8NFE_9a1fd3a_apply_exp_EXP.json)

预测器文件放 `ComfyUI/models/veda_scorers/minimax_h3_t2va_veda_8nfe_600step_preview_fp8_9a1fd3a.safetensors`，在独立 Bundle 节点选择；不是 UNET 或 LoRA。配方沿用非裁剪 H3 INT8 ConvRot、Turbo v4 step600 EMA B 转换 LoRA／强度1、原8步 `dual_clock_euler/native_flow`、shift12/3。网格768×768×124，显式 Trim 输出120帧／24fps／5秒，另存未经裁切的原生 joint AV checkpoint。文件名600是训练步，不是采样步；旧四步配方不改。

Windows Sparse 需在启动前设置 `PYTHONUTF8=1` 并使用 `--disable-dynamic-vram`，以及实际匹配的 Torch／Triton 环境。不要为了载入此图擅自重启或更换正在使用的服务。Loader核验的是实际Bundle SHA；另一个版本不能继承本图的资格。所有12个精确网格仍分项验收，不静默改尺寸／时长。独立 EAV／Relay 组合与两种接线顺序的覆盖边界、模型下载来源、许可证和最新机械结果见[专题说明](../../../docs/VEDA_SPARSE_T2VA_EXP.md)。
