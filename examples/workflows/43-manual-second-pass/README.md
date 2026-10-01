# S09 手动二采分离式工作流（EXP）

这里有 NativeNoise 和 FreeNoise 各一对前端 JSON。`Full_Stages` 显式运行并保存 FIRST／SECOND；`Cold_SECOND` 读取已保存 FIRST 的真实 manifest 路径和 SHA，只运行 SECOND。原有长视频一体节点及工作流没有迁移或替换。

FIRST 是完整原生20步轨迹，交接使用完成阶段的 `output`，不是 `denoised_output`。SECOND 是独立的手工 sigma `0.5,0.412,0.35,0` 三次迭代；没有 learned3D 放大或 FastH3 V2 绝对步窗。两个可见 StageNoise 节点分别对应各阶段：NativeNoise 为 `disabled` 旁路，FreeNoise 为 `variance_preserving_blend`；默认同 seed 和同 segment index。宽、高、帧数原语须与冻结 FIRST 保持一致。

两阶段 MODEL、条件、NOISE、Prompt Relay Plan 与 Stage EAV 可分别编辑。EAV 默认 `report_only`，不增强画质；给 SECOND 单独启用 EAV 是新增的外置能力，不代表旧一体 effects runner 原本就这样执行。

导入后先核对本机底模、Qwen、双 VAE、画布与素材。运行完整图后，从 FIRST 保存输出抄取 `artifact_path` 和 SHA，填到对应噪声模式的冷图；占位值不可直接排队。若改变 FIRST 的模型、提示词、条件、噪声或几何，应重新运行 FIRST，不要把旧冻结结果当成新配方。

既有真实权重测试只覆盖 FL2VA 固定128×64×22的小画布机械执行、媒体解码与全新 Core 仅 SECOND 恢复，不证明本图默认尺寸、其它模型／LoRA／后端、长视频分段、`apply_exp` 画质或人工画音验收。旧工作流继续可用。
