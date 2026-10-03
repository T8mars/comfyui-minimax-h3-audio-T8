# LMS／Orbit／Wallpaper 独立模型对照（本地 EXP）

六张可导入占位图，每个家族各Baseline／Candidate；只换所选模型权重或disabled→LoRA strength1，使用现有加载器，不增加专用模型节点。先选择自己的素材／已完成curveTAIL实际path与完整SHA。素材与模型不打包、不自动接受质量或Queue。

- LMS：完整3D放大权重，已有 `Learned Latent Upscale` 加载；同源完成AV放大2x，音频保持，不重采样。
- Orbit：pruned FL2VA、同首尾图、768方／73帧／28步／CFG1、无Turbo。T8内部仍是joint AV，此图只保存静音视频；不是作者audio-off推理和逐字prompt等价，完整环绕／闭环待人审。
- Wallpaper：R32／Ref2VA／一张参考，`live_wallpaper:`触发；保持既有Ref2V Turbo4小样配方单变量对照，不等同作者R64＋Taomate3＋LMS多因素图。其它秩、镜头和多参考须独立确认。

对应专题：[RADAR模型兼容资格](../../../docs/RADAR_MODEL_COMPATIBILITY_EXP.md)。静音交付不代表内部audio stream关闭；机械解码／耗时不证明画质、冻结人物、精确循环或通用加速。

候选明确选择 `zz_radar/` 子目录：两份LoRA置于 `models/loras/zz_radar/`，LMS置于 `models/latent_upscale_models/zz_radar/`。基线仍为原权重／disabled，原节点默认不变。当前本机三组实际原生画布完整成片资格均通过；其它机器需选择真实本地文件，源素材占位和curveTAIL路径／SHA仍须手动填写，不直接Queue占位图。
