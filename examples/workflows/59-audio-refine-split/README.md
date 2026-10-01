# S26 Audio Refine 分离式图（EXP）

本目录从 `18-audio-refine` 的十张已实现采样工作流各新增一对图，不替换旧图：`freeze_video` 运行原视频生成链并显式冻结**最后一遍视频**的联合 AV latent；`resume_audio` 在新 Core 中验证路径、完整 manifest 与文件 SHA 后，仅运行 Audio Refine 音频尾采。Learned／PDD 4+4 的两遍视频都属于冻结图，不能把第一遍误当最终精修输入。

先在冻结图审核首遍结果，并把 `confirm_save` 显式设为 `true`。再将 Save 返回的 `checkpoint_path`、`manifest_json`、`file_sha256` 三项原样填入同一配方的恢复图。恢复图默认 Quality Gate 仍返回原片，需要人工听审和显式选择候选音频；这两张图不是自动接受或无需前段文件的“续跑”按钮。不要跨配方混用回执，文件损坏或长度不符必须拒绝。

旧图已有的外置 EAV 或 Prompt Relay 路径被保留；长视频 Relay 的精修 Plan 可以独立编辑。新增 [effects 子目录](effects/README.md) 提供26张尾采独立 EAV／Relay／组合图；这里只新增，不迁移本目录原20图。效果图的 tiny 执行、画布和真实权重质量资格分别记录，**不应解读为 S26 完成**。

原20图与此前私有候选逐结构相同，20/20 当前 Core 静态验证、20/20 **公开图**原生浏览器打开／另存／重开、十条冷尾采 CPU 证据已取得；普通 Prompt Relay 的冻结／冷尾图还各通过可见 Plan 编辑和重开。PDD 两条此前另有小尺寸真实权重 GPU 冻结／冷尾采机械证据。这些原图证据不能自动继承给新增效果配置，也不证明原尺寸、其它八条真实权重、精修音质、画质或跨素材适用性。
