# v1.96.0 · VM01 视觉标记空间引导（EXP）

在参考图中明确标出人物和目标，通过现有 H3/Qwen 提供空间语义引导；不需要新增模型、LoRA 或 SOLRICKS 权重。

- 新增四个独立入口：标记图准备、空间提示词、实际 Picture 绑定条件、外置 Prompt Relay 条件。
- 自动矩形与用户手绘图均可用；内置框选编辑器支持读取底图、拖框、应用和源 RGB 校验，不自动运行或保存画布。
- 默认整张标记图进入 VAE/Qwen；另提供显式 clean-VAE／marked-Qwen 实验，不自动擦框。
- EAV／Prompt Relay／Semantic Bridge 继续显式外接，旧688个节点的完整接口、顺序和默认保留；本版共692个节点。
- 保存[三张已审4+4生成工作流和独立框选编辑器](../examples/workflows/84-visual-marker/README.md)，旧工作流不覆盖。

三条指定5秒原生音画已通过真实画布另存、重开、运行及用户人工审核。
整框和分流样片仍观察到色框残留；用户接受这些原样片不等于去框成功、硬坐标锁、多窗动作隔离或新素材成功率保证。
模板使用合法本机素材占位入口，换图后需重选框、改人物/动作并自行验收；不打包私人参考图、视频、模型和审核文件。
本次发布未新增GPU采样或重跑已完成的功能测试；安装包另核对真实注册和全部旧接口。

详见[接线与限制](VISUAL_MARKER_EXP.md)。方法参考 [H3-Visual-Marker-Control](https://github.com/RK-BoilingPoint/H3-Visual-Marker-Control)，固定 commit `0f0274f5f07c0e8dda0bb8cd0286256587f4e959`；本机模板及提示词独立编写。
GitHub正式版本与Comfy Registry上传、审核和可安装状态分别核对，不等同。

## English

Four additive experimental marker nodes plus a source-bound box editor, with explicit native image binding and external Relay support.
Three exact native 5-second clips were accepted by the user; marker residue remains a documented limitation, not a removal or hard-control guarantee.
All 688 legacy schemas/defaults and existing workflows are retained. The four public templates contain placeholders, not private media or model weights.
