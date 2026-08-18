# ComfyUI 可导入工作流

本目录只保留 ComfyUI 前端画布能够直接导入的工作流：

- 所有工作流统一放在 `examples/workflows/`。
- 文件均为 ComfyUI workflow JSON（画布格式），可拖入 ComfyUI 或通过“加载”打开。
- API prompt、测试规格和程序内部夹具不属于用户工作流，已移出 `examples/`。
- 工作流中的模型名和素材名可能是示例值，排队前请检查加载器、输入图像、视频和音频。

## 主要分类

- 基础音画：`H3_Turbo_Stable_4V4A.json`、三种音频输入模式和 4V8A/4V10A 实验采样。
- 图像与参考：22 帧单图编辑、多关键帧、Ref2VA 参考强度和源视频重绘。
- 长视频：手工续写、候选接受、自动恢复、后台生成和场景加身份参考。
- 人脸修复：单人真人、动漫、作者参数对齐，以及 SAM3.1 双人/三人高级工作流。
- 语音：描述音色、参考音色、双人/联合对白、长文本、ADR、音色库和显存预检。
- Hybrid：兼容审计、模型实验、音频/混合参考、VBAR 余量和 artifact 维护。
- 创作与诊断：环境审计、激活分块、Qwen 前缀缓存、时间线、选择性修复、成片交付、音频注入和轨迹探针。
- 高速动态与细节：红色汉服固定输入的尾段3步、平滑模型时间偏置、联合AV RF Restart、H3 STG、
  时序保护后期细节五个独立Advanced工作流；每份都带NOTE，禁止把五种机制叠加后冒充单变量结论。
- 工具：`H3_Latent_Upscale_By32.json` 按 32 倍数约束 latent 放大后的目标宽高。

## 多人脸修复默认行为

`H3_SAM31_2Person_Face_Refine_Advanced_EXP.json` 和
`H3_SAM31_3Person_Face_Refine_Advanced_EXP.json` 的 Composite 节点默认
`accept_candidate=true`，会采用已生成的修复候选。若只想检查 Preview 而不回贴候选，才将其关闭。

多人工作流仍按人物顺序串行处理；每次换镜、遮挡或重新入镜后都要检查分色预览和人物绑定。
它主要修复清晰视频中的五官结构崩坏，不承担对失焦、低码率或本身模糊视频的锐化/超分。
