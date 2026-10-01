# S28 Prepared LTX 生成／解码分离（实验图）

`Generate_Decode` 明确分开生成与解码节点；`DecodeOnly` 只载入先前生成回执并解码，不重新生成。两图均要求有效的 Prepared LTX bundle 路径和同一 serial lease 路径；冷图还需复制生成回执的精确 SHA-256。默认占位值不能直接运行。它们不是任意 H3 latent 转 LTX 的工具。

旧 Prepared LTX 一体节点与原两张图保持不变。另有六张 `External` EXP 图：EAV、Relay、两者组合各自 full／DecodeOnly，外置 Timeline、独立 Encode、Effects Bind 和采样/解码。所有效果默认 report_only；局部提示词空白为明确 global-only 旁路。编码器从正常 models/text_encoders 下拉选文件，但只认证匹配的原生 INT8 provider，不是任意 CLIP 互换。

基线三阶段及一个室内双人物/白杯的 EAV＋Relay apply_exp 组合已分别完成真实原生 Queue、冷解码和第三缓存读取，73帧832×480完整H264/AAC严格解码/原声同值，冷图不重新生成，缓存不重跑编码/生成/解码。六张新图的真实原生打开、可见独立效果编辑、另存、刷新重开和接线/控件审计通过；独立编码器实际中断、零Job遗留、冷进程仅编码重试及第三缓存读取也通过。这不覆盖任意参数/素材、组合采样/解码取消或人工画质，不应当作全路线完成。详见 docs/PREPARED_GENERATION_INTEGRATION_EXP.md。
