# Studio制作、时间线与交付

这一组处理生成前后的制作数据：统一时间线、上下文IR、选择性修复、AV解码安全和最终Reel交付。

## 工作流

- `Studio_Timeline`：构建绝对帧/sample时间线。
- `Context_IR_Provider`：向分段生成提供结构化上下文。
- `Selective_Repair_Execution`：只重做指定片段并保持accepted语义。
- `AV_Decode_Safety`：核对音画latent、时钟和释放策略。
- `Reel_Delivery`：汇总片段、音频、字幕与交付报告。

## 当前成果

这些节点已有typed数据对象、JSON报告、时间边界和异常路径测试，适合把复杂创作从“一张巨型画布”拆成可追踪步骤。

## 使用方法与注意事项

先让Timeline/Context成为事实源，再启动生成或修复；不要在下游节点各自重算帧率和时长。选择性修复必须写入新的overlay并保留原accepted记录。最终交付前检查音频采样率、视频fps、字幕单调性和A/V边界。
