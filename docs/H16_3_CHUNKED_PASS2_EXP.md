# H16-3 分块 PASS2 音频精修（实验节点）

`DeciiaChunkedPass2Sampler` 是正式目录里的可选 H16-3 兼容入口。它复用 T8 的
`chunked_two_pass_upscale_advanced` 执行器，不复制外部仓库的实现，也不打包外部权重。
节点 ID 保持为 `DeciiaChunkedPass2Sampler`，便于导入引用该名称的 ComfyUI 工作流。

## 使用方式

1. 输入原生 MiniMax H3 的嵌套 AV `LATENT`，并把同一组 `Noise`、`Guider`、`Sampler`
   和 `Sigmas` 接入节点。
2. `temporal_strategy` 选择 `full_clip_safe` 可关闭时间分块；选择
   `guarded_overlap_exp` 时使用 `temporal_chunk_frames` 与
   `temporal_overlap_frames` 控制分块及重叠。空间策略固定为 full-frame safe，
   不暗中启用空间切块。
3. `audio_output` 默认是 `preserve_first_pass`：保持一采音频，失败边界最小，适合先验收。
   只有明确需要实验性二采音频时才选择 `refined_exp`。该模式按绝对视频帧坐标放置各块音频，
   仅在重叠区交叉淡化，并对安静尾段使用一采音频保护；合并失败会回退到一采音频而不丢视频。
4. 下游请解码 `output`。`denoised_output` 是为了兼容常见采样器端口保留的同值别名，
   不应作为另一份结果重复解码或保存。

正式示例位于
`examples/workflows/13-latent-upscale/2026-09-20_H3_H16_3_Chunked_PASS2_I2VA_Advanced_EXP.json`。
它保留标准 I2VA 4+4 链的学习型 3D 放大与 HIGH 合同，只原位替换 PASS 2；模板为了验证
H16-3 扩展而显式选择 `refined_exp`，并非改变节点的保守默认值。

## 兼容与边界

- 节点只接受原生 H3 AV 嵌套张量，形状、有限值和正负 conditioning 会在执行前检查。
- 这是实验性兼容层，不声称与外部实现逐值相同，也不声称所有 GPU、长片或音频感知质量已经通过。
  应先运行默认 `preserve_first_pass`，再固定输入、seed 和分块参数单独对照 `refined_exp`。
- Issue #18 提供的上游线索、工作流和许可证记录在 `THIRD_PARTY_NOTICES.md`；外部参考仓库是
  [deciia/ComfyUI_Deciia_All](https://github.com/deciia/ComfyUI_Deciia_All)，其 GPL-3.0-or-later
  代码未被复制进本项目。
