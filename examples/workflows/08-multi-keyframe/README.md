# 多关键帧时间线

这一组在稳定Conditioning之后追加首尾帧之外的中间视觉锚点，不修改原稳定节点的输入、默认值或旧工作流。

## 当前成果

任意中间位置的H3 packed-layout机械路线已通过结构回归，并使用克隆MODEL的局部补丁避免全局修改ComfyUI。空计划会保持原链；参考图、参考视频和音频payload有保留检查。

## 使用方法

在 `2026-08-09_H3_MultiKeyframe_Advanced_EXP.json` 中替换占位图，按frame/seconds/percent设置位置。首尾帧继续交给稳定Conditioning；Advanced计划默认只放中间帧。关键帧越多，packed rows、attention计算和显存都会增加。

## 边界

当前底层统一视觉noise aug不能自然表达“每一帧独立且可标定的参考强度”；没有通过单调性验证前，不要把实验字段解释为精确逐帧权重。与Long Video的组合必须走专门兼容路径，不能叠两层未知补丁。
