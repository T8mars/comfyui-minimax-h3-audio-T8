# 全帧观察一次 / 显式无脸 lazy 旁路（新增 EXP）

六张独立配对图：Standard、Anime、Parity 各 FullSave / ColdDelivery。原56目录及旧图不覆盖；同原 crop、采样、VAE、原音轨、质量门和手动确认。

1. 替换源视频，选择实际本地检测权重。先保持 `confirm_no_face=false`、确认SHA为空运行；两个轻量 Preview 显示完整观察及决定。检测失败、缺模型、部分观察为 unknown，不当无脸。
2. 有脸自动选择原精修支路，检测观察复用给原 Planner，不再跑第二次检测。最终质量仍人工确认。
3. 完整成功零框且你确实要跳过时，在决定节点**手动独立粘贴当前 `observation_sha256`**，开启确认；不把 Observer 的 SHA 自动连到确认输入。源/权重/阈值/代码改变必须重新观察和确认。
4. FullSave 无脸旁路不会保存 Stage，也不返回可恢复路径/SHA。ColdDelivery 只有选择有脸路线才加载实际先前保存的 manifest/SHA；不填写凭空或占位路径。

须使用 Core `ComfySwitchNode` 真 lazy。新 Dependent Audit / Stitch 不作为主动输出根，原操作与 schema 数据不变；新按需 StageSave true 调用原保存器。不要把 true 支路的中间预览/旧主动输出节点另接为输出，否则会强制执行该支路。

检测未找到脸不是质量或真人证明；合成/音轨对象、未知阻断和 Stage 完成严格区分。多人 SAM / 窗口 Plan 未宣称适配。六图有当前 Core 接线/输出根校验，不等于全部素材、模型或整片人工验收。见[技术说明](../../../docs/NO_FACE_LAZY_EXP.md)。
