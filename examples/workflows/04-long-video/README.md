# 长视频与断点续跑

这一组把短窗口H3生成组织成可接受、可恢复、可后台继续的长视频任务。

## 工作流定位

- `Long_Video_22F`：最小分段编排示例。
- `Long_Video_Accepted_22F`：加入accepted manifest，只推进已验收片段。
- `Long_Video_Auto_Resume_22F`：从最后一个已接受片段恢复。
- `Long_Video_Background_22F`：后台队列/租约路线。
- `Long_Video_Background_22F_ScenePlusIdentity`：同时携带场景和身份上下文。

## 当前成果

已实现原子manifest、父哈希、OS租约、accepted不回退、崩溃后续跑和后台任务隔离；本项目稳定使用约30～32秒范围，60秒不是发布硬门。

## 使用方法与注意事项

首次使用从Accepted或Auto Resume开始，先跑短段并人工验收。不要手工改写accepted文件；更换提示词、模型、参考或种子后应开启新任务目录。长视频降低的是整片一次性生成的峰值，不代表单段可以无限提高分辨率或关键帧数量。
