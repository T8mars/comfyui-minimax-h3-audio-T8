# 长视频与断点续跑

这一组把短窗口H3生成组织成可接受、可恢复、可后台继续的长视频任务。

## 工作流定位

- `Long_Video_22F`：最小分段编排示例。
- `Long_Video_Accepted_22F`：加入accepted manifest，只推进已验收片段。
- `Long_Video_Auto_Resume_22F`：从最后一个已接受片段恢复。
- `Long_Video_Background_22F`：后台队列/租约路线。
- `Long_Video_Background_22F_ScenePlusIdentity`：同时携带场景和身份上下文。
- `Prompt_Relay_Long_Video_Turbo8_Advanced`：整条长视频只创建一次全局事件时间线；每段按
  `timeline_start - context_frames`投影到本地渲染窗口，再与既有Long Video上下文补丁隔离组合。

## 当前成果

已实现原子manifest、父哈希、OS租约、accepted不回退、崩溃后续跑和后台任务隔离；本项目稳定使用约30～32秒范围，60秒不是发布硬门。

## 使用方法与注意事项

首次使用从Accepted或Auto Resume开始，先跑短段并人工验收。不要手工改写accepted文件；更换提示词、模型、参考或种子后应开启新任务目录。长视频降低的是整片一次性生成的峰值，不代表单段可以无限提高分辨率或关键帧数量。

Prompt Relay长视频示例必须先运行segment 0并成功保存AV tail，再把Planner的`segment_index`改为1、2……。
不要为每段重建全局Plan，否则事件会重新从第一幕开始。唯一允许的Turbo顺序是
`UNET → Prompt Relay Long Video Conditioning → 修正Alpha8 Bypass LoRA → DualClock`。
默认继续使用论文范围的`video_only_paper`；单组联合AV盲测七项均为平局，没有证据把`joint_av_exp`
设为默认。当前新增路线已完成一条真实segment 0→1链：736×416、Turbo8、22帧上下文，输出
124+102帧，事件顺序没有重启，视频/音频各3轮严格解码通过；整卡峰值约15478/14984MiB。
接缝处没有静音断层，但原始PCM边界跳变仍需最终试听，因此继续标EXP，不宣传普遍画质或16GB安全。
