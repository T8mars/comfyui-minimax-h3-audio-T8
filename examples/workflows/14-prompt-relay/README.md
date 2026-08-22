# Prompt Relay / 分段提示词时间线

本目录用于在同一段 MiniMax H3 视频里保留一份全局人物/场景约束，并让多条局部事件按时间段发挥作用。当前实现是 H3 packed joint-AV self-attention 的 Advanced/EXP 适配，不是把 Wan/LTX 节点原样搬过来。

## 当前工作流

- `2026-08-22_H3_Prompt_Rewriter_8B_Advanced_EXP.json`：独立的 LightX2V MiniMax H3 8B 提示词重写器，输出完整提示词、画面、环境声、非叙事音乐和卸载报告；默认生成后卸载，不把8B模型长期缓存到工作流中。

- `2026-08-20_H3_Prompt_Relay_Plan_Preview_Advanced_EXP.json`：不加载 UNET、CLIP、VAE，也不采样；三条 Event 构建 Plan 后，先计算目标AV/参考素材的 packed 行数和Relay显式chunk bias估算，再在历史记录和节点 UI 中显示逐段帧号、秒数、提示词及 READY 状态。适合先检查时间线和显式bias规模，再打开正式生成工作流。
- `2026-08-20_H3_Prompt_Relay_T2VA_Stock20_Advanced_EXP.json`：T2VA、原生音频、Stock20、BasicGuider（CFG=false）、论文 `paper_v1` 默认公式；三条局部动作已改为可复制的链式 Event，画布内有三份 NOTE。
- `2026-08-20_H3_Prompt_Relay_T2VA_Turbo8_Advanced_EXP.json`：T2VA、原生音频、完整 non-pruned FL2VA、修正后的 Alpha8 Turbo LoRA 与双时钟 8 步；唯一允许的模型顺序为 `UNET → Prompt Relay → Alpha8 LoRA → DualClock`。
- `2026-08-20_H3_Prompt_Relay_Joint_AV_Turbo8_Advanced_EXP.json`：在 Plan 与 Conditioning 之间显式插入 Query Route，选择 `joint_av_exp`，让目标音频与目标视频共同按事件时间路由；论文未验证该音频扩展，因此单独隔离为 EXP。
- `2026-08-20_H3_Prompt_Relay_I2VA_Stock20_Advanced_EXP.json`：首帧 I2VA；`<Picture 1>` 对应首帧，关键帧条件始终全局可见。
- `2026-08-20_H3_Prompt_Relay_Ref2VA_Stock20_Advanced_EXP.json`：单图 Ref2VA；默认加载匹配的 Ref2VA checkpoint，参考块始终全局可见。
- `2026-08-20_H3_Prompt_Relay_RefVideoAudio_Stock20_Advanced_EXP.json`：2秒参考视频与同编号原音轨；视频和音轨必须来自同一素材并分别接到`ref_video_0/ref_video_audio_0`。
- `2026-08-20_H3_Prompt_Relay_RefAudio_Stock20_Advanced_EXP.json`：独立参考音频；它是氛围、节奏与声场的语义参考，不是`lock_source`或原波形复制。
- `2026-08-20_H3_Prompt_Relay_FL2VA_Lock_Turbo8_Advanced_EXP.json`：首帧 + 尾帧 + 锁定原音频；保存节点的音频已正确接到 Conditioning `mux_audio`，并采用 Relay 后再加载修正 Alpha8 LoRA 的 Turbo8 顺序。
- `2026-08-20_H3_Prompt_Relay_L2VA_Turbo8_Advanced_EXP.json`：只连接尾帧的 L2VA Turbo8；用于让三段动作自然抵达指定结束构图。
- `2026-08-20_H3_Prompt_Relay_Hybrid_Stock20_Advanced_EXP.json`：首帧和身份/服装参考图分成两个独立 LoadImage，使用 Ref2VA pruned + Stock20，不加载 Turbo LoRA。
- `2026-08-20_H3_Studio_Prompt_Packet_Relay_Stock20_Advanced_EXP.json`：将 Unified Cast、Sound Canvas 和 T8 Video Prompt Compiler 汇总出的 H3 Prompt Packet 直接转成 Relay Plan；三条局部动作已拆成可复制、可串联的 Event 节点，人物/声音约束始终全局可见。首次为 `report_only`，确认后切 `apply_exp`。

## 使用方法

1. 全局人物、环境、风格、镜头连续性和整体声音写到 `global_prompt`。
2. 推荐复制 `Prompt Relay Event`，把上一项输出接到下一项 `previous_events`，最后一项连入 Plan；未连接事件链的旧工作流仍读取每行一个的 `local_prompt`。
3. 显式范围必须按开始时间从早到晚排列；`frames`只接受整数帧且end包含端点；`seconds/percent`的end是边界，percent使用`0..100`；`auto_equal`会忽略Event的start/end并自动均分。
4. 先执行模型免加载工作流中的 `Resource Estimate → Preview`：把分辨率、query chunk和参考素材数量设置成正式工作流值。看到资源估算和时间线 `READY` 后，再让同一 Plan 进入 Conditioning。当前全部11份生成模板已经内联 Preview。
5. 首次保持 Conditioning 的 `report_only` 检查报告；通过后显式切到 `apply_exp`。
6. 首轮只和相同模型、媒体、分辨率、Stock20、seed 的普通 Conditioning 跑一条 A/B。
7. Turbo8 只能把修正后的 Alpha8 bypass LoRA 接在 Prompt Relay 输出之后；反向连接会被主动拒绝，旧 plain 转换文件也不要使用。
8. 可以临时关闭Event的`enabled`。整条链全部关闭时，Plan只保留`global_prompt`并无补丁直通；只有一个局部事件时也不会安装注意力补丁，通常直接把它并入`global_prompt`即可。
8. 需要保留输入原声时选择 `lock_source`，并把 Conditioning 的 `mux_audio` 接到保存视频节点；不要改接 AV Decode 重新解码出的音频。
9. FL2VA 中 `<Picture 1>/<Picture 2>` 分别对应首/尾帧；L2VA 的 `<Picture 1>` 是尾帧；Hybrid 中 `<Picture 1>` 是首帧、`<Picture 2>` 是第一张参考图。
10. Hybrid 模板虽然初始可让两张 LoadImage 使用同一文件，但它们是两个独立槽位；替换成不同图片后应重新检查身份、服装和首帧约束。
11. 使用 Studio 桥接模板时，不要把完整编译提示词再手工复制到另一份 Plan。Packet 是唯一全局事实源；同一Event链可直接连接Packet桥接。连线存在时事件链优先，`events_json`只作兼容回退。时长会按24fps取整并向上对齐到`17n+5`，实际结果以报告为准。

## 当前边界

- 8B 重写器使用 Qwen3-VL-8B-Instruct 基座和 `MiniMax-H3-Prompt-Rewriter-LoRA-8B`，不复用现有 H3 32B CLIP；当前上游只覆盖 T2VA/I2VA/L2VA/FL2VA，不覆盖 Ref2VA。16GB 本机真实加载和生成成功，但 256 token 短测试约482秒且字段发生截断，因此结论是“可运行但不轻量”，不是实时可用保证。默认 `allow_hub_download=false`，模型须预先放入本地目录。

- apply_exp 已按精确 PackedLayout 合同开放 T2VA/I2VA/FL2VA/L2VA/Ref2VA/Hybrid。`native`、`lock_source`、`remix_source`、`reference_only`沿用稳定 Conditioning 的原合同；Relay 只直接路由目标视频，音频不会获得局部时间权重。
- 参考视频+音轨和独立参考音频均已完成一组736×416×124 Stock20同seed baseline/Relay实测：前者约362/397秒，后者约229/257秒，四条成片严格解码通过。长参考视频会增加固定packed rows和真实耗时；不要把参考音频宣传为波形复刻，也不要把单条结果外推为感知改善。
- `lock_source` 的真实 FL2VA 8 步检查已通过，并通过 `mux_audio`保留原始波形；`reference_only`必须同时启用`add_source_as_reference`。`remix_source`与`reference_only`虽通过结构测试，但声音效果仍需逐条试听，不能写成音频不变。
- Prompt Relay 输入侧不接受已有 LoRA/weight patch/bypass injection；当前只验证了输出侧再接修正 Alpha8 LoRA 的 Turbo8 路线。BlockCache、STG、ActivationChunk、SPEED、MultiRate 或其他 diffusion wrapper 仍未开放，未知组合主动拒绝。
- 只直接修改目标视频 query 到局部文本 key 的 logits。关键帧、参考媒体及其 Qwen 视觉前缀不加时间惩罚；音频 logits 不直接加偏置，但 H3 联合 Transformer 仍可能间接改变声音，必须试听。
- MODEL/CONDITIONING 配对通过标准 `model_conds`携带，不再修改共享`extra_conds`；同一进程连续执行不同 Relay 计划后，L2VA 8步已通过缓存隔离检查。
- Preview只证明Plan的哈希、H3帧网格、事件覆盖和时间换算有效；它不会加载模型、不会采样，也不能预测最终画质、声音或动作遵循。
- Resource Estimate只计算H3 packed行数代理与Prompt Relay显式bias；同一报告会并列736×416、1152×640、1920×1088三档，方便先看分辨率增长带来的rows/chunk变化。它不会运行权威Qwen分词，也不包含模型权重、完整attention激活、CLIP/VAE、后端临时区、VBAR或显存碎片，因此不能拿它判断“16GB一定能跑”。`additional_text_rows`用于补偿系统/媒体token，`manual_extra_packed_rows`用于max-size参考图等未显式覆盖的条件。
- 不创建完整 S×S mask；目标视频 query 分块处理。Stock20、原生 Euler20、full FL2VA + Alpha8 Turbo8、FL2VA lock_source 与连续 L2VA 已分别通过单条本机真实生成和严格媒体解码；这仍不足以宣传稳定提质、音频不变或通用16GB安全。
