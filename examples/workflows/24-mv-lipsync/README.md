# 24 · 全本地 MV / 口型分镜

优先使用 V2 工作流，它把人物参考图、完整原曲和同时间线的隔离人声/清晰对白交给本项目节点：

1. `MV Vocal Lock Scene Planner V2` 要求 `full_song` 与 `vocal_lock_audio` 同起点、同 24fps 时长，并只用本地 CPU 分析人声活动与 5–10 秒分镜。
2. `MV Vocal Lock Prompt Compiler V2` 固定生成官方 `subject_definitions → summary → retention_analysis → detailed_description → overall_soundscape → non_diegetic_music`，绑定 `<Subject 1>`、`<Picture 1>` 和 `<Audio 1>: fully_copy`；没有精确文本时绝不猜歌词或对白。
3. `Local MV Vocal Lock Renderer V2` 只把隔离人声逐段送进 H3 `lock_source`，严格串行调用已连接的本地 MODEL，完成一段就原子保存，可用相同 `chain_id` 续跑。
4. `full_song` 不进入 H3 或分段候选；视频合成后只混入一次完整原曲。

流程不使用远程 H3、ComfyUI HTTP `/prompt` 队列、在线 LLM、TTS 或视频 API。

## 使用

- 换成正确的 Ref2VA 基模、Turbo LoRA、H3 CLIP 和两个 VAE。
- 替换参考图、完整原曲与隔离人声。两条音频必须同起点、同时间线；V2 不提供在线分离，先在本地准备好干声或清晰对白。
- 第一次运行使用新的 `chain_id`。中断后保持全部参数不变并再次运行即可续跑。
- 参考图应清晰、脸部可见，并尽量与目标正面/3/4脸方向一致；侧脸强行转正会要求模型补出不可见信息，可能产生人物轮廓柔化、光晕或身份漂移。推荐 V2 默认锁定机位；主动改成推拉、手持或横移会增加时域重影风险。
- 分辨率必须为 32 的倍数；显存不足先降低尺寸，不要并发。
- 人声镜头由 V2 强制为中近景、正面或 3/4 脸，嘴部全程无遮挡。最终仍要按正常速度检查音素时机、身份、手部、背景和切镜；H3 不是确定性音素求解器。

推荐工作流：`2026-09-01_H3_Local_MV_VocalLock_V2_Ref2VA_8Step_Advanced_EXP.json`

兼容工作流：`2026-09-01_H3_Local_MV_LipSync_Ref2VA_Turbo4_Advanced_EXP.json`。旧版只用完整混音驱动 H3，保留用于兼容，不能作为独立人声口型验收路线。

## 当前成果与边界

- V2 已通过本地人声分镜、17n+5 预量化、官方六段式结构、双音频合同、串行生成/续跑和最终原曲单次混入测试。
- 一条 5.152 秒清晰英语对白的 736×416×124、8 步真实本地 H3 成片严格解码通过。官方 SyncNet 测得候选偏移 0 帧；把画面固定延后 0.400 秒后测得 10 帧，负对照与预期一致。
- Prompt Compiler 仍额外输出标准 Prompt Relay Event；Renderer 本身按场景提示词生成，不宣称启用了 Attention 级 Prompt Relay。
- 用户已按正常速度观看受测原片与嘴部放大审核版并明确反馈“口型通过”。同一次反馈指出人物周围发虚；这是独立的生成画质备注，不是故意效果，也不否定口型结论。推荐工作流因此改为锁定机位并强化人物边缘稳定提示，但没有用未实跑的改动冒充“已消除虚影”。SyncNet 与本次人审只支持这条清晰对白样片，路线继续保持 Advanced EXP。
