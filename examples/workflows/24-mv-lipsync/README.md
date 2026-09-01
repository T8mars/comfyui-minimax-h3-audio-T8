# 24 · 全本地 MV / 口型分镜

优先使用 V3 官方 Ref2V Turbo4 工作流。它沿用 V2 本地分镜，把人物参考图、完整原曲和同时间线的隔离人声/清晰对白交给本项目节点：

1. `MV Vocal Lock Scene Planner V2` 要求 `full_song` 与 `vocal_lock_audio` 同起点、同 24fps 时长，并只用本地 CPU 分析人声活动与 5–10 秒分镜。
2. `MV Vocal Lock Prompt Compiler V2` 固定生成官方 `subject_definitions → summary → retention_analysis → detailed_description → overall_soundscape → non_diegetic_music`，绑定 `<Subject 1>`、`<Picture 1>` 和 `<Audio 1>: fully_copy`；没有精确文本时绝不猜歌词或对白。
3. `Local MV Vocal Lock Renderer V2` 只把隔离人声逐段送进 H3 `lock_source`，严格串行调用已连接的本地 MODEL，完成一段就原子保存，可用相同 `chain_id` 续跑。
4. `full_song` 不进入 H3 或分段候选；视频合成后只混入一次完整原曲。

流程不使用远程 H3、ComfyUI HTTP `/prompt` 队列、在线 LLM、TTS 或视频 API。

## 使用

- 使用对应 Ref2VA 基模、H3 CLIP 和两个 VAE；推荐工作流已固定官方 `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`，strength 1.0。不要再把通用 LarryVrh EMA Turbo LoRA 接到这条 Ref2VA 路线。
- 替换参考图、完整原曲与隔离人声。两条音频必须同起点、同时间线；V2 不提供在线分离，先在本地准备好干声或清晰对白。
- 第一次运行使用新的 `chain_id`。中断后保持全部参数不变并再次运行即可续跑。
- 参考图应清晰且脸部可见。当前 V3 已在清晰正脸参考上验证正面与左右 3/4 镜头；主动改成推拉、手持或横移仍会增加时域重影风险。
- 分辨率必须为 32 的倍数；显存不足先降低尺寸，不要并发。
- 人声镜头由 V2 强制为中近景、正面或 3/4 脸，嘴部全程无遮挡。最终仍要按正常速度检查音素时机、身份、手部、背景和切镜；H3 不是确定性音素求解器。

推荐工作流：`2026-09-01_H3_Local_MV_VocalLock_V3_Official_Ref2V_Turbo4_Advanced_EXP.json`

固定采样：1024×768、4 步、Euler/simple、shift 12/3。旧 V2 8 步工作流仅保留为历史兼容，不是当前验收配置。

兼容工作流：`2026-09-01_H3_Local_MV_LipSync_Ref2VA_Turbo4_Advanced_EXP.json`。旧版只用完整混音驱动 H3，保留用于兼容，不能作为独立人声口型验收路线。

## 当前成果与边界

- V3 沿用已通过测试的本地人声分镜、17n+5 预量化、官方六段式结构、双音频合同、串行生成/续跑和最终原曲单次混入，并增加唯一人物/人脸和逐镜头导演合同。
- 一条 5.152 秒清晰英语对白的 736×416×124、8 步真实本地 H3 成片严格解码通过。官方 SyncNet 测得候选偏移 0 帧；把画面固定延后 0.400 秒后测得 10 帧，负对照与预期一致。
- Prompt Compiler 仍额外输出标准 Prompt Relay Event；Renderer 本身按场景提示词生成，不宣称启用了 Attention 级 Prompt Relay。
- 旧 5.152 秒样片的口型已由用户通过，但人物周围发虚。后续同图、同音频、同失败 Seed 对照已推翻“只是参考脸向或换 Seed”的解释：旧 V2/V3 r1–r3 使用了通用 LarryVrh EMA Turbo LoRA 与非官方 8 步/shift 6:3 组合；换成官方 Ref2V Turbo v0.1 的完整 4 步配置后，同 Seed 双脸/拖影消失。
- 新 `r4_official_ref2v` 真实阶段片为 32 秒、5 个独立 H3 镜头、1024×768、768 帧。5/5 镜头严格解码并合成为一条完整主片，完整原曲只混入一次；最终视频/音频/联合严格解码通过，默认解码重复 20 次为 0 异常。
- 5 镜逐镜抽帧未见重复脸、背景人脸、持续光晕或明显人物边缘涂抹。官方 SyncNet 使用各镜隔离人声测得 `0/-1/0/-1/0` 帧（25fps），把第 2 镜画面延后 400ms 的负对照测得 9 帧。
- 用户已完整观看32秒主片并明确反馈“32秒这个已经没问题了，完美”，同时取消约90秒要求。人工结论绑定最终主片SHA-256 `E833277844E6980FDEACF9BDFD5C61FFE48AEFDB3E1EBA6869C363777B7DD75F`；本轮长MV / Lip Sync验收完成。
- Renderer 的 manifest `accepted` 仍只表示机械保存和断点合同成立；对任意新素材，不能把 accepted 计数直接当成人工画质通过。
