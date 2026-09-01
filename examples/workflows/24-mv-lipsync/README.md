# 24 · 全本地 MV / 口型分镜

这批工作流把完整歌曲和人物参考图直接交给本项目节点：

1. `MV Vocal Scene Planner` 在本机分析 5–10 秒分镜；可选接入本地人声干声。
2. `MV Ref2VA Prompt Compiler` 生成逐段 `<Picture 1>` / `<Audio 1>` 提示词，不猜歌词。
3. `Local MV In-Node Renderer` 严格串行调用已连接的本地 H3 MODEL，完成一段就原子保存，可用相同 `chain_id` 续跑。
4. 视频合成后只混入一次完整原曲，分段生成音频不会进入最终成片。

流程不使用远程 H3、ComfyUI HTTP `/prompt` 队列、在线 LLM、TTS 或视频 API。

## 使用

- 换成正确的 Ref2VA 基模、Turbo LoRA、H3 CLIP 和两个 VAE。
- 替换参考图与歌曲；普通歌曲保持 `assume_vocal`，有干声再连接 `vocal_stem`。
- 第一次运行使用新的 `chain_id`。中断后保持全部参数不变并再次运行即可续跑。
- 参考图应清晰、脸部可见。分辨率必须为 32 的倍数；显存不足先降低尺寸，不要并发。
- 最终检查口型、身份、手部、背景和切镜。该流程不是音素级口型模型，不承诺逐音素精确同步。

工作流：`2026-09-01_H3_Local_MV_LipSync_Ref2VA_Turbo4_Advanced_EXP.json`

## 当前成果与边界

- 已通过本地分镜、17n+5 预量化、媒体标签、串行生成/续跑和原曲单次混入测试。
- Prompt Compiler 额外输出标准 Prompt Relay Event，可供现有 Prompt Relay 研究工作流复用；本 MV Renderer 本身按场景提示词生成，不宣称启用了 Attention 级 Prompt Relay。
- 真正 GPU 成片仍需串行低负载验证和人工审核，未审核前保持 EXP。
