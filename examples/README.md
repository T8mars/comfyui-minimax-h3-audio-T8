# Example

`audio_lock_api.json` 是 API prompt 格式示例，展示完整链路：音频窗口、统一条件、H3
采样、联合解码、同步裁切和 MP4 输出。

`dual_clock_4step_api.json` 展示 H3 Turbo LoRA 的四步原生音画生成。双时钟节点同时
替代 Sigma Shift、sampler 选择器和 scheduler，输出直接接入 `SamplerCustomAdvanced`。

`multirate_exp_api.json` 展示独立 EXP 节点的 4 个视频宏步 / 8 个音频微步设置。
`audio_steps` 是完整联合 H3 DiT 的实际调用次数；测试 4/10 时直接把它改成 10。

`still_image_edit_api.json` 展示 Ref2VA 实验性单图编辑：主图作为 `<Picture 1>`，生成
22帧短视频候选并由 Still Decode 选择中间帧输出 IMAGE。22帧对应原生 `17n+5` 网格的
`video latent_t=7`，比124帧便宜很多，但仍低于约124帧的训练下限。当前示例使用本机已有的
pruned Ref2VA INT8，不加载与其不完整兼容的 Turbo LoRA，默认512×512、20步，并通过
Still Preflight 输出检查报告。需要把 LoadImage 的占位文件名替换为输入图。

`ref2va_visual_reference_strength_exp_api.json` 展示在现有
`MiniMaxH3AudioConditioningT8.positive` 与 `BasicGuider.conditioning` 之间插入
`MiniMaxH3VisualReferenceStrengthEXPT8`。它只写入全局
`minimax_visual_cond_noise_aug`，不修改模型、latent 或采样设置。示例默认0.990用于观察差异；
正式 A/B 应固定其他所有变量，先比较无节点/0.999，再测试0.995、0.990、0.980、0.950。

三份Hybrid Model Advanced API prompt使用相同的严格pair、stock loader和Stock20控制链：

- `hybrid_model_advanced_api.json`：显式视觉参考profile，便于受控矩阵逐项覆盖recipe；
- `hybrid_model_audio_reference_api.json`：独立参考音频，Inspector连接Conditioning并使用
  `auto_match_reference_modalities_exp`，当前会解析成最小audio-row profile；
- `hybrid_model_mixed_reference_api.json`：参考图与参考音频组合，自动解析成video+audio-row profile。

后两份示例中的图像/音频文件名都是占位符，必须替换为用户有权使用的素材。自动模式只按真实
reference类型选择要实验的AdaLN模态行，不负责判断哪个recipe质量最好；无额外reference或未知
类型会明确拒绝。`tools/run_hybrid_model_matrix.py`可从这些API顺序生成FL2VA、Ref2VA和Hybrid
对照，逐路全局释放模型，并输出hash、显存、盲评包、`matrix_summary.json/csv`；可选本地ASR、
InsightFace和WavLM都必须由用户显式指定，工具不会下载模型。

三份实验性语音 API prompt 复用现有 H3 模型对象，不在节点内部重复加载模型：

- `speech_described_api.json`：描述音色的单句英文语音。当前 stock 20步基线使用
  `res_multistep + simple`，最终显式执行全局模型卸载；如后续还要生成 H3 视频，应把
  `release_policy` 改为 `keep_loaded`。
- `speech_reference_clone_api.json`：经授权参考音频的 Ref2VA 语音。先把
  `LoadAudio.audio` 替换为自己的 2–15秒单人参考，并把示例中的本地 ASR/WavLM 模型目录
  替换为实际路径。`trim_exact_target` 只有找到完整目标台词时才裁切；speaker cosine 仅是
  实验报告，不是通用身份判定。
- `speech_dialogue_two_speaker_api.json`：两段独立生成、分别 ASR 精确裁切，再按绝对 sample
  时间轴合成的双人对白。它没有使用未经验证的单次联合多人生成；两个分支都会占用同一 H3，
  由 ComfyUI 串行执行，最后一个 Finalize 才做全局卸载。

对白结束后保留完整背景声提供两套独立示例：

- `dialogue_safe_master_api.json` / `workflows/H3_Dialogue_Safe_Master_EXP.json`：输入已验收的
  独立对白、音乐、环境和 SFX stems；本地 ASR 只有在整条对白 stem 恰好包含一个完整目标时
  才把 `clean_exact` 接给安全混音。默认 strict 要求三个背景 stem 已经是完整目标时长，最终
  母带不会随对白结束而截断。
- `dialogue_timed_bed_lock_api.json` /
  `workflows/H3_Dialogue_Timed_Background_Bed_Lock_EXP.json`：两遍 H3 路线。输入一条不含对白的
  完整背景底轨，2秒前允许 H3 生成，2秒后以0 denoise锁住底轨 latent。标准124帧窗口的真实
  Audio VAE 会产生206步而联合时钟需要207步，所以示例有意使用 `fit_reported` 并报告补1步；
  这不是静默时长修正。

两者都不能把已混合的 H3 母带自动拆成人声、音乐和音效。前者是 sample 精确分轨合成；后者
只有40Hz latent边界，并且真实解码探针在边界后约0.3秒观察到 VAE 时间感受野影响。不要把
它们描述成源分离、无接缝硬切或角色闭嘴保证。

三类语音功能也分别提供 ComfyUI 0.4 画布工作流：

- `workflows/H3_Speech_Described_Stock20_EXP.json`：描述音色单句，默认不运行可选 QA；
- `workflows/H3_Speech_Reference_Clone_Stock20_EXP.json`：参考音色、英文 ASR 精确目标裁切与
  WavLM 余弦报告；导入后先替换 `speech_reference.flac`，且仅能使用有合法权利的参考；
- `workflows/H3_Speech_Dialogue_Two_Speaker_Stock20_EXP.json`：两角色逐 turn 独立生成，两个
  Studio 都保持模型缓存，合成并保存后才由 Finalize 执行一次显式全局卸载。

三份工作流都复用同一组外部 H3 MODEL、Qwen3-VL CLIP 和双 VAE，使用当前实测的
32像素、stock 20步 `res_multistep + simple` 基线，不加载视频 Turbo LoRA。

语音节点和示例都标为 EXP。当前真实生成只验证英文；16GiB 探针的最小整卡余量低于项目
512MiB 安全门槛，因此这些示例不能被称为 `memory_safe`、高保真克隆或绝不 OOM。插件不会
自动下载 ASR/说话人模型，也不会随示例保存用户参考音频。

`workflows/` 内是可直接拖入 ComfyUI 画布的完整示例：

- `H3_Turbo_Stable_4V4A.json`：稳定双时钟 4/4；
- `H3_Audio_Lock_Source_Stable_4V4A.json`：输入音频锁定，最终保留干净原音轨；
- `H3_Audio_Remix_Source_Stable_4V4A.json`：按0.35强度重绘输入音频，保存模型解码音轨；
- `H3_Audio_Reference_Only_Stable_4V4A.json`：输入音频仅作为 `<Audio 1>` 参考，保存全新生成音轨；
- `H3_Turbo_EXP_4V8A.json`：EXP 视频 4 / 音频 8；
- `H3_Turbo_EXP_4V10A.json`：EXP 视频 4 / 音频 10。
- `H3_Still_Edit_22Frames_EXP.json`：Ref2VA 单图语义编辑，512×512、22帧、20步；
  可在 Reference Image Edit 节点点击“＋”添加最多8张附加参考图，也可切换1、5或124帧。
- `H3_Ref2VA_Visual_Reference_Strength_EXP.json`：完整 Ref2VA 生成链中的全局视觉参考强度
  后置节点；导入后替换占位参考图，固定 seed 后从0.999逐步下降。0.950及以下可能明显破坏
  身份、动作、构图和首尾帧，不能宣传为“去油修复”。
- `H3_Long_Video_22F_EXP.json`：手工逐段长视频续写；
- `H3_Long_Video_Accepted_22F_EXP.json`：候选预览、接受与可恢复状态链；
- `H3_Long_Video_Auto_Resume_22F_EXP.json`：总时长规划与人工审核后自动恢复；
- `H3_Long_Video_Background_22F_EXP.json`：显式启用的后台自动排队长链；
- `H3_Long_Video_Background_22F_ScenePlusIdentity_EXP.json`：在后台长链上增加完整场景首帧与
  同人物身份裁剪图两个独立输入，并预设续写段使用 `scene_plus_identity` 双参考。
- `H3_Speech_Described_Stock20_EXP.json`：描述音色的 stock20 实验语音；
- `H3_Speech_Reference_Clone_Stock20_EXP.json`：有授权参考音色、ASR裁切与speaker报告；
- `H3_Speech_Dialogue_Two_Speaker_Stock20_EXP.json`：两角色逐turn生成、sample级合成并统一释放。
- `H3_Dialogue_Safe_Master_EXP.json`：唯一精确目标边界检查后，把独立 stems 合成完整时长母带；
- `H3_Dialogue_Timed_Background_Bed_Lock_EXP.json`：两遍 H3 的40Hz分时背景底轨锁定。
- `H3_Hybrid_Model_Advanced_Stock20_EXP.json`：显式视觉参考Hybrid profile的Stock20实验链；
- `H3_Hybrid_Model_Audio_Reference_Stock20_EXP.json`：自动audio-row Hybrid参考音频链；
- `H3_Hybrid_Model_Mixed_Reference_Stock20_EXP.json`：自动video+audio-row图像与音频混合参考链。

三份音画画布工作流使用相同 seed、prompt、EMA LoRA 和输出设置，便于直接比较。它们已写入
当前 T8 安装中实际存在的非裁剪 H3 INT8 基模、NVFP4 H3 文本编码器和两套 VAE 文件名，
并使用 `LoraLoaderBypassModelOnly`，适配 INT8 基模。要比较非 EMA LoRA，只需在 LoRA
节点中切换到 `minimax_h3_turbo_4步加速_comfyui.safetensors`。

三份输入音频画布统一使用736×416、124帧、稳定4/4和5秒 Audio Window。导入后先替换
`Load Audio` 的占位文件。锁定模式的最终音频来自 Conditioning `mux_audio`；重绘和仅参考模式
来自 AV Decode `generated_audio`，因此不能只改模式下拉框而忽略输出接线。

静态编辑工作流独立使用 pruned Ref2VA INT8，不串联 Turbo LoRA。导入后先在 Load Image
选择主编辑图；默认 Prompt 以 `<Picture 1>` 指向它。预检提示“22帧低于近似训练范围”是
预期警告，但 `ready` 应为 true。

双参考长视频工作流导入后必须同时替换两张占位图：第一张是第0段的完整场景首帧，第二张是
同一人物的清晰正脸或上半身裁剪。它使用独立 `chain_id`，不会和普通后台示例共用 manifest。
这仍是 EXP 基线：32秒单链的身份保持已有正向证据，但动作幅度门槛未全部通过，也不构成所有
显卡和素材的无 OOM 保证。升级节点后需要完整重启 ComfyUI，Python schema 才会出现两个新增输入。

API prompt 内的占位模型名需要手动替换；`workflows/` 中的前端工作流已经使用当前机器的
实际文件名。示例最后使用 VideoHelperSuite 的 `VHS_VideoCombine`；核心 T8 节点本身不依赖
VideoHelperSuite。
