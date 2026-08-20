# 学习型 Latent 放大与二阶段生成

这一组把低分辨率 H3 首次生成的干净联合 AV latent，经独立 3D 学习型模型放大视频空间，再在高分辨率画布继续联合音画去噪。它与普通插值放大不同，也不会修改原来的 `Latent Upscale by 32` 稳定节点。

## 工作流

- `2026-08-19_H3_Learned_Latent_TwoPass_I2VA_Advanced_EXP.json`：I2VA按公开原版schedule执行低分辨率4步、学习型latent放大、高分辨率3步；二采专用Mixer默认旁路，可显式开启Tail/Bias/STG/Restart。
- `2026-08-16_H3_Latent_Upscale_By32.json`：普通插值latent放大，只负责32像素整除和画幅误差报告，不创造学习型细节。

## 模型与成果

学习型工作流读取 `ComfyUI/models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors`。当前固定文件SHA-256为 `043E5A48E161610EF6C3EA974645220354D06FA618ABCA15F76D084812EB55C2`，322个FP16张量、24通道3D输入输出。节点只处理视频latent，音频latent保持联合H3原值；默认每次完成后只定向卸载该放大模型的GPU权重。

当前已确认本地和上游网络的322个参数及固定随机输入输出逐位一致。最新示例固定到上游工作流
`64fc9d4`的I2VA合同：FL2VA full INT8、正确`comfyui_alpha8` LightX2V LoRA、量化底模bypass加载、
shift 12/3、低清`simple8`前4步，以及高分原始3步sigma
`0.9035, 0.6316, 0.3158, 0`。不要使用缺少PEFT alpha处理的普通`_comfyui`文件；它会把LoRA更新
放大约16倍并造成整幅融化。

高分第二阶段从video sigma `0.9035`开始时，shift 12/3对应的audio sigma约为`0.701`。Comfy通用
Custom Sampler会先按video sigma初始化整个packed AV latent；T8自定义双时钟现在会在第一次模型调用前，
只把audio slice从video时钟精确重建到audio时钟。该修复保留video初态和随机噪声，不改上游3D网络、
4+3 schedule或总NFE。

真实I2VA 4+3链使用736×416低清latent和`scale_by=2.0`，节点输出1472×832并自动连接高分
Conditioning宽高，无需维护第二套尺寸。任务626.969秒完成，输出1472×832、124帧、24fps、32kHz
双声道，视频/音频严格解码通过；8帧检查未见前一错误LoRA样本的整幅崩坏。单素材仍不能证明普遍
画质或显存优势，因此继续保留Advanced/EXP。
历史v1.35.0的1120×640链曾只开启二采Mixer的Tail +3：高分辨率阶段由3次增至6次前向，任务
302.85秒，媒体合同与严格解码通过。它不是本轮1472×832、corrected alpha8路线的同图A/B，只证明
节点可以串联，不自动证明画质增益。

## 使用方法

1. 第一套 Conditioning 使用低分辨率；第一采样器必须把 `denoised_output` 接到 Learned Latent Upscale，不能使用仍处在中间噪声状态的 `output`。
2. 第二套 Conditioning 必须用相同提示词、首帧、参考媒体和时长；其宽高直接连接Learned Latent Upscale的`width/height`输出。用户只改`scale_by`，不要再手填第二套高分尺寸。Reconcile会拒绝低分辨率旧keyframe，避免 `cond_video_rows` 错配。
3. 使用`Learned Two-Pass Parity Plan`，默认`base_steps=8 / coarse_steps=4 / refine_steps=3 / shift 12/3`；旧`Two-Pass Sigma Plan`保留老工作流兼容，但不再作为原版复现推荐。
4. Parity Plan的`coarse_sigmas`接第一采样器；`refine_sigmas`接`Two-Pass Detail Mixer.refine_sigmas`。Mixer的MODEL、SAMPLER、SIGMAS三路接第二采样器。
5. Mixer全部开关关闭时严格透传高分辨率refine schedule；需要Tail +3时只打开`enable_tail=true`并保留`extra_tail_steps=3`。Bias/STG/Restart可组合但会改变联合音画预测，必须试听审片。
6. `Temporal Detail Enhance`不能接在latent放大和第二采样器之间；它只能接在AV Decode的IMAGE输出之后，AUDIO直接旁路到保存节点。
7. native作者对齐路线保持`second_pass_audio_source=legacy_policy`，第二采样器`output`直接进入AV Decode。
   第一阶段audio随放大后video一起进入第二阶段，第二阶段继续完成联合AV轨迹；不要把第一阶段尚未完成的
   audio x0估计当作最终声音。
8. `first_pass + second_pass_audio_strength=0.0`是显式音频锁定实验，不是native默认。它会阻止第二阶段
   完成audio，可能改变口型条件和声音质量。只有确实需要冻结某条已验证audio latent时才使用，并单独做
   完整试听与口型检查；`highres_template`同样属于显式实验来源。
9. `Two-Pass Audio Audit`只服务上述零mask锁定实验：它校验采样前后audio latent并回锁。不要把它插到
   native默认图中，否则会把第一阶段中间音频误当成最终成品。
10. 默认 `offload_after` 只释放学习型放大模型；`clear_after`连CPU缓存一起清理，`keep_loaded`会持续占显存，只适合连续批量放大。
11. 示例固定用VHS的H.265 MP4。本机Windows上H.264曾产生一帧损坏；H.265结果已通过124帧、32kHz双声道和`-xerror -err_detect explode`严格解码。
12. 量化H3底模使用`LoraLoaderBypassModelOnly`加载`minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8.safetensors`、强度1.0。普通`minimax_h3_fl2v_turbo_4step_v0.1_comfyui.safetensors`为已否决转换，不要替换回去。

## 边界

这是二阶段生成，不是把最终视频直接锐化。模型最大只允许4倍空间放大、输出仍受H3 2.0MP面积上限；不能据单次运行宣称普遍更清晰或16GB永不OOM。Ref2VA、Hybrid、多关键帧和Long Video需分别验证后再扩展，不应从I2VA结果直接外推。

2026-08-21同prompt/seed/model/4+3 NFE检查中，Comfy原生`ModelSamplingAV + Euler`输出与修正后的
T8自定义双时钟输出均由用户完整试听确认为声音正常；二者解码PCM相关约`0.9491`。此前第一阶段audio
零mask硬锁版本由用户明确判定声音异常，已撤销为默认。这个结论只覆盖当前native I2VA样本，不能外推到
`lock_source`、`remix_source`、`reference_only`或含对白的通用口型质量。
