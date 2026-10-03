# 已采用外片 → 显式续拍（本地 EXP，未发布）

这是独立于原生 accepted-parent 续拍链的新入口。外片可以没有 H3 模型或采样快照；本功能不捏造原始 latent、MODEL、seed、StageResult 或链中前段祖先。旧节点和工作流不改。

## 连接

1. 先在导演台上传、登记外片，手动采用、设置整数裁切范围并保存。未保存草稿不参与采样。
2. `H3 External · Saved Adopted Take` 填写已保存项目、镜头和外片 take UUID，选择5/22/39帧运动尾及音频政策。不是选一个视频参考就自动采用。
3. `H3 External · RGB / PCM Context Encode` 连接 Source 和实际视频 VAE，设定当前条件画幅。LOW/HIGH 可分别放一个编码节点，用不同画幅与各自连接的 VAE；原生 resize 后编码，不能将重编码称原始采样尾态。
4. `H3 External · ONE Conditions / Model` 接 Context、MODEL、CLIP、两 VAE、当前镜头提示词及生成长度，输出独立 MODEL、CONDITIONING 和 AV latent。可接现有独立采样节点；Bridge 有独立输入，EAV／Prompt Relay 不装在此节点里。外片 motion 路径上的效果绑定仍需单独资格，不能把该新 Context 塞进要求原生祖先的旧 Continuation Effects 节点。
5. render 包含开头 context_frames 帧。交付时明确去掉这些上下文帧，再审接缝；不会自动拼回原片、替换采用版本、审批或保存新的原生祖先。

Source 完整报告含已保存项目修订/SHA、原片完整 SHA、登记回执 SHA、实际视频/音频时钟、已采用范围和所选尾部 `[out-context_frames,out)`。更改项目、采用版本、裁切、文件字节、回执或相关代码后原 Source 失效，应重新明确选择。编码后 tensor/VAE 身份也检查；未知用户 VAE wrapper 可正常执行但不认证跨进程便携缓存，不提供该上下文的持久化缓存或原始 latent 恢复。

## 明确边界

- 首版只直接接受完整解码确认的零起点、24fps CFR。VFR、旋转、多轨、非零起点或其他 FPS 先明确另存转换为新素材；原片保留，不偷偷补帧、重采样或改时间轴。
- `video_only` 不使用外片声音作为运动上下文，也不会自动替换新生成音轨。内部原生上下文 schema 所需的音频占位明确标为未使用，不是恢复的外片音频 latent。
- `reencode_stereo_pcm_context` 要求真实双声道音轨、音频覆盖所选尾部且帧边界对应整数原采样点。独立 worker 提取同采样率/同声道 float32 PCM，不混音、不复制单声道；由当前原生音频编码函数按其 VAE 采样率编码，保留原 H3 音频格点 overhang 规则。派生音频 latent 并非原片原始采样状态，也不等同原 PCM 字节；原片与原音轨不会写回或替换。
- 返回的 last frame 来自采用范围真实末帧，非生成预览或浏览器截图。编码窗口只取所选尾部，不取被裁掉的片尾。
- 独立 sampler 生成的是新候选。尚不承诺任意底模、LoRA、语义桥和 EAV/Relay 配置下接缝自然、音色连续或原生双阶段 exact 恢复。不能把这份新重编码上下文塞入旧 `Continuation Select Accepted Parent` 来绕过原祖先守卫。

当前十二个完整 CPU 范围195项通过（实际 RGB/立体声 PCM、5/22/39窗口、原 H3 条件构建、改源/采用/trim/回执拒绝及旧接缝回归）；CUDA未初始化。另有29项完整路由／续拍范围锁住实际 session 相对脚本 GET，计数重叠不相加。

已完成一条实际原生画布 Save As／重开／Queue：对已人工采用的外片 `[226,248)`，video_only、448×256、124帧、原4步 sampler，显式去掉22帧上下文后得到102帧／4.25秒完整 H264/AAC 新候选。阶段完整身份／manifest 校验可通过；独立新GPU进程显式加载该完成阶段，零采样交付，完整解码RGB／PCM与首次逐值一致。原工程、原片、五所选权重、源 epoch 与控制器／Core SHA保持，owned服务关闭。曾误选CPU／float32的冷解码试验已中断保留，不冒称与GPU／float16等价；未重复采样。

这只资格一条 video_only 的采样与保存交付机制，不等于 PCM-context、新双采／效果组合、接缝自然／身份／音色或整项目质量通过。完整接缝人审与新效果的原生成片仍独立验收，不自动采用。

独立外置效果现见 [External Relay／Stage EAV](EXTERNAL_CONTINUATION_EFFECTS_EXP.md)；四张可编辑完整保存／冷交付新图在 [67-radar-external-continuation](../examples/workflows/67-radar-external-continuation/README.md)。新接口的 CPU／画布／媒体资格分别记录，不覆盖上述原 plain 资格。
