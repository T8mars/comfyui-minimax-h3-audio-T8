# Skin Finish / 肤质收尾

这一组位于 H3 最终解码和 Face Refine/Motion Recovery 之后，用于生成一个非生成式、可回退的肤质后期候选。P0处理可靠遮罩内的低频肤色不均、轻度斑驳和油光观感；固定ParseNet节点可把单轨Face Refine Plan收窄成像素语义皮肤MASK，新的多人语义路线则复用SAM3.1逐镜人物轨迹、以YuNet五点对齐逐脸解析并与各自人物MASK相交；P1还提供绝对帧续跑状态、原音频包封装，以及直接接文件VIDEO的两遍低内存路线；P2 Texture Guard在候选之后增加源片相对的亮暗保护、裁切和纹理硬门。它们都不是修脸、去模糊、身份重建、毛孔生成或口型工具。

## 工作流

- `2026-08-24_H3_Skin_Finish_External_Mask_Advanced_EXP.json`：同时展示 Basic、Advanced 和 Preview/Audit。默认 `accept_candidate=false`，分屏左侧为原片、右侧为候选；还输出全分辨率裁切、绿/红 used/rejected mask、放大差异和当前帧前后各两帧循环。
- `2026-08-24_H3_Skin_Finish_MultiPerson_Video_Finalize_Advanced_EXP.json`：读取未裁切的8-bit SDR `VIDEO`，原生SAM3.1只追踪一次并卸载；Multi-Person节点用CPU YuNet把人物轨迹收窄成保守脸部肤区，Video Finalize在人工接受后重编码画面并验证原音频包payload逐包不变。画布内含5个NOTE。
- `2026-08-24_H3_Skin_Finish_Two_Pass_Video_Stream_Advanced_EXP.json`：直接连接Long Video/Studio最终文件VIDEO，不经过`GetVideoComponents`。第一遍只保存固定YuNet脸框、切镜和来源摘要；第二遍用有界CPU chunk处理并立即编码，兼容音频包payload逐包复制和验证。画布内含5个NOTE。
- `2026-08-24_H3_Skin_Finish_Texture_Guard_Advanced_EXP.json`：把现有Advanced的source/candidate/used mask接入独立P2机械护栏。默认保护深阴影和接近饱和高光；逐帧新增裁切或高通纹理下限失败时整帧回退source。默认仍不接受候选，画布内含5个NOTE。
- `2026-08-24_H3_Skin_Finish_Semantic_Mask_Advanced_EXP.json`：同一批来源帧先建立source-bound Face Refine Plan，再用固定SHA的FaceXLib v0.2.2 ParseNet生成皮肤MASK；绿色为候选皮肤，红色为受保护五官/头发/配饰。MASK接Advanced的`external_exact`输入，后面再接Texture Guard；两个接受开关默认false，画布内含5个NOTE。
- `2026-08-24_H3_Skin_Finish_MultiPerson_Semantic_Mask_Advanced_EXP.json`：原生SAM3.1逐镜追踪后卸载，语义节点先运行并释放固定YuNet，再以五点相似变换将每张可靠脸对齐到FFHQ 512，运行固定ParseNet并把皮肤MASK反投影到各自人物轨迹内。可选`identity_assignment`只给跨镜报告加Character标签，不自动改色或证明身份；后接Advanced、Texture Guard和Video Finalize，三道接受门默认false，画布内含5个NOTE。

## 推荐顺序

```text
H3采样 -> Latent放大/二采 -> AV Decode -> Face Refine/Motion Recovery
-> Skin Finish -> 字幕/调色/Tape-FX -> 保存
```

使用外部遮罩时，当前示例的第二个 `LoadImage` 需要带 Alpha 的 RGBA mask；普通无 Alpha 图片会得到空 MASK，节点会 `ABSTAIN` 并保持原片。已有 Face Refine Plan 时，可把 Advanced 的 `mask_source` 改成 `face_refine_plan` 并连接由同一批来源帧生成的 plan；这只是保守脸区代理，不是语义皮肤解析。

## 安全起点

- `preset=subtle`
- `amount=0.35`
- `texture_keep=0.90`
- `shine_control=0.35`
- `mask_feather_px=3`
- `proxy_long_side=640`
- `chunk_frames=4`
- 两个 `accept_candidate` 均保持 `false`，完成首/中/尾帧和 ±2 帧循环复核后再显式接受
- Texture Guard安全起点为`shadow_protection=0.10`、`highlight_protection=0.94`、`minimum_texture_ratio=0.78`、`maximum_new_clipped_fraction=0.0005`；这些是机械拦截器，不是自动美颜分数
- Semantic Mask安全起点为`include_neck=false`、`crop_expansion=1.45`、`minimum_face_weight=0.35`、`minimum_class_probability=0.55`、`feature_protection_px=3`；不要为了扩大面积盲目降低概率或面积门
- Multi-Person Semantic安全起点为`YuNet=0.45`、`minimum_face_height_px=32`、`minimum_person_overlap=0.20`、`minimum_track_quality=0.10`、`minimum_class_probability=0.55`、`maximum_alignment_rms=0.08`、`minimum_ready_frame_fraction=0.50`；任何来源、对齐或覆盖门失败都保持空MASK

Advanced 只原对象透传 AUDIO；Preview/Audit 同时核对输入和透传 PCM 合同。两个P1文件节点都不把音频解码后再编码，而是对兼容MP4音频包的payload做逐包复制与SHA-256复核。旧Multi-Person示例的`GetVideoComponents`仍持有完整IMAGE；新的Two Pass Video Stream自身不输入IMAGE，只保留逐帧小型元数据和默认4帧处理chunk。它降低的是该后期阶段的峰值内存，不代表生成链、编码器、任意长片或普遍16GB都已认证。

两个文件输出节点固定 `libx264 threads=1`，并在原子发布前使用单线程 FFmpeg `-xerror -err_detect explode` 严格解码。原因是代表性实跑曾发现Windows默认多线程PyAV/libx264生成了帧数表面正确但码流损坏的候选；该损坏候选被拒绝，修正后的1088×544×124两遍流式候选通过严格解码和来源/候选PCM逐值哈希一致。缺少FFmpeg时显式接受会失败关闭，不会发布未经严格验证的文件。

## 当前边界

所有路线都禁止运行时下载。语义节点只接受`ComfyUI/models/facedetection/parsing_parsenet.pth`，要求85,331,193字节、SHA-256 `3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2`，并使用`torch.load(weights_only=True)`在CPU加载；缺文件、依赖、大小/hash或来源不匹配时输出空MASK并ABSTAIN/REJECT，执行后释放模型且不建持久缓存。单轨Face Refine Plan没有五点关键点，所以单轨节点仍使用扩展正方形脸框；多人语义节点则直接读取YuNet五点并逐人对齐，但不会传播丢失关键点，也不把可选身份标签当作证明。旧P1 Multi-Person和流式节点的脸内椭圆仍是保守代理；所有现有Skin Finish效果参数仍是共享中性色，不会自动按人物改肤色。六帧0.67584MP双人低负载检查证明了真实YuNet/ParseNet、五点反投影和逐人物相交，但人物区域来自来源绑定夹具，不等于一次新的SAM3.1实跑、全片连续性或人工效果通过。P2高通RMS可能把噪声也视为高频，只可用作过度磨皮的硬失败下限。HDR/Log/广色域、10-bit、旋转/裁切VIDEO、生成式皮肤重建和任意容器/codec仍不支持。
