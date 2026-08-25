# Skin Finish / 肤质收尾

这一组位于 H3 最终解码和 Face Refine/Motion Recovery 之后，用于生成一个非生成式、可回退的肤质后期候选。P0处理可靠遮罩内的低频肤色不均、轻度斑驳和油光观感；Studio Timeline路线允许在每个创作镜头内部按帧改变参数且不跨切镜插值；Frequency Split把P0候选的低频肤色/亮度层与来源已有高频纹理显式解耦；独立Specular-Aware Split Advanced EXP只在亮肤、正高频和原候选确实更暗时，把普通分离丢失的候选处理按比例补回，且绝不越过原候选；固定ParseNet节点可把单轨Face Refine Plan收窄成像素语义皮肤MASK，新的多人语义路线则复用SAM3.1逐镜人物轨迹、以YuNet五点对齐逐脸解析并与各自人物MASK相交；P1还提供绝对帧续跑状态、原音频包封装，以及直接接文件VIDEO的两遍低内存路线；P2 Texture Guard在候选之后增加源片相对的亮暗保护、裁切和纹理硬门。它们都不是修脸、去模糊、身份重建、毛孔生成或口型工具。

## 工作流

- `2026-08-24_H3_Skin_Finish_External_Mask_Advanced_EXP.json`：同时展示 Basic、Advanced 和 Preview/Audit。默认 `accept_candidate=false`，分屏左侧为原片、右侧为候选；还输出全分辨率裁切、绿/红 used/rejected mask、放大差异和当前帧前后各两帧循环。
- `2026-08-24_H3_Skin_Finish_MultiPerson_Video_Finalize_Advanced_EXP.json`：读取未裁切的8-bit SDR `VIDEO`，原生SAM3.1只追踪一次并卸载；Multi-Person节点用CPU YuNet把人物轨迹收窄成保守脸部肤区，Video Finalize在人工接受后重编码画面并验证原音频包payload逐包不变。画布内含5个NOTE。
- `2026-08-24_H3_Skin_Finish_Two_Pass_Video_Stream_Advanced_EXP.json`：直接连接Long Video/Studio最终文件VIDEO，不经过`GetVideoComponents`。第一遍只保存固定YuNet脸框、切镜和来源摘要；第二遍用有界CPU chunk处理并立即编码，兼容音频包payload逐包复制和验证。画布内含5个NOTE。
- `2026-08-24_H3_Skin_Finish_Texture_Guard_Advanced_EXP.json`：把现有Advanced的source/candidate/used mask接入独立P2机械护栏。默认保护深阴影和接近饱和高光；逐帧新增裁切或高通纹理下限失败时整帧回退source。默认仍不接受候选，画布内含5个NOTE。
- `2026-08-24_H3_Skin_Finish_Semantic_Mask_Advanced_EXP.json`：同一批来源帧先建立source-bound Face Refine Plan，再用固定SHA的FaceXLib v0.2.2 ParseNet生成皮肤MASK；绿色为候选皮肤，红色为受保护五官/头发/配饰。MASK接Advanced的`external_exact`输入，后面再接Texture Guard；两个接受开关默认false，画布内含5个NOTE。
- `2026-08-24_H3_Skin_Finish_MultiPerson_Semantic_Mask_Advanced_EXP.json`：原生SAM3.1逐镜追踪后卸载，语义节点先运行并释放固定YuNet，再以五点相似变换将每张可靠脸对齐到FFHQ 512，运行固定ParseNet并把皮肤MASK反投影到各自人物轨迹内。可选`identity_assignment`只给跨镜报告加Character标签，不自动改色或证明身份；后接Advanced、Texture Guard和Video Finalize，三道接受门默认false，画布内含5个NOTE。
- `2026-08-25_H3_Skin_Finish_Per_Person_Advanced_EXP.json`：使用独立的侧脸Advanced EXP语义节点，原严格五点对齐成功时完全沿用旧路径，只有严格残差门拒绝的姿态才使用1.45×原侧脸方形裁切ParseNet回退；随后为Character_A/Character_B串联独立Profile，把不同preset、amount、texture、shine和tone参数路由到各自皮肤区域。精确`shot:track`配置优先于Character；重叠、未匹配、来源/hash不一致的像素保持原片。末端Safety Audit以`unique_track_owner + hard_gate`检查蒙版泄漏、变化越界、时间处理突变、人物串色和音频PCM，失败时把精确source送给Finalizer；通过后仍由Finalizer唯一的人工接受开关决定是否保存候选。画布内含7个NOTE。
- `2026-08-25_H3_Skin_Finish_Frequency_Split_Advanced_EXP.json`：严格按`Skin Finish Advanced → Frequency Split → Texture Guard`接线。两遍低通半径默认按短边1%计算并封顶32px；使用候选低频和来源已有高频重组，mask面积异常或新增裁切超限时逐帧回退。默认不接受候选，画布内含6个NOTE，明确说明来源模糊、噪声、HDR、音频和人工审片边界。
- `2026-08-25_H3_Skin_Finish_Studio_Timeline_Advanced_EXP.json`：两个22帧Studio镜头、四个global关键帧和完整多人语义/Safety Audit示例。`hold / linear / smoothstep`只影响同一Studio镜头内的连续参数，preset到下一关键帧才切换；Studio shot只定义时间，SAM shot:track只定义人物，不能混用编号。输入帧数、Timeline总帧数和24fps track plan必须一致，默认所有接受开关关闭。画布含7个NOTE。
- `2026-08-25_H3_Skin_Finish_Quality_Stream_Advanced_EXP.json`：直接接最终未裁切文件VIDEO，不经过`GetVideoComponents`。第一遍仅保留YuNet脸框、切镜和来源摘要；第二遍默认2帧CPU chunk运行固定ParseNet语义皮肤MASK、Skin Finish、Frequency Split、Texture Guard和跨chunk Safety Audit，并立即单线程H.264编码。默认false完全不分析不写文件，画布含6个NOTE。
- `2026-08-25_H3_Skin_Finish_Oil_Control_Stream_Advanced_EXP.json`：复用同一低内存文件流，但固定经过真实油感素材机械验证的`oil_control / amount 0.35 / texture_keep 0.90 / shine 0.35 / chunk 2 / CRF 16`。来源为v1.0八步LoRA的960×544×124中文说话近景，124/124帧处理且音频packet/PCM精确；匿名人工结果为`ABSTAIN_UNSURE`（8项平局、2项弃权、双方无硬失败，观感“似乎感觉差不多”），所以该参数只作为保守起点，不是已证明有效的默认方案。画布NOTE强调油感不足时不要强行接受，以及高光、蜡像感、眼唇、闪烁和halo的人工复核。

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
- Frequency Split安全起点为`low_frequency_strength=1.0`、`source_detail_gain=1.0`、`separation_radius_percent=1.0`、`maximum_radius_px=32`、`chunk_frames=4`。先保持来源纹理增益1.0；提高它可能同步放大噪声、压缩块和锐化边缘。该节点必须放在Texture Guard之前，不能替代后续硬门
- Specular-Aware Split是尚未进入示例工作流的实验候选，默认`separation_radius_percent=3.0 / highlight_detail_suppression=0.65 / positive_detail_threshold=0.004`，设为0时逐像素等于旧Frequency Split。最新候选边界六帧复核中，0.65/1.0两档都通过Texture Guard和Safety Audit，最终平均变化仅为原始Skin Finish候选的29.7%/32.5%，肉眼仍弱；固定CineStyle默认链虽更明显，但会显著降低纹理代理且使最亮肤区整体变亮，不可当作直接去油答案。不要据此替换现有工作流或跳过整段时间一致性与人工审片
- Guided Surface Finish Advanced EXP仍未进入示例工作流。它是独立实现的逐帧guided-filter表面处理，并新增有界亮度肩部处理宽面积亮区，不复制CineStyle也不冒充物理反射分离；默认保护遮罩外、眼唇、辅助通道和AUDIO，`accept_candidate=false`。单组六帧静态门与一条124帧两帧分块流式门均通过：2帧Surface与2帧Texture Guard按合同回原片，Safety Audit零失败，最大时间跳变0.00381594，音频精确。但匿名review `b3aad4e0d57b`中原片在整体、自然度、油光、肤色均匀和halo五项胜出，其余五项平局，候选0项胜出且双方无硬失败。本参数组主观门未通过，不新增工作流、不替换Oil Control Stream，也不要通过继续增大亮度肩部强度来强行制造差异
- Surface v2改用语义皮肤区域内的多尺度局部亮差，并在硬MASK边界内侧两像素渐退；不会再把均匀明亮肤色直接当作宽油光。高光专用参数`0.90 / smoothing 0.25 / texture 0.96 / compact 0.90 / broad 0.90 / blemish 0.10 / radius 2.5%`只作为review `8e89bff3bc95`的固定候选：六帧和124帧机械门均通过、音频精确、无回退，映射盲时间审计通过；正式人审10项全部平局且双方无硬失败。它消除了v1的明显回退，却没有证明肉眼改善，因此仍无示例工作流或推荐默认值，也不要继续调强同一亮度肩部来制造差异
- Dichromatic Specular Advanced EXP按Shafer双色反射近似在linear-sRGB中估计镜面分量；六帧与124帧机械门、安全审计、音频和映射盲时间审计均通过。anonymous review `b2e13261f44e`的正式盲测表单为原片7胜、3平、候选0胜、双方无硬失败；揭盲后用户复看并修正为“基本一样”。揭盲后意见不覆盖原始盲测记录，保守结论仍是没有证明可感知收益。本路线没有示例工作流、推荐参数或默认推广；机械PASS不能解释为肉眼去油或物理材质恢复
- Timeline关键帧推荐从同镜头首尾两键开始，`smoothstep`、amount差不超过约0.15、texture_keep保持0.90以上、tone保持0。需要逐人物时用经人工复核的`character_id`，只对单条SAM轨迹覆盖时才用`shot_track`；优先级固定为`SAM shot:track > character_id > global > source`。高频来回设置会造成时间泵动，节点不会自动判定美感
- Quality Stream安全起点为`subtle / amount=0.30 / texture_keep=0.95 / shine_control=0.25 / crop_expansion=1.45 / minimum_class_probability=0.55 / mask_feather_px=0 / source_detail_gain=1.0 / chunk_frames=2 / CRF=18`。只在最终文件上运行；不按段重复处理，不为扩大面积降低语义门，显式打开`accept_candidate`也仅表示渲染待审候选。true执行前会在可测平台检查至少2,048MiB可用系统RAM；不足时返回原VIDEO并ABSTAIN，不加载ParseNet、不写文件。该门来自本机32秒实测约1,163MiB进程增量并保留约885MiB余量，不是任意机器安全证明
- 明确需要压制额头、鼻梁或双颊局部油光时，可改用独立Oil Control Stream工作流的`oil_control / 0.35 / 0.90 / 0.35 / chunk 2 / CRF 16`起点。它比`subtle`更有针对性，但不等于自动磨皮：完整审片若出现蜡像感、五官变化、时间泵动或halo，必须保留原片；来源没有明显油光时不应为了看出差异继续加大参数
- Basic/Advanced完整IMAGE P0会在分配候选、两张完整MASK与float16差异图之前，按实际`帧数×宽×高×通道×dtype`、`chunk_frames`、代理尺寸和mask来源估算本节点增量CPU内存：逐项组件和有界scratch合计后乘1.5，再加固定512MiB余量；Windows同时检查可用物理内存与commit。任一可测值不足即返回精确source并`ABSTAIN_INSUFFICIENT_SYSTEM_RAM_NO_CANDIDATE_ALLOCATED`，不会准备MASK或候选；测不到时会明确记录unavailable并保留原有CPU有界路线，不冒充PASS。该门不可由widget降低，也不覆盖输入IMAGE本身、其他节点、ComfyUI整图或GPU显存
- Semantic Mask安全起点为`include_neck=false`、`crop_expansion=1.45`、`minimum_face_weight=0.35`、`minimum_class_probability=0.55`、`feature_protection_px=3`；不要为了扩大面积盲目降低概率或面积门
- Multi-Person Semantic安全起点为`YuNet=0.45`、`minimum_face_height_px=32`、`minimum_person_overlap=0.20`、`minimum_track_quality=0.10`、`minimum_class_probability=0.55`、`maximum_alignment_rms=0.08`、`minimum_ready_frame_fraction=0.50`；逐人物侧脸节点另保持`profile_crop_expansion=1.45`。不要提高RMS来强行把侧脸贴到正脸模板；任何来源、检测、解析或覆盖门失败仍保持空MASK
- 逐人物工作流的SAM文本先用单数`person`，让multiplex检测分别返回人物实例；不要写`two people`等可能被检测成一个整体的数量短语。本机清晰双人素材以`detection_threshold=0.35`得到两条独立轨迹；这只是召回起点，若素材出现误检应提高阈值并人工复核彩色轨迹
- Per-Person Profile安全起点仍建议从`subtle / amount=0.25～0.35 / texture_keep=0.90 / shine_control=0.25～0.35 / tone_adjust=0`开始。`tone_adjust`只是中间调曝光式微调，不是自动肤色估计；先用Character配置，只有确切知道SAM报告中的`shot:track`时才使用镜头覆盖
- Per-Person执行器的优先级固定为`精确shot:track > Character > 可选默认Profile`。安全默认`source_unmatched`不会处理没有可靠身份/路由的人物；多人轨迹重叠像素永远回退source，不能靠降低门槛强行涂抹。报告会为每条路由列出display-referred Rec.709亮度代理、平均/峰值改变量及高低裁切比例，只供不同人物人工复核；不会自动给出肤色公平、美感或身份结论
- Safety Audit在普通单人链可保持`mask_only + report_only`；逐人物工作流使用`unique_track_owner + hard_gate`，默认`maximum_mean_abs_change=0.08`、`maximum_peak_abs_change=0.30`、`maximum_temporal_effect_jump=0.04`、`maximum_track_leak_fraction=0.001`。这些阈值只会拒绝明显机械风险，不是“美肤评分”；不要为了让候选变绿而放宽门。审计节点自身`accept_candidate=false`，工作流只使用其`gated_candidate`，最终仍在Video Finalize人工接受

Advanced 只原对象透传 AUDIO；Preview/Audit 同时核对输入和透传 PCM 合同。三个文件输出节点都不把音频解码后再编码，而是对兼容MP4音频包的payload做逐包复制与SHA-256复核。共享VIDEO门同时读取ComfyUI bit depth、PyAV实际像素分量bit数、格式名和FFmpeg整数颜色枚举；仅接受未标记或BT.709/传统SDR兼容的8-bit输入，明确拒绝10/12/16-bit、PQ、HLG、线性/Log、BT.2020、P3和ICTCP等，再把已接受的primaries/transfer/colorspace/range复制到H.264输出。旧Multi-Person示例的`GetVideoComponents`仍持有完整IMAGE；新的Two Pass Video Stream自身不输入IMAGE，只保留逐帧小型元数据和默认4帧处理chunk。它降低的是该后期阶段的峰值内存，不代表生成链、编码器、任意长片或普遍16GB都已认证。

Preview/Audit运行后会在节点内显示最长边不超过512的JPEG代理：左侧原片、右侧候选。拖动滑杆只改变浏览器中的即时对比；点击“写回位置 / Apply”只修改当前节点已有的`comparison_position`，之后仍需用户手动排队，且不会修改`accept_candidate`。代理图可能受缩放和JPEG影响，只用于定位分界线；肤质、五官、闪烁和边缘判断必须查看节点的全分辨率crop/mask/difference/±2帧输出及最终审片视频。代理编码失败不会让后端审核结果失败。

三个文件输出节点固定 `libx264 threads=1`，并在原子发布前使用单线程 FFmpeg `-xerror -err_detect explode` 严格解码。原因是代表性实跑曾发现Windows默认多线程PyAV/libx264生成了帧数表面正确但码流损坏的候选；该损坏候选被拒绝，修正后的1088×544×124两遍流式候选通过严格解码和来源/候选PCM逐值哈希一致。Quality Stream先完成960×544×5真实CPU小片探针，随后只运行一次736×416×768、24fps、32秒的真实H3最终文件：384个2帧chunk、690帧语义就绪、78帧安全回退source、Safety Audit零失败、最大跨chunk处理跳变0.00052124；1579.531秒完成，峰值working set约1.974GiB，严格解码、音频包/PCM一致且ParseNet已释放。它关闭这一条32秒资源与媒体机械门，但人工观感、任意时长和通用内存安全仍待验证。缺少FFmpeg时显式接受会失败关闭，不会发布未经严格验证的文件。

## 2026-08-25 live多人链实测

- 一条832×736×124、24fps、单人到三人并含手遮脸的素材已实际跑通工作流中的native SAM3.1、固定YuNet、固定CPU ParseNet、Advanced subtle、Texture Guard和Video Finalize。候选严格解码通过，原片与候选的163个AAC包payload及解码PCM完全一致。
- 本机16GB RTX 4060 Ti峰值约5.57GB、最低空闲约10.54GB；SAM追踪后选择性卸载。但是三人物逐帧CPU ParseNet整条链约18分钟，因此该路线适合最终短片/精选镜头的审慎后期，不应按这个数据宣传长片高吞吐。
- 抽样蒙版能分别覆盖三张可靠脸的皮肤，并排除眼眉、鼻孔、嘴唇和头发；这只是机械与抽样目视结果，最终是否更自然仍需观看完整并排视频。
- 相似棚拍段落即使来自两个文件，也可能低于默认0.28切镜阈值而继续留在S0；另一个22帧明显背景硬切探针已确认默认阈值能重建为S0/S1。镜头颜色/编号仍不等于人物身份，跨镜角色必须另行审核。
- 逐人物路线另在960×704×69清晰双人全片上真实运行一次：SAM文本`person`、阈值0.35得到`0:0/0:1`两条全身轨迹；`0:0`使用subtle 0.25，`0:1`使用oil_control 0.55。原严格执行器报告两条路由分别处理564,882和249,606个可靠皮肤像素，重叠与未匹配像素为0；当时较难侧脸/对齐失败帧逐帧保持原片。
- 后续从同一清晰双人源抽取0/32/43/48/51/68六帧做独立CPU ParseNet探针：严格模式每帧只接受1人，侧脸裁切回退每帧均接受2人，只增加6次必要回退；整帧皮肤面积保持2.29%～2.70%，来源tensor不变，绿皮肤/红保护区目视未见背景泄漏。
- 更新后的侧脸路线随后对同一960×704×69双人全片完成一次native SAM3.1→YuNet→CPU ParseNet→双Profile→Per-Person→Texture Guard→Finalize真实运行。覆盖从严格基线96/138人物帧提高到138/138，其中42次为严格拒绝后的侧脸裁切回退；候选/审片严格解码，来源/候选91个AAC包、46,740字节payload及解码PCM SHA完全一致。全链616.062秒，16GB RTX 4060 Ti峰值约6.19GB、最低空闲约9.92GB。证据在`artifacts/skin-finish-per-person-profile-live-validation-20260825/20260825-020540-47378389`。
- 2026-08-25又对同一来源实际重跑唯一一条未压缩Safety Audit全链：native SAM3.1→YuNet→CPU ParseNet→双Profile→Per-Person→Texture Guard→`unique_track_owner + hard_gate`→Finalize。结果为`PASS_HARD_GATES`，69帧零失败、track泄漏与重叠歧义均为0，最大时间处理跳变0.00007348（门限0.04），PCM精确一致；候选/审片严格解码，138/138人物帧覆盖和42次必要侧脸回退保持不变。
- 新实链388.5秒，16GB RTX 4060 Ti峰值5,763MiB、最低空闲10,347MiB，8197结束且8188未触碰。证据在`artifacts/skin-finish-per-person-profile-live-validation-20260825/20260825-032941-cbeb0b91`。另有纯CPU确定性夹具确认：两人左右交叉时参数继续随人工Character而非屏幕位置移动，重叠区逐值回原片；切镜后track编号和人物左右互换时，经hash绑定的人工mapping仍把同一Character路由到同一Profile；深浅两组人物都独立报告亮度、处理幅度和裁切。这些只关闭路由/诊断机械门，仍不证明主观更自然、说话口型、真实不同肤色公平、身份真值或长片吞吐。

## 人工审片

`tools/build_skin_finish_human_review.py`可对最终source/candidate MP4生成匿名A/B审片页，不重新编码。页面同步播放两路视频，支持0.25×/0.5×慢放、逐帧、联动放大和点击改变放大中心；逐项记录肤质自然度、油光、高光、均匀度、纹理、五官保护、闪烁、光晕、跨人物误涂和身份/口型，并将A/B硬失败分开保存。`tools/analyze_skin_finish_human_review.py`用私钥揭盲并校验review ID、manifest hash、每项投票和失败列表；一名审阅者的结果永远不会自动接受候选或建立普遍优越结论。

本机69帧双人最终成片已经为最新Safety Audit PASS候选生成`human-review-safety-audit-v1`页面，路径在上述新实链目录内。该素材可以评审侧脸、双人物、肤色/油光/纹理和时间闪烁，但没有足够说话口型证据；口型项应选择“无法判断”，另用明确说话素材验收，不能从静态微笑推断通过。

## 当前边界

所有路线都禁止运行时下载。语义节点只接受`ComfyUI/models/facedetection/parsing_parsenet.pth`，要求85,331,193字节、SHA-256 `3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2`，并使用`torch.load(weights_only=True)`在CPU加载；缺文件、依赖、大小/hash或来源不匹配时输出空MASK并ABSTAIN/REJECT，执行后释放模型且不建持久缓存。当前没有模型、完整帧或MASK缓存可供清理，因此不提供会误导用户的Clear节点；若未来引入真实缓存，必须再实现内容hash、条目/总字节上限与显式Clear。单轨Face Refine Plan没有五点关键点，所以单轨节点仍使用扩展正方形脸框；旧多人语义节点直接读取YuNet五点并逐人严格对齐，新侧脸节点只在该严格对齐抛出拒绝时使用原姿态裁切，不传播丢失关键点、不提高残差阈值，也不把可选身份标签当作证明。旧P1 Multi-Person和流式节点的脸内椭圆仍是保守代理；Per-Person路线能按人工Profile显式区分人物和镜头，但不会自动估计肤色，也不会把SFace/SAM标签冒充身份真值。Frequency Split只保留来源本来存在的高频，来源模糊时仍然模糊，来源含噪声时也可能保留噪声。Safety Audit只能拦截可测的硬故障：ParseNet若把嘴唇错误分类为skin，它不能凭美学自行纠正；源相对时间向量也不是光流或自然度oracle。六帧低负载夹具、一条124帧live SAM完整链和一条69帧侧脸双Profile完整链已证明各自机械合同；合成交叉/切镜夹具进一步关闭显式路由数学，但仍需完整人工审片。它们都不等于跨镜身份真值、真实不同肤色公平性、真实交叉遮挡重识别、真实长片连续性或效果通过。P2高通RMS可能把噪声也视为高频，只可用作过度磨皮的硬失败下限。HDR/Log/广色域、10-bit和旋转/裁切VIDEO现会在编码前明确拒绝，而不是被静默转成SDR；真正处理这些格式、生成式皮肤重建和任意容器/codec仍不支持。
