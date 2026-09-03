# 长视频与断点续跑

这一组把短窗口H3生成组织成可接受、可恢复、可后台继续的长视频任务。

## 工作流定位

- `In_Node_Long_Video_Loop_Turbo4_Advanced`：一次排队后在同一个输出节点内严格串行完成全部片段，逐段原子落盘并在中断后按相同合同续跑，最后流式合成为一个VIDEO；不需要手工修改`segment_index`或反复点击队列（实验）。
- `In_Node_Long_Video_Prompt_Relay_EAV_Stock20_Advanced`：在同一个严格串行内循环里，把一条全局Prompt Relay时间线投影到每个片段，并为每段独立执行、审计Enhance-A-Video；旧内循环节点保持不变（实验）。
- `In_Node_Long_Video_Prompt_Relay_EAV_Manual_Second_Pass_Advanced`：在上述路线末端追加可断开的Sampling Plan，支持原尾段细分或每段独立低Sigma二次采样；默认手动表为`0.5, 0.412, 0.350, 0`（实验）。
- `FreeNoise_Prompt_Relay_EAV_Long_Video_Advanced`：在上述内循环前加入FreeNoise视频噪声时间重排；它不占用Attention，因此可与Prompt Relay和EAV组合，音频噪声保持原生独立（实验）。
- `Native_Latent_Timeline_Concat`：把两段或更多已采样完成的H3联合AV latent按原生时间格合并，供后续一次VAE解码（实验）。
- `Native_Latent_Continuation_Concat`：核对Long Video Planner/Conditioning报告，并在双时钟上移除续段重新注入的完整5/22/39帧上下文，而不是固定只去5帧（实验）。
- `Native_Latent_Resume_Manifest`：为保存/重载后的完整H3联合AV latent生成精确内容SHA-256，或用旧清单严格核对是否可安全续接（实验；不负责保存文件）。
- `Native_Latent_Checkpoint_Save_Load`：把完整H3联合AV latent、可选嵌套mask及受支持元数据保存为无pickle的原子safetensors检查点，并在同进程或ComfyUI重启后严格重载（实验）。
- `Dual_Clock_NFE_Checkpoint_Resume`：只为本项目`dual_clock_euler + native_flow`在每个完整Euler步后原子保存采样状态，并在新进程中验证合同后输出剩余sigmas（实验；默认关闭）。
- `Long_Video_22F`：最小分段编排示例。
- `Long_Video_Accepted_22F`：加入accepted manifest，只推进已验收片段。
- `Long_Video_Auto_Resume_22F`：从最后一个已接受片段恢复。
- `Long_Video_Background_22F`：后台队列/租约路线。
- `Long_Video_Background_22F_ScenePlusIdentity`：同时携带场景和身份上下文。
- `Prompt_Relay_Long_Video_Turbo8_Advanced`：整条长视频只创建一次全局事件时间线；每段按
  `timeline_start - context_frames`投影到本地渲染窗口，再与既有Long Video上下文补丁隔离组合。
- `Enhance_A_Video_Long_Video_Accepted_22F_Stock20_Advanced`：原生Stock20逐段EAV；Long Video继续拥有上下文/layout，组合节点为每段创建独立Runtime Audit。
- `2026-09-02_H3_Native_Masked_Video_Context_Plan_B_Segment0_Starter_Advanced_EXP`：独立Plan B配套第0段启动器；固定当前核心原生AV Euler并保存供续段使用的context，不替换现有默认路线。
- `2026-09-02_H3_Native_Masked_Video_Context_Plan_B_Advanced_EXP`：独立Plan B续段；续段把上一段原生video latent尾部硬锁到当前段开头，只做画面续接，当前音频与Vocal Lock mask不变。

## 当前成果

已实现原子manifest、父哈希、OS租约、accepted不回退、崩溃后续跑和后台任务隔离；本项目稳定使用约30～32秒范围，60秒不是发布硬门。

候选保存和最终拼接的AAC编码现已隔离到FFmpeg子进程。视频仍由PyAV逐帧编码，不会把整段RGB放进内存；音频按双声道float32暂存到磁盘后由可取消子进程编码并原子封装。这样即使AAC原生编码器异常，也只会让该候选失败并清理临时文件，不再直接终止ComfyUI。为避免极短音频被AAC帧量化截到绝对manifest边界之前，封装时最多增加一个1024采样AAC帧的有界padding，后续读取仍按manifest精确裁样本。该路线要求`ffmpeg`可从`PATH`调用；它修复的是已观察到的Windows主进程崩溃边界，不代表任意FFmpeg版本、磁盘故障或断电均已验证。

原生latent拼接已完成一条低负载真实验证：两个256×256、22帧、Turbo8的T2VA完整latent合并后一次VAE解码，精确得到39帧；两个源FLAC各29600采样，合并FLAC为52000采样，第二段精确去除9个音频latent步/7200采样。相同seed复跑的源A、源B和合并MP4全部字节一致，三条媒体的视频、音频和联合解码各3/3通过。与两段分别VAE解码后再拼RGB/PCM相比，本例边界视频MAD由0.11164降到0.04269，音频单采样跳变由0.02231降到0.00104，但边界前后100ms电平仍相差8.32dB。由于两段只是相关提示词、不同seed，并未使用上一段画面/音频做续段条件，这只能证明时格、无损PCM、单次解码和局部平滑信号；不能称为无缝长视频、画质更好或省显存。

Long Video续接专用拼接已完成CPU结构验证：124帧时间线接一个124帧、22帧上下文续段，精确输出226帧、67个video latent step和377个audio latent step；再接一个39帧上下文续段，精确输出311帧。节点会同时核对Planner与Conditioning的chain、segment、render帧数、motion keyframe数和时间线位置。该结果只关闭结构与双时钟合同，尚未完成真实多段H3内容的人眼/试听结论。

双时钟Euler逐NFE恢复已完成确定性CPU、真实跨Python进程以及一条真实H3低负载验证。固定256×256×22、4 NFE、12/3双时钟任务先完整生成控制，再在第2步真实中断；关闭该ComfyUI进程后由新进程完成剩余2步。最终视频latent、音频latent、解码RGB、解码PCM以及H.264/AAC容器均严格核对，和不中断控制逐位一致。该证据只证明这一组模型、LoRA、Conditioning、seed、采样器和本机kernel合同，不代表任意wrapper、采样器、分辨率、显存档或跨GPU都成立。

## 使用方法与注意事项

首次使用从Accepted或Auto Resume开始，先跑短段并人工验收。不要手工改写accepted文件；更换提示词、模型、参考或种子后应开启新任务目录。长视频降低的是整片一次性生成的峰值，不代表单段可以无限提高分辨率或关键帧数量。

Native Masked Video Context Plan B续段节点不能生成第0段：必须先用配套的
`Plan_B_Segment0_Starter_Advanced_EXP`以目标`chain_id`完成第0段并保存上下文，再打开Plan B续段工作流，
填写完全相同的`chain_id`并从默认`segment_index=1`开始。不要用原`Long_Video_22F`旧双时钟工作流生成
这条链的第0段。其Conditioning必须保持
`context_audio=video_only`；Planner、Context Load、Conditioning和Plan B节点的报告必须直接连线，不能
手抄或跨chain复用。它会在采样前复制上一段原生video latent尾部并把对应video mask设为0，随后仍由
现有Output Trim移除重建头。上一段audio tail不会注入，当前段audio tensor和已有Vocal Lock audio mask
原样保留。合同冲突会直接报错。该工作流固定当前ComfyUI原生AV `euler + native_flow`；不要改回旧
`dual_clock_euler`，后者在当前核心下已通过同Seed单变量复测确认会产生异常PCM直流偏置和噪音。
两份独立Plan B工作流同时固定用户指定的新版step600 EMA_B
`minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors`；旧通用EMA在匿名人审中声音非常轻且不正常，
不得回退。旧Long Video工作流仍保持原配置。

当前已有一组736×416、同一segment 0、同prompt、同seed `2609024101`、同模型/LoRA/4 NFE/12:3 shift
的首轮真实续段对照：旧video-only软上下文和Plan B各输出102帧，三份源片及两份226帧完整审查片的音画
联合严格解码通过，共同context SHA前后不变。用户盲评画面差不多且均无问题，揭盲后A为Plan B、B为旧路线；
两路均有可闻杂音。第二轮纯器乐也被用户判定两条都是严重噪音。416×224同模型/LoRA/Seed/提示词/4 NFE/
12:3只换sampler的复测中，旧双时钟DC为0.21313，原生AV Euler为0.00060。修复配置完整跑通首段和两种续段，
三路DC接近零且无削波；用户认为比旧版好，但仍感觉音频可能有问题，因此不算PASS。追加的416×224古典音乐+
“你在哪里”4/8 NFE对照均严格解码、无削波；8 NFE逐字命中，4 NFE被识别为“你在那里”。只更换通用EMA/
校正FL2V Alpha8 LoRA的4 NFE匿名对照中，用户判定A正常、B声音非常轻且不对；揭盲后A为Alpha8、B为旧
通用EMA。首个新版EMA_B完整链虽严格解码、无削波且DC接近零，但首段和续段重复请求同一句对白，违反
全片只说一句的合同，已作废为仅机械诊断。运行器现固定首段说一次、续段人物静默并延续同一古典音乐；
用户仍试听该作废包并反馈A/B声音都没问题；揭盲A为软上下文、B为Plan B，两条声音诊断通过且无偏好，但
不关闭重复对白合同、口型或独立接缝评价。修正版随后已用同一新版EMA_B完成：三份源片和两份226帧完整片
严格解码，共享context不变；VAD在两个续段均检出0段对白，第0段单独识别为“你在哪里”。用户再次判定A/B
声音都没问题；揭盲后A为Plan B、B为软上下文，本精确样本声音打平。完整AAC上的ASR仍会在“哪里/那里”
之间波动。旧接缝分析器按面积误选Plan B前两帧背景假脸；连续追踪修正后Plan B/软上下文接缝SFace为
0.873/0.875，两条续段都102/102帧追踪到主脸。共享对白段SyncNet在多种裁剪下为-3/-4帧（25fps），400ms
负对照移动+9/+10帧，未过±1帧机械门。另外，仅延后画面
4个24fps帧的诊断候选回到SyncNet 0帧且音频PCM不变；因其会冻结开头4帧、舍弃末尾4帧，真人对比前不写入
工作流。另一个平滑建立并追回延迟的候选同样回到0帧且保留首尾，但会改变说话前后的运动速度并混合分数帧。
用户随后完成原始/固定/平滑三方真人对比并反馈“3组差不多，都还行”，故本哈希原始片真人口型通过；两个校准
没有可感知优势且各有代价，均不接入工作流。既有旧配置4/8 NFE样本的置信度和400ms负对照响应不足，不能证明
增加NFE可修口型。共享对白段不能选择Plan B路线，精确用词与续段主观接缝仍未验收。本次第0段/软续段/Plan B最低显存余量为
490/475/527MiB，整对仍未过512MiB门。不得宣称通用16GB安全或Plan B普遍更优。

因416×224不足以判断画面，随后按完全相同合同补跑960×544（522,240像素，约0.522MP）真实A/B。第0段、
软续段、Plan B精确为124/102/102帧，全部源片与两份226帧完整片严格音画解码，共享context前后不变；
最低空闲显存1009/545/1122MiB，本精确三阶段均通过512MiB余量门。用户盲评两条差不多，但两条在接缝处
都有可见颜色跳变；揭盲后A为软续段、B为Plan B。因此旧接缝质量不通过，也不能归因给某一条路线。

两份工作流现都在`Output Trim`后、`CreateVideo`前接入可选且默认开启的
`MiniMaxH3LongVideoColorMatchT8Advanced`。只做5帧RGB均值偏移的V1已被人审拒绝：B变化较少但两条仍有，
A左侧明显。V2先做pooled Reinhard Lab色彩/对比度匹配，再用8x5局部分区RGB残差场处理空间不一致色偏；
总像素通道改变量不超过0.02并在24帧内渐隐。它只改解码后的SDR画面，不接触原生latent或音频；疑似
切镜、旧schema、校验和/chain/segment/画布不匹配时不猜测校正，关闭时画面逐像素原样通过。
最终960×512（0.49152MP）同Seed实跑得到124/102/102帧并全部严格解码，共享latent context和第0段
颜色参考在两条续段前后不变；软/Plan B最大局部RGB跳变由0.014492降到0.001714、0.008184降到
0.001238，最低空闲显存532/515MiB。机械与本次512MiB资源门已过。用户盲评反馈左侧A仍有一点跳变、
右侧B好很多；揭盲A为软路线、B为Plan B。本样本接受Plan B接缝色彩连续性，软路线仍有轻微残差；
不为追求软路线零残差而放宽0.02幅度上限。一次本机单样本通过不能外推身份/音频、通用16GB安全、
两条路线完全无跳变或Plan B普遍更优。

In-Node Loop路线适合“不需要逐段人工筛选、希望点一次就跑完整条”的任务。默认`124`帧渲染窗、`22`帧上下文、batch 1，内存只保留当前段；完成段与续接上下文写入磁盘。中断、OOM或ComfyUI重启后，保持模型、提示词、参考、采样参数和`chain_id`不变即可续跑；任一合同参数改变时必须换新的`chain_id`。它会自动接受每个成功片段，因此需要逐段挑片时仍应使用Background/Accepted路线。节点只在片段边界释放自身临时对象，不调用全局模型卸载，也不承诺任意显卡、任意模型封装都不会OOM。

Relay/EAV内循环模板固定为`20步 + dual_clock_euler + native_flow`，不接Turbo LoRA。Prompt Relay Plan只建立一次，`length`必须覆盖最终长片；内循环节点的`global_prompt`和`segment_prompts_json`保持为空。每段顺序固定为：投影全局Relay → 组装Long Video上下文 → 建立精确双时钟sigma → 单一Relay/FETA组合器 → 采样 → EAV审计 → 写候选与`effects_audit.json` → 接受。默认512MiB只是下一段启动前余量门，不是显存峰值保证；联合AV仍需最终试听。

Sampling Plan断开或设为`disabled`时，旧内循环逐值不变。`tail_subdivide`只细分主轨迹尾段；`manual_second_pass`在每段主采样完成后创建独立的普通采样器，按手动Sigma再跑一遍。两遍共用同一份有界preview cache状态，Prompt Relay保持全局时间投影；EAV只审计主采样，不把旧EAV runtime错误复用到第二遍。音频仍由H3联合AV双时钟采样，不独立冻结或加噪。

FreeNoise模板默认`paper_permutation`：所有续段共享一个视频噪声池，但每段按确定性时间置换重新排列；音频噪声不重排。`variance_preserving_blend`可用`reuse_ratio`在共享池与该段独立噪声之间混合。该配置写入断点合同，改变mode、base seed或比例后必须换新的`chain_id`。原论文还在同一个长latent中执行滑窗时序Attention融合，而H3路线仍按124帧训练窗口独立续段，因此这里明确称为FreeNoise噪声重排适配，不宣传完整复现或必然改善接缝。

该组合路线已完成一条单任务、严格串行的256×256×6秒真实Stock20验证：两段分别交付124帧和20个新帧，最终为144帧/24fps、32kHz双声道；两段都执行20次模型前向，使用同一全局Relay计划但生成不同的分段投影。最终文件严格联合解码通过，跨段118～129帧没有黑屏、花屏或场景重置；最低空闲显存约1,850MiB。这个低分辨率机械样本只证明组合、时间线、审计、续接和媒体封装可用，不代表画质提升、音频无差异、任意时长或通用16GB安全。

真实短链已各完成一条256×256和1152×640（约0.737MP）的6秒Turbo4运行。两条都是第一段交付124帧，第二段用22帧上下文续接后只交付20个新帧，最终严格为144帧/24fps和32kHz双声道音频。0.7MP单次运行耗时484.969秒，最低显存余量1,948MiB；严格联合解码、六时点抽帧和118～129帧边界抽查均通过，未见黑屏、花屏或明显场景重置。它只证明这两条短链的控制器和交付数学可用；完整运动与音频仍需人工审看，也不代表任意时长、内容或16GB显卡都已通过。

Native Latent Timeline输入必须是每段采样完成后的完整联合AV latent，不能接空Conditioning latent。所有段必须同画布、同dtype、batch 1、完整`5n+2`视频时格和精确音频时钟；默认将新合并结果放在CPU，但输入仍可能被ComfyUI缓存。节点不会自动卸载或采样；本机两次真实运行的最低整卡余量只有341.11/144.18MiB，均未过512MiB安全门，因此不能宣传16GB安全。要改善内容接缝，后续段仍需正确的画面/音频上下文条件，latent拼接本身不会把两个无关状态变成同一镜头。

Native Latent Continuation Concat必须把Planner和**实际生成该续段的**Long Video Conditioning的
`report_json`直接接入节点，不能手抄。`timeline_latent`接第一段或上一次拼接输出，
`continuation_segment`接本段采样完成后的`denoised_output`。默认
`audio_context_policy=require_video_and_audio`，要求Context Load同时提供上一段音画尾部；显式
`allow_video_only`仍会按时长裁音频重叠，但不能声称音频连续。所有段应先在latent域合并，再只做一次
AV VAE Decode；最终段若报告隐藏尾帧，应在一次解码后按
`trim_tail_frames_after_decode`裁掉，不能提前破坏原生latent时间格。报告JSON只能证明本项目节点的
接线/结构合同，不证明第三方采样器实际消费了Conditioning，也不是运行中NFE断点恢复。

Native Latent Resume Manifest首次运行保持`expected_manifest_json`为空，并把latent与输出JSON成对保存；崩溃后重新加载latent，再粘贴原JSON。默认`mismatch_policy=error`，只有`MATCH + resume_verified=true`才证明AV样本、mask、受支持元数据、shape、dtype和checkpoint ID完全一致。默认8MiB分块只限制临时CPU拷贝，分块大小不参与内容哈希。该节点不保存latent、不写文件、不恢复扩散内部NFE状态，也不代表画面/声音续接质量通过。

Native Latent Checkpoint Save默认`confirm_save=false`，只计算清单而不创建文件；确认后固定写入`output/MiniMaxH3/latent_checkpoints`，使用唯一文件名、文件`fsync`、回读核验和原子替换，已有文件永不覆盖。Load始终校验内嵌内容清单，也可同时要求Save返回的外部manifest和整文件SHA。保存会把完整latent复制到CPU主机内存，且只关闭“已完成检查点跨进程精确重载”这一机械边界；它不会恢复某个NFE中间步、Transformer/sampler内部状态或ComfyUI队列。

Dual Clock NFE Checkpoint/Resume首次运行必须先把`model_contract_id`改成可复核的基座模型SHA、LoRA名称/强度和wrapper清单。**不要把Conditioning的普通`report`文本直接接到`run_contract_json`**；示例已新增`MiniMax H3 NFE Run Contract`，把`positive`、`conditioned_prompt`、`media_map_json`和`report`四路接入编译器，再把编译器输出接给断点节点。编译器分块哈希真正的Conditioning张量内容，分块大小不改变摘要。`mode=checkpoint_each_step`还必须显式打开`confirm_checkpoint_write`；默认`disabled + false`不创建目录或文件。重启后保持同一模型、补丁、Conditioning、seed、总steps和12/3 shift，切到`resume`并填写原相对路径；只读续跑可保持确认开关为false，需要继续推进同一检查点才打开。文件固定在`output/MiniMaxH3/nfe_checkpoints`，使用无pickle safetensors、非阻塞同路径锁、写后回读和原子替换。绝对路径、`..`、符号链接、并发写、非有限tensor、内容篡改、sigma/seed/布局/合同不一致均拒绝。此路线不接受DPM++等多步历史、ancestral/SDE随机状态、原生或第三方sampler历史，也不能从一次未完成的Transformer前向内部恢复。

Prompt Relay长视频示例必须先运行segment 0并成功保存AV tail，再把Planner的`segment_index`改为1、2……。
不要为每段重建全局Plan，否则事件会重新从第一幕开始。唯一允许的Turbo顺序是
`UNET → Prompt Relay Long Video Conditioning → 修正Alpha8 Bypass LoRA → DualClock`。
默认继续使用论文范围的`video_only_paper`；单组联合AV盲测七项均为平局，没有证据把`joint_av_exp`
设为默认。当前新增路线已完成一条真实segment 0→1链：736×416、Turbo8、22帧上下文，输出
124+102帧，事件顺序没有重启，视频/音频各3轮严格解码通过；整卡峰值约15478/14984MiB。
接缝处没有静音断层，但原始PCM边界跳变仍需最终试听，因此继续标EXP，不宣传普遍画质或16GB安全。

EAV长视频模板固定为原生Stock20，不使用旧Turbo LoRA。正确顺序是
`Long Video Conditioning → DualClock → EAV+Long Video Composer → Guider`；Planner的
`segment_index/context_frames`必须同时连接Conditioning与Composer。segment 0要求0帧上下文，
续段只接受5/22/39帧并核对运动keyframe偏移。每段独立要求20次前向、每个活跃前向50次FETA测量；
候选保存、人工接受、manifest和断点恢复仍由既有Long Video节点负责。当前只完成低负载合同、注册、
导入接线和项目/用户文件一致性检查，没有压力测试，不宣称接缝更好、音频非劣、提速、省显存或通用16GB安全。
