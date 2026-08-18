# MiniMax H3 Audio T8

面向当前 ComfyUI 原生 MiniMax H3 的独立 T8 节点扩展。当前版本为 `1.30.1`，共注册
118 个节点。新增四个隔离的SPEED Advanced节点，按官方论文实现空间渐进分辨率、DCT频谱扩张、kappa状态缩放与sigma对齐，并为H3联合音画、关键帧和参考模态重建每阶段条件；此前的尾段细化调度、平滑模型时间偏置、联合音画Rectified-Flow Restart、H3时空引导、时序保护细节增强、默认无影响的动态 Guidance 与尾段额外 NFE 因果实验、原生SAM3.1多人分色追踪、参考图身份建议、逐角色顺序修复与审片合成，人工验收的MANUAL512 REL Face Refine机械基线、带源片回退的Face Refine候选质量门、隔离的上游机制Face Refine Parity、32像素整除且比例误差可审计的latent放大、分镜感知的远景脸二次生成规划、严格视频 latent 注入、低去噪双时钟采样与非破坏回贴审计，只读环境审计、克隆局部 MLP 激活分块、有界 Qwen 视觉参考前缀缓存、参考语义 IR、统一角色表、声音画布、多后端提示词编译、可视时间轴、非破坏性局部重做、文件级成片交付、同进程采样轨迹探针、计划式驱动音频实验与安全 AV 解码，以及原生音画条件、Hybrid组合兼容审计、可恢复的Hybrid artifact维护、前置显存/VBAR策略、隔离的 FL2VA×Ref2VA 小型混合补丁、多关键帧时间线、对白边界分析、对白安全分轨混音、分时背景底轨锁定、来源视频音画重绘准备、音频控制与后处理、稳定双时钟采样、实验性多速率采样、
隔离的分段长视频续写、总时长编排、候选/接受状态与文件级合成、Ref2VA 单图/多图
参考的静态语义编辑，以及带异常释放保护、持久分段、精确时长后期和显式音色库的实验性语音链。

节点按稳定性与用途分为以下菜单：

| 菜单 | 状态 | 内容 |
|---|---|---|
| `T8/MiniMax H3/Audio` | 稳定 | 音画条件、音频处理、预检、双时钟采样与 AV 解码 |
| `T8/MiniMax H3/Audio/Experimental` | 实验 | 多速率联合采样、计划式驱动音频注入与显式安全 AV 解码 |
| `T8/MiniMax H3/Still/Experimental` | 实验 | Ref2VA 静态图像条件、预检与候选帧解码 |
| `T8/MiniMax H3/Conditioning/Experimental` | 实验 | 全局视觉参考强度、多关键帧，以及有界 Qwen 视觉参考前缀缓存 |
| `T8/MiniMax H3/Models/Experimental` | 实验 | 只读环境审计、MLP激活分块、显存/VBAR策略、Hybrid工具与严格的同进程采样轨迹探针 |
| `T8/MiniMax H3/Studio/Experimental` | 实验 | 参考语义IR、统一角色、声音画布、多后端提示词、可视时间轴、非破坏性局部重做执行和成片交付 |
| `T8/MiniMax H3/Long Video/Experimental` | 实验 | 总时长分段、断点续作、候选预览/接受、后台逐段执行、原子 manifest、已接受上下文与文件级合成 |
| `T8/MiniMax H3/Speech/Experimental` | 实验 | 描述/参考音色、ASR与身份评估、异常释放保护、逐句对白、长文本断点、ADR和显式音色库 |
| `T8/MiniMax H3/Source AV/Experimental` | 实验 | 来源视频24fps窗口、H3双流latent严格组装、画面/音频独立mask与无VAE拆分 |
| `T8/MiniMax H3/Quality/Experimental` | 实验 | 高速动态审计、sigma因果实验、repair规划，以及远景脸局部二次生成与回贴审计 |
| `T8/MiniMax H3/Quality/Experimental/Face Refine Parity` | 实验 | 隔离复现上游FaceRefine的21/51高斯轨迹、逐帧去噪、音频锁和24/24回贴，并提供MANUAL512 REL机械基线校验 |
| `T8/MiniMax H3/Quality/Experimental/Face Refine Multi-Person` | 实验 | 原生SAM3.1按镜头分色追踪2～3人、CPU参考身份建议、逐角色H3修复与审片后顺序合成 |
| `T8/MiniMax H3/Latent` | 稳定 | 32像素整除、比例误差最小化的普通/H3联合latent空间放大 |
| `T8/MiniMax H3/SPEED/Experimental` | 实验 | H3空间频谱标定、渐进画布计划、原始多模态条件源和整链分阶段采样 |

本包不是把源音频简单塞进 latent：它按 ComfyUI 当前 H3 实现维护媒体展示顺序、
`<Picture N>` / `<Video N>` / `<Audio N>` 标签、联合 AV latent、首尾关键帧、参考媒体和
噪声掩码之间的契约。

## 安装与兼容性

将项目目录放入 ComfyUI 的 `custom_nodes/minimax-h3-audio-T8`，重启 ComfyUI 后即可在上述
菜单中找到节点。基础节点没有额外 pip 依赖，复用 ComfyUI 自带的 PyTorch、torchaudio 和
MiniMax H3 实现；可选语音校验才延迟导入 `faster-whisper` 或 `transformers`，缺少它们不会
阻止整个插件加载，也不会暗中下载模型。Face Refine手工ROI没有额外依赖；默认真人自动检测
延迟使用ComfyUI环境中的OpenCV `FaceDetectorYN`，动漫EXP检测延迟导入`onnxruntime`，旧的
本地Ultralytics路线仍为显式可选项。任何检测路线都不会在执行时联网下载模型。

本机已把固定人脸模型放入`ComfyUI/models/face_detection`，GitHub插件仓库不分发权重：

- `face_detection_yunet_2023mar.onnx`：默认真人/远景CPU检测，OpenCV Zoo MIT，SHA-256
  `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`；
- `anime_face_detect_v1.4_n.onnx`：仅供`local_anime_onnx_exp`动漫EXP路线，deepghs MIT，SHA-256
  `fd860b650a4377046842c3cd80d01b0b408bdfbdb4acee5759630f82c6ef04a9`；
- `face_recognition_sface_2021dec.onnx`：多人参考身份建议，OpenCV Zoo Apache-2.0，SHA-256
  `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`。

多人追踪还需把官方`sam3.1_multiplex_fp16.safetensors`放入`ComfyUI/models/checkpoints`，本机校验
SHA-256为`9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03`。节点按行为检查
`SAM31Tracker + track_video_with_detection`，旧SAM3或未知包装器会明确拒绝，不按文件名猜版本。

目录内的`YUNET_SOURCE.json`、`ANIME_FACE_SOURCE.json`和`YUNET_LICENSE.txt`记录固定revision、
来源、哈希和许可。YuNet不识别纯动漫并非故障；动漫模型也不能作为真人或身份验证器。
当前590项完整回归通过；本次改动文件Ruff与compileall通过，115份非artifact项目JSON、diff check与
白名单启动继续使用ComfyUI
`v0.33.0@7fe8a61385`；上一轮Qwen缓存/H3
真实生成兼容探针使用`v0.32.0-16@ddbaa8752`，较大范围真实生成矩阵仍以
`0.31.0@cbbc9dab1`为主基线。Face Refine已在当前`v0.33.0@7fe8a61385`完成画幅安全的
736×416×124完整H3三冷三暖，冷/暖最低整卡余量717.6/922.1MiB，执行后private spread
3.2/79.1MiB；另一次736×416×362冷启动链以1,176.2MiB余量完成。固定YuNet也在3,226张
WIDER FACE验证图、39,123个有效人脸上完成集成评估：默认0.35的precision/recall为
62.23%/64.99%，0.60为86.10%/56.94%，且`<16px`召回从43.60%降到32.25%。因此不强改默认
阈值，仍要求人工审核Plan；这些证据不授予真实画质、多人身份安全、跨环境或通用显存结论。
运行环境为 Python 3.10+。模型、VAE、CLIP 和可选 LoRA仍需按具体任务自行安装。

`1.27.2`保留此前追加的6个字面以`Advanced`结尾的多人节点，并从多人角色参考节点移除
`rights_confirmed`界面控件与执行门槛；旧API残留字段会被兼容忽略。其余旧节点ID、顺序、输入、
默认值和稳定`sampling.py`均不变。流程先用当前ComfyUI原生SAM3.1对每个镜头做2～3人分色追踪，
再以清晰单人参考图建立YuNet+SFace CPU角色档案。参考图默认`dominant_face_auto`：只有最大检测框
相对第二名同时满足面积比≥1.8、置信度差≥0.20且自身置信度≥0.60时才自动选择，否则仍拒绝；
`require_single_face`保留为严格模式，`largest_face_exp`保留为显式无条件最大脸模式。颜色和`0:0`之类索引只代表镜头内轨迹，
不能直接当人物身份。自动建议低于相似度或间隔门槛会拒绝，用户可用
`{"0:0":"Character_A","0:1":"Character_B"}`逐镜头明确覆盖。

每个修复任务只处理“一个人物×一个镜头窗口”，默认73帧（24fps下约3.04秒）、`manual_512`、
legacy crop2.5、21/51平滑，并复用已人工验收的`relative_to_clip 0.8/0.35`和face-only 24/24回贴。
当所选镜头只比请求窗口少1～16帧时，Repair Job只为H3上下文复制末帧补齐，Composite会自动丢弃
补齐尾帧并保持来源帧数；超过16帧仍明确拒绝，避免把过短镜头伪装成长片。
旧工作流继续使用原`crop_factor`；新2/3人示例显式选择`target_face_px=300`，节点会自动换算有效
crop factor，并报告H3画布内实际脸高与受源边界限制的帧数。新Turbo示例统一为8步，旧4步只保留为
历史验证记录。不同人物顺序生成，不会把
三个人同时送入H3；候选通过`previous_composite`链式合成，默认`accept_candidate=false`且拒绝
重叠mask。SAM默认在追踪后只卸载自身及其clone，不调用全局卸载，再进入H3；最终音轨始终取原视频。

Face Refine的产品边界是“重建崩坏五官”，不是“把模糊视频锐化”。输入脸部已经清楚、但眼口鼻或
脸型发生结构畸变时，参考图和局部H3重绘可以纠正结构；输入本身失焦、低清或只由小图放大时，H3通常
会继续保持这种模糊质感，`target_face_px`也不会凭空恢复来源中不存在的纹理。单人、双人和三人前端
工作流现在都内置Markdown使用说明；当前推荐统一为MANUAL512、`relative_to_clip 0.8/0.35`、
`er_sde + simple`、Turbo 8步和人工完整审片。

真实2人240×416×22帧探针中，SAM轨迹、SFace自动Alice/Bob绑定、两路MANUAL512 H3顺序生成、
mask外逐位保护和最终合成都完成；结果严格解码为22帧24fps，SHA-256为
`C74000515CFED4DB8A7D6E1DCD428F4AF379D3CEA89A432C3AE5EEC806F818E2`。随后把含4人的群像裁成
240×416×22帧低分辨率测试片；`person with a visible face`正确选择3名可修复正脸人物并排除背对镜头者，
三个人物计划均完成SAM约束YuNet定位、人工审核轨迹映射和SFace Guard，再用seed42/43/44依次生成。
prompt `0c37c0b3-e910-405f-9b3f-0a159c048b9e`在95.78秒完成，最终H.264/AAC文件严格解码为
240×416、22帧、24fps、0.917秒，SHA-256为
`C3CCB956397AC7497E8241DAB97D057ABAFFC20C625945662DE2608917B4DC42`；来源/结果解码PCM SHA-256
均为`3645A04B3F853F324732FFB9779EE1C95B01F6E5F68C6A07968ECBEDAAD552C1`。

为避免不到1秒的短片掩盖跟踪和时序问题，随后又完成608×448×73帧（3.042秒）三人验证。普通
`person with a visible face`在这段四人素材上把第3条轨迹给了背对镜头者；改为示例当前默认的
`front-facing person with a visible face`后，三条轨迹稳定对应左侧关羽、中间女性和右侧黄头巾男性。
三张来源脸约50～100px，均送入MANUAL512画布，seed42/43/44三路H3顺序完成。第三人24px羽化区
与已接受区域重叠50,621像素，默认`reject`按设计停止；人工确认后用`keep_old_exp`保留旧像素并只应用
第三人的非重叠区域。最终文件严格解码为608×448、73帧、24fps、3.042秒，SHA-256为
`AB26FC42A0FD9EFA5DA32877100554F1487165DEF2498BCC0495DD7638F656BB`；来源/结果解码PCM MD5同为
`4c7905d4a36f6f9c456b7e074b52707e`。五个时间点的放大对照未见新增鬼脸或串人，局部清晰度变化较轻，
尚无盲评证明普遍修复增益，因此继续保留EXP与人工accept。

随后使用一段原生1920×1408、69帧、24fps的清晰双人侧脸视频验证“清楚但五官崩坏”场景：SAM3.1
追踪69帧，前56帧按合法H3窗口分别使用两张清晰单人参考图、MANUAL512、目标脸高300px、
`relative_to_clip 0.8/0.35`和Turbo 8步顺序修复，末13帧保留为未修复对照。用户完整观看后确认本次
实际修复了五官，并确认正确边界是“结构修复有效，但不会把原本模糊的画面直接锐化”。这是新增的一条
清晰双人素材人工验收证据，不等于所有身份、角度、遮挡或模糊输入都会改善。

这次三人实跑同时发现并修复了一个真实边界错误：Parity Stitch把`alpha == 0`定义为mask外，旧多人
Composite却把`alpha <= 1e-6`也当成mask外，第二个人的羽化尾部因此被误拒。现在两端统一为
`alpha > 0`，并新增mask有限性、0～1范围及微小正羽化值回归门；仍不会放行真正的mask外改动。
精确2人链抽样最低余量约489MiB，前一单分支曾见450MiB；三人冷启动完整链结束后粗采样最低约
375MiB，均低于512MiB保守门。因此2人和3人都只能称本机短片机械通过，不能宣传通用16GB安全、
跨镜头身份可靠或跨素材修复质量。

`1.26.0`不增加节点，也不改变前100个节点或稳定采样数学；它修正此前新追加的
MANUAL512 REL Parity路径，使其真正复现用户选中作者结果的关键机制：Ultralytics接收BGR而不是
RGB、裁剪边缘仍使用作者居中的平滑脸罩、逐帧去噪按`crop height / crop_factor`取脸尺寸、回贴后在
来源坐标做colour match，并直接把Comfy解码出的89帧交给video VAE，不再人为复制第90张像素帧。
机械对照中，修正后裁剪平均绝对误差为`0.00000677`、去噪曲线最大误差`0.00001423`、完整回贴平均
绝对误差`0.00000117`；这些数值来自固定上游commit的相同帧/检测器/参数对照。

修正后的完整T8链在prompt `1ed411fa-9b91-45c4-801d-7f45b3597fe5`中用112.48秒完成。输出
`face_refine_t8_author_parity_v2_seed42_00001_.mp4`严格解码为89帧320×320@24fps和32kHz双声道，
SHA-256为`0DD8C79F95B01647F3BF345B6503C83A5860BE99BA66D8D72114BD274E9A0884`；音频PCM MD5仍与来源/
作者目标完全一致（`26d40526bd022d7237ba183bd8777966`）。相对作者目标的全片SSIM由旧T8路径的
`0.955273`提高到`0.967059`。本次0.1秒级轮询观察整卡峰值15,823/16,380MiB，约557MiB余量，
89帧逐帧SSIM范围为0.943776～0.990705，最低五帧人工抽图未见新的灾难性脸崩。用户随后完整观看
作者目标/T8 v2双栏并确认“两边效果一样好”，因此该固定素材、seed和模型链的“至少同等”人工门已通过。
该链只刚越过512MiB本例门槛，仍不能标成通用16GB安全档，也不能把单人单例结论外推到其他输入。

`1.25.0`完整保留前100个节点的ID、顺序、输入、输出和默认值，只在末尾追加
`MiniMaxH3FaceRefineManual512RelativeBaselineT8Advanced`。推荐Face Refine Parity工作流改为用户
完整审片后选中的`manual_512 + crop_factor=2.5 + relative_to_clip + 0.8/0.35`路线，并继续使用
21/51轨迹平滑、锁定零音频mask、face-only回贴、dilation24、feather24、colour match1。新基线节点
要求Plan、latent注入、逐帧去噪和Stitch四份报告属于同一次执行；任一参数、hash、fallback、非有限像素或crop内
最小脸高低于200px都会明确拒绝。它不再通过1.24.0源相似度代理门把人工选中的候选切回原片，输出
candidate tensor原样传递；这只是固定机械配置，不是跨素材的身份、自然度或画质保证。

本机89帧fixture中，该路线把来源105～195px脸部送入约205～312px的512画布，crop实际倍率
1.60～1.95x；结果严格解码通过，SHA-256为
`19EA5844643B962F6FD197E34705861916D69F7EA70F3E00A2DF022D6A017399`。用户在六路完整视频中选择
该路线为最好，因此人类审片结论覆盖了本案例中偏向absolute的源相似度代理。该次本机16GB运行虽
成功，但最低抽样余量仅161MiB，仍不得宣传为通用16GB安全档。

推荐工作流现在还锁定了该次人工选择实际使用的模型链：Ref2VA pruned INT8、Qwen3-VL NVFP4、
官方video/audio VAE、`minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8-T8-convert.safetensors@0.75`、
两张身份参考图、来源音频锁定、face YOLO、`er_sde + simple + 4步 + denoise0.45 + seed42`。真实素材
由Comfy解码为89帧，节点会显式复制最后一张crop一次形成合法90帧H3内部输入，报告该对齐动作，并在
回贴前严格丢弃这1张内部尾帧；最终视频仍为原始89帧，绝不隐藏裁切任意有效源帧。旧工作流默认的
`require_h3_grid=true`保持不变，只有推荐示例显式关闭它且仅接受0或1帧对齐尾巴。

该推荐API已用本项目自己的6个Parity节点完成一次真实端到端执行，prompt ID
`57741215-c23b-4a9b-87b7-7288ce175ff1`，耗时107.41秒；输出严格解码为89帧320×320@24fps和
32kHz双声道音频，SHA-256为`B91BBE09C2AF4266EDD2975760A13749A0DB819054BE6C8118E144F0D4AF3097`。
其解码音频MD5与源片及人工选中版本均为`26d40526bd022d7237ba183bd8777966`；视频相对人工选中版本
SSIM为0.955273。机械链、时间轴、原声保护和配置Guard因此已实跑关闭，但“效果至少一样”仍是主观
判断，完整双栏视频必须由用户审片，不能由SSIM替代。

`1.24.0` 保留前99个节点ID、顺序、输入和默认值不变，只在末尾追加
`MiniMaxH3FaceRefineQualityGateT8Advanced`；此前`1.23.0`追加的4个Face Refine Parity
Advanced节点。新路线不再复用已经被盲评否决的5帧均值/统一0.45/ellipse12旧链，而是隔离实现
中心21帧、尺寸51帧的反射Gaussian轨迹，crop factor 3、最高768方形画布、30～120px脸尺寸映射
0.8～0.35的逐帧视频noise mask、严格为0的音频mask，以及face-only、dilation24、feather24、
colour match1的回贴。采样示例直接使用原版机制的`er_sde + simple + 4步 + base denoise 0.45`，
LightX2V FL2V Turbo LoRA强度0.75和seed42；最终只复用原视频完整音轨。上游face/person YOLO权重
已安装并校验，固定上游代码的实跑实际使用face YOLO；T8 Plan仍默认使用固定YuNet，除非用户显式
选择本地Ultralytics，因此不称身份跟踪等价。新增质量门连接在Parity Stitch后，只接受结构SSIM、
脸区变化、实测清晰度和残差时序都通过且连续至少3帧的候选；其余帧回到源tensor。代理通过仍不是
身份或画质证明，输出必须观看完整视频。
API示例为`tests/fixtures/api/face_refine_parity_advanced_api.json`；第三方MIT归属见
`THIRD_PARTY_NOTICES.md`。

本机已额外跑完三条736×416×124 T8真实候选，并直接运行固定commit上游的四个原始节点。T8
原版0.8/0.35强度和只换all-50 Hybrid权重的两条
均出现明显鬼脸并否决；只把强度降到0.45/0.15后，脸区SSIM由0.46179升到0.76044、动作相关升到
0.69200，但脸部高频中位比只有0.50535，仍不能证明“修清楚”。首条采样期观测余量约433MiB，
低于512MiB保守门。固定上游节点的0.8/0.35脸区SSIM/动作相关为0.49077/0.42102；只改成
0.45/0.15后为0.77776/0.70423，但清晰度中位比仍只有0.56297，也被否决。默认高强度T8候选接入
质量门后124帧全部被拒，输出在tensor阶段精确返回源片；两次独立H.264编码后相对单独源片直通的
全画面/脸区SSIM仍为0.999885/0.996810。当前结论是：质量门能避免这条已知鬼脸回归，但尚无真实
生成帧证明修复增益；仍需补齐上游原始GGUF、embedded prompt、完整refs和独立VOCALS后做严格盲测。

`1.22.1` 不新增或修改节点契约，只追加可复现的Face Refine离线验证工具与证据：精确24fps/
`17n+5`计划探针、随机本地A/B评审包与严格揭盲汇总、WIDER FACE固定阈值集成评估、受控人物
交叉矩阵、六候选来源相似/时序代理，以及对应回归测试。公开标注集只在本地按
CC-BY-NC-ND-4.0用于非商业验证，不随插件分发。首份完整单评审揭盲中，原片在总体和身份上均
6比0胜出，动作6组全平；候选的身份、表情/嘴型、时序、接缝和自然度均值为1分，原片均为5分，
备注一致指出候选鬼脸和脸部来回跳动。因此这6个当前候选明确否决；至少5名独立评审者的晋级门
仍未达到，不能外推为跨素材通用结论，身份/画质晋级和自动接受继续拒绝。

`1.22.0` 完整保留此前94个节点的 ID、顺序、输入、默认值和稳定 `sampling.py` 数学，只在末尾
追加 `Latent Upscale by 32`。它在像素域选择宽高均可被32整除的合法尺寸，默认以H3的
16像素/latent合同计算；`best_aspect`只在最邻近的四组合法尺寸中选择宽高比误差最小者，并把
请求尺寸、实际尺寸、X/Y实际倍率和比例误差写入JSON报告。任意小数倍率通常无法同时满足“精确
倍率、精确原比例、宽高均32整除”，因此节点不会虚构零误差。H3联合AV latent只缩放视频空间轴，
音频tensor与时钟保持原样；普通SD/SDXL latent需显式改选8像素/latent。

`1.21.1` 完整保留此前90个节点的 ID、顺序、输入和稳定 `sampling.py` 数学，只在末尾追加的
4个Face Refine Advanced节点内补齐自动检测。Plan默认使用固定哈希的YuNet真人CPU路线；纯动漫
可显式切到隔离的ONNX Runtime EXP路线，手工ROI和本地Ultralytics仍保留。检测器对象在Plan结束
后销毁，但OpenCV/ONNX Runtime的进程级CPU分配器可能保留已热身页面，因此报告不承诺RSS回到
第一次执行前。规划先检测硬切并逐镜头重置轨迹和平滑，
记录边缘钳制后的真实脸部偏移、源画面代理哈希和完整transform合同。Conditioning只替换联合
AV latent的视频流，帧数、空间latent或17n+5时钟不一致即拒绝，锁定音频tensor和noise mask
原样复用。Sampler复用稳定双时钟实现，只在隔离模块中截取低噪声schedule；Stitch默认在CPU
生成候选，异常帧及相邻帧回退原图，并验证mask外像素逐位不变。全部输出都只是待人工审核的
候选，不覆盖原片、不自动接受、不改变最终背景音乐/音效，也不承诺身份正确、所有远景脸都会
改善、16GB普遍安全或低去噪数值已经标定。多镜头默认拒绝单次H3处理，应先拆成镜头内窗口。

`1.18.0` 完整保留 `1.17.0` 的62个节点ID、顺序、输入、默认值和旧接线，只在末尾追加24个
Advanced/Experimental节点。`Environment Audit`只读报告当前 H3 core、已知修复、wrapper、
DynamicVRAM和请求负载；`MLP Activation Chunk`只在克隆MODEL上分块token-local MLP，attention
不变且遇到已有`dit/double_block`所有者立即拒绝；`Qwen Reference Prefix Cache`只缓存严格视觉
参考因果前缀的CPU KV，prompt后缀仍逐次计算、音频-only自动旁路、默认`report_only`。

Studio层增加可审查 Context IR、可视时间轴、已接受长视频manifest上的非破坏性repair overlay，
以及按文件流式处理的Reel Delivery。Repair和Reel只在显式接受/确认后写新文件，不改原片。
同进程Trajectory Probe只接受稳定双时钟Euler并拒绝wrapper/`patches_replace`；Scheduled Audio
Injection首轮真实A/B没有消除尾部额外语音，因此保持默认旁路，不能宣传为“闭嘴修复”。
AV Decode Safety默认只预检，不把当前空闲显存或输出tensor估算伪装成VAE峰值预测。

`1.18.1` 不增加或改动节点接口，只加固 Reel Delivery 的异常边界：音频阶段先写临时文件，
完成样本数与峰值校验后才原子替换正式阶段文件；同一项目通过进程级OS锁串行；FFmpeg或宿主
进程被强杀后，下一次同项目执行只复用哈希验证通过的阶段，并在锁内清理本节点命名空间中的
孤儿临时文件。它不会清理其他项目或其他节点文件，也不把本机Windows结果外推为跨平台保证。

`1.18.2` 同样不改节点接口或旧工作流。验证发现MP4默认`movie_timescale=1000`会把非整毫秒
成片的AAC逻辑流时长量化：58帧/48kHz计划应为116,000采样，实际被写为115,968。最终封装现
使用`LCM(24fps, sample_rate)`时间标尺，并在原子替换前从容器头严格复核视频帧边界和音频
逻辑采样边界；32k/44.1k/48k三档均恢复为0采样误差。AAC解码器仍可能输出尾部padding，
因此这证明的是MP4逻辑流时长准确，不是有损AAC逐样本无损。
同轮1920×1088文件级探针还发现PyAV/libx264自动多线程会偶发进程崩溃或生成带参考帧/CABAC
错误的H.264流；约2MP及以上的Reel视频阶段现限定`threads=1`，低分辨率路径仍为自动线程。
高分辨率临时视频和最终mux临时文件在各自原子替换前都必须通过FFmpeg单线程、`-xerror`、
`err_detect=explode`完整严格解码；验证策略以`ffmpeg_single_thread_xerror_v2`持久化。升级时若
旧phase缺少单线程、严格解码或精确策略标记，会自动作废并重编码，不复用证据不足的码流。
本机两个FFmpeg 7.1 Windows构建的自动解码线程对同一1080p码流会随机失败；单线程FFmpeg及
PyAV/libavcodec 62重复解码均通过。因此这里只证明该版本化单线程验证合同，不外推其他平台或
播放器的多线程解码器。
另在Ubuntu 24.04.4 WSL2的Linux内核与ext4 `/tmp`上，使用Linux FFmpeg 7.0.2、PyAV 18.1.0
直接执行同一份Reel产品代码：两段H.264/AAC加一条FLAC lane得到精确66帧、132,000个48kHz
逻辑采样，第二次执行复用视频/音频阶段且输出SHA-256一致；来源哈希不变、无孤儿临时文件，
真实POSIX锁竞争被拒绝并在持锁进程结束后可重新取得。该结果关闭一条Linux/POSIX低分辨率
机械路线，不等于裸机Linux、macOS、高分辨率、任意FFmpeg构建或跨GPU通过。

验证期间ComfyUI先更新到`v0.32.0-15@86aedfd9`，Llama/Qwen新增合并投影、固定KV、预取和原地
残差路径。Qwen缓存现显式在无梯度推理上下文执行，并把直接调用的`TransformerBlock.forward`
也纳入精确源码合同；旧`cbbc9dab1`合同继续保留。当前core的CPU全前向/前缀KV数学探针和真实
32B NVFP4同进程OFF/HIT均通过，但真实配对最低余量只有116.998/337.583MiB，结论仍是兼容而非
无损、显存优化或16GB安全。
随后`v0.32.0-16@ddbaa8752`把MiniMax投影格式检测前移；Qwen前向合同未改变。447项完整回归、
精确源码合同、tiny-Llama等价探针及原生48帧视频参考完整A/V配对再次通过：OFF/HIT为
25.266/15.578秒，110.744MiB条目真实命中，视频SSIM均值/最低0.950934/0.944633，音频相关
0.953029；整卡余量仅344.340/338.833MiB，512MiB安全门仍失败。

Studio规划本身不会加载模型、自动排队、覆盖已接受媒体或宣称其他后端原生支持H3音频。
超过单段362帧的纯视觉镜头可显式拆段；带精确对白的长镜头拒绝自动断句，要求创作者先按
语言语义划分。所有新节点未连接时旧工作流行为不变；当前仍不授予普遍`memory_safe`、
bit-exact、精确语音时序或质量提升声明。

`1.17.0` 保留 `1.16.0` 的61个节点ID、顺序、输入、默认值和旧接线，只在末尾追加一个
`Hybrid Compatibility Audit Advanced`节点。它放在所有MODEL补丁与采样设置之后、`BasicGuider`
之前，默认`report_only`并返回同一个MODEL对象；检查Hybrid offset-set、LoRA顺序/AdaLN重叠、
Block Cache、Sage、Long Video、多关键帧、采样协议以及VRAM/VBAR policy provenance。旧工作流不
创建该节点时行为完全不变。机械通过不等于质量胜出或显存安全，报告始终保持
`memory_safe_claim=false`。

`1.16.0` 保留 `1.15.1` 的60个节点ID、顺序、输入、默认值和旧接线，只在末尾追加一个
Hybrid Artifact Maintenance Advanced输出节点。默认`inspect_only`不创建事务目录、不移动文件；
所有变更都要求`confirm_action=true`和正整数`operation_epoch`。节点只处理由严格Hybrid plan
推导出的`models/h3_hybrid_artifacts`内容寻址文件，使用可恢复隔离区与原子事务日志，不扫描或
删除diffusion checkpoint，也不负责卸载MODEL或释放显存。

`1.15.1` 保留 `1.15.0` 的60个节点 ID、顺序和旧接线；只把新VRAM策略节点的固定值建议与
示例从未过512MiB门槛的2.0GiB改为三冷三暖通过的4.0GiB，并把固定模式示例改为不执行全局
卸载。旧工作流没有该节点或不连接policy时，行为仍完全不变。

`1.15.0` 保留 `1.14.0` 的59个节点 ID、顺序、默认值和旧接线，只在末尾追加1个字面以
`Advanced` 结尾的 VRAM/VBAR 策略节点；Hybrid Loader 仅在末尾新增可选 policy 输入，旧工作流
不连接时不会清理模型、修改全局预留或改变 stock loader 行为。

`1.14.0` 保留 `1.13.0` 的56个节点 ID、顺序、默认值和旧接线，只在末尾追加3个字面以
`Advanced` 结尾的 Hybrid Model EXP 节点；未连接时不会检查第二份 checkpoint、构建 artifact、
克隆 MODEL 或改变任何旧工作流。`1.13.0` 保留此前54个节点的 ID、顺序、默认值和旧接线，只在末尾追加2个字面以
`Advanced` 结尾的多关键帧 EXP 节点；旧工作流不会创建中间关键帧、克隆 MODEL 或修改 H3
内部条件行。只有用户主动接入 Advanced 计划时才启用局部能力探针和补丁。`1.12.0` 保留此前51个节点的 ID、顺序、默认值和旧接线，只在末尾追加3个完全隔离的
Dialogue Safe Audio EXP 节点；旧工作流不会自动运行 ASR、改变联合 latent、分离混合音轨或
替换最终音频。`1.11.0` 保留此前48个节点的 ID、顺序、默认值和旧接线，只在末尾追加3个完全隔离的 Source AV
EXP 节点；旧工作流不会自动读取来源视频、改变 latent 或启用重绘 mask。`1.10.0` 保留此前36个节点的 ID、顺序、默认值和旧接线，只在其后追加12个语音可靠性/
创作 EXP 节点；旧工作流不会自动启用异常全局卸载、持久音色库、长文本或 Joint 多人。
`1.9.0` 只在原35个节点之后追加一个视觉参考强度 EXP 后置节点；原节点 ID、顺序、默认值和
旧工作流接线不变，只有用户主动插入节点时才写入实验字段。`1.8.0` 只在原25个节点之后追加10个实验性语音节点；旧节点顺序不变，原工作流不会自动进入
语音链。`1.7.0` 只在原23个节点之后追加 Background Start 与 Auto Accept & Continue；旧节点顺序不变。
`1.6.0` 在此前22个节点之后追加一个总时长编排节点；`1.5.0` 的四个候选/接受/合成节点与
`1.4.0` 的四个手工分段节点继续保留。原有14个节点的 Node ID、
schema 顺序和数值路径保持不变。`1.3.3` 在稳定双时钟节点末尾追加可选的采样器与调度器下拉框；原有三个控件的顺序、
默认双时钟 Euler、原生 flow sigma 和旧 API 缺省行为均保持不变。`1.3.2` 保留 `1.3.1`
对两代 H3 采样协议的兼容：旧版 ComfyUI 的 slope-scaled 音频速度，以及当前
`FLOW_AV` / `ModelSamplingAV` 的原始音频速度。兼容性由实际 H3 基模能力检测，不依赖用户
手动选择，也不会对新版 ComfyUI 再次应用音频 carry/scale。本版本还兼容
VideoHelperSuite 的延迟 `AUDIO Mapping`，用 H3 latent 契约识别视频/音频 VAE，并把画布
像素面积上限放宽到 `1920×1088 = 2,088,960`；超过旧 0.98M 档只提示显存风险，不再阻止执行。
当前插件随后在 H3 初始期 ComfyUI `0.30.0@563b98eef` 和当前 `cbbc9dab1` 上使用同一
256×256×22、一阶、同seed工作流完成真实配对：22张PNG逐字节一致，32kHz双声道音频相关
0.999688、SNR 36.12dB、最大差一个16-bit PCM步长。该证据只覆盖稳定
`dual_clock_euler`默认路线。另用当前`1.18.2@c7f5080`插件在同一旧版快照做了完整导入/schema
探针：86/86个插件节点和其中24/24个Advanced节点均出现在`/object_info`，没有导入traceback。
随后再真实执行Trajectory Advanced的256×256×22四步full、2+2 split和save/load/resume：3/3
prompt成功，两份最终checkpoint逐字节一致。三段最低整卡余量仅94.153MiB，未过512MiB门槛。
Environment Audit、Studio/Prompt/Repair规划、本地Context IR和Reel只规划不写入的4张API图也全部
执行成功，覆盖另外12个Advanced节点；Qwen Prefix Cache先通过report-only、`memory_lru_exp`包装安装与
Stats，随后又以48帧真实参考视频完成OFF/MISS→HIT及完整A/V对照：缓存为1 miss+1 hit，HIT较OFF
快21.80%，但输出非逐位一致（视频SSIM均值0.950934、音频相关0.953028），且最低余量334.508MiB，
所以只证明该短链兼容，不证明无损或16GB安全。AV Decode Safety真实解码成功22帧
和音频，而Activation Chunk在应用前正确拒绝旧H3未知源码合同。Repair的Bind/Stage/Accept/Compose
四节点使用已持久化真实链执行成功：`accept_repair=false`，原manifest及27个accepted素材哈希保持不变，
只在忽略的验证目录生成一个base-rollback成片。随后隔离的两段fixture又完成`accept_repair=true`与
repair-overlay合成：替换索引1、保留索引0、原manifest/accepted素材不变、输出44帧；AAC流时长比
58,667逻辑样本多21样本且解码含padding，所以不宣称sample-exact或真实H3修复质量。
Scheduled Audio Injection的默认`report_only`与实际
`scheduled_injection`路线均完成真实256×256×22一阶生成并输出22帧和音频；两者最低余量分别只有
15.685MiB和97.335MiB，且这不证明注入能抑制多余说话。
至此新增的24/24个Advanced节点都在该精确旧core上取得了执行或明确fail-closed证据，但证据强度按
路线受限：Qwen只测一条短视频参考、Activation拒绝应用、Repair接受仅为隔离fixture、Scheduled
质量结论仍为否决。它不代表
所有模式、任意旧版本或16GB安全。

## 项目目录

| 路径 | 内容 |
|---|---|
| `tools/` | MiniMax H3 Turbo LoRA 转换工具 |
| `docs/` | LoRA 使用说明与验证记录 |
| `examples/` | API 与 ComfyUI 前端工作流 |
| `artifacts/` | 历史发布包和代码迁移归档；已由 `.gitignore` 排除 |

项目源码、文档、工具和本地交付资产均以当前项目目录为唯一事实源，不依赖其他盘符
中的工程副本。模型权重不存放在本项目中，应继续使用 ComfyUI 的标准模型目录。

## 节点

| 节点 | 用途 |
|---|---|
| MiniMax H3 Audio Conditioning (T8) | T2VA、I2VA、FL2VA、L2VA、Ref2VA 和关键帧+参考媒体 Hybrid 的统一条件节点 |
| MiniMax H3 Audio Latent Control (T8) | 对已有 H3 AV latent 锁定或重绘源音频，并保留已有视频 mask |
| MiniMax H3 Duration Planner (T8) | 把场景时间换算成 24fps、`17n+5` 的渲染窗口和最终裁切参数 |
| MiniMax H3 Audio Window (T8) | 直接切取/补零 AUDIO，短场景可自动扩展到 124 帧训练下限 |
| MiniMax H3 Prompt Tags (T8) | 把 `Image 1`、`Audio1` 等写法规范为官方标签并严格校验编号 |
| MiniMax H3 AV Decode (T8) | 用视频/音频 VAE 分别解码联合 AV latent |
| MiniMax H3 Audio Mix (T8) | 源音轨与模型生成音轨重采样、增益、ducking、峰值限制后混合 |
| MiniMax H3 Output Trim (T8) | 把 Planner 的时间窗口同时应用到解码帧和音频 |
| MiniMax H3 Preflight (T8) | 在采样前检查模型、尺寸、帧数、音频、参考数量和参考视频时长 |
| MiniMax H3 Dual-Clock Sampler (T8) | 默认配置 12/3 shift、原生 flow sigma 与双时钟 Euler，也可选择当前 ComfyUI 的原生采样器和调度器 |
| MiniMax H3 Multi-Rate Sampler (EXP/T8) | 实验性视频宏步/音频微步采样；独立实现，不替换稳定双时钟节点 |
| MiniMax H3 Reference Image Edit (EXP/T8) | 用 Ref2VA 对单张主图进行语义编辑，并支持最多 8 张附加参考图 |
| MiniMax H3 Still Preflight (EXP/T8) | 检查单帧 OOD、画布、参考数量、模型和 VAE 契约 |
| MiniMax H3 Still Decode (EXP/T8) | 只解码视频 latent，并从 1/5/22/124 帧候选中选出一张图 |
| MiniMax H3 Segment Planner / 长视频分段规划 (EXP/T8) | 计算当前段渲染帧、重叠裁头、有效时长、绝对时间和是否允许保存下一段上下文 |
| MiniMax H3 Previous Context / 读取上一段上下文 (EXP/T8) | 第0段返回空上下文，第N段只读取并校验固定的 N-1 状态文件 |
| MiniMax H3 Long Video Conditioning / 长视频续写条件 (EXP/T8) | 合并运动尾部、音频 timeline、原有关键帧/参考媒体，并只给克隆 MODEL 加局部补丁 |
| MiniMax H3 Save AV Tail / 保存下一段上下文 (EXP/T8) | 只保存最多39帧所需的 CPU AV latent 尾部，校验哈希并原子替换当前段槽位 |
| MiniMax H3 Save Candidate / 保存候选片段 (EXP/T8) | 把当前裁后 A/V 和有界 latent tail 原子写入候选目录，不修改已接受历史 |
| MiniMax H3 Review & Accept / 预览并接受候选 (EXP/T8) | 默认只预览；确认后提交 manifest，替换中间段时使所有依赖后段失效 |
| MiniMax H3 Accepted Context / 读取已接受上下文 (EXP/T8) | 只按 manifest 读取 N-1，并输出父候选 ID/修订号防止陈旧续接 |
| MiniMax H3 Compose Accepted / 合成已接受片段 (EXP/T8) | 校验每段哈希后流式重编码；内存上限为单帧视频加单段 PCM，不聚合整条 tensor |
| MiniMax H3 Chain Orchestrator / 总时长自动分段 (EXP/T8) | 把总时长量化为固定内部窗口的完整时间轴，按 accepted manifest 自动定位下一段，并提供分段 prompt、seed、进度与完成阻断 |
| MiniMax H3 Background Start / 后台长视频启动 (EXP/T8) | 在模型执行前登记当前 prompt；显式启用自动接受、失败重试和段间释放策略，安全默认仍为 `review_only` |
| MiniMax H3 Auto Accept & Continue / 自动接受续跑 (EXP/T8) | 接受当前候选、可选合成最终 MP4，并且一次只校验和排入一个下一段 prompt |
| MiniMax H3 Voice Profile / 音色档案 (EXP/T8) | 建立描述音色或经授权的内存参考音色；规范化真实剩余时长并输出质量报告 |
| MiniMax H3 Speech Plan / 语音规划 (EXP/T8) | 按语言分句，把实际台词与演绎方向严格分离为可复现段计划 |
| MiniMax H3 Speech Conditioning / 语音条件 (EXP/T8) | 描述音色走 T2VA，参考音色走 Ref2VA；复用外部 MODEL/CLIP/双 VAE，不重复加载 |
| MiniMax H3 Speech Decode / 语音解码 (EXP/T8) | 只解码联合 latent 的音频部分，可选保守能量裁边 |
| MiniMax H3 Speech Verify & Align / ASR校验裁切 (EXP/T8) | 可选 CPU ASR、完整目标顺序定位、说话人余弦报告与最终衰减式峰值保护 |
| MiniMax H3 Speech Assemble / 语音合成 (EXP/T8) | 以绝对 sample 时间轴合并多段、停顿、重叠、淡化、声像和增益 |
| MiniMax H3 Dialogue Script / 对白脚本 (EXP/T8) | 把2–3个角色的普通文本或 JSON 变成逐 turn 对白计划 |
| MiniMax H3 Dialogue Turn Select / 对白段选择 (EXP/T8) | 逐 turn 选择正确角色档案和单段语音计划，避免把未验证联合多人模式当稳定能力 |
| MiniMax H3 Dialogue Boundary Analyzer / 对白边界分析 (EXP/T8) | 本地 CPU ASR 仅在出现唯一、连续、精确目标词序列时报告边界；重复目标或被插话打断时拒绝猜测 |
| MiniMax H3 Dialogue Safe Master / 对白安全混音 (EXP/T8) | 把已验收的独立对白 stem 与独立音乐、环境和 SFX 组成完整时长母带，不在对白结束处截断背景声 |
| MiniMax H3 Timed Background Bed Lock / 分时背景底轨锁定 (EXP/T8) | 两遍 H3 路线：用独立完整背景底轨替换音频 latent，并在显式对白边界后锁住底轨尾段 |
| MiniMax H3 Speech Finalize / 语音完成与释放 (EXP/T8) | AUDIO 直通后执行 keep/cache-clear/全局卸载策略，并明确报告作用域 |
| MiniMax H3 Speech Studio / 语音工作台 (EXP/T8) | GraphBuilder 一站式串起条件、stock采样、音频解码、校验和释放 |
| MiniMax H3 Speech Abnormal-Exit Guard / 异常释放保护 (EXP/T8) | 在条件节点前登记 prompt 生命周期；异常、取消或非 OOM 错误令 Finalize 未执行时补发释放请求 |
| MiniMax H3 Speech VRAM Preflight / 显存预检 (EXP/T8) | 报告当前整卡空闲、PyTorch占用和 DynamicVRAM 配置；只做当前态门槛，不授予 `memory_safe` 标签 |
| MiniMax H3 Voice Library Save / Load / Delete (EXP/T8) | 显式保存/读取经授权音色；不允许默认同名覆盖，删除移到可恢复回收目录 |
| MiniMax H3 Speech Performance Direction / 演绎控制 (EXP/T8) | 添加情绪、语速、音高、能量和非语言提示方向；当前标定矩阵未通过，只能作 prompt EXP |
| MiniMax H3 Speech ADR Exact Fit / 配音精确时长 (EXP/T8) | 拒绝、补裁或有界相位声码器变速，并可确定性移调；输出精确到 sample，不声称口型同步 |
| MiniMax H3 Speech Long Form Start/Accept/Control/Compose (EXP/T8) | 原子 accepted manifest、断点恢复、合作式取消、分段可播放预览、哈希校验和最终精确时间线合成 |
| MiniMax H3 Joint Dialogue Conditioning / 多人同段条件 (EXP/T8) | 2–3人 Ref2VA 同段实验；真实两人探针质量门槛失败，不是稳定推荐路径 |
| MiniMax H3 Keyframe Plan / 中间关键帧计划 (Advanced) | 链式加入一个中间图像，使用帧/秒/百分比定位，并记录该帧的原始视觉 `noise_aug` |
| MiniMax H3 Multi-Keyframe Conditioning / 多关键帧条件 (Advanced) | 在独立 MODEL 克隆上组装首帧、1–7张中间帧、尾帧及可选 Hybrid 参考；不修改稳定 Conditioning |
| MiniMax H3 Hybrid Pair Inspector / 混合模型配对检查 (Advanced) | 在分配GPU前校验精确FL2VA/Ref2VA pruned pair、完整SHA-256、曲线、tensor合同、recipe和预计artifact大小 |
| MiniMax H3 Hybrid Artifact Builder / 小型混合补丁构建 (Advanced) | 把Ref2VA所选AdaLN模态行曲线重基到FL2VA基底，原子生成约13.84～83.06MiB的内容寻址target-slice artifact |
| MiniMax H3 Hybrid Model Loader / 混合模型加载 (Advanced) | 继续使用ComfyUI stock diffusion loader加载FL2VA，再给克隆MODEL应用小型offset-set patch；保留DynamicVRAM/VBAR路径 |
| MiniMax H3 VRAM Policy / VBAR显存预留策略 (Advanced) | 默认只报告；显式连接Hybrid Loader后在模型加载前设置ComfyUI总预留与AIMDO simple headroom，并报告主机commit边界 |
| MiniMax H3 Hybrid Artifact Maintenance / 混合补丁安全维护 (Advanced) | 默认只检查精确artifact状态；可显式隔离、还原、恢复中断事务或处理过期构建残留，永不永久删除源checkpoint |
| MiniMax H3 Hybrid Compatibility Audit / 混合模型组合兼容审计 (Advanced) | MODEL原样直通；审计Hybrid/LoRA/BlockCache/Sage/长视频/多关键帧/采样/VBAR组合，默认只报告、可选阻断硬冲突 |
| MiniMax H3 Visual Reference Strength (EXP/T8) | 在现有 H3 positive Conditioning 后写入全局视觉参考强度；可能缓解过度平滑/蜡感，也可能削弱身份、构图和首尾帧 |
| MiniMax H3 Source Media Window / 来源视频窗口 (EXP/T8) | 把已有IMAGE/AUDIO按24fps、`17n+5`和32kHz裁为严格同步的短窗口；不是文件流式解码 |
| MiniMax H3 Source AV Prepare / 来源音画重绘准备 (EXP/T8) | 严格校验并组装视频/音频latent，保留元数据与mask，显式处理时钟差并提供双流lock/remix/regenerate |
| MiniMax H3 AV Latent Separate / 联合潜空间拆分 (EXP/T8) | 不调用VAE即可校验、拆出H3视频/音频latent并保留各自mask |
| MiniMax H3 Environment Audit / 环境兼容审计 (Advanced) | 只读检查core修复、wrapper归属、DynamicVRAM、当前显存/主机状态和请求负载；默认只报告 |
| MiniMax H3 MLP Activation Chunk / MLP激活分块 (Advanced) | 仅在克隆MODEL上分块token-local MLP；attention不变，未知core或已有double-block owner时拒绝 |
| MiniMax H3 Qwen Reference Prefix Cache / 参考前缀缓存 (Advanced) | 有界CPU LRU复用完全相同的视觉参考因果前缀；prompt仍重算，音频-only自动旁路 |
| MiniMax H3 Qwen Prefix Cache Stats / 前缀缓存统计 (Advanced) | 报告命中、未命中、条目数、CPU MiB和超限拒绝，不写磁盘 |
| MiniMax H3 Unified Cast / 统一角色表 (Advanced) | 把角色身份、服装、行为规则、禁止变化和参考槽位编译为确定性文本合同，不加载识别模型 |
| MiniMax H3 Sound Canvas / 声音画布 (Advanced) | 在绝对时间上规划对白、音乐、环境和SFX；对白结束后只禁止额外语音，不截断完整声音床 |
| T8 Video Prompt Compiler / 多后端提示词编译 (Advanced) | 输出H3、Wan 2.2、LTX-Video或通用视觉提示包；非H3音频始终作为sidecar，不虚构原生支持 |
| MiniMax H3 Studio Timeline / 创作时间轴 (Advanced) | 将多镜头量化为独立`17n+5`窗口、确定性seed和绝对时间；长视觉镜头可显式拆段 |
| MiniMax H3 Studio Shot Select / 镜头选择 (Advanced) | 从时间轴输出单镜头prompt、negative、length和seed，连接现有Conditioning/Sampler |
| MiniMax H3 Selective Segment Repair / 选择性分段重做 (Advanced) | 根据显式索引或质量证据生成非破坏性重做列表，不删除、覆盖或自动接受媒体 |
| MiniMax H3 Repair Segment Select / 重做段选择 (Advanced) | 输出一个重做项的prompt、length、seed与策略；最终接受仍由用户或既有manifest节点完成 |
| MiniMax H3 Selective Repair Bind/Stage/Accept/Compose (Advanced) | 把重做项绑定到不可变manifest修订，暂存、显式接受为独立overlay并可回退到原成片 |
| MiniMax H3 Scheduled Drive Audio Injection (Advanced) | 对完整驱动音频latent做计划式重复锚定；默认旁路，不能只控制语音或保证消除额外念叨 |
| MiniMax H3 AV Decode Safety / 音视频安全解码 (Advanced) | 检查联合latent、VAE角色、时间网格、有限值和当前资源；默认只预检，tiled路径单独标EXP |
| T8 Context IR Provider / 参考语义理解 (Advanced) | 默认只校验本地IR；外部视觉服务需显式确认，仅上传抽样JPEG和用户文本，不上传原始音频 |
| T8 Context IR Prompt Compiler / IR提示词编译 (Advanced) | 将已审查IR编译进既有Prompt Packet；远端不能控制路径、模型、节点、采样或改写精确对白 |
| MiniMax H3 Reel Delivery Plan/Compose / 成片交付 (Advanced) | 指纹化24fps文件片段、有限crossfade和音频lane，显式确认后以有界内存原子重编码MP4 |
| MiniMax H3 Trajectory Probe / 采样轨迹探针 (Advanced) | 只拆稳定双时钟Euler sigma，绑定同进程MODEL/SAMPLER，并拒绝wrapper与patches_replace |
| MiniMax H3 Trajectory Checkpoint Save/Load / 轨迹保存续跑 (Advanced) | 显式保存联合latent并校验完整合同；第二阶段必须使用Load输出的resume_noise，重启后不会伪装成同一模型栈 |
| MiniMax H3 Face Refine Plan / 远景脸修复规划 (Advanced) | 在原画幅24fps、17n+5窗口中用默认YuNet真人CPU检测、动漫ONNX EXP、手工ROI或本地Ultralytics做分镜内轨迹和平滑，输出源绑定crop计划与预览 |
| MiniMax H3 Face Refine Latent / 脸部二次生成条件 (Advanced) | 把crop序列精确编码进联合AV latent的视频流，复用锁定音频与mask，拒绝任何静默时空补齐 |
| MiniMax H3 Face Refine Sampler / 低去噪双时钟采样 (Advanced) | 复用稳定双时钟采样实现并截取低噪声sigma尾段；denoise仅是实验参数，不是已标定修复强度 |
| MiniMax H3 Face Refine Stitch Audit / 脸部回贴审计 (Advanced) | 按真实边缘偏移回贴ellipse/rect区域，颜色限幅、异常帧回退并证明mask外像素逐位不变；只输出待审候选 |
| MiniMax H3 Face Refine Parity Plan / 原版机制规划 (Advanced) | 隔离输出21/51 Gaussian轨迹、crop factor 3、最高768画布、最佳来源参考crop和完整审计报告 |
| MiniMax H3 Face Refine Parity Latent / 原版机制Latent (Advanced) | 仅替换联合AV latent的视频流，严格保留音频tensor和已有mask，不做静默trim/pad |
| MiniMax H3 Face Refine Per-Frame Denoise / 逐帧去噪 (Advanced) | 按脸尺寸逐帧生成视频noise mask，音频mask保持全零；默认0.8/0.35、30/120px、smooth9 |
| MiniMax H3 Face Refine Parity Stitch / 原版机制回贴 (Advanced) | 默认face-only矩形、dilation24、feather24、colour match1；mask外逐位不变，只输出待审候选 |
| MiniMax H3 Face Refine Quality Gate / 候选质量门 (Advanced) | 兼容保留的保守源片回退实验；源相似/锐度代理不能判断结构修复成功，不再作为推荐工作流出口 |
| MiniMax H3 Face Refine MANUAL512 REL Baseline / 人工验收512相对模式基线 (Advanced) | 严格校验manual512、crop2.5、relative 0.8/0.35、21/51、音频零mask和24/24回贴后原样放行候选；不宣称通用质量保证 |
| MiniMax H3 Face Character Profile / 多人角色参考 (Advanced) | 对清晰单人参考图运行YuNet+SFace CPU，建立执行期内存角色档案；无授权状态开关，相似度只作匹配建议 |
| MiniMax H3 Face Cast Merge / 2-3人角色表 (Advanced) | 合并2～3个唯一角色ID；拒绝重复ID且不把身份向量持久写盘 |
| MiniMax H3 SAM3.1 Multi-Person Track / 多人分色追踪 (Advanced) | 按镜头运行原生SAM31Tracker，输出2～3条分色轨迹并在结束后默认选择性卸载SAM |
| MiniMax H3 Face Track Assign / 轨迹绑定角色 (Advanced) | SFace一对一建议加逐镜头JSON覆盖；低分、低间隔或未绑定轨迹默认拒绝 |
| MiniMax H3 Multi-Face Repair Job / 单角色修复任务 (Advanced) | 为一个角色的一个镜头窗口生成source-bound、17n+5、默认MANUAL512的Parity修复计划；可选自动目标脸高，旧crop-factor模式保持默认 |
| MiniMax H3 Multi-Face Composite / 多人候选合成 (Advanced) | 顺序应用已审片候选，默认拒绝人物mask重叠并验证mask外像素逐位不变；音频不参与回贴 |
| Latent Upscale by 32 / 32整除潜空间放大 (T8) | 按显式8/16像素latent合同放大，输出宽高严格32整除；比例优先模式报告不可避免的残余误差，H3联合latent不改音频 |

最小可运行示例见
[`examples/workflows/2026-08-16_H3_Latent_Upscale_By32.json`](examples/workflows/2026-08-16_H3_Latent_Upscale_By32.json)。
示例使用ComfyUI普通`EmptyLatentImage`，所以显式选择8像素/latent；连接本包H3 Conditioning输出的
联合AV latent时应保留默认16。不要把已包含首尾帧/参考图空间条件的AV latent随意放大后继续沿用
旧尺寸Conditioning；新节点只负责latent几何与音频保持，不会自动重编码那些条件媒体。

`MiniMax H3 Audio Conditioning (T8)` 与 Long Video Conditioning 的 `task_type` 下拉框会显示中英双语说明：

| 选项 | 中文含义 |
|---|---|
| `auto` | 自动判断（按已连接输入） |
| `T2VA` | 文生音视频 |
| `I2VA` | 图生音视频（首帧） |
| `FL2VA` | 首尾帧生音视频 |
| `L2VA` | 尾帧生音视频 |
| `Ref2VA` | 参考生音视频 |
| `Hybrid` | 关键帧与参考媒体混合生成 |

中文仅用于前端显示，后端和 API 仍提交原有英文枚举，因此旧工作流与 API JSON 无需修改。

## Advanced：环境、低峰值复用与 Studio 创作层

### 1. 先审计，再决定是否启用实验优化

`Environment Audit Advanced`是只读输出节点。它检查当前ComfyUI H3源码合同、若干已知修复
是否存在、H3 wrapper归属、DynamicVRAM/VBAR状态、当前整卡余量和所填负载，但**不是峰值预测器**。
`status=pass`只代表没有发现已知硬阻断，不代表该配置必然不OOM。默认`report_only`不修改设置、
不卸载模型、不下载依赖，也不接管任何全局H3类。

审计报告还包含当前ComfyUI进程RSS/private/pagefile、page faults、累计磁盘读写、pinned-memory
开关/当前量和可选NVML温度、功耗、频率与热降频。单点计数只是当前快照，不能把累计270GiB读取
误判为本次任务；配套`tools/validate_h3_vram.py run`会对本地ComfyUI服务计算运行前后差分，并
保守分类为`fits / fits_with_thrashing / unsafe / unknown`。当前高读取筛查阈值为64GiB且不是
存储基准，也不把“能跑完”自动写成“可用”。

`MLP Activation Chunk Advanced`仅把每个H3 block中token-local MLP按行分块，attention仍处理完整
packed序列。节点只补丁克隆MODEL；当前如发现Block Cache或其他`dit/double_block`所有者会
fail closed，不覆盖、不叠套，默认`report_only`。真实256×256×22小链保持逐帧PNG与PCM一致；
随后FL2VA pruned INT8、736×416×124、1步受控A/B给出否决结果：chunk256冷态整卡峰值比baseline
高约288.32MiB，暖态只低约22.88MiB（未过128MiB实质差异阈值），耗时也无稳定收益。内核探针
同时确认当前TensorWise INT8已融合SwiGLU，未分块的大fc1激活代理不适用。因此该节点**不推荐用于
当前INT8路径省显存**，只保留给其他精度/后端的EXP研究；`memory_safe_claim=false`保持不变。

### 2. Qwen视觉参考前缀缓存

推荐连接为`CLIP Loader -> Qwen Reference Prefix Cache -> 原有H3 Conditioning`。同一组参考图/
参考视频但多次修改文字提示时，H3的Qwen输入具有严格“参考在前、prompt在后”的因果结构；节点
只缓存完全相同视觉前缀的每层KV与前缀最终hidden，prompt后缀仍逐次计算。缓存是有界CPU LRU，
默认1条/1024MiB、从不写盘；超过预算只拒绝保存该条，不无限增长。音频-only参考没有视觉计算，
会自动旁路；token权重、schedule/hooks、非原生H3 CLIP或未知core均旁路或拒绝。

真实ComfyUI小型Llama探针中，完整因果前向与“缓存前缀KV+新后缀”在`2e-6`容差内一致；本机
32B Qwen NVFP4已完成同参考/改prompt命中、两条LRU淘汰和64MiB超预算拒绝。一个
512×512×22、1步最终A/V对照中，命中为14.719秒、关闭缓存为19.985秒，但视频SSIM均值
0.9777、音频相关0.9581，并非bit-exact。随后三组全新进程冷配对和三组同进程暖配对中，命中端
每次都更快，配对平均分别约11.97%和11.01%；但六组结果仍不逐位一致，视频SSIM最低0.9246，
一组暖态音频相关仅0.2323，冷/暖最低整卡余量也只有75.63/168.08MiB，均未过512MiB门槛。
双图参考的冷/暖OFF对HIT短链随后都取得真实命中，60.70MiB缓存条目的耗时分别
下降6.04%/6.87%；视频SSIM均值/最低为0.91869/0.91130，音频相关0.95956，仍非逐位一致，
最低余量311.85MiB。另一条使用ComfyUI原生`LoadVideo`/`GetVideoComponents`读取48帧、2秒、24fps
真实视频参考的完整A/V配对也命中110.74MiB条目，耗时下降13.81%，峰值低166.31MiB；
但OFF/HIT仅余下145.15/311.46MiB，视频SSIM均值/最低为0.95093/0.94463，音频相关0.95303。
又完成了真人×机械、机械×城市角色、城市角色×真人三组双图组合×2 seed的同进程暖态矩阵：
6/6真实命中且HIT均更快，平均耗时变化-11.09%；视频SSIM的组均值0.9314、最低单帧0.8531，
音频相关均值0.9771。六组后进程private memory最大正向跳发为59.91MiB，未见256MiB阶梯，
但最差整卡余量仅111.93MiB，而且1步画面不足以做感知结论。
同三类素材各取1个seed的Stock20对照也已完成：3/3真实命中且HIT更快，平均耗时变化-5.00%；
但视频SSIM组均值只有0.8227，组间0.6790～0.9073、最低单帧0.6052；音频相关均值0.7188，
组间0.2603～0.9894，最低显存余量190.68MiB。这说明缓存数值差异会被完整扩散链放大，自动质量门不通过。
ComfyUI更新到`v0.32.0-15@86aedfd9`后，精确源码能力探针先确认tuple-KV路径在推理语义下仍与
完整因果前向等价，再完成一组真实32B NVFP4同进程受控配对：HIT记录1次命中/1次未命中，
108.283MiB条目，完整链13.297秒降到9.375秒；22帧SSIM均值/最低为0.951217/0.924603，
32kHz双声道音频相关0.956522。OFF/HIT最低余量仅116.998/337.583MiB，仍未过512MiB门槛。
当前`v0.32.0-16@ddbaa8752`又完成精确合同、完整447项回归及原生48帧视频参考的完整A/V复跑：
110.744MiB缓存真实命中，OFF/HIT耗时25.266/15.578秒；视频SSIM均值/最低
0.950934/0.944633，音频相关0.953029。OFF/HIT余量344.340/338.833MiB仍低于512MiB，且
输出仍非逐位一致。
因此默认仍为`report_only`：只能称精确短链的多图/视频参考机械路线已打通，不能称无损、显存优化、
16GB安全或固定加速；多素材自动重复已完成，Stock20非劣性尚需人类盲评但已有明显自动风险，跨GPU主机内存仍需验证。需要观察命中时可接
`Qwen Prefix Cache Stats`。

### 3. Studio：角色、声音、镜头和局部重做

建议顺序：

1. `Unified Cast`定义角色ID、稳定外观、默认服装和禁止变化；它只是文本合同，不做人脸识别。
2. `Sound Canvas`在绝对时间上列出对白、音乐、环境和SFX。开启
   `no_unrequested_speech`时，提示会明确要求对白结束后不再念叨，但继续指定的背景音乐、环境和
   音效；它不截断最终混合音轨，也不假装能从已有混音中分离stems。
3. `Studio Timeline`把镜头JSON编译成24fps、每段`17n+5`的H3窗口和确定性seed。超过362帧的
   纯视觉镜头可拆成连续parts；含精确对白的长镜头拒绝自动断句，避免破坏中文/多语言语义。
4. `Studio Shot Select`输出一个镜头的prompt、negative、length和seed，直接接现有Conditioning、
   RandomNoise与Sampler；它不隐藏加载器或采样器。
5. `Selective Segment Repair -> Repair Segment Select`只列出需要重做的镜头，支持手工索引、失败
   状态或预先给定阈值；不选中的片段不进入计划，节点不删除、覆盖或自动接受任何媒体。
6. 需要真正替换已接受长视频片段时，再连接`Repair Bind -> Stage -> Accept -> Compose`。Bind把
   重做项锁到原manifest修订和源文件哈希；Stage只预览验证；Accept默认false并写独立overlay；
   Compose可选择`base_rollback`忽略overlay，原manifest和原segment始终不变。
7. `Reel Delivery Plan -> Compose`处理已经落盘的24fps片段与对白/音乐/环境/SFX文件lane。
   当前要求同尺寸、精确24fps，crossfade是像素混合而非运动插帧；Compose默认不执行，开启后
   H.264/AAC重编码且不是bit-exact。CRF变化会强制重做视频阶段，完整哈希一致才会恢复旧阶段。
   本机Windows/NTFS已完成30分钟、50个独立路径片段、4类音频lane的机械压力测试：输出
   43,200帧和86,400,000个48kHz采样（精确1,800秒），来源哈希不变，重复运行复用已验证阶段。
   同时覆盖音频FFmpeg、最终mux FFmpeg及宿主Python进程强杀后的恢复；50个路径使用同一小型
   fixture的硬链接，因此证明的是时间线规模、内存/磁盘和恢复合同，不是多编码器/多素材质量。
   后续本机同一条实验成片又混用H.264/AAC、HEVC/MP3和VP9/Opus三种128×96、24fps合成素材，
   并叠加WAV/FLAC/Opus/AAC四类lane；结果精确132帧、计划264,000个48kHz采样和5.500秒流时长，
   来源哈希不变，二次执行复用视频/音频阶段且输出哈希稳定。这关闭了本机合成素材的编码多样性机械门；
   三段真实256×256×22帧H3素材又组成精确58帧/116,000采样成片；三段真人、机械龙、城市角色
   736×416×124帧H3素材组成精确348帧/696,000采样、14.5秒成片，12帧转场缓冲估算10.51MiB。
   两条真实H3成片均保持来源哈希、阶段复用和复跑输出稳定。由此关闭本机真实H3多素材与
   736×416吞吐门。相同三段真实H3内容另被确定性放入1920×1088画布做文件级交付；自动线程
   路径复现一次native crash和3/3可解码码流错误，限定单线程后3个独立项目按v2策略完成阶段与
   最终mux双重严格解码，阶段/最终SHA三次一致，成片仍为348帧/696,000采样，转场缓冲估算
   71.72MiB。两个本机FFmpeg 7.1构建的自动解码线程仍会随机失败，而单线程FFmpeg与
   PyAV/libavcodec 62重复解码通过；它只证明
   派生1080p的Reel机制，不证明H3原生1080p生成质量；非Windows文件系统/FFmpeg仍未验证。

Repair执行器另在隔离的14段/60秒已接受链上覆盖了6个进程强杀点，重试后原manifest与27个
已接受资产哈希均保持不变；一个真实H3第7段重做也成功写入overlay并合成为精确1440帧/
1,920,000 samples。不过当前重做不会自动级联重生成依赖它的后续段：该探针的进入边界接近原片，
退出边界SSIM却从原链约0.933降至0.804。候选复制与合成输出现在分别在精确目标命名空间的OS锁内
清理前次进程强杀留下的`.*.tmp`；重建的14段真实链六点强杀矩阵中，复制写半与音频合成中断的
orphan都在重试前被报告并清零，六项事务/哈希/恢复门全部通过。该结论只关闭受测Windows/NTFS
故障点的crash-clean，不解决上下文级联。因此相邻段必须单独复核；不能宣传
无缝局部重做，后续需增加级联重生成或明确的退出边界阻断。

`Prompt Compiler`还可输出`wan_2_2`、`ltx_video`和`generic_cinematic`文本包。除MiniMax H3外，
声音计划只保存在`audio_prompt` sidecar和JSON报告中，不能据此宣称相应后端原生生成音频。所有
编译器输出仍是生成式方向，不保证逐帧时间、逐字对白、角色一致性或最终质量。

`Context IR Provider`默认`validate_local`，只做schema与控制权边界校验。外部
OpenAI-compatible视觉请求必须同时选择外部模式并开启确认；API key只从指定环境变量读取，
最多上传32张显式抽样/缩小的JPEG与用户主动提供的transcript，原始音频不上传。远端返回中的
路径、URL、节点、模型、采样器、seed、steps和凭据会递归拒绝；精确对白必须逐字保持。

`Studio Timeline`执行后会在节点中显示只读彩色时间轴；这是前端预览，不改变工作流JSON、
seed、调度或生成状态。

`Trajectory Probe`只用于稳定双时钟Euler的同进程诊断。最终v2使用专用trajectory MODEL与Load
节点输出的`resume_noise`直接传递内部`x_sigma`，不再用`DisableNoise`重建中间状态。RTX 4060 Ti
16GiB、FL2VA pruned INT8、Qwen NVFP4和双H3 VAE的四步2+2验证中，736×416×124与
256×256×362的full/续跑最终video、audio latent均逐位一致，最大误差为0。124帧另完成3个
全新进程冷周期和3个同进程暖周期：18/18个full/split/resume prompt成功，6/6对最终checkpoint
逐位一致；暖态full峰值无阶梯增长。124重复矩阵最低整卡余量587.15MiB；362单次full最低
520.51MiB，只比512MiB门槛高8.51MiB，故仍不能宣传通用16GiB安全。124和362最终checkpoint
约4.10/2.66MiB；体积取决于空间与时间latent共同大小，帧数不能单独预测磁盘成本。暖态三组
`split+resume`总耗时均值约72.30秒，full约70.75秒，没有吞吐收益证据。所有`patches_replace`
仍拒绝，保存默认关闭，重启续跑与跨GPU均不支持。
当前ComfyUI `v0.32.0-16@ddbaa8752`又独立完成124/362两档的full、split和resume共6个真实prompt；
两档full与resume最终checkpoint SHA-256分别完全一致。124/362 full整卡余量为749.019/
548.502MiB；362仍只比512MiB门槛高36.502MiB，而且该档仅为256×256，因此不扩大16GiB安全声明。

`AV Decode Safety`真实128×128×22普通解码已输出22帧及finite 32kHz音频。源码/行为探针进一步
确认当前H3 Video VAE默认内部tile为256像素：大于该边界时，`decode_regular`也会进入内部空间分块；
公开`decode_tiled(...)`则委托普通解码并忽略传入的tile/overlap。因此当前core若缺少full-frame
维度与tile offset坐标合同，Advanced会把任一边大于256的普通或显式tiled解码报告为high-risk，
严格模式直接阻断。验证进程中曾直接把每个空间tile改为整图坐标：256×256单tile控制逐位一致，
但736×416的真人、线稿与平滑物体3/3重建均退化，平均SSIM下降0.0828、PSNR下降1.141dB，且
x/y接缝比没有一例改善，肉眼出现栅格和重影。这条直接修法已被否决，没有合入核心或稳定节点；
它也不能证明经过训练或不同架构的修复一定失败。当前仍不声称显式tiled更省显存或视觉等价。
计划式驱动音频注入的真实A/B同样给出负结论：基线额外语音约
2.26秒开始，完整强度处理后仍约2.10秒开始，因此它不是对白终止器；保留它只是为了受控研究完整
驱动音频latent重锚，默认旁路。

API与可导入前端示例：

- `tests/fixtures/api/environment_audit_advanced_api.json` / `examples/workflows/2026-08-13_H3_Environment_Audit_Advanced.json`
- `tests/fixtures/api/activation_chunk_advanced_api.json` / `examples/workflows/2026-08-13_H3_Activation_Chunk_Advanced.json`
- `tests/fixtures/api/qwen_prefix_cache_advanced_api.json` / `examples/workflows/2026-08-13_H3_Qwen_Prefix_Cache_Advanced.json`
- `tests/fixtures/api/studio_timeline_advanced_api.json` / `examples/workflows/2026-08-13_H3_Studio_Timeline_Advanced.json`
- `tests/fixtures/api/context_ir_provider_advanced_api.json` / `examples/workflows/2026-08-13_H3_Context_IR_Provider_Advanced.json`
- `tests/fixtures/api/selective_repair_execution_advanced_api.json` / `examples/workflows/2026-08-13_H3_Selective_Repair_Execution_Advanced.json`
- `tests/fixtures/api/reel_delivery_advanced_api.json` / `examples/workflows/2026-08-13_H3_Reel_Delivery_Advanced.json`
- `tests/fixtures/api/scheduled_audio_injection_advanced_api.json` / `examples/workflows/2026-08-13_H3_Scheduled_Audio_Injection_Advanced_EXP.json`
- `tests/fixtures/api/av_decode_safety_advanced_api.json` / `examples/workflows/2026-08-13_H3_AV_Decode_Safety_Advanced.json`
- `tests/fixtures/api/trajectory_probe_advanced_api.json` / `examples/workflows/2026-08-13_H3_Trajectory_Probe_Advanced_EXP.json`
- `tests/fixtures/api/face_refine_advanced_api.json` / `examples/workflows/2026-08-16_H3_Face_Refine_Advanced_EXP.json`
- `tests/fixtures/api/face_refine_anime_advanced_api.json` / `examples/workflows/2026-08-16_H3_Face_Refine_Anime_Advanced_EXP.json`
- `tests/fixtures/api/face_refine_parity_advanced_api.json` / `examples/workflows/2026-08-09_H3_Face_Refine_Parity_Advanced_EXP.json`
- `tests/fixtures/api/multiface_sam31_2person_advanced_api.json` / `examples/workflows/2026-08-17_H3_SAM31_2Person_Face_Refine_Advanced_EXP.json`
- `tests/fixtures/api/multiface_sam31_3person_advanced_api.json` / `examples/workflows/2026-08-17_H3_SAM31_3Person_Face_Refine_Advanced_EXP.json`

Face Refine示例故意不经过固定736×416的来源视频窗口：它要求输入已经是24fps且帧数严格满足
`17n+5`，直接保留原始宽高比，再把脸部crop送入512×512二次H3。默认真人示例使用固定哈希的
YuNet CPU检测；纯动漫另有`2026-08-16_H3_Face_Refine_Anime_Advanced_EXP.json`，明确使用动漫专用EXP模型。
两者都要求先审核Plan preview；手工ROI仍可在缺少检测依赖时使用，Ultralytics自带模型须由用户
自行确认许可和类别。示例最终用Conditioning的`mux_audio`
重新封装，因此H3第二遍生成的音频被丢弃，原背景音乐、环境声和音效不会随脸部回贴被截断。
若Plan报告硬切，Conditioning默认拒绝一次处理，应先按镜头拆开。先看preview与Stitch报告，再
人工选择是否接受；该示例没有身份识别，也不构成“远景脸一定修好”或16GB通用保证。

多人示例在上述单人Parity机制外增加SAM3.1人体轨迹和SFace身份建议。导入后依次完成：替换源视频与
2～3张清晰单人参考图、审核分色预览、按每个镜头修改
`manual_assignments_json`、分别检查每个角色的完整candidate window，最后才把对应Composite的
`accept_candidate`切为true。Composite默认全部为false，第一次排队只会保存源片，不会自动采用候选。
双人、三人画布各有1块顶部总览和4块就地NOTE，分别贴近参考图、SAM追踪、Repair/Denoise与
采样/Composite区域；建议按NOTE顺序从左到右检查，不需要再到画布末尾查参数。
镜头切换后`0:0`可能指向另一人，必须重新绑定；遮挡、重新入镜或过小脸也不能只信颜色。

当前单人推荐工作流与2/3人示例统一使用Turbo 8步；多人显式使用
`manual_512 + target_face_px=300`，单人沿用人工验收的`manual_512 + crop_factor=2.5`。300px是H3
裁剪画布内的目标脸高，不是来源视频凭空获得300px真实细节；如果输入本身经过放大或严重失焦，自动
放大只能保证重绘区域足够，不能单独保证恢复真实纹理。报告中的实际脸高和
`source_boundary_limited_frames`必须先审核。

每个角色分支都锁定源音频、顺序复用同一个H3模型，并在最终`CreateVideo`重新接回源音轨；因此不会
为了修某张脸截掉音乐、环境声或音效。若源视频确实没有音轨，应显式接一个同长度32kHz双声道
`EmptyAudio`作为锁定静音，不能把`native`误当成“无声等于已锁定”。当前2人和3人完整顺序合成均已
实跑，清晰双人素材又通过一次用户人工结构修复验收；跨镜头长期身份、模糊输入锐化和其他素材显存仍需
逐项目验收。

本机真实机械探针使用FL2VA pruned INT8、Qwen3-VL NVFP4、双H3 VAE、736×416×124、12步
低去噪双时钟和锁定原音频。手工ROI链12/12步完成、总耗时176.93秒；自动YuNet链同样12/12步
完成、总耗时171.855秒，整卡峰值15,814MiB/16,380MiB。自动输出严格解码为124帧24fps、
32kHz双声道，文件SHA-256为
`56df397c0789694ef9919da2593297785d8b0a2ca4e70439261154809b0526ca`。手工链也输出124帧24fps
H.264与32kHz双声道AAC。来源与输出解码音频均为164,864个双声道采样，相关系数0.997751，
差异来自AAC重新编码，不能称压缩码流或PCM逐位一致。该来源取自上一轮已被非等比横向拉宽的
盲测素材，所以本次仅证明真实H3链、回贴、封装和音频保留能完成，绝不作为脸部修复质量证据。

后续扩展验证改用画幅安全真人舞蹈来源：124帧完整链3冷3暖全部成功，冷/暖最低余量分别为
717.6/922.1MiB，执行后private spread为3.2/79.1MiB；单次362帧完整链在295.235秒完成，余量
1,176.2MiB、private峰值41,311.6MiB。另一个124帧真实群像中113帧有多脸，抽样轨迹在硬切后
重置并持续跟随同一灰衣人物；但362帧台球硬负例曾把墙灯和卡通图案当脸，且跟踪器没有身份
模型，所以仍不能自动接受或宣传多人不串脸。

5组受控双人交叉矩阵进一步确认：一般交叉和检测顺序变化未换人，但目标在交叉点遮挡3帧时，
轨迹切换到另一人并保持到结尾，31帧跟错。六个真实二次生成候选的脸区SSIM均值仅
0.479～0.539、Laplacian中位比0.549～0.684、动作差分相关均值0.358～0.407，没有自动非劣
信号；这些代理不测身份，也不能代替人类盲评。

固定阈值评估工具为`tools/evaluate_face_refine_yunet_wider.py`，计划/主机内存探针为
`tools/probe_face_refine_plan.py`，匿名评审包生成器为`tools/build_face_refine_blind_review.py`。
受控交叉审计为`tools/audit_face_refine_tracker_crossing.py`，候选代理汇总为
`tools/summarize_face_refine_candidates.py`。
详细分层召回、显存口径、多人限制与盲评状态见
[`docs/FACE_REFINE_ADVANCED_VALIDATION.md`](docs/FACE_REFINE_ADVANCED_VALIDATION.md)。

## Advanced：FL2VA × Ref2VA 小型混合模型实验

`1.14.0` 没有照搬“同时 mmap 两份完整模型再拼 state dict”的 loader。默认路线依次连接：

1. `Hybrid Pair Inspector` 对两份用户已有 checkpoint 做完整 SHA-256 与结构检查；
2. `Hybrid Artifact Builder` 只读取选定 Ref2VA AdaLN 行，生成小型内容寻址 artifact；
3. `Hybrid Model Loader` 用 ComfyUI 原生 loader 加载 FL2VA，再把 artifact 应用到克隆 MODEL；
4. 如需 LoRA，必须接在 Hybrid Loader **之后**；示例为隔离变量而使用 Stock20、无 LoRA。

P0 只接受以下精确文件，文件名相同但哈希不同也会拒绝：

- `minimax_h3_fl2va_pruned_int8_convrot.safetensors`：
  `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a`；
- `minimax_h3_ref2va_pruned_int8_convrot.safetensors`：
  `9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779`。

Builder 默认把 artifact 放在 ComfyUI 标准模型目录 `models/h3_hybrid_artifacts/`，不会覆盖源模型，
也不会生成另一份约19.5GiB的完整融合 checkpoint。默认
`blocks_25_49_video_audio_exp` artifact 为27.69MiB、100个精确offset-set操作；完整实物构建的
曲线拟合相对误差为`4.9343e-5`，最差已保存调制重构误差为`2.3021e-5`。当前 DynamicVRAM
实测仍产出原生`ModelPatcherDynamic`，普通clone、non-dynamic delegate与同设备deepclone均保留
patch和来源报告。

这些 recipe 都使用中性 `_exp` 名称。当前 ComfyUI 只有video/text/audio三种AdaLN tag，视觉
参考和目标视频共用video行，音频参考和目标音频共用audio行；所以静态替换不是“只改参考通路”，
也不能先验保证“FL2VA质量 + Ref2VA全部参考能力”。`base_only`是明确的stock FL2VA对照；
`header_only_exp`只作诊断，永远不能授权artifact构建。

Inspector 的 `auto_match_reference_modalities_exp` 可读取现有 Conditioning：额外图片/视频只选
video行，独立音频只选audio行，有声视频或图像+音频组合选video+audio行；仅有首尾关键帧、没有
额外reference或遇到未知reference类型时会拒绝，不会猜测“最佳配置”。这只是最小模态路由，
不是经过训练的质量自动选择。

真实Stock20顺序矩阵已覆盖视觉、音频和图像+音频参考，共15路成功生成。混合参考单案例中，
25～49 video+audio Hybrid的人脸余弦中位数为0.523，FL2VA/Ref2VA对照为0.449/0.443；说话人余弦
为0.868，介于FL2VA的0.467与Ref2VA的0.945之间，三路ASR均为零词错。这是值得继续多seed盲评的
Pareto候选，不足以宣布去油、身份更准或优于原生Ref2VA。精确显存轮询的最差余量只有41.34MiB，
因此所有Hybrid示例继续禁止标16GB `memory_safe`。可恢复矩阵工具会顺序释放模型、校验输出hash、
生成盲评包与`matrix_summary.json/csv`，并可选使用本地ASR、InsightFace和WavLM；不会自动下载模型。
详细记录见
[Hybrid Model Advanced 验证报告](docs/HYBRID_MODEL_ADVANCED_VALIDATION.md)。

## Advanced：Hybrid artifact安全维护

`MiniMax H3 Hybrid Artifact Maintenance (Advanced)`不接受任意路径，而是连接同一个严格
`Hybrid Pair Inspector`的plan，重新推导唯一内容寻址路径。安全默认工作流是：

1. `inspect_only + confirm=false + epoch=0`：只校验artifact/sidecar、内嵌manifest、SHA、锁、temp
   和已有事务，连内部事务目录都不会创建；
2. `quarantine_artifact_exp`：完整校验后，把artifact和sidecar移到同卷`_recycle`，每移动一个文件
   都原子更新并fsync事务日志；同一epoch重复执行只返回已完成状态；
3. `restore_quarantined_exp`：使用同一epoch把完整隔离对还原，活动路径已被占用时拒绝覆盖；
4. `recover_interrupted_exp`：当进程在两文件移动之间被终止时，按日志中逐文件SHA恢复到活动目录；
5. `quarantine_stale_build_residue_exp`：只处理该plan对应、超过年龄门槛且锁owner未被证明仍活着的
   孤立artifact/sidecar、build lock与匹配temp，隔离后Builder可安全重建。

它没有永久删除动作；符号链接、越界路径、非canonical manifest、损坏journal、哈希或大小不一致
均fail closed。隔离区仍占磁盘，用户应先审核再手动做最终清理。此节点不会清Comfy执行缓存、卸载
已加载MODEL或释放VRAM。安全说明见[Hybrid Artifact维护文档](docs/HYBRID_ARTIFACT_MAINTENANCE.md)。

## Advanced：Hybrid组合兼容审计

`MiniMax H3 Hybrid Compatibility Audit (Advanced)`应放在最终MODEL链末端：

```text
Hybrid Loader -> 可选LoRA -> 可选Sage -> Block Cache/Long Video/MultiKeyframe
              -> 稳定或EXP采样设置 -> Compatibility Audit -> BasicGuider
```

可选`positive`输入用于核对Long Video/多关键帧的MODEL与Conditioning是否成对，并读取真实参考
模态。默认`report_only`即使发现问题也不阻断，且输出与输入是同一个MODEL对象；显式选择
`block_hard_conflicts`才会在已证明的机械冲突上报错。节点检查：

- Hybrid attachment身份、fingerprint、recipe、payload和全部offset-set条目；
- `Hybrid -> LoRA`顺序，以及后加LoRA是否覆盖所选AdaLN行；
- H3 Block Cache首/末block replacement和两个wrapper是否完整；
- Sage是否覆盖全部H3 blocks，未知attention forward patch会拒绝；
- Long Video与多关键帧的局部patch版本、Conditioning配对和双向互斥；
- stock、稳定双时钟/native AV、EXP多速率或未知采样路由；
- Loader记录的VRAM策略是否真实应用，以及执行当下整卡空闲、DynamicVRAM和主机commit。

Hybrid Loader现在只把必要的策略应用事实作为轻量MODEL附件随clone传递，不携带完整遥测树。
`require_applied_vram_policy=true`会拒绝未连接策略或`report_only`策略。512MiB与16GiB是可配置的
当前态门槛，不是下一次denoise峰值预测。通过审计只表示已知patch合同没有硬冲突，不能证明某个
recipe去油、画质更高、参考更准、音频更好或16GB不会OOM；AdaLN重叠LoRA在该精确组合完成数值与
质量矩阵前会被严格模式阻断。

2026-08-13又对完整组合做了RTX 4060 Ti 16GiB实测：736×416、124帧、Stock20、固定seed、
27.69MiB Hybrid artifact、KJ H3 Sage、T8 Block Cache默认0.12阈值、4GiB policy和严格审计。
三个全新进程冷态与同一进程三个暖态共6/6成功，每次均真实命中`6/20`，CPU cache 117.7MiB；
最差整卡余量766.38MiB，暖态基线最大正向变化82.94MiB，744张PNG和6份FLAC在六次运行间
解码后逐字节一致。这只让该精确本机profile通过机械/重复性门槛。

同日又完成同栈、同seed的Cache OFF三冷三暖，以及同一热进程三组OFF/ON交错对照。暖态完整链
平均由169.93秒降至129.19秒，节省23.98%；采样器平均由146.24秒降至105.31秒，节省27.99%，
每次ON仍命中`6/20`。但ON不是无损：相对OFF，视频平均SSIM 0.8432（最低帧0.7577）、8-bit
MAE 10.37，音频相关系数0.9207、SNR 7.99dB。OFF一个冷态峰值仅余239.40MiB，也未过512MiB
门槛。因此这里只确认该精确profile有稳定性能收益；后续六组单评审盲评仅通过主观冒烟门，
跨GPU和通用16GB安全仍未证明，`memory_safe_claim=false`与`quality_validated=false`不变。

进一步扩展到真人、机械龙、城市屋顶超级英雄三类视觉素材、每类两个seed，共6组ON/OFF质量
对照。5组暖态公平性能对照的完整链提速22.05%～28.47%，均值24.35%；采样器提速
27.94%～33.05%，均值29.07%，命中范围6～7/20。自动质量差异并不稳定：6组视频平均SSIM
总体为0.7020，组间0.5192～0.9373，最低单帧0.4774；音频相关系数均值0.9329，范围
0.8792～0.9806，没有一组bit-exact。统一六组单评审盲评现已完成：画面6平、声音6平。
评审观察所有B组颜色略浅，但揭盲后前3组B=OFF、后3组B=ON，不对应固定处理；逐帧客观统计中
B仅1/6组平均亮度更高、4/6组平均饱和度略低，因此不能归因于`0.12`。该结果通过这个精确
profile的单评审主观冒烟门，但自动差异范围仍明显大于`0.08`，不能证明统计非劣或感知无损。
因此`0.12`最多记为性能优先EXP候选，`0.08`仍是质量优先候选；节点与旧工作流默认值不变，
`quality_validated=false`和`memory_safe_claim=false`继续保留。

随后对差异最大的机械龙/超级英雄先试`0.08`与`0.10`，再把较保守的`0.08`扩展到完整
三类×两seed矩阵。`0.10`在超级英雄音频上出现非单调退化，因此不作为推荐证据。`0.08`的
5组有效暖态性能对照完整链提速9.05%～14.38%（均值12.35%），采样器12.20%～18.39%
（均值15.10%），命中3～4/20。6组视频SSIM均值0.8598、组间0.6013～0.9840、最低帧
0.5294；音频相关均值0.9635、范围0.8883～0.9927。视频和音频代理指标均为6/6优于
`0.12`，但仍非bit-exact，难例差异仍大。OFF对`0.08`随机六组单评审盲评现已完成并揭盲：
画面为`0.08` 1胜、5平、0负，唯一明确偏好来自真人第1组；评审认为真人有轻微差异、动漫
基本不可辨。音频为OFF 1胜、5平、0负，但唯一明确偏好来自两边都近似静音的第6组，且整体
判断为差异很小。这个结果通过了该精确profile的单评审主观冒烟门，因此`0.08`继续作为
“质量优先候选”；它不是统计非劣或感知无损证明，没有修改旧工作流或节点默认值，也不能称
通用推荐或16GB安全，`quality_validated=false`仍保持不变。

匿名模型侧盲审又抽查每边6个时间点和每组最大差异帧：6/6组均未见黑帧或采样帧结构坍塌，
视觉偏好全部记平局；12条音轨均无削波，但未进行真人听音，且超级英雄seed 2两边都近似静音。
该低置信度检查只能排除部分明显失败，不能证明运动/听感不劣，不能替代人类盲评，也不改变
`quality_validated=false`或任何默认值。

示例：

- `tests/fixtures/api/hybrid_compatibility_audit_api.json`；
- `examples/workflows/2026-08-09_H3_Hybrid_Compatibility_Audit_Stock20_EXP.json`。

完整合同与issue code见
[Hybrid Compatibility Audit文档](docs/HYBRID_COMPATIBILITY_AUDIT.md)。

## Advanced：显存预留与 DynamicVRAM/VBAR 策略

`MiniMax H3 VRAM Policy (Advanced)` 是一份有类型的策略计划，不是独立执行的万能通配节点。
默认 `report_only` 只报告当前整卡空闲、PyTorch池、ComfyUI预留、AIMDO状态和主机commit余量；
只有把 `vram_policy` 输出连接到 `Hybrid Model Loader` 新增的可选末尾输入后，才会在 stock 模型
加载前应用。未连接时 Loader 的调用、输出MODEL和旧工作流路径不变。

提供两种显式实验策略：

- `fixed_total_reserved_exp`：设置一个总预留值；当前16GiB Hybrid Stock20示例使用本机验证过的
  4.0GiB保守起点；
- `external_usage_plus_margin_exp`：先显式全局卸载ComfyUI模型，再用整卡已用显存加用户余量计算，
  并受最大预留上限约束。为了不把ComfyUI缓存模型误当成外部程序占用，此模式强制要求
  `clean_before_load=true`。

本实现与 `ComfyUI-ReservedVRAM` 的产品方向相同，但没有复制其代码，也没有调用
`aimdo_control.init(...)`。本机 `comfy-aimdo 0.4.13` 的 `init` 在库已加载时还会写入
`nvml_pressure`；本节点只调用底层 `lib.set_simple_vram_headroom(...)`，同时更新当前ComfyUI的
总预留。它不会再次初始化设备，也不会修改启动参数 `--vram-headroom`。

预留显存能让ComfyUI/VBAR更早减少GPU权重驻留，降低贴着16GiB上限运行时被其他进程或瞬时分配
顶穿的概率，但不能推出“虚拟内存足够就不会OOM”。VBAR管理的是模型权重页；activation、attention
workspace、VAE/CLIP、CUDA上下文、其他GPU进程、pin memory和系统commit仍可能失败，VBAR本身也有
OOM返回路径。`clean_before_load`还是全局卸载，不是只卸载H3。策略在进程内持续有效，直到再次修改
或重启ComfyUI。

新示例为：

- `tests/fixtures/api/hybrid_model_vbar_headroom_api.json`；
- `examples/workflows/2026-08-09_H3_Hybrid_Model_VBAR_Headroom_Stock20_EXP.json`。

示例使用4.0GiB固定总预留、不做全局清理、DynamicVRAM必需、512MiB当前门槛和16GiB主机commit
门槛。RTX 4060 Ti 16GiB、736×416、124帧、Hybrid Stock20真实A/B中，未启用策略最低余量
41.879MiB；2GiB为343.086MiB、3GiB为528.828MiB。4GiB三冷三暖全部成功，最差冷态余量
1028.117MiB、最差暖态余量1401.415MiB，连续暖态基线上升最大12.25MiB；同seed基线与
2/3/4GiB的124帧解码视频和PCM哈希逐位一致。完整证据见
[VRAM Policy Advanced验证报告](docs/VRAM_POLICY_ADVANCED_VALIDATION.md)。

这只证明该GPU和该工作流的4GiB起点通过项目门槛，不覆盖0.6M/362帧、1080p、长视频、语音、
其他GPU或并发CUDA程序，因此节点报告仍把`memory_safe_claim`和`never_oom`保持为false。

## EXP：来源视频音画重绘

`1.11.0` 新增的 Source AV 链不是时间轴上的视频拼接器，而是把已有视频与声音规范化、VAE编码
后作为 H3 的起始联合 AV latent。它支持以下实验组合：

该方向受到 `ptmaster/ComfyUI-PT_H3ConcatAVLatent` 的产品思路启发，但本项目没有复制、打包或
依赖该仓库代码；实现基于 ComfyUI 原生 GPL AV latent 契约与本项目已有的校验、mask和时间轴
基础设施独立完成。当前 ComfyUI 已自带通用 `LTXVConcatAVLatent`，本项目新增的是 H3 专用的
媒体规范化、严格时钟检查、双流模式和风险报告，而不是重复注册一个无校验的通用包装器。

- `video=remix, audio=lock`：重绘画面，尽量保留来源音频 latent；
- `video=lock, audio=remix/regenerate`：尽量保留画面，重绘或重生声音；
- 双流 `remix`：画面和声音分别使用自己的 denoise mask；
- `regenerate`：对应流的 mask 为1，但在真实矩阵通过前不把它描述成完全摆脱来源信息。

推荐连接：

1. Core `Load Video -> Get Video Components`；
2. frames、audio、fps 进入 `Source Media Window`，选择画布、起点和124帧窗口；
3. 输出 frames/audio 分别进入标准 `VAE Encode` 与 `VAE Encode Audio`；
4. 两个 latent 进入 `Source AV Prepare`；
5. 使用它的 `av_latent` 替换 `SamplerCustomAdvanced.latent_image`，同时接到双时钟节点的
   `av_latent`；Conditioning 仍只负责 prompt 和媒体条件；
6. sampler 输出继续使用现有 `AV Decode -> Create Video -> Save Video`。

`Source Media Window` 会把目标帧数向上对齐到 `17n+5`，按来源 fps 选择最接近的帧，并输出
精确时长的32kHz双声道音频。视频不足默认拒绝；用户显式选择 `hold_last_frame` 才会保持末帧。
音频不足默认补静音并写入报告；不连接音频时会生成报告标记的静音轨，供 Audio VAE 构造合法
联合 latent。这个节点接收的完整 IMAGE 已经在内存中，因此不能把它称为长视频流式或低内存解码。

`Source AV Prepare` 严格要求视频 `[1,24,T,H,W]`、音频 `[1,32,2,T40]`、视频
`T=5n+2`，并校验音频长度是否等于 `round((17n+5)*40/24)`。音频不一致时只能按用户选择的
`strict`、裁切或补零生成尾部策略处理，不会静默修改。视频元数据优先保留，非冲突音频元数据
会合并，冲突字段和所有时间调整都会写入 report。

当前没有证据证明 `0.25/0.5/0.75` 会形成视觉上严格单调或线性的重绘权重，也没有完成真实
H3画质、身份、动作、音频保真和16GB显存矩阵。因此三个节点均为 EXP，不标 `memory_safe`、
“任意视频”或“精准局部重绘”。API与前端示例分别为：

- `tests/fixtures/api/source_video_repaint_api.json`；
- `examples/workflows/2026-08-09_H3_Source_Video_Repaint_Stock20_EXP.json`。

本机已用真实来源有声视频、FL2VA pruned INT8、Qwen3-VL NVFP4 与双 H3 VAE 完成一次
256×256、124帧、1步机械整链检查；结果成功保存为24fps H.264 + 32kHz stereo AAC，视频与
音频时长均约5.167秒。该探针只证明加载、双VAE、latent组装、采样、双解码和封装能连通，
不证明1步画质。运行期间整卡最低空闲仅44.52MiB，远低于512MiB安全门槛，所以16GB环境
仍属于高风险实验档；在多模式、多强度、三冷三暖及质量矩阵完成前不提供显存安全承诺。

## Advanced：首尾帧 + 多个中间关键帧

`MiniMax H3 Keyframe Plan (Advanced)` 与 `MiniMax H3 Multi-Keyframe Conditioning
(Advanced)` 是一条完全独立的 FL2VA/Hybrid 实验路线。稳定
`MiniMaxH3AudioConditioningT8` 没有增加输入、改变默认值或修改执行路径；已有工作流继续按
原方式运行。Advanced 路线当前要求同时连接首帧与尾帧，并可链式加入 1～7 张中间图，位置可用
24fps 绝对帧、秒或 `0～100` 百分比表达。重复位置、首尾端点、越界和无法识别的底层补丁会直接
报错，不会静默改成别的位置。

推荐接线：

1. 多个 `Keyframe Plan Advanced` 按时间顺序或任意顺序链起来，最终由节点解析后排序；
2. MODEL、CLIP、双 VAE、首帧、尾帧和最终 plan 接入 `Multi-Keyframe Conditioning Advanced`；
3. Advanced 输出的 `model` 接双时钟节点，`positive` 接 `BasicGuider`，`av_latent` 同时接双时钟
   节点与 `SamplerCustomAdvanced.latent_image`；解码和保存继续使用原节点；
4. 如果还连接普通图片/视频/音频参考，使用 `Hybrid`，提示词中的 Picture/Video/Audio 序号仍按
   节点报告的媒体顺序编写。

每个时间线关键帧的 `visual_noise_aug` 是 MiniMax H3 底层原始混噪参数，不是“参考强度百分比”。
默认先用 `0.999`；更低数值可能削弱锚点，也可能改变动作、身份或构图。非统一数值只会在克隆的
MODEL 上同时作用于对应条件 latent 的混噪与对应 packed timestep 行；如果当前 ComfyUI 的内部
契约不能被验证，节点会 fail closed。普通非时间线视觉参考目前仍共用
`reference_visual_noise_aug`，并不是每张普通参考图独立控制。

本机真实检查使用 RTX 4060 Ti 16GB、FL2VA pruned INT8、Qwen NVFP4 与双VAE。736×416、
124帧、1步下，0/1/3/5张中间帧各完成3次冷启动和3次暖运行，共24/24成功；每个视频都是
124帧/24fps，独立音频均为finite 32kHz stereo，连续运行没有显存基线阶梯增长。该矩阵最差
余量672.95MiB；最大7张中间帧（连同首尾共9张）的单次上限探针也成功，余量1819.42MiB。
这些只是机械/内存证据，不能推导任意配置安全；另一条256×256、22帧、非统一raw值探针仅余
202.82MiB，非双时钟、Block Cache兼容探针也低于512MiB，所以仍不标16GB `memory_safe`。

Stock20质量检查覆盖3类素材×3个seed、每条首/25%/50%/75%/尾共45个锚点：42/45在自动
全局相似度代理中命中目标±2帧，全部27个中间锚点命中，9/9顺序正确，未检出黑白/灾难跳变代理。
这是自动代理加人工查看，不等于身份、动作或盲评保证。单独改变中间一帧的
`[0.999, 0.995, 0.990, 0.980, 0.950]` 时，其他锚点基本稳定，证明不是全局覆盖；但目标相似度
Spearman `rho=0.70`，未达到预设 `0.8` 门槛且不严格单调。因此示例继续默认全部 `0.999`，
该字段只称 raw EXP，不称线性参考强度。4步 FL2V Turbo机械可运行，但当前样本明显融化，
Advanced质量推荐仍是Stock20；加速档需要独立质量验证。

报告中的 `added_rows_vs_target_video_rows_percent` 只是 DiT packed视觉条件行比例，不是显存
百分比；它不包含CLIP图像处理、普通refs、VAE峰值、allocator/offload行为或attention的非线性交互。
Advanced节点也不允许与本项目Long Video MODEL补丁或第三方全局Motion Context/PackedLayout
补丁叠加。完整边界、矩阵和否决项见
[`docs/MULTIKEYFRAME_ADVANCED_VALIDATION.md`](docs/MULTIKEYFRAME_ADVANCED_VALIDATION.md)。

API 示例：`tests/fixtures/api/multikeyframe_advanced_api.json`；可拖入画布的示例：
`examples/workflows/2026-08-09_H3_MultiKeyframe_Advanced_EXP.json`。导入后需要替换四张占位图片；两个
中间节点均默认 `0.999`，低值只建议在固定素材/seed/采样设置下做A/B。

## EXP：视觉参考强度（Ref2VA 纹理 A/B）

`MiniMax H3 Visual Reference Strength (EXP/T8)` 是轻量 Conditioning 后置节点。把现有
`MiniMaxH3AudioConditioningT8.positive` 接入它，再把它的 `positive` 接到
`BasicGuider.conditioning`；`av_latent`、MODEL、VAE、采样器、调度器、shift 和步数仍走原接线。
节点把 `reference_strength` 原值写入 ComfyUI 当前支持的
`minimax_visual_cond_noise_aug`，不会产生随机数或增加 DiT 前向次数。

建议固定参考图、prompt、seed、尺寸和采样设置，从 `0.999` 基线依次比较 `0.995`、`0.990`，
再谨慎测试 `0.980` / `0.950`。降低数值可能减少参考纹理被过度复制或平均化，但不能称为
“修复油感”；它也会全局影响参考图片、参考视频以及 first/last-frame keyframe。`0.950` 及以下
属于激进实验，可能明显损失身份、动作、构图和首尾帧一致性。没有视觉参考时节点会明确拒绝，
只有音频参考也不会误报生效。当前核心只支持一个全局强度，不能为每张参考图单独设置。

API 示例：`tests/fixtures/api/ref2va_visual_reference_strength_exp_api.json`；可拖入画布的工作流：
`examples/workflows/2026-08-10_H3_Ref2VA_Visual_Reference_Strength_EXP.json`。前端示例使用完整
`minimax_h3_ref2va_int8_convrot.safetensors`、736×416、124帧和20步基线，导入后先替换
占位参考图。该参数本身不要求这些采样值，接入旧工作流时保持用户原有 sampler/scheduler 即可。

本机单参考、单 seed 的完整矩阵已经跑通：无节点与显式 `0.999` 的解码视频/音频最大绝对
误差均为0，`0.950` 重复两次也逐帧逐样本一致。该案例没有得到“数值越低越去油”的证据：
`0.995～0.950` 会改变姿态、表情、动作轨迹或构图，`0.950` 的输出偏移最大，面部高频代理还
低于 `0.999`。因此节点只适合受控 A/B；不能把某个默认值宣传成稳定修复。当前矩阵的最小
显存余量约35MiB，远低于项目512MiB安全门槛，也不能据此称16GB安全档。

## EXP：对白结束后保留完整背景声

这里处理的是一个与普通语音裁切不同的问题：H3 的最终音频往往是对白、音乐、环境声和音效的
同一条立体声母带。如果模型在目标台词之后继续念叨，直接把整条母带裁到台词结束会同时删除
后续音乐、环境和音效；当前节点不会用这种方式假装修复成功。

`1.12.0` 提供三层、默认拒绝猜测的实验能力：

1. `Dialogue Boundary Analyzer` 使用用户本地的 faster-whisper，在且仅在 ASR 中找到一个连续、
   完整且唯一的目标词序列时报告边界。目标重复两次、目标被额外台词插断或未找到时都不选
   “第一个/最后一个”；尾部能量只报告“还有信号”，不会把音乐或音效误判成语音。
2. 推荐的确定性路线是“对白 stem 与背景 stem 分开生成/准备”。`Dialogue Safe Master` 要求
   上游传入已验收的独立对白，并将独立音乐、环境和 SFX 放到目标 sample 时间线上。默认
   `strict` 不会暗中循环、补零或截断任何已连接 stem；只有用户显式选择策略才会调整。最终
   母带保持完整时长，对白结束后背景 stem 继续存在。
3. 如果创作流程必须走联合 H3，可使用两遍生成：先准备一条不含对白、完整时长的背景底轨，
   再由 `Timed Background Bed Lock` 编码成音频 latent；边界之前允许 H3 生成对白，边界之后
   默认 `tail_denoise_strength=0` 锁住背景底轨。它保留原视频流和已有视频 mask；已有音频 mask
   只作为上限，不会被节点偷偷放宽。

真实机械探针使用当前 FL2VA pruned INT8、256×256、124帧、稳定双时钟4步。标准124帧 Audio
Window 经真实音频 VAE 编码得到206步，而联合 H3 时钟需要207步，因此 `strict` 会正确拒绝；
示例显式使用 `fit_reported`，记录补1个零 latent 步。4步采样后，2秒前可编辑头部相对底轨
latent 的平均绝对变化为 `0.50223`，2秒后锁定尾部的最大绝对误差为 `2.38e-7`，在 `1e-6`
绝对容差内保持。解码对照同时发现音频 VAE 的时间感受野会让边界后最初约0.3秒仍受头部变化
影响；从2.3秒起100ms窗的最大差异降到约 `3.97e-4` 或更低。这证明 mask 机械生效，但不是
“样本级硬切”“绝对无接缝”或主观质量保证。

边界分析也在此前真实 Joint 两人失败样本上复测：一个样本的目标台词被额外内容插断，节点
返回 `target_not_found`；另一个样本在17个多余词后出现唯一完整目标，节点报告7.00–9.72秒、
`clean_exact=false`，而不是自动裁切或验收。ASR 会漏掉含混、非词汇人声，因此报告不是模型
真值。

当前没有集成自动源分离。机器虽安装了 `audio_separator` 包，但没有已选择/校验的分离权重；
常见 vocal separator 以音乐人声/伴奏分离为目标，不等价于“只移除目标人物的额外对白”，还
可能删除原本想保留的歌声或损伤音乐/SFX。必须先用合成可知真值与真实 H3 混音建立泄漏、
音乐损伤和听评门槛，未过门槛前不会靠模型名猜一个默认分离器。

示例：`tests/fixtures/api/dialogue_safe_master_api.json`、
`examples/workflows/2026-08-10_H3_Dialogue_Safe_Master_EXP.json`，以及两遍 H3 的
`tests/fixtures/api/dialogue_timed_bed_lock_api.json`、
`examples/workflows/2026-08-09_H3_Dialogue_Timed_Background_Bed_Lock_EXP.json`。所有输入文件都是占位符；
底轨必须是不含对白的独立完整背景，而不是已混合的 H3 最终母带。

## EXP：原生语音、参考音色与逐句对白

这套节点已经完成真实 H3 生成检查，但仍标记为 `Experimental`。它不是额外的确定性 TTS
模型，而是把联合音视频 H3 用在 32/64/128 像素暗色视频画布上，并只保留解码音频。小画布
降低 activation，不会消除 H3、Qwen3-VL 和双 VAE 权重；因此不能把“小画布”解释成
“16GB 必然安全”。

### 推荐连接方式

1. 将现有工作流的 H3 `MODEL`、Qwen3-VL `CLIP`、video VAE 和 audio VAE 直接接到
   `Speech Studio`；节点内部没有隐藏加载器，也不会再加载一份 32B 文本编码器。
2. 合成音色使用 `Voice Profile.voice_mode=described_voice`；参考音色使用
   `reference_voice` 并连接 ComfyUI `AUDIO`。参考模式必须显式确认已取得权利，节点只在
   当前工作流内保存 CPU 音频。只有用户主动连接 `Voice Library Save` 才持久化；同名默认
   拒绝覆盖，`Delete` 只移动到本地可恢复回收目录。
3. 使用 `Speech Plan` 明确输入实际台词、语言、演绎方向和渲染时长。当前质量探针使用
   20步 `res_multistep + simple`；未经 A/B，不把视频 Turbo LoRA 静默套到语音上。
4. 参考模式可能在目标台词前生成参考内容或无关引导声。需要严格文本输出时启用
   `trim_exact_target`，并指定本地 faster-whisper CTranslate2 模型；只有 ASR 找到完整目标
   token 顺序时才裁切，找不到会在报告中拒绝，不做模糊猜测。
5. 两人/三人对白采用“每个 turn 独立生成 → Assemble 绝对 sample 时间线”的方式，支持
   单句重做、停顿、重叠、声像和最终混合。联合一次生成多人身份尚未通过角色交换/串音门槛，
   `Joint Dialogue Conditioning` 仅保留 EXP；两次真实探针都产生大量额外语音，不能作为稳定模式。
6. 长文本使用 `Long Form Start/Resume -> Speech Studio -> Long Form Accept`。每次排队只生成
   manifest 的下一段；接受后先原子写 safetensors 和可播放 FLAC，再推进 manifest。重新排队
   会读取新指纹继续下一段，完成后用 `Long Form Compose` 合成。当前段正在采样时用 ComfyUI
   Stop；`request_cancel` 是段与段之间的合作式取消，不是后台线程硬中断。
7. `Performance Direction` 的 pace/pitch/energy/intensity 是未标定提示方向。需要精确输出时长或
   确定性移调，使用 `ADR Exact Fit`；超出显式安全变速范围会拒绝，不会静默强拉伸。

### 可选校验模型

- ASR：`faster-whisper` + 本地 CTranslate2 模型。输入可用绝对目录，或放在
  `ComfyUI/models/TTS/<folder>`。本机已安装并校验 pinned 多语言 small 模型；一条中文
  Stock20 探针的原始 CER 为 7.14%，但远未达到每语言30条的稳定门槛，不能宣传“中文已验证”。
- 说话人：`transformers` + 本地 `WavLMForXVector` 目录，同样支持绝对目录或
  `models/TTS/<folder>`。`report_cosine` 只报告余弦，不控制结果；`require_threshold` 的阈值
  依数据集而异，不能把单一默认值描述为通用真假判定。
- 两项模型都在 CPU 延迟加载，并可在每次校验后卸载。插件不自动下载，也不随仓库分发权重。

### 释放策略

| 策略 | 实际含义 |
|---|---|
| `keep_loaded` | 保持 ComfyUI 模型缓存，适合同一工作流后面还要继续生成 H3 |
| `clear_execution_cache` | 请求清理执行缓存；不声称只卸载 H3，也不等于显存归零 |
| `unload_all_models` | 强制全局卸载 ComfyUI 模型；会影响工作流中其他已加载模型，只有用户显式选择才执行 |

所有正常完成的校验、拒绝和裁切结果都会返回 JSON 报告。`Speech Studio` 会在 Conditioning
之前自动插入 `Abnormal-Exit Guard`：若取消、非 OOM 异常或上游失败使 Finalize 未执行，prompt
结束回调会补发所选异常释放请求并写入 `output/minimax_h3_t8/speech_recovery`。当前 ComfyUI 对
识别出的 CUDA OOM 本身也会全局卸载。真实“采样成功后故意在 ASR 报错”的探针已触发该回调并
回落显存；这仍不是所有 ComfyUI 版本、驱动崩溃或进程强杀场景的绝对 finally 保证。
最终音频默认使用 `-1 dBFS` 的衰减式峰值保护，只降低超限音频，不把较安静的语音自动放大。

### 2026-08-10 本机真实探针

测试环境为 RTX 4060 Ti 16GiB、Windows、ComfyUI `cbbc9dab1`、FL2VA pruned INT8、
Qwen3-VL NVFP4 和两套 H3 VAE；采样为 stock 20步，不是 Turbo 质量外推。

| 探针 | 结果 | 不能推出什么 |
|---|---|---|
| 描述音色 | 10.125秒英文；ASR逐词一致；同 seed 重复 PCM 完全一致 | 不代表不同硬件/版本位级一致 |
| 参考音色 | 原始10.125秒含无关前导；精确目标裁成4.465秒后英文逐词一致；同 seed 重复 PCM一致 | 不代表所有参考都能自动干净裁切 |
| 早期说话人信号 | 同参考生成样本 WavLM余弦0.949587；刻意不同的男性负对照0.484272 | 仅是随后10人集合前的单对照，不单独作为结论 |
| 两人对白 | 两段独立生成后合成9.81秒；合并台词逐词一致；两段余弦0.247203 | 不证明多人联合生成、角色长期稳定或重叠对白自然度 |
| 显存/释放 | 整卡峰值约16262–16316MiB；显式全局卸载后隔离服务 torch pool 15秒回到32–64MiB | 观测最小余量约64–118MiB，低于512MiB门槛，不能标 `memory_safe` 或“绝不OOM” |
| 异常释放 | Stock20采样完成后故意制造上游校验异常；Finalize未执行，生命周期 Guard 仍请求全局释放并落盘恢复事件 | 不代表进程强杀、驱动崩溃或所有旧版 ComfyUI 都可回调 |
| 三冷三暖 | 三个冷进程均成功；同进程三次 warm 峰值没有阶梯抬升，但 `keep_loaded` 基线驻留增加约15.1GiB | 最小余量仍只有约17MiB，16GB安全档继续否决 |
| 中文 | 一条10秒中文 Stock20 成功，原始 CER 1/14 = 7.14% | 样本数1，未过每语言30条门槛，不代表中文/多语言稳定 |
| 10人音色集合 | 10名有许可 LibriSpeech 说话人、90个冒充者配对；10/10 genuine 高于 impostor 95百分位 | 只有每人一句，ABX包尚无人类听评，仍不能称“高保真克隆” |
| 演绎控制 | 同seed 7案例全部生成；语速、F0和响度三组单调门槛均失败 | pace/pitch/energy/intensity 仍是未标定 prompt 方向 |
| 长文本状态 | 真实32秒四段输出；另一个四段工作流连续排队推进并完成哈希合成；32秒/2分钟/10分钟合成状态矩阵样本数与SHA均精确 | 2/10分钟只验证持久状态，不证明真实H3长期音色连续性 |
| ADR | 安全范围内变速/移调后输出误差为0 sample，超范围明确拒绝 | 不证明音素边界或视频口型同步 |
| Joint多人 | 两次两人Stock20都完成生成，但WER分别225%和237.5%，含大量额外语音 | 稳定Joint路径明确否决，推荐逐turn合成 |

仍未完成：每语言30条中文/多语言矩阵、至少3名听者的盲听ABX、情绪/语速/音高可感知标定、
真实H3的2分钟/10分钟音色连续性、当前采样中的后台硬取消、token/帧级实时流、ADR音素/口型
同步、持久音色库跨进程/网络盘压力测试和跨GPU验证。16GB `memory_safe`、高保真克隆和稳定
Joint多人均未放行。

API 与前端示例除原有描述音色、参考音色和逐turn对白外，新增：

- `speech_performance_adr_api.json` / `2026-08-10_H3_Speech_Performance_ADR_Stock20_EXP.json`；
- `speech_longform_resume_api.json` / `2026-08-10_H3_Speech_LongForm_Resume_Stock20_EXP.json`；
- `speech_longform_compose_api.json` / `2026-08-10_H3_Speech_LongForm_Compose_EXP.json`；
- `speech_voice_library_save_api.json`、`speech_voice_library_load_api.json` 及对应前端工作流；
- `speech_voice_library_delete_api.json`、`speech_vram_preflight_api.json`、
  `speech_longform_control_api.json` 及对应维护工作流；
- `speech_joint_dialogue_exp_api.json` / `2026-08-09_H3_Speech_Joint_Dialogue_Stock20_EXP.json`，仅用于复现
  已知质量风险，不作为推荐模板。

参考与Joint示例导入后必须替换 `speech_reference*.flac` 占位音频；完整环境、输出哈希、显存和否决项见
[语音真实生成验证报告](docs/SPEECH_VALIDATION_REPORT.md)。

声音参考必须获得说话人授权，不得用于未经同意的冒充；生成内容应按适用许可和平台规则
披露为 AI。MiniMax H3 权重许可独立于本仓库 GPL 代码，插件不分发任何权重。请同时阅读
[MiniMax H3 官方许可证](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)。

## EXP：分段长视频续写

`T8/MiniMax H3/Long Video/Experimental` 是与原有14个节点隔离的实验子系统。它没有
安装或依赖 `ComfyUI-H3-Motion-Context`，也不会在插件导入时全局修改 ComfyUI 的
`PackedLayout` 或 `MiniMaxH3.extra_conds`。Long Video Conditioning 会克隆输入 MODEL，
只在该克隆上挂接一个 `extra_conds` object patch；不带本项目长视频标记的 conditioning
直接旁路，因此稳定 Conditioning、Hybrid、Still 和其他 H3 工作流不受该补丁影响。

设计研究参考了 NikoDemon80 的
[ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
固定提交 `15fc6a7` 所展示的内部关键帧和音频 timeline 思路；本项目没有复制或捆绑该插件，
重新实现了局部模型补丁、多参考合并、直接 latent tail 和有校验的状态系统。上游代码采用
GPL-3.0-only，本项目代码采用 GPL-3.0-or-later；二者均不包含模型权重。

当前链路：

1. Planner 输入固定 `chain_id`，把 `segment_index` 从 `0` 开始逐段递增；默认上下文为22帧。
2. Previous Context 在第0段自动返回空值；第N段只读取
   `output/minimax_h3_t8_long_video/<chain_id>/segment_(N-1).context.safetensors`，不会猜“最新文件”。
3. Long Video Conditioning 直接截取 sampler 的视频 latent 尾部作为 5/22/39 帧运动条件，
   不解码上一整段 IMAGE，也不做视频 VAE 重编码；`video_and_audio` 还会把音频 latent 尾部
   放到当前目标头部的正确 timeline，`video_only` 则只续运动。
4. sampled AV latent 同时进入 Context Save 和现有 AV Decode；Context Save 只保存最多39帧
   所需的 CPU tail，并使用 tensor SHA-256、metadata 校验、同目录临时文件和原子替换。
5. Planner 的 `trim_start_seconds` / `final_duration_seconds` 连接现有 Output Trim，同步删除
   重建的画面与音频头部，再交给 ComfyUI 原生 `CreateVideo -> SaveVideo`。该链按帧数裁音频
   tensor，并避免 VHS `apad + -shortest` 在独立 MP4 上造成约79–90ms声轨短缺。

针对60秒实测暴露的逐段人物年龄/身份漂移，Long Video Conditioning 在输入末尾追加三个高级项，
且默认值继续严格保持旧工作流：

- `first_frame_reuse=segment0_only`：默认值；`first_frame`只作为第0段精确关键帧，续写段仍只用前段尾部。
- `first_frame_reuse=persistent_identity_reference`：只在续写段增加非时间轴身份参考，同时保留
  5/22/39帧运动上下文；第0段仍由原始`first_frame`精确控制。
- `persistent_identity_image`：可选的续写专用身份裁剪图，建议清晰正脸或上半身；不会改变第0段。
- `persistent_identity_strategy=single_reference`：兼容默认。连接身份裁剪图时优先只用裁剪图，未连接时
  回退到完整`first_frame`。
- `persistent_identity_strategy=scene_plus_identity`：续写段把完整`first_frame`和身份裁剪图作为两张
  独立参考图；未连接`persistent_identity_image`会失败关闭。持续参考与用户ref images合计最多9张，
  `task_type`应使用`auto`或`Hybrid`。
- `persistent_identity_interval=1`：高级实验项。`1`保持每个续写段都注入双参考的现有行为；`2`只在
  续写段1、3、5、7……注入，间隔段只使用有界运动/音频上下文。它只是固定频率控制，不会检测
  身份漂移，也不能保证减少参考后动作更自由。

新输入全部追加在旧schema末尾，因此旧API JSON和旧`widgets_values`前缀不变；缺省interval为1。
方案不加载新模型、
不保存完整历史视频，也不改稳定采样；但每个续写段会多1或2个reference block及对应VAE编码，序列、
耗时和显存可能上升，也可能与运动上下文竞争。原来的“完整首帧单参考”在32秒/8段仍从首续段
0.613漂到末段0.134，因此已经被否决为长期身份方案。

重构后的短链验证分两步。身份裁剪图单参考在三seed足球/手臂高运动探针中，相对旧行为的配对余弦
均值/中位数为+0.08272/+0.09845，54/59帧更高；但相对完整首帧单参考有一个seed回退，不能单独作为
统一答案。随后“场景+身份裁剪”双参考完成三seed、6/6条双段冷链，所有12次采样成功且每组第0段
逐比特一致；相对旧行为的余弦均值/中位数为+0.11278/+0.11628（56/59更高），相对完整首帧为
+0.08454/+0.09455（52/60更高），三个seed的中位收益均为正，接触图未见冻结或新伪影。

最终用seed `2608097101`完成一条独立32秒/8段“场景+身份裁剪”链：8/8次采样一次成功，无OOM、
重试或缓存复用，成片精确768帧/32.000秒。七个续写段的身份余弦中位数为
0.699/0.639/0.644/0.609/0.737/0.601/0.574，末段/首续段保持率0.821；相对旧行为58个配对样本
均值/中位数+0.42945/+0.46670，相对完整首帧63个样本+0.33771/+0.35802。最低空闲显存
3906.07MiB，post-15秒占用回到1231.63MiB；这只证明本机固定档的一条链，没有通用无泄漏或显存
安全含义。预设动作门槛没有全过：第2、5个续写段flow-P90仅为旧行为0.546/0.538，第5段MAD比
0.647；2fps抽帧仍显示足球、手臂和姿态持续运动，不是冻帧，但动作幅度/轨迹受限警告成立。
因此功能继续标为EXP并默认关闭，暂不进入三seed 60秒矩阵；下一步先做未参与调参的多seed/多素材
32秒复验并解决动作强度回退，不能宣传为“身份锁定”“动作无损”或“显存安全档”。

后续先做了同一开发案例的`persistent_identity_interval=2`对照。该链8/8段一次成功，成片精确
768帧/32.000秒，运行825.92秒，整卡峰值12,744.43MiB、最低余量3,635.07MiB，post-15秒回到
1,231.63MiB。身份末续段/首续段比为0.525，低于每段注入方案的0.821；相对旧行为的续写
flow-P90仍在第4/5/6段降到0.648/0.457/0.652，动作地板继续失败。失败段并不只对应参考注入段，
说明简单奇偶段降频不是稳定的“解除动作约束”方案。该控件因此只保留为EXP研究项，默认仍为1。

随后使用未参与策略选择的新人物、seed和动作完成两条32秒/8段冷链；两条均是736×416、124帧
窗口、22帧AV context、4步Stock+DynamicVRAM h2、每段双参考、段间`unload_all_models`：

- 旗袍扇舞/鼓点：8/8成功，运行872.95秒，峰值12,170.76MiB、最低余量4,208.74MiB，post-15
  回到1,231.63MiB。逐段时间线保持同一人物、服装与院落，扇舞持续且最差接缝未见硬切；稀疏
  人脸抽样的末/首续段比约0.937。音频最大半秒响度差2.68dB、首末高频变化-3.41dB，但要求
  120 BPM时Librosa描述性估计约104.17 BPM，不能判为严格节奏遵循通过。
- 双人物对白：8/8成功，运行862.41秒，峰值12,227.57MiB、最低余量4,151.93MiB，post-15同样
  回到1,231.63MiB。两个源人物在8段×10个抽样点均同时检出，末段仍保留两人；但第6→7段从
  全身景别跳到近景，构图连续性失败。经校验的`faster-whisper-small.en`在32秒内持续识别两句
  目标对白，两句最佳词错误率均为0，但内容主要重复这两句，不是自然长对话。裁脸后的嘴部代理
  检测覆盖约85.4%/27.1%，与音频包络相关仅0.042/-0.008；由于没有训练型SyncNet证据，口型同步
  仍是未证实项，不能宣传稳定。

用户当前日常目标约30秒，因此本轮把32秒作为完整链验收长度，保留现有任意总时长/60秒能力，但
不再要求新增60秒实跑。现有证据支持“本机固定配置下32秒分段机械与显存稳定”，不支持把节奏、
镜头接缝或口型质量描述成已经普遍解决。跨GPU、高分辨率32秒、真人盲听/盲评和训练型唇同步评分
仍需独立资源与硬件门槛。

### 双参考身份续写示例

可直接导入的 `examples/workflows/2026-08-09_H3_Long_Video_Background_22F_ScenePlusIdentity_EXP.json`
基于后台长链示例，专门演示“完整场景首帧 + 人物身份裁剪图”的双参考续写：

1. 完整场景图连接 Long Video Conditioning 的 `first_frame`，继续精确控制第0段首帧。
2. 同一人物的清晰正脸或上半身裁剪图连接 `persistent_identity_image`；续写段同时保留完整场景与
   身份裁剪两个独立 reference block。
3. 节点预设 `task_type=auto`、`first_frame_reuse=persistent_identity_reference`、
   `persistent_identity_strategy=scene_plus_identity` 和 `ref_image_size=match`。若身份图未连接，
   `scene_plus_identity` 会明确报错，不会静默退回单参考。
4. `persistent_identity_interval=1`保持当前每个续写段都注入双参考的已验证基线；提高该值只适合
   受控对照，当前`2`没有通过动作复合门槛。
5. 示例沿用 736×416、124帧内部窗口、22帧上下文和4步采样的实验基线；导入后必须先替换两张
   示例输入图，并检查模型、VAE、CLIP、LoRA 与输出目录是否符合本机环境。

如果升级节点后画布仍看不到 `persistent_identity_image`、`persistent_identity_strategy` 或
`persistent_identity_interval`，先完整
重启 ComfyUI 再导入该工作流；仅刷新浏览器不会重新加载 Python 节点 schema。

身份裁剪图应只保留一个主要人物，尽量包含稳定可辨识的脸部、发型和上半身服装，不要用多人合照、
过小人脸或严重遮挡图。这个工作流只提供已验证接线与参数基线；它仍是 EXP，32秒单链虽显著改善
身份保持，但动作幅度门槛未全部通过，也没有证明所有素材、时长和显卡都不会 OOM。

中间段有一条重要约束：裁后输出必须一直保留到本次 sampler 的末尾，否则下一段会从用户
没有看到的隐藏尾帧续写。Planner 因此把可继续段的有效时长量化到当前 H3 `17n+5` 网格；
只有最后一段开启高级项 `is_final_segment=true` 后，才允许按请求时长裁掉隐藏尾部，同时
自动输出 `save_context=false`，防止把这个最终裁尾误用为后续上下文。默认4.25秒、22帧上下文
时，第0段为124帧（约5.167秒），后续可继续段为裁头22帧后保留102帧（4.25秒）。

`1.4.0` 的手工 P1 状态文件采用固定槽位：重抽第N段只覆盖 `segment_N`，而第N段读取的仍是 `segment_(N-1)`；
中断或 OOM 前没有完成原子替换时，不会破坏上一段。状态只保留尾部，不缓存完整历史 IMAGE
或完整 AV latent，因此链长增加不会让当前 H3 序列或状态内存随历史总时长线性堆积。

`1.5.0` 进一步提供推荐的“候选→接受”状态链：

1. 用 `Accepted Context` 代替旧 `Previous Context`；第0段仍为空，第N段只读取 manifest 中
   已接受的 N-1，并输出父候选 ID 与 manifest revision。
2. 解码、裁头后接 `Save Candidate`。它以绝对24fps时间轴计算音频起止 sample，原子写入
   当前候选 MP4、可选 continuation context 和 `candidate.json`，不会改变已接受历史。
3. `Review & Accept` 默认 `accept_candidate=false`，可直接预览候选。满意后改为 `true` 再排队；
   同一候选重复提交是幂等的。若有意替换已接受的第N段，必须显式选择
   `replace_and_invalidate_following`，manifest 会保留失效历史，但 N 之后都必须重新生成。
4. manifest 写入有同目录锁、SHA-256、临时文件原子替换和一代有效备份。主清单损坏时可回退
   上一修订；回退可能丢失最后一次接受记录，但候选文件仍保留，可重新提交而不必重新采样。
5. 最后一段必须由 Planner 标为 final。所有段接受后，用 `Compose Accepted` 校验连续帧/sample
   边界和文件哈希，再流式生成最终 MP4；默认不会把未标 final 的半成品误当成完整长片。

`1.6.0` 新增推荐的“总时长→人工审核→自动恢复下一段”路线：

1. `Chain Orchestrator` 只输入一次目标总时长。时长先量化到24fps的精确总帧数，再拆成固定
   `17n+5` 内部窗口；默认窗口124帧、上下文22帧。总时长增加只增加片段数，不增加单段内部窗口。
2. 第0段有效新增124帧，后续完整段有效新增102帧；最后一段自动标记 final 并精确裁到剩余帧。
   例如60秒严格规划为14段：`124 + 12×102 + 92 = 1440帧`，所有段内部仍采样124帧。
3. manifest 已接受段数就是恢复点。重新打开工作流或执行失败后，节点会输出第一段未接受的
   `segment_index`、时间轴、裁切参数、prompt 和 seed；已接受时间结构与新设置冲突时明确拒绝。
4. `global_prompt` 为默认提示词；高级 `segment_prompts_json` 可按段覆盖 prompt、seed 和镜头备注。
   seed 支持固定、递增或按 chain/segment 哈希派生，同一计划可以确定性复现。steps、视频/音频
   shift、sampler 和 scheduler 也由同一个 Orchestrator 同时连接采样节点和候选元数据；用户不必
   在两处重复填写，已接受链改变采样身份时会在下一次采样前拒绝续接。
5. 最后一段接受后，节点输出完整进度并阻断下游采样，避免多生成一段。它不会在一个节点里保留
   完整历史 IMAGE/AUDIO tensor，也不会通过后台循环绕过 ComfyUI 的模型管理。

`1.7.0` 在不移除上述人工审核路线的前提下增加一条显式后台路线。加载
`examples/workflows/2026-08-09_H3_Long_Video_Background_22F_EXP.json` 后，`Background Start` 的默认值仍是
`review_only`；只有主动改为 `auto_accept_and_continue` 才会跳过人工预览，自动接受每个成功候选。
终端节点每次只排入一个下一段 prompt，不在单个 Python 循环里长期持有完整 IMAGE/AUDIO 历史。

后台节点提供 `status / 状态`、`pause / 当前段后暂停`、`resume / 继续` 与 `cancel / 取消` 按钮：

- 暂停不会丢弃已接受段；当前段成功后停在下一段恢复点。排队但未开始的段会直接撤回。
- 取消只按当前 background prompt ID 删除或中断，不清空用户的整个 ComfyUI 队列。若候选已经跨过
  manifest 原子提交点，它可能完成本次提交，但不会再排下一段。
- `max_retries` 表示同一失败段的额外尝试次数。重试复用完全相同的 API prompt，绝不静默降低
  分辨率、帧数、上下文、采样步数或改 seed；同一错误超过上限后进入 `failed`。
- `clear_execution_cache` 是默认释放策略：显式设置 `free_memory=true`、`unload_models=false`，
  清执行 tensor/软缓存但不声称卸载模型。策略在每次候选原子接受后应用，包含继续排队、段后暂停
  和 final；`unload_all_models` 会调用ComfyUI全局卸载标志，连H3以外的模型一起卸载；
  `keep_loaded`不请求释放。接受后释放失败会保留manifest并把任务标记failed，不会重生成。
- `background_job.json` 只保存状态和 prompt SHA-256，不保存 prompt 正文。ComfyUI 重启后 manifest
  仍可恢复，但内存中的 prompt snapshot 已丢失，需把后台工作流重新排队一次完成重新附着。
  错误状态采用字段白名单，不落盘 `current_inputs/current_outputs`、媒体 tensor 或提示词。
- 最终段接受后可自动调用流式合成器。若接受已经成功但最终合成失败，任务会停止并保留完整
  manifest，不能通过“重试生成”越过已提交边界；此时单独运行 `Compose Accepted`。

隔离 ComfyUI 的模型无关实测已覆盖两段自动排队/合成、当前段后暂停再继续、定向取消以及一次
原参数失败重试。另一次真实 H3 机械探针使用 FL2VA INT8、Standard Turbo EMA LoRA、NVFP4 CLIP、
双 H3 VAE、256×256、124帧窗口、22帧 context、1步、DynamicVRAM headroom 2GiB，并选择
`unload_all_models`：两个独立 prompt 均成功，manifest 为 `124+20=144` 帧，最终 H.264/AAC
音画流均严格6.000秒。该探针只证明后台执行与强释放后重载闭环，不是画质基准，也不能外推成
四步、高分辨率、其他GPU的通用 `memory_safe` 或“绝不 OOM”。

随后完成了一条代表性的真实四步后台长链：RTX 4060 Ti 16GB、FL2VA INT8、Standard Turbo
LoRA、736×416、124帧窗口、22帧音画context、DynamicVRAM headroom 2GiB，段间选择全局
`unload_all_models`。60秒被严格拆为14个独立prompt，14/14均一次成功，无重试或OOM；manifest
revision 14 为 `124 + 12×102 + 92 = 1440` 帧、1,920,000音频samples，最终H.264/AAC的
视频、音频和容器均严格60.000秒。0.5秒轮询观察到整卡峰值约12,823MiB、最低余量约3,556MiB；
全局峰值在第3段后基本平台化，第9段只增加约39MiB，此后不再上升，本轮没有阶梯式泄漏迹象。
13个视频接触帧没有明显硬切，但音频半秒窗响度变化最大仍约13.75dB；5ms bridge仅把单样本
跳变中位降低约96.2%，不能修复响度、语义、音乐节奏或口型。该结果只证明本机固定配置的一个
单prompt/seed后台长链，不等于跨GPU、高分辨率、多参考或通用显存安全档。详细证据位于本地
`artifacts/background-four-step-check/REPORT.md`。

长链完成后的审计发现旧实现只在“准备排下一段”时请求释放，暂停和final会继续持有模型。
现已改成每次接受后统一请求所选策略，并用另一条真实256×256一阶双段链复测：完成瞬间整卡约
8,124MiB，状态记录 `last_release_policy=unload_all_models`，随后自动回落到约1,230MiB，释放约
6,894MiB，不再需要手动调用`/free`。这证明final释放时机生效，不代表所有第三方模型都能无副作用重载。

同条件释放策略对照随后扩展为3个配对seed：每档3次全新ComfyUI进程冷态，并在独立同进程
primer后连续测3次暖态，共21条双段链、其中18条正式测量。全部成功且无重试/OOM；每个seed在
三档×冷暖六种条件下的首段MP4、首段AV tail tensor、第二段MP4和最终成片SHA-256完全一致。

| 策略 | 冷态耗时均值 | 暖态耗时均值 | 冷/暖整卡峰值均值 | 冷/暖15秒残留均值 |
|---|---:|---:|---:|---:|
| `keep_loaded` | 170.89s | 153.10s | 13,449.95 / 13,467.30MiB | 8,083.22 / 7,987.22MiB |
| `clear_execution_cache` | 188.28s | 185.08s | 13,434.52 / 13,408.94MiB | 1,229.63 / 1,229.63MiB |
| `unload_all_models` | 189.08s | 197.13s | 13,421.52 / 13,384.03MiB | 1,229.63 / 1,229.63MiB |

三档所有配对峰值差均低于128MiB差异阈值，旧单次探针中强释放低约1GiB的现象没有重复。
`keep_loaded`相对默认档冷/暖平均快17.39/31.97秒，但15秒后多占约6.85/6.76GiB；只适合用户
明确愿意为单工作流吞吐量保留显存时选择。强释放相对默认档冷态只慢0.80秒，暖态平均慢
12.05秒，还会卸载其他ComfyUI模型。默认因此继续使用`clear_execution_cache`。

首次`keep_loaded`探针还发现ComfyUI会把运行期`is_changed`指纹写回prompt；若原样续排，会把
整张下一段图误判为缓存命中。后台prompt快照现会双重剥离该字段。18条正式链中，keep-loaded
只缓存节点1–5的加载器，编排、采样、保存和终端均正常重跑；另外两档没有节点缓存命中。
这些结论仍只绑定本机RTX 4060 Ti 16GB、当前模型/插件和736×416双段配置，不能外推跨GPU、
高分辨率、多参考或通用不OOM保证。

后台恢复又完成了一次真实进程强杀检查：256×256、一阶、双段H3链在第0段已持久接受、manifest
revision 1且下一prompt运行时终止ComfyUI。重启后的状态查询会把磁盘残留的`running`纠正为
`detached`，显示已接受1段和“需重新排队工作流一次”；重新排同一工作流一次后从第1段继续，
旧第0段的候选ID、MP4、AV tail tensor哈希和修改时间均未改变，最终revision 2为144帧且A/V/容器
严格6.000秒。加入v2磁盘schema、操作系统锁与后台进程租约后再次复测通过，恢复阶段整卡峰值约
13,537.02MiB；杀进程前的background state和manifest均已原生写成带明确format marker的schema 2。
随后两个独立真实H3后台链在同一ComfyUI队列按
`A0 -> B0 -> A1 -> B1`交错完成，两个job、prompt、manifest、父链、目录和成片保持隔离，整卡峰值
约13,511.44MiB，无OOM，两个链的state和manifest也都保持原生schema 2。

本地Windows/NTFS多进程门槛也已补齐：manifest改用进程死亡时由操作系统自动释放的
`manifest.lock.v2`；两个进程争同一chain/index/revision时只有一个提交，四进程×25次共100次锁内
更新无丢失，持锁进程强杀后2秒内可接管。后台job另有整链进程租约，第二个ComfyUI进程会在生成前
被拒绝，首进程强杀后第三进程可接管并写入`previous_job_id`。旧版活锁会被尊重，死锁残留不阻塞且
不被破坏性删除；未知新schema明确拒绝，不会回退旧backup，same-schema附加字段会跨下一次提交保留。
损坏的辅助后台状态会隔离留档，再由accepted manifest恢复。现有schema 1 manifest/background
state会只在内存中规范化为schema 2，纯读取不改原文件；下一次受锁保护的manifest提交会原子写入
schema 2主文件并保留原始schema 1备份，后台重启接管也会写schema 2并记录来源schema。已有真实
H3 schema 1链的只读迁移检查保持两个原文件哈希不变，新建原生schema 2强杀恢复链也已通过。

接受事务另补8个确定性故障注入场景。完全相同的候选若已接受MP4丢失或context损坏，会从仍通过
哈希校验的候选文件安全修复，manifest revision不增加；同一规范化`candidate_id`不得绑定不同
内容，失效历史中的ID也不能覆盖原归档文件。context复制失败、备份写完但主manifest写入失败时，
旧manifest仍是唯一权威，重试同一候选可完成提交。主manifest缺失时必须先读取有效backup，即使
调用方允许新建链也不会重置历史；未知schema或损坏backup则拒绝创建空链。这些测试覆盖明确的
程序步骤边界，不等于任意CPU指令或掉电边界已经全部证明。

其中两个最高风险步骤又提升为Windows/NTFS真实进程强杀：worker持有真实`manifest.lock.v2`，
分别在“accepted MP4已完整复制、context尚未复制”和“旧revision已写入backup、新primary尚未替换”
时由父进程直接`kill()`。两个断点各做3轮独立重复，共6/6恢复；每次OS锁自动释放，旧/空manifest
仍是权威，同一候选在2秒内完成重试，媒体哈希和最终revision正确，没有残留worker。这比Python
异常注入更强，但仍使用小型测试媒体，不是H3 CUDA生成中的任意时刻强杀，也不覆盖机器掉电或网络盘。

这些结果证明的是“持久接受边界后的强杀恢复”“单队列多链隔离”“本地schema 1→2迁移契约”和
“单机NTFS同链所有权/提交串行化”，不等于不同已发布插件/ComfyUI组合的完整升级降级矩阵、网络
共享盘锁、同时CUDA执行或多GPU并行。详细证据位于本地
`artifacts/background-crash-recovery/REPORT.md`。

合成器的视频逐帧处理，音频一次只解码一个已接受片段。默认 `cosine_bridge` 在每个边界把
当前段开头的值连续地拉到上一段末样本，并在默认5ms内余弦衰减修正；它不做会缩短时长的
overlap acrossfade，最终 sample 总数严格取 manifest 的绝对边界。该处理只能降低瞬时幅值跳变，
不能证明相位、节奏或语义已经无缝；视频会重编码为 H.264，音频会重编码为 AAC，也不是无损拼接。

既有124/102/102帧真实 H3 三段已完成一次 `none`/5ms bridge 文件级对照：两份输出均为
328帧，视频13.6667秒、AAC 13.667秒；最终 AAC 解码后的两处边界跳变从约
0.04226/0.03509降至0.00434/0.00704，约下降89.7%/79.9%。这是单素材的零阶幅值指标，
尚未完成盲听和多素材 click energy/响度/频谱验证，不能据此宣传“音频无缝”。

这不是“显存优化节点”的证明。124帧目标加22帧运动条件约增加18.9%的视频条件行，音频
timeline 也增加约17.9%的音频 reference 行；这些是 packed rows 比例，不是显存百分比。
分段只能让总时长的峰值有界，单个带上下文片段一定比同档普通片段更重。Block Cache、Sage
和 DynamicVRAM headroom 的首轮受控矩阵及本机60秒门槛现已完成，结论见本节后文；它只支持
一个固定本机保守档，不提供通用`memory_safe`宣传，也不承诺任意尺寸、任意帧数下不会OOM。

2026-08-08 的四步实测使用非裁剪 FL2VA INT8、NVFP4 H3 CLIP、两个 H3 VAE、Standard
四步 LoRA、736×416、124帧窗口和 DynamicVRAM：direct 22帧 AV context 的三段链全部完成，
原生输出为124/102/102帧，视频与音频流时长一致。三次设备峰值约15,998/15,881/16,135MiB，
余量都低于512MiB候选门槛。相同素材的三路 A/B 中，单末帧接缝显著差于两条22帧路线；
VAE重编码22帧没有显示出优于直接 sampler latent 的充分证据，且暖态运行多约17.25秒，
所以正式节点继续只保留 direct latent 默认。未处理的原始分段音频边界跳变仍接近局部最高值；
该首版 bridge 检查当时只覆盖上述三段单素材；后续14段证据见下文。多素材长期退化矩阵和
跨配置通用16GB安全档仍未完成，因此本功能继续保持 Experimental，不宣传无缝或绝不 OOM。

旧手工链的画布/API 示例仍为 `examples/workflows/2026-08-09_H3_Long_Video_22F_EXP.json` 与
`tests/fixtures/api/long_video_segment_api.json`。接受状态画布/API 示例为
`examples/workflows/2026-08-09_H3_Long_Video_Accepted_22F_EXP.json` 与
`tests/fixtures/api/long_video_candidate_accept_api.json`；完成全部片段后再单独运行
`tests/fixtures/api/long_video_compose_api.json`。推荐的总时长自动恢复画布/API 示例为
`examples/workflows/2026-08-09_H3_Long_Video_Auto_Resume_22F_EXP.json` 与
`tests/fixtures/api/long_video_auto_resume_api.json`；它自动管理 index、final、时间轴和断点位置，但保留
逐段人工预览/接受。显式后台画布/API 为
`examples/workflows/2026-08-09_H3_Long_Video_Background_22F_EXP.json` 与
`tests/fixtures/api/long_video_background_api.json`；只有这组示例连接 Background Start 与 Auto Queue。

本机已对这条自动恢复 API 做一次真实执行探针：非裁剪 FL2VA INT8、Standard Turbo LoRA、
NVFP4 H3 CLIP、双 H3 VAE、736×416、124帧内部窗口、1步、目标1秒，并启用 DynamicVRAM。
真实联合采样、裁成24帧候选、接受、完成后再次排队阻断，以及 accepted 文件合成全部成功；
候选和最终 MP4 都是24fps、24帧，视频/音频/容器均为1.000秒。该结果只证明新工作流执行闭环，
不代表四步多段画质、60秒质量或16GB显存安全档已经通过。

随后又完成一条相同推荐 API 的真实四步双段检查：6秒精确拆为124帧与20帧，第二段自动读取
已接受的22帧 AV context 和父候选身份；最终 manifest 覆盖144帧，完成后重排没有新增候选。
`none` 与5ms bridge 合成都严格为24fps/144帧，视频、音频和容器均6.000秒。bridge 使最终
AAC边界的单样本跳变约下降80.2%，但段前后仍有约33.3dB响度落差；视频边界静帧没有明显
身份/构图跳切，不过MAD和SSIM不连续度均是附近16个片内转场中的最高值。两段设备峰值约
15,461.4/16,181.5MiB，第二段只余约198MiB，因此仍只能称为“本机跑通、可续接”，不能称为
音画无缝或16GB安全档。该双段检查本身不代表长期链已经通过。

同日又在一个未重启的 DynamicVRAM 进程中完成了首条真实四步60秒长期链：14段按
`124 + 12×102 + 92 = 1440` 精确接受并合成，视频24fps/1440帧，视频、音频和容器都严格为
60.000秒。整个链没有显式调用 `/free`；14段设备峰值位于15,480.0–16,228.2MiB，暖态峰值的
描述性线性斜率约为每段+28.0MiB，基线没有单调阶梯增长，因此这一次运行没有显示累积型显存泄漏。
但第12段只余约151.3MiB，共5段低于512MiB安全余量，所以该配置仍不能标为16GB安全档；
0.25秒轮询也可能漏掉更短的分配尖峰。

13个视频边界的MAD中位数为0.01618、最大0.01906，SSIM中位数为0.96374、最小0.92868。
最差接缝的接触图没有显示主体或背景的硬切，但14段中间帧时间轴可见人物外观和曝光逐步漂移，
像素/光流指标也不能证明身份保持。音频的长期退化更明显：接缝前后半秒响度变化中位数约
-9.51dB，最大绝对变化约40.83dB；首末段8kHz以上能量占比相差约-36.30dB，说明递归续写出现
明显变闷/频谱漂移。5ms bridge 将最终AAC边界单样本跳变的中位数降低约97.23%，但不能修复
响度、音色、对白语义或口型连续性。因此这条结果证明了“60秒、14段、可恢复、定长合成”的
执行闭环，不证明无缝、身份无漂移或长期音频无损。仍需补人物对白/口型、快速运动、节奏音乐、
多素材多seed、盲听/ASR/说话人/唇形评估，以及跨GPU/高分辨率/多参考显存档。

最后一段接受后的首次完整重排在当前 ComfyUI 中会以预期的 `ExecutionBlocked` 终止（空 traceback，
只执行到编排节点）；命中缓存后的复排也可能显示成功但只运行审核节点。两种情况下候选数都保持14，
不会生成第15段。这是安全完成阻断，不应把预期的 `ExecutionBlocked` 当作生成故障。

5帧/22帧现已完成0.3M与0.6M的重复矩阵，而不再只是单次试探。两档均固定复用各自仅接受
第0段的基线；同分辨率下第0段MP4以及视频/音频tail tensor均bit-identical。每档使用3个
配对seed、交替顺序执行3次独立冷启动；另在同一进程primer后执行3次暖态。相同context+seed
的全部冷/暖候选也bit-identical，VRAM按0.10秒采样。

0.3M（736×416）冷启动5/22帧整机峰值均值为15,279.5/15,224.0MiB，但三组`22-5`配对差为
+96.6/-78.3/-184.9MiB，方向不一致。Sampler PyTorch pool则稳定为约3,189.9/3,495.3MiB，
5帧少约305MiB；冷态耗时均值86.53/93.08秒，暖态69.27/78.01秒。暖态最低余量仅97.6MiB，
5/6次低于512MiB门槛。三个seed平均上，22帧的视频MAD/SSIM与音频响度/NCC更好。

0.6M（1056×608）冷启动5/22帧整机峰值均值为15,739.0/15,724.2MiB，三组`22-5`配对差却为
-752.0/+10.8/+696.7MiB；绝对峰值同样不能归因于context。Sampler pool稳定为约
5,753.4/6,381.2MiB，5帧少约628MiB；冷态耗时均值200.29/230.38秒，暖态187.89/218.40秒。
暖态6/6次均低于512MiB余量，最差只剩33.6MiB。该组三seed中5帧MAD/SSIM平均更连续，但可能
包含运动被压低；22帧音频响度/NCC明显更好，且有一个seed出现正面到侧面的明显画面边界跳变。

39帧随后在0.3M完成同一批3个seed的3次独立冷启动和primer后3次暖运行。六次均成功，且相同
seed的冷/暖MP4 bit-identical；5/22/39三条链的第0段MP4及AV tail tensor也完全一致。39帧的
sampler pool冷/暖均约3,799MiB，相对22帧稳定多约303–304MiB；冷/暖耗时均值为101.65/87.38秒。
但暖态3/3次低于512MiB，最低只剩77.35MiB。人工检查三张接缝图时，只有1个seed较连续，
1个有明显姿态/构图跳变，另1个发生严重人物身份和镜头关系变化。

因此5帧只保留为`fast_context_5_experimental`候选：它确实更快并减少sampler activation pool，
但没有可重复的整机峰值优势。22帧继续作为当前默认平衡候选；39帧降级为
`context_39_high_risk_experimental`，既不是质量档，也不是安全档。旧策略下0.6M/39帧没有强行执行：
0.6M/22帧暖态已经6/6低于512MiB、最低33.6MiB，继续增加5个latent step没有建立安全档的
可能且有真实OOM风险。这是预定义安全门槛否决，不等于已经证实0.6M/39必然OOM。任何档位在
绑定具体硬件、模型、分辨率和插件组合后通过至少512MiB余量前，都不能命名为`memory_safe`。

2026-08-09 又完成了原生 DynamicVRAM `headroom=2.0GiB`、Stock/Sage和Block Cache的受控检查。
736×416与1056×608各用3个seed完成3冷3暖；Stock和Sage的全部试次均高于512MiB，相同策略的
冷/暖同seed输出bit-identical。默认Block Cache在4次前向中0次命中、CPU cache约117.7MiB，
不能跳过首次完整前向，因此不作为OOM默认方案。Sage虽更快，但同headroom下整机峰值反而比
Stock高；0.6M的3个seed中有2个出现明显镜头/姿态/运动轨迹分叉，只保留为高风险近似加速实验项。

最终采用`Stock + DynamicVRAM headroom 2.0GiB`重跑真实60秒/14段链，全程不重启且不显式`/free`。
14/14段、manifest revision 14、1440帧和1,920,000 audio samples全部完成；两份合成都是
736×416、24fps、1440帧，视频/音频/容器严格60.000秒。峰值范围12,829.44–13,640.09MiB，
中位13,137.67MiB，最低仍空闲2739.41MiB；暖态峰值不单调，未见典型阶梯泄漏。

相对旧`headroom=0.5`的同提示词/同seed Stock链，14/14段MP4 SHA-256以及13/13个续写
`video_tail`/`audio_tail`张量完全一致，说明调整的是原生内存调度而不是采样数值；峰值中位数
降低约2635MiB，总生成时间增加约1.63%。因此该组合可以称为**本机固定配置已验证保守档**，
但只覆盖RTX 4060 Ti 16GB、FL2VA INT8、Standard四步LoRA、736×416、124帧窗口、22帧context
和本次插件集合。其他GPU、0.6M长链、更多参考媒体或桌面显存占用仍可能OOM，不能宣传通用
`memory_safe`或“绝不爆显存”。0.6M/39帧也需在新策略下另做多seed显存与质量门槛。

随后又保持相同prompt、模型、画布、窗口、context和Stock+h2策略，只改变base seed为
`2608082000`、`2608083101`、`2608083202`，分别启动三个独立ComfyUI冷进程执行60秒/14段链。
三链共42/42段一次成功，没有OOM、重试或候选复用；manifest、父候选/revision链、候选与
accepted视频/context SHA-256、1440帧、1,920,000 samples、完成阻断以及六份60.000秒成片均
独立复核通过。每条链的最大峰值分别为13,640.09、13,414.01、13,426.72MiB，最差空闲余量
2739.41MiB，没有片段低于512MiB。这关闭了**本机固定档跨base-seed冷启动机械/显存门槛**，
但同seed整链暖重复、跨prompt/多素材、其他GPU和桌面负载仍未验证。

质量门槛没有随之通过：三个14段中间帧时间轴都出现逐段面部年龄与身份漂移，seed
`2608083101`最严重；三链音频相邻半秒窗最大响度差为23.59–48.06dB，描述性NCC中位仅
0.127–0.206，首末段8kHz以上能量占比下降9.66–36.30dB。5ms bridge把后AAC单样本跳变中位
降低94.93%–97.33%，但不能修复响度、音色、语义或递归变闷。因此仍不能宣传长期身份稳定、
音频无损或无缝；本地完整报告在
`artifacts/long-video-generation-check/stock-headroom2-60s-multiseed/analysis/REPORT.md`。

同一 ComfyUI 提交在 `--novram` 下，连本体自带的
`EmptyMiniMaxH3LatentAV -> VAEDecodeAudio` 也会独立复现 CUDA 输入与 CPU filter 的设备不一致；
因此这不是 T8 AV Decode 或 Orchestrator 引入的错误。当前建议使用本机已验证的 DynamicVRAM
路线；在 ComfyUI 本体修复或本项目有可靠的局部兼容方案前，不宣称 H3 Audio VAE 的
`--novram` 解码可用。

## EXP：参考图像编辑

`MiniMax H3 Reference Image Edit (EXP/T8)` 位于
`T8/MiniMax H3/Still/Experimental`，复用 H3 Ref2VA 的 Picture 条件生成静态候选。
`edit_image` 始终是 `<Picture 1>`；附加参考图依次成为 `<Picture 2>` 至 `<Picture 9>`。
Prompt 应明确每张图的职责，例如主体身份、服装、背景或光照。

目标模式：

- `direct_1_frame`：直接创建 `video latent_t=1`，成本最低，但严重偏离训练帧数；
- `micro_video_5_frames`：生成 H3 最短 5 帧，再在 Still Decode 中选帧；
- `short_video_22_frames`：生成下一档原生 `17n+5` 网格的22帧，视频 latent T=7，
  音频 latent T=37；比124帧便宜很多，但仍低于约124帧的训练下限；
- `trained_124_frames`：按近似训练下限生成 124 帧，作为质量基准，成本最高。

默认 `reference_strength=0.999` 与 H3 参考条件的原始噪声增强接近；降低该值会向参考
latent 注入更多噪声，可能增强重绘幅度，也可能损坏身份与构图。`generate_and_discard`
让联合模型正常生成短音频但最终不解码；`lock_silence` 锁定零音频，仅用于对照。

推荐链路：

1. 加载 H3 Ref2VA 模型、H3 Qwen3-VL CLIP 和视频 VAE；
2. 将主图和附加参考图接入 Reference Image Edit；
3. 同一个 `av_latent` 同时连接到双时钟采样设置与 `SamplerCustomAdvanced.latent_image`；
4. 采样输出接 Still Decode，再接 `SaveImage`。

本机现有 Ref2VA 是 pruned INT8，不能完整应用本项目转换的 Turbo LoRA；示例因此不加载
LoRA，并以 20 步作为结构基线。若以后安装非裁剪 Ref2VA，再单独进行 Turbo LoRA 对照。
这项能力是参考引导的语义重绘，不是 mask/inpainting，也不保证未编辑区域像素不变。
API 示例见 `tests/fixtures/api/still_image_edit_api.json`；可直接拖入画布的完整示例见
`examples/workflows/2026-08-07_H3_Still_Edit_22Frames_EXP.json`。两者默认使用512×512、22帧、20步，
并连接 Still Preflight；在 Reference Image Edit 节点上点击“＋”可追加最多8张参考图。

本机真实模型验证中，pruned Ref2VA INT8 在 512×512、20 步、`direct_1_frame` 下成功
保留手袋主体并把黑色皮革改成深红色；相同任务在 128×128 下结构明显崩坏。因此默认推荐
`canvas_mode=from_edit_image`，自定义画布短边不要低于 512。该结果只是单个可用案例，
不能代替多图、不同主体、不同编辑类型和多种 seed 的系统质量评估。

## H3 Turbo 四步双时钟采样

H3 的视频流默认使用 shift 12，音频流使用 shift 3。旧版 ComfyUI 的 H3 DiT 会把音频
速度乘上 `d(sigma_audio)/d(sigma_video)`；当前 ComfyUI 已改为 `FLOW_AV`，模型返回原始
音频速度，并由原生 `ModelSamplingAV` 支持音频 carry/scale。T8 双时钟节点自己维护两个
时钟，因此会检测实际基模协议：旧版移除 schedule slope，当前版直接按音频 sigma 差积分，
同时把自定义 sampling 的 `audio_scale` 固定为 `1.0`，避免重复缩放。

`MiniMax H3 Dual-Clock Sampler (T8)` 每步仍只做一次联合 AV 模型前向，不拆开模型，
但更新 latent 时执行：

- 视频：`delta_video * velocity_video`；
- 音频：旧协议先除去 schedule slope，当前协议直接使用原始速度，再乘 `delta_audio`；
- mask=0 的锁定区域保留 ComfyUI 原有的 inpaint 时钟，完整生成区域使用音频时钟。

四步 Turbo 推荐连接：

1. `UNET/Diffusion Model Loader -> LoraLoaderBypassModelOnly -> Dual-Clock Sampler.model`；
   当前 INT8/量化模型不要改用普通 LoRA 合并链并假设结果等价。
2. Conditioning/Empty H3 AV Latent 的同一个 `av_latent` 同时连接到
   `Dual-Clock Sampler.av_latent` 和 `SamplerCustomAdvanced.latent_image`。
3. Dual-Clock 的 `model` 接 `BasicGuider.model`，`sampler` 和 `sigmas` 分别接
   `SamplerCustomAdvanced` 的同名输入。
4. `steps=4`、`shift_video=12`、`shift_audio=3`、`sampler=dual_clock_euler`、
   `scheduler=native_flow`。LoRA 强度使用作者建议值。

节点内部现在可选择采样器和调度器：

| 控件 | 默认值 | 行为与兼容范围 |
|---|---|---|
| `sampler / 采样器` | `dual_clock_euler` | 原有 T8 显式双时钟 Euler，数值路径不变；兼容旧版与当前 ComfyUI |
| 其他采样器 | 无 | 使用当前 ComfyUI 自带的 sampler，并切换到原生 `ModelSamplingAV` carry/scale；旧版 ComfyUI 不提供这些选项 |
| `scheduler / 调度器` | `native_flow` | 原有 shifted-uniform H3 flow sigma，数值路径不变 |
| 其他调度器 | 无 | 调用当前 ComfyUI 的同名 scheduler；改变 sigma 时间网格，不承诺一定改善 Turbo 画质或音质 |

`dual_clock_euler` 配其他调度器时，仍由 T8 显式维护视频/音频两个时钟；其他采样器则由
当前 ComfyUI 原生 `FLOW_AV` 协议把联合 latent 映射为单一求解时钟。两条路径不能混用
carry/scale。标准采样器只在新版原生协议存在时开放，因为旧版 H3 没有可证明等价的通用
多阶求解适配。

这个节点已经代替 `MiniMax H3 Sigma Shift`、`KSamplerSelect` 和 scheduler 三个节点。
不要再串联一次 Sigma Shift，也不要外接 `KSamplerSelect` 或 `BasicScheduler`；需要更换时
直接使用本节点新增的两个下拉框。
`SamplerCustomAdvanced`、`RandomNoise` 和 `BasicGuider` 仍照常使用。

可导入的 API 结构示例见 `tests/fixtures/api/dual_clock_4step_api.json`。其中模型文件名是占位符，
请替换为本机的 H3 基模、两个 VAE、Qwen3-VL CLIP 和已转换 LoRA 文件名。旧 API JSON
可以不提供 `sampler_name` 与 `scheduler`，后端会使用上述两个默认值。

## EXP：视频 4 步、音频更多步

`MiniMax H3 Multi-Rate Sampler (EXP/T8)` 位于独立的 `Experimental` 分类，代码也在独立
模块中，并使用与稳定版相同的新旧 ComfyUI 音频速度协议检测。EXP 节点把视频
Euler 更新保持为 `video_steps` 个宏步，同时在每个宏步内部为音频安排更多微步。例如：

- `video_steps=4, audio_steps=8`：每个视频区间 2 个音频微步；
- `video_steps=4, audio_steps=10`：四个区间均衡分配为 2、3、2、3 个音频微步；
- 四个视频宏时间边界与稳定 4 步网格完全一致。

H3 是联合音画 Transformer，无法只计算音频分支。因此 `audio_steps` 也是实际的完整 H3
DiT 前向次数：4/8 约是稳定 4/4 的 2 倍计算量，4/10 约是 2.5 倍，并会同时受到显存和
耗时影响。视频 latent 只在四个宏边界提交更新，但每个音频微步仍需联合模型前向。

建议先用相同 seed、prompt 和输入做 4/4 稳定版与 EXP 4/8 对照；若音频仍明显不够，再试
4/10。更多步不保证一定更好，因为 Turbo LoRA 的训练设计点仍是四步，额外中间时间点可能
改善音频数值积分，也可能产生分布外误差。EXP 不应直接替代已验证的生产工作流。

连接方法与稳定版相同，只把三个输出接入 `BasicGuider` / `SamplerCustomAdvanced`；不要再
叠加 Sigma Shift 或外部 scheduler。示例见 `tests/fixtures/api/multirate_exp_api.json`。

## 四种音频模式

| 模式 | 目标音频 latent | 源音频是否作为参考 | 适用场景 |
|---|---|---|---|
| `lock_source` | 源音频，denoise mask=0 | 默认是 | 画面严格跟随音频，最终保留原音轨 |
| `remix_source` | 源音频，按 strength 重绘 | 默认是 | 保留节奏/语音结构，同时让模型改造声音 |
| `reference_only` | 空白、完整生成 | 是 | 源音频只提供语义/节奏参考，输出使用模型音频 |
| `native` | 空白、完整生成 | 否 | 纯 H3 原生音画联合生成，无需输入音频 |

`drive_audio` 是给模型的驱动轨，`final_audio` 是最终 mux 的干净轨。二者分开可以让你把
外部人声分离器得到的 vocal stem 用作驱动，同时把原混音或另一条 stem 送到最终输出；
本包不会假装内置了一个未经验证的分离模型。

可直接拖入画布的音频示例：

| 工作流 | `audio_mode` | 最终 MP4 音轨 | 用途 |
|---|---|---|---|
| `2026-08-06_H3_Audio_Lock_Source_Stable_4V4A.json` | `lock_source` | Conditioning `mux_audio` | 锁定源 latent，保留干净输入原音轨 |
| `2026-08-06_H3_Audio_Remix_Source_Stable_4V4A.json` | `remix_source` | AV Decode `generated_audio` | 以默认0.35强度保留节奏/语音结构并重绘声音 |
| `2026-08-06_H3_Audio_Reference_Only_Stable_4V4A.json` | `reference_only` | AV Decode `generated_audio` | 输入音频仅作 `<Audio 1>` 参考，目标音频重新生成 |
| `2026-08-06_H3_Turbo_Stable_4V4A.json` | `native` | AV Decode `generated_audio` | 无需输入音频的原生音画联合生成 |

三份输入音频示例均预设736×416、124帧、稳定4/4双时钟、原生 flow 调度，并通过 Audio Window
把用户选择的5秒场景对齐到合法 H3 窗口，再由 Output Trim 恢复精确5秒。导入后必须先在
`Load Audio` 中选择或上传音频。切换 `audio_mode` 时不要只改下拉框：`lock_source` 的最终音轨
应取 `mux_audio`，而 `remix_source` / `reference_only` 应取模型解码的 `generated_audio`。
其中 `reference_only` 仍需要把输入音频接入 `drive_audio`，只是不会把它注入目标音频 latent。
若另接一条干净轨到 `final_audio`，它只会替换 Conditioning 的 `mux_audio` 输出；要把它用于最终
MP4，仍需将 `mux_audio` 明确连接到 Output Trim 的 `audio`。

## 推荐连接

锁定原音频生成画面：

1. `Load Audio -> MiniMax H3 Audio Window (T8)`。
2. `context_audio`、视频 VAE、音频 VAE、CLIP 接入统一 Conditioning，选择 `lock_source`。
3. Conditioning 的 `positive` 和 `av_latent` 进入原生 H3 sampler。
4. sampler 输出进入 `MiniMax H3 AV Decode (T8)`。
5. 解码 frames、Conditioning 的 `mux_audio`、Audio Window 的两个 trim 输出进入
   `MiniMax H3 Output Trim (T8)`。
6. 将裁切后的 frames/audio 交给 VideoHelperSuite 或你现有的保存节点。

短场景开启 `ensure_minimum_context` 时，节点会添加上下文，但不会再让动作时间轴悄悄漂移：
`prompt_timing_note` 给出主场景在渲染窗口中的真实开始/结束时间，最终 trim 参数再恢复用户请求时长。

## 媒体编号

H3 的展示顺序是：所有 Picture；然后每个参考视频（其声轨 Audio 标签位于对应 Video
标签前）；最后是独立 Audio。因而两个参考视频都带声轨时，主驱动音频会是 `<Audio 3>`，
而不是 `<Audio 1>`。统一 Conditioning 会输出完整 `media_map_json`，并把 prompt 中配置的
`prompt_primary_audio_ordinal` 自动映射到主驱动音频的真实编号。设为 0 可关闭重映射。

严格模式会拒绝引用未连接媒体的标签，避免模型收到看似合法、实际无对应条件的 prompt。

## H3 边界

- 固定 24fps，帧数向上对齐到 `17n+5`。
- 当前模型近似训练区间为 124–362 帧；区间外允许规划但 Preflight 会警告。
- 生成画布像素面积不能超过 `1920×1088 = 2,088,960`，宽高必须是 32 的倍数。
- 超过 `1344×768 = 1,032,192` 像素不再报错，但 Preflight 会提示显存需求显著增加；
  模型支持该画布不代表所有帧数、参考数量和显卡都能在相同显存内运行。
- 原生 H3 目前只支持 batch size 1。
- 引用上限：9 张 Picture、3 个 Video、3 个独立 Audio；参考视频官方建议 2–15 秒。
- `Hybrid` 同时使用精确首/尾帧和参考媒体。节点包含针对当前 ComfyUI `PackedLayout`
  行为的运行时契约检查；上游若改变结构会明确停止，而不是生成错位条件。

官方建议的 16:9、32 倍数尺寸可直接使用：

| 约百万像素 | 输出尺寸 |
|---:|---:|
| 0.2 | 608×352 |
| 0.3 | 736×416 |
| 0.4 | 864×480 |
| 0.5 | 960×544 |
| 0.6 | 1056×608 |
| 0.7 | 1152×640 |
| 0.8 | 1216×672 |
| 0.9 | 1280×736 |
| 0.98 | 1344×768 |
| 1.0 | 1376×768 |
| 1.2 | 1504×832 |
| 1.5 | 1664×928 |
| 1.8 | 1824×1024 |
| 2.0 | 1920×1088 |

## 高速动态质量实验（Advanced）

v1.19.0 追加两个完全旁路旧工作流的实验节点：

- `MiniMaxH3AVSigmaTailSubdivisionT8Advanced`：接在双时钟节点的 `sigmas` 与
  `SamplerCustomAdvanced` 之间。默认 `report_only + extra_substeps=0`，逐位透传输入；开启实验时
  只在base-flow时钟中插点并保留全部原始knots，同时报告video/audio sigma、真实NFE与schedule SHA。
  Turbo增加中间时间点属于训练网格外实验，必须显式确认；每个新增点都是一次完整联合A/V DiT前向。
- `MiniMaxH3MotionQualityAuditT8Advanced`：对已解码IMAGE帧做只读时序代理审计，可用全帧、人工静态ROI
  或外接MASK，输出风险区间与`17n+5`合法修复窗口。它不修改成片、不加载或自动下载人脸模型，
  也不声称已完成人脸检测、身份验证或质量改善。

新的 Turbo 双时钟质量、兼容、性能和显存测试统一按视频8步/音频8步。旧4步工作流继续保留，
用于验证向后兼容和复现历史结果，不再作为后续质量结论的统一测试标准。新示例
`2026-08-09_H3_Motion_Quality_Advanced_8Step_EXP.json`默认不应用sigma插点，先提供严格8步基线和只读审计。

v1.28.0 又追加两个隔离的 Advanced 节点，旧107个节点及稳定采样数学不变：

- `MiniMaxH3DynamicCFGGuiderT8Advanced`：默认`passthrough_basic + 1.0→1.0`，不安装CFG或
  model wrapper，等价回到BasicGuider路线。`single_condition_gain_exp`只对正条件预测施加随sigma变化的
  单路增益，**不是真正CFG**；`true_cfg_exp`需要匹配的negative、双分支成本确认，仍未做H3质量验证。
- `MiniMaxH3DynamicGuidanceAuditT8Advanced`：放在采样后，原样透传联合AV latent，并报告实际Guider
  调用、物理模型前向和cond/uncond分支批次。它只负责审计，不修画面或声音。

对应前端示例为`2026-08-09_H3_Motion_Quality_Dynamic_Guidance_8Step_EXP.json`与
`2026-08-09_H3_Motion_Quality_Extra_Tail_NFE_8Step_EXP.json`。二者均默认关闭实验效果并附带NOTE。推荐只做四条
单变量检查：8步Basic基线、尾段额外2 NFE、普通10步因果对照、动态单路增益`0.90→1.10`；不要再扩成
大矩阵。尾段2个插点会把8次提高到10次完整联合A/V DiT前向，约增加25% NFE；动态引导与尾段调度
都是生成阶段预防实验，不是Face Refine、锐化或后处理。

本机现已完成同一清晰736×416首帧、同prompt/seed、124帧、24fps、Turbo Standard 8步且
Block Cache关闭的四条真实生成：S0原始8 NFE、S1尾段增加2次完整前向、S2普通10 NFE、G1单条件
增益`0.90→1.10`。四条均严格解码为H.264 736×416×124@24fps与32kHz双声道AAC；G1审计实测
8次predict、8次CFG callback、8次物理模型前向、8次cond与0次uncond，证实它不是true CFG。
这只关闭机械执行门，完整视频/声音人工判断仍未完成；可选SG1组合按计划未生成。冷态S0最低整卡
余量仅311.1MiB，低于512MiB门槛，因此`quality_guarantee=false`与通用
`memory_safe_claim=false`保持不变。

### v1.29.0 五条细节路线

本版在旧109个节点之后追加五个隔离节点；稳定`sampling.py`、旧节点ID、输入输出、默认值及既有
工作流均未改动：

- `MiniMaxH3AVTailDetailScheduleT8Advanced`：连接后默认在最后一个双时钟区间增加1次完整联合
  AV前向；设为3时，视频sigma从最后非零值按75%、50%、25%逐级降到精确0，音频按shift12→3
  的原生时钟映射。末尾0只是积分终点，不会多调用模型，也没有随机升噪。
- `MiniMaxH3ModelTimeBiasSamplerT8Advanced`：真正的`sin²`尾段包络，只改变共享AV Transformer
  看到的模型时间，积分轨迹和NFE保持不变。它借鉴Navyblue/Detail Daemon的科学机制，但不是复制
  Navyblue的常数sigma倍率，也不是CFG或锐化。
- `MiniMaxH3RectifiedFlowRestartSamplerT8Advanced`：先完成基础双时钟采样，再按
  `x_sigma=(1-sigma)*x_clean+sigma*epsilon`对联合AV端点重新加噪并二次下降；报告视频/音频sigma、
  seed和真实额外NFE。只允许二值视频mask且要求完整音频latent参与；锁音频或分数mask会拒绝，
  不会把实际video-only误报成联合重启。H3音画共用Transformer，因此首版不提供未经验证的
  “只重启视频”伪隔离。
- `MiniMaxH3SpatioTemporalGuidanceT8Advanced`：在选定进度内增加一条跳过指定H3 double block的
  正条件分支；每个生效步增加一次完整联合AV前向，节点执行和真实采样时都会检查同block
  replacement，后接Block Cache也不能静默覆盖。非H3 MODEL和非零共享AV rescale会拒绝。
- `MiniMaxH3TemporalDetailEnhanceT8Advanced`：解码帧上的运动门控亮度细节增强，可选按32倍数放大；
  高运动区自动减弱以控制闪烁，不接触音频，也不能凭空重建缺失的脸、身份或几何。默认8帧分块
  带一帧halo，放大采用保比例优先且不缩小的32对齐，默认2.1MP输出预算会阻止危险组合。

这五条路线必须分别比较，不能叠在一起后声称因果成立。前三条和STG都会改变共享音画预测，因而
音频是否非劣必须靠完整试听判断；“音频冻结”没有被当成硬约束。Restart是真随机重启，风险与成本
都高于尾段细分；STG在生效步额外跑一次Transformer，显存和耗时也不等同于普通8步。

固定红色汉服高速旋转实测使用`10A.jpg`、1152×640（737,280像素）、124帧、24fps、seed
`2608172801`、非pruned FL2VA INT8、Qwen NVFP4、官方双VAE、Turbo LoRA和双时钟8步。五条候选均
完成真实生成，并分别输出124帧H.264、32kHz双声道AAC；每个新文件3/3次FFmpeg `-xerror`
完整解码通过，A/V流时长差14.67ms，小于一帧。tail+3、RF Restart+3分别执行11次联合AV前向；
STG只在25%～85%区间增加skip-block分支；model-time bias仍为8 NFE；temporal detail在解码后处理。

旧8步对照文件保留用户此前接受的一个已知坏帧，OpenCV仍读出124帧，但严格解码为失败；它只作为
历史匿名对照，不计入新节点通过数。model-time bias首轮采样已完成，但VHS写出的中间MP4缺少moov，
删除该损坏文件后利用Comfy缓存重存成功；这是保存节点异常，不是模型时间数学或生成失败。代理锐度、
运动和音频电平只用于发现异常，不能据此选胜者。完整六路匿名页位于本地
`artifacts/h3-detail-routes-v1/blind/blind_review.html`，在人工完整观看与试听前，五条路线仍保持EXP，
不声明普遍更清晰、音频非劣或通用16GB安全。

2026-08-16 已完成首轮3类素材×3seed×Stock20/Standard8/EMA8/FL2V8×control/same-NFE-tail，
共72/72次真实124帧运行。严格视频解码6轮共432/432次通过；每条输出均为有限32kHz双声道音频，
A/V时长差不超过一帧。FL2V处理组的预注册运动代理0/9通过，已否决。其余27个已完整评审的
profile-pair中，画面偏好为20平、5次same-NFE-tail、2次control，声音27/27判平；这不足以证明
尾部重分配具有稳定感知优势。完整36组最终记录为画面29平、5次same-NFE-tail、2次control，
声音36/36平；其中FL2V原先漏选的5个偏好仅在评审者明确说明“漏填就是平”后补记为平，未补造
缺失的1–5评分或语音判断。

这批首轮素材还暴露出一个测试设计错误：I2VA首帧以`crop=disabled`直接缩放至固定736×416，
两张900×1600竖图被横向相对拉宽约3.15倍，另一张3027×1531图也发生约10.5%的非等比变化。
因此该批运行只能保留机械执行、解码、音频时长和显存观察，不能用于绝对脸部质量、身份保持或
产品晋级。修正矩阵应使用与素材匹配的横/竖画布或已审核的同画幅素材后重跑。各profile至少有
一次整卡最低余量低于512MiB，3冷3暖尚未完成，故仍不能称为“修脸”“高速动态修复”或
`memory_safe`，`quality_guarantee=false`与`memory_safe_claim=false`保持不变。

保持源图比例的第二轮计划现已生成到
`artifacts/motion-quality-same-nfe-v3-aspect-safe/`，但尚未启动生成。它为3027×1531横图使用
768×384（相对宽高比误差约1.16%），为两张900×1600竖图使用416×736（误差约0.48%）；
72/72个API prompt与36/36组A/B均通过静态指纹检查。逐任务对照确认，除画布、时间线画幅标签和
输出前缀外，其余模型、LoRA、步数、seed、prompt、调度和same-NFE变量与首轮一致。计划仍保持
`quality_decision=not_evaluated_aspect_safe_matrix`与`memory_safe_claim=false`。2026-08-16用户明确
决定不再执行这72条复测，并确认其16GB实际工作流没有问题、直接通过。项目因此将精确本机/用户
使用场景记为`local_16gb_operational_acceptance=passed_by_user_confirmation`，将72条复测与3冷3暖
记为`waived_by_user`而不是失败；原话、manifest哈希和边界保存在同目录`user_acceptance.json`。
这不是伪造的重复测量，也不外推到所有16GB显卡、分辨率、帧数、驱动、wrapper或并发负载。

## v1.30.1：老工作流参数错位兼容热修

本版没有调整任何稳定节点输入顺序、类型、默认值或`sampling.py`数学。问题来自此前的
API→前端工作流转换器按API字典顺序写入`inputs/widgets_values`，而ComfyUI按节点schema顺序读取，
导致部分示例在重新打开时出现宽高、长度、枚举和布尔值串位，甚至显示`NaN`。

- 转换器现在强制依据在线`object_info`的required/optional完整顺序序列化，并重建真实连线槽位；
- 独立修复工具只在检测到输入顺序、控件或已连接槽位异常时修改旧文件，正常旧工作流保持不变；
- 40份项目工作流及用户目录中的对应40份副本已恢复，原文件备份保存在`artifacts/workflow-order-repair-*`；
- 63份项目与60份用户前端JSON已严格解析，在线schema复扫为0残留，完整610项测试全部通过。

磁盘修复不会改写浏览器内已经打开的画布；请关闭旧标签后从工作流菜单重新打开，或刷新页面后重载。

## v1.30.0：MiniMax H3 SPEED 空间渐进采样（Advanced）

本版不是复制`ComfyUI-MiniMax-H3-SPEED`。实现以
[`howardhx/speed@ca7801c9`](https://github.com/howardhx/speed/tree/ca7801c9bdffe681742e9592345bcf4885959be5)
和论文[`arXiv:2605.18736v3`](https://arxiv.org/abs/2605.18736v3)为数学基线，clean-room实现：

- 早期在较小空间latent上去噪，时间轴、帧数和音频长度不变；
- 切换分辨率时用正交DCT保留原低频系数，用当前视频sigma幅度的高频高斯系数补齐新频段；
- 按`κ(t,r)=r/[1+(r-1)t]`缩放状态，并把下一阶段起始sigma改为`tκ`；
- 总NFE保持不变，例如20步两阶段仍是20次DiT前向，不是Block Cache、跳层或额外细化；
- 不使用WAN的`A=219.48、β=2.4227`，也不使用WIP猜测的`A=150、β=2.0`冒充H3标定。

四个节点职责分开：

1. `SPEED Spectrum Harvester Advanced`从已分离的H3视频latent拟合`P(ω)=A|ω|^-β`。单片只标
   `research_probe_only`；必须在一次输入里实际提供至少100条batch样本、声明其独立数据来源、填写
   checkpoint/VAE指纹并达到设定R²才可作为dataset profile。节点能核对batch数量，不能从tensor本身
   证明统计独立性；手填更大的样本数不能把单片升级成已标定profile。
2. `SPEED Plan Advanced`把每级画布解析为同时被32整除、宽高比误差受控的实际尺寸，生成手工sigma或
   delta-optimal阶段表，并证明NFE守恒。κ严格使用官方定义的请求scale比，不把32整除后的grid比误当成r；
   两个值都会写入报告。默认用手工sigma，因为当前尚无通过门槛的H3数据集profile。
3. `SPEED Stage Source Advanced`保存原始prompt、首尾帧、参考图/视频/音频和双VAE/CLIP引用，不预编码
   第二份H3模型。这样每个分辨率阶段都能重新缩放、VAE编码并重建keyframe/ref/PackedLayout；若使用
   delta-optimal，还会把profile的task/checkpoint/VAE指纹与当前Source绑定，不匹配即拒绝。严格纯T2VA
   没有空间条件，因此只编码一次Qwen文本，后续阶段复用文本条件并只重建对应尺寸的空AV latent。
4. `SPEED Whole-Chain Sampler Advanced`按阶段运行原生H3`ModelSamplingAV + Euler`。只有视频做空间DCT
   扩张；音频不补空间高频，但由于音画共享Transformer，音频状态仍需从旧公共flow sigma同步重参数化
   到对齐sigma，不能伪装成冻结不动。

默认`strict_t2va_stock20`只允许T2VA+native audio+严格20步；要运行I2VA、FL2VA、L2VA、Ref2VA或Hybrid，必须
主动选择`multimodal_research_exp`。这些模态的阶段重建代码已完成，但真实H3生成、身份/锚点、mask、
参考声音和感知质量尚待逐项实机验证。节点遇到现有DiT block replacement或已知采样wrapper会直接拒绝，
首轮不得叠Block Cache、STG、Activation Chunk、Restart、MultiRate或Dynamic Guidance。

当前已完成官方公式测试、20 NFE守恒、画布/latent尺寸、DCT对SciPy数值对齐、确定性随机种子、联合AV
分段状态往返、频谱profile门槛、节点注册和工作流静态合同；全项目606项测试、Ruff、compileall和
63份前端工作流JSON解析均通过。尚未运行真实ComfyUI GPU生成，因此此版本
不声明速度提升、质量非劣、音频非劣、16GB安全或任意参考模态已经通过；这些字段在报告中全部为false。

## 示例与测试

可直接拖入画布的稳定 4/4、三种输入音频模式、EXP 4/8、EXP 4/10、Ref2VA 22帧静态候选编辑、
对白安全分轨母带、两遍 H3 分时背景底轨锁定、Hybrid Model Advanced Stock20，以及以下长视频示例位于 `examples/workflows/`：

- `2026-08-09_H3_Long_Video_22F_EXP.json`：手工逐段续写基线。
- `2026-08-09_H3_Long_Video_Accepted_22F_EXP.json`：候选预览、接受和可恢复状态链。
- `2026-08-09_H3_Long_Video_Auto_Resume_22F_EXP.json`：总时长编排与人工审核后自动恢复。
- `2026-08-09_H3_Long_Video_Background_22F_EXP.json`：后台自动排队长链。
- `2026-08-09_H3_Long_Video_Background_22F_ScenePlusIdentity_EXP.json`：完整场景与身份裁剪双参考的后台长链。
- `2026-08-09_H3_Motion_Quality_Advanced_8Step_EXP.json`：Turbo双时钟8步测试基线、默认关闭的sigma尾段实验与只读质量审计。
- `2026-08-09_H3_Motion_Quality_Dynamic_Guidance_8Step_EXP.json`：默认Basic透传的动态单路引导实验与实际分支/NFE审计。
- `2026-08-09_H3_Motion_Quality_Extra_Tail_NFE_8Step_EXP.json`：默认关闭的尾段额外NFE实验，NOTE说明8→10 NFE成本与普通10步因果对照。
- `2026-08-18_H3_Hanfu_Tail_Detail_3Step_Advanced_EXP.json`：红色汉服高速旋转固定例，8+3联合AV尾段细化。
- `2026-08-18_H3_Hanfu_Model_Time_Bias_Advanced_EXP.json`：同输入8 NFE、平滑共享AV模型时间偏置。
- `2026-08-18_H3_Hanfu_RF_Restart_Advanced_EXP.json`：同输入基础8步后联合AV Rectified-Flow Restart 3步。
- `2026-08-18_H3_Hanfu_STG_Advanced_EXP.json`：同输入H3 block 25时空引导。
- `2026-08-18_H3_Hanfu_Temporal_Detail_Advanced_EXP.json`：同输入基础8步后解码帧时序保护细节增强。
- `2026-08-09_H3_Hybrid_Model_Advanced_Stock20_EXP.json`：精确pair检查、默认小artifact、stock-loader Hybrid MODEL与Ref2VA参考图链。
- `2026-08-09_H3_Hybrid_Model_Audio_Reference_Stock20_EXP.json`：独立音频参考，Inspector按Conditioning自动选择
  最小audio-row实验profile。
- `2026-08-09_H3_Hybrid_Model_Mixed_Reference_Stock20_EXP.json`：参考图+参考音频，自动选择video+audio-row
  实验profile；仍需用户盲评，不能视为最佳profile。
- `2026-08-18_H3_SPEED_T2VA_Stock20_Advanced_EXP.json`：默认严格T2VA、20步、0.5→1.0、手工sigma的首个实机候选。
- `2026-08-09_H3_SPEED_FL2VA_Stock20_Advanced_EXP.json`：首尾帧逐阶段重编码的多模态研究示例，必须显式EXP。
- `2026-08-09_H3_SPEED_Ref2VA_Stock20_Advanced_EXP.json`：参考图逐阶段条件重建示例，必须显式EXP。

API 示例见 `tests/fixtures/api/audio_lock_api.json`、
`tests/fixtures/api/dual_clock_4step_api.json`、`tests/fixtures/api/multirate_exp_api.json` 和
`tests/fixtures/api/still_image_edit_api.json`、`tests/fixtures/api/hybrid_model_advanced_api.json`、
`tests/fixtures/api/hybrid_model_audio_reference_api.json`与`tests/fixtures/api/hybrid_model_mixed_reference_api.json`；
高速动态实验另见`tests/fixtures/api/motion_quality_advanced_8step_api.json`；
对白安全音频另见
`tests/fixtures/api/dialogue_safe_master_api.json` 与 `tests/fixtures/api/dialogue_timed_bed_lock_api.json`。
替换 API 示例里的模型、VAE、CLIP、可选 LoRA、
输入图像和音频文件名后即可使用；
保存节点使用已安装的 VideoHelperSuite。

从 ComfyUI 根目录、使用启动 ComfyUI 的同一 Python 环境运行：

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -q .\custom_nodes\minimax-h3-audio-T8
```

自动化测试用于验证节点注册、条件与 latent 契约、sigma 数学、mask/callback、工作流结构
和静态图像路径；它不等同于对所有模型、提示词、种子和画布的感知质量保证。

## 显存与 DynamicVRAM 验证

项目提供独立诊断工具 `tools/validate_h3_vram.py`，用于排查 H3 工作流在
DynamicVRAM/VBAR、`LoraLoaderBypassModelOnly` 和双时钟采样组合下的 OOM。工具不修改
采样数学或模型权重，可完成 API 工作流静态检查、生成 stock Euler/双时钟严格 A/B、生成
Hybrid Loader未启用/启用VRAM Policy的严格单变量A/B、按节点和采样进度记录显存曲线，以及
比较两次运行的控制变量与峰值增量。

从v1.19.0开始，新的 Turbo 双时钟对照必须统一为8步、相同模型/LoRA/Prompt/seed/尺寸/帧数，
并建议关闭预览。既有4步记录仅保留为历史与兼容证据。完整命令、判定规则和限制见
[显存验证方法](docs/VRAM_VALIDATION.md)。在取得真实 OOM
traceback 和有效 A/B 前，不应把高显存直接归因于双时钟节点，也不应盲目替换 INT8 旁路
LoRA 或关闭 VBAR。

2026-08-07 的本机暖缓存实测中，`0.6M`、362 帧、4 步的 stock Euler 与双时钟设备峰值
分别为 16,213.5 MiB 和 16,182.2 MiB，PyTorch 峰值均为 14,573.5 MiB；未发现双时钟路径
存在实质峰值增加。两条路径都已非常接近 16 GiB 上限，这个单机结果不能替代反馈用户的
精确工作流、OOM traceback 和冷启动换序复测。
