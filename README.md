# MiniMax H3 Audio T8

面向 ComfyUI 的 MiniMax H3 视频与音频节点包。它保留原生 H3 的工作流接口，并在此基础上提供双时钟采样、音频控制、长视频、关键帧、脸部修复和显存诊断等能力。

当前版本：**1.47.2** · 节点 211 个 · GPL-3.0-or-later

## 能做什么

- 文生音视频（T2VA）、图生视频、首尾帧、参考图/参考视频/参考音频和 Hybrid 工作流
- 双时钟视频/音频采样，以及音频锁定、重混、参考和最终混音
- 学习型双采样在非1.0起点把音频重建到自己的sigma时钟，再由第二阶段完成联合AV；显式锁源审计仍可选
- `13-latent-upscale` 中标准、native语音、lock_source、remix_source 0.20 和 reference_only 五份学习型二采工作流已统一到作者对齐的低清4步+高清4步，共8次联合AV前向
- 同一清晰语音素材上的 native、lock_source、remix_source 0.20 与 reference_only 双采成片已由用户完整审核通过；结论限定于该素材
- 32 倍数对齐的 3D latent 放大，避免官方放大节点造成宽高偏移
- 长视频分段、断点续跑、时间线报告和音视频同步检查
- 单人/多人脸部修复、SAM3.1 追踪、多帧关键帧和实验性动态细节增强
- Motion Recovery动作过载分析、自动ABSTAIN懒旁路、局部时间超采样二采、原时钟恢复和分窗断点续跑（实验）
- 语音、对白、演绎、ADR 和生产辅助节点（实验功能会明确标注）
- 音频完整性、参考相对音色漂移与多人对白路由预检：报告首尾边界、DC、削波、持续频谱/响度漂移、参考音频绑定和歧义，遇到风险返回 `ABSTAIN`，不自动修改素材
- 提示词预算与角色媒体编译：统计字符/token、检查 `<Picture>/<Video>/<Audio>` 数量、顺序与人物绑定覆盖，原提示词连首尾空白也不静默删除
- 提示词服务路由：默认本地原文直通，可显式连接 OpenAI/LM Studio/llama.cpp 或 Ollama；密钥仅从环境变量读取，Ollama默认请求后卸载（实验）
- 提示词语义合同审计：用用户声明的必需/禁用动作词组、精确对白和媒体标签检查重写候选；默认继续输出原文，机械通过并经人工显式接受后才切换候选（实验）
- Creator Workspace：在既有 Studio Timeline 上追加镜头覆盖、seed变体、运行区间、hold-map sidecar、显式运行回执与断点续跑计划；可选桥接现有Long Video后台取消/重试，同时提供不拉伸画面、A/B音轨分别试听的同步音画审片。候选保留计划默认只生成清单；另有双阶段、SHA锁定、可恢复的Quarantine节点，显式确认后只把文件移入输出目录内的隔离区，不提供永久删除（实验）
- ClipProj 与 Sol-Attn 外部兼容审计：只检查插件版本、矩阵维度、硬件和补丁所有权，不复制、加载或执行外部实现（实验）
- 原生 H3 latent 时间线拼接：按视频24fps/音频40Hz双时钟移除后续片段重复的5帧前缀，并在CPU组合完整AV latent（实验）
- 原生 H3 Long Video续接拼接：同时核对Planner与Conditioning报告，按5/22/39帧真实上下文移除双时钟重叠；旧5帧通用拼接保持不变（实验）
- 原生 H3 latent 恢复清单：分块计算联合AV、可选mask和元数据的精确SHA-256；重载后默认不一致即报错，不写文件、不冒充扩散内部断点恢复（实验）
- 原生 H3 latent 检查点 Save/Load：把已经形成的完整联合AV latent、可选mask和受支持元数据保存为无pickle的safetensors，并在另一进程中按文件SHA与内容清单严格恢复；默认不保存，不等于采样中断后续跑NFE（实验）
- 双时钟 Euler NFE 检查点/续跑：仅支持本项目 `dual_clock_euler + native_flow`；新增运行合约编译器会绑定最终提示词、媒体映射、报告和实际Conditioning张量，显式启用后在每个完整NFE边界原子保存联合AV状态，并可在新的ComfyUI进程中从剩余sigma继续。默认关闭，不支持DPM++、SDE/ancestral、第三方sampler或中断在Transformer前向内部的恢复（实验）
- 预检、显存/缓存报告、模型兼容检查和异常路径清理
- LightX2V Turbo-SLA LoRA、85% 动态块稀疏 Sage2、逐次运行审计，以及可选的KJ Sage单入口组合器（实验）
- RAVEN Streaming外部运行时桥接：统一发布参数、在巨大权重载入前检查硬件/内存，并严格审计仅T2VA的因果分块请求（实验）
- Skin Finish 非生成式肤质收尾：P0在可靠遮罩内生成低频肤色/油光候选；固定ParseNet节点只选择皮肤并保护眼眉鼻唇头发，多人语义节点复用SAM3.1逐镜轨迹并与各自人物MASK相交。逐人物示例使用独立侧脸Advanced EXP节点：先做原有YuNet五点FFHQ对齐，只有严格残差门拒绝时才在原侧脸方形裁切中解析，不降低阈值、不把侧脸正面化。Per-Person Advanced可按人工Character或精确`shot:track`配置参数，重叠和未匹配区域保持原片，并逐路报告亮度代理、处理幅度和裁切供人工复核；不会自动判定肤色公平。Studio Timeline路线按创作镜头内局部帧为global、Character或SAM轨迹设置hold/linear/smoothstep关键帧，绝不跨切镜混合。Preview提供512边长代理的原片/候选拖动对比，但全分辨率输出仍是判断依据。Frequency Split把候选低频变化与来源已有高频纹理解耦，再交给Texture Guard和Safety Audit；独立Specular-Aware Split Advanced EXP现只把普通分离丢失的高光处理按比例补回，并严格限制在“普通分离候选—原始Skin Finish候选”之间，默认`3% / 0.65`且0强度逐像素等于旧节点。六帧同参复核全部通过机械门，但最终只保留原候选约29.7%的平均变化，肉眼仍弱，当前不作为默认工作流。完整IMAGE P0会在候选、双MASK和float16差异图分配前，按实际帧数、尺寸、通道、dtype、chunk和代理尺寸计算增量系统RAM/Windows commit下限；可测资源不足时原片ABSTAIN且不进入遮罩或候选计算。Quality Stream直接读取最终文件VIDEO，以两遍YuNet元数据和默认2帧CPU chunk依次执行固定ParseNet、Frequency Split、Texture Guard与跨chunk Safety Audit，不物化完整IMAGE或完整语义MASK。通用示例保持轻量`subtle`，新增Oil Control Stream给明确油光素材使用经过单次124帧机械验证的`oil_control / 0.35 / 0.90 / 0.35`起点；没有明显油光时不要为了制造差异强行使用。唯一32秒/768帧真实CPU运行峰值working set约1.97GiB，严格解码且音频包/PCM一致；据其约1,163MiB进程增量，显式执行前会在可测平台要求至少2,048MiB可用系统内存，不足时原片ABSTAIN且不加载ParseNet、不写文件。两种预检都只覆盖本节点增量或已审阅文件流，不是整张ComfyUI图、任意机器或普遍16GB安全证明。专用油感素材的一次匿名评审为`ABSTAIN_UNSURE`：8项平局、2项弃权、双方无硬失败，备注“似乎感觉差不多”；固定CineStyle默认链的六帧参考审计虽约有2倍平均改变量，但纹理代理更低且最亮肤区整体变亮，不是可直接复制的去油答案。因此当前只保留机械通过，不宣称肉眼改善。它不会锐化模糊来源、生成毛孔或自动判断美感，所有候选仍需显式人工接受（实验）
- Guided Surface Finish Advanced EXP是严格追加、未接入工作流的clean-room候选：逐帧guided filter保留边缘，再以有界亮度肩部处理旧算法漏掉的宽面积油光；它不冒充物理高光分离。遮罩外与辅助通道逐值不变，AUDIO原对象旁路且默认不接受候选。固定油感来源六帧中，相对当前Quality Stream的平均变化由`0.00013440`提高到`0.00840873`，纹理代理仍为`0.99715394`，最亮肤区变化`-0.01996538`且全部硬门通过。唯一124帧流式机械复核为62个两帧块、124/124语义脸、2帧Surface和2帧Texture Guard安全回原片、Safety Audit零失败、最大内部时间跳变`0.00381594`、音频packet/PCM精确、峰值working set约1.95GiB；公开A/B不读取私钥的独立解码审计最大ROI跳变`0.00305909`，未发现粗粒度闪烁峰值。匿名review `b3aad4e0d57b`已完成：原片在整体、肤质自然度、油光、高低频肤色均匀和halo五项胜出，其余五项平局，候选0项胜出，双方均无硬失败。因此该节点仅保留为未通过主观门的实验实现，不接入工作流、不替换Oil Control，也不宣称可见改善（实验）
- 针对上述否决，Surface v2不再把绝对亮度直接视为宽油光：它只压缩相对同一语义皮肤区域大尺度照明更亮的局部成分，并在硬MASK边界内侧两像素渐退；盒滤波改成数学等价的水平/垂直两遍。唯一v5六帧候选6/6通过，最终平均变化`0.00387333`、最亮肤区`-0.00787700`、纹理代理`0.99280846`、边界/内部变化比`0.67757654`，Surface六帧耗时从约57秒降至4.35秒。唯一124帧流式复核无任何Surface/Texture Guard回退、Safety Audit零失败、最大内部跳变`0.00173517`、音频包/PCM精确；映射盲审计最大ROI跳变`0.00052624`。匿名review `8e89bff3bc95`已完成：整体、自然度、油光、肤色、纹理、五官、时间闪烁、跨人物串色、halo和身份/口型10项全部平局，双方均无硬失败。v2消除了v1的明显主观回退，但没有建立可感知收益，因此仍不接入工作流、不改默认值；后续若继续应换用实质不同的方法，而不是继续调强同一亮度肩部（实验）
- Dichromatic Specular Advanced EXP是实质不同但未获推广证据的实验路线：在线性sRGB中以中性照明双色反射近似分离漫反射与镜面分量，并同时要求正镜面估计、局部色度稀释和方向一致。固定六帧与唯一960×544×124文件流均通过机械门，Safety Audit零失败、音频精确，公开A/B时序审计也未发现粗粒度异常。anonymous review `b2e13261f44e`的正式盲测表单在揭盲后显示原片7胜、候选0胜、3平且双方无硬失败；映射公开后用户再次复看，修正主观描述为“基本一样”。后者属于揭盲后意见，不能改写盲测JSON；两者共同只能证明没有建立可感知改善。因此该路线不接工作流、不改默认值，也不冒充物理皮肤BRDF、去模糊、毛孔生成或身份修复（实验归档）

稳定节点与 `Advanced`/`Experimental` 节点分开。高级节点采用追加式设计：未连接时不改变旧工作流、旧节点 ID、输入顺序和默认采样数学。

## 安装

### Comfy Registry（推荐）

在 ComfyUI Manager 中搜索 `MiniMax H3 Audio T8`，或在终端执行：

```powershell
comfy node install minimax-h3-audio-t8
```

### 手动安装

将仓库放到：

```text
ComfyUI/custom_nodes/minimax-h3-audio-T8
```

本插件不自动下载 H3 权重，也没有强制的额外 pip 依赖。请先准备与任务匹配的模型、CLIP、VAE 和 LoRA，并重启 ComfyUI。Long Video 的候选保存和最终拼接要求 `ffmpeg` 可从 `PATH` 调用；常规 ComfyUI 整合包通常已经包含，缺失时节点会在写入前给出明确错误。只有使用 8B 提示词重写器时才安装 `requirements-prompt-rewriter.txt`；依赖被限制在 Transformers 4.x、PEFT 0.18–0.20 和 Pillow 10.x，避免无关升级影响其他节点。LightX2V SLA 实验节点还需要与当前 Torch/CUDA 匹配的 `spas-sage-attn`；只有使用SLA + KJ Sage组合工作流时才需要另外安装 `ComfyUI-KJNodes`。可选的Skin Finish语义遮罩节点还需要FaceXLib代码和固定的`models/facedetection/parsing_parsenet.pth`；多人路线另复用本项目现有的本地YuNet与原生SAM3.1轨迹。ParseNet节点会核对85,331,193字节及完整SHA-256，不会联网下载或接受任意模型路径。

## 第一次使用

1. 打开 `examples/workflows/`，先从 `01-basic-generation` 中的基础工作流开始。
2. 将工作流中的模型、图片、视频和音频替换为自己的文件。
3. 初次运行建议使用 22 帧、较小画布和 stock sampler，确认能正常解码后再增加时长、分辨率或参考条件。
4. 使用工作流中的 NOTE 节点作为参数说明；报告输出可用于排查标签、显存、音频和兼容性问题。

### Quick Start 子图

`subgraphs/` 提供 7 个 ComfyUI 原生 Quick Start 子图：T2VA、I2VA/FL2VA、Ref2VA、Audio Drive、Long Video、Repair 和 Creator AV Review。它们只封装现有节点并减少外露参数，不改变旧节点的输入顺序或默认值。Creator审片入口保持A/B音轨独立、并排视频无声且默认`ABSTAIN`，不会自动选择胜者。文件名统一使用日期开头的 ASCII 名称，避免 Registry 打包在部分 Windows/GBK 环境漏文件；导入后的子图标题和 NOTE 仍保留中英说明。详细说明见 [`subgraphs/README.md`](subgraphs/README.md)。

### 可选高级扩展

| 功能 | 入口 | 额外要求 | 当前结论 |
| --- | --- | --- | --- |
| 8B 提示词重写 | `14-prompt-relay/2026-08-22_H3_Prompt_Rewriter_8B_Advanced_EXP.json` | `requirements-prompt-rewriter.txt`、Qwen3-VL-8B 基座和 LightX2V LoRA | 16GB 本机可运行并能生成结构化结果，但 CPU 分片很慢；默认生成后卸载 |
| LanPaint 局部 AV 修复 | `03-image-video-edit/2026-08-22_H3_LanPaint_AV_Local_Repair_Advanced_EXP.json` | 单独安装 `scraed/LanPaint` | 画面蒙版与音频秒区间分离，未声明区域回贴原始内容；尚未做高负载质量验证 |
| 外部 BlockSwap | `12-system-memory/2026-08-22_H3_External_BlockSwap_Stock20_Advanced_EXP.json` | 单独安装 `xiaolibai-sys/ComfyUI-MiniMaxH3` | 只服务外部 `MINIMAX_H3_*` 类型，不接受官方 `MODEL`；16GB 未做压力认证 |
| LightX2V Turbo-SLA | `15-sla-attention/2026-08-22_H3_LightX2V_SLA_FL2VA_4Step_Advanced_EXP.json` | H3 SLA LoRA、FL2VA、匹配的 `spas-sage-attn`；官方参考仍为4 NFE/6V/3A | 默认4步机械验证通过；节点现按结构和完整模型映射接受其他H3 SLA LoRA，并按实际NFE审计。8步可运行但属于实验兼容，不宣称上游质量背书 |
| SLA + KJ Sage组合器 | `15-sla-attention/2026-08-22_H3_LightX2V_SLA_KJ_Sage_Composer_FL2VA_4Step_Advanced_EXP.json` | 上述SLA条件，加单独安装的 `ComfyUI-KJNodes` | 解决KJ完整forward绕过SLA的问题；每次Attention只选一个后端，当前完成结构与回归验证 |
| 音频完整性审计 | `05-speech-dialogue/2026-08-22_H3_Audio_Integrity_Audit_Advanced.json` | 无；纯CPU信号分析 | 输出PASS/ABSTAIN和证据，不修音；循环音乐可能触发尾首相似提示 |
| 音色远近漂移审计 | `05-speech-dialogue/2026-08-22_H3_Audio_Perceptual_Drift_Audit_Advanced.json` | 同内容、同时间线的基准和候选音频 | 真实试听异常的纯二采在1.4～3.6秒触发ABSTAIN，正常一采与本例80%混音PASS；它是声学复核提示，不诊断远场/混响/换声 |
| 多人对白路由预检 | `05-speech-dialogue/2026-08-22_H3_Speaker_Routing_Audit_Advanced.json` | 每个角色独立参考音频 | 编译`<Audio N>`映射并检查重复波形、未结构化笑声/喘息和描述歧义 |
| 提示词预算与角色编译 | `14-prompt-relay/2026-08-22_H3_Prompt_Budget_Role_Compiler_Advanced.json` | CLIP为可选输入 | 默认7000匹配当前官方H3 CLI提交上限；7000/7001与三人物映射已测。真实Qwen3-VL 8B/Boogu编译文本为140个token、规划估算153。当前ComfyUI无7000字符硬拦截，故不把API/CLI规则冒充本地tokenizer硬上限；视觉/时间戳token在Conditioning阶段另行加入 |
| 提示词服务路由 | `14-prompt-relay/2026-08-23_H3_Prompt_Provider_Router_Advanced_EXP.json` | 本地服务无需密钥；远程服务的密钥通过环境变量提供 | 默认不联网；原始`<d>`对白先替换为不可猜测令牌，provider只有逐字返回唯一令牌且位置正确才会恢复，真实对白不上传；可选0～2次合同修复且不重传参考图，默认0保持旧请求数；CPU-only 8B已有一条严格合同通过，但把“旋转”改成“站立”，仍未通过语义质量门 |
| 提示词语义合同审计 | `14-prompt-relay/2026-08-23_H3_Prompt_Semantic_Contract_Audit_Advanced_EXP.json` | 无；纯本地字符串审计 | 已知“旋转→站立”真实Provider回归会被拒绝并保留原文；空锚点ABSTAIN，非法合同、对白变化和源媒体标签丢失fail closed。词组PASS不等于通用语义等价，仍需人工复核 |
| Creator Workspace | `11-studio-production/2026-08-22_H3_Creator_Workspace_Run_Window_Advanced.json` | 复用现有Studio Timeline | 真实3镜头计划、3候选seed、hold-map和显式选择已执行；不隐式排队、不写文件、不删除候选 |
| 同步A/B预览 | `11-studio-production/2026-08-22_H3_Creator_Synchronized_AB_Advanced.json` | 输入两个IMAGE帧批次 | 真实39帧A/B已保持源像素并严格解码；几何不一致可强制ABSTAIN；仍不比较音频 |
| 同步音画A/B审片 | `11-studio-production/2026-08-22_H3_Creator_Synchronized_AV_AB_Advanced.json` | 两条同内容、同起点且均带音轨的视频 | 保存无声并排画面并分别试听A/B音轨。1088×544近景真人合同以同一124帧latent得到243帧/10.125秒两路成片，精确双时钟、严格解码及1,125MiB最低余量通过；一名审阅者在所有画面/接缝/对白维度判平且无硬失败。该固定简单素材结果不等于普遍等价，仍不自动接受 |
| Creator运行回执、续跑与可恢复隔离 | `11-studio-production/2026-08-23_H3_Creator_Run_Receipt_Resume_Advanced.json` | 复用Creator Workspace；运行结果由用户或外部执行器显式填写 | 记录completed/accepted/rejected/cancelled/failed并给出render/review/retry/complete；Retention Plan仍只生成keep/proposed-delete清单。Quarantine节点在示例中默认静音，先`prepare_only`核对路径/字节/SHA，再以精确plan hash、新epoch和显式确认移动到`output/MiniMaxH3/creator_quarantine`；同一receipt/epoch可restore或recover，不提供永久删除 |
| Creator × Long Video后台桥 | `11-studio-production/2026-08-23_H3_Creator_Long_Video_Background_Bridge_Advanced_EXP.json` | 必须嵌入现有Long Video Background候选保存/自动接受成片链 | workspace hash绑定任务；accepted_count选镜头、retry_count选seed；默认review_only；真实256×256×22 H3采样在1/4步定向取消后显存回到基线+90MiB；另以合法124+119帧轻量控制链验证终态重新挂接、自动续排、Candidate Save/Auto Accept和最终AV合成，真实H3恢复成片质量仍待验收 |
| ClipProj兼容审计 | `12-system-memory/2026-08-22_H3_ClipProj_Compatibility_Audit_Advanced_EXP.json` | 单独安装`ComfyUI-ClipProj` 0.1.13+ | 检查4B/8B投影维度、Qwen3-VL声明与加载模式；不加载模型，也不替换32B默认路径 |
| Sol-Attn兼容审计 | `12-system-memory/2026-08-22_H3_Sol_Attn_Compatibility_Audit_Advanced_EXP.json` | 单独安装`ComfyUI-sol-attn` 0.6.2+ | 检查CUDA/BF16、架构与完整H3补丁所有权；不运行kernel，不作速度/显存结论 |
| ClipProj 4B低负载T2VA桥 | `12-system-memory/2026-08-23_H3_ClipProj_4B_T2VA_Bridge_Advanced_EXP.json` | 官方ComfyUI格式Qwen3-VL 4B、4B v3.1矩阵和外部ClipProj | 短T2VA和1088×544近景真人I2VA均已真实生成。固定简单素材中4B与8B、4B与原生32B的单人盲评均为全维度平局且无硬失败，但这不建立普遍等价或非劣；32B继续默认。原生32B本次最低余量643MiB，仅证明这一条通过512MiB门，不代表普遍16GB安全；纯文本`qwen_3_4b`仍不可替代 |
| ClipProj 8B完整T2VA桥 | `12-system-memory/2026-08-22_H3_ClipProj_8B_T2VA_Bridge_Advanced_EXP.json` | 8B Qwen3-VL、v3.1投影矩阵和外部ClipProj | 本机固定seed短T2VA已真实编码、生成并严格解码；只证明链路，其他模态和质量A/B仍待补 |
| ClipProj 8B完整I2VA桥 | `12-system-memory/2026-08-22_H3_ClipProj_8B_I2VA_Bridge_Advanced_EXP.json` | 同上，并连接一张首帧图 | 本机256×256×22、4步I2VA已走通视觉塔与联合AV生成并严格解码3/3；不等于32B画质或对白遵循度已通过 |
| ClipProj 8B完整FL2VA桥 | `12-system-memory/2026-08-22_H3_ClipProj_8B_FL2VA_Bridge_Advanced_EXP.json` | 同上，并连接独立首帧与尾帧 | 本机256×256×22、4步FL2VA已走通双视觉关键帧与联合AV生成并严格解码3/3；只证明短链执行，不证明长插值或32B画质等价 |
| ClipProj 8B完整Ref2VA桥 | `12-system-memory/2026-08-22_H3_ClipProj_8B_Ref2VA_Bridge_Advanced_EXP.json` | 8B ClipProj、Ref2VA pruned INT8与一张参考图 | 本机Stock20短链及原生32B同seed机械对照均通过严格解码；身份代理未稳定过线且8B只余约62MiB，故不宣称质量等价、省显存或16GB安全 |
| Sol-Attn保守T2VA桥 | `12-system-memory/2026-08-22_H3_Sol_Attn_T2VA_Conservative_Advanced_EXP.json` | 外部Sol-Attn 0.6.2+ | 4步默认dense百分比为0，首三/末块dense、exact conditioning KV、首次strict；本机0.737MP实跑记录5139 tokens并进入kernel，但单次顺序A/B未显示速度或显存优势；一名审阅者给出全平且没有失败备注，仍不足以提升为推荐路线 |
| RAVEN Streaming受保护T2VA | `16-raven-streaming/2026-08-23_H3_RAVEN_Streaming_T2VA_Guarded_Advanced_EXP.json` | 单独安装外部RAVEN插件0.1.0、完整BF16 H3和 mandatory RAVEN LoRA | 真正采样/预览仍由外部节点执行；T8统一4步、12/3、2/2、cpu_pinned参数并检查T2VA合同。当前16GB/128GB机器低于已审阅资源范围，默认会在加载前拒绝，不宣称可运行 |
| Skin Finish肤质收尾 | `17-skin-finish/2026-08-24_H3_Skin_Finish_External_Mask_Advanced_EXP.json` | 可靠外部MASK，或同一来源帧的Face Refine Plan | Basic/Advanced/Preview-Audit均默认保留source；仅做SDR非生成式低频肤色和油光候选，不修复五官、模糊、身份或口型。缺mask、面积异常、来源不一致和音频不一致均fail closed |
| 多人Skin Finish与原音频封装 | `17-skin-finish/2026-08-24_H3_Skin_Finish_MultiPerson_Video_Finalize_Advanced_EXP.json` | 原生SAM3.1 track plan、CPU YuNet、未裁切8-bit SDR文件VIDEO | 复用一次追踪结果，不重复加载SAM；镜头内bbox EMA和绝对帧state支持重叠续块。最终候选默认不保存，人工接受后仅重编码视频，兼容音频包payload逐包复制并SHA复核；HDR/10-bit、旋转、裁切和未知codec拒绝 |
| 两遍文件流式Skin Finish | `17-skin-finish/2026-08-24_H3_Skin_Finish_Two_Pass_Video_Stream_Advanced_EXP.json` | Long Video/Studio最终未裁切8-bit SDR文件VIDEO、固定YuNet | 不连接完整IMAGE：第一遍只保存脸框/切镜/来源摘要，第二遍按默认4帧chunk处理并以单线程H.264立即编码，发布前执行FFmpeg严格解码。1088×544×124单次机械验证通过；默认false时不分析、不写文件，共享中性色、无身份识别/语义parser，不等于去模糊或修脸 |
| ParseNet低内存Quality Stream | `17-skin-finish/2026-08-25_H3_Skin_Finish_Quality_Stream_Advanced_EXP.json` | Long Video/Studio最终未裁切8-bit SDR文件VIDEO、本地固定YuNet与ParseNet | 第一遍只留脸框/切镜元数据，第二遍默认2帧chunk依次运行语义皮肤MASK、Skin Finish、Frequency Split、Texture Guard和跨chunk Safety Audit并立即编码；只额外保留上一帧。唯一736×416×768、32秒CPU长片运行384个chunk，690帧语义就绪、78帧安全回退source、Safety Audit零失败、峰值约1.97GiB，严格解码且音频包/PCM一致。显式执行前在可测平台要求2,048MiB可用系统RAM，不足时原片ABSTAIN且零模型/零写入；默认false连预检都跳过。共享媒体门按真实像素分量与FFmpeg枚举拒绝10/12/16-bit、PQ、HLG、BT.2020/P3/ICTCP等，并复制已接受SDR的颜色元数据。主观效果、HDR处理、任意时长和通用内存安全仍未证明 |
| Skin Finish专用Oil Control Stream | `17-skin-finish/2026-08-25_H3_Skin_Finish_Oil_Control_Stream_Advanced_EXP.json` | 有明确额头、鼻梁或双颊油光的最终8-bit SDR文件VIDEO | 复用同一低内存Quality Stream，固定`oil_control / amount 0.35 / texture_keep 0.90 / shine 0.35 / chunk 2 / CRF 16`。v1.0八步LoRA的960×544×124中文说话近景已完成一次124/124帧、零安全拒绝和音频包/PCM精确的机械验证；这不是主观去油通过或自动磨皮，必须检查蜡像感、眼唇、闪烁和halo，来源油感不足时保留原片 |
| Skin Finish纹理/曝光机械护栏 | `17-skin-finish/2026-08-24_H3_Skin_Finish_Texture_Guard_Advanced_EXP.json` | 已生成并审核来源绑定的source/candidate/used mask | 默认保护源片深阴影和近饱和高光；新增裁切或源片相对高通RMS下限失败时整帧回退。0.592MP×124单次默认门机械通过且音频PCM一致；它不是自然肤质评分、语义parser、去模糊或毛孔生成器 |
| Skin Finish低频/纹理解耦 | `17-skin-finish/2026-08-25_H3_Skin_Finish_Frequency_Split_Advanced_EXP.json` | 已生成且来源绑定的source/candidate/used mask；后接Texture Guard | 按短边比例建立两遍低通层，以候选低频叠加来源已有高频；默认`1.0/1.0/1%`、4帧CPU分块且不接受候选。相同输入和关闭低频为精确no-op，mask外/alpha/音频保持不变；来源模糊或只有噪声时不会自动变清晰，仍需硬门和人工审片 |
| Skin Finish Studio Timeline关键帧 | `17-skin-finish/2026-08-25_H3_Skin_Finish_Studio_Timeline_Advanced_EXP.json` | 最终完整IMAGE、总帧数完全匹配的Studio Timeline、同来源SAM3.1/ParseNet合同 | 按Studio镜头内局部帧插值amount/texture/shine/tone；preset只在目标键切换。Studio镜头负责时间，SAM shot:track负责人物，两套编号独立；优先级为精确轨迹、Character、global、source。跨切镜不插值，重叠/未匹配/合同错误回原片，音频旁路且默认不接受候选 |
| Skin Finish固定ParseNet语义遮罩 | `17-skin-finish/2026-08-24_H3_Skin_Finish_Semantic_Mask_Advanced_EXP.json` | FaceXLib代码、固定v0.2.2 ParseNet权重和同一来源帧的Face Refine Plan | 仅CPU逐帧解析并在结束后卸载；默认选择skin，排除鼻、眼、眉、嘴唇、头发和配饰。真实3帧固定权重机械检查通过；现有plan无五点关键点且只代表单轨人脸，尚未通过多人/跨镜和人工肤质验收 |
| 多人五点ParseNet语义遮罩 | `17-skin-finish/2026-08-24_H3_Skin_Finish_MultiPerson_Semantic_Mask_Advanced_EXP.json` | 同一来源IMAGE、hash-valid SAM3.1逐镜track plan、本地YuNet、固定ParseNet；identity assignment可选 | 每个可靠脸只匹配一个人物轨迹，五点对齐到FFHQ 512后解析并反投影，与各自人物MASK相交；覆盖不足整批返回空MASK。0.67584MP六帧双人真实YuNet/ParseNet机械检查12/12 READY；该低负载检查使用来源绑定的人物区域夹具，不冒充一次新的SAM实跑或人工肤质通过 |
| 逐人物/侧脸Skin Finish参数与安全审计 | `17-skin-finish/2026-08-25_H3_Skin_Finish_Per_Person_Advanced_EXP.json` | 同一SAM轨迹、固定ParseNet；Character配置还需人工复核的identity assignment | 严格五点优先，仅被拒绝的姿态使用1.45×原侧脸裁切回退；960×704×69双人真实完整链达到138/138人物帧覆盖，严格基线为96/138，共42次必要回退。最新实链在未压缩候选上通过`unique_track_owner + hard_gate`：69帧零失败、零串色/重叠歧义、PCM精确一致，严格媒体解码通过。CPU夹具另确认交叉人物仍随Character路由、重叠回原片，以及切镜后人工mapping按Character续接；逐路亮度/改变量/裁切只作审片辅助。它仍只代表可进入人工审片，不能证明自然、更美、真实肤色公平或身份真值；仍需说话口型/闪烁、真实交叉遮挡、不同肤色和跨镜人眼验收 |
| 原生latent时间线拼接 | `04-long-video/2026-08-22_H3_Native_Latent_Timeline_Concat_Advanced_EXP.json` | 输入完整H3嵌套AV latent，batch=1且画布一致 | 真实22+22→39帧单次解码、52000采样无损音频和重复hash已通过；旧32B路线两次未过512MiB余量门，另一个固定8B ClipProj短链以605MiB最低余量通过一次0.25秒采样门。独立段仍会换景，不能外推为无缝或通用16GB安全 |
| 原生latent Long Video续接拼接 | `04-long-video/2026-08-23_H3_Native_Latent_Continuation_Concat_Advanced_EXP.json` | 上一段/累计时间线latent、本段采样完成latent，以及同一Planner与Conditioning的直连报告 | 精确移除5/22/39帧上下文；124+124在22帧context下得到226帧，默认要求音画上下文。最终隐藏尾帧须一次解码后再裁；不等于NFE断点恢复、无缝质量或省显存证明 |
| 原生latent恢复清单 | `04-long-video/2026-08-23_H3_Native_Latent_Resume_Manifest_Advanced_EXP.json` | 输入已保存/重载的完整H3嵌套AV latent；恢复核对时同时粘贴旧`manifest_json` | SHA-256分块大小不影响摘要；默认`error`阻断内容、shape、dtype、mask或checkpoint ID不一致。节点不保存latent、不恢复NFE内部状态，也不作续段质量/显存保证 |
| 原生latent检查点Save/Load | `04-long-video/2026-08-23_H3_Native_Latent_Checkpoint_Save_Load_Advanced_EXP.json` | 输入完整H3嵌套AV latent；首次保存需显式打开`confirm_save` | 文件限制在ComfyUI输出目录，唯一文件名、写入后校验和原子放置；独立进程严格恢复已逐tensor/mask/元数据核对一致。它只恢复已完成latent，不保存Transformer、采样器、队列或CUDA状态 |
| 双时钟Euler NFE检查点/续跑 | `04-long-video/2026-08-23_H3_Dual_Clock_NFE_Checkpoint_Resume_Advanced_EXP.json` | 仅限`dual_clock_euler + native_flow`；必须填写精确模型合同，并把Conditioning的4路输出接入`NFE Run Contract`编译器，不能把普通`report`文本直接接到JSON输入 | 默认`disabled`且不写文件；一条256×256×22、4步真实H3任务已在第2步中断，由新ComfyUI进程续跑，最终联合AV latent、解码RGB与PCM均和不中断控制逐位一致。仅证明该固定合同，不支持多步历史、随机SDE或任意wrapper |

这些扩展均为追加节点。未放入旧工作流时不会改变旧采样路径。外部项目及模型权重不随本仓库分发。

以后生成的匿名评审页会先要求选择“可判断 / 原素材不足 / 播放问题 / 不确定”。只有“可判断”
的组才统计 A/B/平；其余组明确记为 `ABSTAIN`，不会被误算成方法平局或质量通过。旧版评审 JSON
仍可解析，已经完成的盲测页和私钥不会被重建工具覆写。

## 工作流目录

| 目录 | 用途 |
| --- | --- |
| `01-basic-generation` | 基础 T2VA、I2VA、FL2VA 和 Ref2VA |
| `02-audio-control` | 音频驱动、音频锁定、重混、参考音频和双时钟 |
| `03-image-video-edit` | 图像编辑、视频编辑和音视频参考 |
| `04-long-video` | 分段生成、长视频、断点恢复和拼接 |
| `05-speech-dialogue` | 单人语音、对白、演绎和 ADR（实验） |
| `06-face-refine` | 单人/多人脸部修复与 SAM3.1 追踪（实验） |
| `07-motion-detail` | 动态细节、尾段采样、FETA 与 Motion Recovery 二采修复（实验） |
| `08-multi-keyframe` | 中间关键帧和关键帧计划（实验） |
| `09-hybrid-model` | FL2VA/Ref2VA 混合模型研究（实验） |
| `10-speed` | SPEED 多分辨率采样研究（实验） |
| `11-studio-production` | 生产、字幕、时间线和批量辅助 |
| `12-system-memory` | 预检、显存、缓存、VBAR 和释放策略 |
| `13-latent-upscale` | 32 倍数对齐的 latent 放大 |
| `14-prompt-relay` | 全局提示词常驻、局部事件按时间接力（实验） |
| `15-sla-attention` | LightX2V Turbo-SLA 动态块稀疏 attention（实验） |
| `16-raven-streaming` | RAVEN因果分块流式T2VA与资源/合同保护（实验） |
| `17-skin-finish` | 解码后肤质收尾、遮罩门禁和人工审计（实验） |

每个目录都有自己的 `README.md`，说明适用场景、参数建议和已知限制。完整索引见 [`examples/workflows/README.md`](examples/workflows/README.md)。

## 常用稳定节点

| 节点 | 作用 |
| --- | --- |
| `MiniMaxH3AudioConditioningT8` | 统一处理任务类型、提示词、图像/视频/音频参考和 AV 条件 |
| `MiniMaxH3DualClockSamplerT8` | 使用独立的视频/音频时钟采样 |
| `MiniMaxH3AVDecodeT8` | 解码 AV latent 并保持音视频时间线 |
| `MiniMaxH3AudioLatentControlT8` | 音频锁定、重混和参考策略 |
| `MiniMaxH3LatentUpscaleBy32T8` | 按 32 像素网格放大 3D latent |
| `MiniMaxH3PreflightT8` | 在生成前检查模型、标签、尺寸、帧数和显存风险 |

## 提示词媒体标签

显式引用建议使用官方标签：`<Picture 1>`、`<Video 1>`、`<Audio 1>`。标签编号必须与实际输入对应。节点会规范化大小写和空白，并在报告中列出最终绑定结果。

- 只有一个明确的旧式参考输入时，可兼容历史工作流的零编号/单参考写法。
- 多个参考输入出现歧义、缺失或重复标签时，严格模式会直接报错，不会静默猜测。
- 看到 `prompt media tag validation failed` 时，请先检查任务类型、标签编号和对应输入是否真的接入。

## 模型与尺寸

常见目录：

```text
models/diffusion_models   H3 DiT 模型
models/text_encoders      Qwen/MiniMax CLIP
models/vae                视频 VAE、音频 VAE
models/loras              Turbo、质量或任务 LoRA
models/latent_upscale_models
models/face_detection/checkpoints
```

不要只按文件名判断兼容性；任务类型、模型家族、LoRA、VAE 和 ComfyUI 版本必须匹配。工作流中的 NOTE 和预检报告会提示推荐组合。

重要约束：

- H3 是联合 AV Transformer，视频和音频共享部分计算；参考图、参考视频、参考音频越多，显存和序列长度越高。
- 宽高应为 32 的倍数；`1920×1088`作为官方高分辨率参考面积，而不是学习型放大节点的执行禁令。该节点允许用户选择更大画布并报告高显存风险，实际能否完成取决于显卡、帧数和参考数量。
- 常用帧数遵循 H3 的网格约束，例如 22、124、362；不要只按“秒数”猜可用帧数。
- 16GB 显卡不要默认视为所有任务安全。先用较小画布/短片段通过预检，再逐步增加规模。
- 人脸检测模型可以通过 `ComfyUI/models` 下的目录符号链接或 Junction 指向其他盘；节点仍拒绝绝对路径和路径穿越，并在报告中标记外部存储。
- 环境审计会标记 RTX 50 系/compute capability 12.x 的 SageAttention 高 token 风险；Strict Sage 在 50,000 packed rows 以上直接拒绝执行，避免把可导入误当成高分辨率输出正确。
- SPEED、Prompt Relay、动态细节、Hybrid、多帧关键帧、多人脸修复和语音节点目前按实验功能使用；未通过本机验证的组合会 fail-closed。
- Motion Recovery 是独立的“pass-1分析 → 问题段扩时 → 部分去噪V2V → 恢复原时钟”链，不是光流插帧、修脸或单采样器增强。Analyzer默认使用`auto_conservative_exp`，Auto Gate通过ComfyUI原生lazy input实现真正旁路：平静片ABSTAIN时不请求任何二采节点，并原样输出pass-1画面和音频。736×416×124真实平静T2VA已验证`second_pass_requested=false`，旁路与pass-1的MP4、解码画面及PCM哈希均一致。Stock20 T2VA、I2VA、FL2VA和独立Ref2VA模型现各有一条真实二采成片，均恢复为124帧并通过严格音视频解码；I2VA还生成了`pass1_original`、`pass2_recovered_exp`和`blend_exp`三种完整对白音轨。完整人工试听确认默认原音正常；纯`pass2_recovered_exp`在中段会突然变成远处声音再恢复，已降为诊断用途；`blend_exp`在本次`pass1_mix=0.8`素材中正常，但仍只是单素材EXP结论。多模态单条成功不代表稳定提质或通用16GB安全，FL2VA/Ref2VA/I2VA粗采余量均曾低于512MiB，不要在首轮叠加EAV/STG/RF Restart/BlockCache。
- Enhance-A-Video / FETA 节点按论文公式从目标视频 Q/K 计算跨帧 CFI，只直接增强目标视频 attention 输出。原节点继续支持 Stock20 的 T2VA / I2VA / FL2VA / L2VA，以及严格限定的修正 Alpha8 Turbo8 T2VA；独立 Reference Composer 只开放原生 Stock20 Ref2VA 和任务型 Hybrid，不改变旧节点合同。上述七类路线现均至少完成一组 0.7MP 同输入、同 seed 的 disabled/apply 机械与媒体对照；Ref2VA 增强端最低显存余量仅约 417MiB，低于项目 512MiB 门槛。自动指标只能证明轨迹和联合音频发生变化，不能证明画质、参考遵循或声音更好，因此所有路线仍不宣称稳定提质、音频非劣或通用16GB安全。
- `Enhance-A-Video + Strict Sage Advanced EXP`由一个组合节点同时拥有 FETA 路由和本机 SageAttention HND 后端，避免第三方整块 Attention patch 绕过 FETA。它不会静默回退到 PyTorch attention；本机一条 1152×640×124、Stock20 实测完成 1000 次 FETA 测量和 1000 次 Sage 调用，失败/回退为 0，并通过三轮严格音视频解码。该单条机械验证不代表画质更好、声音非劣、速度更快或通用 16GB 安全；使用时不要再叠加 KJ Sage、BlockCache、STG 或其他全局 attention patch。
- `Enhance-A-Video + Prompt Relay Composer Advanced EXP`解决两个独立节点争用同一Attention入口的问题：它验证现有Relay绑定后，在一次路由中先执行局部事件Relay，再只对目标视频输出行应用FETA；关闭FETA时保留原Relay MODEL。当前仅开放Stock20 T2VA，一组736×416×124、20步基础机械对照及严格媒体检查已通过，但余量低于512MiB；不作为稳定提质、音频非劣或16GB安全路线宣传。
- `Enhance-A-Video + BlockCache / STG / Long Video`是三个隔离追加组合器：BlockCache只接受已核对合同的CPU缓存并按full=50/hit=1审计实际执行块；STG把同一FETA规则用于主分支和弱分支，默认审计50/49块及额外联合AV前向；Long Video保留原上下文/layout拥有者，并为每个`segment_index/context_frames`创建独立Stock20审计。三者当前只通过低负载确定性合同、注册和工作流导入检查；按要求未做压力测试，不宣称提质、音频非劣、提速、省显存或通用16GB安全。
- 基础 `MiniMax H3 LightX2V SLA Loader + Attention (Advanced EXP)`仍必须直接接干净的Dual-Clock模型并独占attention。若要保留KJ MiniMax H3 Sage，请使用专门的 `SLA + KJ Sage Composer`：连接顺序为 `Dual-Clock → KJ Sage → Composer`。已有的ComfyUI内置PyTorch/Comfy Kitchen `ModelAttentionBackend`会被识别并由Composer替换，外部Sol-Attn等未知Attention仍拒绝。SLA生成路径运行block-sparse Sage2，KJ只负责非SLA调用或 `dense_lora_control`；同一次Attention不会重复跑两个kernel。LoRA不再按单个固定SHA白名单判断，而是要求完整A/B结构并在当前H3基座上全部映射；这不能证明任意文件真实经过SLA训练。官方参考默认4步/6V/3A，8步及其他native_flow NFE会按实际前向次数审计并标为实验。当前结论仍不等于画质更好、速度更快、音频非劣或普遍16GB安全。
- Prompt Relay 与 Turbo8 组合时，连接顺序必须是 `UNET → Prompt Relay Conditioning → 修正 Alpha8 Bypass LoRA → DualClock Sampler`；把 LoRA 接在 Relay 前面会被主动拒绝。
- Prompt Relay 需要保留输入原声时使用 `lock_source`，并把节点的 `mux_audio` 接到最终保存节点；`native/remix/reference_only`仍可能因 H3 联合 AV Transformer 而改变声音。
- `drive_audio`是联合生成条件，不是确定性音素/口型约束。希望尽量保留原声时，使用
  `audio_mode=lock_source`、`add_source_as_reference=true`，提示词引用实际编号的`<Audio N>`；
  已知台词时再把逐字内容放入`<d>...</d>`。最终保存必须接`mux_audio`，不要误接AV Decode的
  `generated_audio`。这些设置能够保留最终输入音轨并提高语义提示强度，但不能保证嘴型逐音素同步；
  广播级精确口型需要在H3生成后使用专用唇形/人脸重定向工具。
- Prompt Relay 默认 `video_only_paper` 不改变旧工作流；只有显式插入 `Query Route`并选择`joint_av_exp`，才会把局部事件时间扩展到目标音频。该模式是H3实验扩展，不是论文已验证能力。
- `14-prompt-relay` 已提供 T2VA、I2VA、FL2VA、L2VA、Ref2VA、Hybrid 六种视觉任务模板，以及参考视频+同编号音轨、独立参考音频、联合AV和Turbo8模板；各任务的图片/视频/音频接线、模型顺序和已验证边界写在画布 NOTE 中。
- 六种视觉任务以及“参考视频+同音轨”“独立参考音频”现均至少有一组同输入、同seed、同NFE的baseline/Relay机械对照：16条成片均为736×416×124、24fps、32kHz双声道，严格音视频解码通过且无黑帧、冻结或削波；I2VA/FL2VA/L2VA/Hybrid的首尾帧锚点代理没有出现明显回退。该结果只关闭媒体与锚点合同，不代表Relay必然改善动作、身份、画质或声音；参考音轨不是原波形复制，感知优劣仍需观看和试听。
- 局部动作可用多个 `Prompt Relay Event Advanced` 节点逐项复制串联，并连接普通 Plan 或 Studio Packet桥接；普通Plan未连接时继续读取旧`local_prompts/time_ranges`，Packet桥接未连接时继续读取旧`events_json`。Studio 的 `Unified Cast + Sound Canvas + Prompt Compiler`编译结果仍是全局画面/声音事实源，节点不会自动拆剧情或改写事件。
- Prompt Relay显式时间范围必须按开始时间排列；`frames`只接受整数帧，`percent`只接受`0..100`。零个事件（包括整条Event链全部关闭）会只保留全局提示并无补丁直通；只有一个事件时也不安装补丁，通常应直接并入`global_prompt`。
- `Prompt Relay Preview Advanced`可以在不加载UNET、CLIP、VAE且不执行采样的情况下检查事件帧段、秒数、覆盖关系和Plan哈希；`Prompt Relay Resource Estimate Advanced`还能按画布、参考素材数量和query chunk估算H3 packed行数及Relay显式bias峰值，并在同一报告中列出736×416、1152×640、1920×1088三档矩阵。后者只是一项内存规划代理，不包含模型权重、完整attention激活、VAE/CLIP、VBAR或碎片，绝不等同于“16GB安全”。`14-prompt-relay`里的11份生成模板已内联Preview，另附一份纯时间线/资源预检工作流。
- Long Video需要分段事件时，使用独立的`Prompt Relay Long Video Window/Conditioning Advanced`；全局Plan只创建一次，本地渲染起点按`accepted timeline start - context overlap`计算。旧Long Video节点与旧工作流未改。本机已完成一条736×416、Turbo8、22帧上下文的segment 0→1真实链：输出124+102帧、事件没有从头重启、视频/音频各3轮严格解码通过，整卡峰值约15478/14984MiB。该单条结果仍只证明机械与时间线可用；最终音频接缝需试听，不能外推为普遍画质或16GB安全。

## 故障排查

遇到 OOM、NaN、花屏、音频丢失或人脸跳动时，请保留完整 ComfyUI Error Report，并记录：任务类型、模型文件、宽高、帧数、采样器/调度器、是否双时钟、参考输入、音频采样率和所有显式媒体标签。常见原因是：输入尺寸未对齐、重复连接 Sigma/采样器、参考标签错位、把实验节点接到不兼容的旧 core，或显存不足后继续复用缓存。

若KJNodes的`MiniMax H3 Mem Eff Sage Attention Patch`报
`sageattention is not new enough ... or could not determine CUDA architecture`：

1. 先运行`MiniMax H3 Environment Audit / 环境兼容审计 (Advanced)`，把
   `attention_backend`设为`sage_attention`、`enforcement`保持`report_only`。
2. 查看报告中的`environment.sageattention_runtime`。诊断会分别显示包/`core`导入错误、当前
   KJNodes所需的六个core符号、wheel报告的`smXX`架构和当前GPU架构是否一致；它不加载H3模型，
   也不执行attention kernel。
3. “已安装最新版”本身不能证明兼容；Torch、CUDA、Python、GPU架构与Sage wheel必须同时匹配。
   若诊断不通过，先移除KJ Sage节点改用stock attention，或为当前ComfyUI Python重装匹配wheel并
   完整重启。不要让工作流静默降级后仍宣称正在使用Sage。

## 开发与回归

```powershell
$env:PYTHONPATH = 'F:\AI-T8-video-onekey\ComfyUI'
& 'F:\AI-T8-video-onekey\python\python.exe' -m pytest -q
```

修改节点前请先阅读本地 `SKILL.md`、`features.json`、`meta.json` 和对应工作流目录说明。新增高级功能应保持旧工作流可加载；检测到未知 ComfyUI core 或第三方全局 patch 时，应明确报错而不是静默降级。

## 链接与许可

- [GitHub](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)
- [Comfy Registry](https://registry.comfy.org/)
- [MiniMax H3 官方仓库](https://github.com/MiniMax-AI/MiniMax-H3)

## 社区与服务

- [B站：T8star](https://space.bilibili.com/385085361)
- [YouTube：T8star AI](https://www.youtube.com/@T8star-Aix/)
- [API 服务注册](https://api.seedance.nz/sign-up?aff=5f4w)
- [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd)
- [模型网盘](https://pan.quark.cn/s/c9c267081fbf)
- [Hugging Face](https://huggingface.co/t8star)

插件代码采用 GPL-3.0-or-later。H3 权重、Qwen、ASR、SAM3.1 和第三方 LoRA 各自遵循其上游许可；本仓库不随插件分发这些权重。使用人声、人物参考图或克隆功能时，请自行确认素材权利并遵守适用法律和模型许可。
