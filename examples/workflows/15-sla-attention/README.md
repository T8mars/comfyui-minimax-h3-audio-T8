# LightX2V SLA Attention 工作流

本目录提供 MiniMax H3 FL2VA 的 LightX2V Turbo-SLA 实验工作流：基础版由SLA节点独占attention；组合版允许保留KJNodes的MiniMax H3 Sage完整forward，并由专用Composer逐调用选择唯一后端。普通SLA节点仍不接受KJ、Sol-Attn、FETA、Prompt Relay、BlockCache、STG或其他attention/block patch。

## 四份入口

- `2026-09-02_H3_SLA_Precision_V2_FL2VA_FP8_8Step_Advanced_EXP.json`：新画质修复入口。固定PlagueKind v1.4.3提交`066ada9`的FP32路由、直接Triton稀疏核、首尾Dense、精确语言/音频保护和动态LoRA旁路；运行后必须通过逐逻辑步Audit。当前仍为Advanced EXP。
- `2026-08-26_H3_Turbo_SLA_Profile_Router_FL2VA_Advanced_EXP.json`：推荐入口。默认使用普通修正Alpha8 Turbo8；只有明确选择实验Profile时才进入SLA精确路线。
- `2026-08-22_H3_LightX2V_SLA_FL2VA_4Step_Advanced_EXP.json`：基础严格版，Dual-Clock后直接接SLA。
- `2026-08-22_H3_LightX2V_SLA_KJ_Sage_Composer_FL2VA_4Step_Advanced_EXP.json`：KJ组合版，连接顺序为 `Dual-Clock → KJ MiniMax H3 Sage → SLA + KJ Composer`。`ModelAttentionBackend`不是必需；若已有且属于ComfyUI内置PyTorch/Comfy Kitchen后端，Composer会明确记录并替换它。Sol-Attn和未知Attention仍拒绝。

## 使用方法

0. 新的Precision V2是一条独立路线：使用FP8 FL2VA基座、8 NFE、shift 12/3，并按工作流固定顺序连接`Dual-Clock → SLA Dynamic LoRA Bypass V2 → SLA Precision V2 Attention → BasicGuider`。不要与下面三条旧LightX2V/Sage2路线混接；旧节点只为兼容和历史诊断保留。
1. 默认可准备 `minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors`，放入 `ComfyUI/models/loras`。节点不再把所有SLA能力绑定到这一个文件SHA；其他H3 SLA safetensors必须具有完整A/B LoRA对，并在加载时全部映射到当前H3基座，否则仍会拒绝。
2. 使用 FL2VA 基座，同时接入首帧与尾帧。推荐模板默认重复同一张近景图，避免把首尾条件冲突误判成SLA失效；替换尾帧时应保持人物占比、视角和场景尺度接近。近景人物直接跳到航拍小点属于高风险镜头，应延长时长、拆分镜头/加入中间关键帧，或改为只接首帧。上游发布证据覆盖 BF16 checkpoint 和 LightX2V FP8 配方；没有证据覆盖本地 INT8 ConvRot + SLA。Profile Router 的SLA精确档因此会拒绝INT8；仅研究旁路档允许INT8动态LoRA，默认在采样进度15%～90%启用SLA。
3. SLA发布路线使用 `native_flow`、4 NFE、视频 shift 6、音频 shift 3。官方配置的`infer_steps=5`是5个sigma网格点，调度器实际执行4次模型前向。旧节点仍可记录其他NFE实验；Profile Router不会把它们称为上游精确路线。
4. `apply_lightx2v_sla`目前只保留为未发布的诊断计划：低于50K packed token会主动全稠密，较长序列会安排首尾稠密、中间稀疏并保护文本/首尾帧/音频key；它不再被推荐为“质量安全”模式。只有研究上游固定85%行为时才选 `apply_lightx2v_sla_upstream_exact_exp`。`dense_lora_control`只做“同一SLA LoRA改用稠密Attention”的机械归因，不能代替普通Turbo LoRA基线；完全旁路选 `disabled_identity`。
5. 采样后必须连接 Runtime Audit。Profile Router默认Turbo8应报告8次模型前向和0次SLA调用；SLA精确档应看到4×50次稀疏内核。INT8旁路档默认应报告`dense/sparse/sparse/sparse`，对应4步模型前向开始点0%、25%、50%、75%；90%以后没有新的模型前向，不能误称为“末步恢复稠密”。这些计数只证明路由执行，不代表画质通过。

若同时安装了旧版 `ComfyUI-PainterNodes`，它可能给 `PackedLayout` 安装一个仍转发已移除 `frame_count` 参数的全局补丁。SLA节点现在会复用T8 Hybrid的可执行兼容探针：只有在该补丁带有已知标记、其闭包中的原生构造器独立通过当前键帧+参考排列合同后，才解除这个过时包装；未知全局补丁仍会拒绝。更新后需要完整重启ComfyUI，旧的红框状态不会自行证明新代码仍失败，应重新执行节点并查看新的错误文本。

组合版中，短序列诊断性稠密fallback的200次主调用会全部进入KJ dense；长序列4 NFE自动模式由首尾100次KJ dense和中间100次SLA block-sparse组成；显式上游精确EXP才是200次SLA sparse和0次KJ。`dense_lora_control`始终为200次KJ。每次Attention只计算一个后端，并不是把两个加速结果相乘；这些计数只说明路由真实发生，不说明画质可用。

## 已完成验证

- Precision V2固定上游为`PlagueKind/ComfyUI-PlagueKind-Nodes` v1.4.3提交`066ada9eb2378f392cc815663f63c4eef1060b4a`（MIT）。`block_map.py`和`kernel.py`保持上游逻辑；`patch.py`只增加返回运行状态和逐逻辑步计数，不参与Attention数学或路由选择。
- RTX 4060 Ti / sm89 BF16尾块微探针相对Dense参考的relative MAE为`0.000434825`、cosine为`0.99999988`，FP32 Router top-k精确匹配独立参考且重复运行bit-exact。
- 736×416×124真实复跑逐步记录：步骤0/7各50次Dense，步骤1–6各50次Sparse，总Sparse 300次；序列12,785 token、400个key block、选中59个（含20个语言/音频保护块），kernel failure/fallback为0。新增计数前后decoded video/audio哈希完全相同。
- 同输入、同Seed、同FP8底模、同SLA LoRA、同8 NFE/12:3的Dense XFormers对照也已完成。Precision V2首轮137.828秒、Dense 157.297秒；采样约107.188秒对131.922秒。两条严格解码、ASR对白可辨、SyncNet均-1帧，400ms画面延迟负对照均+9帧。正常速度匿名A/B人审仍待用户完成。
- 最新Precision V2最低空闲236MiB（此前211MiB），Dense为245MiB，均未过512MiB门；不得写成通用16GB安全或正式稳定默认。
- 固定 LightX2V 代码 revision：`f8aee98b5462cca8d7288888146ebd95592bf266`。
- 固定模型 revision：`10ade67cd15ff7a135fa35c2a0673ea96c839247`。
- 当时机械验证使用的参考LoRA SHA-256为`5CAE6DF40A06EA825F85FC8876C9EA1C9692C833A9AF07BB8B3BAC9CE2A71BAC`，208个LoRA patch全部映射并加载；这只是历史证据，不再是运行时唯一白名单。
- RTX 4060 Ti / sm89 上，一条 256×256×22、4 NFE 的 FL2VA INT8 兼容机械运行完成：4×50 次稀疏调用，失败和回退均为 0。
- 普通Turbo8以736×416×124、近景→航拍的不同首尾帧完成一次串行真实复测：124帧和32kHz双声道严格解码通过，但完整人审确认约一秒后进入持续的不合格强制变景，不能称为画面连贯。该次运行没有进入SLA路由（8次模型前向、0次SLA调用），因此它证明的是首尾条件冲突，不是SLA内核崩坏；最低剩余显存418MiB，亦低于512MiB安全门。
- SLA精确档以同景别首尾帧完成一次736×416×124机械成片，但后续盲评导出把SLA候选标记为硬失败且所有维度都偏向普通Turbo8；由于评审同时选择`unsure`，形式结论为ABSTAIN而不是胜负票。无论哪种解释，都不能推荐当前INT8标准weight-patch SLA。Profile Router新增的`sla_4step_int8_bypass_exp`只用于验证动态LoRA旁路能否避免底模重数量化，默认仍是普通Turbo8。
- 用户本轮提供的文件名含`124f`的问题视频实际是704×416、22帧、0.9167秒；容器信息优先于文件名。

首尾素材的低负载审计也支持上述归因：近景与航拍图缩放到相同画布后，像素相关系数仅`0.032`、边缘IoU仅`0.0828`，ORB只有3个有效匹配且无法求得单应性；它们不是一个可由普通小幅镜头运动连接的同尺度锚点对。另一方面，同一稀疏块图上的随机张量检查显示，当前`spas-sage-attn`量化Sage2相对FP32参考的RMSE约`0.00517`，高精度Triton稀疏核约`0.00030`。这说明量化误差值得后续单独A/B，但不能解释那条0次SLA调用的失败过渡。

## 科学边界

这里实现的是 LightX2V 发布版 learned dynamic block router 数学与 Sage2 block-sparse 内核接入，不是可脱离SLA LoRA的通用加速开关，也不是通用SLA论文全部 sparse+linear 分支。现有证据只能把普通Turbo8提升为当前消费级推荐回退；SLA仍是BF16/LightX2V FP8证据族上的4-NFE实验路线。结构映射、内核零失败和单条连贯成片都不能证明画质更好、速度更快、音频非劣或普遍16GB安全。
