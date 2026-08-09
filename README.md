# MiniMax H3 Audio T8

面向当前 ComfyUI 原生 MiniMax H3 的独立 T8 节点扩展。当前版本为 `1.10.0`，共注册
48 个节点，覆盖原生音画条件、音频控制与后处理、稳定双时钟采样、实验性多速率采样、
隔离的分段长视频续写、总时长编排、候选/接受状态与文件级合成、Ref2VA 单图/多图
参考的静态语义编辑，以及带异常释放保护、持久分段、精确时长后期和显式音色库的实验性语音链。

节点按稳定性与用途分为六个菜单：

| 菜单 | 状态 | 内容 |
|---|---|---|
| `T8/MiniMax H3/Audio` | 稳定 | 音画条件、音频处理、预检、双时钟采样与 AV 解码 |
| `T8/MiniMax H3/Audio/Experimental` | 实验 | 视频宏步/音频微步的多速率联合采样 |
| `T8/MiniMax H3/Still/Experimental` | 实验 | Ref2VA 静态图像条件、预检与候选帧解码 |
| `T8/MiniMax H3/Conditioning/Experimental` | 实验 | 对已有 H3 视觉参考/关键帧 Conditioning 设置全局参考强度，不修改采样链 |
| `T8/MiniMax H3/Long Video/Experimental` | 实验 | 总时长分段、断点续作、候选预览/接受、后台逐段执行、原子 manifest、已接受上下文与文件级合成 |
| `T8/MiniMax H3/Speech/Experimental` | 实验 | 描述/参考音色、ASR与身份评估、异常释放保护、逐句对白、长文本断点、ADR和显式音色库 |

本包不是把源音频简单塞进 latent：它按 ComfyUI 当前 H3 实现维护媒体展示顺序、
`<Picture N>` / `<Video N>` / `<Audio N>` 标签、联合 AV latent、首尾关键帧、参考媒体和
噪声掩码之间的契约。

## 安装与兼容性

将项目目录放入 ComfyUI 的 `custom_nodes/minimax-h3-audio-T8`，重启 ComfyUI 后即可在上述
菜单中找到节点。基础节点没有额外 pip 依赖，复用 ComfyUI 自带的 PyTorch、torchaudio 和
MiniMax H3 实现；可选语音校验才延迟导入 `faster-whisper` 或 `transformers`，缺少它们不会
阻止整个插件加载，也不会暗中下载模型。当前真实生成验证基线为 ComfyUI `0.31.0`、提交
`cbbc9dab1`，运行环境为 Python 3.10+。模型、VAE、CLIP 和可选 LoRA
仍需按具体任务自行安装。

`1.10.0` 保留此前36个节点的 ID、顺序、默认值和旧接线，只在其后追加12个语音可靠性/
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
| MiniMax H3 Speech Finalize / 语音完成与释放 (EXP/T8) | AUDIO 直通后执行 keep/cache-clear/全局卸载策略，并明确报告作用域 |
| MiniMax H3 Speech Studio / 语音工作台 (EXP/T8) | GraphBuilder 一站式串起条件、stock采样、音频解码、校验和释放 |
| MiniMax H3 Speech Abnormal-Exit Guard / 异常释放保护 (EXP/T8) | 在条件节点前登记 prompt 生命周期；异常、取消或非 OOM 错误令 Finalize 未执行时补发释放请求 |
| MiniMax H3 Speech VRAM Preflight / 显存预检 (EXP/T8) | 报告当前整卡空闲、PyTorch占用和 DynamicVRAM 配置；只做当前态门槛，不授予 `memory_safe` 标签 |
| MiniMax H3 Voice Library Save / Load / Delete (EXP/T8) | 显式保存/读取经授权音色；不允许默认同名覆盖，删除移到可恢复回收目录 |
| MiniMax H3 Speech Performance Direction / 演绎控制 (EXP/T8) | 添加情绪、语速、音高、能量和非语言提示方向；当前标定矩阵未通过，只能作 prompt EXP |
| MiniMax H3 Speech ADR Exact Fit / 配音精确时长 (EXP/T8) | 拒绝、补裁或有界相位声码器变速，并可确定性移调；输出精确到 sample，不声称口型同步 |
| MiniMax H3 Speech Long Form Start/Accept/Control/Compose (EXP/T8) | 原子 accepted manifest、断点恢复、合作式取消、分段可播放预览、哈希校验和最终精确时间线合成 |
| MiniMax H3 Joint Dialogue Conditioning / 多人同段条件 (EXP/T8) | 2–3人 Ref2VA 同段实验；真实两人探针质量门槛失败，不是稳定推荐路径 |
| MiniMax H3 Visual Reference Strength (EXP/T8) | 在现有 H3 positive Conditioning 后写入全局视觉参考强度；可能缓解过度平滑/蜡感，也可能削弱身份、构图和首尾帧 |

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

API 示例：`examples/ref2va_visual_reference_strength_exp_api.json`；可拖入画布的工作流：
`examples/workflows/H3_Ref2VA_Visual_Reference_Strength_EXP.json`。前端示例使用完整
`minimax_h3_ref2va_int8_convrot.safetensors`、736×416、124帧和20步基线，导入后先替换
占位参考图。该参数本身不要求这些采样值，接入旧工作流时保持用户原有 sampler/scheduler 即可。

本机单参考、单 seed 的完整矩阵已经跑通：无节点与显式 `0.999` 的解码视频/音频最大绝对
误差均为0，`0.950` 重复两次也逐帧逐样本一致。该案例没有得到“数值越低越去油”的证据：
`0.995～0.950` 会改变姿态、表情、动作轨迹或构图，`0.950` 的输出偏移最大，面部高频代理还
低于 `0.999`。因此节点只适合受控 A/B；不能把某个默认值宣传成稳定修复。当前矩阵的最小
显存余量约35MiB，远低于项目512MiB安全门槛，也不能据此称16GB安全档。

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

- `speech_performance_adr_api.json` / `H3_Speech_Performance_ADR_Stock20_EXP.json`；
- `speech_longform_resume_api.json` / `H3_Speech_LongForm_Resume_Stock20_EXP.json`；
- `speech_longform_compose_api.json` / `H3_Speech_LongForm_Compose_EXP.json`；
- `speech_voice_library_save_api.json`、`speech_voice_library_load_api.json` 及对应前端工作流；
- `speech_voice_library_delete_api.json`、`speech_vram_preflight_api.json`、
  `speech_longform_control_api.json` 及对应维护工作流；
- `speech_joint_dialogue_exp_api.json` / `H3_Speech_Joint_Dialogue_Stock20_EXP.json`，仅用于复现
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

可直接导入的 `examples/workflows/H3_Long_Video_Background_22F_ScenePlusIdentity_EXP.json`
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
`examples/workflows/H3_Long_Video_Background_22F_EXP.json` 后，`Background Start` 的默认值仍是
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

旧手工链的画布/API 示例仍为 `examples/workflows/H3_Long_Video_22F_EXP.json` 与
`examples/long_video_segment_api.json`。接受状态画布/API 示例为
`examples/workflows/H3_Long_Video_Accepted_22F_EXP.json` 与
`examples/long_video_candidate_accept_api.json`；完成全部片段后再单独运行
`examples/long_video_compose_api.json`。推荐的总时长自动恢复画布/API 示例为
`examples/workflows/H3_Long_Video_Auto_Resume_22F_EXP.json` 与
`examples/long_video_auto_resume_api.json`；它自动管理 index、final、时间轴和断点位置，但保留
逐段人工预览/接受。显式后台画布/API 为
`examples/workflows/H3_Long_Video_Background_22F_EXP.json` 与
`examples/long_video_background_api.json`；只有这组示例连接 Background Start 与 Auto Queue。

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
API 示例见 `examples/still_image_edit_api.json`；可直接拖入画布的完整示例见
`examples/workflows/H3_Still_Edit_22Frames_EXP.json`。两者默认使用512×512、22帧、20步，
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

可导入的 API 结构示例见 `examples/dual_clock_4step_api.json`。其中模型文件名是占位符，
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
叠加 Sigma Shift 或外部 scheduler。示例见 `examples/multirate_exp_api.json`。

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
| `H3_Audio_Lock_Source_Stable_4V4A.json` | `lock_source` | Conditioning `mux_audio` | 锁定源 latent，保留干净输入原音轨 |
| `H3_Audio_Remix_Source_Stable_4V4A.json` | `remix_source` | AV Decode `generated_audio` | 以默认0.35强度保留节奏/语音结构并重绘声音 |
| `H3_Audio_Reference_Only_Stable_4V4A.json` | `reference_only` | AV Decode `generated_audio` | 输入音频仅作 `<Audio 1>` 参考，目标音频重新生成 |
| `H3_Turbo_Stable_4V4A.json` | `native` | AV Decode `generated_audio` | 无需输入音频的原生音画联合生成 |

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

## 示例与测试

可直接拖入画布的稳定 4/4、三种输入音频模式、EXP 4/8、EXP 4/10、Ref2VA 22帧静态候选编辑，
以及以下长视频示例位于 `examples/workflows/`：

- `H3_Long_Video_22F_EXP.json`：手工逐段续写基线。
- `H3_Long_Video_Accepted_22F_EXP.json`：候选预览、接受和可恢复状态链。
- `H3_Long_Video_Auto_Resume_22F_EXP.json`：总时长编排与人工审核后自动恢复。
- `H3_Long_Video_Background_22F_EXP.json`：后台自动排队长链。
- `H3_Long_Video_Background_22F_ScenePlusIdentity_EXP.json`：完整场景与身份裁剪双参考的后台长链。

API 示例见 `examples/audio_lock_api.json`、
`examples/dual_clock_4step_api.json`、`examples/multirate_exp_api.json` 和
`examples/still_image_edit_api.json`。替换 API 示例里的模型、VAE、CLIP、可选 LoRA、
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
采样数学或模型权重，可完成 API 工作流静态检查、生成 stock Euler/双时钟严格 A/B、按节点
和采样进度记录显存曲线，以及比较两次运行的控制变量与峰值增量。

第一轮稳定 Turbo 对照必须统一为 4 步、相同模型/LoRA/Prompt/seed/尺寸/帧数，并建议关闭
预览。完整命令、判定规则和限制见 [显存验证方法](docs/VRAM_VALIDATION.md)。在取得真实 OOM
traceback 和有效 A/B 前，不应把高显存直接归因于双时钟节点，也不应盲目替换 INT8 旁路
LoRA 或关闭 VBAR。

2026-08-07 的本机暖缓存实测中，`0.6M`、362 帧、4 步的 stock Euler 与双时钟设备峰值
分别为 16,213.5 MiB 和 16,182.2 MiB，PyTorch 峰值均为 14,573.5 MiB；未发现双时钟路径
存在实质峰值增加。两条路径都已非常接近 16 GiB 上限，这个单机结果不能替代反馈用户的
精确工作流、OOM traceback 和冷启动换序复测。
