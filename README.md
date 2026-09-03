# MiniMax H3 Audio T8

简体中文 | [English](README_EN.md)

MiniMax H3 的 ComfyUI 节点包：生成视频和声音，也支持参考控制、长视频、修脸和加速。

当前版本：**1.67.0** · GPL-3.0-or-later

## 主要功能

- 文生、图生、首尾帧、参考图/视频/音频、混合参考
- 视频和音频双时钟采样
- 原声保留、参考音频、音轨合成；Audio Refine 可选接入 Turbo4/8、双采、PDD、EAV、Prompt Relay 和长视频8步
- 多关键帧、长视频续写、节点内一键串行、断点恢复、Latent 放大；双采可用v8人物安全RGB后处理，仅在人工逐帧alpha内采用T2、其余画面与音频保留D0
- 单人/多人脸部修复、SAM3.1 追踪、Skin Finish
- Prompt Relay、SPEED、SLA、PDD、Enhance-A-Video
- 全本地 MV Vocal Lock V3：官方 Ref2V Turbo4 配置、独立人声驱动、逐镜头视觉合同、串行生成、断点续跑、原曲最终单次混入
- FastH3 Preview：T2VA 4步，可选真实 learned-gate VSA 90% 稀疏执行
- OpenVDN MiniMax H3：独立T2VA混合注意力架构，支持DMD 8步与Stage B 50步
- NVIDIA H3 Super Acceleration：H3 4步草稿经完整 LTX VAE 编码后接 LTX-2.5 3步 Refiner（TAEHV仅最终解码）
- FlashVSR v1.1：成片2×/4×超分、固定LCSA、动态预算候选和低显存分块，原音频不处理
- RAFT运动审计、轨迹控制、RealBasicVSR、FreeNoise、AYS校准契约、CADS视觉参考退火

带 `Advanced` 或 `EXP` 的节点属于高级/实验功能，建议直接使用配套工作流。

## 安装

在 ComfyUI Manager 搜索 `MiniMax H3 Audio T8`，安装后重启 ComfyUI。

> **先更新 ComfyUI 本体：** 本节点包使用新版 ComfyUI 的原生 MiniMax H3、`comfy_api.latest`、模型补丁和权重适配接口。只更新节点包但保留旧版 ComfyUI，可能导致整套 T8 节点同时爆红或显示缺失。请把 ComfyUI 本体、前端和 Manager 一起更新，再完全退出并重启。

手动安装：

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git minimax-h3-audio-T8
```

## 快速开始

1. 打开 [`examples/workflows`](examples/workflows)。
2. 新用户先用 `01-basic-generation` 或 `02-audio-control`。
3. 把 JSON 拖进 ComfyUI，替换模型和素材后运行。
4. 高级工作流先看同目录的 `README.md` 和画布 NOTE。

希望只排队一次就自动完成全部长视频分段时，使用`04-long-video`中的
`In_Node_Long_Video_Loop`工作流；需要逐段人工挑片时继续使用原有Background/Accepted路线。
需要同时使用Prompt Relay和Enhance-A-Video时，使用同目录的
`In_Node_Long_Video_Prompt_Relay_EAV_Stock20_Advanced`工作流（原生20步，不接Turbo LoRA）。
需要尾段细分或独立低 Sigma 二次采样时，使用同目录带
`Long_Video_Sampling_Plan`的工作流；断开该节点即恢复旧路径。

`2026-09-02_H3_Native_Masked_Video_Context_Plan_B_Segment0_Starter_Advanced_EXP.json`和
`2026-09-02_H3_Native_Masked_Video_Context_Plan_B_Advanced_EXP.json`组成独立的长视频Plan B，不替换上述
默认路线。必须先用前者生成并保存第0段，再以相同`chain_id`用后者从第1段继续；不能用旧双时钟默认工作流
生成这条链的第0段。续段工作流把上一段校验过的原生video latent尾部复制到当前段开头，
并用ComfyUI原生mask锁住这部分画面；`context_audio`固定为`video_only`，上一段音频不注入，当前段
audio tensor及已有Vocal Lock audio mask原样保留。首轮736×416同上下文、同Seed雨声A/B严格解码、
帧数、运行拓扑和上下文不变性通过；用户盲评画面“都差不多、都没问题”，揭盲后A为Plan B、B为旧路线，
但两条都有可闻杂音，所以只关闭单样本画面不劣门。随后同合同的纯器乐A/B也被用户判定为两路严重噪音。
单变量复测确认旧`dual_clock_euler`是第一项独立问题：同一416×224首段仅改为当前ComfyUI原生AV
`euler + native_flow`后，PCM直流偏置从`0.21313`降到`0.00060`。推荐工作流现固定原生AV Euler；
修复配置的首段、软上下文和Plan B续段均无削波、严格解码
通过，直流偏置分别为`0.00060/-0.00150/-0.00105`，共享context不变。用户试听后认为比旧版好，但仍感觉
音频可能有问题，因此没有通过音频门，也没有选出软续段/Plan B赢家。追加的416×224“古典音乐 + 人物只说
一次你在哪里”4/8 NFE对照均严格解码、无削波；8 NFE逐字命中，4 NFE实际被识别为“你在那里”。随后又
建立只更换旧通用EMA/校正FL2V Alpha8 LoRA的4 NFE匿名对照：用户判定A声音没问题、B声音非常轻且不对；
揭盲后A为Alpha8，B为旧通用EMA，结论已绑定成片和LoRA哈希。用户随后指定新版step600 EMA_B
`minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors`作为后续配置；两份独立Plan B工作流已固定该文件，
SHA-256为`80FCC655…90DFAE`，旧Long Video工作流不改。首个新版EMA_B 416×224完整链虽严格解码、无削波、
DC接近零，但复核发现首段和续段重复请求“你在哪里”，违反全片只说一句的合同，已标记为不可人审、只保留
机械证据。用户仍实际试听并明确反馈“A和B都没问题声音”；按完整片SHA揭盲后A为软上下文、B为Plan B，
两条声音诊断均通过且无偏好，但不覆盖重复对白合同、口型或独立接缝评价。运行器随后改为首段说一次、续段人物
静默且古典音乐继续，并已用同一新版EMA_B完成修正版真实A/B：三份源片和两份226帧完整片严格解码通过，
共享context哈希不变；VAD筛查在两个续段均检出0段对白，第0段单独识别为“你在哪里”。用户试听修正版后再次
确认“A和B都没问题声音”；揭盲后A为Plan B、B为软上下文，所以本精确样本的声音非劣门通过且打平。完整AAC
上的ASR受音乐影响在“哪里/那里”之间不稳定。接缝分析器原先按最大面积误选了Plan B前两帧的背景假脸；改为
从第0段人脸连续追踪后，Plan B/软上下文接缝SFace为`0.873/0.875`，两条续段均102/102帧追踪到主脸，旧的
Plan B低身份分属于分析误报。共享对白段SyncNet中心裁剪为-3帧、三种动态人脸裁剪均为-4帧（25fps），400ms
负对照移动+9/+10帧；置信度偏低且未过±1帧机械门。另提供两个音频PCM不变、SyncNet回到0的诊断候选：
固定延后4个24fps帧会冻结开头4帧并舍弃末尾4帧；平滑首尾保留会改变说话前后运动速度并混合分数帧。
用户随后以正常速度完成原始/固定/平滑三方真人对比并反馈“3组差不多，都还行”。因此本哈希绑定的原始片
真人口型通过；两种校准没有可感知优势且各有已知画面代价，均不接入工作流。既有旧配置4/8 NFE样本置信度
及负对照不足，仍不能证明加步数能修。该对白段在A/B中逐字节相同，故本结论不选择Plan B路线，也不关闭
精确用词或续段主观接缝。本次修正版第0段/软续段/Plan B最低余量为490/475/527MiB，整对仍低于512MiB项目门；
因此继续标为Advanced EXP，不宣称普遍口型稳定、通用16GB安全或Plan B普遍更优。

由于416×224不足以判断接缝画质，随后按完全相同合同补跑960×544（约0.522MP）真实A/B。第0段及两条
续段精确为124/102/102帧，所有源片与两份226帧完整片严格音画解码，共享context前后不变；本次
第0段/软续段/Plan B最低空闲显存为1009/545/1122MiB，三阶段均通过512MiB项目余量门。用户盲评两条
差不多，但指出两条在接缝处都有颜色跳变；揭盲后A为软路线、B为Plan B。因此路线本身未分出胜负，
而旧接缝主观质量不能算通过。本次资源通过只绑定这一个本机profile，不外推通用16GB安全或Plan B普遍更省显存。

两份独立Plan B工作流现追加可选且默认开启的`MiniMaxH3LongVideoColorMatchT8Advanced`，位于
`Output Trim`之后、`CreateVideo`之前。最初只比较5帧RGB均值的V1虽通过机械门，但用户盲评B跳变较少、
两条仍有且A左侧明显，故V1已拒绝；揭盲后A为软路线、B为Plan B。V2改为当前ComfyUI内置
ColorTransfer同类的pooled Reinhard Lab色彩/对比度匹配，再叠加8x5局部分区RGB残差补偿；每像素通道
总改变量仍硬限0.02并在24帧内渐隐。疑似切镜、旧状态schema、状态/校验和/chain/画布不匹配或非SDR
输入不会猜测校正；关闭后逐像素原样通过，开启与关闭都不改原生AV latent或音频。

V2最终960×512（0.49152MP）同Seed真实A/B得到124/102/102帧，全部源片、完整片和匿名审查传输严格
解码，共享latent context及第0段颜色参考在两条续段前后不变。软/Plan B最大局部RGB跳变分别从
0.014492降至0.001714、0.008184降至0.001238；最大整帧RGB均值跳变降至0.000021/0.000004。
第0段/软/Plan B最低空闲显存612/532/515MiB，本精确运行通过512MiB门。用户完成哈希绑定盲评后反馈
“左边还是能看到一点点跳变，右边就好很多”；揭盲后A（左）为软路线，B（右）为Plan B。因此本样本
接受默认开启Color Match的Plan B接缝色彩连续性，软路线仍保留轻微残差。该单样本结论不等于两条都
完全消除跳变，也不外推身份、音频、通用16GB安全或Plan B普遍更优；继续限制校正幅度，避免为追求
软路线零残差而引入不自然偏色或慢回色。

[查看全部工作流分类](examples/workflows/README.md)

## SLA Precision V2（画质修复路线）

新的 [`15-sla-attention/2026-09-02_H3_SLA_Precision_V2_FL2VA_FP8_8Step_Advanced_EXP.json`](examples/workflows/15-sla-attention/2026-09-02_H3_SLA_Precision_V2_FL2VA_FP8_8Step_Advanced_EXP.json) 是本轮 SLA 画质修复入口。它不会修改或删除旧 SLA 节点，而是追加三个 V2 节点：动态 model-only LoRA 旁路、Precision V2 Attention 和采样后 fail-closed Runtime Audit。

根因不是简单换 Seed。旧路线使用 pooled BF16 路由、`spas-sage-attn` Sage2 Q/K 量化、128×64块、按模型调用次数判断步骤、粗前缀保护和全程稀疏；Precision V2 固定到 PlagueKind v1.4.3 提交 `066ada9` 的 FP32 路由与直接 Triton FP32 online-softmax 稀疏核，并使用真实 sigma 逻辑步骤、精确语言/音频段保护、首步与末步 Dense。FP8 底模上的 SLA LoRA 作为动态残差注入，不合并回底模，也不触发二次量化。

推荐模板固定 736×416×124、8 NFE、shift 12/3、32×32块、请求90%稀疏、步骤0/7 Dense、步骤1–6 Sparse。真实复跑逐步审计精确记录到两端各50次 Dense、中间每步50次 Sparse（总计300）、20个保护块、0次 kernel fallback；生成的 decoded video/audio 与修改审计计数前的同 Seed 成片逐字节一致。同输入、同 Seed、同 SLA LoRA 的 Dense XFormers 对照中，Precision V2 本轮端到端用时126.438秒，对照157.297秒；按首轮成对计时分别约快12.38%和18.75%（端到端/采样），只代表这一个环境和素材。

两条对白成片均严格解码，ASR识别为“你在干嘛呢，我在这里啊，看看效果如何”，官方 SyncNet 均为-1帧（25fps），把画面延后400ms后均测得+9帧。32帧接触表未见旧问题中的一秒后人脸/运动崩坏；用户随后正常速度匿名观看并判定“两个差不多，都还行”。揭盲后A为Dense、B为Precision V2，因此本素材通过感知非劣门，但不建立Precision V2普遍更优结论。RTX 4060 Ti 16GB 上 Precision V2 最低空闲236MiB（此前同画面211MiB），Dense 对照245MiB，均未达到项目512MiB安全门，因此本路线保持 Advanced EXP，不宣称所有16GB环境安全。

## 全本地 MV / 口型分镜

优先使用 [`24-mv-lipsync`](examples/workflows/24-mv-lipsync) 中的 `VocalLock_V3_Official_Ref2V_Turbo4` 工作流：同时加载人物参考图、完整原曲 `full_song` 和同时间线的隔离人声/清晰对白 `vocal_lock_audio`。它沿用 V2 本地分镜，只让独立人声逐段进入本地 H3 `lock_source`，完整原曲不进入 H3 或分段候选，只在画面合成后一次性混入最终成片。

V3 提示词使用官方 Ref2VA 六段顺序和 `<Subject 1>` / `<Picture 1>` / `<Audio 1>: fully_copy` 关系，增加唯一人物/人脸和逐镜头视觉合同；人声场景强制中近景、正面或 3/4 脸及无遮挡嘴部。旧工作流保留用于兼容，但不能替代独立人声口型验收。整个路线不调用远程 H3、LLM、TTS、分离或视频 API，也不在节点内部提交 HTTP `/prompt`。

早期一条 5.152 秒清晰英语对白样例通过了 SyncNet 与用户正常速度口型审核，但其人物周围发虚。后续同图、同音频、同失败 Seed 对照证明，主要问题不是 Seed 或 H3 基模本身，而是把通用 LarryVrh EMA Turbo LoRA 和非官方 8 步/shift 6:3 组合用于 Ref2VA。当前推荐路线固定为官方 Ref2V Turbo v0.1 LoRA（strength 1.0）、4 步、Euler/simple、shift 12/3、1024×768；同 Seed 重测已消除持续双脸/拖影。

官方 Ref2V 配置下的 32 秒 / 5 镜真实成片现已完成：5/5 镜头、768/768 帧、1024×768、24fps，完整原曲只混入一次；严格视频/音频/联合解码通过，默认多线程视频解码重复 20 次为 0 异常。逐镜抽查未见重复脸、背景人脸或持续人物边缘拖影。官方 SyncNet 对 5 镜隔离人声测得 `0/-1/0/-1/0` 帧偏移，400ms 画面延迟负对照测得 9 帧。用户完整观看后明确反馈“32秒这个已经没问题了，完美”，并取消约 90 秒要求；该结论绑定最终主片 SHA-256 `E833277844E6980FDEACF9BDFD5C61FFE48AEFDB3E1EBA6869C363777B7DD75F`，本轮长 MV / Lip Sync 验收完成。

## 模型目录

| 模型 | 放置目录 |
| --- | --- |
| H3 主模型 | `models/diffusion_models` |
| 文本编码器 | `models/text_encoders` |
| 视频/音频 VAE | `models/vae` |
| Turbo、SLA、PDD 等 LoRA | `models/loras` |
| Latent 放大模型 | `models/latent_upscale_models` |
| H3 Fun Control（新版 Model Patch / 旧版 ControlNet） | `models/model_patches` / `models/controlnet` |
| TAEH3 快速预览模型 | `models/vae_approx` |
| RAFT 光流模型 | `models/optical_flow` |
| RealBasicVSR | `models/upscale_models` |
| FlashVSR v1.1 整套目录 | `models/FlashVSR-v1.1` |

FL2VA、Ref2VA、pruned 和完整基模不要混用。

## 常用设置

- Turbo 双时钟常用 shift：视频 `12`，音频 `3`
- 保留原声：`audio_mode=lock_source`，保存节点连接 `mux_audio`
- 媒体标签：`<Picture 1>`、`<Video 1>`、`<Audio 1>`，编号必须对应输入
- 宽高使用 32 的倍数；显存不足时先降分辨率、帧数和参考数量
- `1920×1088` 只是风险参考面积，不是执行上限；更大画布会警告但不拦截，OOM 与画质风险由用户承担
- 不要同时叠加多个接管 sampler、attention 或 MODEL forward 的节点

## PDD 8 步

工作流在 [`19-pdd-acceleration`](examples/workflows/19-pdd-acceleration)。PDD 必须使用专用节点，不能当普通 LoRA 加载。

转换后的 FL2VA / Ref2VA PDD 加速 LoRA 下载：[t8star/MiniMax-H3-Acc-8Step-comfy](https://huggingface.co/t8star/MiniMax-H3-Acc-8Step-comfy)。下载后放到 `ComfyUI/models/loras`，并选择与基模一致的版本。

默认：Euler/simple、8 NFE、shift `12 / 3`、CFG 1。

新版 ComfyUI 会自动使用官方 PDD FinalLayer；旧版继续走本项目的兼容回退。判断只看运行时能力，不按 ComfyUI 版本、模型哈希或文件大小拦截。

目录中同时提供 FL2VA / Ref2VA 的学习型 latent 双采工作流：严格把同一条 PDD 轨迹分成 LOW 4 步和 HIGH 4 步，总 NFE 仍为 8。正式 Ref2VA 预设为 864×480×22、1.5×；FL2VA 双采暂保留实验标记。

## FastH3 VSA 4 步

工作流在 [`10-speed`](examples/workflows/10-speed)，文件名含 `FastH3_VSA_T2VA_4Step`。下载官方
[VSA Data-Free 适配器](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/blob/main/vsa-datafree/adapter_model.safetensors)，放到
`models/loras/FastH3-VSA/vsa-datafree/adapter_model.safetensors`。

该路线只支持普通 T2VA、4 NFE、shift `12 / 3`。真正 VSA 还需要带
`topk_ratio / block_len / coarse_gate`接口的 Comfy Kitchen；缺少内核或50层 learned gate 时会明确回退
Dense 4步，不会冒充稀疏执行。安装与源码构建说明见[`10-speed/README.md`](examples/workflows/10-speed/README.md)。

## OpenVDN MiniMax H3（Advanced EXP）

工作流在[`10-speed/2026-09-03_H3_OpenVDN_DMD8_T2VA_0p5MP_Advanced_EXP.json`](examples/workflows/10-speed/2026-09-03_H3_OpenVDN_DMD8_T2VA_0p5MP_Advanced_EXP.json)。它用ComfyUI原生H3模型承载OpenVDN发布的50层混合注意力分支，保留`chunk=5 / radius=1 / 首尾锚点 / 双向线性分支 / text state / alpha bridge / VDN solve`合同；Windows RTX默认使用精确分组原生SDPA，不依赖FA4、Triton或Diffusers运行时补丁。

把[OpenVDN/vdn-minimax-h3](https://huggingface.co/OpenVDN/vdn-minimax-h3/tree/main)固定revision `18be6bcc4ee72585eee322ba28b5ccac2cf85ef0`的`stage-dmd-step-250`放入`models/diffusion_models/OpenVDN/vdn-minimax-h3/`。DMD默认内部加载default和turbo adapter并固定8 NFE；Stage B内部只加载default adapter并固定50 NFE。不要再叠加EMA_B、Turbo、SLA、VSA、Sol-Attn、BlockCache或其他MODEL/Attention补丁。权重遵循MiniMax H3 Community License，Applicable Territory排除欧盟、英国、韩国和美国；下载或运行前必须阅读模型仓库完整协议。

v1严格只支持普通T2VA；FL2VA、I2VA、L2VA、Ref2VA和任何混合参考都会明确拒绝。当前配套工作流使用结构匹配的本地INT8/ConvRot H3底模并显式开启`allow_structural_base`，报告会保留“未证明等同上游BF16 base”的边界。960×512×73真实DMD8样本已通过分支/adapter完整应用、严格音画解码和本机512MiB余量门，但仍是Advanced EXP，不能据此宣称普遍画质、声音、口型或16GB安全。

## 官方核心兼容

[`20-core-compatibility`](examples/workflows/20-core-compatibility) 提供 AV latent、H3 Attention Hook 和每步 host sync 的按需兼容节点。H3 Audio VAE 会自动关闭旧版按对齐长度裁尾，非整倍数音频不再少掉最后一个 latent step；新版核心和非 H3 VAE 不受影响。Tiled VAE 全局坐标候选仍只保留默认旁路审计。旧工作流不需要修改。

## NVIDIA H3 Super Acceleration

这是两阶段方案，不是 H3 Attention 开关：H3 先跑 4 步草稿，完整 LTX-2.5 Video VAE 编码后经官方 x2 latent upscaler 放大，LTX-2.5 再做 3 步 Refiner，最后才由 TAEHV Wide 快速解码。TAEHV Encode 不得作为 Refiner 输入；H3 音频直接旁路并在最终保存时复用。

目录内同时提供低 Sigma 保脸实验版：`0.5 → 0.412 → 0.350 → 0`，仍是3次 Euler 更新，默认 Dense Attention。它保留更多放大后的原 latent，适合官方完全降噪版导致人脸变化过大时做 A/B；不是官方 Sigma 对齐路线，也不保证所有素材都更好。

配套模型整包：[t8star/Minimax-H3-Super-Acceleration-Comfy](https://huggingface.co/t8star/Minimax-H3-Super-Acceleration-Comfy)。下载时保留仓库中的目录结构，把各目录复制到 `ComfyUI/models` 即可。完整文件名和路径见 [`22-sol-engine-h3-super`](examples/workflows/22-sol-engine-h3-super)。Sol-Attn 是可选依赖，未安装时自动使用 Dense Attention。

## FlashVSR 视频超分

工作流在 [`23-flashvsr`](examples/workflows/23-flashvsr)。下载官方 [FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1)，把整个目录放到 `ComfyUI/models/FlashVSR-v1.1`；模型仓库未包含的 `posi_prompt.pth` 从[官方 FlashVSR 仓库](https://github.com/OpenImagingLab/FlashVSR/tree/main/examples/WanVSR/prompt_tensor)补到同一目录。再从 [SpargeAttn](https://github.com/thu-ml/SpargeAttn) 或其 [Windows wheels](https://github.com/woct0rdho/SpargeAttn/releases)安装与当前 Torch/CUDA 匹配的 `spas_sage_attn`。

先用 `Quality Locked`：固定 `2.0 / 3.0 / 11` LCSA。`Balanced Dynamic` 会改低运动块预算，必须人工看成片；`Memory Safe` 用分块和卸载换显存，通常更慢。官方主要面向4×，项目也提供保守2×实验路线。节点不设模型哈希、文件大小或像素上限，音频原对象直通。

## 社区创作增强

[`21-community-advanced`](examples/workflows/21-community-advanced) 提供 Fun Control、长视频人物音色/句界、接缝漂移审计、低显存策略、Creator语义缓存、TAEH3原生快速预览检查和只读诊断。Fun Control 同时兼容新版官方 `MODEL_PATCH` 合同和旧版 ControlNet：新版模型放`models/model_patches`，旧模型放`models/controlnet`，运行时按能力选择，不按版本号或模型哈希拦截。模型可从 [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/controlnet)取得。从[madebyollin/taehv](https://github.com/madebyollin/taehv)下载`taeh3.safetensors`放入`models/vae_approx`。

需要把Qwen参考前缀缓存与外部[T8 BlockCache](https://github.com/T8mars/comfyui-minimax-h3-blockcache-T8)组合时，使用[`12-system-memory`](examples/workflows/12-system-memory)中的Ref2VA Stock20模板。它是性能优先EXP，不保证bit-exact、省显存或16GB安全。

## 论文能力实验

- [`07-motion-detail`](examples/workflows/07-motion-detail)：RAFT运动审计/MASK传播、轨迹控制、RealBasicVSR时序恢复、H3双时钟AYS校准契约。
- [`04-long-video`](examples/workflows/04-long-video)：FreeNoise视频初始噪声重排，可接普通或Prompt Relay/EAV内循环。
- [`03-image-video-edit`](examples/workflows/03-image-video-edit)：CADS视觉参考退火；只改视觉条件，不改音频条件。

AYS没有可直接套用的H3官方最优时间表；默认仍是原生flow。FreeInit和PAG目前没有可靠的H3联合音视频数学/Attention合同，因此没有做同名伪实现。

## 常见问题

- **8 月 22 日后更新，所有 T8 节点同时爆红/显示缺失：** 这是插件在启动时整体导入失败，不是模型或工作流参数问题。先把 ComfyUI 本体、前端和 Manager 一起更新，完全退出后重启。
- **根据启动终端的第一条报错判断：** 缺少 `comfy_api.latest`、`comfy.weight_adapter`、`comfy.patcher_extension` 或 `comfy.ldm.minimax`，说明 ComfyUI 本体过旧；缺少 `torch`、`torchaudio`、`numpy`、`safetensors` 或 `PIL`（安装包名是 `Pillow`），说明当前 ComfyUI 使用的 Python 基础环境不完整。
- **修复依赖：** 本项目根目录的 `requirements.txt` 为空是正常设计，基础依赖由 ComfyUI 提供。请用启动 ComfyUI 的同一个 Python，重新安装 **ComfyUI 本体**的 `requirements.txt`；整合包用户优先使用整合包更新器。不要在系统 Python 中安装，也不要为了基础节点盲装 SLA、Transformers、OpenCV 等可选依赖，以免替换 Torch/CUDA。反馈时请附第一段完整 `IMPORT FAILED` / `ModuleNotFoundError`、ComfyUI 版本和本节点版本。
- **参数错位或 NaN：** 完整重启 ComfyUI，再重新载入工作流。
- **媒体标签报错：** 检查素材连接和标签编号。
- **Prompt Relay tokenizer 报错：** 优先使用原生 `Load CLIP` 并选择 `type=minimax`；1.52.3起兼容隐藏内部 tokenizer 的CLIP包装，但实际token必须与原生H3逐项一致。
- **没有保留原声：** 使用 `lock_source`，并连接 `mux_audio`。
- **OOM：** 关闭并发，降低分辨率/帧数，再逐个关闭高级节点排查。

## 文档与反馈

- [工作流索引](examples/workflows/README.md)
- [完整使用说明](docs/README_ComfyUI.md)
- [验证结果与限制](docs/VERIFICATION_REPORT.md)
- [问题反馈](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues)
- [第三方项目与许可](THIRD_PARTY_NOTICES.md)

## 相关链接

- [B站](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [API](https://api.seedance.nz/sign-up?aff=5f4w)
- [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd)
- [模型网盘](https://pan.quark.cn/s/c9c267081fbf)
- [Hugging Face](https://huggingface.co/t8star)

模型和第三方组件遵循各自许可证；人物、声音和参考素材的使用权由使用者自行负责。
