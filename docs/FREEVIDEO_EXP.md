# FreeVideo 独立引擎（T8 EXP）

v1.91.0验收状态：基础8+2、音频参考＋外置EAV/Relay的8+2，以及新增真正前4＋后4，三段指定完整5秒音画均获用户通过。实际原生画布保存六张Full/Cold图，旧工作流保留。配方448×256→原learned1.2×实际512×288，完整AV同步Trim120帧/24fps/5秒。8+2保留完成LOW音频；4+4继续实际未完成audio_a4，两类缓存不混用。受影响CPU141项通过/0skip/CUDAfalse；没有追加Cold GPU或重跑已通过片。EAV测量和Relay实际执行不等于任意素材的可见增强或质量保证。[六张公开模板](../examples/workflows/72-freevideo-split/README.md) · [发行说明](RELEASE_1.91.0.md)。

固定来源：[FlashML-org/FreeVideo](https://github.com/FlashML-org/FreeVideo)，代码 `98b3550ae551a70ae3f0a9a463f9dfc9a23b21a7`，VDN `30b6b380c2482f3519469350810c2955d8847fd9`，patched Diffusers tree `37068ab7331d8b28f4cf718dba7015c742a306d2`。模型为 [OpenVDN/vdn-minimax-h3-edge](https://huggingface.co/OpenVDN/vdn-minimax-h3-edge)，本入口使用 Ada/Ampere rowwise revision `ae041e5aec51f8516a3416bd86d579cb29c4c796`。

这是 VDN-H3 DMD8 的 FP8 流式推理引擎，不是另一个普通 H3 UNET。官方 slim 包包含固定八步 AdaLN 表、量化尺度和混合线性注意力，省略原调制投影。不能把它拼成普通 UNET safetensors 再声称无损转换；本入口使用可在 ComfyUI 接线的独立加载器。

## 节点与接线

新增 `FreeVideo Loader`、可串联的 `LoRA`、`LOW`、`HIGH`、`Stage Save`、`Stage Load`，以及独立外置 `EAV` 和 `Prompt Relay 条件`；另有下文四个Split／MID入口，共十二节点。旧节点/默认/工作流不变。8+2效果与纯音频参考合并固定片已获真人确认；视频＋音频参考为结构支持，未另加GPU质量测试。

完整图：原生 H3 Conditioning → LOW（完整8步）→ 标准AV LATENT → 外置学习型3D放大 → HIGH（原八步表末尾2步）→ 原生AV Decode/Trim/保存。HIGH有独立模型、条件和seed。LOW Stage连接HIGH，确认完成态并保留LOW音频。

冷图：加载已保存LOW Stage + 实际manifest SHA → 外置放大 → HIGH。不要连接旧LOW生产节点；填真实缓存，空值/占位不是有效缓存。Save每次新建唯一目录，不覆盖旧缓存。

LOW/HIGH可显式连接 `clip_to_offload`，只释放这个CLIP家族，避免条件编码器占住独立引擎显存；不是卸载所有Comfy模型。外部进程退出释放自己的GPU，取消只终止自己拥有的进程树。

## 环境准备

使用独立 Python 3.12，官方匹配 Torch/VDN/patched Diffusers/Triton。不要升级主 ComfyUI 环境或直接导入官方一体化插件。工具不会安装依赖、覆盖源代码、删除模型或自动下载原投影。

`tools/prepare_freevideo_runtime.py --help` 列出需要的固定源、源码验证副本、独立Python、模型目录和配置overlay。工具逐文件核对官方model catalog与固定源码；`--output-config`只能新建。正常装在 `user/default/T8/freevideo-runtime.json`，或在Loader填写本地配置路径。大模型可留在已有目录只读复用，缺失的小配置放独立overlay。不要将本地绝对路径配置、缓存、权重或私人报告提交GitHub。

准备后执行 `tools/check_freevideo_kernels.py <runtime_config>`，只做实际选定的FP8/weight-only linear与cuDNN两个小检查；结果用于兼容准入，不是完整视频质量或性能认证。Windows实测用 `native-fp8-rowwise/scaled-mm-epilogue`，不宣称不存在的CUTLASS实现。现场资源预算按实际空闲VRAM/RAM选择分块/流式权重，没有隐式降分辨率、换采样数学或OOM自动重试。

内核收据绑定实际Python、Torch/Triton安装记录、GPU UUID/driver与固定计算源码，并核对原始日志SHA。配置/缓存均create-only；换环境或改计算源码后旧probe不自动沿用。仅同一session的已完成旧日志可显式 `--bind-existing`，工具标注身份在probe之后采集，要求计算源未在probe后编辑，不冒称重跑或历史字段预先采集。完成Stage同时核对源pin、producer SHA、AV形状与实际NFE；LOW必须完整8步，HIGH必须1..7整数尾步，不接受只有一个“完成”布尔值的缓存。

## 范围与限制

原入口为官方匹配的完整八步LOW及其1..7步HIGH尾部细化；默认8+2。新增分段入口见下文，不改变原入口。时钟是 Diffusers H3 `t=1-sigma`，不是 Core `1000*sigma`。124帧采样后同步Trim到120帧才是5.000秒。

### 真正分段4+4（EXP路线，指定完整片人审通过）

使用 `分段LOW 前4步` → `保存4步MID`，LOW输出的 `partial_x0_av` → 原外置learned3D → `分段HIGH 后4步`；HIGH还要连接同一MID。它沿用原DMD8表的0..3/4..7索引，不是把完整4步模型再采4步，也不是8+4。两个阶段独立模型、条件、seed与外置EAV/Relay。

MID保存最后一次预测的干净video x0、实际更新后的video_x4和audio_a4、完整原时钟及tensor/源SHA。LOW音频尚未完成，不要把LOW预览当成片；新MID类型无法连接旧completed LOW入口。HIGH将放大的video x0按原index4重噪，而音频直接从a4继续后4次联合更新，不能沿用8+2的冻结完成音频策略。`冷加载4步MID`恢复实际path/SHA即可，不连接LOW生产节点；已有8+2缓存不能冒充MID。

该路线为 `dmd8_split4_lift_x0_continue_audio_exp_v1`，高采视频重新加噪是显式质量变化，不宣称作者严格续采或同尺寸单次8步逐位等价。EAV默认start=.15，原前四步video t均小于该值，因此LOW默认合法无测量；要覆盖前四步统计，须在新分段图显式设置LOW start=0，旧节点默认保持。

独立验收：一个新增实际画布Queue完整跑完前4＋后4，总8NFE，448×256→原learned1.2×实际512×288、完整AV同步Trim120帧/24fps/5秒；MID音频输入SHA精确接续，HIGH输出音频确实更新，不冻结半成品。LOW/HIGH各50层完整覆盖，Relay实际调用；EAV LOW显式start0有200次测量、gain范围1..1.0507023，HIGH有100次测量且gain=1，不是一般增强质量承诺。整图约608秒，不是同条件8+2性能对照。MID真实CPU冷加载通过，Full／Cold两图已原生保存及重开、执行widget与边核对；没有追加Cold GPU。完整受影响CPU141项无skip/CUDAfalse，另四layout的实际pinned tiny CPU方程对照通过；tiny不等于学习权重或画质证明。原325份用户JSON及原八FreeVideo schema保持不变。最终HIGH完成Stage用原Stage Load的`role=HIGH`读取；继续二采用MID Load，不互换。

原生 `[1,L,5120]` H3条件、已应用语义桥、匹配表的首尾帧/视觉参考在结构上支持；参考质量仍需自己的实际片验收。音频/视频音频参考可使用显式派生缓存：`tools/prepare_freevideo_audio_tables.py --help`，再给runtime准备工具传 `--audio-reference-cache`。它只复用已验证权重，并按每个clock/modality从指定官方常量行复制两种新任务的表；CPU逐行字节复核、50块完整命中，不暗中下载约26GB原投影。官方不同任务批次的同clock行可能有BF16舍入差异，因此明确为 `derived_constant_rows_exp_v1`，不能声称重新计算投影逐位等价。已应用语义桥只保留receipt，不重复应用。标准audio轴转换不改变缩放/均值/标准差。

普通 attention/FF/refiner 在线 LoRA逐目标检查A/B、形状、alpha/rank，不忽略未知权重。零强度旁路，不要求该文件存在。暂不支持需要fused重建/调制投影的LoRA，不自动回退或写借用模型。

现有 Core MODEL型 EAV／Prompt Relay 不适用于 FreeVideo自定义模型类型，请使用本家族两个外置节点。每端可以独立选择效果/计划/条件；Relay输出的模型和条件必须成对连接同一采样器，改参考或尺寸需重新编码，不可交换绑定。EAV按全56头统计FETA并只增强目标视频，默认report_only；Relay复用原生Plan/Query Route和精确token绑定，支持native/reference_only音频，不支持Core lock/remix和未实现Hybrid。

Relay保留官方softmax window keyset，在显式cuDNN上加目标query的局部时间偏置；线性分支使用时间加权beta后再按原VDN非线性逆/solve构造文本种子，保留原video特征/alpha/双向扫描/anchor pruning/0.5系数。这是 `beta_weighted_text_seed_exp_v1` 扩展，不冒称论文纯softmax等价。不开效果走原producer；report_only/单事件或全中性路线不应用偏置。apply多事件前需 `tools/check_freevideo_masked_kernel.py <runtime_config>` 的一个带mask tiny，旧无mask检查不替代此检查。EAV统计和Relay偏置/扫描矩阵预算只限定新增显式workspace，不是整个引擎显存上限；硬阈值越界报错，无静默降级/额外NFE。

没有作者2×放大、任意LoRA/后端/参考质量、严格RNG等价或相对T8速度提升承诺；外置1.2×属于兼容路线，不是作者2×配方的严格复现。

实现/CPU检查、内核兼容、完整画布采样、真人质量审核和GitHub/Registry发布是不同状态。没有真人确认不能称质量通过；本地实现不等于已发布。

## 许可

FreeVideo和VDN代码使用Apache-2.0；权重受MiniMax H3 Community License及其使用范围限制。保留模型的LICENSE、NOTICE、MODIFICATIONS与OPENVDN-CODE-LICENSE，模型下载和转换不改变权重授权。不要将代码Apache许可误当成模型可无条件使用的授权。
