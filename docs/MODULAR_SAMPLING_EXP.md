# 分离式采样：当前本地实验入口

v1.90.0 的 [Kijai 九原件入口](KIJAI_EXPERIMENTAL_LORAS_EXP.md) 与 [18张Full_Save／Cold_HIGH](../examples/workflows/71-kijai-experimental-split/README.md) 已追加；旧S01–S29／Veda路线不替换，Acc8保持绝对联合音频。

这是S01–S29独立阶段与外置效果的实验入口。旧一体节点、旧图、原采样公式不替换、不自动迁移。新节点需正常重启ComfyUI后出现；开发工具不会替用户重启服务。节点／存取／画布功能验收不等于人工质量或全部组合认证。

## 1.88.0 当前发行内容与历史说明

当前候选的入口、兼容和限制见[发行说明](RELEASE_1.88.0.md)。下方长期开发记录中的“尚待／tiny／M5待完成”保留为当时阶段，不能据此覆盖后续真实资格；也不能把后续代表样例推广成所有后端／素材通过。

新增多人Source Save／Load已完成双人、三人原8步完整图及独立新进程0采样冷图，完整124帧画音逐值相同；四张opt-in图真实编辑、另存、刷新并侧栏重开通过。详细接线、保护与边界见[多人来源说明](MULTIFACE_FROZEN_SOURCE_EXP.md)。原84份Face图不变。

当前旧575→581审查确认原2098份JSON字节保留，仅12份新图追加；十处实现和八处schema变化逐项精确审查及负例检查，不修改严格M0原始失败报告。外置EAV和Relay的实际调用、阶段冷恢复及原画音保持均分别验证；未知组合保留／委托并记录非可移植边界。

未来新采样准入已经实现：`tools/audit_modular_new_sampling.py`在发行布局冻结86处实际调用及完整导演台构图指纹；原105项研究清单保留，其19项未发布根目录旧副本不混入安装包，实际h3_t8调用点未删减。新的调用需要声明路线、独立阶段／效果与恢复图，额外变化或缺图明确拒绝。该门禁不是自动开发或画质认证。当前271个完整文件4416项回归及另38个导演台完整文件429项通过，无跳过／排除；当前581节点索引和发行布局单独精确复验，重叠计数不相加。实际发行包的Core注册／旧schema／序列化和旧JSON逐字节是额外发布门；主观画质最后集中审核，不恢复用户已取消矩阵。

## 旧工作流默认值与 schema：精确限定审查

多人 Face 的显式来源保存／加载另见 [使用说明](MULTIFACE_FROZEN_SOURCE_EXP.md)。新四张 opt-in 图不会替换旧图；冷恢复使用原保存的计划和编码输入，严格绑定当前完整父片、原音频与独立 Stage，不重新推断 SAM 或放宽来源守卫。

FastH3 V2 的旧保存图不重写，只在测试比对时允许四种已审查的可选控件尾部，
且必须是确切历史控件数和当前 schema 的同类型默认值。原缺省仍是90帧渲染、68帧交付，
Relay `accepted_end_frame=0` 的原时间投影、配方 `continuation_render_frames=90` 的原 payload／SHA 不变。
原控件、连接和其它图数据仍逐值比较；非默认124、额外控件、改边和默认值类型变化均不能被隐藏。
十个完整 CPU 文件范围178项通过，包含实际端口缺省／显式值对照和双模型接缝守卫。

另一轮十四个完整 CPU 文件范围402项通过，检查六份原 schema 快照及旧注册／效果功能。
审查仅允许按原顺序保留所有旧选项、且新增文件真实存在的明确资产菜单，以及五个确切命名的
SemanticBridge／StageSave 描述、tooltip 或允许 `chunk_tokens=0` 的范围扩展；
旧 `alpha=0.1`、`chunk_tokens=256` 和其它默认、端口、属性仍严格匹配。
删除／重排旧选项、未知 `model_name` 归属和其它字段变化拒绝。文件存在不等于权重内容认证。

两轮源码与可选运行时稳定、CUDA未初始化、无跳过或排除；它们有重叠范围，不能相加当独立覆盖数。
原 JSON／schema 快照及原严格 M0／总回归失败证据保留。
这是明确列出的兼容审查，不是自动迁移、原严格 M0 全绿、所有路线 GPU／人审或发布资格。

## Chunked v5：已选历史窗口的限定兼容

显式 Window Load 现在识别两种已精确审查的历史存储实现：`a60d8561…b7d614` 的原完整放大报告身份，以及 `518a7d43…b1048d` 的稳定遥测身份。其余五份采样／效果源码身份必须与当前读取器相同；未知存储版本、缺失／额外模块或其他源码变化仍拒绝。新保存的清单同时绑定兼容读取器源码，不关闭实现身份校验。

原格式仍要求完整报告逐值相同，不能因新增兼容而放宽原内存／模型缓存字段。稳定格式只沿用已有的六项过程遥测排除，不排除放大模型、几何、noise 或 seed。源／计划／已完成窗口、AV 内容、文件 SHA、路径、大小、描述符和 lease 检查均保留；Load 不改写旧档，不重采样，也不声明自动缓存命中或当前 MODEL／条件执行认证。报告明确记录实际历史格式。

两个历史源码由当前代码精确反投影后仍匹配原全文 SHA，真实执行旧 Save → 当前 Load → 新进程仅续剩余窗口。包含原 v5 采样、外置 EAV／Relay、保存工作流及双模型接缝守卫的十三个完整 CPU 文件范围共 195 项通过，无跳过或排除；这是限定内容与冷恢复兼容证据，不是全仓通过、完整 GPU 画质或发布认证。

## HyperFlow continuous：HEAD／TAIL独立节点（实验）

`MiniMaxH3HyperFlowHeadStageEXPT8`只执行绝对0:split，输入独立MODEL、条件和NOISE；
`MiniMaxH3HyperFlowTailStageEXPT8`只执行split:8，可接另一组MODEL／内容LoRA／条件。
两路从同一个完整H3底模分支，各自经专用HyperFlow Loader加载同一原始adapter；不能把
HyperFlow原文件当普通内容LoRA。旧一体`HyperFlowSplitT8Advanced`保留，不自动迁移旧图。

HEAD输出专属continuous_boundary，内部是直接捕获的模型坐标AV `x_sigma`，不是预测x0、
普通LATENT或Progressive的clean_video/audio_next。TAIL直接注入这个状态，**不抽新噪声、
不重置音频时钟、不放大**；其seed是模型执行seed，并非第二份RandomNoise。
TAIL还输出核验完成AV／实际调用／输入身份的completed_result，可接下述专属磁盘Save／Load。
当前保留原连续split的同分辨率、无denoise mask约束，不把它冒充8+4、partial4+4或P7。

`tools/build_modular_hyperflow_workflow.py --output-dir <新的artifacts子目录>`生成1+7、4+4、7+1
三张独立阶段frontend/API候选，位于`artifacts/development/modular-sampling-m3-hyperflow-20260923/candidate-v1/`。
共享几何不依赖TAIL提示词；可在各自HyperFlow Loader之前串联多内容LoRA。两份正条件独立，
CFG1候选复用本阶段正条件作负口；改CFG须提供真实负条件。默认两采执行seed相同。

真实50+2层tiny Core＋synthetic adapter已验证与原split、single8逐位一致，独立内容补丁和HIGH条件；
实际Core缓存验证仅改TAIL提示词／内容补丁／执行seed只跑TAIL，HEAD提示／初始噪声变化两采重跑，
TAIL取消后重试不回跑HEAD。坏边界／错base／adapter／时钟、运行中输入或模型变化不签发完成回执。
未知用户wrapper保留委托执行，但不认证其跨进程复用。已认证的专用HyperFlow owner支持下述存取。
最初三图逐输出Core校验及序列化通过；当批旧420节点接口／顺序和382旧JSON保持，仅追加2节点。
阶段EAV/Relay现有下述continuous专用入口；**尚待**其余HyperFlow三路线、更多后端、完整资产/GPU/浏览器/成片与人审；
新图能导入或tiny通过不等于这些项目已经支持。

### Continuous专属存取与仅二采恢复

新增四节点：`MiniMaxH3HyperFlowHeadSaveEXPT8`、`MiniMaxH3HyperFlowHeadLoadEXPT8`、
`MiniMaxH3HyperFlowTailSaveEXPT8`、`MiniMaxH3HyperFlowTailLoadEXPT8`。
存储位置为 `output/MiniMaxH3/hyperflow_stage_artifacts`；Save返回相对路径和精确SHA，两者都要保留。

- HEAD Save保留原始model-space `x_sigma`和独立Core scaffold；Load直接接TAIL，不能接普通clean-x0放大。
- TAIL Save保留完成AV；Load后只需VAE解码／交付，不加载扩散底模或CLIP，不执行任何采样。
- 每次保存创建新目录，OS锁保护，单个safetensors文件原子提交。不覆盖旧结果，不使用pickle。
- 缺文件、错误SHA、越界路径、partial、类型错配、坏tensor／回执或锁占用会明确报错，不静默重采。
- 重开后仅TAIL可换独立提示和内容LoRA；原始底模、原HyperFlow adapter及双时间执行合同必须匹配。
  不用文件名、随机owner或Python对象id充当持久身份；实际52层闭包、delegate、端点权重／FP32缓存、
  原始权重和所选补丁内容分别校验。未知用户补丁仍可执行／归档，但不伪造可移植恢复证明。
- 显式选择旧HEAD意味着冻结该已完成一采；不是证明今天修改后的一采参数与旧文件相同。
  旧HEAD节点的process-local说明属于初版历史文字；为保持已有完整schema，本批未改旧节点描述。

`tools/build_modular_hyperflow_storage_workflow.py --output-dir <新的artifacts子目录>`
按1+7／4+4／7+1各生成 `save_head`、`save_both`、`resume_tail`、`load_completed`，共12张frontend/API图。
本地候选在 `artifacts/development/modular-sampling-m3-hyperflow-storage-20260923/candidate-v1/`。
恢复图已真实删除一采输出／随机噪声分支；完成AV图已删除底模／CLIP／采样分支。
Load中的路径和SHA是占位值，必须换成实际Save结果后才能执行，不自动查找或猜测旧文件。

六组实际新进程tiny CPU对照证明TAIL-only输出与同进程逐位一致，包括独立二采内容LoRA／提示；
真实Core文件指纹测试证明文件被替换后不会复用原Load缓存或偷偷重采。FP32/FP16/BF16 adapter
原始dtype与运行时FP32端点缓存均有tiny测试。不是完整预训练模型／GPU或音画质量认证。
最新批次测试与剩余项见上述目录CHECKPOINT；本次只新增四存取节点，旧422 schema和382JSON不变。

### Continuous外置EAV／Prompt Relay

新增 `MiniMaxH3HyperFlowHeadEffectsBindEXPT8`、`MiniMaxH3HyperFlowTailEffectsBindEXPT8`，
以及对应 `HeadEffectsAudit`／`TailEffectsAudit` 节点（完整ID同样以 `MiniMaxH3HyperFlow` 开头、`EXPT8` 结尾）。
旧采样节点描述中的效果pending是初版历史文字；为保持旧schema不修改，该项以本节说明为准。

```text
HEAD HyperFlow MODEL → 可选Relay Conditioning → HEAD Effects Bind → 可选Stage EAV Apply → HEAD Only
TAIL HyperFlow MODEL → 可选Relay Conditioning → TAIL Effects Bind → 可选Stage EAV Apply → TAIL Only
HEAD完成边界 → HEAD Audit → 可选HEAD Save/Load → TAIL Bind及TAIL Only
TAIL完成结果 → TAIL Audit → 可选TAIL Save → AV Decode
```

- 两路可以使用不同内容LoRA、提示、Relay Plan和EAV Config，也可只启用一采或二采。
  原底模与原HyperFlow adapter仍须一致。Relay Conditioning的MODEL和positive必须成对接入Bind。
- EAV复用现有 `MiniMaxH3StageEAVConfigEXPT8` 和 `MiniMaxH3StageEAVApplyEXPT8`。
  Bind的MODEL、SIGMAS、stage_template、stage_context分别接Apply；Apply的MODEL接对应采样器。
  HEAD Bind与HEAD采样器使用相同split；TAIL Bind从真实continuous_boundary取得区间。
  stage_template只是几何/scaffold，不是raw x_sigma，不得代替TAIL采样器的continuous_boundary。
- EAV默认report_only，不施加增益；apply_exp才应用，disabled为旁路。窗口是绝对 `1-video_sigma`，
  不是在二采重新从0计时。增益超过所选hard limit仍报错，report_only不取消原资源/数值保护。
  原生Relay的disabled／report-only／零或单事件旁路语义不变。活跃阶段效果当前要求CFG1。
- Audit读取该次完成结果中的不可变调用记录。它不从旧Apply节点的可变计数猜测采样是否成功。
  未知用户wrapper/后端保留执行；若实际绕过效果，报告unverified且不认证持久复用，不偷偷换dense。
- 已识别的效果可随HEAD保存，重开后只执行独立TAIL；原生Relay两路可用不同路由，TAIL EAV也可重新配置。
  新效果不会自动启用在旧图上。已保存的HEAD代表显式冻结历史一采，不证明修改后的HEAD参数仍等价。

`tools/build_modular_hyperflow_effect_workflow.py --output-dir <新的artifacts子目录>`生成81张frontend/API图：
1+7／4+4／7+1 × EAV／Relay／组合 × 仅HEAD／仅TAIL／两者 × minimal／save／resume_tail。
本地候选位于 `artifacts/development/modular-sampling-m3-hyperflow-effects-20260923/candidate-v1/`。
恢复图真实移除HEAD模型、编码、噪声、效果和采样输出链；Load需填实际Save路径与SHA，不能执行占位符。

已覆盖实际50+2层tiny/synthetic adapter的关闭保真、真实效果调用、两路不同配置、取消/缓存、未知补丁，
以及带效果HEAD跨进程仅TAIL逐位对照。最新结果与失败历史见该目录CHECKPOINT。
此资格不代替预训练GPU、浏览器保存重载、完整音画或人审；S14/S15/S16不能用本适配冒充完成。

## HyperFlow full8＋fresh4／partial4＋fresh4（实验）

新增五个独立节点，ID均为 `MiniMaxH3HyperFlowFresh`＋以下名称＋`EXPT8`：
`Loader`、`StageSetup`、`LiftInput`、`StageAudit`、`StageLoad`。
它们不替换旧HyperFlow Loader／Sampler，也不是continuous或P7长片节点。

```text
LOW底模／内容LoRA → Fresh Loader → 可选Relay Conditioning → Stage Setup → 可选Stage EAV → Stage Sampler
LOW Stage Result → Audit → Lift Input → 原learned3D放大器 → HIGH的latent_image
HIGH底模／内容LoRA → Fresh Loader → 独立Relay／Stage Setup／EAV → 独立Guider与Stage Sampler → Decode
```

- Stage Setup准备原SAMPLER／SIGMAS和阶段描述，不执行采样。实际阶段仍经标准Core执行一次；
  Stage Sampler只是额外给出可审计／保存的Stage Result，前两个输出保持Core原值。
- `hyperflow_low_full8`执行完整0:8；Lift Input选择完成的`output`。`hyperflow_low_partial4`
  执行0:4；Lift Input选择真正的`denoised_output`预测x0，不把非终点output当完成图。
- 两种HIGH分别为`hyperflow_high_after_full8`和`hyperflow_high_after_partial4`，均保留原4:8、
  外部独立NOISE及原音频start rebase。前者总12次求值，后者总8次；不是continuous无噪续采。
- 新Fresh Loader使用Core原权重备份建立端点快照，防止后创建分支把前一阶段驻留的LoRA当原底模。
  只改新MODEL自身的待安装描述，不卸载共享网络、不删除内容LoRA、不改旧Loader。两路可以独立
  选择模型／内容LoRA；不像continuous，不额外强加两阶段底模必须相同的限制。
- EAV Config／Apply和原生Relay Plan／Conditioning外置，可仅LOW、仅HIGH或分别配置两者。
  候选EAV默认report_only，实际增益需选apply_exp；绝对时钟不在HIGH归零。活跃效果要求CFG1。
  Audit读取完成回执的实际调用，而非旧Apply UI计数；未知用户补丁保留运行但可能无法认证持久复用。
- 保留原fresh路线learned AV直接传HIGH的音频路径，不额外插入旧Dual音频锁定策略。
  HIGH条件尺寸连接放大器实际输出。更多参考／录音组合仍须各自验证，不能由T2VA候选推定全支持。
- 保存复用已有Stage Save，位置为`output/MiniMaxH3/stage_artifacts`。Fresh Stage Load需要实际path、
  SHA和确切阶段；旧通用Load的下拉列表不变。冻结LOW可经Lift Input重新放大并只采HIGH；
  冻结HIGH可以直接解码，不重新编码或采样。显式冻结结果不等于自动匹配当前LOW参数。

`tools/build_modular_hyperflow_fresh_workflow.py --output-dir <新的artifacts子目录>`生成62组frontend/API候选：
两种路线×（无效果、EAV／Relay／组合各三种作用范围）×minimal／save／resume_high，再加两张load_high。
resume_high图真实裁掉LOW模型、条件、噪声、采样、效果和输出节点，Load占位符必须替换。
当前可导入候选在`artifacts/development/modular-sampling-m3-hyperflow-fresh-20260923/candidate-v2/`。
该目录的`CHECKPOINT.md`、`scoped-summary-v2.json`与两份兼容记录是本批证据；早期candidate-v1已过时。

阶段数值和效果测试用真实50＋2层tiny Core及synthetic adapter；跨尺寸测试调用原learned外层流程，
网络和编码器明确使用CPU替身。不得把这些证据写成完整预训练learned3D／GPU／浏览器重载／音画人审通过。
受影响旧路线52模块联合回归1690通过、0失败、1项可选真实权重跳过，178份冻结文件SHA保持。
候选图和兼容记录以检查点为准；完整预训练、GPU成片、浏览器重载与人审仍待，整个M0–M5计划仍未完成。

## Progressive：公开分离节点与专属阶段存取（实验）

当前已追加10个阶段/存取节点和5个外置效果节点，旧整体Progressive／Avatar入口不替换。主接线为：

```text
中性目标AV＋完整SIGMAS → Plan/LOW Source → LOW单阶段条件 → LOW Sampler
LOW Boundary → Lift Input → 原learned 3D放大 → HIGH Handoff → HIGH单阶段条件 → HIGH Sampler
```

LOW模型／LoRA／条件／实际联合噪声与HIGH模型／LoRA／条件／视频噪声分别可编辑。
Plan的目标AV不要接HIGH专属提示词的条件节点输出，否则图依赖会让HIGH提示词改变时也重算LOW。
候选图使用Core中性Empty AV作为共同几何；HIGH条件尺寸接实际放大器输出。
LOW和HIGH各只调用一次原生阶段执行器，不回调旧整体两阶段入口。LOW保存模型坐标下的
`clean_video`、按原Euler公式推进的 `audio_next`，以及HIGH音频clean anchor所需的原LOW噪声。
视频先还原到VAE坐标交给图外原learned放大器，再按原噪声缩放恢复HIGH；音频继续演化，
不能把LOW的预测音频当成完成音频锁死，也不能把这个边界直接冒充普通AV LATENT。

HIGH可独立替换MODEL／LoRA／条件／视频噪声；可选择新的剩余SIGMAS，但必须从冻结LOW的
resume sigma开始，并保留原生坐标与时间网格。HIGH原始clean source和mask与噪声restart分离；
保留原生已知区域及Avatar音频mask=0的计算语义，不采用采样后粘贴音频的替代方法。
资源reserve仍默认1024MiB、最低512MiB，开始及每个callback检查；错误原样传出，无自动重试。

独立 `progressive_storage.py` 仅保存有完整LOW调用证据的专属类型，使用唯一目录、OS lease、
无pickle safetensors和精确SHA，原子改名是唯一完成点。读取不执行LOW，不宣称当前LOW配置
仍与所选冻结结果相同。未知用户wrapper仍可运行和归档，但未适配的执行身份不能认证跨运行复用。
原普通Stage Save／Load和旧Progressive整体checkpoint均不改。

HIGH Sampler还输出专属 `high_result`，可接“Save Completed HIGH AV”。独立HIGH Load返回
完成AV和回执，不执行MODEL加载／LOW／放大／HIGH，可直接送解码或后续处理；额外LATENT
metadata一并保存。LOW与HIGH是不同文件类型，不能混接；保存结果未知身份可归档但不认证
跨运行复用。两种Load均必须填真实artifact_path与artifact_sha256，不是自动请求缓存。

`tools/build_modular_progressive_workflow.py --output-dir <新的artifacts子目录>`
构建T2VA／首帧I2VA各四图：普通分离、保存阶段、仅恢复HIGH、仅加载完成HIGH，共8图。
本地候选与逐输出校验记录在
`artifacts/development/modular-sampling-m3-progressive-public-20260923/candidate-v2/`。
所有frontend／API边和输出分支均通过真实Core校验；不是浏览器重载或完整模型成片证据。
默认原生Stock20分成10＋10，不添加Turbo/EMA加速LoRA；可独立插入内容LoRA。
HIGH的原生Setup只提供MODEL坐标和Euler采样器，完整steps输出未接HIGH剩余表。
要改变HIGH步数，显式向Handoff的可选high_sigmas口接新的剩余表，起点/dtype必须匹配冻结边界。
默认HIGH视频噪声seed为LOW seed＋1，而HIGH采样器seed与LOW相同；修改LOW seed后按需同步。
I2VA候选引用已有本地图片，换机器须自行选择真实首帧。CFG1候选复用本阶段正条件作为负输入；
要使用其它CFG需自行提供正确负条件，并单独验证。旧工作流不自动升级。

CPU验证包含真实Core tiny T2VA／I2VA、原noise_scale／选定sampling对象、初始化二值／小数／
零遮罩、Avatar型音频锚、真实LoRA、取消／重试／资源失败，以及四个新进程只恢复HIGH。
放大环节使用明确标注的插值替身，**没有运行真实learned权重或完整预训练GPU成片**。
新增实际Core执行器/cache测试确认HIGH专属提示词、模型补丁、噪声修改只执行HIGH，LOW提示词
修改两段执行；相同图零采样调用，HIGH取消重试不回跑LOW。这只是在缓存仍保留的本进程内；
跨重启请使用显式Save/Load。新进程完成HIGH加载亦经检查，不加载模型或调用采样/放大。
阶段外置EAV／Relay的执行已接入，效果持久身份按下节限定组合适配；Avatar与accepted-parent continuation
各有下述专属入口，长片可编辑body接入仍待完成；不将内部mask测试写成完整Avatar／续段资格。完整预训练／learned权重、实际GPU、
浏览器编辑保存重载、成片音画和人审仍未验收。

### Progressive逐阶段外置EAV／Prompt Relay

每一采都可以独立使用以下接线，也可以只启用其中一采：

```text
独立Relay Plan → 原Relay Conditioning（本采MODEL、目标尺寸）
             → Progressive Relay Stage Apply（LOW/HIGH） → 本采MODEL/positive/negative
独立Stage EAV Config → Progressive LOW EAV Apply 或 HIGH EAV Apply → 本采MODEL
本采完成结果 → LOW/HIGH Effects Audit → 实际调用报告
```

Relay Stage Apply本身已经准备本阶段guide与实际packed layout，应替代普通Stage Conditioning，
不要重复串联缩放。LOW与HIGH不要求同一个Relay Plan，HIGH也不要求LOW启用Relay。
原Relay编码器的零/单事件或report_only输出是合法旁路：不安装效果、不清用户补丁，仍准备
本阶段条件；残留的错配MODEL/CONDITIONING标记不是旁路，会正常报错。
需要同时使用两种效果时先Relay Stage Apply，再专属EAV Apply；既有通用Stage EAV Apply
不接受ProgressivePlan。HIGH EAV读取typed restart的原clean-source mask与实际HIGH剩余表。
所有效果维持完整原生表的`1-video_sigma`绝对时钟，不按局部step从0重计。此适配为CFG1。

Apply不执行模型；专用单阶段采样器开始时重置本次计数，结束时把实际Relay/EAV报告写进
该结果的不可变回执。取消/资源失败释放本阶段锁并标记aborted，再次执行不累计旧计数；
Audit读取完成结果，不读取可能来自另一轮的可变Apply计数。未知补丁保留，绕过效果时报告
unverified。EAV默认report_only是数值旁路，明确选apply_exp才应用增益；disabled保留原MODEL。
FETA不直接缩放音频行，但联合模型传播和Relay音频query可影响最终音频，不能承诺音轨不变。

`tools/build_modular_progressive_effect_workflow.py --output-dir <新的artifacts子目录>`
提供T2VA/I2VA × EAV/Relay/组合 × 仅LOW/仅HIGH/两采，共18图。
本地候选在`artifacts/development/modular-sampling-m3-progressive-effects-20260923/candidate-v1/`。
真实Core tiny已检查独立阶段、原算法逐位对照、真实缓存依赖及取消重试。原生/KJ/Sol/未知
delegate保留；CPU Sol委托证据不冒称CUDA Sol内核验证。Core正常清理产生的原生bound-forward
别名仅在新模块的执行身份检查中规范化；替换类方法、未知forward、hook、权重变化仍失配。

原生自动注意力及已认证的普通Core委托现在支持Relay、EAV、两者组合的显式阶段存取。
身份适配核验真正的factory wrapper/selector/observer、MODEL–条件绑定、事件布局、阶段
Plan、原生mask实际值和执行中的类方法；仅在检查用克隆中投影，不卸载或清除用户补丁。
遥测计数不影响身份，改真实执行方法或配置仍使身份失配。原CONDConstant按精确Core类型
及实际cond内容绑定，未知条件对象不伪装成可移植数据。

上述工具加 `--variant save`、`--variant resume_high` 或 `--variant load_high`，各产生18张
新候选。保存图串联本阶段Audit→Save；恢复图真正删除LOW模型/编码/效果/采样及预览支路；
完成HIGH读取图没有模型/放大/采样，仅读取、审计、解码和交付。须填写真实path和SHA。
当前候选存放于 `artifacts/development/modular-sampling-m3-progressive-effect-identity-20260923/`。
tiny CPU已验证新Python进程读取带效果LOW后仅执行HIGH，以及独立更改HIGH权重补丁，
输出与同进程逐位相同；完成HIGH可直接读取。原无效果阶段Save/Load合同不变。

**当前限制**：KJ/Sol/未知委托和Sage延迟创建的有偏置内核仍需各自的持久身份适配；可执行
和归档不等于允许认证恢复。未识别组合不会被清除、替换或因“未认证”而禁止新采样，
Load仍拒绝其未认证记录。效果owner可识别也不保证整套用户MODEL可移植，以结果总资格为准。
GPU/learned资产、浏览器、完整音画和人审仍须单独完成；旧节点提示中的较保守资格文字
不构成更多后端保证，本批没有改变旧节点接口或默认值。

## Avatar：原录音驱动的独立阶段（实验）

追加四个独立节点，不替换旧Avatar整体入口或普通Progressive节点：

- `MiniMaxH3AvatarSourceBindEXPT8`：连接编码AV与原录音选区，校验显式双路mask、音频mask全0、有限PCM及采样率。记录所选编码AV／PCM对的内容身份，不假称证明了任意手接LATENT的VAE来源；不编码、不采样。
- `MiniMaxH3AvatarLowStageEXPT8`：只执行原生LOW。输入必须与绑定源按原initialized规则准备的LOW一致，完成回执附加录音／源绑定。复用通用LOW Save／Load和外置learned3D，不执行隐藏HIGH。
- `MiniMaxH3AvatarHighHandoffEXPT8`：校验冻结LOW属于同一录音／编码源，保留原clean audio与全0音频mask。默认用绑定源；可另接HIGH视频源／mask，但不能悄悄换录音锚点。其输出接普通Progressive HIGH Sampler及外置HIGH EAV／Relay。
- `MiniMaxH3AvatarDeliveryAuditEXPT8`：接受完成HIGH及同一原录音选区，核验实际HIGH的clean anchor／mask／LOW绑定，报告输出音频latent与原锚点差异。原样返回PCM用于显式最终mux；不会事后把PCM粘回采样latent伪造数值一致。

建议接线：中性Empty AV → 独立Audio Latent Control（lock）→ Source Bind →
Progressive Plan（initialized_av_exp）→ Avatar LOW → Lift Input → 外置learned3D →
Avatar HIGH Handoff → 独立HIGH效果／条件 → Progressive HIGH → Avatar Delivery Audit。
共同源不依赖HIGH提示词／模型；两采的LoRA、条件、EAV与Relay分别可接。
已有录音不再作为音色参考重复注入；候选条件编码用native，录音锁定由独立源节点承担。

`tools/build_modular_avatar_workflow.py --output-dir <新的artifacts子目录>`
生成T2VA／I2VA的最小图及三种效果×三个阶段作用域，每种包含minimal／save／resume_high／load_high，
共80张frontend/API候选。当前路径是`artifacts/development/modular-sampling-m3-avatar-20260923/candidate-v1/`。
默认原生8步拆4＋4，沿用原learned放大器；不是FastH3 V2配方。
AudioWindow显式选择73帧对应原录音区间；换素材需重新选择实际录音／首帧。
最终只解码视频latent，CreateVideo的audio明确连接Audit返回的原录音选区，不连接生成音频。

保存LOW可在新进程重建同一编码源／录音后只采HIGH；更换编码器导致源字节不同不能冒认原冻结来源。
完成HIGH另外保存用于核验的原音频latent锚点，直接加载时只接回同一录音，不需要音频VAE、
source编码、LOW、learned放大或HIGH采样。仍须真实artifact_path及SHA，不是自动缓存。
原生已认证效果组合可持久恢复；未知用户补丁保留执行／归档，但不假授予跨进程资格。

当前tiny CPU验证包含T2VA／I2VA、独立HIGH权重、原Avatar逐位数值对照、实际逐阶段EAV／Relay，
坏源／mask／录音与运行中修改拒绝、取消恢复、四次新进程仅HIGH和完成HIGH只读交付。
实际Core缓存验证HIGH专属提示词／模型补丁／噪声仅HIGH、LOW条件改变两段重算、HIGH取消重试不回跑LOW。
80图每个输出分支和frontend序列化检查通过。编码／放大是明确测试替身，未运行真实learned权重或
完整预训练GPU；不证明录音身份、口型质量、长片接缝或浏览器编辑重载已验收。

## Continuation：已接受前段的独立续采（实验）

新增12节点，不替换旧Progressive continuation／长视频Loop。这里的续段不是把普通Empty AV
或任意LATENT当成前段：必须选择已有链的直接accepted parent、准确candidate、revision和
previous_job_sha256，核验实际MP4、完成AV context及其SHA。previous_job_sha256只证明
所选旧来源，不证明今天修改的提示词／MODEL仍属于旧job；可编辑body的持久交付另属M5。

```text
Select Accepted Parent → Prepare Accepted Contexts → Geometry + Stage Plan
                       ├→ LOW ONE Phase Conditions → LOW Sampler
                       └→ HIGH ONE Phase Conditions ───────────────┐
LOW Boundary → Progressive Lift Input → 外置原learned3D → HIGH Handoff → HIGH Sampler
完成HIGH + 同一Accepted Parent → Deliver Completed Window → 解码／交付
```

共同Contexts不依赖HIGH提示词／MODEL。它只给LOW使用真实accepted MP4最后39帧RGB24、
原resize、当前视频VAE的结果；HIGH已完成AV和两采共享audio对象不改，22/39时间坐标不改。
LOW/HIGH各自编码提示词和参考，可独立接MODEL／LoRA／noise／条件和Semantic Bridge。
只在HIGH锁原已知视频前缀；LOW不能用缩小的HIGH prefix代替accepted-picture motion guides。
两个专属Sampler各只执行标明阶段，固定原生CFG1，不转发旧整体runner。
Plan保留原生Euler的clean_video/audio_next边界与原HIGH重噪，不能当FastH3／HyperFlow配方。

外置效果分别接到相应阶段：

- `ContinuationRelayProjectEXPT8`按直接前段绝对时间投影外部全局Plan；compiled_prompt先接本采Conditions。
- `ContinuationRelayApplyEXPT8`绑定该投影、本采实际编码条件和motion layout，返回成对MODEL/positive/negative。
  LOW与HIGH可以使用不同全局Plan，不要求另一采启用Relay。joint_av_exp与锁定音频的真实冲突仍报错。
- 外部Stage EAV Config接`ContinuationLowEAVApplyEXPT8`或`ContinuationHighEAVApplyEXPT8`，
  HIGH读取本采typed restart的原clean-source mask；有Relay时先Relay后EAV。绝对1-sigma时钟不重置。
- 完成结果接现有Progressive LOW/HIGH Effects Audit，读取本次实际调用，不以Apply接线代替生效证据。

以上节点实际ID均以`MiniMaxH3`开头；分类为Modular Sampling/Continuation Experimental。
未知用户wrapper仍委托执行，不清LoRA、mask或注意力补丁；未适配身份不能伪造可移植记录。
原生motion owner及已识别原生EAV/Relay组合支持已有Progressive LOW/HIGH Save/Load。
新进程加载LOW后仅采独立HIGH；只加载完成HIGH时不再做Contexts VAE／条件编码／采样，
但仍通过Source重新校验accepted parent。父段revision或媒体变化会失效，不能拿旧内存缓存遮住。
Delivery不自动写Candidate、Accept或Compose，也不更新链时间线。

`tools/build_modular_continuation_workflow.py --output-dir <新的artifacts子目录>`生成80张候选：
22/39 × 普通或EAV/Relay/组合的仅LOW/仅HIGH/两采 × minimal/save/resume_high/load_high。
本地在`artifacts/development/modular-sampling-m3-continuation-20260923/candidate-v2/`。
默认原生8拆4+4和外置learned3D；parent、保存path/SHA必须自行填真实值。默认native音频图接
解码后的生成音频；若明确选择final_audio，应把Delivery的mux_audio接CreateVideo，不能把
None直接当有效PCM。Delivery原样保留该明确选择，不剪切／增益／回贴latent伪造一致。

CPU检查包含旧native续段4+4逐位对照、真实阶段效果、独立HIGH模型/Relay、跨进程仅HIGH、
父段变动/链锁/错源/错motion grid拒绝、取消重试和实际Core缓存依赖；80图逐输出校验通过。
VAE、文本编码、learned环节仍是明确测试替身，不是完整资产/GPU/成片接缝/浏览器/人审资格。
兼容核验保留旧408接口/注册顺序及382旧JSON，本批仅追加12节点。共享mask runtime新增可选
accepted_source入口，默认数学未变，但其文件SHA可能让旧源码glob阶段缓存安全失配；
不删除旧成片/accepted/回执、不假命中或静默重采，接口兼容不代表所有旧缓存还能命中。

## VDN：完整首采与独立后段

`MiniMaxH3VDNStageSetupEXPT8` 接原 OpenVDN Composer 的 MODEL 和本阶段 AV latent，
选择 `vdn_complete` 或 `vdn_refine`，输出标准 MODEL／SAMPLER／SIGMAS 和 stage_context，
不加载权重、不采样。DMD8／B50首采完整执行原轨迹；VDN后段使用自身原表尾部及fresh noise，
不是LBH表或隐藏4+4。二采还要接完成的 first_pass_latent，保留原几何、有限值和非缩小检查。
LOW和HIGH模型／内容LoRA／条件／噪声独立；放大与Reconcile在图上可编辑。

两种路线分别提供构图工具，均只写新的私有artifacts候选目录：

- `tools/build_modular_vdn_workflow.py`：DMD8／B50→自身VDN尾段，各最小／外置EAV／保存／只恢复HIGH，共8图。
- `tools/build_modular_vdn_native_workflow.py`：DMD8／B50完整首采→独立干净原生MODEL＋EMA B＋LBH HIGH3／4／5，共24图。HIGH不继承VDN branch、DiT或layout；LOW不使用LBH coarse输出。带效果版本还外露**仅native HIGH**的Relay Plan。

VDN完整首采使用 **output槽0** 交给原learned 3D放大，不能照搬partial路线的denoised槽1。
默认原Reconcile保留已完成首采音频，解码前仍接原AudioAudit检查浮点误差并重锁；仅有零音频mask
不能宣称采样出口逐位不变。保存后的恢复图真正删除LOW模型／条件／采样／效果／输出链，
填写冻结LOW path／SHA，只执行新的HIGH。不是自动请求缓存或长视频恢复。

VDN EAV通过独立适配接入原window softmax与linear两路输出投影之前，保留原窗口、双anchor、
gate、linear recurrence和原SDPA后端。使用有界额外QKV统计，不增加扩散步、不换Dense。
`disabled`原对象旁路；`report_only`逐位不变；`apply_exp`才改变结果。
原Composer additional branch的实际权重、补丁、模块与hook身份纳入只读检查；认证的普通加载路径
可保存/读取，未知可执行组合保留新运行而不伪授予跨运行身份。low-VRAM side-model身份仍待适配。

### VDN专属外置Prompt Relay（实验适配）

原grouped SDPA和linear全局文本状态绕过普通selector，必须显式增加
`MiniMaxH3VDNRelayApplyEXPT8`，不能只接普通Relay配置就声称生效。每阶段的顺序为：

```text
独立Relay Plan → Relay Conditioning（本阶段实际尺寸、成对MODEL/positive/latent）
               → VDN Stage Setup → VDN Relay Apply → 可选Stage EAV Apply
               → BasicGuider / 单阶段Sampler → 可选保存/EAV Audit → VDN Relay Audit
```

Apply的`disabled`返回输入MODEL原对象；`apply_exp`启用真正的VDN执行路径适配。
两阶段各一份节点，可分别启用/关闭，也能只给某一采接EAV，不依赖一体Loop。
`MiniMaxH3VDNRelayAuditEXPT8`报告实际window调用、linear帧/头分块/seed求解次数、
预期和实际阶段block数、中性旁路、工作区估计与异常；未知producer绕过不冒称完成。

窗口分支保留原窗口key集合、global行和双anchor，只对目标video/可选audio query到事件text key
加入原Relay时间惩罚。linear分支没有softmax logits，因此采用明确命名的
`beta_weighted_text_seed_exp_v1`：按每个query帧的`exp(-penalty)`缩放事件text beta，
重新计算原非线性文本seed，通过不变的视频扫描传递。全局text不被事件权重覆盖；
不是把分事件seed相加，也不是偷偷换成全Dense。它是**VDN实验扩展，不是paper softmax
数学等价或预训练画质认证**。中性权重直接走原函数逐位旁路；启用时的分解扫描用FP32容差
与逐帧原扫描参考对照，不能声称所有浮点重排逐位相同。

默认64MiB预算约束额外bias并估计head-chunk的scan/factor工作张量；不包括原模型特征、
输出、内核allocator工作区或总显存。预算不足正常报错；SDPA保留原VDN安全后端上下文，
选定GPU内核不支持mask时正常报错，不静默改后端。无额外扩散NFE，但新增文本求解有计算成本。
已识别实际hook、binding、config与阶段身份支持显式保存/读取；未知组合保留运行而不伪认证。

`tools/build_modular_vdn_relay_workflow.py`生成32张新候选：DMD8/B50×VDN自身尾段或
native HIGH3/4/5×Relay-only、Relay＋EAV、阶段保存、只恢复HIGH。两阶段Plan独立；
VDN阶段实际接专属Apply/Audit，native HIGH保持原普通Relay，不能相互冒认覆盖。
恢复图删除LOW Plan/模型/条件/采样/效果/输出全部链，只读冻结完成LOW output槽0。
原不含Relay的8图/24图仍保留，旧正式工作流不覆盖。

新增证据是小张量窗口参考、逐帧原linear扫描参考、真实tiny Core长/短VDN、
EAV组合、冷/热/重建、主/side旧补丁门禁及新进程只恢复带Relay/EAV的HIGH。
完整权重、GPU内核、实际媒体质量/人审仍未认证。原生side-model低显存身份见下节，
不能把CPU强制内存路径的测试称为真实GPU显存资格。

当前证据为真实tiny Core／原VDN模块的DMD8、B50、短片full-cover和长片window+linear、
原两输出对照、冷／热／重建身份、用户LoRA及音频mask、阶段保存。原VDN阶段批次有十个新Python
进程只执行VDN HIGH或native HIGH；后续Relay批次另有四个VDN、两个native恢复进程。
前者同尺寸，后者使用明确的合成4×4→4×8交接，均不代表真实learned权重。
候选图通过Core／序列化检查；完整预训练权重、GPU／浏览器编辑重载／音画人审仍未认证。

### VDN侧模型低显存加载与实际补丁路径

阶段身份现可只读识别原Core `ModelPatcher` 的正常、低显存、预准备补丁、卸载和clone状态。
只接受原 `LowVramPatch` 的确切目标、补丁内容/顺序及预准备副本，保留原权重backup；
不会为生成身份卸载模型、删除用户LoRA、替换采样器或清除外部回调。
未知子类/回调/转换函数不因名称相同得到可移植身份，仍遵循未知组合可执行但不假认证的策略。

**逻辑模型身份相同不等于所有加载方式逐位相同。** BF16存储的补丁在加载时合并会先按存储精度
舍入，低显存延迟补丁可能按计算精度应用。新阶段回执的`request.vdn_weight_execution`记录
每次denoiser调用实际使用的`core_materialized_storage_dtype`或`core_delayed_compute_dtype`、
目标key/存储dtype及调用索引。不同路径不会合并成同一个请求SHA；同路径冷/热/重建可核对。
显式读取LOW仍读已选完成张量和SHA，不隐式重新采样；自动请求缓存尚未启用。

验证使用原Core加载/补丁/前向，只在测试中强制side模型的低内存预算；主干为tiny FP32，
side包含FP32/BF16、diff/LoRA及bias补丁。不是整模型BF16或CUDA异步搬运/真实VRAM峰值认证，
也不是动态patcher、任意量化side模型或完整预训练媒体的通用保证。

## PDD：原32头、绝对LOW0:4／HIGH4:8

`MiniMaxH3PDDStageSetupEXPT8` 接已有 PDD 8-Step Setup 的 MODEL、**完整九点SIGMAS**、
该阶段实际 AV latent，选择 `pdd_low_0_4` 或 `pdd_high_4_8`。不重新加载PDD权重、不采样，
保留原native head-bank或legacy dynamic injection。两阶段可各用独立PDD MODEL及内容LoRA。

LOW沿用旧Loader的Euler/simple；HIGH沿用旧双采图的dual-clock Euler/native_flow几何设置。
HIGH重新生成的全表不使用，两阶段均取连接进来的原九点表及dtype，绝对头组为0–3／4–7，
不把HIGH重置到头0，也不是8+8。LOW denoised_output接原learned3D，重建HIGH尺寸条件后
通过原Reconcile的`auto/legacy_policy`继续联合音频，HIGH另接fresh NOISE；不要锁死未完成LOW音频。

EAV Config／Apply／Audit可分别配置，Relay需按各阶段实际尺寸成对绑定MODEL与CONDITIONING。
原生head-bank及可认证的原legacy动态注入都有阶段结果身份、保存／显式读取适配。
动态身份检查原backbone LoRA、输出头、strength、真实闭包与注入生命周期；只投影检查用克隆，
不卸载或改写运行中的MODEL。未知第三方补丁保留执行，但不能拿未识别运行状态认证恢复。
原动态输出头补兼容当前Core追加的sigma/sample_sigmas/shifts参数；选头仍由原绝对时钟wrapper负责，
不改变旧四参数调用、强度混合或音视频计算。

`tools/build_modular_pdd_workflow.py --output-dir <新私有artifacts目录>` 生成FL2VA／Ref2VA
各最小／外置效果／保存／仅HIGH恢复共8张候选。两路模型、条件、噪声在画布上可编辑；
恢复图实际移除LOW模型／采样／输出链，需填写已保存LOW的path和SHA。示例`10A.jpg`请替换；
FL2VA首尾图先统一crop再送两阶段。旧图原字节不改，未认证候选不冒充正式通过样例。

目前证据是随机tiny Core的真实原生PDD头与legacy backbone/head注入、原采样窗口两输出一致、
较大HIGH几何、LoRA／mask保留、实际EAV／Relay调用、冷／热／重建身份，以及原生两个、
动态四个新进程只执行HIGH。动态分支覆盖strength 0／0.5／1、原生float32／BF16适配器转换和失败重试；
额外用户LoRA确实保留并影响输出。任意新增可执行字段不因重试适配而获得虚假恢复认证。
测试变体标签不代表完整预训练FL2VA／Ref2VA素材质量；同尺寸恢复交接也不是learned权重执行。
浏览器保存重载、完整权重GPU／音画／人审、自动缓存与更多后端仍未完成。

## 已有原生／LBH／完整首采工作流：不重建采样器

`MiniMaxH3NativeStageBindEXPT8`接入已有原生 Dual-Clock Setup 的 MODEL、SAMPLER，
以及实际选择的 SIGMAS、AV latent，选择 `native_low` 或 `native_high`。
它原样传出 SAMPLER／SIGMAS，只在MODEL新分支上绑定不可变阶段描述，不采样、不生成新表，
也不隐藏二采。原生Euler和dual-clock Euler已接完成审计；其它连接仍可运行，未适配的
多次求值采样器不会获得假的完成／持久身份（Heun三间隔实际五前向有测试）。

```text
原Dual-Clock Setup MODEL／SAMPLER + 原计划SIGMAS + 实际AV
              ↓ Native Stage Bind
        可选Stage EAV Apply → BasicGuider → 标准采样器或Stage Sampler
```

需要保存结果时用Stage Sampler并接Bind的stage_context；LOW和HIGH各接一套，
中间仍用原learned 3D upscaler、实际尺寸的HIGH条件及原Reconcile，不换成交给大节点执行。
Relay在各阶段实际尺寸分别生成成对MODEL／CONDITIONING；EAV、Bias/STG的实际时钟和
身份适配复用公共实现。节点描述的是**实际连接的原生阶段**，不是靠一个配方标签证明SIGMAS来源，
更不能拿它宣称PDD／VDN／FastH3 V2的专属状态与效果已经适配。

构图工具 `tools/build_modular_native_explicit_workflow.py --output-dir <新私有artifacts目录>`
提供40张候选（10种组合×最小／外置效果／保存／只恢复HIGH）：

- base-flow 4+4：两阶段分别调用原TwoPassSigmaPlan，以各MODEL的实际AV shift投影base q，保留restart_base_noise。
- LBH 4+3、4+4、4+5：原simple8前四区间和发布的独立HIGH raw-video表，不重映射成另一个shift的“剩余半段”。
- 完整原生8或20 + HIGH3／4／5：LOW取原native_flow完整表，HIGH仍独立LBH表；总次数为11–13或23–25，不写成总8步。

其中纯原生完整8步是原有单阶段节点的显式可组合示例，不是旧VDN8→原生HIGH比较器的
替代品或验收证据；真正VDN首采组合使用上面的独立VDN→native构图工具和专属证据。

LOW的denoised_output仍接learned放大，HIGH条件宽高直接来自放大器；每阶段自己的调度计划
不依赖另一阶段MODEL，因此修改HIGH的独立LoRA／计划不会构造LOW的反向依赖。
base-flow／LBH候选保留原Reconcile的`auto/legacy_policy`联合音频；完整首采候选默认
`first_pass/0`保留完成音频，并在解码前接原Audio Audit校验／重锁。用户改变音频强度时须同步其审计。
这里不替换旧Dual长片runner特有的前缀／迁移逻辑，也没有自动覆盖任何旧JSON。

恢复图显式读取`native_low`的path/SHA与denoised_output，全部LOW模型／条件／调度／采样／效果节点
从图中删除，只执行HIGH。当前仅有tiny CPU实际原生数值、冷／热／重建、LoRA／mask／效果和
10个新Python进程恢复证据；同尺寸交接不是完整learned权重验证。40图通过Core与序列化检查，
浏览器编辑重载、完整权重GPU／人审仍待对应验收；VDN首采组合证据不与本批40图混算。

## 手动二采：完整 FIRST → 独立 SECOND

`MiniMaxH3ManualPassStageSetupEXPT8` 把两个原长视频 runner 的
`manual_second_pass` 公开为 `manual_first` 和 `manual_second`，每个 Setup
只准备 MODEL／SAMPLER／SIGMAS／阶段合同，不运行采样。两个单阶段采样器的模型、
LoRA、条件、噪声和效果都可独立编辑。默认首采完整 native_flow20，二采手工视频
sigma 为 `0.5,0.412,0.35,0`；AV shift 沿旧合同分别转换。

接线是 **FIRST output → SECOND latent_image**，不是 V2／learned 路线使用的
denoised_output，不自动插入放大器。第二采自行准备噪声；要复现旧 runner，
两阶段保留同一 seed、batch_index、native mask 和 FreeNoise segment_index。
`tail_subdivide` 仍是一采，不计成此路线。旧 effects runner 的 EAV 原先只绑定首采；
分离图可显式给二采增加独立 EAV，这属于用户新选择，不宣称与旧默认输出相同。

每阶段可按 Relay Conditioning → Setup → EAV Apply → Guider → Stage Sampler →
EAV Audit 接线。FIRST 的 Stage Save／Load 使用 output 槽0；读取冻结 FIRST 后的
恢复图删除首采模型、条件、噪声、效果、采样器及输出节点，只运行 SECOND。
恢复时需填真实路径／SHA，并让条件几何尺寸与冻结结果一致；不是自动请求缓存。

旧 Core 采样器选项仍保留。当前可认证完成及跨进程复用的是 `dual_clock_euler`
和 `euler`；其他采样器可新运行，但其多次求值／中间 sigma 的完成与效果覆盖适配
尚未齐全，不虚报一间隔一次前向或授予持久复用资格。

`tools/build_modular_manual_pass_workflow.py --output-dir <新的私有artifacts目录>`
生成 native noise／FreeNoise 两组，每组有最小、外置效果、保存效果、只恢复二采四图。
这是可导入候选；CPU tiny、Core 图校验不等于浏览器编辑重载、完整 GPU 或人审通过。

### 外置 FreeNoise

`MiniMaxH3StageNoiseEXPT8` 放在 RandomNoise 与阶段采样器之间，公开
`paper_permutation`、`variance_preserving_blend`、`from_model_plan` 和 `disabled`。
它只包装一次原 NOISE 的 generate_noise，调用原 FreeNoise 重排，不执行模型；
输入 NOISE 继续负责 seed／batch_index，音频噪声保持原对象，段号显式可编辑。
disabled 或 MODEL 没有计划时返回原 NOISE 对象。from_model_plan 需连接携带旧计划
的 MODEL；仅在 MODEL 上保留计划而未接此 Noise 节点，标准 RandomNoise 不会执行它。

该 NOISE 端口也可用于原生 Dual／V2 阶段。已有视频重排、音频对象、seed／段号／
batch_index 及阶段身份 CPU 对照；未声称所有后端、全尺寸长片 FreeNoise 组合已验收。

## 原生 Dual：LOW4／LOW20 与 HIGH3／4／5

新增 `MiniMaxH3NativeDualStageSetupEXPT8`，每个节点仅准备一个阶段，不采样、不调用Loop：

| stage | 原调度 | 后续交接 |
| --- | --- | --- |
| `dual_low_4` | Core simple8的前4区间，末端非零 | denoised_output用于放大，音频尚未完成 |
| `dual_low_20` | 完整native_flow20，末端零 | denoised_output用于放大，音频已完成 |
| `dual_high_3`／`dual_high_4`／`dual_high_5` | 各自发布的LBH原始视频sigma表 | 最终解码output |

HIGH使用自己的MODEL和video/audio shift，不是LOW轨迹的剩余部分；不能换成FastH3 V2 DMD调度。两个公开Setup接两个标准采样器，或两个单阶段Stage Sampler。模型、LoRA、条件、noise、EAV和Relay都可以分别编辑。修改HIGH专属输入不会成为LOW的上游依赖。

完整接线为 `LOW denoised_output → 原learned3D放大 → Native Dual Handoff → HIGH采样`。
HIGH条件的宽高接放大器的实际输出，`MiniMaxH3NativeDualHandoffEXPT8`同时接放大latent、HIGH模板和HIGH positive。
交接节点复用原Dual runner的音频策略，不自行改写数学：

- `first_pass_steps=4`＋`auto`继续未完成的联合音频；模板mask为0的区域使用模板已锁音频，其余保留coarse音频。
- `first_pass_steps=20`＋`auto`保留已完成的一采音频。
- 显式`first_pass`、`highres_template`和强度沿用旧解释；LOW4的`first_pass/0`仍按旧兼容逻辑迁移为joint continuation，并在报告中注明。

节点不执行放大或额外扩散，不改变旧接缝、accepted-picture LOW参考或长片runner。此批只公开单段；长片逐段可编辑body仍待实现。

旧 `FreeNoise Long Video` 节点只向MODEL记录长片噪声计划，由旧runner在prepare_noise之后消费。分离图需显式接上述 Stage Noise 并选择 from_model_plan，标准RandomNoise不自动消费计划。旧runner行为不变，完整长片组合仍待验收。

Stage EAV和普通原生Relay的阶段身份、保存／读取已扩展到这些原生Dual阶段，不借用V2 owner。tiny CPU检查涵盖6种组合的原调度和原生输出逐位对照、冷／热／重建身份、非零LoRA与native AV mask，以及6个新Python进程读取LOW后只执行HIGH3／4／5。测试交接是同尺寸tiny数据，没有冒充真实学习型放大权重或完整成片验证。

`tools/build_modular_native_dual_workflow.py --output-dir <新的私有artifacts目录>`一次生成24张候选：6种LOW/HIGH组合分别提供最小分离、外置效果、外置效果＋保存、读取LOW只跑HIGH四种图。默认原生底模；LOW4和HIGH有独立EMA B加速LoRA节点，LOW20使用原生底模。图不强制用第三方attention。效果图EAV默认report_only；LOW/HIGH Relay在实际尺寸各自绑定，可复制Plan让两阶段使用不同事件。

恢复图真正删除LOW模型、条件、采样、效果及其输出节点；填写真实保存path和SHA后才可运行。24张frontend/API候选已通过真实Core校验，浏览器编辑／保存重载、完整GPU／画质／音频及接缝人审均未认证，未放入正式示例目录。

## RF Restart：真正独立的后段

新增三个节点，覆盖独立 RF、Detail Mixer 与 Two-pass Detail Mixer 的第二次下降：

- `MiniMaxH3RFBaseStageSetupEXPT8`：显式接入首采 SIGMAS，只配置一次原生 dual-clock 下降；完整表、外来 partial 表、tail 加步均由外部节点提供。
- `MiniMaxH3RFHandoffEXPT8`：接首采 `output` 和原始模板，绑定两个内容身份，不采样、不生成噪声。Stage Sampler 的 RF 输出或对应保存资产已经携带模板时可不接模板口；输出实际视频宽高供后段条件使用。
- `MiniMaxH3RFRestartStageSetupEXPT8`：只配置设备上的联合 AV 重噪与后段下降，输出独立 MODEL、SAMPLER、SIGMAS、latent 和 stage_context。不会隐藏执行首采；steps=0 或 sigma=0 对应空 SIGMAS、零模型调用，不假报一次采样。

```text
独立首采 MODEL／LoRA／条件／NOISE → RF Base Setup → Stage Sampler
                                                    │ output（可Save）
                                              RF Handoff
                                                    ↓
独立后段 MODEL／LoRA／条件／NOISE → RF Restart Setup → Stage Sampler → 解码
```

两种 Mixer 的 tail、Model Time Bias、STG 都在画布上单独连接，不是换一个大封装。
Two-pass Detail Mixer 原本是在 LOW→learned 放大→HIGH 之后追加 RF，候选图保留三个
实际采样阶段，不能为了拆 RF 而删掉前面的空间双采。三种入口各提供最小、外置效果、
带保存和只恢复 RF 后段四种候选，由 `tools/build_modular_rf_workflow.py --output-dir <新私有artifacts目录>`
生成；共12张 frontend/API 图。恢复图读取 `rf_base` 的 output 槽0、原始模板和原始终点，
真正移除所有 LOW／BASE 采样链。必须填写真实 path/SHA，并保持冻结片段的长度与 Relay 布局一致。

### RF 的原模板、精度与噪声不是普通 LATENT 的隐含属性

旧 RF 的 inpaint 和 `_rebase_partial_audio_start` 使用**首采原始模板**，不是完成的首采结果。
独立后段保留旧算法中再次换算音频起点的实际行为；这次拆分没有把它当作算法缺陷擅自修改。
重噪使用独立 `restart_seed` 在实际设备生成完整 packed AV 噪声；NOISE 端口仍提供原 inpaint
noise/seed。复现旧图时，应连接相同的确定性 NOISE 配置（包括 FreeNoise 的种子、模式和分段参数），
不能把 `restart_seed` 当成 NOISE 端口的替代品。未知或有内部可变状态的噪声源不认证跨运行复用。

标准 Core 在采样节点出口将 packed 结果转换成 float32；旧内部 RF 则在这之前继续运行。
外部 float64 SIGMAS 会使 Euler 内部状态升为 float64，因此只保存普通 LATENT 和 sigma dtype
不能精确复现旧路径。RF Stage Sampler 现在同时保存转换前的原始模型空间终点、其内容身份与
对应可见 output 身份；两个公开 LATENT 输出保持 Core 原样。Restart 检查身份、形状、有限值和
当前模型的 latent 格式，使用原精度终点，而不是从舍入后的 output 或 denoised 猜测恢复。
denoised 不冒充原始终点；修改可见 output 或保留终点后携带旧描述会明确失败。

使用普通 `SamplerCustomAdvanced` 仍可以接 Handoff，但需要显式 `original_template`。
可选 `base_sigmas` 只补齐调度 dtype（省略时默认为 float32），**不会凭空补回已丢失的内部精度**；
需要和隐藏 RF 数值一致及跨进程精确恢复时，使用 RF Stage Sampler 产生的绑定状态。
额外原模板／原始终点会增加保存体积；沿用既有不可变存储与大小限制，不覆盖旧记录。

### 外置效果及验证边界

RF 同样支持 Relay 成对 MODEL–CONDITIONING 和独立阶段 EAV。
Bias 改变模型实际看到的 sigma；STG 在原积分间隔之外增加 weak forward，并可能跳过部分 attention
block。RF 的 EAV 审计按真实时钟、主／弱前向和实际 block 数分别检查，不用积分步数假充网络调用数。
已识别的原生 Bias/STG、Relay、EAV 身份只在检查用克隆上投影，实际运行中的 LoRA、callback、mask
和注意力 delegate 不清除。未知补丁保留执行但不冒充可移植身份或完整效果覆盖；非标准 guider
等组合的实际计数可能仍标 unverified。

当前证据是随机 tiny Core CPU：三旧入口两路 AV 和两个输出逐位对照、四种 SIGMAS dtype、
原生 mask、非零 LoRA、Bias/STG 与 Relay/EAV 的冷／热／重建身份、原始终点篡改拒绝、
四个新 Python 进程只恢复 RF 后段。候选图验证不是浏览器编辑保存重载，也不是完整权重 GPU、
画质／声音／接缝人审；旧一体节点和旧正式图未替换。

## FastH3 V2 单段接线

新增 `MiniMaxH3FastH3V2StageSetupEXPT8`，分别设置 `low_0_4` 和 `high_4_8`：

```text
LOW MODEL → 独立LoRA → LOW Stage Setup → BasicGuider → SamplerCustomAdvanced
LOW条件 ────────────────────────────────────↑          │ denoised_output
                                                   原有 learned 3D upscaler
                                                           ↓
HIGH实际尺寸条件 ───────────────────────────────→ 原有 Reconcile
                                                           ↓
HIGH MODEL → 独立LoRA → HIGH Stage Setup → BasicGuider → SamplerCustomAdvanced → 最终解码
```

每阶段使用自己的噪声节点；HIGH的模型、LoRA、提示词、适用效果可独立修改。LOW的 `denoised_output` 是供现有学习型放大/协调使用的预测，不要误接 `output` 的未完成状态。音频沿旧 `auto/legacy_policy` 继续联合采样；不要默认锁住未完成的一采音频。HIGH条件使用放大器实际输出宽高。

此节点仅公开既有DMD绝对0:4/4:8，AV shift仍10/3。`official_comfy_template_exp` 使用另一种调度/采样器，不是本节点的4+4配方。标准采样器仍在画布上，并没有隐藏第二采样。

`stage_context`是不可变描述，不是张量来源证明、完成回执或断点缓存。普通ComfyUI同一会话缓存仍由真实依赖决定；需要显式跨进程保留阶段时，使用下述独立结果/存储节点。

## 外置 Enhance-A-Video

新增三种节点：

- `MiniMaxH3StageEAVConfigEXPT8`：配置mode、tau、全局视频sigma进度窗、工作空间和增益上限。可一份配置接两阶段，或两份独立设置。
- `MiniMaxH3StageEAVApplyEXPT8`：接在对应Stage Setup后、BasicGuider前，输入同阶段MODEL、SIGMAS、实际AV latent和stage_context。每个Apply有独立运行时。
- `MiniMaxH3StageEAVAuditEXPT8`：接对应采样器结果和Apply的runtime，查看实际前向、sigma、FETA及Relay调用。不以配置存在冒充效果执行。

只给LOW接Apply就是仅一采启用；只给HIGH接就是仅二采；两边都接可分别配置。`disabled`返回输入MODEL原对象，不安装补丁；它不会删除上游别的节点已安装的效果。`report_only`计算统计但不改输出；`apply_exp`才应用现有FETA增益。tau=0不等于关闭。

窗口始终是绝对 `1-video_sigma`，不是本阶段步数百分比。默认0.15–0.90可能让V2 LOW4没有激活步骤；审计会给出 `observed_no_steps_in_effect_window`，不会把窗口重置伪装成有作用。

目前真实tiny Core机械验证的主要路径是明确选择 `dense_compat_exp` 的LOW/HIGH。保留用户注意力delegate、LoRA/DiT补丁和native音视频mask；真实错误正常传出。EAV只直接缩放目标视频attention输出，后续层仍可能间接影响音频，不能声称整体音频永不变化。

### 必须明确的覆盖缺口

原生VSA chunked producer不构造完整Q/K，且绕过普通注意力selector。现在为源码可认证的V2 producer增加了独立FETA桥接：按目标视频空间列分块，额外调用原生QKV／RMS／RoPE投影计算时间统计，在原out_proj之前只给目标视频行应用增益；原VSA计划、conditioning sinks、coarse gate、头分组、pooled state及稀疏内核保留。增加的是受限QKV计算，不是额外扩散步，也没有构造整段视频QKV或静默换Dense。workspace限制针对桥接拥有的中间张量估算，不是整个模型或CUDA内核的总显存上限。

当前证据包含CPU精确FETA数学、真实Core的producer／分块／gate／头分组接线；集成测试将CUDA资格和最终Kitchen内核替换为CPU测试替身，因此**仍没有真实CUDA稀疏结果或完整成片资格**。审计分别报告selector和sparse producer调用；CPU正常不满足CUDA资格而走Dense的结果也不能证明VSA成功。稀疏Relay需要的逐query时间偏置仍未适配，组合会明确报告 `unverified_relay_coverage`，不将其遗漏或伪装成已实现。

未知第三方producer也可能绕过入口；保留它并报告覆盖不足，不以未验证为由阻止用户选用。新Apply目前支持V2、原生Dual、显式原生阶段绑定、PDD、VDN、手动二采及RF阶段合同，其他配方的外置效果适配将逐项增加，旧EAV节点继续按原合同工作。

## 外置 Prompt Relay

复用现有 `Prompt Relay Plan` / `Query Route` / `Prompt Relay Conditioning`，不用把时间事件塞回Loop。LOW、HIGH分别在实际尺寸调用Conditioning，使用各自成对的MODEL、positive和AV latent。Plan可共用；若希望二采不同事件，复制Plan后单独接HIGH，不能串用另一个阶段的MODEL–条件绑定。

顺序为 Relay Conditioning → Stage Setup → 可选Stage EAV Apply → BasicGuider。新EAV在认证后将该阶段的Relay wrapper合成一个运行时入口，保留其注意力delegate与真实时间偏置；错MODEL/条件配对仍报错。HIGH Reconcile继续保留HIGH条件元数据。

目前可用的候选图由 `tools/build_modular_fast_h3_v2_workflow.py` 生成；加 `--with-effects` 得到外置Relay和两份EAV配置版本。工具只生成新的私有候选目录并做CPU真实Core图校验，不提交GPU任务、覆盖旧图或发布未经审片的正式示例。每个候选的说明和审计列明资格边界。

## 保存一采、只重跑二采

新增三个可选节点，不替换原来的标准采样器图：

- `MiniMaxH3StageSamplerEXPT8`：五个标准采样输入保持分离，另接同阶段 `stage_context`；只调用一次原生 `SamplerCustomAdvanced`。前两个输出仍分别是 `output` 和 `denoised_output`，另输出 `stage_result` 及执行记录。
- `MiniMaxH3StageSaveEXPT8`：接 `stage_result`，向 `output/MiniMaxH3/stage_artifacts` 写一个新目录，输出 `artifact_path` 和 `artifact_sha256`。两者都要保存。它还原样传出两个latent，便于在保存之后接原有放大节点。
- `MiniMaxH3StageLoadEXPT8`：填写相对路径、精确SHA和预期LOW/HIGH阶段。只读取选中的完成记录与张量，不加载模型、不采样。交接槽位按配方选择：VDN完整首采使用 `output` 槽0；V2/PDD等partial路线使用 `denoised_output` 槽1，再接原有learned3D放大→HIGH条件→Reconcile→HIGH采样。

构图工具的 `--with-results` 生成双采+阶段保存候选；`--resume-high` 生成只含HIGH采样的读取候选。后者真正移除了LOW模型、条件、采样器和LOW输出节点，须先填入保存节点返回的路径及SHA，占位值不能执行。

这里是**显式冻结已选一采结果**，不是自动缓存查找。读取图不读取现在的LOW提示词/LoRA，也不宣称它们仍与旧结果相同；要改变LOW内容，应重新执行LOW并保存新的结果。之后的HIGH模型/LoRA/条件仍独立可改。

保存内容为无pickle的safetensors及最后提交的manifest。原始x_sigma、预测denoised、native AV结构与可序列化metadata分别校验；坏SHA、错阶段、缺完成文件、路径越界、超限metadata、执行期间输入变化都会报错，不静默重采。20GiB单文件和4MiB元数据上限；操作系统锁保护写入/读取，持锁进程终止后可重新打开，不靠删除旧锁猜测任务结束。保存永不覆盖原目录。文件指纹变化会使读取节点重新校验，不用旧内存结果掩盖损坏。

目前普通V2 dense/trained配置的tiny CPU路径已检查冷/热加载、重新建立条件、非零原生LoRA、阶段输出精确保存/读取、新Python进程只执行HIGH并与原进程结果一致。条件UUID仅在已识别Core条件槽位作稳定映射，未知可执行组合不因此获得可移植身份；读取不会卸载或解除共享模型的补丁。

新增效果身份适配已覆盖原生Stage EAV、普通 `dense_compat_exp` Relay，以及普通Dense Relay＋Stage EAV组合。身份校验实际wrapper／selector／callback／V2 producer的源码闭包和绑定，不靠节点名字；记录EAV参数、绝对阶段、native mask、Relay事件／布局／query route和实际分块参数。只在检查用MODEL克隆中剥离已认证效果，运行中的权重、LoRA、sampling对象和callback不卸载、不重装。运行中配置、mask或效果owner变化不签发原身份的完成回执。

这些已识别组合在tiny CPU上验证了标准原生采样输出逐位一致、冷／热／重建条件身份相同、非零LoRA保留、保存读取，以及新Python进程读取LOW后仅执行带效果HIGH4。`trained_vsa_exp` EAV持久身份的数值恢复证据仍是CPU资格下的Dense执行，不外推真实CUDA VSA。稀疏Relay、未适配的第三方Relay后端／observer／未知补丁仍可新采样和归档，但认证跨运行读取拒绝缺少身份的记录；它们仍是后续开发范围，不是被删除的需求。

构图工具现在允许 `--with-effects --with-results` 生成带独立EAV／Relay及阶段保存的双采图；`--with-effects --resume-high` 生成读取LOW、只跑带效果HIGH的图。LOW采样／模型／条件／效果及其输出节点真正移除，HIGH在原learned放大器实际尺寸上重建成对Relay条件。保存结果经过对应实际效果审计再交给下游。两种候选都已通过真实Core图校验，仍需填写真实模型和保存路径／SHA，并做浏览器及实机验收。普通采样器、旧一体节点和旧缓存合同未改。

这些是tiny随机权重与同尺寸交接证据，不是完整预训练learned放大、真实GPU成片或人审；自动请求缓存/依赖失效与长片恢复也尚未完成。

## 历史阶段待办（以本页顶部后续资格为准）

S27 RGB → LTX 新增独立输入 Save／Load，支持显式冻结 RGB／原音轨／编码放大后的 LTX 输入，再只运行独立精修链；旧图不替换。见 [RGB/LTX 输入存取说明](LTX_RGB_SOURCE_STORAGE_EXP.md)。这只是数据交接，不是已完成采样回执，也不覆盖 learned adapter／Prepared worker 或 LTX 外置效果。

原生稀疏FETA的真实CUDA资格、稀疏Relay时间偏置、其他后端的效果持久身份、自动阶段缓存/依赖失效、完整预训练GPU/人审、浏览器编辑保存重载、其余双采/多阶段路线、长片/导演台可编辑body和未来配方门禁都仍在总计划中。不要将本页的局部节点视为整个项目改造完成。

## 全路线登记与查漏

`h3_t8/modular_sampling/catalogue.py`将S01–S29写成不加载Core或模型的路线目录，分别记录阶段名称、交接语义、变体、实际源码函数和已列出的公开节点。当前191处源码绑定包含S01–S03的显式原生Bind、VDN阶段／逻辑身份与实际补丁执行路径／EAV／专属Relay与三类构图器、PDD阶段和动态身份、原生Dual公开stage／handoff、两个manual runner及其外层第二采调用点、手动阶段／FreeNoise、RF三旧入口与公开阶段／交接／效果时钟，以及Progressive阶段／存取／效果、Avatar专属录音绑定／分离／交付／构图、Continuation来源／独立条件／阶段／外置效果／motion身份／交付／构图、HyperFlow连续HEAD/TAIL及强类型边界／完成结果／公开节点／持久身份／专属存取／构图。同一个执行函数可以服务多条不同分支，但不会因此把LOW4／LOW20、HyperFlow连续／fresh-noise或Chunked五种合同合并。RF重噪是两次下降之间的交接，不误计为第三次扩散采样；音频精修和局部／逐窗精修仍在范围内。

开发核对工具为 `tools/audit_modular_route_catalogue.py --live-core --progress <本地进度文件> --output <新的私有证据文件>`。它解析当前实际函数而非相信旧行号，记录函数AST／源文件SHA和调用名，检查声明的公开节点确实在根入口注册，并生成“公开阶段、可编辑图、EAV、Relay、恢复、旧图回归、GPU、人审”八维状态。证据文件缺失、定义改名／删除、缺节点或范围丢项会明确失败；不会覆盖旧基线。

这是源码／登记／证据绑定检查，不是算法验收或任意新源码的自动双采发现器。只有部分CPU和候选图证据的历史路线记录仍标partial；没验证的维度不能用目录存在自动勾掉。未来新入口的自动扫描／配方与三类接线准入已另行实现，以本页顶部当前门禁为准；不借它补写未运行的GPU或人工审核。
