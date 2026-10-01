# H3 → LTX RGB：独立输入保存／读取（本地 EXP）

## 当前显式加载策略的新验收（2026-10-01，本地）

新增可选 [LTX 原生一致流式策略](LTX_NATIVE_LOAD_POLICY_EXP.md)，默认仍原 MODEL／旧加载方式，只有显式开启才请求 Core 原生一致流式准备。一条公开节点 Identity＋EAV／Relay 的原生画布 full→fresh cold 已完成原三步／完整121帧1024×576画音，候选latent与全部decoded RGB／PCM精确一致，原音轨不变；六选定权重、源码、Core、存档及控制器九门全真。旧混合驻留模式下的失败链仍失败，不能由新模式改判。不是人审、任意素材／后端或全部S27认证；用法与精确范围见专题。

## 当前 RGB Identity＋EAV／Relay 的验收缺口（2026-10-01，本地）

一条当前原生完整／fresh 冷读取链两边都完成原 3 步，完整 121 帧 1024×576 的 H.264＋AAC 严格解码、原音轨旁路、EAV 非单位有界增益与 Relay 实际覆盖均通过，但候选 latent／RGB 没有逐值一致，整链仍失败。121 帧来自旧 `trim_to_8n_plus_1` 对 124 帧源的既定裁尾，不是新改时间轴。旧失败证据保留；先前其他素材、尺寸或 source-only 结果不能代替这一条当前链。

新的独立输入诊断只执行原生 full 新 SourceSave→fresh SourceLoad，不运行扩散；同一配置的采样前 latent 值、dtype、设备、stride／offset、原生条件、sigma、noise seed 与 CFG 都一致。它支持该次输入存储的逐值保真，不证明后续 MODEL 加载或采样结果一致。驻留／流式 LoRA 的实际计算路径正在另行核对，尚未认定 GPU 差异根因，不更改存储格式、旧加载器或 LoRA 数学，也不降低完整／冷候选一致性门。当前仍非全部 S27、人审或发布资格。

## 2026-09-28 LTX 外置 EAV 真实权重小画布 A/B/A

普通与Identity各用自有8956 Core同一进程，以真实Gemma/LTX权重对组合图连续执行EAV `report_only`→`apply_exp`→`report_only`，Relay始终`apply_exp`、EAV `tau=4`。两条路线的两次基线候选latent逐值相同，开启EAV时各144次视频块施加、增益大于1，候选latent与基线有数值差异；普通max1.41726／RMSE.32284，Identity max.56691／RMSE.06916。见私有`trained/20260928T131755Z-ordinary-eav-aba`与`133209Z-identity-eav-aba`。首次Identity `132036Z`第三段前中断，无完整结论，保留原工件。这里只导出128×64候选latent，不是1024×576完整音画、冷恢复、画质、人审或稀疏／第三方后端资格；原已保存工作流仍默认EAV `report_only`，先前完整图codec失败不改判。

## 2026-09-28 四路线冷恢复原生画布：机械通过，非质量验收

四张私有`resume_ltx`图已在独立8958 GPU Core中由真实Chromium原生Queue逐张执行：普通／Identity各Relay-only及EAV＋Relay。冻结输入逐文件SHA绑定，不挂原视频；每条history精确对应画布POST、3步真实采样、1024×576／113帧24fps完整画音严格解码，并与各自先前独立API cold结果的解码画面／音轨SHA相同。报告在`artifacts/development/modular-ltx-relay-20260928/canvas/20260928T130430Z-ordinary-relay-cold`、`130559Z-ordinary-eav-relay-cold`、`130727Z-identity-relay-cold`、`130854Z-identity-eav-relay-cold`。首个普通尝试因Core自动创建空`input/3d`目录被过严检查误判，原失败回执保留；限定空目录后重跑的四份报告全真。先前Identity＋EAV**完整图**的原严格解码失败仍然有效，不能被冷图通过覆盖。组合EAV仍是`report_only`，没有验证`apply_exp`画质；只有一份历史素材，也没有人审、多素材、长片或全部后端资格。8958服务均已停止，8940本轮前后无监听且未操作。更详细边界见私有roadmap顶节。

## 2026-09-28 四路线完整图原生画布：3通过／1严格出口失败

继普通Relay-only画布成功后，普通＋EAV、Identity Relay-only亦用独立GPU Core/真实Chromium原生Queue完成，三张完整片各自与此前同路线API控制的解码画面／音轨SHA相同；两组合图 EAV 均`report_only`，不是EAV实际增益资格。Identity＋EAV画布请求和三步采样已执行，但隔离出口原运行在`strict_decode_video`报坏帧，`video.partial.mp4`及raw音画保留，未提交合格成片。该失败批的raw YUV／音频与此前通过的API full相同，服务退出后原partial在不变SHA下可严格解码，只作故障分层证据，不反写通过或断言根因。前一次Identity私有控件索引错误批次同样留证；新测试固定backend与verbose槽。8958／所有自有进程停止，8940不变。精确路径见roadmap顶节。冷图画布、其它素材/后端/长片与人工画音仍欠。

## 2026-09-28 原生画布一路机械成片（非全部S27）

在此前8图原生编辑／另存重开及8组API全／冷媒体之后，自有8958/GPU Core对私有普通 `full_save_external_relay` 图由真实Chromium点击原生Queue。浏览器POST、prompt history、3/3真实采样步与隔离出口MP4绑定；当前前端仅为LoadVideo附加空的`video-preview`显示字段，独立后验严格限定这一个差异。`canvas/20260928T002807Z-ordinary/postflight-v2.json` 的12项检查全真；1024×576、113帧24fps、4.708333秒的完整画音严格解码，解码画面及原音轨SHA分别与此前独立API同配置完整图相同。源为历史H3 5秒原生片，不是本次H3首采；只做一路全图画布机械运行，没有冷图画布、组合EAV画布、人工看完整片／听审、多素材及全部后端。第一次浏览器URL匹配超时及第二次控制器过严对比均保留失败原报告，后验不覆盖。8940未改，自有服务已停；旧图与默认模式不变。精确回执及后续门见roadmap顶节。

## 2026-09-28 新增完整媒体机械验收范围

私有 `run_modular_ltx_relay_cold_gpu.py --include-media --confirm-run` 在默认 latent-only 探针之外，显式保留隔离候选导出，并从冻结 source frames/audio 各增加一条原片参考导出。普通／Identity×Relay-only／EAV+Relay 四路线各在128×64及1024×576，执行真实训练权重 full_save→全新Core resume_ltx；每阶段三步，均113帧24fps。八组回执全部通过严格MP4画音解码、完整／冷候选画面解码SHA相同、四份音轨与原片参考相同、候选latent逐值相同；32个MP4的文件SHA另只读复核。私有路径、精确证据和历史失败见 `roadmap.md` 顶节。组合EAV仍为`report_only`，这里只证明独立测量与Relay施加可共存。先前非Relay图两次cold编码／解码失败不改判。

在这一较早截面，这仍只是一份历史H3原生片在两档几何的API运行；后来画布进展见本文件顶节。无论哪一截面，都不代表视觉或听觉质量、多素材/长时长、所有后端或整个双采分离计划完成。用户8940已在再次确认保存后重启并载入四个Relay节点；本地候选不可当作公开推荐工作流。

其后另在自有8957 CPU Core/真实Chromium对八张私有图逐张打开，可见编辑独立Plan/Apply（四组合另编辑EAV配置），原生Save As、刷新并重开，具名连线和widget语义审计8/8通过；证据 `artifacts/development/modular-ltx-relay-20260928/browser-roundtrip-v2.json`。首次严格审计因前端给已连接的两个STRING报告输入补空显示widget而停，限定这两个节点和两个字段的规则修订后仅复审已保存副本，不抹首错。队列/history为空；**不是画布点击运行或主观效果验收**。原八图未改，QA拷贝的效果模式编辑未排队。

## LTX Prompt Relay 外置 Plan → Encode → Apply → Audit（本地 EXP）

四节点 `MiniMaxH3LTXPromptRelayPlanEXPT8`／`EncodeEXPT8`／`ApplyEXPT8`／`AuditEXPT8` 追加在原完整563节点之后，当前567。Plan 读取**实际 LTX LATENT** `[B,128,T,H,W]`，以 `(T−1)×8+1` 得出输出帧数；T=15 是113帧，不借用 H3 的17n+5。全局／逐事件提示词可自动均分，或用帧／秒／百分比范围及已有链式 Event 节点；内容哈希与 latent 形状绑定。旧 H3 Plan、旧节点前缀和旧工作流不改。

Encode 对用户选定 LTX CLIP 分别编码全局与每个局部事件，按实际输出 embedding 长度绑定 token span，再合并成正向 CONDITIONING；负向 CLIP 线路保留原样。选定 Gemma 的 connector 可能附加 register token，这些尾部 token 不施加事件偏置。Apply 接 Setup/LoRA 后的 MODEL、原 SIGMAS、阶段 LTX latent、Encode 的正向条件及进程内 binding；`disabled` 原样直通，`report_only` 只观测，显式 `apply_exp` 才对视频 cross-attention 文本段施加有界时间偏置。原 `attn2.forward` 的 Q/K/V、RoPE、mask、STG、gate、to_out、已选 attention 后端及第三方可委托 callable 不被清空；音频 attention 不改。负向 CFG 行与 guide tail 保留原输出，已有遮罩与事件偏置合并。未知不委托的 producer 仅报告未覆盖，不因未知补丁本身禁用运行。Audit 报实际调用次数与未覆盖原因，**不**认证候选来源、可移植缓存、完整采样、画音或质量。

`tools/build_modular_ltx_relay_workflows.py --output <新项目artifacts目录>` 从隔离媒体输入图生成普通／Identity Preserve 各 Relay-only、EAV+Relay 的 full_save 与 resume_ltx，共八张私有 opt-in 图；freeze_source 不附加效果。Plan 从当前冻结或读取的 LATENT 接入，Apply MODEL 接 CFGGuider／Stage Bind／Stage Audit，Apply 正向接 LTXVConditioning，原负向与原 H3 音轨旁路不变；组合图让 EAV 和 Relay 分别占视频 self-/cross-attention。默认两效果均 `report_only`。当前 Core 静态全输出8/8通过，旧六张输入图、旧四张 EAV 图和公开原图不变。

真实 Gemma4／LTX-2.5 INT8／distilled LoRA／原视频 VAE／learned x2 输入，在独立8956服务按保存API跑 128×64、113帧时间线、3步：普通 Relay `apply_exp` 的48个视频 cross-attention块各被观测／施加3次，原后端委托144次；普通与 Identity 的 EAV+Relay 组合各自 Relay 144次、EAV 144次测量，EAV 保持 `report_only` 零施加。实际 Gemma 合并文本46个原始 token，connector 追加978个尾 token；仅前述46个绑定 token 接受事件偏置。

跨独立服务即使同为 `apply_exp`，本机结果也曾有数值差异，不能仅凭跨服务 A/B 差断言效果。因此另用 `tools/run_modular_ltx_relay_pair_gpu.py` 在**同一个自有 Core PID**按 A=`report_only` → B=`apply_exp` → A=`report_only` 连续跑三次，每次真实重新采样3步；提交API规范化后只差 mode 和输出文件名前缀。两次A的float32候选latent逐值完全相同，max_abs=0、RMSE=0；B与A不同，形状`[1,128,15,2,4]`、max_abs=1.55644536、RMSE=0.36869201；A/ B/ A分别记录Relay施加0/144/0次，均观测144次。最终回执 `artifacts/development/modular-ltx-relay-20260928/trained/20260927T232007Z-ordinary-aba/report.json`，仍不是媒体或质量验收。所有自有服务已停止，未触碰用户8940。证据在 `artifacts/development/modular-ltx-relay-20260928/{candidate-v1,trained}`；首个GPU探针因自有input目录未配置而在提交验证被拒，未采样，错误回执保留，修探针后重跑成功。

`tools/run_modular_ltx_relay_cold_gpu.py` 又对四种路线（普通／Identity × Relay-only／EAV+Relay）分别用真实权重执行 full_save→**全新 Core 进程** resume_ltx。四组 full/cold 均各真实采样3步、Relay每组两边各144次施加；两组合图的EAV两边各144次测量且保持report_only。冷图无原视频解码／准备／VAE编码／latent放大，严格从内容SHA绑定的 Source Load 读冻结输入。四组候选 latent 均`[1,128,15,2,4]`、逐值相同、max_abs=0，服务／端口全部退出；四个回执位于 `trained/20260927T232752Z-ordinary-cold`、`20260927T232929Z-identity-cold`、`20260927T233148Z-ordinary-eav-relay-cold`、`20260927T233321Z-identity-eav-relay-cold`。这关闭了四路线**小尺寸候选latent冷恢复**门，不是完成的 MP4、音轨、画布或质量门；此前隔离媒体cold失败仍有效。

上述是**独立API小尺寸采样与候选latent冷恢复**，不是画布点击、原尺寸、full/cold成片、长片、多素材、主观画音或论文级 LTX Relay 质量验证。八图沿用的隔离媒体出口此前两条 cold 整图失败仍有效；新四路线未做完整 MP4／音轨严格媒体门。用户8940在新 Relay 四节点添加前重启，未获再次画布保存确认不得重启或宣称这些节点在该服务中可见。原时间线单独CPU 411项与后续单元／工作流测试分属不同截面，不相加作全仓结论。

## S27 外置 LTX 视频 EAV（本地候选，未完成全部验收）

两个新节点 `MiniMaxH3LTXEAVApplyEXPT8`／`MiniMaxH3LTXEAVAuditEXPT8` 追加在此前完整 561 节点之后，不修改旧节点与保存工作流。前者可复用现有 `MiniMaxH3StageEAVConfigEXPT8` 的配置；接线为：LTX Setup/LoRA 的 MODEL＋该阶段 SIGMAS＋输入 LTX LATENT → Apply，Apply MODEL 同时接 CFGGuider、Stage Bind、Stage Audit；原 Stage Audit 候选 LATENT → 新 Audit → TAEHV Decoder。音轨仍走原 H3 AUDIO 旁路。`disabled` 返回原 MODEL；`report_only` 只测量；`apply_exp` 才放大视频自注意力的完整输出。前者与后者均不新建采样器或改变 sigma／时间坐标。

效果保留实际安装的 attention forward 与 Core 内部 STG、GuideAttentionMask、普通遮罩、RoPE、门控及已选后端；只在 post-RoPE 的视频 Q/K 上按时空布局计算有界 chunked CFI，然后对原 forward 已完成 `to_out` 的 noisy video 行施加增益。参考 guide 行和 audio attention 不受 EAV 增益影响；已有 optimized-attention override 继续委托。统计本身不把 attention mask 加进时间 CFI，这一点在报告中公开。外部覆盖路径如果不调用底层选择器，报告为未覆盖，不能把采样完成冒充 EAV 生效。增益超过配置硬限时原样抛错，不静默截断或降级。报告不证明候选 LATENT 的来源、采样完成、可移植缓存或画质。

`tools/build_modular_ltx_eav_workflows.py --output <新的项目artifacts目录>` 已产出私有四图：普通/Identity Preserve 的 full_save 与 resume_ltx，使用当前隔离媒体候选；freeze_source 只负责输入保存，不引入效果。EAV 默认 `report_only`，用户需显式选择 `apply_exp` 才会更改数值。四图静态 Core 全输出校验通过，既有六图、公开两图及源文件字节保持。当前 CPU 回归含真实小型 Core LTX Euler 三步、两种 Setup、0/0.7 权重增量、三模式、重复生命周期、Core attention 遮罩/STG/RoPE/门控/旧委托，均不等于预训练权重或真实成片。训练权重完整／冷媒体、画布编辑保存重开、人工画音、Prompt Relay 外置及其它 S27/S28 路线仍待；前述隔离出口的两项冷媒体失败仍独立保留。

普通 LTX Refiner 与 Identity Preserve 的 RGB 路线现在可显式冻结二采输入。新增两个节点，不替换原 Stage Bind／Sampler／Stage Audit，也不改原工作流。

- `MiniMaxH3LTXRGBSourceSaveEXPT8`：接原视频帧、原 AUDIO、准备后的 RGB、准备报告、已编码并放大的 LTX latent，以及原视频位深。默认 `confirm_save=false` 只直通；开启才保存。
- `MiniMaxH3LTXRGBSourceLoadEXPT8`：填写 Save 返回的相对 `artifact_path` 和精确 `artifact_sha256`，读取相同数据，并输出原报告的 fps／裁切时长及原位深。

保存位置是 `output/MiniMaxH3/ltx_rgb_sources`。每次创建新目录，保存 safetensors，最后原子提交 manifest；不覆盖既有记录、不用 pickle、不序列化 MODEL／CLIP／VAE／可执行补丁。保留原音轨的采样率、波形及支持的附加元数据。原视频无音频时 `None` 保持不变。

恢复接线为：Load → 原 Stage Bind → 原独立 `SamplerCustomAdvanced` → 原 Stage Audit → TAEHV Decode／Output Trim／Create Video。LTX MODEL、LoRA、提示词、NOISE／GUIDER／SAMPLER／SIGMAS 仍在画布外置重建；Load 不调用原视频读取、RGB 准备、VAE 编码、latent 放大器或 H3 采样。

这是**选择并冻结输入数据**，不是采样完成回执，也不证明当前修改后的一采与历史输入相同。输入来自用户选定的已生成视频；不能凭此称 H3 首采或 LTX 二采已经执行，不能作为通用 StageResult 或自动缓存。learned H3→LTX adapter 路线和 Prepared worker 路线不是此格式。

坏 SHA、缺文件、未提交的 partial、路径越界、符号链接／junction、锁占用、NaN、输入几何与报告不符均拒绝；不会偷偷重跑前段。读取节点以 manifest 和 tensor 内容哈希参与 Core 缓存，内容未变可复用，外部改写／删除后重新验证并报错。

## 候选工作流与验收边界

本节及下方 serial PyAV 章节保留早期候选的失败证据；最新隔离出口和六图集成状态见文末，不能用后续独立复查覆盖原失败。

`tools/build_modular_ltx_rgb_source_workflows.py --output <新的项目artifacts目录>` 生成普通与 Identity Preserve 两配方各三图：

- `full_save`：原分离精修链增加可选输入保存，独立采样控制保留。
- `freeze_source`：只准备并保存输入，没有 LTX MODEL／CLIP／采样器。
- `resume_ltx`：严格读取指定输入，仅跑 LTX 后段；删除原视频读取、编码和放大分支，保留音轨、fps、裁切时长和位深。

六候选已通过当前 Core 全部输出的静态校验；用于静态 API 校验的原视频占位换成本机已有文件名，不代表读取或生成成功。CPU 测试验证小型合成数据跨进程精确读回、原 Stage 重建、取消／损坏／并发锁保护及实际 Core 文件缓存失效；CPU 证据不等于训练权重或质量认证。

新增 `tools/run_modular_ltx_rgb_source_gpu.py`，默认只读预检，显式 `--confirm-run` 才在新私有目录启动独立 Core。普通和 Identity 两配方已分别实际执行 full／freeze-only／新进程 cold：真实 LTX-2.5 权重、原 VAE、learned x2 放大器、Gemma 编码器及 0.8 distilled LoRA，无模型替身；明确选择 dense-reference，原 sigma／采样器不改。两组完整／冷恢复的候选 latent 均逐值相同，最大绝对差为 0；冷图实际不执行原视频读取／准备／编码／放大，仍执行独立三步 LTX 采样。输入为已生成的 5 秒 H3 原生视频，不在本次重跑 H3 首采；按旧 8n+1 规则输出 113 帧、24fps、1024×576，约 4.708 秒。

**整体资格仍失败**：同一视频重新读取并冻结的源 RGB 有差异；普通 full/cold 和 Identity full 的输出有严格 H.264 解码错误，不能以 latent 一致代替成片一致。Identity cold 单片完整解码通过，四路解码音轨一致，仍不等于整组通过或质量认可。新报告区分每份媒体、音频旁路、几何时长与解码失败，不将两个失败的空哈希视作一致。

独立 CPU 媒体诊断（无 CUDA／无模型）也在 Core 1024×576 导出复现失败，512×288 导出及纯 PyAV 对照通过；仅在短寿命诊断进程内显式设编码线程为 1 后，该例通过。尚未证明根因或通用修复，没有给服务安装 monkeypatch，没有改 Core／旧视频节点／旧图；不得把诊断线程代理当成正式修复或将失败文件转为合格文件。

候选仍仅在本地 artifacts，不替换公开旧图。真实权重完整／冷恢复媒体一致性、画布操作、长片／多素材、LTX 外置 EAV／Relay 和人工画音验收仍待；S27 和全双采分离计划尚未完成。

## 独立媒体节点：诊断阶段，尚非合格出口

新增两个 opt-in EXP 节点，追加在完整旧节点列表之后，不全局修改 PyAV，也不替换原 Core 节点：

- `MiniMaxH3VideoComponentsSerialEXPT8`：前五输出与 GetVideoComponents 一致；第六输出记录真实 RGB／音频身份。原生文件逐流单线程，沿用 Core 的裁切、旋转、浮点转换及音轨选择；逐 packet／frame 检查取消，解码异常不静默跳过。外部 VIDEO provider 仍委托其自身实现，不能据此认证其解码线程或行为。
- `MiniMaxH3SaveVideoSerialEXPT8`：明确 H.264/AAC、8/10 位、sRGB/HLG/PQ；逐帧取消、RGB→YUV 及编解码线程显式设置为 1，保留原 fps 舍入和音频区间。每次新建目录，partial 完成后才提交成片，失败不覆盖旧媒体。报告包含输入 RGB／音轨、量化 RGB 和有效 YUV 值哈希；不把无效的平面填充字节当作图像内容。

当前仅在本机 PyAV 18.1.0 验证接口；不是任意 PyAV 版本的兼容保证。原生 codec 调用仍在本进程，尚无硬超时／进程树隔离；取消检查不等于可中断底层挂死。报告的 `strict_decode_verified=false`、`quality_accepted=false` 是有意保留的边界，不因成功写文件就改为 true。

`tools/run_modular_ltx_media_replay.py` 默认只读预检，确认后使用两个自有 Core 进程，只读取之前真实采样保存的 full/cold latent、原冻结输入和 TAEHV；**扩散 NFE=0，不重跑模型采样**。每组分别在解码导出前后重读源视频，严格检查完整画音和精确几何，保存所有原始回执，不更改旧失败文件。

最终仍失败：普通路线初次回放通过，但增补观测后的最终回放 cold H.264 严格解码失败；Identity 最终两片能解码、音频相同，画面不一致。两组编码前浮点 RGB、量化 RGB、有效 YUV 均相同，且重复源读取一致；因此不能将媒体差异归为采样或 TAEHV 输出差异，也不能宣称“单线程已修复”。一次独立 ffprobe 进程异常退出后的同文件复查成功，原失败单独保留，未豁免严格检查。

独立 FFmpeg 单线程重新编码对照三次文件及解码画面均相同，仅定位后续方向：需验证可取消、超时受限、进程树受控的独立编码出口，并保留同样的位深／颜色／音轨／时序语义。该对照不是节点实现，不可把二次有损编码样例当作正式修复；六张候选尚未接入或完成新的媒体／画布验收。

## 后续新增：显式隔离出口（仍未完成整图资格）

`MiniMaxH3SaveVideoIsolatedEXPT8` 追加完整旧560节点之后。它不替换上述 serial writer 或 Core SaveVideo：输入 VIDEO 与输出 VIDEO／路径／报告分离，Windows Job 管理独立 FFmpeg 编码及严格完整解码进程。原 RGB→YUV 位深／色彩顺序、原音轨采样率与声道、原 fps 舍入及裁切区间保持；不是将已有坏片再次有损转码。原生 AAC 编码器版本不同，不能承诺与旧 PyAV 的 AAC 字节或解码 PCM 相同；验证使用同后端的原音轨对照，同时核对输入波形身份。

原始数据指纹按不超过4MiB块计算，包括非连续 tensor，不复制整段视频；逐帧／音频块检查取消，显式磁盘预算与剩余空间门。子进程有总期限和进程树清理，stdout／stderr有上限；这不等于为父进程外部 VIDEO provider 的底层解码提供硬中断。metadata 经有限大小数据文件传递。成功返回前检查几何、位深、帧率、音轨、完整解码哈希和输入不变；10bit 的解码审计保留到 RGB48，不缩为8bit比较。失败保留 raw／partial／回执且不提交 `video.mp4`，不自动重试或切换算法。

已完成的范围：两条真实历史 latent 的零采样输出回放，各45门通过，包括完整画音哈希、编码前RGB/YUV、四次源读取、原数据及源码不变、自有进程退出。新的六图候选可用构图器显式 `--isolated-media` 生成，旧缺省输出不变；当前 Core 全输出静态6/6通过。执行工具需同时指定 `--isolated-media --saved-graphs <该候选目录>`，防止误用旧图作为新出口验收。

**整图仍未通过**：两配方新的 full／freeze 成功且冻结输入完全一致，两组 full／cold 的真实采样latent逐值一致，staged YUV及音频身份相同；普通 cold 在编码进程异常退出，Identity cold 在严格解码中报坏帧，两者均没有提交候选成片。普通失败原始帧在服务退出后三次独立编码精确一致；Identity同一partial文件在服务退出后由两版FFmpeg严格复查通过，文件SHA未变。前者的Windows异常为 `0xc000008f`（[官方状态码说明](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-erref/596a1078-e883-4972-9bbc-49e60bebca55)）；这些证据仍不足以确定硬件、负载或编解码器根因，不能静默忽略失败并称全面修复。

`tools/replay_isolated_video_request.py` 只用于对精确SHA绑定的私有失败raw输入作最多三次独立诊断，不加入节点的正常执行或恢复路径，不启动采样。画布实跑、完整稳定媒体、人审、LTX EAV／Relay及其他路线门禁仍未完成，六图继续留作私有候选，不发布为合格示例。
