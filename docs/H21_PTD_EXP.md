# H3 PTD 专用专家与采样（实验）

PTD 是独立的逐 hop 专家配方，不是普通加速 LoRA、PDD 32 头或 HyperFlow。完整采样、同画布连续 HEAD／TAIL 和实际审计入口已随 [v1.102.0](RELEASE_1.102.0.md) 发布；默认旧工作流不变。指定已审来源见[工作流目录](../examples/workflows/90-h21-reviewed/README.md)，不能把指定片验收扩为任意模型或任务资格。

## 接线与安全边界

`H3 PTD 专家加载与采样` 接用户所选 MODEL，读取已经由可信来源转换的 safetensors，输出匹配的 MODEL、SAMPLER、SIGMAS 和报告。将输出 MODEL 与提示条件 positive 接 `BasicGuider`；再将其 GUIDER、PTD 的 SAMPLER／SIGMAS，以及噪声和初始 latent 接 `SamplerCustomAdvanced`。需单独接线时，`H3 PTD 采样器与完整网格` 从同一个 PTD MODEL 取出其采样器和网格。`H3 PTD 实际专家与时钟审计` 的排序输入只建立依赖，不把已有 latent 当作本轮采样证明。

文件名默认相对于 `models/ptd`，也可明确指定绝对 safetensors 文件。运行时不下载、不安装依赖，不调用 `torch.load`、DCP reader 或 pickle，不自动转换文件。原始 DCP 的 `.metadata` 使用 pickle；必须另外确定文件来源并显式接受信任边界，不能因为允许下载而默认执行。当前没有提供这种 unsafe 转换入口。

Python 入口为 `ptd_advanced.load_ptd_setup(model, path, recipe=None)`；已有受信任张量可用 `configure_ptd(model, tensors, metadata)`。文件必须带 PTD 版本化映射、专家数及配方 metadata；普通 LoRA 缺少这个合同，不能静默按 PTD 加载。实际缺 A/B、专家 hop、投影维度、损坏文件、非有限值、坏网格正常报错。

## H24 开发中的 Compact 适配

同一个专家加载入口现在可按实际 `ptd_num_hops` 元数据识别 stacked-hop 文件，包括 schema1 `int8 per-row` 存储。普通 LoRA 仍需普通 LoRA 加载器。无需凭文件名选择一种算法；旧 EPFL 4／6 步配方保持原来的网格。

Compact 的量化因子按文件中的 scale 字面还原为 BF16，再进入实际逐 hop 投影。零或负的有限 scale 不会被擅改；形状、缺键和非有限值仍检查。文件缩小不等于推理 INT8 或显存变少。该训练合同使用从文件推导的完整 hop 网格、32 区间绝对输出头融合及对应的音频修正，不能套用旧 EPFL 的网格或 HEAD／TAIL 回执。

用户内容 LoRA 和现有 callable 保留。当绝对输出头会覆盖用户已选输出头补丁时，保留用户计算并明确报告部分专家消费；这类报告不能当作完整 Compact 头已经生效。随机微型 Core 的完整4／8 hop、refiner、头、LoRA 顺序与异常恢复已验证。Compact 连续 HEAD／TAIL 尚缺独立状态交接协议，当前入口明确解释这一计算缺口。

2026-10-26 的指定真实样例已完成：固定 Nynxz `d3a95b9` 的 `ptd_h3_ref2va_4step_int8.safetensors`（2,862,719,912 字节，SHA256 `f12f71bace7759c9bab9c9270976e4289f3c72e673ef3c5a8902429abd788b1b`），完整 Ref2VA INT8 底模、双人持杯参考图、832×480、56帧、seed26092137。普通40步与Compact4步的实际条件、初始latent和噪声摘要相同；Compact完整4 hop、专家、头和原生refiner消费报告为完整，前后普通1步诊断的最终视频和音频latent逐位相同，所有forward/hook归属恢复。

该次主采样实测普通40次前向277.25秒、Compact4次48.64秒，Compact另有19.891秒设置成本；两片均完整56帧，原生PCM74400样本，保存器明确补齐至74667样本。Compact成片SHA256为 `98df53ff8fcdcc4adf30373fe49235729bef9f7c06dc15f8a1f1c095e54f53ea`。这是同机器顺序对照，算法与网格不同、冷热顺序未平衡，不能写成普遍加速或显存收益。2026-10-27，指定对照及Compact候选的画面、声音已获完整人工看听接受；不扩大为其它底模、任务、8 hop训练资产、冷恢复或双采资格。候选工作流由真实Core schema校验导出，公共图本身尚未另做GPU队列运行。

固定来源、完整代码许可和模型条款分别见 [H24 来源说明](../THIRD_PARTY_NOTICES/H24-SOURCES.md)。本节是开发候选能力，不扩大上述 v1.102.0 已发布样片的验收范围。

## 发布的两套配方

| 配方 | 完整视频共享时钟 | 专家数 |
| --- | --- | --- |
| `h3_4step` | `.999, .947, .857, .667, 0` | 4 |
| `h3_6step` | `.999, .968, .923, .857, .750, .545, 0` | 6 |

这些是发布的固定十进制网格，不重新计算近似 shift-6 曲线。完整单采仍运行整个网格；另有[同画布连续 HEAD／TAIL](H21_PTD_CONTINUOUS_EXP.md)，4 步按全局专家拆为 2+2、6 步拆为 3+3，不重新加噪或从零开始。这不是旧 LOW／放大／HIGH 4+4，也不支持用普通 x0 冒充连续状态。

音频物理时钟的上端同为 `.999`，相对 shift 为 `.25`，不是上端 1 的固定 3/12 原生时钟。每个 hop 传入其对应 native audio shift，并在已添加专家的音频输出头上乘物理时钟导数。原噪声先以 float64 乘 `.999` 再转 float32，跨 hop 状态始终 float32；网络自身的计算 dtype、FP32 时间／输入／输出岛分别保留。检查报告中的实际时间与专家调用，不以节点数代替执行。

## 用户组合与生命周期

加载器使用共用原权重的 MODEL clone，保留原 LoRA 顺序、强度和已有 model wrapper。不凭模型名字、裁剪／量化标签或未知类名拒绝用户生成；非发布秩、不同来源字段、参考 buffer 与当前底模数值不同等分别警告。真实映射、shape 或缺失 API 仍是输入错误。

不同 hop、独立 Q／K／V 的秩可以不同：各对 A/B 必须自身配对，并逐个满足实际目标投影的输入／输出维度；无需为方便堆叠而强制同秩。不属于作者发布形状的组合会保留警告，不改用户所选因子。

原生已识别 DiT／attention 使用 PTD 所需的分次 BF16 调制、残差和 split-half RoPE 次序；未知现有 forward 保持原调用并记 partial，不清除用户注意力。MLP 的融合 SwiGLU 可能绕过 fc2 模块；PTD 在可识别的请求私有实例 dispatch 中显式调用 fc2 来应用当前专家，不改全局 Core 函数，也不设置共享 `comfy_force_forward` 标志。Q／K／V 因子分别加在原 fused QKV 的对应第三段，SwiGLU 的 B 行从源 `[up,gate]` 映射为原生 `[gate,up]`，输出头保持其完整矩阵，系数为 1。

原生文字预处理只保留原 Qwen embedding，使 refiner 每 hop 重算；未知自定义预处理保留并报告可能缺失的逐 hop 覆盖。线性输入／原模块权重准备／已有 hooks 继续执行。专家只保存在私有 CPU 张量中，按当前层／输出 dtype 临时搬运，不引入 prepared-weight 或 GPU 专家缓存。

原生 forward 本身未改，并不证明它调用的实际调制或融合激活 helper 未被用户替换。发现这类未知调用时保留完整原路径并报告 partial，不绕过用户逻辑来强行套 PTD 数学。时间精度修正也不能覆盖已有 pre-hook 已经修改的 AdaLN 输入。

每次采样开始和正常返回前，读取所选全部 hop 的私有 CPU 因子及其结构并核实际内容 SHA。最后一步即使只改了已使用的早期专家，也不能发布完成记录；失败后恢复原字节可以重试，不靠永久张量版本号拒绝合法恢复。原计算或取消已经抛错时保留其原错误，不让第二个完整性错误覆盖它。完整扫描会增加真实 CPU／内存带宽成本，尚未单独测量这部分开销，不能据扫描机制宣称低成本或提速；上文指定 Compact 对照的耗时仅适用于该次完整执行。

临时 hook 和实例方法由请求 ContextVar 及真实网络身份租锁约束，异常后恢复；其它普通线程不继承 PTD 专家。这个租锁不保证第三方在锁外进行 ModelPatcher 权重加载本身并发安全。报告中的 `completed` 只指固定 hop 递推返回；实际 wrapper 和专家未观察到时仍是 partial，不能冒称所有专家已运行。缓存身份始终 `portable_cache_identity=false`。

## 当前验证与未验证

实际安装的 Core／Kitchen CPU 微型模型已验证 4／6 步完整 AV、真实 A/B 投影、原生一层 token refiner 每 hop 重算、BF16 网络与 FP32 岛、物理音频导数次序、实际普通下一请求逐位恢复、嵌套 hop、普通其它线程、已有 LoRA／wrapper／自定义 attention 保留与异常恢复。随机微型张量不是训练权重，也不是成片画质资格。

截至 2026-10-22，原精度底模和 4／6 步专家已完成安全转换及独立源字节核对。已实际完成完整 4 步 T2VA 与同画布连续 2+2：两份最终 raw AV 逐字节相同，两份 120 帧、1024×576、24fps 的完整视频／音轨／PTS／strict 检查通过。证据为开发目录 `r4-fourstep-trained-verification-v1.json`；这是当时的机械验证，不单独代表人审或公平速度测试。后续指定样片的人审接受按下述发布记录分别认定。

6 步完整与连续 3+3 也已在另一个新私有准备件上实际完成：原始 AV 逐字节相同；实际专家投影为完整 1260 次、HEAD／TAIL 各 630 次；两片完整 120 帧、1024×576、24fps 的 RGB／PCM／PTS／strict 检查通过，MP4 均为 `0a8d6e888d18e05a54f45602550e464e9013adde6866231ab1b75fd194312aa7`。证据为开发目录 `h22-ptd6-media-v1/actual-v1/collection.json`，完整源／Core／模型末尾校验和自有进程退出分别有原始记录。

原 2 秒只读 history 查询超时导致的失败记录保留；新控制器只恢复 GET 轮询，不重提采样。上述机械对照不代表 native40 对比、公平速度测试、图像条件、其它底模组合、冷恢复、learned 放大或人工画质。v1.102.0 发布时已另行接受指定来源的 PTD 完整4／6步及同画布2+2／3+3样片，公开JSON保存在已审目录；这不增加其它任务、模型、冷恢复或性能资格，也不需要重做人审。未验证组合仍按用户选择委托，不借用作者 GH200 用量作为最低显存，也不把作者成功称成本机通过。

## 固定来源与归属

本实现依据固定 [PTD-inference `43ae22c3`](https://github.com/vita-epfl/PTD-inference/tree/43ae22c3c94f8a24cc5e82ca82e4e3a5f736de85) 的公开合同独立适配。官方原生 ComfyUI 资格仅为其未剪枝原底模的 4 步 T2V，6 步及图像条件的整模型资格并未包含在同一报告内。[官方说明](https://github.com/vita-epfl/PTD-inference/blob/43ae22c3c94f8a24cc5e82ca82e4e3a5f736de85/docs/COMFYUI.md)

MiniMax 提供原始 MiniMax-H3；MiniMax 与 Hugging Face 的源组件归属保持不变。EPFL VITA Lab 提供 PTD 训练专家、逐 hop 适配和相关推理实现。代码采用 [Apache-2.0](https://github.com/vita-epfl/PTD-inference/blob/43ae22c3c94f8a24cc5e82ca82e4e3a5f736de85/LICENSE)，原项目 [NOTICE](https://github.com/vita-epfl/PTD-inference/blob/43ae22c3c94f8a24cc5e82ca82e4e3a5f736de85/NOTICE) 另记上述来源。T8 新增安全读取、作用域、未知委托、实际计数与实验节点；没有直接安装上游全局节点包。

底模、PTD 权重与输出另受 MiniMax H3 Community License Agreement 约束，不由代码许可证替代。固定 [PTD 权重说明 `8a70e31c`](https://huggingface.co/epfl-vita/PTD-weights/blob/8a70e31c82a7ce6520dbfa334ebbe633780f85ed/README.md) 列出 4／6 步 DCP 目录；权重体积不是本机内存／显存需求测量。发布或分发具体权重／派生物前需复核相应模型条款与 NOTICE。
