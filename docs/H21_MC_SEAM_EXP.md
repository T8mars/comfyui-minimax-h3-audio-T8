# MC 接缝实验：三个独立算子

这是可选实验，不是对已有双采／长视频接缝的默认替换。三个节点已在工作区追加注册，均默认关闭；
关闭返回输入原对象，且不执行采样。v1.102.0发布独立算子，不自动接入导演台，不重启用户服务。

2026-10-23下述固定HIGH-only与五个完整双窗口来源成片已获人工通过，未转授冷恢复、任意组合或性能资格。完整双窗口实测使用私有显式集成适配器，该探针不随包发布；不能将仅含原Loop节点的图冒称会启用MC。公开节点仍为下述显式外置算子，下文“待审”保留其历史时点。

2026-10-22 的前期验证实际完成 HIGH-only OFF4、Sigma Tail ON4、额外 Refine2 和历史 Whole 四组；
前三组共 10 次真实前向，Relay 路由分别 200／200／100。四片各 192 帧、896×448、24fps，
全部 RGB／PCM／PTS／strict 与独立原始 latent／mask／noise 检查通过。开发证据为
`h22-r7-media-review-v1/actual-acceptance-v1/verification.json`；这不是完整 LOW4＋learned3D＋HIGH4 重跑、
冷 TAIL、内核、性能或人工画质资格。Sigma ON 的音轨实际改变；Refine RETURN 与新 OFF 音轨逐 PCM 相同，
不能把两者都描述成“保留原声”。旧接受片和默认接缝均未改。

同日，私有 Relay 保留修复副本也已完成同四组：真实 Relay 调用 200／200／100，
768 帧完整音视频核验通过。与上述原运行比较的 9 组原始数据、33 对张量均为相同
dtype／shape／原字节，误差为零；这只覆盖指定 HIGH-only 配方，不扩大到完整双采。
收片比较器误把 Whole 的“未观测采样”说明当文件，已在独立 v2 收片工具修复，原失败保留。
Whole 不因此获得采样器认证。证据为 `h22-relay-preserve-media-v2/actual-verified-v1/verification.json`。
该副本没有热更新用户 Core；其后的完整路线回归结果见下段，冷恢复及人工画质仍须分别记录。

同日进一步完成原 8 秒两窗口的完整 LOW4→真实 learned3D fp16→HIGH4 路由回归，
每窗口各四次 LOW／HIGH 前向，总计16次原生前向、800次 Relay 路由；四阶段实际 Sage 返回计数
3200／11600／3200／11600，无 fallback。六份阶段张量和原音频合同读回通过，声音生成区在 HIGH
继续演化，锁定上下文差值为零；续段 LOW 来自前段实际成片，HIGH 仍保留原7单元前缀规则。
最终192帧896×448@24、全部RGB／PCM／PTS／strict及原源／模型／后端末尾校验通过，
审计为 `h22-accepted8-full-media-v1/audit.json`，原张量读回为
`h22-accepted8-collect-v2/actual-stage-payload-v1.json`，已独立整份重推。
这次没有启用三个MC算子：它补齐的是 Relay 保留修复的完整路线回归，
不能据此宣布 MC 三算子完整集成、旧接受片逐位相同、公平性能或人工接缝验收。
`high_input` 保存于 HIGH 前缀锁定之前，不能当作锁定后的独立输入见证。
Relay输入193按原规则归一化为209计划帧，与192交付分开；旧193硬编码收片误判已保留并窄修。
该路由保留修复已移植当前工作区源码，用户运行中的服务和所有冻结批次没有热改。

同日随后完成了新的五组完整两窗口对照：OFF、Sigma、Refine、Whole、KeepPin，
每组均实际走 LOW4→原学习型3D→HIGH4 两次，交付192帧896×448@24。
OFF／Sigma／Whole／KeepPin 各16次原生前向、800次 Relay；Refine 另有显式2次细化，
合计18／900。五组阶段张量与原音频合同读回、960帧完整 RGB／PCM／PTS／strict、
整批源码／模型／后端校验及隔离进程退出均已核对，私有114组索引已独立整份重推。
证据为 `h22-MC-review-v5/root-actual-third-batch-closure-v1.json` 与
`h22-MC-review-v5/actual-five-index-v1/merge.json`。

本例 Whole／KeepPin 的均值校正确有触发，spread／detail 条件未触发，不能说三项都应用。
Whole 原算法会改变首7单元；KeepPin 变体保留这7单元，不把两者混成同一算法。
Sigma 的解码音轨与新 OFF 不同；Refine／Whole／KeepPin 的解码 PCM 与该 OFF 相同。
这仍不认证冷恢复、任意组合、公平性能、旧接受片逐位一致或视觉／听觉质量；Whole 完整
公式没有另做一次独立数学重放。五片已进入待审数据，最终是否改善接缝仍须整片人工对比。

来源为 NikoDemon80 的
[ComfyUI-H3-Motion-Context 固定提交](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/tree/518c083d11d80fa5920222ad89889924ee4401c9)。
这是有明确改动的 GPL3 改编，不是 clean-room。完整原许可和作者版权逐字保存在
[许可](../licenses/MC_MOTION_CONTEXT_GPL3.txt)，来源与文件 SHA 见
[NOTICE](../licenses/MC_MOTION_CONTEXT_NOTICE.md)。作者链式测试结果不等于 T8 测试成绩。

## 先核历史片，而不是先承诺修好

实际串行读取原人审接受片、三个旧失败片及 B/C 审片候选，共六份完整 MP4：
每份 192 帧／24fps，完整解码 RGB、逐视频 PTS、解码原生 float PCM、音频 PTS，
没有重编码、裁音、VAE 编码、模型加载或采样。原接受片 SHA 仍为
`3ff583bc817dd845fa288aaf0b16c2d2be24a6ce282f50c1b37a823edc91ad73`。

已有 bounded RGB 算子对这六份均以 `abstain_no_material_drift` 保留源像素。
原接受片边界 MAD 约 0.01508，旧 high-prefix／context39／drop-guide 失败片
约 0.02284／0.02911／0.02445；这只说明这些失败不能简单当成全局均值偏色，
不证明新 latent 算子会更好，更不能以边界数值排序冒充真人偏好。

采用真实解码纹理构造、意图已知的独立拒绝对照：小恒色阶 0.012 可以通过旧
bounded RGB 的纹理门；明确硬切 0.8 和 SDR 范围内单帧闪光 0.16 都保持原像素，
后者实际命中 transient-flash 拒绝。它们是受控构造，不是假称历史片中拍到了闪光。
初版对照闪光超 SDR 范围，原报告／脚本保留；第二版只修对照以真实检查闪光门，
没有放宽生产阈值。

B/C 原音频压缩包、完整解码 PCM SHA 与 >=8kHz 分帧 FFT 能量均相同；
PCM SHA 为 `d20c947c2e53bb99349da19d1cc4612dee4076d5afb6aa04203cd6c58ecf47f0`。
解码有 257024 样本／声道，含 AAC padding，未伪称刚好 256000 或偷偷裁剪。
频谱值只记录是否被改动，不是声音质量分。完整本地证据见开发收据
`r7-historical-media-evidence-v2.json`，并非新的质量通过片。

## Refine Blend：只混显式提供的结果

`MiniMaxH3MCRefineBlendEXPT8`：

- `original` 接原采样完成 AV LATENT；`refined` 接用户显式添加的细化采样器结果。
- `enabled=false` 时 LAZY 输入不请求 `refined`，不为一个关闭的实验支付额外采样。
- 节点本身始终 0 NFE；上游细化采样的真实 NFE／耗时不计入本节点，不能说总成本为零。
- H3 时间单元为 `1,4,4,4,4` 原帧周期。`hold_frames=22` 正好保留前 7 单元；
  随后的权重按各单元**自己的起始原帧**计算。渐变首点权重也为零，因此默认有 8 个零权重单元。
- `blend_frames=0` 是用户显式选择的硬切，不保证无跳变。`clip_index=0` 是首段，整片采用 refined；
  不自动寻找 Chain 分组或覆盖别的节点 widget。
- 默认 `audio=original` 保留原音频 tensor 对象。只有明确选择 refined 才换原声音流；
  不把音频 latent 混合宣称为可听域交叉淡化。
- 原 LATENT 全部元数据对象保留；原／细化视频实际 shape 必须一致。
  AV pair 是 list／tuple 子类时，其自有字典／slot 状态也浅保留；先获得新容器再换流，
  不会因为自定义 copy 返回自身而改写原 pair。无法重建真实两流时明确报容器 API 错误。
  合法浮点精度提升／复制使用原公式的 PyTorch 行为，不因 dtype 不同硬禁；真正无法计算的设备组合由实际 API 报错。
  显式选择 refined 音频时不混合、不强制改回旧音频的 dtype／通道／时间尺寸；其下游兼容性不伪称已认证。

这是两个 latent 的显式混合，可能产生重影、身份或运动退化，不能因为保住 pin 就宣称安全。
它没有恢复此前失败的 endpoint-bridge 生产路径，不接入已接受 runner；必须保留原片并单独审片。

## Join Match：原算法与保头变体必须区分

`MiniMaxH3MCJoinMatchEXPT8` 只计算采样完成后的 latent 统计，不会自动接入任何 runner。
`context_latent` 接真实前段 AV；`context_length` 须与真实采样时使用的 pin 长度一致。
目前只有用户显式长度，不伪造对已采样 pin 的身份认证。

- `source_whole_clip_exp`：保留原算法对**整片，包括 pin** 的均值／空间 spread／细节修正。
- `keep_pin_T8_exp`：明确的新 T8 变体；同样计算候选，但把前 22 原帧对应的 7 单元恢复为原字节。
  这不等于原作者算法，也不能保证保头与后部修正的交界更自然。
- 测量使用五单元周期同相位：前段末单元与当前 hold 后第一个同相位单元，
  再与后五单元比较。太短、零强度、首段无上下文时原对象直通。
- 原作者 auto 的 `0.48` 阈值及 `.9..1.1` spread/detail 增益属于算法，不是 T8 的通用画质校准。
  多 batch 扩展逐 batch 独立计算，不把原源码 batch0 的统计广播到不同样本。
  允许一份前段上下文显式广播到多个当前样本，分别与其自身统计比较。
- 原声 tensor 和元数据不变；不对模型名／LoRA 做准入限制。实际张量尺寸、NaN、
  空数组及不足两个空间元素时却要求 unbiased spread 等真实数学错误仍报错。
- 不从 latent 均值推断一个真实曝光闪光、硬切、遮挡或人物运动应当被消除。
  若局部纹理／身份／运动或原声高频变差，拒绝候选，保留原片。

普通有限统计保持固定源码的原运算；只有平方／归约因极大有限输入溢出时，
才改用缩放等价统计，避免把可表达的 FP64 结果误报成 JSON 错误。
真实均值差、统计比值或输出超出实际 dtype 时仍报告具体 Join Match 计算错误，
不截断输入、不允许 NaN JSON、不伪造质量分；这也不是零开销承诺。

## Sigma Tail：只产生显式的新表

`MiniMaxH3MCSigmaTailEXPT8` 关闭／零强度／零尾步时保留原 SIGMAS 对象。
开启时复制表，保持第一层、尾部以前以及末尾零值，按原算法
`new = anchor * (old / anchor) ** (1 / (1 + strength))` 改指定尾部。
`strength=-1` 会使公式分母为零，属于真实数学错误；其它仍能实际计算的负指数／
非降序用户表保留执行，报告实际指数、原排序及原算法逐尾单元保序修正次数，不能仅按 widget 意图拒绝。
负 noise、零的负幂或实际溢出仍说明真实计算错误；不设任意步数／模型大小上限。

这张新表**不是原训练表**，节点不自动改 FastH3、HyperFlow、PTD 或标准 4+4 的表，
不换采样器／步数。显式接入某个采样协议后如该协议确实要求固定网格，仍须遵守其真实接口；
“用户承担未知兼容风险”不代表可以伪称不同数学是同一张训练表。
如果用户把新表接入联合音画采样器，重新采样可能同时改变声音和画面；这不是原声保持保证。
只有 Refine／Join 默认选择原音频 tensor 的纯后处理路径具有上述音频对象保持合同。

## 验证边界及后续 GPU 单变量计划

CPU 合同检查应含：真实 Core V3 schema 与 LAZY 输入、OFF 原对象、原生周期大整数、
22→7 字节保持、元数据／默认原声对象保持、B1 与固定源码数值对照、逐 batch 独立、
坏真实维度／NaN／SIGMAS、实际 Core 取消与下一请求恢复。不是训练权重／内核／画质验收。

上述已完成的完整4+4对照使用统一 controller 单独准备的新 opt-in 图，没有修改已接受示例：固定原 0.4MP／8 秒两窗口、
22 context、4+4、已接受 LOW picture 和原声；每次只比较一个算子及 OFF 控制。
Refine 必须显式记录额外上游采样与 seed，Sigma Tail 必须单独列出新旧完整表，
Join 两种 pin 策略不能混为同一实验。核 192 帧／24fps、全部逐 PTS、原声 PCM／高频、
实际 NFE、pin latent、纹理与完整后段，再从接点看至片尾由用户判断。
上述 HIGH-only 证据不能替代完整配方资格；新五组只补齐其明确范围，`quality_accepted` 和
`portable_cache_identity` 仍为 false，不能用笼统 `GPU_validated=true` 掩盖未验证范围。

## 最小 CPU 检查和 API 用法

仓库内的 `tools/check_director_d1_cpu.py -q tests/test_h21_mc_seam.py`
检查真实 AV 运算、Core schema、LAZY 状态和 API 图校验，不启动服务、不需要模型权重。
完整回归还需独立 review 与前述五份接缝门禁；不要把重复运行次数累计成测试数量。

在主线已注册三个节点的 Core Python 上下文中，可通过实际节点映射使用以下调用。
`original`、`refined`、`previous` 是上游真实 AV LATENT，`sigmas` 是用户当前明确选定的表，
不是从 MP4 伪造的恢复状态。此片段本身不会采样：

```python
from nodes import NODE_CLASS_MAPPINGS as N

# 关闭时保留原对象，连上的 refined 在真实 Core 图中应使用 LAZY 路由。
same, off_report = N["MiniMaxH3MCRefineBlendEXPT8"].execute(original).result
assert same is original

blended, blend_report = N["MiniMaxH3MCRefineBlendEXPT8"].execute(
    original, enabled=True, refined=refined,
    hold_frames=22, blend_frames=22, audio="original", clip_index=1,
).result
joined, join_report = N["MiniMaxH3MCJoinMatchEXPT8"].execute(
    blended, enabled=True, context_latent=previous, context_length=22,
    variant="keep_pin_T8_exp", tuning="auto",
).result
new_sigmas, tail_report = N["MiniMaxH3MCSigmaTailEXPT8"].execute(
    sigmas, enabled=True, tail_steps=3, strength=0.5,
).result
```

这些调用演示各接口，**不表示建议一次同时开启三个算子**。实际对照应逐个开启；
`new_sigmas` 只有用户显式接入后才影响后续采样，不会回写任何现有节点。
Core API JSON 的 `class_type` 就是上面三个映射名，`inputs` 使用相同参数名；
`LATENT`／`SIGMAS` 输入必须接上游 `[node_id, output_index]`，不能上传假的完成 receipt。
实际开发若尚未注册，不能直接拿此片段声称用户服务已可调用。
