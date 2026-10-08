# RES外置EAV／Prompt Relay（EXP）

新增独立 `MiniMaxH3RESEffectsBindEXPT8`，只给既有RES History的MODEL／SAMPLER／SIGMAS
添加采样前的精确阶段描述；不采样、不加载模型、不替换采样器、不改旧节点schema。

接线：外置Relay Plan／Conditioning的MODEL → RES History → RES Effects Bind →
既有Stage EAV Apply → BasicGuider；配对Relay CONDITIONING直接接BasicGuider。
NOISE和原始AV latent仍直接接RES完成态采样器；采样后接Stage EAV Audit，
需要显式冻结最终数据时接既有Stage Save／新的RES Stage Load。

EAV `disabled`、`report_only`、`apply_exp`均使用原完整轨迹的绝对video progress。
RES恢复的实际覆盖只计算剩余sigma区间；不会重新从0计时或重新触发前四步。
检查点仍绑定原seed／条件／音画输入／遮罩／effect配置和真正solver历史。
改EAV参数或换成未配对条件不能复用旧边界，不关闭原有坏SHA和Euler守卫。

身份投影只在检查副本上移除已认证的效果层。原版Sol的实际选定Relay delegate
另由RES-only来源适配器认证；未知delegate正常保留运行，但不授予冻结复用或恢复资格。
原Sol包装器通过`**kwargs`转发refs／keyframes时，只读核其已认证的原生Core构造器
参数；不更换全局构造器，不对任意`**kwargs`包装器放行。

新CPU行为覆盖实际完整8步／第4步保存／余4恢复、EAV两模式＋真实Relay路由，
两槽完整AV tensor与连续结果一致；report-only无效果时与原RES一致。
还覆盖效果配置／配对条件变化、未知delegate保留且不复用、外国构造器不被执行，
以及真实安装来源KJ／FFN／原版Sol的只读身份。未初始化CUDA。
上述CPU证据本身不代表真实GPU效果、完整成片或人审通过；
原版Sol/KJ可能在部分forward绕过selector，必须看实际Audit，不能把非零调用当完整覆盖。
新验收图获人审之前不作为推荐公开工作流，旧图和已通过媒体保留。

真实首轮5秒发现原Sol小尺寸门控下，KJ前向绕过selector：8次完整forward、
EAV／Relay selector均0，明确为覆盖不足；该成片、Stage和旧源码epoch证据保留。
新增RES-only KJ／Sol组合适配保留原KJ投影、RMSNorm／RoPE与原Sol门控。
原Sage内核不支持Relay浮点bias；带bias的查询明确走原Sol已选mask-capable委托，
不声称Sage处理了bias，不静默丢弃mask。Sol门控外的无偏置查询仍调用原KJ内核。
原KJ会就地中心化K，适配只给它私有K副本，避免改掉后续Relay查询和FETA所需K。
这会增加该新分支的瞬时K副本，不构成内存／速度或任意后端质量保证。
CPU定向来源／bias／旧投影／两恢复模式5项通过，旧批次未重跑。

## 当前固定组合的真实画布验收

修正版在两个独立后端分别由画布另存、关闭、从侧栏重开、点击运行。
仍用原Ref2VA INT8／FFN2／KJ Sage／原Sol参数和seed，512×288、5秒；
外置两事件Relay为`apply_exp`，EAV为`report_only`。
连续RES8真实8次前向、400次selector／400次Relay；保存POST4边界后，
独立进程只恢复后4步、200次selector／200次Relay，绝对sigma与完整轨迹后半相同。
两次完成态、两个native输出槽、全部AV张量，以及120帧RGB／PTS、
原生32k立体声160000采样PCM和最终MP4逐值／摘要一致。原365个保护文件未变。
期间画布页崩溃后，已核同一个live任务并在新标签恢复观察，没有再次提交采样。
首轮0覆盖失败证据保留，修正版恢复没有使用该失败边界。

这个结果证明指定配对组合的实际覆盖、完成态与跨进程恢复，不是通用
MODEL／CUDA数值资格或人审通过。EAV报告模式不改变增强系数，不能据此
声称增强画质已通过。该配对恢复证据属于此前源码批次，不能冒充下面新分支的恢复证明。

## 不接 Relay 的独立 EAV

普通 MODEL → RES History → RES Effects Bind → Stage EAV Apply → BasicGuider；
普通 CONDITIONING 与原 AV latent 沿各自输入直接连接，不需要 Relay Plan 或配对 Relay 编码。
只在新 RES 分支认证原 KJ／Sol 组合并安装独立路由；保留原投影、RMSNorm、RoPE、
Sol 门控和已选委托，不制造 Relay wrapper、时间计划或伪 CONDITIONING。
EAV 关闭时不安装此适配，未知组合继续保留运行而不授予恢复资格。

指定原栈的实际画布另存、保存、关闭、侧栏重开及运行已经完成：
512×288、5 秒、完整 RES8，29 个执行节点／43 条连线，400 次 selector、
0 次 Relay、100 次 FETA 测量，2 次 active forward；完整 120 帧 RGB／PTS、
32k 立体声 160000 采样原生 PCM、最终 MP4、两个 native AV 槽与完成态均已检查。

第一条仍用原默认 tau=4，所有 gain=1：路由有覆盖，但没有数值增强。
严格非零增强检查的失败保留，不把它包装成增强证据。
随后仅把私有实验图 tau 改为16，模型、seed、轨迹、时间窗、几何与1.5硬上限不变；
实际 gain 最大1.285774，已确认目标视频增强不再是恒等变换。
节点默认仍为4，16不是新默认或通用推荐；该单变量诊断不是多seed／多尺寸矩阵。
超过硬上限仍报错，不静默夹紧或更换算法。

直接 gain 只作用于目标视频行，但联合音画模型可能间接改变生成音频；
这两条的实际原生 PCM 摘要不同，不承诺启用 EAV 后音轨逐值不变。
当前证明的是这个固定栈的独立 apply_exp 覆盖与完整成片，
不证明更好看、更好听或所有后端可用。

上述同一 tau16 POST4 边界已在第二个独立后端通过真实画布另存、保存、关闭、
侧栏重开及运行验证：只执行剩余4步，空执行缓存，130.368秒完成。
实际 selector 200次、Relay 0次、FETA 100次／2个活动前向；
绝对 sigma 后半段和增强系数与原连续8步相同。
完整120帧 RGB、PTS、32k立体声160000采样原生 PCM、最终 MP4、
两个 native AV 槽和两个 Stage AV 槽的全部 tensor 及其他成员均逐值一致。
检查点本身未改写；新后端只卸载模型内存，服务、历史和持久资产均保留。
这是指定栈的独立恢复资格，不扩大为通用 MODEL／CUDA 认证。
人审与其他 H05 样片最后集中交付；未把默认tau4中性片或旧配对报告冒充本次增强恢复。
