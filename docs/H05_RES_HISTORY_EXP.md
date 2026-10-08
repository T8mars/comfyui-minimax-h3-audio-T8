# RES 历史步边界（EXP）

新增 `MiniMaxH3RESHistoryEXPT8`，输出 MODEL／SAMPLER／SIGMAS 接原生
BasicGuider／SamplerCustomAdvanced。旧节点、旧图、Euler 恢复守卫与 Core
采样器不变。本入口只支持确定性 `res_multistep`、eta=0、非 CFG++、
`native_flow` 和原生 H3 batch=1 的共同音画坐标。

## 用法

- `disabled`：完整 RES，无文件读写；这是新节点自己的缺省，不改变旧采样器选择。
- `checkpoint`：在 `checkpoint_step` 整步更新之后保存一次，之后继续完整采样。
  需要明确开启 `confirm_checkpoint_write`，并使用新文件名。中断发生在该步之前，
  不会产生该边界；发生在之后，已写入的边界保留。
- `resume`：保持总 steps、两路 shift、seed、原始 av_latent／mask、最终条件和
  模型／LoRA／补丁一致，读取该文件并只输出剩余 sigma。不要将一采 x0／部分
  输出代替原始 av_latent；不要开启写确认，恢复不会覆盖原文件。

`run_contract_json` 建议连接现有 NFE Run Contract 编译器，使用实际最终条件、
提示词、媒体映射和报告。`model_contract_id` 声明底模、LoRA 顺序与强度、
注意力和其他补丁，供人阅读；该文字本身不是权重证明。
执行时另外检查实际加载的原始权重存储、原生 LoRA 张量／顺序／强度、选定的
AV 坐标与实际代码。INT8／ConvRot 使用原始量化存储与尺度，不用反量化副本
冒充原件。`disabled` 不扫描这些权重、不读写检查点。
实现也绑定实际 guider 原始条件中的可识别 tensor 内容、seed、运行时类型、
Torch／数学精度和代码。未知补丁或条件继续允许完整采样，但不授予跨运行恢复
身份；应正常重建任务，而不是关闭完整性检查来复用未经认证的检查点。

## 文件及边界

文件位于 `output/MiniMaxH3/res_checkpoints/`，使用相对短名
`*.h3res.safetensors`。拒绝越界、绝对路径、符号链接、Windows 保留名和错扩展名。
上限2 GiB；不使用 pickle，不执行文件内配置。检查点包含运行合约和提示词，
属于私人资产，不应提交或上传仓库。

保存的是更新后的 x、上一预测 `old_denoised`、上一 `old_sigma_down`、完整 sigma
表和原始 processed noise／latent／mask。原 dtype 不变，全内容摘要校验；读取
时确认全局步偏移，恢复 inpaint 原始输入和完整 `sample_sigmas`。Core 的原生
预更新 callback 不能冒充这个整步边界。

临时文件校验／fsync 后，以同目录硬链接原子发布；仅新建，不覆盖。
不支持这种发布方式的文件系统明确失败，没有非原子降级。
实际 Core 的源与已加载代码都必须匹配本次资格记录；更新 Core 后需要重新资格，
不会因此禁用普通 Core／T8 旧节点。

## 已验证与尚缺

12项新 CPU 检查通过，无跳过、CUDA 未初始化。涵盖实际 Core float32／float64
每步对照、新进程第4步恢复、实际 packed 音画及 MASK 输出、输入不变、错 seed／
条件／原始 latent 拒绝、内容／代码变化、越界、写入失败和取消边界。
节点尾追加后，原678个 schema 的全部 JSON 字段及顺序精确保持。

新增保存／恢复两份短图已完成原生 SaveAs、关闭标签、从侧栏重开、画布运行。
固定 Ref2VA INT8 ConvRot 底模、无 LoRA，512×288、实际原生音画、完整8步保存
第4步边界，再只读恢复剩余4步。两次任务约47.085／26.415秒；最终 output 和
denoised_output 的视频／音频四个 tensor 逐值、dtype、shape 精确一致；完整120帧、
24fps、5秒成片的解码 RGB 与 PCM 也一致，检查点 SHA 未改。MP4 的 AAC 尾部
padding 与 FLAC 预览并非逐 PCM 相等，不把两种封装混作原声保留。

上述是前一源码阶段、同一个已加载模型／同一条件缓存的真实 GPU 文件恢复。
之后独立的自动权重身份阶段有10项直接相关 CPU 检查通过，包括原始量化存储、
原生 LoRA 顺序／强度变化和新 CPU 进程读实际边界后只选剩余4步。再新增的
3项混合身份检查覆盖真实 T8 FFN＋已安装 KJ low-memory 工厂、检查时不改执行
模型，以及未知 forward／hook 和原版 Sol 组合仍保持未经认证状态。
这些分属明确源码阶段，不把历史通过全部冒充当前全仓测试。

自动身份阶段的新原生画布任务已完成完整8步并保存第4步边界，耗时约100.953秒；
实际120帧／5秒的完整 RGB、PCM 与原连续 RES 结果一致。当前实际
T8 FFN＋KJ Memory Efficient Sage＋原版 Sol 组合仍不可移植，未排队跨进程恢复。
新增的 RES-only 原版 Sol 结构检查已认证选择器／forward composer 的实际工厂
闭包、原生 `_forward`／RoPE 委托、PackedLayout 构造器及有序 block hooks。
3项新 CPU 结构检查已通过，包含实际 Memory Efficient Sage 组合、安装前后
内容身份保持，以及未知 hook／构造器不执行、不删除、不授予恢复资格。
原版 Sol 文件、实际执行 MODEL 和采样数学未改；未初始化 CUDA、未新增 GPU。
结构报告仍明确 `portable_cache_reuse=false`。后续 RES-only 检查已绑定实际
comfy-kitchen 的公共函数、CustomOpDef、原生冷热包装、注册表选择顺序／约束、
递归计算辅助代码，以及实际 CUDA 扩展文件内容与选定原生导出。没有调用 CUDA
内核、修改注册表或安装替代实现；包／版本名和函数名称不代替内容身份。
Sol 布局 lookup 的真实 owner／segment／sink 边界、Morton CPU／设备缓存的
派生排列内容、当前所选函数的参数缓存也已核验，检查不回写原缓存。
地址与无关旧缓存数目不进入摘要，正常的缓存预热不会改变执行身份。

本源码阶段5项直接相关 CPU 检查已通过（其中一次真实 tiny eager 调用），
覆盖冷热内容身份保持、优先级／约束改变、外国 helper／binary／dispatch
不被执行或认证、错排列／bounds／签名，以及缺失依赖与坏缓存仍保留。
最后一次只重跑错误消息期望不符的单项；此前红证据保留，不把它说成整批绿。
CUDA 未初始化，旧 N01／solver 不重跑。

随后新 RES-only adapter 已绑定原始 KJ SM89 的真实 C++ dispatcher／两个原生
binary、int64-safe Triton quantizer 的实际源码／JIT 设置，以及 RMS／partial
split-half RoPE 的实际 CustomOpDef／native fallback／binary。原生 eager RoPE
CPU 调用前后身份一致，不用普通 SDPA 夹具替代 Sage 计算证明；分支夹具只检查
已安装真实源码和入口内容，不代表本进程 CUDA 或其他架构资格。

还新增实际 processed conditioning 的位置检查：由真实 latent geometry、参考块
及关键帧重建原生 Core PackedLayout，逐值核 position_ids、img/audio 行号、更新
遮罩及 segment 表，连同实际 payload／context 内容接入 RES runtime contract。
只在私有实例调用已认证原生构造器，不调用 Sol constructor hook、不写其 lookup。
尺寸相同但坐标被改、来源时间变更、未知 helper／constructor 与越界预算均拒绝
位置身份；普通完整采样不因未知组合被禁止。

这两组分别取得6项计算链和3项位置的不同 CPU 检查，保留明确源码阶段；
最终仅复查懒加载／反例和全缓存不写的3个直接受影响用例，不累加重复计数或
冒称当前全仓通过。首轮的三红属于 JIT 夹具 AST-only 编译的 call opcode 差异
和未导入 fixture，已用完整原源码 code 与真实 triton.jit、正确 fixture 修正，
没有删源码比较守卫；失败报告保留。CUDA 未初始化，旧 N01／solver 未重跑。
在该阶段，已编译 device-cache 的执行内容和只读 Sol 正规化仍待，portable 仍 false。
上述内容检查不等于
CUDA 数值／速度资格，也不授予跨进程恢复许可。不能删补丁或换裸模型取通过。
改身份实现后使用新的检查点源码阶段；
旧 EXP 文件不静默升级，也不覆盖。

后续独立阶段新增只读 Sol 组合投影：先认证原版结构、实际 raw KJ 计算链和完整
attention owner，再仅浅复制 MODEL／diffusion／block／attention 容器。只在检查副本
移除已认证的 Sol hooks／composer／selector，恢复其已认证原生委托，并通过原有
严格权重与混合 FFN 适配器绑定实际 raw storage、原生 LoRA 字节／顺序／强度及
Sol／FFN 设置。执行 MODEL、原 Sol、全局 PackedLayout、权重、receipt 与缓存
不改；未知 hook／owner 或未认证 Sage 仍不投影、不调用，完整采样路径保留。
该已知真实源码 CPU 夹具的 raw weights 现在可绑定，但整套 MODEL 的
`portable_cache_reuse` 仍 false，不借“权重已绑定”放行跨进程恢复。

同时绑定 Triton 原生运行时的实际类执行代码、参数／constexpr 顺序与缓存属性、
default factory 和原生依赖哈希。只在私有 JIT 计算 CPU 哈希，不写所选 JIT 的
参数缓存／hash／device cache；原生冷热哈希内容身份一致。外国 annotation／
比较对象、类方法或缓存工厂不执行、不认证。实际已编译 GPU cache、binder／
launcher 与内核内容仍独立待验，容器正确不等于执行内容正确。

新增4项投影、3项 JIT 状态 CPU 检查分明确源码阶段通过，并核两个直接受影响的
既有用例；最终源码只复查4个直接受影响范围，不累加重复或冒称全仓测试通过。
首轮一红为普通 CPU Sage 夹具替换真实 package 导致实际 binary baseline 无效；
修正夹具请求顺序，仅失败项重跑通过，生产守卫未放宽、红证据保留。
CUDA 未初始化、0 新 GPU、旧 N01／solver 不重跑。既有 GPU 检查点属于旧源码，
新代码未热注入旧服务；后续需新检查点和真实新进程剩余4步验收。

后续 RES-only 编译缓存检查已接入原生 JIT 身份入口：认证固定 compiler／backend
源码、实际类函数和生成 binder 的代码／默认值／依赖，以及实际 libtriton 扩展的
specialization 导出和文件内容。检查原生 device cache 的参数特化键、源 signature／
constexpr／attributes，并核 metadata、IR、cubin 在内存与磁盘上的一致性。
只生成一个私有 CPU 参数绑定函数作比较，不运行它、不初始化设备句柄、编译或
填充所选 JIT 缓存。未知 binder／parser／异步条目及外国比较对象不执行。

稳定 producer 身份与实际已观察 artifact 清单分开：普通冷热缓存增长不进入
MODEL 摘要；可选只读 collector 返回文件内容摘要供后续检查点绑定。缓存文件
自洽不证明其编译来源；native compile key、已加载 launcher／句柄及实际 GPU
程序资格仍待，因此 `compiled_device_cache_content_qualified` 仍 false。
CPU 结构测试明确使用不可执行的假 cubin 字节，仅验证拒绝篡改与不触发执行，
不拿它作 CUDA 二进制资格。该轮4项新缓存测试＋4项直接受影响既有测试通过；
补比较对象保护后只复查2项直接范围，两个源码阶段分别留证，不重复累加。
旧 N01／solver 不重跑，CUDA 未初始化、0 新 GPU；跨进程实际验收仍在下面范围。

已另接入只读 native launcher 检查：固定实际 NVIDIA driver／build／cache／knobs
源码和类函数，先认证类描述符再读取实例。对当前 scalar／pointer 签名，只调用
已认证原生 C 字符串生成函数，不构造 CudaLauncher、编译、加载扩展或启动设备。
检查时另外确认全局 launch／kernel-load HookChain 为空，未知 hook 不调用、不删除。
已实现实际已加载 launcher 的字段／元数据一致性、native export 所属模块和文件、
生成 C＋平台形成的原生缓存键、扩展内容摘要，以及 handle 类型观察。冷 program
不为检查初始化句柄；tensor descriptor／未知包装器未授予该路径资格。

两项新 CPU 检查通过：真实 C 源生成／原生冷对象未改，与 foreign generator、
类 getter、全局 hook、字面映射及 launcher 不执行。另核两项直接既有范围；
补类型／描述符守卫后，仅复查两新项和一直接范围。分阶段留证，不重复累计。
实际 warm launcher／扩展和句柄的 GPU 正向证明仍待；CPU 构造的对象只作拒绝
反例，不作已运行设备证明。compile-key、实际 warm 资产绑定及跨进程恢复未完成，
`compiled_device_cache_content_qualified` 继续 false，旧服务未载入这一源码阶段。

后续已补独立 native compile-key 重建：认证实际 compiler／knobs／package 文件、
libtriton 和 bundled ptxas 内容，使用完整 `--version` stdout，不用短版本号代替。
只作版本查询，不编译；保留原生 tuple 配置、实际 defaults／kwargs、extern 库内容，
以只读私有 JIT hash 重建 source／backend／options／environment 组合与缓存目录键。
检查不调用所选 JIT 的 mutating cache_key、binder、driver 或 kernel。
实际 Windows 空 tensordesc metadata 以及 native 文本读入时的换行转换已处理；
IR 原磁盘字节仍单独 SHA 绑定，不改写缓存文件。未知环境函数／default descriptor／
listener 或可执行 kwargs 不执行；版本查询失败不给恢复资格，保留普通采样路径。

两项新编译检查和直接受影响既有检查分源码阶段留证；最终仅复查五项直接范围，
不相加冒称更多新测试或全仓通过。首轮 native PyBind 导出 owner、空 initializer，
以及测试导入 runtime.driver alias 的失败均保留；不弱化编译来源或 binary 检查。
另以一份本机真实 KJ SM89 cache 核 source 及编译键、metadata、IR 和实际 ELF cubin
内容：C=128／BLK=32，key `4eb9bd7e…ef206`；不是 CPU 假 cubin，也没有加载／
运行该 GPU 程序。只该固定已有文件资格，不扩大为全部几何或已验证设备句柄。
原生 compile-key 门已完成；上述早期轮次的“待补”属于历史状态。
实际 warm launcher／handle、检查点资产绑定和跨进程 GPU 仍未完成，
`compiled_device_cache_content_qualified` 与 portable 恢复资格仍 false。

后续已把只读 observer 从实际 MODEL 经原版 Sol／KJ／RoPE 接到 JIT，并新增
编译资产边界清单：在 POST-step 检查点中单独保存 producer 摘要、实际编译文件
与已观察 launcher 内容，不将缓存预热或进程地址混入稳定 MODEL／初始 run contract。
恢复入口先核清单摘要、当前 cold producer 与原生缓存目录中实际文件的内容；
拒绝错根目录、链接、篡改与未知缓存工厂／manager，不导入清单路径、不初始化
设备句柄或执行配置。资产完整性检查本身仍不是 CUDA 或 portable 恢复证书。

该源码阶段3项新 CPU 检查和3项直接受影响的 Core 保存／恢复检查通过；没有
复跑旧 solver／N01。另将同一实际 ELF cubin／IR 清单存入数据-only serializer
夹具，并在冷 cache 状态核对原文件，文件及检查点 SHA 不变、0 NFE。这不是
真正的 GPU 轨迹边界；新源码真实 warm launcher 检查点与 fresh GPU 余4步仍待。

后续已在新的原生画布另存、关闭重开并实际运行完整8步：512×288／120帧／
24fps／5秒，115.354秒，实际POST-step4检查点保存两个KJ Q/K编译程序、
真实warm launcher扩展与已加载句柄类型。完整RGB／PCM和两个输出槽的四个
最终AV tensor均与原连续采样逐值一致；不是partial x0或CPU假binary证明。
该实际检查点仍为nonportable，不借已加载句柄自动认证数值或跨进程恢复。

实际GPU诊断另定位到CUDA RoPE的call rule被错误绑定为constraints.py的函数：
它实际属于已固定源码的CUDA backend。RES-only检查器已改为认证真实源／globals
owner，不调用rule、不制造CUDA可用性、不修改第三方kernel或执行MODEL。
新增正反两个CPU检查及三个直接受影响已有检查通过；原红证据保留。
这项修正发生在上述GPU终态之后，未热注入8972，不能把旧GPU片当新源码通过。
后续需固定源码的完整已识别MODEL边界及真正fresh GPU剩余4步验证。

后续新增独立 `scoped_RES_history_content`：只有已识别的原始权重、实际 Sol／
KJ SM89／RoPE／JIT 计算内容、完整 attention 路径和只读正规化都验证后，才标记
内容可重建。恢复时还必须验证实际条件位置，以及两个量化程序的原生编译文件、
producer 和已观察 warm launcher。运行合约、seed、sigma、原 noise／latent／mask
和所有 solver history 仍逐项检查。未知 hook／权重 owner 保留，不授予恢复资格。
这是 RES 专用内容边界，不把 `portable_cache_reuse=false` 翻为通用 Stage 或 CUDA
数值证书；不更改原 Sol、KJ、Core、执行 MODEL 或采样数学。四项新 CPU 检查和三项
直接相关回归通过，无跳过、CUDA 未初始化。

该固定源码阶段随后完成真正的两进程 GPU 验收。完整8步任务约161.559秒，
保存POST-step4边界；独立新进程重新加载同一原始模型和条件，只执行剩余4步，
约108.803秒。两边均为原生画布另存、关闭侧栏重开和运行，22执行节点／29连线，
没有复用节点执行缓存。完整120帧、24fps、5秒的RGB、解码PCM和两个最终输出槽
中的四个float32 AV tensor与原连续采样逐值一致；8,738,352字节的检查点未改。
这只批准这一固定FFN／KJ SM89／原版Sol计算链的RES历史恢复，不批准任意补丁、
其他GPU、通用Stage或CUDA内核数值资格；旧Euler恢复守卫保持。

临时PreviewAudio文件不应当作为长期交付资产。上述两个实例旧启动参数共用
临时目录，Core启动清理后，第一实例的临时FLAC已缺失。原文件当时的完整音频
审计与SHA记录保留，不能宣称旧FLAC字节仍在。持久MP4、检查点和最终AV数据
保持且已核验；当前恢复实例的FLAC另存了逐字节相同的持久副本，后续独立实例
使用各自临时目录。没有重新采样、伪造旧文件或覆盖其失败记录。

后续已新增 RES 专用完成态采样／读取入口并完成实际剩余4步保存、0NFE读取图
的原生保存重开和完整5秒音画／两槽四tensor核验，详见
[RES完成态保存读取](H05_RES_STAGE_EXP.md)。这不授予通用 MODEL／Stage 或 CUDA资格。
外置 EAV／Relay 的新恢复资格与通用可移植 Stage 仍未完成，
人声、口型与画质仍待集中人审。CPU内容身份验证不是任意GPU内核／组合效果保证。
新 `tools/prepare_h05_res_history_workflow.py` 只创建指定的新文件，不排队；
保存／恢复模式需显式选择，同一检查点必须使用同一模型声明和 run contract。
未获人审的验收图保存在本机，尚不作为推荐公开示例。
这是同一轨迹的历史恢复，不是 LOW→放大→HIGH 双采，不是 vLLM 两个独立 solver
复现，也不承诺加速。外置 EAV／Relay 旧入口保留，不能据此声称新恢复组合已验收。
