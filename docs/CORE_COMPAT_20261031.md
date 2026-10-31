# ComfyUI新旧版本兼容与Issues／PR核对

2026-10-31，本机ComfyUI `7c4390a6f70cd8eb41014d6af52969397a752fc4` 与当时上游HEAD一致。另用本地Git中的原始源码检查了v0.43.0（`426f5ee`）和v0.41.1（`45bf90d`），没有修改用户Core、安装包或下载模型。

## 修复的两处兼容问题

**原生SigmaShift被整个Core文件哈希误拒。** 三个版本的`MiniMaxH3SigmaShift.execute`实现相同，但同文件新增其它节点、文件换行或辅助代码更新会改变整文件SHA。之前会提前报`Core SigmaShift source is outside the audited sampling factory`，影响组合采样、保存／恢复等路径。

现在继续核对实际Core来源、类和父类、作用域、执行函数的精确源码，以及按当前文件重新编译后的实际字节码。整个文件的真实SHA仍记录在缓存身份中，只是不再作为“是否允许这个未改变函数”的判断。函数改变、执行方法／hook替换、字节码与文件不符仍不能借用可移植缓存资格。没有更改sigma、采样步数、音频或用户LoRA。

**显式导出非持久buffer的状态字典被误判损坏。** 对应[#32](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/32)的错误类型：`MODEL nonpersistent buffer unexpectedly appears in persistent state`。部分原生／自定义`state_dict`导出hook会明确加入已注册的非持久buffer。动态加载改变dtype后，不能仅凭“它出现在导出字典”就拒绝。

现在只对确实匹配已注册live buffer或原始backup内容的导出项，还原原始字节并保留该导出键；旧版没有导出的非持久项依旧不伪造键。未知路径、形状错误、NaN及伪造导出内容仍报错。回归使用真实Core Dynamic加载器和原生VAE的状态导出hook，核对加载前后身份、live对象和backup均不被修改。

## 语义桥验证边界

原有语义桥154项基础CPU检查在改动前即通过；复现的失败在其组合使用的原生采样身份路径，不是已证实的Bridge权重计算错误。修复前组合检查18项失败，修复后同批193项通过。旧v0.43.0、v0.41.1所选检查各227项通过。各批均另有3项要求真实Python 3.13解释器的编译器元数据检查跳过；当前运行时为Python 3.12.10，已有注入式元数据正反例仍执行，不能宣称做了真实3.13运行验收。

最终本机Core综合回归为462项通过、3项上述跳过；独立源码门禁批次25项通过。两批存在覆盖重叠，不相加成独立功能数。旧Core另各完成2项原生Bridge到AV采样检查。所有本轮运行均未初始化CUDA。

额外检查原生Bridge应用节点到SigmaShift、Guider和完整AV采样的文字／参考图两条tiny Core路径；本机已有Original、BUNNY、Comic Combat和Wushu四份实际Bridge权重也完成了CPU原生AV衔接。H3去噪器是明确标注的随机小模型，因此这些检查证明软件接线，不代表新增完整训练H3的GPU画质。用户原报告未提供完整trace，不能断言它与这两处复现完全相同。

旧节点ID／输入／默认参数、已审工作流、Bridge数学路径均保留。修改的是身份兼容与完整性恢复。新增有限源码后继单独绑定六份明确源码／CI文件和原始preimage；历史H21／H24／H27／Pika账本、采样调用站点和人工成片资格不重录。

### 多桥反馈的补充核查

用户进一步说明“老单桥正常，新多桥报错”后，额外执行了真正的组合路径，不以四份权重分别通过代替多桥通过：

- 在最新Core、v0.43.0、v0.41.1分别检查自动参数节点、链接图预检、Core V3执行分发、组合节点和Apply；每版所选多桥／Transformer／输入验证批次56项通过，其中新增6项完整多桥集成回归。
- 本机已有Original、BUNNY、Comic Combat、Wushu权重组成五种顺序：Original→BUNNY、Comic→Wushu、BUNNY→Comic、Wushu→Comic、四桥串联。每版Core各完成文字／参考图两种原生AV CPU衔接，共各10例。H3仍为随机tiny测试模型，只证明软件兼容。
- 最新Core另在RTX 4060 Ti上执行五种真实训练Bridge组合、FP32／FP16／BF16、CPU条件显式送CUDA及CUDA条件自动设备两种路径，共30例全部通过，输出有限、dtype／设备及原输入保持正确。本次仅加载Bridge适配器，峰值张量分配约46 MiB；没有加载完整H3检查点或生成新的验收视频。

在这些条件下未复现新的多桥专属异常，因此本次补检**没有猜测并修改多桥运行代码**。前面两处共享身份兼容修复已经包含在GitHub源码中；是否修复反馈者那一条错误，仍需其实际报错节点、完整堆栈及所选模型确认。原报告缺失不等于所有其它环境或模型组合已通过。

## 开放PR处理结论

- [#31 Python 3.13](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/31)：现有`_sampling_class_members`已覆盖，并额外检查元数据的真实类型及可执行成员。本次保留更完整实现，不重复合入较弱版本。PR正文提到的Core整文件哈希问题在本次另行修复。
- [#29 Topaz输入时间轴](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/29)：未直接合并。对其固定head `71a99d6`的实际代码，用CPU生成的合法MP4／MKV复现两处模式分发错误：CFR MP4显式`reencode`仍提前返回原片；合法MKV显式`reencode`仍在调用FFmpeg前被拒绝。新增节点还插入旧Topaz注册列表中间，未保持完整节点前缀；FFmpeg子进程缺少超时／持续取消检查和失败产物清理。容器表修复也需要更多时间表、音轨和编辑列表一致性验证。修正这些边界前不能把该PR视为已安全集成。

## 15个开放Issue的核对

| Issue | 当前结论 |
| --- | --- |
| #32 扩画状态字典 | 本次修复匹配的非持久buffer导出误判；原报告的具体完整工作流仍未收到复测。 |
| #30／#24 Topaz解码 | 已有v1.98.2显式software／CUVID修复和既有验证，等待原环境复测；不把所有Topaz异常归为同一根因。 |
| #23 Topaz多GPU | 已有独立Topaz NVML读取器；无真实双NVIDIA硬件的新验收。 |
| #28 导演台打不开 | 已有模块路由404修复；本次没有新原环境复测，不盲关。 |
| #22 LoRA选择无变化 | 已有LOW／HIGH独立列表和保存恢复修复；时间轴、许可与第三方工具建议不算同一bug。 |
| #27 路径过长 | 现有短文件名和Windows路径门禁仍保留；不重命名用户已保存文件。 |
| #25 放大尺寸 | 现有width／height直连及Reconcile契约；相同MP不保证两种算法选出相同尺寸，不能静默改旧默认。 |
| #21 HyperFlow | 已有专用加载、采样与工作流；不因此认证任意组合。 |
| #20 tokenizer byte contract | 已有慢／快ByteLevel兼容和精确token等价检查，不用任意tokenizer替换。 |
| #19 Registry | 本次官方只读API显示1.103.0为公开最新，目标1.104.0仍Pending。保持审核问题独立，不用GitHub推送或绿色Action代替可安装状态。 |
| #18 空间分块二采 | 能执行不等于无接缝；真实质量验收仍未满足，不能宣布完成。 |
| #17 MPS音频前导 | 仍缺少同平台的最终修复／试听确认，本轮不凭空宣称解决。 |
| #12 CUDA能力识别 | 历史桌面／Registry元数据识别反馈，缺原安装环境的新复测。 |
| #26 云端原版材料 | 外部云端组件、素材和准确部署版本无法凭本地代码补造，仍需原材料。 |

这些Issue的旧回复已与现有源码核对，没有批量关闭尚需用户、硬件或外部服务确认的条目。

更新时应更新**T8自定义节点源码**并重启对应ComfyUI后端；仅更新ComfyUI本体或刷新浏览器不会加载本次Python修复。本次维护不自动覆盖或重启用户正在使用的服务。
