# H3 投影分块

这是可选的计算工作集优化，不是新的采样器，也不是加速 LoRA。它分批处理能确认逐行独立的
输入投影、norm/QKV、输出投影和最终层；完整 K/V、生成长度、参考坐标及采样时间表不缩减。
没有插入节点时，现有工作流不变。

## 接线

在需要的模型和 LoRA 配置后接 `H3 投影分块 · Projection Chunks (EXP/T8)`，将它输出的
MODEL 送给原来的采样配置和 guider。正负条件、音频、SIGMAS、SAMPLER、latent 和解码路径
保持原接线。Apply 的报告是“已选择”，不能证明采样实际用到了它。

把同一份选中 MODEL 接到 `H3 投影分块 · 实际调用审计 (EXP/T8)`，`after_sampling` 接对应
完整采样的结果。这个输入只保证执行顺序；随便接一个对象或旧结果不会凭空产生调用记录。
Audit 返回实际 wrapper 调用、逐路径行数、委托原因和清理状态。

双采仍可共用底模，LOW 和 HIGH 各自接一个 Apply，分别设分块参数及 `phase` 标签。
两阶段 LoRA 的文件、顺序和强度仍由原节点控制；原 learned 3D upscaler、4+4 或其它显式
时间表不由本节点改动。`phase` 仅是报告标签，不是自动选择阶段或证明 HIGH 已执行。

## 参数

| 参数 | 含义 |
| --- | --- |
| `enabled` | 关闭时返回输入 MODEL，不安装本优化。它不移除输入链上用户已经显式选择的其它节点。 |
| `rows` | 投影和最终层每批最多处理的行数，缺省 1024；必须为正整数，没有人为固定上限。 |
| `query_rows` | 已确认原生 dense SDPA 路径的每批 Q 行数，缺省 1024；每批仍能看到完整 K/V。 |
| `stage_queries` | 仅在上述已确认 dense 路径将完整 Q 暂存 CPU，缺省关闭；会增加主存占用及传输。 |
| `norm_qkv` | 选择 norm/QKV 的分批路径。不能确认原调用语义时，保留完整原调用。 |
| `output` | 选择 attention 输出投影的分批路径。 |
| `embedding` | 选择输入 embedding 投影的分批路径，不替代 Qwen 编码或前缀缓存。 |
| `final` | 选择最终层的分段原生委托，保留实际 AV 调制行及 PDD/已适配 HyperFlow 调度。 |
| `phase` | 自定义报告标签，通常填 SINGLE、LOW 或 HIGH。 |

小批次不保证更快。最终输出缓冲、完整 K/V、权重驻留、offload、MLP 和 VAE 仍可能是主要开销；
逐路径行数及字节数不能当作显卡 allocator 峰值或整片提速成绩。

## 用户补丁与请求隔离

Apply 在 MODEL clone 上选择 wrapper，不在加载时修改共享网络。前向期间按实际网络取得可重入
租锁，并用请求局部选择分派；结束及异常路径恢复原方法，后来安装的外部方法不被覆盖回旧值。
共同使用租锁的算子可以嵌套，租锁本身不安装、关闭或改变另一个算子的选择。
这不代表任意第三方对同一 ModelPatcher 的并发换权重都安全。

自定义 forward、`__call__`、hooks、权重回调、内层 cast、注意力后端或 Relay 的语义不能确认时，
继续运行用户原路径，并报告 `partial`/`delegated`。不会删除 LoRA、换模型、删除时间偏置或重试
一个已经失败的计算。Relay/未知注意力保留完整 Q 路径；这不是禁止用户组合。
真实尺寸、API、NaN、文件及本节点显式配置被修改等错误仍正常报出。

## 如何读审计

- `completed_diffusion_calls` 是实际 diffusion wrapper 返回次数，不等于配置步数、NFE 或内核数。
- `calls` 只有实际命中的路径；`selected_not_executed` 或空计数不能写成“优化已生效”。
- `delegated` 列保留原调用的路径；`partial` 不表示用户生成失败，也不表示全部优化生效。
- cleanup 后保留本次观测，下一次真实 pre-run 重新计数；旧 LATENT 不算一次新采样。
- `portable_cache_identity=false`：当前未给此新 owner 授予跨进程 Stage 缓存身份。普通生成仍可运行，
  不得把历史缓存搬来冒充本节点的冷恢复成功。
- `allocator_peak_measured=false`、`performance_qualified=false`、`quality_accepted=false`
  不能从统计行数或成片能解码推翻。

## 当前验收范围

真实 tiny Core 的 CPU 回归覆盖完整采样、共享底模的独立 LOW/HIGH、实际 LoRA、dense 全 K/V、
Relay 委托、PDD/最终层、异常与后续普通请求、未知用户调用，以及采样最后一次前向的配置变更。
合法自定义模块、动态 cast、全局/局部 hook 和被原地修改的后端均有实际反例。
另验证了共同租锁不会关闭外层选择，异常后其它线程能继续取得租锁。

上述结果不是训练模型的显存、速度或成片质量资格。完整 GPU 对照、真实性能归因和人工审片
仍单独进行；实验草稿不能冒充已通过的推荐示例。

当前固定 Ref2VA 单采 20 步又完成 OFF、R1、R2、R1+R2 四组实际训练模型对照，各完整 120 帧 736×416／24fps 原生音视频机械通过。R1 的实际 wrapper、逐行投影消费及 cleanup 有原报告；此次 `stage_queries=false`，完整 query 委托和 `observed_partial` 范围照实保留，不等于所有投影／query 都已分块。原文件末尾核验与退出、四份完整 AV、逐类型结果及原索引前缀在本地 `h22-R12-batch-merge-v2/actual-four-index-v1/merge.json` 独立重推。它不是双采 4+4、任意组合、显存节省、公平速度或人工质量证明。
