# H3 INT8 行分块（实验）

此优化只改变符合实际调用合同的 INT8 矩阵乘行调度，不改底模、LoRA 顺序、采样步数、sigma、音频时钟或 learned 3D 放大器。它与 QKV／query 工作集分块、普通 MLP 分块是不同的开关；默认旧图不受影响。

## 接口与行为

Python 入口：

```python
from h3_t8.int8_row_chunk import Int8RowPlan
from h3_t8.int8_row_chunk_advanced import configure_int8_row_chunks, audit_int8_row_chunks

selected, setup = configure_int8_row_chunks(
    model, Int8RowPlan(rows=2048, targets="all"), phase="LOW"
)
report = audit_int8_row_chunks(selected)
```

`targets="all"` 检查该扩散网络实际命中的 INT8 Linear；`"mlp"` 只调度真实 MLP 内部。`rows=0`／`enabled=False` 不新增此优化，保持传入 MODEL。两采可以从同一底模各建 LOW／HIGH 分支，设置不同的行数；Apply 不复制权重，也不表示两个 MODEL 物理独立。

当前实现分开覆盖两条实际路径：

- 普通 `QuantizedTensor` 的 Linear／mm／addmm：使用当前线程的 Torch dispatch scope，保持模块输入处理和权重准备原调用，之后分行执行实际原算子。
- 原生 MLP 的融合 SwiGLU：该路径绕过上述算子入口。仅在原 Core／布局调用合同对应时，使用请求私有的函数 globals 副本，把实际公共 INT8 调用分行；原 Cast／权重准备仍在分块之外执行一次。

不替换全局 `comfy_kitchen.int8_linear`、Core 全局函数／类或 Kitchen 私有 dispatch 表。私有表只读用于识别实际执行所有者。MLP 临时实例方法在 `finally` 恢复；无此节点的其它线程没有当前请求的 dispatch scope。R1／R2 共用真实网络身份租锁，避免两种锁顺序造成死锁；这不保证第三方绕过该租锁的 ModelPatcher 权重加载天然并发安全。

未知前驱、用户 callable、自定义 backend／约束或无法确认的布局保留原调用，并报告 `partial` 或无合格调用。BF16、FP8、被 Core 转成普通权重的分支不按 INT8 命中计数。真实 Kernel／张量错误照常抛出，不重试或静默换后端。

## 输入处理与已有融合边界

普通模块的 `pre_quant_scale`、forward hooks、实际权重回调在原模块内执行；本优化不改成 raw `F.linear`。LoRA 按用户原顺序／强度准备，不新增 prepared-weight 缓存。

当前 Core 的 `linear_input_act` 存在已观察到的边界：融合分支会绕过模块 `forward`，因此可能绕过运行时额外设置的 `pre_quant_scale`／模块 hooks。R2 遇此类分支保留完整原路径并明确 `partial`，不谎称已优化或已修复，也不私自设置 `comfy_force_forward` 改变既有输出。需要模块语义时，应显式选择 Core 已有的 forward 路径；这与行分块是否生效分别检查。

## 如何看报告

`calls` 是实际进入的算子／融合调用及行数，不是声明步数、NFE、GPU kernel 数或完整任务成功凭据。`selected_not_executed` 表示仅完成配置；`observed_no_eligible_calls` 不代表显存或速度改善。新请求通过真实 pre-run 重置统计，cleanup 后可读取最后一次观察。审计本身不会排队或生成。

完整输入、完整输出和 attention 的完整 K/V 仍存在。`completed_output_bytes` 是本次已完成分块输出的累计字节体积，不是同时驻留峰值；它不能证明减少多少显存。权重／offload、量化／LoRA准备、传输、矩阵 scratch、完整 hidden／KV、VAE／放大器需分别测量。

所有报告保留 `portable_cache_identity=false`，不授予 Stage 冷恢复／缓存复用资格。输出排序输入或已有 latent 不能制造“本轮已采样”记录。

## 当前验证边界

已用实际 Core／Kitchen CPU INT8 加载和算子检查普通矩阵与融合 SwiGLU，包含真实 tiny Core 两步 AV、全部 Linear 命中、只 MLP、LOW／HIGH 共底模、ordered LoRA、原输入缩放、嵌套／真实线程、异常恢复和下一普通请求。测试中的随机微型权重不是训练模型样片。

当前固定 Ref2VA 原生单采 20 步的 R2 单独、R1+R2 联合组已观察到真实 INT8 分块调用、非零 chunks 及请求 cleanup；连同 OFF／R1，各完整 120 帧 736×416／24fps 原生音视频机械通过，原源／模型末尾核验及进程退出齐。本地证据为 `h22-R12-batch-merge-v2/actual-four-index-v1/merge.json`。三份既有 `cudaMallocAsync` 记录仅是相应整请求的分配器读数，起始状态不同，不是整显卡显存节省或公平测速。

4+4 成片、其它设备／组合、数值逐位一致性和人工画质仍需独立验收；不能从 CPU 逐位一致推出所有 GPU 逐位一致或通用收益。若权重或完整 K/V 主导，实际收益可能为零。

研究参考：[固定上游 INT8 RowChunk](https://github.com/barkeslim-ops/ComfyUI-H3-Int8RowChunk/commit/269e9ef20a2c6d36c6fb93361c38df4140547ad9)（MIT）。本实现独立使用当前 Core／Kitchen 接口，不安装其全局 wrapper／私表改写；作者 GB10 成绩不是 T8 测量结果。
