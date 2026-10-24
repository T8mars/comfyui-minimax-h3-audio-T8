# PTD 原精度基模转换

这是独立的 CPU 转换工具，不是节点，也不改默认工作流。它把 Diffusers 原 H3 权重
映射为原生 Core 文件；不量化、不剪枝、不合并 PTD 专家或 DMD，不执行 checkpoint
中的代码。来源、完整许可与 NOTICE 见
[来源记录](../THIRD_PARTY_NOTICES/H21-PTD-BASE-SOURCES.md)。

截至 2026-10-22，完整 66,280,488,088 字节输出已转换并独立核对全部 534 份原权重 payload 和 16 个 FP32 RoPE 值；输出 SHA 为 `8dcc55e31dd99fcefaf58ab77efa44e457b1f62fb337e2c17003370de2da4c59`。该原精度底模已实际用于 PTD 4 步／连续 2+2 对照，不代表所有任务或人工画质通过。证据见开发目录 `base-independent-byte-audit-v1/full-v1/result/verified.json`。

固定参考是 `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`。
原配置／index 的 638 个 tensor 映射成 534 个原权重，再加 16 值 FP32 RoPE，共535个。
Q/K/V 按该顺序拼行；两类 MLP 的第一投影交换 `[up,gate]` 两半；其它原值及每个 tensor
的实际 dtype 保留。不是把已有 INT8／PDD／裁剪模型冒充原底模。

## 准备与显式执行

`--source` 指包含 `transformer/` 的目录，不是 transformer 目录本身。
`--manifest` 是包含完整 `files` 数组的 SHA 清单；每项含 `file`（或 `path`）、
`bytes`、`sha256`。正式下载工具产生的 `verified.json` 可以直接使用。

默认／`--prepare` 只读 manifest、config、index 和实际 safetensors 头，不读取 tensor
payload，也不导入 Torch。它不能宣称权重有限、完整 SHA 已通过或模型可运行。

```powershell
python tools/convert_h21_ptd_base.py --prepare `
  --source G:/your-fixed-source --manifest G:/your-verified.json `
  --report G:/new-prepare-report.json
```

完整输入下载并校验后，另行显式执行：

```powershell
python tools/convert_h21_ptd_base.py --convert `
  --source G:/your-fixed-source --manifest G:/your-verified.json `
  --job G:/new-plain-base-job --output G:/new-model/minimax_h3_original.safetensors `
  --report G:/new-conversion-report.json --threads 1
```

不要拿正在下载、尚未完成 SHA 校验的目录启动正式转换。工具不下载、不装环境，
输出不能放进源或 job 目录，也不会覆盖／采用一个没有本 job 发布记录的现存文件。
`--chunk-bytes` 为用户选择的读取缓冲大小，不是模型大小上限。

## 有界内存与恢复的真实成本

转换先扫描完整源 SHA，并保留分块内容摘要。后续复制每段字节时重新验证对应源块，
避免只靠文件日期或“前后恰好恢复原值”而使用中途篡改的数据。每个原生 tensor 单独
生成可恢复 part；校验浮点有限性只使用分块 CPU 数据，不建立完整 state 或矩阵。
只有小 RoPE 按官方 FP32 CPU 公式计算。

完成后完整复核源 SHA，再流式校验／组装全部 parts，使用无覆盖原子发布。
取消或异常保留已校验 parts；同一输入／实现／缓冲设置可恢复。坏 part、源 SHA、
状态或未知 job 目录不会自动修好或改名绕过。发布前／后中断通过本 job 日志恢复，
不是采用任意已有输出。

恢复时，已有 part 不能只凭它和可编辑状态中相互匹配的 SHA 取得资格。每个恢复的
payload 都须重新逐块对照当前完整 SHA 已验的原 source ranges，按同一 QKV／SwiGLU
次序比较；派生 RoPE 用同一个官方 FP32 CPU 公式重建对照。part 与状态 SHA 同时
改成有限、同 shape 的其它数值也必须拒绝，不自动覆盖或修补该 part。首次新生成的
part 已从已验证源块直接写入，不为这项恢复检查额外重复读源。
首次和恢复共用相同分块有限性／元素边界 carry 检查；即便 source 本身含 NaN 且它与
伪造的 part／状态全部 SHA 自洽，仍须拒绝。`convert`／恢复允许导入 CPU Torch；只有
metadata-only 准备承诺不导入 Torch，不能为了维持错误观测而跳过真实坏张量校验。
已有发布日志和输出时，同样逐块将真实输出与这些已绑定源的 parts 对照，并核对同次
读取摘要；不会仅因输出和可编辑 publication SHA 同时一致就采用。发布后回调中断、
尚无最终 result 的情况也受此检查，不覆盖或修复用户已改动的输出。

这不是零复制、零内存或零耗时。源 payload 约66.28GB，输出 payload 多64字节 RoPE，
此外还有文件头；源＋parts＋输出约需199GB，SHA 扫描和组装会产生大量真实 I/O。
恢复已有 parts 还会额外读取其对应源范围／小 RoPE，不能把这项内容绑定称为免费。
恢复已有完整输出还要对照读取 parts 与输出；不会另外写一份模型，但同样有真实 I/O 成本。
内存还包括源块摘要、JSON、有限性临时缓冲及 OS 文件缓存，不能据显式小缓冲保证总 RSS。
应另记实际峰值、耗时、完整文件 SHA、Core 加载和 GPU 行为。

## 验证边界

微型 safetensors 必须与固定官方函数逐 tensor 比较 dtype 和原字节，并覆盖跨 shard
QKV、SwiGLU、RoPE、FP32 岛、损坏／NaN／shape、篡改恢复、取消／发布崩溃恢复、
真实 Core strict load。CPU 通过不等于66GB完整转换、训练采样、速度、显存或画质通过。
实际大文件转换、原模型完整加载及 GPU 审片需要各自明确的收据，不能从工具存在推断完成。
