# MiniMax H3 显存验证方法

本方法用于验证 MiniMax H3 双时钟采样、旁路 LoRA 和 ComfyUI DynamicVRAM/VBAR 之间的
显存关系。它只采集运行数据，不修改采样数学、模型权重或 ComfyUI 显存策略。

诊断工具：`tools/validate_h3_vram.py`。

## 能验证什么

- 工作流是否使用 4 步双时钟、旁路 LoRA、额外 scheduler 或不一致的模型参数；
- ComfyUI 启动日志是否明确记录 `DynamicVRAM support detected and enabled`；
- 每次执行前的显存基线、执行峰值及相对基线增量；
- 峰值对应的 ComfyUI 节点和采样进度；
- PyTorch 当前活动显存，以及包含 VBAR/其他进程影响的设备可用显存；
- OOM 的节点、异常类型、错误信息和 ComfyUI traceback；
- 两次运行的模型、LoRA、CLIP、VAE、媒体、Prompt、尺寸、帧数、seed 和其他非采样输入
  是否保持一致。

工具不能仅凭一次运行证明某个组件是 OOM 的唯一原因。`/system_stats` 是按间隔采样，极短的
瞬时峰值可能落在两个采样点之间；桌面程序和其他 CUDA 进程也会影响设备显存。因此应使用
同一机器、相同后台负载和至少两次重复结果。

## 准备工作

1. 在 ComfyUI 中加载用户实际失败的工作流。
2. 使用 `Save (API Format)` 导出 API 工作流。普通前端工作流包含 `nodes`/`links`，不能直接
   提交到 `/prompt`，工具会明确拒绝。
3. 暂停其他队列任务，关闭会明显占用显存的程序。
4. 第一轮统一使用：相同模型、LoRA、Prompt、seed、分辨率、帧数、输入媒体和
   `preview-method=none`。
5. Turbo 稳定对照先固定为 4 步。12 步应作为第二个独立变量测试，不能与 4 步旧基线直接
   比较。

以下命令均从项目目录执行，并使用启动 ComfyUI 的 Python：

```powershell
$python = 'F:\AI-T8-video-onekey\python\python.exe'
$tool = 'F:\AI-T8-video-onekey\ComfyUI\custom_nodes\minimax-h3-audio-T8\tools\validate_h3_vram.py'
```

## 1. 静态检查

```powershell
& $python $tool inspect '.\failing_api.json' `
  --server 'http://127.0.0.1:8188' `
  --log auto `
  --output '.\artifacts\vram-validation\failing-inspect.json'
```

`DynamicVRAM: enabled (source=log)` 表示工具在启动日志中找到明确启用记录。
`available_not_proven` 只表示安装了 comfy-aimdo 且硬件看起来支持，不能当作已启用证据。

## 2. 生成严格 A/B 工作流

从包含一个 `MiniMaxH3DualClockSamplerT8` 的失败 API 工作流生成两份工作流：

```powershell
& $python $tool make-pair '.\failing_api.json' `
  --steps 4 `
  --output-dir '.\artifacts\vram-validation\pair'
```

生成结果：

- `*-stock-euler-4step.json`：使用原生 `MiniMaxH3SigmaShift`、stock Euler 和
  `BasicScheduler(simple)`；
- `*-dual-clock-4step.json`：使用项目双时钟 Euler。

工具会保留相同的模型、旁路 LoRA、Conditioning、AV latent、Prompt、seed 和输出链。只有
采样设置是实验变量。如果双时钟的 MODEL/SAMPLER/SIGMAS 三个输出没有全部接入正式采样链，
工具会拒绝生成，避免得到无效对照。

stock 对照只用于显存归因。它对四步 H3 音频的数值积分不等价，不能用其音频质量评价双时钟
算法。

## 3. 执行并记录

```powershell
& $python $tool run '.\artifacts\vram-validation\pair\failing_api-stock-euler-4step.json' `
  --label 'stock-4step' `
  --preview-method none `
  --poll-interval 0.25 `
  --timeout 3600

& $python $tool run '.\artifacts\vram-validation\pair\failing_api-dual-clock-4step.json' `
  --label 'dual-4step' `
  --preview-method none `
  --poll-interval 0.25 `
  --timeout 3600
```

报告默认写入 `artifacts/vram-validation/`。即使 ComfyUI 返回 OOM，工具仍会保存已采集显存
曲线和 `execution_error`；命令退出码为 1，便于自动化识别失败。

建议分别做两类测试：

- 暖缓存 A/B：不重启 ComfyUI，连续运行两份工作流；公共 loader/conditioning 缓存命中是
  预期行为，报告会记录 `execution_cached`；
- 冷启动 A/B：每次运行前重启 ComfyUI，并保持相同后台显存基线。至少交换一次运行顺序，
  避免把先后顺序误判为节点影响。

## 4. 比较结果

```powershell
& $python $tool compare `
  '.\artifacts\vram-validation\20260807-120000-stock-4step.json' `
  '.\artifacts\vram-validation\20260807-121000-dual-4step.json' `
  --material-mib 256 `
  --output '.\artifacts\vram-validation\comparison.json'
```

判定字段：

| verdict | 含义 |
|---|---|
| `not_comparable_control_inputs_changed` | 模型、媒体、Prompt、尺寸等控制变量发生变化 |
| `not_comparable_incomplete_run` | 至少一次运行没有成功完成 |
| `no_material_peak_difference` | 峰值差小于指定阈值 |
| `second_run_has_higher_peak` | 第二次运行峰值显著更高 |
| `second_run_has_lower_peak` | 第二次运行峰值显著更低 |

比较使用“峰值减去各自运行前基线”，不是单纯比较 Windows 任务管理器中的绝对占用。

## 分阶段判定

1. 如果显存在进入 `SamplerCustomAdvanced` 前一次性增加约一个 LoRA 文件量级，优先检查
   旁路 LoRA adapter 驻留。
2. 如果每一步结束后显存持续单调增长，检查 callback、预览、缓存、旁路输出生命周期或
   分配器碎片；此时 12 步可能只是把增长放大。
3. 如果双时钟 4 步与 stock 4 步峰值接近，而只有 12 步失败，先按逐步增长问题调查，不能
   归因于双时钟 clone。
4. 如果控制变量一致且双时钟 4 步稳定多出大量显存，再检查 ModelPatcherDynamic clone、
   object patch 和 bypass injection 的组合生命周期。
5. 如果两次运行前基线相差数 GB，结果无效；先处理后台程序或交换执行顺序重测。

在取得真实 OOM traceback 和至少一组有效 A/B 前，不建议修改稳定采样数学、关闭 VBAR、
替换 INT8 旁路 LoRA，或把普通 LoRA 合并路径作为生产修复。

## 2026-08-07 本机实测检查点

用户提供的可运行工作流实际启用非裁剪 FL2VA INT8、SageAttention、Standard bypass Turbo
LoRA 和双时钟采样；参考图节点处于静音/禁用状态，不参与执行。以该链路构造了
`0.6M`、15 秒对齐 362 帧、关闭预览的暖缓存 A/B：

| 路径 | 步数 | 结果 | 设备峰值 | PyTorch 峰值 | 耗时 |
|---|---:|---|---:|---:|---:|
| stock Euler | 4 | 成功 | 16,213.5 MiB | 14,573.5 MiB | 1,210.9 s |
| T8 双时钟 | 4 | 成功 | 16,182.2 MiB | 14,573.5 MiB | 1,631.4 s |
| T8 双时钟压力测试 | 12 | 成功 | 16,245.5 MiB | 14,573.5 MiB | 3,280.2 s |

4 步配对的控制输入完全一致。双时钟减 stock 的设备峰值为 `-31.3 MiB`，小于 128 MiB
实质差异阈值；两边 PyTorch 峰值完全相同。因此本次数据不支持“双时钟节点没有适配
VBAR，导致模型权重额外常驻”的假设。

这不等于否定用户 OOM：三次运行都距离 16 GiB 上限极近，其他 CUDA 程序、预览、显存碎片、
模型缓存顺序或工作流替换时连带改变的节点，均可能造成一边刚好成功、另一边刚好失败。
本检查点只是一台 RTX 4060 Ti 16 GiB 上的一次暖缓存序列。下一轮应在每次重启 ComfyUI 后
交换运行顺序重复，并优先取得反馈用户“官方可跑/替换后 OOM”两份 API 工作流及完整 traceback。
