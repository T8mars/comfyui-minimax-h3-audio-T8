# MiniMax H3 Audio Refine 三节点设计规范

状态：设计评审稿；本文件批准前不进入生产代码实现。

日期：2026-08-25（2026-08-26 恢复并完成设计）

## 1. 目的与结论

本功能用于对已经完成首遍采样的 MiniMax H3 联合音视频 latent 再执行一段生成式音频尾段采样。视频流在最终输出中严格恢复为输入视频 latent，音频流在同一视频、提示词和参考媒体上下文中重新生成。

第一版只实现精确、无 Frozen Cache 的 T8 双时钟路线，并追加三个 `Advanced EXP` 节点：

1. `MiniMax H3 Audio Refine Audit (T8 Advanced EXP)`
2. `MiniMax H3 Audio Refine Plan (T8 Advanced EXP)`
3. `MiniMax H3 Audio Refine Dual-Clock Setup (T8 Advanced EXP)`

它不是波形降噪、EQ、锐化或分轨处理。它可能改变文字、音节、音色、演绎、音乐、环境音、瞬态和同步。因此第一版只产生原始候选之外的精修候选，不自动覆盖用户原始音频。

## 2. 固定依据

### 2.1 上游依据

- 固定来源：`Adudeguyman/ComfyUI-H3-AudioRefine`
- 固定提交：`b6abccd572215491c8bb25406bec42cb5ef33138`
- 上游版本：`1.0.2`
- 许可证：MIT
- 核心机制：联合 AV `NestedTensor` 的视频噪声蒙版为 0、音频噪声蒙版为 1，再执行 partial-denoise。

上游一体化 sampler 内部调用普通 `comfy.sample.sample()`，会自行创建普通 KSampler 调度。该路线不会执行 T8 的独立音频时钟推进和 partial-start audio rebase，因此不能作为本实现的 sampler。

### 2.2 本项目依据

- 当前发布版本：`1.47.0`
- 当前已发布节点位置：`0..210`
- 项目记录的已验证 ComfyUI 提交：`187eda8ef5e588c6a5765cad53e482765edae052`
- 当前本机 ComfyUI 提交：`b78cec879b9460d5cb25228a83a942fb78d2cd24`
- 必须复用：
  - `sampling.py::_rebase_partial_audio_start()`
  - `sampling.py::time_shift_sigma()`
  - `sampling.py::setup_dual_clock_sampling()`
  - `sampling.py::sample_minimax_h3_dual_clock_euler()`
  - `core.py::nested_av_parts()`
  - `core.py::split_noise_masks()`
  - `nfe_run_contract_advanced.py` 的强内容哈希规则
  - `vram_policy.py::runtime_snapshot()`

兼容判断以运行能力检查为准，不以单一 ComfyUI commit 字符串作为通过条件。缺失 MiniMax H3 联合 AV、嵌套逐流蒙版或自定义采样协议时返回 `ABSTAIN`，不得猜测兼容。

## 3. 范围

### 3.1 第一版包含

- batch 1 的 MiniMax H3 联合 AV latent。
- 视频 latent `[1, 24, T, H, W]`。
- 音频 latent `[1, 32, 2, T40]`。
- BasicGuider 等价的单条件、CFG=1 路线。
- `dual_clock_euler + native_flow`。
- 固定 video shift 12、audio shift 3。
- `video_denoise=0`、`audio_mask=1`。
- 显式固定 seed。
- KSampler 等价的 partial-tail sigma。
- 精确无缓存执行。
- `ALLOW / ABSTAIN / REJECT` 决策、可审计报告和无操作回退。

### 3.2 第一版不包含

- Frozen Video Cache、磁盘缓存、跨执行缓存。
- CFG>1、动态 CFG、负向条件、多条件 guider。
- 视频同时重绘、局部视频 mask、时间范围音频 mask。
- native multistep、EXP multi-rate、第三方 sampler 或任意 scheduler。
- 隐式启用、禁用或替换 Turbo LoRA、base model、Hybrid artifact 或文本编码器。
- Block Cache、Activation Chunk、STG、EAV、SLA、Prompt Relay、Long Video scoped patch 及未知 transformer wrapper 的组合放行。
- 自动 CER/WER、音色、口型或听感优劣判定。
- 自动接受精修候选。
- 大模型压力矩阵、并发测试和跨 GPU 验证。

## 4. 数据流

```text
首遍采样后的 AV LATENT ───────────────┐
首遍使用的 MODEL ─────────────────────┤
首遍使用的 CONDITIONING ──────────────┤
conditioned_prompt / media_map / report ─┤
                                       v
                              Audio Refine Audit
                                       |
                                  audit contract
                                       v
                              Audio Refine Plan
                                       |
                                   refine plan
                                       v
                     Audio Refine Dual-Clock Setup
                         | MODEL | NOISE | GUIDER
                         | SAMPLER | SIGMAS | LATENT
                                       v
                           SamplerCustomAdvanced
                                       v
                              现有 H3 AV Decode
                                       v
                        原始候选与精修候选人工试听
```

Audit、Plan 和 Setup 都不调用 H3 Transformer。只有用户把 Setup 输出连接到 `SamplerCustomAdvanced` 后才发生模型采样。

## 5. 对原路线输出清单的必要修正

`SamplerCustomAdvanced` 的真实输入是 `NOISE / GUIDER / SAMPLER / SIGMAS / LATENT`。原 roadmap 将 Setup 简写为 `MODEL / SAMPLER / SIGMAS / LATENT / report`，无法机械保证确定性 noise，也无法阻止用户误接 CFG>1 guider。

因此正式设计将 Setup 输出修正为：

`MODEL / NOISE / GUIDER / SAMPLER / SIGMAS / LATENT / report`

其中 `GUIDER` 由 Setup 使用同一 MODEL 和同一 positive conditioning 组装为 BasicGuider 等价对象；`NOISE` 使用计划中显式固定 seed。Setup 同时输出 MODEL 供审计和可视化，但受支持的工作流必须把 Setup 的 `GUIDER` 直接接到 `SamplerCustomAdvanced`，不再另接外部 CFGGuider。

该修正不改变任何旧节点，也不影响旧工作流。

## 6. 公共类型合同

### 6.1 自定义类型

- Audit 类型：`H3_T8_AUDIO_REFINE_AUDIT`
- Plan 类型：`H3_T8_AUDIO_REFINE_PLAN`
- Audit schema：`t8.minimax_h3.audio_refine.audit.v1`
- Plan schema：`t8.minimax_h3.audio_refine.plan.v1`

两个运行对象只保存 JSON 可表达的标量、列表、形状、摘要和决策，不保存 MODEL、CONDITIONING、LATENT 或 tensor 引用，避免自定义输出延长大对象生命周期。

### 6.2 绑定规则

Audit 建立以下绑定：

- MODEL 当前 Python 对象 ID；只用于同一运行进程内防止接错分支。
- MODEL 类、基础模型类、model sampling 类、LoRA/weight patch 结构、attachment key、transformer wrapper 与 patch-replace 结构摘要。
- positive conditioning 全内容 SHA-256；GPU tensor 按固定 CPU chunk 顺序读取，不用 `sum()` 或 `abs().sum()` 代替身份。
- conditioned prompt UTF-8 SHA-256。
- canonical media map SHA-256。
- conditioning report UTF-8 SHA-256。
- 视频和音频 latent 分别按 dtype、shape 和逐字节内容计算 SHA-256。
- 原 noise mask 的布局、取值类别和摘要。

Setup 必须重新计算并匹配上述绑定。任何 MODEL 对象、conditioning 内容、提示词合同或 latent 内容变化均为 `REJECT_CONTRACT_MISMATCH`，不能静默继续。

MODEL 权重本体不做全量哈希。报告中的 model fingerprint 是结构指纹，不宣称为权重文件密码学身份；同一运行内由对象 ID 加结构指纹共同约束。

### 6.3 Canonical JSON

- UTF-8。
- key 按字典序。
- 不允许 NaN 或 Infinity。
- descriptor 含 payload SHA-256。
- Plan 只能保持或降低 Audit 决策，不能把 `ABSTAIN` 或 `REJECT` 升级为 `ALLOW`。
- descriptor schema、摘要或 payload 被修改时返回 `REJECT_DESCRIPTOR_TAMPERED`。

## 7. 节点一：Audio Refine Audit

### 7.1 节点 ID 与分类

- node_id：`MiniMaxH3AudioRefineAuditT8Advanced`
- display name：`MiniMax H3 Audio Refine Audit (T8 Advanced EXP)`
- category：`T8/MiniMax H3/Audio/Experimental`
- 预期注册位置：211

### 7.2 输入

| 输入 | 类型 | 默认 | 作用 |
|---|---|---:|---|
| `model` | MODEL | 必接 | 首遍实际使用、且准备继续精修的显式 MODEL |
| `positive` | CONDITIONING | 必接 | 首遍实际使用的正向条件 |
| `av_latent` | LATENT | 必接 | 首遍采样完成的联合 AV latent |
| `conditioned_prompt` | STRING | 必接 | Conditioning 节点最终输出文本 |
| `media_map_json` | STRING | 必接 | Conditioning 节点 media map |
| `conditioning_report` | STRING | 必接 | Conditioning 节点报告，用于解析 `audio_mode` 等合同 |
| `protected_audio` | AUDIO | 可选 | 若外部音轨必须作为最终权威音频，连接后直接 `ABSTAIN` |
| `minimum_free_vram_mib` | INT | 512 | 当前整卡最低余量；最小可设值仍为 512 |
| `minimum_commit_headroom_gib` | FLOAT | 16.0 | Windows/Linux commit 最低余量；最小可设值仍为 16.0 |
| `hash_chunk_megabytes` | INT | 8 | conditioning/latent 哈希的单次 CPU 传输上限，范围 1..64 |

资源门允许用户提高，不能低于上述下限。

### 7.3 输出

| 输出 | 类型 | 说明 |
|---|---|---|
| `audit` | `H3_T8_AUDIO_REFINE_AUDIT` | 不持有大对象的审计 descriptor |
| `decision` | STRING | `ALLOW`、`ABSTAIN` 或 `REJECT` |
| `report_json` | STRING | 完整可读审计报告 |

### 7.4 结构检查

以下条件全部满足才可能 `ALLOW`：

- `samples` 是恰好两个成员的 nested latent。
- video/audio rank 分别为 5/4。
- batch 都为 1。
- channel 分别为 24/32，音频 stereo 维为 2。
- dtype 支持逐字节哈希，tensor 不是 sparse、quantized 或 meta。
- video/audio 全部 finite。
- positive conditioning 可按现有 NFE Run Contract 规则强哈希。
- media map 是含 `pictures/videos/audios` 对象的合法 JSON。
- conditioning report 可唯一解析 `audio_mode`。
- 当前 ComfyUI 提供联合 AV、嵌套逐流 mask 和 custom sampler 所需能力。

### 7.5 原 noise mask 和 Audio Lock

- 无 `noise_mask`：视为首遍已完成且无残留锁定声明，可继续检查。
- nested mask 且 audio 全 1：可继续检查。
- nested mask 且 audio 全 0：`ABSTAIN_AUDIO_LOCKED`。
- nested mask 的 audio 为分数、混合 0/1 或非有限值：`ABSTAIN_PARTIAL_AUDIO_MASK_UNSUPPORTED`。
- conditioning report 为 `audio_mode=lock_source`：`ABSTAIN_AUDIO_LOCKED`，即使 mask 丢失也不放行。
- `protected_audio` 已连接：`ABSTAIN_PROTECTED_FINAL_AUDIO`。
- legacy video-only mask：记录兼容警告；只要 conditioning report 明确为 `native` 或 `reference_only` 才可继续，否则 `ABSTAIN_AUDIO_MASK_PROVENANCE_UNKNOWN`。

`remix_source` 在第一版返回 `ABSTAIN_REMIX_SOURCE_NOT_VALIDATED`。原因是其首遍音频已经带分数 mask，精修时强制 audio mask=1 会改变用户的源音频保留比例。

### 7.6 MODEL 与补丁门禁

第一版允许：

- 未经 transformer wrapper 改写的 MiniMax H3 MODEL。
- 标准 ModelPatcher weight patch/LoRA；只记录结构摘要，不隐式启停。
- MODEL 已有的显式 Hybrid/weight attachment 仅作为连接模型的一部分记录；节点不创建或切换它。

第一版返回 `ABSTAIN_PATCH_STACK_UNVALIDATED`：

- 任意 `transformer_options.patches_replace` 非空。
- 任意 transformer wrapper group 非空。
- 检测到 Block Cache、Activation Chunk、STG、EAV、SLA、Prompt Relay、Long Video scoped execution、EXP multi-rate 或未知采样状态。
- packed layout owner 无法确认来自当前 ComfyUI MiniMax H3 core。

MODEL 不是 MiniMax H3 或 model sampling 协议明确不兼容时为 `REJECT_NOT_MINIMAX_H3_MODEL`。

### 7.7 资源门

Audit 调用只读 `runtime_snapshot()`，不卸载模型、不清缓存、不设置全局 VRAM policy。

硬门：

- whole-device free VRAM 可测且不少于 `minimum_free_vram_mib`。
- host commit headroom 可测且不少于 `minimum_commit_headroom_gib`。
- host available RAM 可测，且不少于 `1.5 × node_owned_incremental_bytes + 512 MiB`。

`node_owned_incremental_bytes` 由一个同 dtype AV noise、一个同 dtype AV output 和两条 float32 逐流 mask 的字节数相加得到。它只估计节点直接拥有的新增 tensor，不包含 H3 权重、attention workspace、VAE/CLIP、CUDA context、其他进程或第三方补丁。

任一遥测未知或低于门槛返回 `ABSTAIN_RESOURCE_TELEMETRY_UNKNOWN` 或 `ABSTAIN_INSUFFICIENT_HEADROOM`。`ALLOW` 只表示当前机械前置条件通过，不等于 16GB 普遍安全或不会 OOM。

## 8. 节点二：Audio Refine Plan

### 8.1 节点 ID 与分类

- node_id：`MiniMaxH3AudioRefinePlanT8Advanced`
- display name：`MiniMax H3 Audio Refine Plan (T8 Advanced EXP)`
- category：`T8/MiniMax H3/Audio/Experimental`
- 预期注册位置：212

### 8.2 输入

| 输入 | 类型 | 默认 | 作用 |
|---|---|---:|---|
| `audit` | `H3_T8_AUDIO_REFINE_AUDIT` | 必接 | 上一节点的审计合同 |
| `refine_steps` | INT | 4 | 实际额外 H3 NFE，范围 1..8 |
| `audio_denoise` | FLOAT | 0.50 | KSampler 风格 partial depth，范围 0.01..1.00 |
| `refine_seed` | INT | 0 | 显式固定 64-bit seed；不自动 randomize |
| `model_strategy` | COMBO | `connected_model_explicit` | 第一版唯一选项，禁止隐式模型/LoRA 切换 |

video shift、audio shift、sampler、scheduler、CFG 和 video mask 不开放为 widget。第一版固定为 12、3、`dual_clock_euler`、`native_flow`、1 和 0。

### 8.3 输出

| 输出 | 类型 | 说明 |
|---|---|---|
| `plan` | `H3_T8_AUDIO_REFINE_PLAN` | 可复现的纯数据计划 |
| `decision` | STRING | 继承后的决策 |
| `report_json` | STRING | 参数、公式、有效 denoise 和 sigma 列表 |

### 8.4 KSampler 等价 partial-tail 公式

定义：

- `N = refine_steps`
- `d = audio_denoise`
- `M = int(N / d)`
- `k = 0..N`
- `b_k = (N - k) / M`

`M` 必须不小于 `N`。`int()` 采用与当前 ComfyUI `KSampler.set_steps()` 相同的正数截断规则。

视频时钟：

```text
sigma_video(k) = 12 * b_k / (1 + 11 * b_k)
```

音频时钟仅用于报告和 sampler 内部推进：

```text
sigma_audio(k) = 3 * b_k / (1 + 2 * b_k)
```

Setup 传给 `SamplerCustomAdvanced` 的 `SIGMAS` 是 `sigma_video[0..N]`。T8 sampler 在每一步用 `time_shift_sigma(video_sigma, 12, 3)` 得到音频 sigma，并在第一步调用 `_rebase_partial_audio_start()`。

实际有效 denoise 为 `N / M`，可能与用户请求值略有差异，报告必须同时写 `requested_audio_denoise` 和 `effective_audio_denoise`。

默认 `N=4, d=0.5` 时：

```text
M = 8
base sigma  = [0.5, 0.375, 0.25, 0.125, 0.0]
video sigma = [0.9230769231, 0.8780487805, 0.8, 0.6315789474, 0.0]
audio sigma = [0.75, 0.6428571429, 0.5, 0.3, 0.0]
actual NFE  = 4
```

Plan 不生成 tensor、不克隆 MODEL、不创建 noise、不执行 sampler。

## 9. 节点三：Audio Refine Dual-Clock Setup

### 9.1 节点 ID 与分类

- node_id：`MiniMaxH3AudioRefineDualClockSetupT8Advanced`
- display name：`MiniMax H3 Audio Refine Dual-Clock Setup (T8 Advanced EXP)`
- category：`T8/MiniMax H3/Audio/Experimental`
- 预期注册位置：213

### 9.2 输入

| 输入 | 类型 | 说明 |
|---|---|---|
| `plan` | `H3_T8_AUDIO_REFINE_PLAN` | 已签名计划 |
| `model` | MODEL | 必须是 Audit 时同一对象和同一结构合同 |
| `positive` | CONDITIONING | 必须与 Audit 强内容哈希一致 |
| `av_latent` | LATENT | 必须与 Audit 的视频/音频内容哈希一致 |

### 9.3 输出

| 输出 | 类型 | 说明 |
|---|---|---|
| `model` | MODEL | T8 双时钟 clone；ABSTAIN 时为原 MODEL |
| `noise` | NOISE | ALLOW 为固定 seed random noise；ABSTAIN 为零分配 bypass noise |
| `guider` | GUIDER | 同一 MODEL 与 positive 绑定的 BasicGuider 等价对象 |
| `sampler` | SAMPLER | T8 dual-clock Euler；ABSTAIN 为零步 no-op sampler |
| `sigmas` | SIGMAS | ALLOW 为 N+1 尾段视频 sigma；ABSTAIN 为空 tensor |
| `latent` | LATENT | ALLOW 为只替换逐流 mask 的浅拷贝；ABSTAIN 为输入对象 |
| `report_json` | STRING | setup、绑定复核、资源复检和回退报告 |

### 9.4 ALLOW 装配

Setup 的顺序固定为：

1. 验证 plan schema 和 payload SHA。
2. 重新验证 MODEL 对象 ID、结构摘要、positive 内容和 AV latent 内容。
3. 重新读取资源状态；运行时余量下降时把决策降为 `ABSTAIN`。
4. 调用 `setup_dual_clock_sampling(model, av_latent, M, 12, 3, "dual_clock_euler", "native_flow")`。
5. 从完整 `M+1` sigma 截取最后 `N+1` 项。
6. 浅拷贝 latent metadata；保持 samples tensor 本体不变。
7. 创建 float32、同设备、同形状的 video 全 0 mask 和 audio 全 1 mask，组成 NestedTensor。
8. 创建显式 seed 的 deterministic random noise 对象。
9. 用 patched MODEL 和同一 positive 创建 BasicGuider 等价对象。
10. 返回标准对象，不调用 guider、sampler 或模型 forward。

`setup_dual_clock_sampling()` 的 sampler 在首个 partial sigma 自动执行 `_rebase_partial_audio_start()`：只把 KSAMPLER 已建立的视频时钟音频起点恢复为相同 noise/latent 下的音频时钟起点，视频 slice 不变。

### 9.5 ABSTAIN 无操作回退

ABSTAIN 不抛出硬错误。Setup 返回：

- 原 MODEL。
- 零分配 bypass noise；当前 `SamplerCustomAdvanced` 会先调用 `generate_noise(input_latent)`，因此该方法只返回输入 `samples` 引用、不创建同尺寸 tensor；随后空 SIGMAS 让 `CFGGuider.sample()` 在 `prepare_sampling()` 前直接返回 latent，该引用不会进入模型计算。
- 原 MODEL 与同一 positive 的 BasicGuider 等价对象。
- 不调用 model 的 no-op sampler。
- 空 `SIGMAS`，不是 `[0.0]`。当前 ComfyUI `CFGGuider.sample()` 只对长度 0 执行采样前直接返回；长度 1 的 `[0.0]` 仍会进入 `prepare_sampling()` 并可能加载 MODEL，因此不能作为资源不足时的回退。
- 原始 `av_latent` 对象，不替换 samples 或 mask。
- 明确 `bypassed=true` 的报告。

连接到 `SamplerCustomAdvanced` 后必须满足：不调用 `prepare_sampling()`、MODEL 加载和 MODEL forward，bypass noise 不分配同尺寸 tensor，输出 samples 与输入逐值一致。`SamplerCustomAdvanced` 仍可能执行其最终 `.to(intermediate_device())`；设备已经一致时不得复制，设备不一致时只允许设备搬运，数值必须一致。该合同必须由当前真实 `SamplerCustomAdvanced` 机械测试锁定；若未来 ComfyUI 不再在空 SIGMAS 上提前返回，Audit 必须将该 core 标记为不兼容并 `ABSTAIN`，而不是继续使用回退。

### 9.6 REJECT

REJECT 或 Setup 发现绑定被篡改时，在创建 noise、mask、MODEL clone 或 guider 前抛出明确 `ValueError`。错误必须包含稳定 code 和修复建议。原始 latent 仍可由工作流的原始分支直接解码，节点不得删除、覆盖或原地修改它。

## 10. 决策语义

| 决策 | 含义 | Setup 行为 | 是否允许宣传音质改善 |
|---|---|---|---|
| `ALLOW` | 当前结构、合同、补丁和资源前置检查通过 | 装配精确双时钟 refine | 否，仍需生成和人工试听 |
| `ABSTAIN` | 输入可保留，但当前路线未验证、被锁定、遥测未知或余量不足 | 输出零步 no-op，保留原候选 | 否 |
| `REJECT` | 输入不是合法 H3 合同、内容被换线/篡改或参数无效 | fail closed，采样前报错 | 否 |

同一计划内优先级为 `REJECT > ABSTAIN > ALLOW`。报告可包含多个原因，最终决策取最高优先级。

## 11. 异常与生命周期

- Audit 和 Plan 只读，不修改 MODEL、conditioning 或 latent。
- Setup 只创建本节点需要的 clone、mask、noise 和 guider，不调用全局 `unload_all_models()`。
- Setup 任意异常前后都不原地修改输入 latent。
- Setup 创建中途异常时仅释放自己的局部引用；不移除其他节点 hook，不清理全局缓存。
- 第一版没有持久缓存、后台线程、文件输出、磁盘临时文件或进程级单例。
- ComfyUI 取消或下游 OOM 时，第一版不声称能够主动释放上游或 ComfyUI 拥有的模型；它只保证自己没有持久引用。

## 12. 兼容边界

### 12.1 支持

- 新节点直接接现有 T8 Conditioning 输出和首遍 SamplerCustomAdvanced 输出。
- MODEL 必须是首遍实际使用且准备精修的同一连接对象。
- 标准 LoRA/weight patch 可作为显式连接 MODEL 的一部分保留；节点只记录，不改变。
- 已发布的稳定 Audio、双时钟、Decode 和所有旧工作流保持原序列化。

### 12.2 拒绝或弃权

- `lock_source`、连接的受保护最终音频、`remix_source` 和 fractional audio mask。
- 任意未知 MODEL、conditioning、media-map、latent 或 descriptor 身份变化。
- CFGGuider、DualCFGGuider、动态 CFG 或负向条件路线。
- 任何非 `dual_clock_euler + native_flow + shift12/3` 组合。
- Frozen Cache 和所有 transformer wrapper/patch-replace 组合。
- file MP4、decoded AUDIO 或 waveform 不能伪装为可恢复的 H3 audio latent。

### 12.3 不破坏旧工作流

- 不改 `MiniMaxH3AudioConditioningT8` schema、widget 或输出顺序。
- 不改 `MiniMaxH3DualClockSamplerT8`。
- 不改 `sampling.py` 的既有函数语义；第一版只调用和组合它们。
- 不改位置 `0..210` 的 class list。
- 三节点只在 `nodes.py` 当前最后一个已发布 class list 后追加。
- 第一机械阶段不修改任何现有 workflow JSON。

## 13. 串行低负载 TDD 验收

实现阶段必须先写失败测试，并严格串行执行：

1. 三个 schema、node_id、自定义类型和输出类型。
2. Audit 接受正确 nested AV shape，拒绝 batch/channel/rank/finite 错误。
3. `lock_source`、audio mask 全 0、fractional mask、`remix_source` 和 protected audio 的决策。
4. media map、conditioning report 和强内容哈希的确定性与篡改拒绝。
5. MODEL 对象与 patch 结构绑定，未知 wrapper/patch-replace 弃权。
6. 资源遥测通过、未知、VRAM 低和 commit 低四条分支。
7. Plan 与当前 ComfyUI KSampler partial-tail 在多组 `N,d` 上逐值一致。
8. 默认 4/0.5 的 video/audio sigma 精确参考。
9. Plan 不能升级 Audit 决策，descriptor 篡改必须拒绝。
10. Setup 只装配不采样；执行 Setup 后 model forward 计数为 0。
11. 同 seed noise 逐值一致，不同 seed 不同。
12. video mask 全 0、audio mask 全 1，samples tensor 未被原地修改。
13. partial-start audio rebase 与现有 `tests/test_sampling.py` 参考一致。
14. 通过真实 `SamplerCustomAdvanced` 的 stub 模型验证视频 latent 逐值一致。
15. ABSTAIN no-op 的 forward 计数为 0、输出 latent 逐值一致。
16. REJECT 在 clone/noise/mask 分配前失败。
17. 新节点位置为 211、212、213，总数为 214。
18. 旧位置 `0..210` 的 node_id 顺序、schema 摘要和全部现有 workflow JSON 字节不变。
19. Ruff、py_compile、目标 pytest 和全量低负载项目测试通过。

机械测试不加载 H3 大模型、不运行 CUDA、不生成视频、不并发、不做压力矩阵。

## 14. 机械实现后的唯一真实 smoke

只有第 13 节全部通过后，才允许执行一条低负载真实 H3 smoke：

- 单条清晰中文对白。
- 低分辨率机械验证画布，不作为画质结论。
- 首遍 Turbo4，精修 4 NFE，audio denoise 0.5。
- shift 12/3、CFG1、同一 MODEL/conditioning/media-map。
- 不启用 Frozen Cache、Block Cache、Sage/Sol、STG、EAV、SLA 或其他 wrapper。
- 只运行一次，不做压力或重复矩阵。

该 smoke 只验证执行、媒体可解码、video latent 冻结和音频非空；音质价值仍必须在后续 0.6～0.7MP 公平 A/B 与人工试听阶段确认。

## 15. 后续阶段边界

以下工作在三节点机械版之后单独开发，不进入同一提交：

- 同栈 Turbo 与显式 base MODEL 的公平 A/B。
- Turbo4、普通 8 NFE、同栈 4+4、base 4+4 对照。
- Quality Gate 与原始候选自动回退编排。
- `18-audio-refine` 日期前缀工作流和多个 NOTE。
- README、features 和验证报告的正式功能说明。
- CER/WER、speaker embedding、响度、削波、DC、声道和同步辅助指标。
- 时间范围、Long Video、Prompt Relay。
- Frozen Video Cache。

Frozen Cache 必须经过独立 Windows 内存门、hook 生命周期、真实 block 命中、精确路线对照和异常释放审计后，才可成为另一个 Advanced EXP 候选；它不属于本设计的初版实现。

## 16. 设计批准门

本设计批准后才执行以下顺序：

1. 写 `docs/superpowers/plans/2026-08-25-minimax-h3-audio-refine.md`。
2. 以失败测试开始，串行低负载实现三个节点。
3. 证明位置 `0..210` 和旧工作流不变。
4. 完成机械回归。
5. 执行唯一低负载真实 smoke。

批准本设计不等于批准 Frozen Cache、默认工作流推广、音质宣传或自动覆盖原始音频。
