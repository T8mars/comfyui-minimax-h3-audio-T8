# MiniMax H3 Audio Refine Phase 2：显式双模型路由设计

日期：2026-08-26  
状态：实现与三组人工盲测已完成，待合并发布

## 1. 目标与边界

本阶段只解决两个问题：

1. 在不改变现有四个 Audio Refine Advanced EXP 节点 schema、widget 顺序和旧工作流的前提下，允许首遍 MODEL 与精修 MODEL 明确分开。
2. 建立可审计的四臂基线：Turbo4 原始、无 Turbo 基座普通 8 NFE、Turbo4 + 同 Turbo 栈 Refine4、Turbo4 + 无 Turbo 基座 Refine4。

本阶段不开发 Frozen Cache、时间范围精修、Long Video、并发运行、压力矩阵或跨 GPU。节点仍不承诺音质、台词、音色、口型或音乐/音效无损。

## 2. 为什么不能只加一个 MODEL 插口

ComfyUI 的标准 LoRA Loader 把权重补丁写入 `ModelPatcher.patches`，并更新 `patches_uuid`；它不会把 LoRA 文件路径作为稳定、可证明的 MODEL 字段保留下来。只比较类名或补丁 key 数会把不同 LoRA、重复叠加和未知补丁误判成同一栈。

因此 Phase 2 使用运行期联合指纹，而不伪称能够从 MODEL 反推出磁盘文件名：

- 同一进程基座对象身份；
- `clone_base_uuid` 与 `patches_uuid`；
- 权重补丁 key、每 key 条目数、强度、offset、function 和 payload 类型签名；
- `lora_metadata` 的规范化 SHA-256；
- 模型、采样对象、object patch、transformer option 和 wrapper 结构；
- 与首遍 Audit 已签名的 MODEL 对象和结构绑定。

真实验证器另外记录磁盘模型与 LoRA 的绝对路径、字节数、mtime 和有界文件指纹。节点报告与验证器报告必须区分“运行期栈指纹”和“磁盘资产指纹”。

## 3. 新增节点（append-only）

### 3.1 Audio Refine Model Route

节点 ID：`MiniMaxH3AudioRefineModelRouteT8Advanced`

输入顺序：

1. `audit: H3_T8_AUDIO_REFINE_AUDIT`
2. `first_pass_model: MODEL`
3. `refine_model: MODEL`
4. `route_strategy: COMBO`
5. `declared_first_pass_nfe: INT`

`route_strategy` 只有：

- `same_turbo_stack`
- `base_without_turbo`

输出：

1. `refine_model: MODEL`（原对象透传）
2. `route: H3_T8_AUDIO_REFINE_MODEL_ROUTE`
3. `decision: STRING`
4. `report_json: STRING`

路由节点必须重新校验 Audit 签名，并验证 `first_pass_model` 正是 Audit 绑定的对象与结构。运行期栈无法识别时返回 `ABSTAIN`，描述符被篡改或对象合同变化时抛出 `REJECT_*`。

### 3.2 Audio Refine Phase 2 Plan

节点 ID：`MiniMaxH3AudioRefinePhase2PlanT8Advanced`

输入顺序：

1. `route: H3_T8_AUDIO_REFINE_MODEL_ROUTE`
2. `refine_steps: INT`，当前固定 4
3. `audio_denoise: FLOAT`，只接受 0.35 或 0.50
4. `refine_seed: INT`

输出：`plan / decision / report_json`，其中 plan 类型为 `H3_T8_AUDIO_REFINE_PHASE2_PLAN`。

`audio_denoise=1.0` 不进入该基线节点；如后续研究，必须新建明确标注“full audio regeneration”的独立 EXP 路线。

partial-tail 公式与 Phase 1 完全相同：

```text
M = int(refine_steps / audio_denoise)
b(k) = (refine_steps - k) / M
sigma_video(k) = 12*b(k) / (1 + 11*b(k))
sigma_audio(k) = 3*b(k) / (1 + 2*b(k))
```

报告同时写请求 denoise、有效 denoise、首遍实际/声明 NFE、精修实际 NFE、总 NFE和双栈指纹，并明确“总 NFE 相同不代表训练分布相同”。

### 3.3 Audio Refine Dual-Model Setup

节点 ID：`MiniMaxH3AudioRefineDualModelSetupT8Advanced`

输入顺序：

1. `plan: H3_T8_AUDIO_REFINE_PHASE2_PLAN`
2. `refine_model: MODEL`
3. `positive: CONDITIONING`
4. `av_latent: LATENT`

输出顺序与现有 Setup 完全一致：`MODEL / NOISE / GUIDER / SAMPLER / SIGMAS / LATENT / report_json`。

Setup 必须重新验证：

- Plan、Route、Audit 三层签名；
- refine MODEL 对象及运行期栈指纹；
- positive conditioning 与 Audit 的 run contract；
- 首遍 AV latent 完整内容 hash；
- 执行前 VRAM、RAM 和 Windows commit headroom；
- 固定 CFG 1、12/3 双时钟、`dual_clock_euler`、`native_flow`、video mask 0、audio mask 1。

`ABSTAIN` 输出零长度 SIGMAS 和原始 latent，不启动模型采样；`REJECT` 不进入采样。

## 4. 栈识别规则

### 4.1 首遍 Turbo4 栈

当前基线只接受可识别的单层 Turbo4 LoRA：

- 权重补丁非空；
- 每个补丁 key 恰有一个条目，禁止重复叠加；
- strength_patch 与 strength_model 均为 1.0；
- offset 和 function 为空；
- `lora_metadata.base_model` 为 MiniMax-H3；
- `lora_metadata.sampler_steps` 为 4；
- `conversion_source_sha256` 是合法 64 位十六进制；
- 没有未知 wrapper、patches_replace 或已列入禁用清单的 runtime marker。

任何一项无法证明时 `ABSTAIN_UNKNOWN_FIRST_PASS_STACK`，不靠名称猜测。

### 4.2 `same_turbo_stack`

要求：

- 两端共享同一基座对象；
- 权重补丁结构 SHA、`patches_uuid` 和 LoRA metadata SHA 完全相同；
- `clone_has_same_weights()` 可用时必须返回 true；
- 允许一端额外带经审计的 T8 12/3 sampling object patch，因为该补丁不改变权重栈。

### 4.3 `base_without_turbo`

要求：

- 两端共享同一基座对象；
- 首遍端通过 Turbo4 单层栈规则；
- 精修端权重补丁数为 0；
- 精修端没有 `lora_metadata`；
- 精修端没有未知 wrapper、patches_replace、附件或 runtime marker。

这条路线的含义只是“同一已加载基座对象、移除全部权重补丁”，不能宣传成官方非蒸馏 8 步质量保证。

## 5. 四臂公平基线

固定素材：清晰中文对白，0.6～0.7MP，124 帧，24fps，相同 prompt、seed、conditioning。

四臂：

1. `turbo4_original`：已有工件，4 NFE。
2. `base_ordinary8`：同一无 Turbo 基座，单次 8 NFE；这是成本匹配控制，不是训练分布匹配控制。
3. `turbo4_same_turbo_refine4`：已有工件，4 + 4 NFE，0.50。
4. `turbo4_base_refine4`：只补缺失臂，首遍 latent 必须与 arm 1/3 内容合同一致；4 + 4 NFE，先用 0.50。

已有 arm 1/3 不为凑矩阵重复运行。若旧工件没有可恢复 latent，只允许在同一新 prompt 内生成一次共享首遍 latent，同时把它标为依赖，不把重复首遍结果伪装成新的独立样本。

## 6. 机械验收门

进入用户盲听前，每个候选必须：

- ComfyUI 前端工作流可导入，不是 API-only JSON；
- 实际 NFE 与报告一致；
- 严格 FFmpeg 解码无警告；
- 1056×608 或同等级 0.6～0.7MP、124 帧、24fps；
- 音频 32kHz 双声道、finite、非空，无声道坍塌；
- 无 NaN/Inf，削波、DC、响度和频谱漂移只做提示；
- Quality Gate 最终视频严格取原始 video latent；
- 串行单任务，运行前资源门通过；
- 报告记录真实模型栈、LoRA 栈、NFE、seed 和工件 SHA。

四臂后最低再做一条音乐+环境+瞬态混合素材，以及一条 I2VA 或 Ref2VA 说话素材。自动指标不选冠军，最后才交用户匿名试听。

## 7. 兼容与发布约束

- 旧 215 个节点 ID、顺序和 schema 原样保留；新节点从 215 起追加。
- 不修改现有 Audio Refine 四节点 widget 顺序。
- 不修改既有工作流；新建日期前缀工作流。
- stable `sampling.py` 和 EXP multirate 实现不因本阶段改写。
- 用户最终盲测已完成；发布时保留人工 Quality Gate，不把候选自动设为默认。
