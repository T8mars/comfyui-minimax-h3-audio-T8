# MiniMax H3 Audio Refine Phase 2 实施计划

日期：2026-08-26  
执行方式：串行、低负载、TDD；三组最终盲测已完成，待合并发布

## Task 1：锁定现状

- [x] 读取项目 SKILL、roadmap、meta、features、README、相关源码与测试。
- [x] 运行 Audio Refine + stable/EXP sampling 基线：76 passed。
- [x] 核对现有 0.64MP 工件、模型/LoRA、prompt、seed、NFE、资源与私有盲审记录。
- [x] 确认 ComfyUI 标准 LoRA Loader 不保留可证明的文件路径，但保留 patch UUID、补丁结构和 metadata。

## Task 2：先写 Phase 2 红灯测试

修改 `tests/test_audio_refine_advanced.py`，新增：

- 三个新 descriptor/type/schema 常量测试；
- single Turbo4 栈识别、重复 LoRA 和未知 metadata fail closed；
- same stack 接受同权重 clone，拒绝不同 patch UUID/metadata；
- base-without-Turbo 只接受同基座零补丁模型；
- Route/Audit/Plan 篡改拒绝；
- 0.35/0.50 可用，其他 denoise 与非 4 NFE 拒绝；
- Setup 重绑 refine MODEL、conditioning、latent 和资源门；
- ABSTAIN 零步旁路；
- 新节点 schema 追加且旧四节点输入输出顺序不变；
- 全注册顺序由 215 增至 218。

先运行新测试并确认因符号/行为缺失而 RED，不接受语法或环境错误作为红灯。

## Task 3：最小领域实现

修改 `audio_refine_advanced.py`：

- 新增 route/phase2 plan schema 与 type；
- 新增规范化 runtime weight-stack fingerprint；
- 新增 Turbo4 单层栈识别与两种 route 验证；
- 新增三层签名 descriptor；
- 抽取 conditioning/latent 重绑逻辑，保持旧 Setup 行为不变；
- 新增 Phase2 Plan 与 Dual-Model Setup；
- 复用 stable `setup_dual_clock_sampling`，不修改 stable/EXP sampler。

每完成一个行为只运行对应小测试，再运行完整 Audio Refine 测试。

## Task 4：追加节点并注册

修改：

- `nodes_audio_refine_advanced.py`
- `nodes.py`（只通过原列表自动追加，不重排其它节点）
- `features.json`
- `tests/test_audio_refine_advanced.py`
- `tests/test_preflight_and_registration.py`（如存在固定总数）

新增顺序固定：

1. `MiniMaxH3AudioRefineModelRouteT8Advanced`
2. `MiniMaxH3AudioRefinePhase2PlanT8Advanced`
3. `MiniMaxH3AudioRefineDualModelSetupT8Advanced`

验证旧 0..214 前缀逐项相等。

## Task 5：四臂前端工作流与验证器

新增日期前缀 ComfyUI 前端工作流，不覆盖旧工作流：

- 四臂总览/生成工作流；
- same stack 与 base refine 路线 NOTE；
- ordinary8 成本控制 NOTE；
- Quality Gate、风险与人工试听 NOTE。

扩展或新增低负载串行 runner：

- preflight 只读；
- 固定单进程、单 prompt、单 H3 任务；
- 记录磁盘资产路径/字节/mtime/有界文件指纹；
- 记录模型运行期双指纹和真实 NFE；
- 严格解码与 AV 机械门；
- 不自动选择听感冠军。

## Task 6：机械验证

先运行：

```powershell
$env:PYTHONPATH='F:\AI-T8-video-onekey\ComfyUI'
F:\AI-T8-video-onekey\python\python.exe -m pytest -q `
  tests\test_audio_refine_advanced.py `
  tests\test_audio_refine_workflow.py `
  tests\test_sampling.py `
  tests\test_sampling_multirate_exp.py `
  tests\test_preflight_and_registration.py --maxfail=1
```

随后运行 compile/ruff、frontend workflow validator、registry prefix gate。不得启动 CUDA。

## Task 7：真实四臂补齐（严格串行）

资源门满足后：

1. 复用已有 Turbo4 original 与 same-stack Refine4 媒体和盲审事实。
2. 只补 `base_ordinary8`。
3. 只补 `turbo4_base_refine4`；若无法恢复首遍 latent，只在同一任务内生成一次共享 Turbo4 latent，并明确记录为依赖。
4. 每次只跑一个 H3 prompt，完成并释放后再开始下一条。
5. 基础机械报告通过后生成四臂匿名试听页。

用户只在最后进行人工盲测。

## Task 8：最小外推验证

四臂机械门通过后，串行补：

- 1 条音乐 + 环境 + 瞬态混合音频；
- 1 条 I2VA 或 Ref2VA 清晰说话。

仍只比较必要候选，不扩成压力矩阵。失败项记录到 roadmap，不伪称完成。

## Task 9：暂停/交付门

- 三组最终盲测已经完成，允许合并 main 并发布。
- 质量门仍默认回退原始音频，用户试听后再接受候选。
- 最终报告列出通过项、失败项、未知项、资源数据与本地试听入口。
- 只有用户明确批准后再执行提交/合并/推送。
