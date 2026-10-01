# Audio Refine：独立尾采效果接口（本地 EXP）

这组新增接口接在已有 Audio Refine Setup 和 Stage Bind 后面。旧十种配方、旧节点、原二十张冻结／冷恢复工作流不替换；原来的 NOISE、SAMPLER、SIGMAS、联合 AV 和视频 0／音频 1 遮罩继续使用。没有新增隐藏采样器，也不重跑冻结的视频生成阶段。

## 接线

```text
原 Setup → 原 Stage Bind ── NOISE / SAMPLER / SIGMAS / stage_latent ───────┐
                 │                                                    │
                 └→ External Effects Context → [外置 Relay] → Stage EAV│
                                                positive ─┐       │    │
                                                          └→ Effects Guider
                                                                 │    │
                                                                 ▼    ▼
                                                       SamplerCustomAdvanced
                                                                 │
                                             Stage EAV Audit → 原 Stage Audit
                                                                 │
                                                原 Delivery Audit → 原 Quality Gate
```

- `MiniMaxH3AudioRefineEffectsBindEXPT8` 接原 Stage Bind 的 `stage_boundary` 及六个采样输入，输出独立 MODEL 和 `T8_STAGE_CONTEXT`。不改采样参数。
- 现有 `Stage EAV Config`／`Stage EAV Apply`／`Stage EAV Audit` 分别保留为画布节点。Apply 的 SIGMAS 和 AV 必须来自同一个尾采边界。
- `MiniMaxH3AudioRefineEffectsGuiderEXPT8` 接效果后的 MODEL、上述 context／SIGMAS／AV。`positive` 不接时沿用原 Setup 条件；接时检查原媒体／时间线及 Relay 的 MODEL–CONDITIONING 配对。输出连接原 Core `SamplerCustomAdvanced.guider`，CFG 保持 1。
- 可在未绑定 Relay 的尾采 MODEL 上使用现有独立 Relay Plan／Conditioning，再将其 MODEL 接 EAV、positive 接 Effects Guider。Relay 新建的空 AV **不能**替换冻结的尾采 AV。
- 原本已有 Relay 的两种配方，继续修改冷恢复图中独立的精修 Relay Plan；不要向同一 MODEL 重复叠加两个 Relay Conditioning。任意后置替换已装 Relay 的便利节点尚未实现。
- 长视频必须连接实际 `segment_index`／`context_frames`，首段为 0／0；续接上下文支持原有 5／22／39。接口核对实际 motion keyframes 与原长视频修复合同，不重算时间坐标或续接音频。

## 边界与兼容

ABSTAIN 仍为空 SIGMAS、零模型调用、原 AV 直通；即使 EAV 选择 `apply_exp`，也不会把 ABSTAIN 升为采样。效果节点不会接受候选音频，原 Quality Gate 默认保留原片，人工听审和明确选择仍必需。视频的逐值回锁仍由原 Quality Gate 完成，不把视频零遮罩当成采样输出逐位不变的保证。

EAV 的原定义是对目标视频 attention 输出施加 FETA 增益。它在联合 AV 尾采中的执行不等于“专门音频增强”或已证实音质更好。进度仍取 `1 - video_sigma`，不会按四步尾采重新归一化；原 `g_hard_limit` 保护保留。很短的测试网格可能触发保护，不会自动钳制增益或关闭报错。

未知用户 LoRA、attention delegate、hook 保留；实际覆盖不足按原 EAV 审计标为未验证。这组接口不授予可移植 StageResult 缓存资格；恢复继续走已有最终视频 AV 的显式 checkpoint 路径。

## 当前验收范围

已取得三类签名计划及十种尾采配方的 tiny 原生 H3 CPU 效果调用证据，包括长视频首段／续接、Relay＋EAV、disabled／report-only 数值不变、ABSTAIN 零调用、错误 AV／遮罩／SIGMAS／配对拒绝。CLIP／VAE 和权重使用标明的测试替身，这不是预训练 GPU 成片或音质验收。

`tools/build_modular_audio_tail_effect_workflows.py --matrix --output <新的artifacts目录>` 生成26张私有效果草稿：十张 EAV、另外八种配方各一张 Relay-only 和 Relay＋EAV。逐一校验来源图，不覆盖公开文件。已有 Relay 的两图保留其独立 Plan。Relay-only 不包含 EAV 节点；所有新增 Relay 都保留原全局提示词，默认空事件、`report_only`、`video_only_paper`，不会自动添加新剧情。需要实际时间路由时，填写至少两个局部事件，再显式选择 `apply_exp`；独立 Query Route 可选 `joint_av_exp`，它是实验性联合音画扩展，不是论文质量保证。Relay 的空 AV 输出不连接尾采。

这26张图已增量保存在 [59目录的effects子目录](../examples/workflows/59-audio-refine-split/effects/README.md)，旧20张不变。与交付图仅有 CRLF／LF 换行差异的候选26/26原生画布可见编辑／另存／刷新重开通过，未排队；八种新增配方的已保存图经 tiny Core 实执行，验证独立 Relay 的真实调用及 EAV 组合，八条还各在全新 Core 进程仅运行尾采。画布连线序列化、CPU尾采、真实权重及人审是不同资格。S26 尚未完成。

### 原尺寸 PDD8 原生画布完整／冷恢复补验（2026-09-30，本地）

新增一条独立链使用原 PDD8 配方尺寸：124 帧、736×416、24fps，约 5.167 秒。真实 Chromium 画布点击 Queue，完整执行 8 步视频生成＋4 步尾采并显式写入原生 AV checkpoint；全新 Core 严格读取相同路径／manifest／文件 SHA，仅执行 4 步尾采。模型、原参考图、采样器、sigma、3D／音频政策与原公开 JSON 保留，没有搬入历史 tiny 完成阶段。

组合尾采保留原 EAV 窗口、tau4 和 report-only 默认，四次认证 forward、200 次配对 Relay 调用已观察。分别保存原版、未经接受的候选、默认交付三路完整 H.264＋AAC；三路解码画音签名、候选／默认交付的 AV 张量内容都在完整／冷恢复间一致，原冻结文件不变。Quality Gate 仍为 `ABSTAIN_HUMAN_REVIEW_REQUIRED`，默认交付的原视频和音频逐值不变，解码画音与原版相同。候选的解码画面和音频与原版不同，不能把零视频遮罩／小数值差当成候选视频逐位不变或质量通过。

十种原配方的四类效果选择另有当前实际 Core／ClipProj 0.1.13 schema 准备检查：40 组完整／冷图的具名边、独立编号和 DAG，加 8 项错误连线／质量自动接受等拒绝门。它不加载权重、不 Queue，不替代剩余配方的原尺寸成片。其余路线、任意 `apply_exp` 强度／窗口、多素材和人工听审继续单列；旧失败证据保留。以下两条小尺寸结果仍只具有原来的历史资格。

### 十种原配方的分项 full／fresh-cold 补验（本地，2026-10-01）

十条原配方目前各自取得真实 Chromium 画布 full/fresh-cold 三路完整 AV 收据：PDD8、PDD 4+4、Phase2 base/same、Turbo4/8、EAV Turbo8、H3 learned upscale、Relay 与 long Relay。它们沿用各自原尺寸、步数、sigma、VAE、音频和交付区间；PDD 4+4 原来只有22帧，long Relay 原120帧，其余124帧，不把短配方改成五秒。冷图只执行原尾采，不重新生成已冻结的视频。各条候选／默认 AV tensor 和原片／候选／默认的完整解码 RGB/PCM 都在 full/cold 间一致，默认选择仍严格保留原片，没有候选被接受。

这些分项发生在不同冻结源码 epoch，仅覆盖独立外置 Relay apply 加 EAV report-only，不是统一当前源码的全效果矩阵、任意 apply 强度／窗口、更多素材或音质通过。Relay 这次明确预留4GiB，原512MiB余量和16GiBcommit保护未降低；Turbo8 保留原2.5GiB和stock保存器。旧资源／codec失败和证据保留，codec根因尚未证实。所有 `apply_exp` 和人工完整看听仍各自验收，不把候选视频零遮罩或成片可解码当作质量保证。

### 历史两条 PDD 小尺寸冷尾采

2026-09-28 补充了 PDD8（128×128）和 PDD 4＋4（192×192）两条来源的验证。两份源 AV 均来自先前已经核验的真实 PDD 视频生成及 checkpoint；本轮不重跑这些前段。每次在新隔离 Core 中，核对公开分离图与当前 Core 序列化结果，严格读取原冻结文件，仅执行四步尾采。模型、Qwen 编码器、VAE 使用本机真实权重，没有 tiny loader 或资源快照替身。

两条都使用独立 Relay Plan、`joint_av_exp`、`apply_exp`，EAV 效果窗口明确为 0–1。初次 `tau=0.25` 的 200 次 FETA 测量全部为增益 1，只证明路由执行。随后以 `tau=4` 补测，并要求观测到非单位增益：两条均记录 4 次模型前向、200 次 Relay attention 和 200 次 FETA 测量，最大增益分别约 1.2002／1.2482，原 `g_hard_limit=1.5` 保持不变。这不是全部默认配置或增强质量认证。

每次分别保存原片、未经接受的候选片、默认选择片，三份 H.264＋AAC 媒体均完整解码，均为 22 帧／24fps（约 0.917 秒）。默认 Quality Gate 保留原片，两者解码后画音 SHA 相同；候选音频变化且有限，两条候选的视频解码 SHA 也与各自原片相同。没有自动接受候选音频。

这次是隔离 API 的真实权重执行证据，**不是新增画布点击执行，也不是 5 秒成片**。公开工作流和原采样控制不变。原尺寸、多素材、十配方完整生成与冷恢复对照及人工听审仍需继续，不能以这两条小尺寸尾采代替。

### 已有 Relay 两配方的完整／冷恢复 CPU 对照

普通 Prompt Relay 与 Long Video Prompt Relay 的已保存效果图，另有 tiny CPU 整图对照：保留上游已经绑定的 Relay，不新增第二个 Relay；将原冻结图的最终 AV 直接接入效果尾采，完整执行 8＋4 步，再在全新 Python／Core 进程严格读取同一 checkpoint，仅执行四步尾采。覆盖 EAV disabled／report_only／apply_exp；普通配方还覆盖真实 rank-1 测试 LoRA 的 0／0.7 强度，共九种组合。

九组合的原始／候选／默认选中 AV 哈希，以及三路完整解码媒体哈希，均在完整图与冷恢复之间相同。默认 Gate 保留原片；长视频续接输出也保留原 AV，磁盘 context 与冻结文件未被尾采改写。长视频媒体比较采用相同的原 Output Trim 区间，不能把裁切前音轨与裁切后音轨混作同一对照。相同权重设置在完整图内由 Core 正常缓存共享，执行步数由真实 Relay 调用计数核验，不能以“只载一次模型”推断漏跑前段。

模型为固定随机小网络，CLIP／VAE／资源信息明确用测试替身；VAE 解码随 latent 内容变化，非恒定黑帧／恒定音轨。这里只证明 CPU 状态与输出恢复合同，长视频仅首段 segment 0。它不替代这两配方的真实权重、多段续接、原尺寸或人工画音验收。
