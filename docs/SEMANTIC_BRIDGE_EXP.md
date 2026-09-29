# Semantic Bridge / BUNNY — v1.84.0

## T8 动漫战斗训练权重（后续源码兼容）

[T8 Comic Combat Semantic Bridge](https://huggingface.co/t8star/semantic_bridge_T8-comic-combat) 是另行训练的跨 token Transformer 权重，结构不同于下文的六张量 MLP。将 `semantic_bridge_T8_comic_combat.safetensors` 放入 `ComfyUI/models/semantic_bridge/t8_compat/`，使用本仓库最新 GitHub 源码，重启 ComfyUI 并刷新模型列表。节点为 **H3 Semantic Bridge / 模型与设置 (T8 EXP)** 和 **H3 Semantic Bridge / 应用条件 (T8 EXP)**；普通工作流按「原生 H3 CONDITIONING → 应用条件 → 原采样条件输入」连接，配置节点接应用节点的 `semantic_bridge`。Relay 或内循环使用各自的内置 `semantic_bridge` 插口，避免重复应用。

该权重的固定应用契约为 `alpha=1.0`、`magnitude_match=per_token`、`token_scope=all_tokens`、`chunk_tokens=0`（整条 token 序列一起计算）。配置不符会报错；0.10／256 是下文六张量 MLP 的默认值，不适用于此权重。它只改变文本条件，不是视频 LoRA；部分同提示词视频 A/B 观察到局部改善，不保证所有镜头有效。[T8 训练器](https://github.com/T8mars/MinimaxH3_Semantic_Bridge_Trainer-T8)导出的 `trans` 和 `mlp` 桥结构也由本节点读取，其他自训权重按各自的模型卡设置。软件兼容不等于生成效果已验证。

The T8 comic-combat model is a separately trained cross-token Transformer, not the six-tensor MLP below. Install it under `ComfyUI/models/semantic_bridge/t8_compat/` with the latest GitHub source. Set `alpha=1.0`, `magnitude_match=per_token`, `token_scope=all_tokens`, and `chunk_tokens=0` (whole sequence). Other settings fail the model's fixed application contract. Connect the T8 Config and Apply nodes as above, or use a Relay/loop's built-in Bridge input once. The node also reads `trans` and `mlp` bridge exports from the [T8 trainer](https://github.com/T8mars/MinimaxH3_Semantic_Bridge_Trainer-T8); use each weight's own inference settings. Software compatibility does not establish video quality.

## 2026-09-17 最新验收与发布范围

用户在8810复审后明确“可以了没问题了”，授权正式工作流及GitHub发布。
通过B的完整8秒成片SHA256为
`6da418003510e040d2b3ccc7d9961dcb8d6b652395c0b50425f326159348ce9c`。
推荐[正式双采图](../examples/workflows/34-semantic-bridge/2026-09-17_H3_SemanticBridge_DualIndependent_8s_Advanced.json)：
LOW640×320→HIGH896×448、4+原learned3D+4、两阶段Original/BUNNY均0.10；
画面、声音及接缝按这份指定样片验收，不扩大到所有素材或组合。
五份示例进入本版；其他四路仍作为EXP用法示例，不给未评项目追加通过。
H16／Meridian继续暂停。本节覆盖下方历史开发／待审文字；详见[发布说明](RELEASE_1.84.0.md)。

## 2026-09-17 独立完成范围及模型下载

本次仅接续 Semantic Bridge，不恢复 H16／Meridian。基于正式最新源码更新，保持原340节点顺序（包括Sol），末尾追加两个Bridge节点，总计342；只有可选插口追加，已保存图显式参数、EAV新建默认15%–90%、用户补丁警告放行及已验收接缝配方不迁移。

转换模型及完整安装说明：[t8star/Semantic-Bridge-Comfy](https://huggingface.co/t8star/Semantic-Bridge-Comfy)。示例使用 `models/semantic_bridge/t8_compat` 中原版FP16／BUNNY FP32无损封装，保留六张量数值，不是INT8、LoRA或新训练主模型。已有原作者文件仍兼容；模型选择相对路径必须对应实际文件。

示例位于 `examples/workflows/34-semantic-bridge`：外接BUNNY、内部原版、Relay、单模型8秒、双模型独立4+4共8秒。它们是EXP示例，不能冒称任意素材的画质保证。Python更新须重启ComfyUI后加载；不自动重启用户服务。取消／恢复使用原有链机制；改变配置换新chain_id。

本次集中人审已返回：其他八组未报告失败，双采戏院样片画面出现白色飞物；该组音乐／人声、口型与接缝可接受，未评项不算通过。旧双采448×224→896×448不是推荐画质示例。相同提示词、seed和4+4下，只关闭HIGH Bridge或关闭全部Bridge，完整8秒仍有漂浮光斑，不能认定是BUNNY独有问题或靠关闭二采修复。新的双采示例候选提高一采到640×320，二采仍为896×448，保留原版／BUNNY两份Bridge0.10、原learned3D放大器及原采样／音频／接缝参数；完整核验与人审须分别记录。此为固定测试的空间配置候选，不是适配器数学或放大器已证明有错。

最新实现、回归、Core及精确同步证据以本地 `artifacts/semantic-bridge-completion-20260917/verification.json` 的实际终态为准；旧研究树媒体／UI／包证据仅对应原冻结源码，不充当最新源码GPU或人审通过。下方历史339／341节点包描述不是当前342节点交付资格。功能仍为EXP，最终画音验收单独记录，不自动提交／发布Registry。

## 中文快速接线与参数

这是可选的条件语义适配器，不是 LoRA、放大器或速度优化节点。
原版与 BUNNY 是两份独立训练的权重，不要串联叠加。新功能仍待集中画音审片，
已有工作流不接这个输入时继续走原路径。

**普通工作流：** 原生 H3 条件节点的 CONDITIONING → Bridge应用条件 →
原来的 Guider/采样器；Bridge模型与设置 → 应用条件的 semantic_bridge。
原生条件节点也可直接接 semantic_bridge，这两种用法二选一，不要同时使用。

**Prompt Relay：** Bridge设置接 Relay条件节点的 semantic_bridge。
MODEL先进入 Relay，再在其MODEL输出后加载权重LoRA，然后采样。
不要对已经绑定Relay的CONDITIONING额外接一次Apply；global/local/时间线仍由原Relay计划提供。

**内循环：** Bridge设置直接接长视频节点的 semantic_bridge。
双模型4+4另外有 semantic_bridge_pass1 / semantic_bridge_pass2：
空槽继承公共配置；显式接 enabled=false 只关闭对应阶段的Bridge增强，**不关闭该阶段采样**。独立二采的MODEL／LoRA与4+4保持不变，不需要为了关闭增强删除二采节点或接线。
例如公共接原版、一采留空、二采接BUNNY配置，就分别使用两份模型；
一采与二采都是各自新编码后应用一次，不是把两桥连续应用到同一条件。

| 设置 | 用法 |
| --- | --- |
| model_name | 选择原作者文件或T8无损封装；放入 models/semantic_bridge，子目录也会列出。 |
| enabled / alpha | 默认启用、0.10；关闭或0强度完全旁路。先做同seed对照，不把提高强度当修复。 |
| magnitude_match | 默认per_token逐token匹配幅度；global按整条条件统计，none不匹配。不同模式不是等价效果。 |
| token_scope | 默认all_tokens。text_only_preserve_reference只改原生tag=1行，是独立实验，不能保证声音不退化。 |
| device | auto跟随条件张量；cpu/cuda仅高级覆盖。不会因此卸载其他模型或启动SenseNova教师。 |
| chunk_tokens | 六张量 MLP 默认256；跨 token Transformer 的训练契约可要求0（整条序列），正数会把它切为独立序列，可能改变结果。不改变视频长度。 |
| report_json | 记录模型内容身份、配置、编码来源、输入输出身份和应用次数，用于诊断，不是画质评分。 |

遇到“重复应用”时撤掉多余Apply并从原生条件重新编码；遇到Relay绑定错误时使用内部插口。
模型缺失时刷新列表并核实实际目录；不需要安装原作者/社区节点，也不会自动下载模型。
改变Bridge模型、强度、范围或采样配方后使用新chain_id，避免借用旧链结果。
测试内循环先用两段共8秒，从约5.17秒边界一直听看到结尾；首次检查中文对白、歌声、
背景、物体归属和人物身份。保留原音量，不用增益或替换音轨掩盖退化。

This opt-in feature is released in1.84.0 with the repaired dual recipe accepted
in its exact review scope. Historical development evidence below is not universal qualification.
Implementation lives in `h3_t8`. Existing models, sampling schedules, audio masks,
upscalers and accepted clips are not changed. No SenseNova teacher is required.

## Models

Use either the author's original safetensors or the lossless T8 Compat wrapper in
`ComfyUI/models/semantic_bridge/`. Extra model paths are also enumerated. Duplicate
relative names show their actual paths; the loader does not silently select one.

- [Semantic Bridge v1](https://huggingface.co/speach1sdef178/MiniMax-H3-Semantic-Bridge):
  original FP16 six-tensor 5120→512→512→5120 SiLU adapter.
- [BUNNY ActionLogic v1](https://huggingface.co/JOKER141/BUNNY_H3_Conditioning_Bridge):
  independently trained FP32 adapter, not a conversion of the first model.

`tools/convert_semantic_bridge.py SOURCE` inspects the six tensors. Add
`--output DEST.safetensors` to produce a non-overwriting lossless wrapper. Original
metadata is retained and provenance/per-tensor checksums added. Tensor values and
dtypes must remain identical. File SHA changes because metadata changes.

T8 Compat is not a new trained model and grants no new model redistribution license.
Retain author attribution, source revisions and upstream model terms. This implementation
is independently written from the published mathematical contract; it does not copy
the unlicensed community custom-node source into this repository.

## Connecting nodes

Native H3 conditioning → **Semantic Bridge Apply** → existing sampler.
**Semantic Bridge Config** feeds Apply's `semantic_bridge` input.
For Relay and in-node loops use their optional `semantic_bridge` socket instead of
applying externally after Relay. Dual-model loops additionally allow independent
`semantic_bridge_pass1` and `semantic_bridge_pass2`; unset overrides inherit the common
config, while an explicitly disabled override bypasses just that pass.

The controlled standalone Relay example uses base MODEL then downstream weight
LoRA; existing user patches remain allowed with compatibility advisories and are
not rejected merely for being unqualified. Select `apply_exp` to
actually route the timeline: `report_only` does not activate Relay. The candidate
Relay workflow uses this explicit base → Relay → LoRA order.

Start with alpha **0.10**, magnitude match **per_token**, token scope **all_tokens**.
The text-only option uses native `minimax_token_tags` (1=text, 0=visual) and refuses
missing/invalid tags. It is a separate experiment, not guaranteed voice protection.
Global magnitude matching uses the complete conditioning item, including all batches,
not separate chunk statistics. Scheduled items are processed independently.

Zero strength, disabled config or no connection bypasses unchanged without reading a
model or adding receipt metadata. Active application records content identity and
input/output tensor receipts, preserves other metadata and never mutates the input.
Do not stack adapters; use a fresh native encoding for each separate LOW/HIGH stage.

## Boundaries

- Native unprojected H3 `[B,T,5120]` embeddings only; width alone does not qualify a
  different encoder's training distribution.
- FP32 RMS / MLP / blending with original epsilon rules; output returns to original
  dtype/device. Auto follows the conditioning device. Token chunking bounds workspace.
- Ref2VA, Hybrid, AV continuation and singing are EXP: the author reports degraded
  singing/lip-sync in reference-audio experiments. Original audio passthrough does not
  prove generated audio is unaffected. No loudness normalization disguises failures.
- No new attention/model patches. Existing user patches are preserved with advisories;
  an allowed composition is not proof that all VSA/Relay kernels or optimizations execute.
- Changed weights/strength/scope invalidate bridge-enabled cache contracts; use a new
  chain when changing generation settings. Cache entries are content-keyed and CPU-only.
- Bound native-encoder short AV, internal/external Apply, active Relay, single-model
  and independent dual-model two-segment8s outputs have passed strict media checks.
  Completed-final cache reuse and actual single/dual new-process interrupted-stage
  recovery are verified for the fixed8s recipes. Both resumed MP4s exactly match
  their uninterrupted baselines; this is not an arbitrary-input guarantee.
  All new samples still require human picture/audio review. Three-way comparisons with
  the INT8 encoder, Mandarin first-frame I2VA and Korean image-reference Ref2VA have
  also completed their bound media checks. Hybrid Stock20 plus actual first frame
  and voice reference has completed native/original/BUNNY media checks; generated
  speech remains human-pending. Additional backend qualification is ongoing;
  these results are not universal support claims.
- Eighteen bounded CUDA checks cover two models, three magnitude modes and three
  input dtypes, CPU/CUDA error bounds, exact same-profile wrapper output, cancellation
  and reference-token preservation. These are math checks, not image-quality claims.

## Qualification and delivery

The latest source baseline preserves340 nodes including Sol; this release adds
exactly two Bridge nodes at the end,342 total. Legacy workflow files remain byte-identical. The historical FastH3 V2
package gate stays339-specific and must reject this341-node candidate. Test its old
contract against the pinned1.83.0 release rather than relaxing that release gate.
No new sample inherits old FastH3, SelfLift, audio or seam human acceptance.

Five candidate workflows have been opened and saved in an isolated native ComfyUI
frontend and exported as API JSON. After a full page refresh, all five native API
exports remained byte-identical; explicit widget values and execution edges match
the source recipes. Evidence: `artifacts/semantic-bridge/ui-native-v1/native-audit-v3.json`.
The evidence normalizes only the declared Chinese task-type display label and the
current Core's hidden legacy SaveVideo codec default. No graph execution or frontend
injection was used for this check. This is workflow serialization qualification,
not human sample acceptance or final example publication.

A representative trained FastH3 V2 + T8 head_chunks4/FFN chunks2 + original Bridge0.10
sample completed 73 frames at832×480/24fps with generated audio. Runtime audit records
400 actual VSA dispatches across eight sigma values, without a Dense fallback.
The media strictly decodes, but picture/audio and performance are not accepted by
this mechanical result. Evidence: `artifacts/semantic-bridge/p3-fastv2-vsa-h4c2-original-v1`.
This is not Sage/Sol qualification and does not relax trained-VSA/Relay restrictions.

For a first comparison keep the native encoder, same seed, model, LoRA, prompts,
resolution and sampler; compare no Bridge / original / BUNNY. Do not change the
encoder at the same time and attribute all changes to Bridge. Listening to original
reference audio does not test generated speech. Do not normalize away a quiet/noisy
failure. Use actual generated video/audio and listen to the entire second segment.
