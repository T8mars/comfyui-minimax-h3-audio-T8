# Veda R2VA 独立节点（EXP）

当前交付：实现及指定已审来源图随[v1.98.0](RELEASE_1.98.0.md)发布。下方“尚未发布”等表述是来源验收时的历史状态，资格范围不扩大。

本页对应新增 R2VA 预测器接入，不替换已有 T2VA Veda 或 Heuristic TripPool64。
当前已完成权重读取、CPU 合同、小规模 CUDA 数值检查和真实画布5秒 Dense／Sparse 成片；2026-10-11集中反馈revision15的A1已通过这一代表案例，不扩展为所有配方认证，不承诺提速或修复双采闪烁。[Dense／Sparse短名图](../examples/workflows/86-h09-reviewed/README.md)已保存，保持原参数，资源适配版未重新采样，尚未发布。

## 模型与目录

下载 [官方 R2VA Preview](https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview)，将
`minimax_h3_r2va_veda_preview_fp8.safetensors` 放入 `ComfyUI/models/veda_r2va/`。
此文件约275MB，是辅助 attention tile-score 预测器，不是 H3 底模或 LoRA。
原件已是 safetensors，无须转换 H3 checkpoint；加载器按真实 `.__scale` per-head scales 解量化为 BF16。

接入核定 revision：`1598b407905b7c0af85aa0c13acdfa6f01ff2982`；SHA256：
`23edace948693772be33d5cc945d302d2b037a29bd44bd57071bfbdd8b02517d`。
加载器检查50层、56头、128维、全部投影／scales、两组预算和嵌入 plans。
未知但格式合法的其他预测器允许加载并提示未验证；错误 dtype、非有限值、变更文件或损坏缓存仍拒绝。

旧预测器继续放 `models/veda_scorers`，旧节点 ID、类型、默认值和旧工作流不变。
不要把新 R2VA 文件塞进旧 T2VA Bundle；新 header 没有旧格式的 `keep_ratio`。
H3 Ref2VA 底模、Qwen、视频／音频 VAE，以及 Turbo LoRA 仍需另外准备。

## 三个独立节点与接线

类别：`T8/MiniMax H3/Acceleration/Experimental`。

1. `H3 Veda R2VA · Bundle (EXP)`：从 `models/veda_r2va` 读取预测器。
2. `H3 Veda R2VA · Apply Flex Sparse (EXP)`：接标准 MODEL 和 Bundle，输出克隆 MODEL 与 runtime。
3. `H3 Veda R2VA · Audit After Sample (EXP)`：接采样器真实 LATENT 和对应 runtime，再接后续解码／保存；输出审计 JSON。

基本接线：`H3 Ref2VA Loader → LoRA／兼容内存补丁 → R2VA Apply → Guider／Sampler`；Bundle 接 Apply；采样后 LATENT 经过 Audit，再 AV Decode／Save。
Apply 放在模型与 LoRA 后，不改原模型权重。旧图不必迁移；删除／旁路新增 Apply 即可回到原路线。

双采可在 LOW／HIGH 的 MODEL 分支分别放 Apply，**每个 Apply 使用自己的 runtime／Audit**，各自绑定实际网格。
EAV／Prompt Relay 沿用已有外置节点，不合并进新采样器。已有 attention／DiT owner 未确认兼容时保留并警告／委托，可能没有 sparse 覆盖，不能仅凭节点接上就称加速。
MASK／Relay bias 和 partial-query 调用走原 dense delegate，不丢 mask，不伪报 sparse 次数。
目前独立完整8步代表例已获上述人工接受；4+4、长多窗以及全部补丁组合没有新增质量资格。

## 参数

|参数|默认与含义|
|---|---|
|`mode`|`report_only`：保持 dense，观察布局；主动选择 `apply_exp` 才启用新 sparse 路径。|
|`plan_mode`|`strict_exact`：必须有 exact plan；`nearest_exp` 是显式实验借用最近 plan，不 resize、不偷偷放宽旧 T2VA。|
|`generated_budget`|`trained`：读取预测器生成区预算，官方为32个 key tiles。|
|`reference_budget`|`trained`：读取预测器参考区预算，官方也为32个 key tiles，与生成区**分别 top-k**。|
|`max_copy_mib`|默认64，限制分组 head gather 的估算复制预算，不是整个模型的显存上限。|

两种预算也可输入1–4096的整数 tile 数；`dense` 或 `0%` 表示该区全保留，不是保留0个 tile。
预算32是32个128-token key tiles，不是32%、32步或32帧。
小参考图不足32 tiles 时全部参考 key 可能被选中，但这**不等于** reference=`0%` 的双向 global dense。
文本、VLM及所有音频段保持双向 dense；这不保证生成音频 latent 或最终音轨与 Dense 完全相同。
不能验证真实 Cartesian row-major 网格的参考段保留 global dense，审计说明原因，不猜网格、不重写原时钟。

官方 config 的 R2VA 微调长度为 T37；包中12个 plans 不等于12组本地验收或全部长度都已微调。
864×480的124帧输入对应实际 `(37,15,27)`，需显式 `nearest_exp` 借用 `16x9_t37` 的 `(37,24,42)`。
审计分别记录 actual_grid／plan_grid／exact；不为 exact 而自动升到更高分辨率。
官方配方是完整8 NFE Turbo，文件说明的600是训练更新次数，**不是采样步数**。

## 执行、安全与审计边界

本接入复用 T8 PyTorch FlexAttention，采用官方 R2VA 预测器和多参考 span／独立预算语义；
**不是上游 Triton INT8 内核，也不继承上游 INT8 的性能数据**。不强换 Torch/CUDA 或额外安装依赖。
本机 Windows 启用 `apply_exp` 需 UTF-8 环境和 `--disable-dynamic-vram`，否则在改 MODEL 前报明确错误；不要在其他任务采样中擅自重启共享后端。

Audit 记录真实调用、sparse／delegate 次数及原因、实际布局、两预算、plan与实现内容、已载入预测器参数、派生 tile／head 缓存完整性。
实际执行函数／compiled Flex／predictor hooks 被未知补丁替换时仍允许运行，但记录 `unverified_sparse_calls`，不冒报原内核 sparse 覆盖；仍存活的原生 layout／position_ids 被改动会使审计失败。
`executed_sparse_kernel=null` 表示没有实际 sparse 执行，不能按请求配置推断已加速。
新 `t8.veda-r2va.execution-local-content.v1` 仅是执行内审计，**不赋予 portable Stage／冷续跑资格，不沿用旧 T2VA 资格或旧缓存**。
未经确认的组合仍可运行原 owner，但不要把 execution-local 结果跨进程冒充已验证可重用 Stage。

已做的最小检查：17个新 CPU 合同；实际官方100个投影解量化逐值一致、调用者 RNG 不变；4060 Ti 上1层2头、23,308 tokens、参考64 tiles与目标多 tiles的独立32/32稀疏核对，抽样13个 query rows 对 FP32 token-level sparse SDPA 最大误差约0.000261。
这些是模型解析／算子证据，不是 H3 完整采样、真人质量通过、全部 GPU 资格或端到端加速测量。

2026-10-10 已完成最小媒体对照：相同参考、seed、正常中文对白“你好，今天天气不错。”，864×480（0.41472MP），完整8步，Dense／Sparse各一条完整5秒原生音画。
两图均实际保存、导出 API、关闭后从已保存工作流重开，再点击 UI 运行；20个执行节点全部输入／接线与成功 frontend history 一致。两片均完整120帧／24fps／5秒及有限非零音轨。
Dense实际委托400次；Sparse实际400次稀疏调用，0未验证、0回退，参考与生成区各32预算，显式最近plan。Sparse实际生成网格仍是`(37,15,27)`，不是训练分辨率。
Dense首次运行389.90秒、Sparse后续运行266.95秒，加载／条件缓存状态不同，不能据此宣称加速；独立采样段也未证明提速。画面出现可见字幕等现象，质量未自动通过。
2026-10-11用户对绑定这两片SHA的A1明确通过；不重跑或重审这份已接受案例，不把未通过草稿收入正式推荐工作流。

2026-10-22 本地开发另完成原生8步五档对照：dense委托基线、trained32、64、97与Flex全保留。
五片均完整120帧、864×480、24fps及32kHz双声道，完整RGB／PCM／PTS和严格解码通过；
原始图、Audit、raw WS、全批源／模型后核及8个自有进程退出已独立复核。
每档实际400次attention返回：基线为400次dense委托，其余四档为400次Flex返回、无delegate；
64／97是独立生成区与参考区tile预算，Flex全保留不是稀疏加速。五档均显式借用最近plan，
实际网格仍为`(37,15,27)`，不冒充训练尺寸。以上调用数不是kernel launch数或端到端提速证据；
共享IO／加载状态未作公平性能测量，也不认证逐行mask、MC接缝、4+4或新片人审质量。
本地证据是独立的新批次，不继承2026-10-11旧片的人工接受，也未发布到GitHub／Registry。

## 致谢与许可

感谢 [Veda-Sparse／Miowtion 与 Veda-on-ComfyUI 作者](https://github.com/veda-sparse/Veda-on-ComfyUI)。本次格式／布局／选择研究 pin 为
[`8d7ac153`](https://github.com/veda-sparse/Veda-on-ComfyUI/commit/8d7ac153ac215d636eaf1b5910b0536c3de9e6ed)。
原有 vendored tiling／mask／predictor 保持不变，MIT notice 保留在 `h3_t8/veda_vendor/LICENSE`。
权重另受 MiniMax H3 Community License 约束，不随节点发行包捆绑；本接入未复制或发行上游 Sage 派生 INT8 kernel。

English summary: Three additive R2VA Bundle/Apply/Audit nodes, separate `models/veda_r2va`, independent trained reference/generated tile budgets, explicit exact/nearest plans, and T8 Flex execution. Existing T2VA graphs remain unchanged. CPU, bounded kernel and two actual five-second native canvas/full-AV checks passed; the specific Dense/Sparse case received human acceptance on 2026-10-11. No upstream INT8, speed, flicker-fix, all-recipes or portable-stage claim; not yet published.
