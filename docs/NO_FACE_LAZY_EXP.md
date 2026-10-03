# 全帧检测一次 / 显式无脸 lazy 旁路（EXP）

本地新增九个独立节点，不改原 Standard / Parity Planner、检测器、采样、VAE、音轨、合成及旧图：

2026-10-03人工反馈：此前22帧黑屏／测试音虽然验证了机械旁路，但不适合作为正常画音人审，原片未通过、不作推荐。已另用真实舞蹈素材的明确脚步／地面裁切（124帧／24fps，原音轨）跑完整观察、未确认拒绝、literal确认及独立fresh画布。直接交付与旁路全RGB／PCM相同、重量依赖未执行；用户同日定向复审第2版已明确通过新的R08真实素材两路。只接受该完整样片，原合成失败保留，不把一次真人看听推广为任意素材无脸证明。

- `MiniMaxH3CompleteFaceObserveEXPT8`：完整原 RGB 经已有本地 detector 一次，绑定完整张量/权重/代码/声明24fps区间/颜色策略。仅完整成功扫描零框为 `no_face_detected`；有框为 `face_found`；缺模型、异常、部分扫描、坏框为 `unknown`。
- `MiniMaxH3NoFaceDecisionEXPT8`：`face_found` 输出 true；`no_face_detected` 只在明确开启 `confirm_no_face` 并手动填写当前 `observation_sha256` 时输出 false。unknown 和未确认结果阻断交付，不当 false；默认不确认、SHA为空。源、模型、阈值或代码变化后旧确认失效。
- `MiniMaxH3ObservedStandardPlanEXPT8` / `MiniMaxH3ObservedParityPlanEXPT8`：true 支路调用未经改动的原 Planner，但在仅本次调用的独立命名空间中读取已绑定观察，不修改模块全局或做第二次 YOLO。输出 schema、次序、原追踪/平滑/裁剪与原节点相同；无脸/unknown 不伪造 Plan 或 Crops。
- `MiniMaxH3FaceBranchStageSaveEXPT8`：FullSave 输出根也必须按需。true 才请求真实 StageResult 并调用原 immutable Save；false 不请求采样、不写文件，路径/SHA/latent 输出为静默 blocker，报告明确“没有保存 Stage”。
- 四个 `FaceDependentStageAudit` / `FaceDependentParityStageAudit` / `FaceDependentStitch` / `FaceDependentParityStitch` EXP：旧 Audit / Stitch 原本也是输出根，会主动执行。新适配仅取消自身主动输出标记，输入/输出全字段和内部原函数保持；放在 true 支路作为依赖，不改变旧节点的执行方式。

Boolean 应接**当前 Core 的 If/Else Switch（ComfySwitchNode）**。图像 on_false 接完整原 RGB，音频 on_false 接原 AUDIO；on_true 接原精修/合成路线。只有真实 lazy 调度才能省去未选支路的 CLIP/MODEL/VAE/采样。不要使用提前求值的普通切换节点。

不能只给最终 RGB 加开关，却保留一个会主动触发采样的 Save / Audit / Stitch 输出根；也不能把中间 Preview 接成独立输出强制执行 true 支路。当前六张新增配对图只保留最终 SaveVideo、按需 StageSave 和两个轻量观察/决定 Preview 根，详见[新增示例](../examples/workflows/66-radar-no-face-lazy/README.md)。Standard / Anime / Parity 可选；多人 SAM / 窗口规划不伪造兼容一次检测观察。

确认 SHA 必须是独立手动保存值，**不能把 Observer 的当前 SHA 自动接入确认输入并永久开启确认**，否则新素材会继承旧批准。检测器没找到脸不等于真人证明没有脸；最终决定和质量仍属人工，不自动完成或接受任何 Stage。

颜色输入沿原函数：Parity 的 Ultralytics 使用原 BGR 翻转，YuNet/动漫及 Standard 保留原输入规则。调用方需完整RGB3通道/24fps声明，不隐式裁切、RGBA转换、resize、retime或音轨处理。旧函数可手动运行不受此新节点的未知/确认逻辑影响。

可执行身份复用项目已有完整 typed code key（字节码、常量、异常表、参数、闭包名称等），不用首调用会变化的 marshal sharing/intern 标志。未知 callable 不冒称持久缓存便携，OBS 只在当前执行中使用；没有新的全AV/模型内容缓存。

当前十三完整 CPU 范围245项通过，包含五个接缝范围、一次 detector / 原 Standard-Parity Plan、Crops、Preview逐值一致、并发不互改、未知/确认/完整源绑定、六图当前 Core 校验和全部输出根检查。真实 Core PromptExecutor 四状态测试证明已确认负例只检测一次、RGB/AUDIO原对象交付且重量测试支路/Save不执行，未知/未确认不交付；正例复用原Plan而非第二检测。Core本地注册600（保留旧前缀），CUDA未初始化，不累计重叠测试。

已完成隔离CPU服务四次真实画布 Save As／重开／Queue：直接原片交付对照、未确认无脸阻断、独立填写已测OBS SHA的旁路、同确认值的新进程复跑。实际原YuNet完整扫描合成22帧192×128／24fps负例；未确认没有视频，确认后完整RGB和PCM解码与同原Core编码对照、新进程逐值相同。完整FullSave图仍保留所选UNET／CLIP／两VAE／StageSampler／原Plan／Audit／Stitch依赖，但真实执行事件证明这些支路均未运行，未写Stage manifest、未提供虚构路径或SHA。源、四个未选权重完整SHA、检测器、公开源码、Core／助手和私有已认证CV2保持，所有owned进程关闭。

这是Standard负例旁路的机械资格，不证明真实素材一定无脸、Anime／Parity等所有组合画质或任意用户补丁。确认值仍默认为空/false，独立QA填写不是用户审批；用户另行通过的是上述真实裁切代表的完整画音，不自动修改节点确认或QualityGate。功能本地未发布。
