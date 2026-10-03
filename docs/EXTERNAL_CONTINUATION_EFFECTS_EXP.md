# 外片续拍：外置 Prompt Relay／Stage EAV（EXP）

这四个追加节点只处理已保存、已手动采用的外片 RGB／PCM 重编码上下文。不是恢复外片的原采样 latent，也不是原生 accepted-parent／P7 的祖先凭证；旧节点、工作流及默认配方不变。

## 独立接线

`External Context Encode → External Relay Window → External Relay Conditions / Model`

原 global Relay Plan 仍在外部。窗口的 `generated_start_frame` 是明确指定的全局 Relay 时间坐标；不是自动读取原片结束、trim 或已采用工程的时间。渲染起点为它减去 `context_frames`，新生成区间为 `[generated_start_frame, generated_start_frame + length - context_frames)`。窗口必须落在 Plan 范围内，并保留原 24fps、17n+5 网格及事件时间公式。旧 projector 中命名为 accepted 的数字窗口在这里不代表已审核、已采用或原生祖先。

`External ONE / Relay Conditions → External Motion Effects Bind → 原 Stage Setup / Bind → External Stage EAV Apply → Guider → Stage Sampler → 原 Stage EAV Audit`

Effects Bind 同时连接本阶段的 External Context、MODEL、positive 和实际 AV source。LOW／HIGH 使用各自的上下文、模型和条件；HIGH 在原放大／Reconcile 后绑定其实际 HIGH AV。这里只校验本阶段内容与当前模型，不取代原调度、学习型放大器或音频策略。原 EAV Config 可共享或独立连接，默认 `report_only`；`disabled` 原样返回 MODEL。不强迫变成20步。

不需要 Relay 时保留 External ONE Conditions，直接使用 Effects Bind 与外置 EAV。Relay `report_only` 不安装路由偏置，但新外片入口仍返回原生运动 payload MODEL patch。

## 保护与恢复边界

完整源片／保存工程／采用／trim／编码器／上下文／实际 motion guides、条件、AV、实现与明确时间窗绑定。改变任一后重新准备，不套旧绑定。运行时再次核对实际 motion payload；不清空用户 LoRA、注意力代理、遮罩或音频。未知编码器可执行但不可认证跨进程缓存；未知补丁仍遵循项目的可使用／未验证策略。

Effects Bind 本身不采样，EAV 配置也不证明效果执行；查看采样后的真实 audit 和 StageResult。保存／加载仍需原 StageSave／StageLoad 的实际路径及 SHA，不自动缓存、选择、采用或接受。渲染包含上下文前缀，交付前显式裁掉；不自动拼回原片或保证接缝。

## 当前资格

三份完整 CPU 文件59项通过：真实外片注册／采用／RGB 重编码，真实 tiny Core 四次运动前向，独立 Relay 与 EAV 的实际调用，关闭／报告模式逐值一致、apply 修改实际视频注意力，以及晚到内容修改拒绝。随机 tiny 权重／测试编码器不代表预训练质量。首轮六个失败保留，未以放宽源或缓存校验通过。

后续十四份完整 CPU 文件327项通过，覆盖旧 EAV／Native Explicit／P7／Relay、实际独立冷进程和五组接缝。CUDA未初始化，源码1648文件稳定；23条 warning 中21条为已有子进程管道GBK解码告警，保留真实记录，不改测试或伪造静默成功。与59项范围重叠，不累计。

当前追加604完整注册、2029旧JSON／受保护暂存区和采样源保持；四张独立 FullSave／ColdDelivery 图见 [67目录](../examples/workflows/67-radar-external-continuation/README.md)。十份完整 CPU 文件161项通过，CUDA未初始化、1649源文件稳定：补齐立体声5／22／39上下文 EAV-only、真实错误 motion payload 中止与变更实现拒绝、四图逐字段与当前 Core 验证，以及旧 EAV／五接缝。此前一项新版测试的错误异常消息预期已纠正，生产拒绝闭包修改保持，原失败记录保留。各CPU范围重叠不累计。

一条当前预训练权重的真实画布 SaveAs／重开／Queue 资格已完成：外片22帧上下文、124帧渲染，独立外置 Relay `apply_exp` 与 EAV `apply_exp`；四次真实采样回调，Relay与EAV选择器各200次，时钟匹配。显式裁掉22帧后完整102帧448×256／24fps／4.25秒成片，以及同一保存输出根的新GPU进程 StageLoad 零采样交付，全部RGB／PCM精确一致。实际portable StageSave、所选权重完整SHA、原工程已有素材／配置、公开源及Core／助手保持，两个自有进程已关闭。

此例EAV有一次active forward／50次测量，但gain均为1；证明真实执行与恢复，不证明增强收益。只是一条video_only外片上下文组合，不泛化PCM-context、多阶段双采、其他工作流或接缝质量。首次保存旧QA文件名冲突在采样前被保护拒绝，失败证据保留，新资格使用唯一文件名。尚未发布或通过人审，不能把CPU、机械恢复或可导入图称为全目标完成。
