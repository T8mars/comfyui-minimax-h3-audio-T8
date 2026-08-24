# Studio制作、时间线与交付

这一组处理生成前后的制作数据：统一时间线、上下文IR、选择性修复、AV解码安全和最终Reel交付。

## 工作流

- `Studio_Timeline`：构建绝对帧/sample时间线。
- `Context_IR_Provider`：向分段生成提供结构化上下文。
- `Selective_Repair_Execution`：只重做指定片段并保持accepted语义。
- `AV_Decode_Safety`：核对音画latent、时钟和释放策略。
- `Reel_Delivery`：汇总片段、音频、字幕与交付报告。
- `Creator_Workspace_Run_Window`：在现有Timeline上追加镜头级提示词、seed变体、媒体角色、保留策略与hold-map，并显式选择运行区间和候选。
- `Creator_Synchronized_AB`：把两个同源候选按共同帧段并排显示，顶部写明A/B标签，不改变源像素比例。
- `Creator_Synchronized_AV_AB`：从两条同内容视频拆出帧与音轨；保存无声并排画面、分别提供A/B音频播放器，并用参考相对漂移审计给出复核时间段。
- `subgraphs/2026-08-23_H3_Quick_Creator_AV_Review.json`：上述音画审片工作流的原生Quick Start入口，只外露A/B视频、标签、seed、人工结论/备注、严格画布开关和输出前缀。
- `Creator_Run_Receipt_Resume`：把外部执行结果写成不可变回执，从accepted/completed/failed/cancelled状态计算下一次render/review/retry/complete动作，生成只读的候选保留/拟删除清单，并演示默认静音的可恢复Quarantine执行节点。
- `Creator_Long_Video_Background_Bridge`：把Creator workspace hash绑定到现有Long Video后台控制器，用accepted manifest进度选择镜头，用自动重试序号选择seed变体。

## 当前成果

这些节点已有typed数据对象、JSON报告、时间边界和异常路径测试，适合把复杂创作从“一张巨型画布”拆成可追踪步骤。

Creator第一阶段已完成CPU、结构和一条真实素材会话：由两个真实39帧H3候选建立3镜头Timeline，运行窗口0-2；shot B生成3个确定seed并显式选中variant 2，sidecar记录视频/音频上下文角色和5帧hold-first。同步A/B输入均为`[39,256,256,3]`，输出`[39,288,520,3]`，源像素逐值保持，保存MP4严格解码3/3通过，整条8节点API用时1.688秒且没有增加GPU占用。该会话仍保持`ABSTAIN`，因为未进行人眼胜负判断。保留计划现在可以机械生成清单；追加的Quarantine节点只实现输出目录内文件的SHA锁定、可恢复移动和异常回滚，不提供永久删除，仍属于实验功能。

同步音画A/B的独立低负载实跑复用了同两条39帧、256×256候选。无声并排结果为520×288、24fps、39帧，严格视频解码3/3通过；A/B音轨分别生成可播放预览。参考相对审计返回PASS，波形相关0.9842、频谱漂移p90 0.0735、响度差p90 0.2308dB。这里的PASS只证明默认阈值没有触发持续漂移，不是人耳质量认证；实际胜负仍为ABSTAIN。

Creator AV Review Quick Start复用完全相同的工作流节点，不创建第二套审片逻辑。旧六份Quick Start文件的SHA-256保持逐字节不变；新入口默认`ABSTAIN`，并排视频的`CreateVideo.audio`保持未连接，A/B音频仍从两路独立输出分别试听。它改善入口数量，不会把机械漂移报告升级为自动质量判定。

运行回执/续跑工作流已在隔离的CPU ComfyUI API中执行：两镜头73/124帧计划先记录shot 0的`completed`，同一次attempt再记录`accepted`，续跑节点随后选择shot 1、variant 0、attempt 1。完全相同的API任务第二次运行时ComfyUI报告整条1～5号节点缓存命中；只修改`base_seed`后依赖链全部重新执行。这证明普通节点缓存身份与失效传播，不代表H3模型内部计算可以断点复用，也没有测试正在采样时取消。

后台桥不复制队列实现：它复用本项目原有Long Video的定向删除排队prompt、运行中interrupt、history监控、失败重试、进程租约和accepted manifest。新增绑定只保存workspace hash与run_count，不保存完整提示词；不同workspace复用同一chain会拒绝。CPU测试已覆盖绑定持久化、跨workspace拒绝、accepted_count推进镜头、retry_count切换seed、暂停/失败/取消状态阻断下游。另有一条真实256×256×22、4步H3低负载测试在采样进度1/4定向取消：history为`execution_interrupted`、accepted/retry均为0、两类队列清空，`unload_all_models`无释放错误；整卡观察值从2999MiB升至10794MiB后回到3089MiB。

后续机械闭环没有放宽正式Long Video的124帧下限，而是使用原生空H3 AV latent与64×64确定性媒体跑合法两段窗口。第一段124帧保存上下文，后台只续排一次；第二段去掉5帧上下文后接受119帧，最终合成243帧/10.125秒H.264+44.1kHz双声道AAC。两条history均success、manifest完整、队列归零、释放无错误。该过程还发现并修复failed任务持有旧chain lease、阻断同进程新job重新挂接的问题。它证明恢复控制面、Candidate Save、Auto Accept与合成机械正常，但不证明真实H3续作画质/听感或无人审核质量。

隔离CPU ComfyUI中还完成了一次真实PromptServer运行中取消：绑定任务进入`running`后，cancel路由向完全一致的prompt ID发出interrupt，history记录`execution_interrupted`，accepted/retry保持0且运行/等待队列清空。它独立证明控制链；上述后续H3测试再证明一次真实采样中断和释放观察。两者都不证明恢复成片质量。

## 使用方法与注意事项

先让Timeline/Context成为事实源，再启动生成或修复；不要在下游节点各自重算帧率和时长。选择性修复必须写入新的overlay并保留原accepted记录。最终交付前检查音频采样率、视频fps、字幕单调性和A/V边界。

Creator的`run_from_index/run_to_index`只生成计划，不会自动操作ComfyUI队列。`retention_policy`由Retention Plan节点编译为keep/proposed-delete清单；`confirm_artifact_paths_reviewed`默认关闭。Quarantine节点在示例中默认静音且默认为`prepare_only`：先核对文件仍位于当前ComfyUI输出目录、不是目录/链接/越界路径，再记录字节数和SHA而不改文件。需要隔离时必须把完全相同的manifest、plan hash、新正整数epoch和显式确认同时提供；文件只移动到`output/MiniMaxH3/creator_quarantine`。还原或异常恢复必须使用同一plan/manifest/epoch，项目没有永久删除入口。纯画面A/B节点不比较声音；音画A/B工作流也故意不给并排视频绑定任一候选音轨，避免误导。两个候选帧数或尺寸不同且开启`require_equal_geometry`时会返回`ABSTAIN`，应先统一解码几何再人工判断。

运行回执节点只接受显式事实：`cache_observation`不会扫描历史记录或隐藏队列，`artifact_manifest_json`只保存元数据而不打开、移动或删除路径。`completed`之后必须由同一次attempt明确写入`accepted`或`rejected`；`failed/cancelled/rejected`之后重试必须递增attempt。Retention Plan遇到同镜头多个accepted、同一路径同时进入保留/拟删除或拟删除manifest没有显式`path`时会`ABSTAIN`。Quarantine目前已通过prepare、隔离、还原、越界/篡改拒绝和部分移动失败回滚的CPU文件系统测试；它不能替用户判断素材是否该被隔离。真实H3中途取消和后台终态新job重新挂接均已完成低负载机械验证，但前者的真实NFE跨进程恢复媒体仍待显存空闲后验证。

后台桥是另一条明确选择的路径：必须把它嵌入完整`Long_Video_Background_22F`图，并继续使用原Candidate Save与Auto Accept terminal。默认`review_only`不会排队且会阻断下游；切到`auto_accept_and_continue`代表用户同意每个成功候选无需人工复核直接进入accepted manifest。一个Creator shot必须严格对应一个Long Video segment。
