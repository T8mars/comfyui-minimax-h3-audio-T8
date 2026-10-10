# MiniMax H3 T8 工作流目录

这里保存可直接拖入或通过 ComfyUI“工作流”菜单打开的前端 JSON、目录说明与文件名对照表。带日期的文件名只表示该工作流的发布日期，不代表功能稳定等级；带 `EXP` 或 `Advanced` 的路线应先阅读所在目录说明和画布 NOTE。

Windows 命名预算：仓库相对路径≤140、JSON文件名≤96字符；新文件建议≤64。37份旧长名示例已仅重命名，内容不变，见[旧名→新名与原SHA](filename-map.tsv)及[更新失败处理／维护约束](../../docs/WINDOWS_PATHS.md)。不要把全部功能描述叠到文件名。

分离式 S01–S29 / Veda 的保存位置见 [总索引](SEPARATED_WORKFLOWS.md)；最新 B8 / C1 / C2 音频修正版完整/恢复配对图见 [64-reviewed-audio-followup](64-reviewed-audio-followup/README.md)，不覆盖旧图。

本次新增：[VM01 视觉标记已审工作流与独立框选编辑器](84-visual-marker/README.md)；此前 [H05 11项已审短名工作流](83-h05-reviewed/README.md)及旧入口全部保留。

| 目录 | 主要用途 |
|---|---|
| [84-visual-marker](84-visual-marker/README.md) | 已接受的干净图／整框图／显式clean-VAE与marked-Qwen分流三张4+4，以及不采样的独立框选编辑器；残框限制保留，无新权重 |
| [75-radar-r6-reference](75-radar-r6-reference/README.md) | 10张原生保存来源图：参考包、独立Full/Cold采样、Full/Light解码、master模块与配方卡UI；须选自己的合法资产和实际缓存 |
| [74-radar-r6-delivery](74-radar-r6-delivery/README.md) | 三张先保存master再裁切／留边的模块，原声逐包保持，失败保留原片 |
| [73-temporal-dialogue-scope](73-temporal-dialogue-scope/README.md) | 8张v5／H16 Full-Cold；窗口对白新分支，外置EAV／Relay；固定v5通过、H16及效果模板仅CPU资格 |
| [72-freevideo-split](72-freevideo-split/README.md) | FreeVideo FP8六张Full／Cold；基础8+2、外置EAV／Relay音频参考8+2、真正4+4，须独立环境 |
| [71-kijai-experimental-split](71-kijai-experimental-split/README.md) | 九个Kijai非二步原件18张Full_Save／Cold_HIGH，独立MODEL／条件／Relay／EAV；须配对Full／Pruned与FL／Ref |
| [65-radar-visible-face-mask](65-radar-visible-face-mask/README.md) | 16张可见区域MASK模板；建议先用通过定向复审的逐帧MASK／显式0.05轻修配对图，不推荐旧失败静态遮罩配方 |
| [66-radar-no-face-lazy](66-radar-no-face-lazy/README.md) | 6张完整检测、显式SHA确认与真正lazy无脸旁路模板；不自动判无脸或接受结果 |
| [67-radar-external-continuation](67-radar-external-continuation/README.md) | 4张外片续拍Full／Cold图，EAV与Prompt Relay独立外置 |
| [68-radar-hyperflow-curves](68-radar-hyperflow-curves/README.md) | 3张独立Curve HEAD／TAIL、冷TAIL与零采样交付图；须匹配底模／教师／adapter／fit，dynamic模式未认证 |
| [69-radar-native-recipes](69-radar-native-recipes/README.md) | 4张Union2原生40步与A/B交替独唱完整／恢复图；不推荐失败的Turbo4+4 |
| [70-radar-model-compatibility](70-radar-model-compatibility/README.md) | 6张LMS、Orbit、Wallpaper R32单变量对照图，使用现有加载器，须自行选素材和模型 |
| [58-hypervae-2x](58-hypervae-2x/README.md) | HyperVAE 2× 视频 VAE：5秒画布实跑、一次采样同潜空间原生／2×解码双路对照（EXP）；旧图不变 |
| [36-avatar-voice](36-avatar-voice/README.md) | 正式Avatar录音驱动、原生音色／狂怒、标准4+4、独立20+4、两段8秒及EAV对照；可选TAEH3预览 |
| [37-meridian](37-meridian/README.md) | 独立ConvRot INT8四节点、授权Omega几何、空间／源时间编辑器，图片横移、视频冻结及源相机 |
| [38-diagnostics-preview](38-diagnostics-preview/README.md) | 只读来源／音频说明、可选末端开头静音淡入，示例默认关闭处理 |
| [40-modular-two-pass](40-modular-two-pass/README.md) | S01 base-flow、S02 LBH 和 S03 完整首采→独立 HIGH 的分离式实验图；外置 Relay／EAV、仅恢复 HIGH，旧图不迁移 |
| [19-pdd-acceleration](19-pdd-acceleration/README.md) | PDD FL2VA／Ref2VA 原图与新增 S04 独立 LOW4／HIGH4 完整、冷 HIGH 分离式 EXP 图；外置 Relay／EAV，旧图保留 |
| [41-vdn-two-pass](41-vdn-two-pass/README.md) | S05 OpenVDN DMD8／B50 完整首采、独立 VDN／原生 HIGH 的完整与冷恢复 EXP 图；外置 Relay／EAV，旧图保留 |
| [42-dual-model-split](42-dual-model-split/README.md) | S06／S07 旧 Dual MODEL LOW4／LOW20 × HIGH3／4／5 的完整与冷 HIGH 分离式 EXP 图；保留旧音频交接和一体图 |
| [34-fasth3-v2](34-fasth3-v2/README.md) | S08 FastH3 V2 首段／续段完整与冷 HIGH、显式 Review／Accept 和 Compose 分离式 EXP 图；旧图保留 |
| [43-manual-second-pass](43-manual-second-pass/README.md) | S09 手动 FIRST20→SECOND3 的 NativeNoise／FreeNoise 完整与冷 SECOND 分离式 EXP 图；外置 Relay／EAV，旧长视频图保留 |
| [44-progressive-split](44-progressive-split/README.md) | S10 Progressive T2VA／I2VA 原生 LOW／HIGH、外置 Relay／EAV 效果矩阵的完整、冷 HIGH 和已完成 HIGH 读取 EXP 图；旧图保留 |
| [45-progressive-continuation-split](45-progressive-continuation-split/README.md) | S11 已接受父片续段 22／39 上下文、独立 LOW／HIGH 与外置 Relay／EAV 的完整、冷 HIGH 和已完成 HIGH 读取 EXP 图；旧一体图保留 |
| [46-avatar-progressive-split](46-avatar-progressive-split/README.md) | S12 原录音驱动 Avatar T2VA／I2VA 及旧 I2VA 参数配方的独立 LOW／HIGH、外置 Relay／EAV、冷 HIGH 与原 PCM 交付 EXP 图；旧 Avatar 图保留 |
| [47-hyperflow-continuous-split](47-hyperflow-continuous-split/README.md) | S13 连续 HyperFlow HEAD／TAIL 的 1＋7、4＋4、7＋1 分界，外置 Relay／EAV、冻结 HEAD 冷 TAIL 和完成结果读取 EXP 图；不重加噪或放大，旧图保留 |
| [48-hyperflow-fresh-split](48-hyperflow-fresh-split/README.md) | S14 full8→fresh4、S15 partial4→fresh4 的独立 LOW／learned3D／新噪 HIGH，外置 Relay／EAV、冷 HIGH 与完成 HIGH 读取 EXP 图；旧图保留 |
| [49-hyperflow-p7-split](49-hyperflow-p7-split/README.md) | S16 P7 已接受父片续段的独立 LOW／HIGH、外置 Relay／EAV、冷恢复及显式接受 EXP 图 |
| [50-speed-split](50-speed-split/README.md) | S17 SPEED T2VA 两／三阶段分离研究图；手工 sigma 不作画质或速度推荐 |
| [51-speed-multimodal-split](51-speed-multimodal-split/README.md) | S17 六种多模态输入的两阶段外置效果与冷恢复研究图 |
| [52-chunked-v1-split](52-chunked-v1-split/README.md) | S18 Chunked v1 固定三段、逐段 Relay／EAV 与冷段恢复 EXP 图 |
| [53-chunked-v234-split](53-chunked-v234-split/README.md) | S19–S21 Chunked v2/v3/v4 独立 LOW／HIGH、外置 Relay／EAV、显式 LOW 保存及冷 HIGH EXP 图 |
| [54-chunked-v5-split](54-chunked-v5-split/README.md) | S22 Chunked v5 固定 2／3／4 窗 joint AV 分离、逐阶段 Relay／EAV、逐窗保存与冷恢复 EXP 图 |
| [55-h16-split](55-h16-split/README.md) | S23 H16 固定七窗、LOW 与各窗外置 Relay／EAV、逐窗保存及任一后窗冷恢复 EXP 图 |
| [56-face-refine-split](56-face-refine-split/README.md) | S24 Face Refine 七配方×四效果各独立 Stage／完整保存／按 SHA 冷交付共84张 EXP 图；局部 Relay Plan 按修复窗口编辑，旧图保留 |
| [57-motion-recovery-split](57-motion-recovery-split/README.md) | S25 Motion Recovery Fullclip／Windowed 的效果外置、首采保存与冷二采 EXP 图；旧图保留，仍缺 Windowed 单检查点实权重终验 |
| [59-audio-refine-split](59-audio-refine-split/README.md) | S26 十条 Audio Refine 已实现路线的最终视频冻结／独立音频冷尾采20张 EXP 图；效果通用矩阵与画音终验未完成 |
| [60-ltx-rgb-stage-split](60-ltx-rgb-stage-split/README.md) | S27 已有 H3 视频进入 LTX RGB 精修，外置 Stage Bind／Sampler／Audit 两张 EXP 图；不是同图 H3 首采 |
| [61-prepared-ltx-split](61-prepared-ltx-split/README.md) | S28 Prepared LTX 独立生成／解码及仅解码两张 EXP 图；要求有效 prepared bundle／生成回执 |
| [62-rf-restart-split](62-rf-restart-split/README.md) | S29 三条 RF Restart 入口×四变体，显式 BASE／Handoff／RESTART、效果与 Stage Load 12张 EXP 图 |
| `30-trt-vae` | 可选 TRT VAE：安装检查、本机编译、Decoder/Full 和同潜空间双路对照；无总耗时提速承诺 |
| `01-basic-generation` | 稳定双时钟与不同音频步数组合的基础生成 |
| `02-audio-control` | 音频锁定、重混、只参考及计划式音频注入 |
| `03-image-video-edit` | 单帧语义编辑、实验性五视角角色图、源视频重绘、参考强度实验与 LanPaint 局部AV修复 |
| `04-long-video` | 分段长视频、双模型 4+4、T8/KJ/Sol 可选路线、节点内一键串行、断点恢复、Prompt Relay/EAV，以及可选尾段细分或低Sigma二次采样 |
| `05-speech-dialogue` | 单人语音、参考音色、对白、长文本和音色库实验 |
| `06-face-refine` | 单人/动漫/多人脸部五官修复与追踪回贴 |
| `07-motion-detail` | 动态引导、尾段细化、Restart、STG与组合采样 |
| `08-multi-keyframe` | 首尾帧之外的中间关键帧时间线 |
| `09-hybrid-model` | FL2VA/Ref2VA混合权重补丁、兼容审计和显存策略 |
| `10-speed` | SPEED空间渐进采样、频谱标定、FastH3 T2VA 4步VSA，以及OpenVDN H3混合注意力DMD8/Stage B路线 |
| `11-studio-production` | 时间线、上下文、选择性修复、解码安全和交付工具 |
| `12-system-memory` | 环境审计、激活分块、Qwen前缀缓存、外部BlockCache组合、轨迹诊断和外部BlockSwap桥接 |
| `13-latent-upscale` | 普通32整除放大、学习型3D latent放大与二阶段H3生成 |
| `14-prompt-relay` | 全局提示词常驻、局部事件按时间接力、可选联合AV路由与8B提示词重写 |
| `15-sla-attention` | SLA Precision V2 FP32路由/直接Triton修复路线，以及旧LightX2V Sage2/KJ兼容诊断与强制运行审计（实验） |
| `16-raven-streaming` | 外部RAVEN因果分块T2VA、统一参数、加载前资源保护与请求合同审计（实验） |
| `17-skin-finish` | 最终解码后的肤色/油光候选、专用Oil Control低内存文件流、Studio镜头内参数关键帧、候选低频与来源高频解耦、单轨及SAM3.1逐镜多人五点ParseNet语义皮肤MASK、可续跑状态、YuNet代理两遍流和ParseNet语义Quality Stream，以及源片相对的曝光/纹理/裁切P2硬门（实验） |
| `18-audio-refine` | Turbo4/8、最终双采、PDD、EAV、Prompt Relay与长视频8步的可选音频精修；保留原视频、人工试听、默认回退（实验） |
| `20-core-compatibility` | 官方H3 AV Latent、Attention Hook、逐步同步优化和tiled VAE全局坐标的可选兼容节点 |
| `21-community-advanced` | Fun Control、长视频人物音色/句界、接缝漂移、低显存驻留、Creator语义缓存、TAEH3原生预览检查与只读诊断 |
| `22-sol-engine-h3-super` | NVIDIA H3 Super Acceleration：H3草稿经TAEHV、LTX-2.5 x2 latent放大与三步Refiner处理，H3原音频旁路回帖（实验） |
| `23-flashvsr` | FlashVSR v1.1 解码后视频超分：固定质量、动态预算候选和低显存分块，原音频完全旁路（实验） |
| `24-mv-lipsync` | 全本地 MV Vocal Lock V2：独立人声分镜与H3驱动、官方六段式Ref2VA提示词、串行续跑及完整原曲最终单次混入；保留旧V1兼容工作流（实验） |
| `25-dlss-nr` | Windows RTX 可选的 DLSS-NR v1.3 图片、短视频帧和文件视频超分；外部运行时审计、原音频保留和盲测验收 |
| `26-h3-world` | H3-World 首帧 I2VA 人物/镜头控制；固定 832×480×124、37 段 WASD/IJKL/F 动作时间线和安全音画保存 |
| `27-video-outpaint` | H3 视频扩画：范围预览、四阶段或折叠生成、首窗候选审图/确认/续跑、区域提示、完成缓存重存和可选 DLSS-NR 2x |
| `28-progressive-sampling` | H3 渐进首采 T2VA / I2VA，小画幅 6 步＋学习放大＋大画幅 2 步（EXP；短片画面可用，游戏音频及32秒限制见目录说明） |
| `29-dlss-fi` | 独立 DLSS 视频插帧 2x，保留原音轨和时长；需要单独运行文件（EXP；已审短片可用，不保证所有素材无伪影） |
| `31-topaz` | 正式 Topaz 环境检查与视频增强开发候选；仅1x机械验证，2x/星光/人审待完成，不自动下载 |
| `33-selflift-taomate` | 已完成绑定样片验收的 SelfLift 4+4、TST/EAV/Relay/KJ/Sol 组合、两段8秒接缝与 TaoMate 原生3/4步工作流（EXP；不作普适画质或提速承诺） |
| `34-semantic-bridge` | Semantic Bridge／BUNNY 普通条件、Relay、内循环及独立双采；640×320→896×448两段8秒4+4配方已验收，其他路线仍按EXP说明 |
| `36-avatar-voice` | 本地 Avatar 录音驱动4+4、可选TAEH3预览／定向取消、原生Ref2VA普通／情绪对白与双采／长视频接线；人审待、不发布 |

新增入口：[H07七项已审配方](85-h07-reviewed/README.md)：可选新音频时钟、清唱、有限背景合成、姿态、同Stage X2、动作拟音和专用PDD7+1。旧配方不覆盖。

新增入口：[FreeVideo 新四档 Full／Cold](76-freevideo-quality/README.md)。Light8+3／Medium12指定样片人审通过，16／20仅CPU结构支持；[旧8+2／4+4](72-freevideo-split/README.md)保持。

使用顺序建议：先从稳定基础/音频工作流确认模型链可运行，再按具体目的进入 Advanced/EXP 目录。不要把不同高级采样器直接串联；组合能力应使用专门的 Mixer 工作流或遵循画布 NOTE。
