# H07：同完成 Stage 的可选 X2 解码（指定两路人审通过）

2026-10-10定向复审revision10，独立**专用PDD** 7＋1新配方A7通过。
保存[A7_PDD_7plus1.json](../examples/workflows/85-h07-reviewed/A7_PDD_7plus1.json)：
864×480→既有learned1.2×实际1024×576，专用PDD动态AV头、两路新几何Euler/simple。
不是旧普通Turbo闪光片被接受，不是同Stage X2或高清多窗接缝资格；旧失败保留。
下方新PDD片“待审”现为过程历史，未据此证明闪光根因或普遍修复。

复用 H07 N01 已保存的完整 HIGH Stage，原生 VAE 与既有 HyperVAE 各解码
一次，音频 VAE 和最终 5 秒裁剪相同。没有新增采样器／二采／NFE，不改
旧图或默认放大倍率。HyperVAE 模型原件 SHA 与既有获核实的权重一致。

`tools/prepare_h07_x2_canvas.py` 只准备 10 节点原生图，不排队。
独立 `H07_X2_Draft.json` SHA256 为
`9206e005d9d76b4ea9f1f5f9769ed699917e5c63943e91ddc2092697c542a880`。
浏览器恢复后已真实另存 `H07_X2_Native`、导出 API、关闭重开和 UI Run。

实际原生 512×288、X2 1024×576，共用真实完成 AV latent，最终各 120 帧。
job `ec105ca6-665a-41df-93e3-325fdd6d3c7f` 成功 33.89 秒，0 新增 NFE。
两路完整帧与 PTS、音轨、Stage／payload／VAE 身份核验通过，浏览器原生预览
实际可播放且无解码错误；集中人工审核 A5／revision37 两路通过，旧 HyperVAE 的既有资格保留。本项也不等于
已完成高清尾部／跨段接缝或独立 7+1 轨迹资格，不默认 4K。

## 只读验收检查

`tools/audit_h07_x2_canvas.py` 不加载模型或提交任务。`--prepared-only`
仅核真实完成 HIGH 的 manifest 与 tensor 文件 SHA、实际运行服务的 schema／
VHS 所选格式动态输入，以及草稿全部参数与 14 条类型边。该模式明确标为
NOT_UI／NOT_RUN，不当成画布保存、关闭重开、播放或质量通过。

准备核原三个合同通过；真实保存省略零 widget 的 link-only 节点及实际 AAC
帧量化，各补一个针对新增情况的合同通过，没有重跑旧测试矩阵。
首次测试启动未启用 Core CPU 参数、首次检查假设草稿有具名 widget 均失败，
原失败记录保留。现在只在校验副本里按实际 schema 对应位置，不改草稿原字节，
不忽略未知值或放宽参数门。

控制恢复后，先在原 8991 画布另存、导出 API、关闭重开，再用 `--native` 与
`--api` 记录预检；只从真实 UI Run 一次。随后带 `--preflight` 和真实
`--job-id` 读取已成功 frontend history。完整检查两路各 120 帧、全部 PTS、
512×288／1024×576、原件路径、完整音轨与同 Stage 身份；音轨不一致时拒绝
并保留成片供诊断。真实收集分支已经执行；首次严格样本数检查失败保留。
实际两路 32kHz／双声道 AAC 各 159744 sample，解码 PCM 两路逐值一致；
比名义 160000 少 256 sample（8ms），明确为最多一实际 1024-sample AAC 帧
的量化范围，不补音频、不换片、不宣称精确 5 秒 PCM。越界缺音仍拒绝。
关闭重开与可见播放由浏览器记录，细节收益仍最后人审。

## 独立 LOW7＋放大＋最后1步（本机 dense 实验，人审未通过）

感谢 [AI-ONE-STUDIO 原作者](https://github.com/designloves2/AI-ONE-STUDIO)
的 [固定 7+1 图构建来源](https://github.com/designloves2/AI-ONE-STUDIO/blob/988250ddeefa10fa7851d65285979888a42bfcdf/src/tools/minimax_h3/graphBuilder.ts)。
该来源把八步表在第七步切开，LOW 使用 `denoised_output` partial x0，视频
learned lift 保留这一时刻的音频，HIGH 用同一 RandomNoise 加噪并走最后一步。
本机图是显式 T8 dense／fp16 lifter 小尺寸适配，不是作者强制 SOL、memory
patch 或 BF16 upscaler 的逐值复现；没有复制或安装作者插件、恢复 Sol5090。

只新增独立 T2VA 配方，448×256 → scale_by=1.2、实际网格对齐 512×288。
LOW／HIGH 两个 DualClockSampler 必须分别按对应 latent 几何初始化，HIGH
共享同 seed／原最后一段 sigma，不能直接复用 LOW packed sampler。首个
真实 Run 被尺寸门拒绝，LOW7 与 lift 原件保留；不移除 guard。修正版准备
曾出现 NOTE 编号重复，被实际导出预检拒绝，未 Run；这两份失败证据均保留。

最终 `H07_Finish7_Fixed2` 保存、API 导出、关闭、目录重开、预检通过后，真实
UI job `70725ad7-083b-4e28-8d8a-e0776180b3d9` 成功 19.489 秒。这个时间是
**复用已完成 LOW7／lift 后的最后一步＋解码保存**，不是完整冷启动八步耗时。
完整 120 帧／24fps／5 秒原生 AV、三份 native checkpoint CPU loader 实核通过；
lift 前后 partial 音频逐值相同，最后一步确实更新音频。未插桩 DiT 调用数，
不称速度收益或不变音轨。新成片 SHA256：
`55ca5c16aa7e409aba71d2bf151006b054ac9414af99377cbcd399091aae6a0e`。

全帧表可见不符合提示词的黄色亮斑变化。集中人工审核 A7／revision37 未通过，
用户明确反馈持续闪光，不能填质量通过或推荐为无闪烁。用户提及 EAV 只是外观
类比，本图没有接 EAV，尚非根因定位。普通对白仍完整保留，未换 TTS／静音。
后续定向验收采用 LOW 约 0.4MP／learned scale_by=1.2，记录真实网格尺寸；
不重跑 A5 已通过的同 Stage 两路解码，不以放大旧成片冒充修复。
本项不是 HyperFlow continuous 7+1、Progressive 边界或高清跨段接缝资格；
I2VA／FL／Ref 需其各自 HIGH 尺寸条件，不能继承本 T2VA 的通过结论。

## R37 后的定向准备：PDD 配方差异

复核上述固定上游源码后，确认其 HIGH 使用 PDD 动态输出头和 Euler 调度；
本机失败配方却使用普通 Turbo8 LoRA。这是实际配方差异，**尚不是闪光根因
或修复已证明**。原失败图、成片及只改尺寸的诊断准备件保留，不静默重写。

`tools/prepare_h07_pdd_finish.py` 仅新增独立准备：复用本机完整 FL2VA 与已有
FL2VA PDD 权重，通过既有专用 PDD Setup 接入动态视频／音频输出头；原8步表
切7+1，HIGH按lift后的几何重建Euler/simple sampler，复用同一个PDD MODEL，
不重复加载PDD或混入普通Turbo LoRA。LOW864×480、learned1.2×，计划HIGH1024×576。
实际文件头、PDD完整SHA和当前schema检查通过；全基模仅核文件头／stat，不冒称
扫描全文件SHA。两个新增stdlib准备合同通过，未重跑旧CPU矩阵。

随后真实另存 `H07_R37_PDD7`、导出API、关闭并从保存树重开，31执行节点／
52边精确核验；UI job `6e99005e-f049-477e-be23-d354df5187b7` 空缓存完成
277.270秒。实际LOW864×480／7步、VIDEO-only learned1.2×／1024×576、HIGH1步。
真实报告 native_core_used=true，258个adapter／四个动态AV输出头、完整50块
2688输入宽度；三原生checkpoint均经CPU loader SELF_VERIFIED。lift前后AUDIO
逐值相同，末一步实际继续更新AUDIO。最终全120／24fps／5秒音画完整检查通过，
SHA256 `65d7fe6fb0f667109311b3beb3fabf2a0d7a3a76ac30f2c8a06972be7b5565a0`。
两页全部120帧已看，未观察到旧例明显黄色闪斑，仍不是闪光根因证明或真人接受。
原生画布视频loop=true，播放进度／跨循环可观察，不伪写ended=true。
最终集中人审待定，不收入推荐；未恢复作者forced-SOL／memory-patch，不宣称
逐值复现、高清跨段接缝或加速收益，旧普通Turbo失败证据保留。

已通过的同Stage两路解码模板见[H07已审目录](../examples/workflows/85-h07-reviewed/README.md)，
该目录不包含此待验收PDD7+1或原失败图。
