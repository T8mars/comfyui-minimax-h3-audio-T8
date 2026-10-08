# H05 外部 Encode 视频／音频参考导入（EXP）

两个独立入口：`MiniMaxH3ExternalReferenceVideoImportEXPT8` 和
`MiniMaxH3ExternalReferenceAudioImportEXPT8`。旧图像 Import、原生 Create／Load、
Save／Route、外置 EAV／Prompt Relay及采样器保持；旧节点字段、默认和工作流接口不变。
Apply 仅为选定的 H3 音频参考补齐已有 crop 兼容边界，见下方冷加载说明。

## 共用接线

选择配置的 `models/refmods` 中的 v4 单体或 v5／RefLoRA 参考半部，明确整个文件
SHA256、从 1 开始的成员序号、role_id 和来源使用确认。提供实际编码时用的准备后
RGB／PCM及对应原生 VAE；节点重新编码验证 shape、dtype 和全部原字节 SHA。
匹配后保留**外部原 latent**，不是把重新编码结果偷偷当作导入成功。

输出可直接接现有 Route → fresh Apply。需跨运行保存时，先接现有 Save（新文件、
create-only），下一张图从原生参考 Load 的目录菜单选择已保存文件，核 SHA 后
接 Route → fresh Apply。Load 的菜单不是 Save 输出 STRING 的动态文件路径接口。
视频角色和独立声样角色必须显式声明；不同包的同名 role 不能靠猜测合并。
音频参考不等于 drive_audio／final_audio，两个入口也不自动建立音画同步关系。

只加载选定的一个参考张量，不加载其他成员或 LoRA 半部，不执行包内配置、
路径、训练程序或 attach。未知／训练／池化模式、不匹配的字节及坏 SHA 正常报错。
现有 1 MiB 文件头、8192 描述符及 256 MiB 参考资产预算保持。

## 视频入口

输入 `source_frames` 是准备后的 float32 RGB，范围 0..1，恰好等于编码尺寸和帧数。
支持现有原生 `17n+5` 图像帧网格和相应 `[1,24,T,H,W]` latent，最多 360 帧。
`source_fps` 必须明确为 24；不会裁切、缩放、补帧或推测原片 PTS、时间原点。
这是调用者声明的准备后时钟，不是原片时钟认证；需要实际源 PTS 证据时先走
既有 Source Clock／Conform／Map。

- `native_exact`：外部 latent 与当前原生 VAE 输出完全一致。
- `author_encode_fp16`：明确把新原生编码转成 FP16 后比较，保留外部 FP16 字节。
  不承诺与 FP32 原件相同画质。

作者的因果帧网格与 T8 网格并非所有帧数都相同；不能靠重新标注尺寸来兼容。
不适配的原件需明确准备／重建，不能称为本次无损导入。此入口只导入视频参考，
不会把其声音轨道顺带当作声样或最终音轨。

## 音频入口

输入 `source_audio` 必须是实际准备后的 `[1,2,L]` float32、32000 Hz、有限值 -1..1
PCM，长度大于零且最多 30 秒。不会自动重采样、截断或把单声道复制成双声道。

- `native_exact`：一次原生 VAE.encode。
- `author_10s_native_exact`：每 320000 样本调用一次原生 VAE.encode，再沿时间轴连接。
  对应作者固定 10 秒分块的明确验证路径；**不使用图像 FP16 转换**，也不宣称它与
  整段编码在分块边界相同。报告保存每块实际样本区间及 codec 的 800 样本右侧补零。

原生 H3 codec 的 encode 已执行 latent mean/std 归一化；当前 Core 音频 VAE wrapper
process_input 为恒等。只使用连接的原生 wrapper，不另外猜缩放或重复归一化。
已有 H3 非整齐尾部 crop 兼容在 producer 记录前建立；不改变其他 VAE。
带预算截断的原件若不能与提供 PCM 完整编码匹配，保持不支持，不猜被丢弃的声音。

包内保存的是声样 latent 和来源 PCM 的内容 SHA，不保存原 PCM 或最终音轨。
需保存最终原声请走既有 final_audio，不能从这个声样包反推原声音字节。

## 当前证据与限制

新增 9 项定向 CPU 单元测试通过；旧 686 个完整 finalized schema 精确前缀保持，
仅末尾追加两个入口到 688，365 个受保护文件保持。实际 Core VAE wrapper 与
MiniMaxH3AudioVAE.encode 方法的轴、补零、归一化测试使用 tiny 学习子层，
属于接口单元证据，**不是真实权重或 GPU 资格**。

新增的一条已知来源实际案例完成了真实权重和原生画布验证，三张独立图均执行
SaveAs→Save→关闭→侧栏重开→Run：

- 原生源编码：73 帧、256×160、24fps；音频为 320801 个 32kHz 立体声样本，
  明确分成 320000＋801 两块。使用现有真实视频／音频 VAE，不是 tiny 代替权重。
- 视频／音频导入：视频明确 FP16，音频保留原生 FP32，两入口各自 fresh 编码
  与选定外部 latent 原字节匹配，create-only 保存两个独立原生参考包并实际 Route。
- 独立新后端从目录 Load 两包后，实际 fresh Qwen Apply 用时 31.902 秒、无缓存节点。
  visual_A 与 voice_A 分别映射到 Video 1／Audio 1，参考行数 880＋804；
  source_audio_tag=none，没有把声样当作 drive_audio／final_audio。

第一次冷加载 Apply 发现原生 H3 音频 wrapper 默认 crop_input=True，与 Create／
Import 在编码前建立的既有 crop_input=False 身份不同。现在仅在选定 H3 声样的
producer 捕获前调用同一个已有兼容函数；不跳过权重／配置／实现身份检查，不改
其他 VAE、未选声样或包字节。初次失败证据保留。3 个新增回归行为已通过：
首轮两个未改病例通过，修正一个测试的显式 native 音频输入后仅重跑该病例通过；
冷加载、错误权重仍拒绝、未选／非 H3 音频不被误改均有证据。原 688 个完整
finalized schema 未变，原保护基线保留，364 个文件未改，只有明确记录的
reference_runtime.py 冷加载修复例外；不能把这次例外说成 365 个全未改。

本次是 **0 DiT NFE 的真实编码／导入／保存／冷加载／条件接线资格**，没有生成
新视频，不是成片人审或任意第三方原件资格，不借旧图像导入或 N01 人审批准。
旧缺原 RGB／PCM／producer 的社区视频＋音频样本仍不支持。匹配不认证历史作者、
素材许可、原片同步或任意多成员包可完整推理。测试素材、模型、私有交接和运行
证据不随插件发行。

格式及音频分块参考固定[作者版本](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/tree/6131c1f62c612a074d6aac8435c26ffa19c00563)，
本实现不打包或运行第三方编码程序。
