# 局部外部 PCM 音效候选

`prepare_local_sfx_candidates` 接收原始 AUDIO 和外部生成的 PCM 候选。
可选接入需要保留的同轴音轨及真实 H3 PCM 候选。此入口不生成模型音频、不分离人声，
也不把外部 PCM 伪装成 H3 AV latent；原画面继续直连既有 Safe AV Save。

输入 `start_seconds` / `end_seconds` 是已对齐源 PCM 的本地时间窗，
`candidate_start_seconds` 是候选的本地读取起点，对两个候选使用同一偏移。
候选可以是局部片段或更长音频，必须实际覆盖所需样本；不循环、不自动补零。
候选可重采样、按原声声道数变换；源人声轨必须与源 PCM 率、形状、dtype、
时间轴相同，不能通过重采样或裁切假装同轴。

`protected_voice_audio` 也可以接原片整条音轨。此时保留的是原有对白、环境声等
全部声音，残差为零，候选是局部新增声音；这不表示已取得分离人声。下面的
可编辑示例采用这种整轨保护方式，并显式设置候选目标 RMS。

接入人声轨时，残差为 `source - voice`。输出在窗口内替换残差并保留源人声，
默认在两端各 10 ms 交叉淡化；窗口外 PCM 位相同。数值重构误差只是算术检查，
不证明源分离正确、残差没有人声或候选没有幻觉对白。未接人声轨时不会宣称
保护人声，局部替换可能删除该窗中的对白。节点报告这些限制，听审仍必要。

默认 `level_policy=match_source_rms`：把两个候选的局部 PCM RMS 调到源残差
窗口 RMS，再做交叉淡化。它不是 LUFS / 感知响度测量；更响不代表更好。
源残差静音时需显式选择 `target_rms`，用 `target_rms_dbfs` 指定新增声音的电平；
静音候选不能被放大成非静音。默认 `headroom_policy=none` 只报告真实峰值。
可选 `target_rms_ceiling` 将目标作为上限，只衰减过响候选，安静或静音候选保留
原生电平。它避免强制放大模型底噪，不判断或改善生成内容质量。
可选 `attenuate_candidate` 只降低候选贡献以容纳 PCM 峰值，不动源人声、窗口外
声音或使用全局 limiter。若保护的人声与候选组合无法满足峰值限制，报具体错误。
衰减会改变最终候选 RMS，报告同时列出匹配增益和峰值余量增益。

`selection=original` 是默认。用户选择 `external` 或 `h3` 后直接输出所选预览，
没有额外确认开关或自动优胜判定。H3 未连接时其预览口返回原声旁路，报告明确
标为 `absent`；此时选择 H3 会报缺少输入，绝不声称生成过 H3 音频。

返回顺序：所选 AUDIO、外部预览 AUDIO、H3 预览 AUDIO、JSON 报告 STRING。
报告包含源/人声/候选字节身份、真实采样端点、采样率/声道变换、局部 RMS、
峰值和数值重构误差。不要让技术 JSON 代替真实可播放同步对照的人审。

CPU 消费者回归使用合成谐波/脉冲类分轨，不冒充真实说话或 MMAudio 生成。
已覆盖既有 Dialogue Safe Master、实际 WAV 编解码、候选重采样与三臂 MP4/AAC
完整导出，三条视频的源 RGB8 SHA256 相同。窗口外/人声轨保持、默认原声、
直接选择、缺席候选、静音目标、峰值余量和实际时间轴错误另有数值回归。
训练后 MMAudio/H3 对照、含真实讲话的分离泄漏、击打时刻、边界 click 与听感
需另行真实素材验收；这些回归不证明模型质量、模型许可或 GPU 性能。

## 可编辑的三选对照示例

工作流：[`Local_SFX_AB_EXP.json`](../examples/workflows/96-nov05-exp/Local_SFX_AB_EXP.json)。
使用 Core `LoadVideo`、`GetVideoComponents`、`LoadAudio`、`TrimAudioDuration` 和本项目
两个公开节点，不依赖 VHS 或安装 MMAudio 插件。MMAudio/H3 候选在其它工作流中
生成后，以真实 WAV 接入；本图只作对齐、混合和成片对照。

上传以下四个文件到 ComfyUI，在左侧对应加载节点重新选择。JSON 不包含媒体，
不会用替代声音或自动下载来填补缺项。下表是本轮真实验证产物约定的名称和位置；
本轮候选文件已经实际产出；源码仓库不包含这些媒体，其它机器可使用审片附件或自己的真实音视频。

| 输入文件 | 本轮开发产物位置（相对 `artifacts/nov05/`） |
| --- | --- |
| `TaoMateHD_2LOW_1HIGH_5s_00001_.mp4` | `gpu/run-taohd_2plus1-v2/output/Nov05/` |
| `source_original.wav` | `sfx/run-v1/`，由上面原片提取并按有效音频端点去除 AAC 尾部 padding |
| `external_mmaudio.wav` | `sfx/run-v1/`，MMAudio 实际生成的 PCM |
| `h3_candidate.wav` | `sfx/run-v1/`，从 `gpu/run-sfx_h3-v1/output/Nov05/SFX_H3_5s_00001_.mp4` 提取 |

连接保持可见，便于替换自己的素材：

| 连接 | 用途 |
| --- | --- |
| 原片 `LoadVideo → GetVideoComponents.images` | 同时接四个 Safe AV Save，四路保持同一来源画面 |
| `source_original.wav → TrimAudioDuration(0, 5)` | 按采样数裁到 5 秒，防止原片 AAC padding 进入混合轴 |
| 上一裁切结果 → `source_audio` 和 `protected_voice_audio` | 同一份完整原声同时充当来源和保护成分；不是分离人声 |
| `external_mmaudio.wav → external_sfx_audio` | 外部真实音效候选 |
| `h3_candidate.wav → h3_sfx_audio` | H3 真实音效候选，只取声音，不用候选视频画面 |

`selection` 默认 `original`。改成 `external` 或 `h3` 后，“所选成片”直接跟随。
右侧 A 原声、B 原声加 MMAudio、C 原声加 H3 三路仍独立输出，便于同画面听审；
这四个保存节点均为输出节点，一次执行会各保存一个文件，选中原声时“所选成片”
与 A 是相同内容。文件分别写入 `output/Nov05/SFX_{original,external,h3,selected}_5s_*.mp4`，
不覆盖既有成片。

固定对照参数为源窗 `[1.2, 3.5)` 秒、候选起点 `1.2` 秒、两端 `0.01` 秒淡化、
修正版采用 `level_policy=target_rms_ceiling`、`target_rms_dbfs=-56`、`headroom_policy=none`。
淡化控件显示三位小数、步进 0.001 秒；三个时间窗控件显示两位小数、步进 0.01 秒。
这些仅修正控件显示与编辑精度，保留已有默认值及混音算法。
源整轨被保护时不能用零残差做 RMS 匹配；显式目标电平表示新增音效的强度。
候选必须覆盖读取区间，不会循环或自动补零。Safe AV Save 为 24 fps，当前例子
是 120 帧、5 秒；其它帧率素材应另接匹配其帧率的保存节点。

## 验证范围

需要自己生成 H3 候选时，先导入
[H3_SFX_Generate_EXP.json](../examples/workflows/96-nov05-exp/H3_SFX_Generate_EXP.json)。
该图用原片视频条件、video=0/audio=1 遮罩和原生 40 步生成声音，再把原片画面与
新声音保存为 5 秒 MP4，不生成替代画面。它不加载 MMAudio，也不分离人声。
若不想先导出 WAV，可用 `LoadVideo → GetVideoComponents.audio` 读取该输出 MP4，
直接接混合节点的 `h3_sfx_audio`；同理可将其它真实视频候选的声音接入对应 AUDIO。
示例 WAV 是本轮试听附件的方便入口，并不是候选必须采用的文件格式。

`tests/test_nov05_sfx_example.py` 使用真实 Core 加载、裁切和验证器、真实 WAV
编解码及 Safe AV Save，检查额外 768 个源样本被裁除、三种选择确实改变所选
输出、候选缺失确实阻止依赖输出、三路同源画面和窗口外 PCM 保留。测试素材是
明确的合成数值夹具，不是模型成片，也不进入人工审片页。

示例已进行独立无界面浏览器导入、三选切换、原生图序列化和刷新重载验证；未操作
用户浏览器、未排队采样。当前 Core 前端在 LoadVideo 导出中会附带空的
`video-preview` 预览字段，已单独核对其为空及 Core 接受该导出，不把它当作音视频
处理参数。机械与浏览器验证不代表 MMAudio/H3 音质或原片的人工验收。

本轮真实候选已使用本机已有 MMAudio 44 kHz v2 权重，配方为 25 步、CFG 4.5、seed 8301，
目标为轻微衣袖摩擦声；负提示排除对白、音乐等。私有验证器按真实视频时间戳
分别取 CLIP 8 fps / Synchformer 25 fps 条件，避免把 24 fps 源的不同前缀误当完整
5 秒视频。该适配属于本轮验证脚本，没有修改第三方插件。MMAudio 和 H3 的实际
生成结果、声音是否含幻觉对白、动作同步与听感以最终成片及人工反馈为准。

实际记录观察到 MMAudio 的 50 次 predict_flow 调用（25 步、每步 CFG 两次），
H3 候选采用无加速 LoRA 的原生 40 次网络调用。公开混合节点实际消费这两条声音，
保存原声／H3／MMAudio 三份 120 帧 AV；三份成片的完整解码 RGB 相同，窗口外
编码前 PCM 与原声逐样本相同。不是用合成音频替代训练模型候选，也未自动评为
音质通过；本轮待人工听审。

2026-11-08 初审备注为“都是噪音，需要修复”。旧例将 H3 放大 66.8 倍，已新增
只衰减策略并重做混音。旧失败片保留，新片待复审。见[反馈记录](NOV05_REVIEW_20261108.md)。

最新补充复审：指定 H3 与 MMAudio 两份修正版声音均通过，两个旧版均失败。
