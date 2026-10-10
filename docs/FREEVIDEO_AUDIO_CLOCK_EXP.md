# FreeVideo参考音频时钟（可选增量 EXP）

感谢[FlashML-org/FreeVideo](https://github.com/FlashML-org/FreeVideo)和OpenVDN作者。此增量固定作者v0.3.5 commit `566251c88707322e4674ba1c6f587834e07a91b6`，参考音频行采用t=1；不是训练新模型，也不修改参考音频的全局时间。

旧FreeVideo 8+2／4+4／四质量loader、默认值、工作流及旧Stage保留。新增“FreeVideo · 新参考音频t=1加载（v0.3.5）”显式接入原分离采样、HIGH3及外置EAV／Relay。新入口不自动迁移旧LOW；不同源版本的Stage不能混用。旧Stage继续由旧入口读取。

## 独立准备

1. `tools/prepare_freevideo_quality_source.py --audio-reference-v3 --help`：新目录固定Git源码，核每文件blob，不执行安装器。
2. `tools/prepare_freevideo_quality_runtime.py --audio-reference-v3 --help`：借用实际SHA验证的主体、Python及VDN，按选定profile/task取表，create-only新配置。`--relocate-t8-from`仅用于明确的插件根迁移，未改helper仍须旧SHA一致。
3. `tools/check_freevideo_quality_kernels.py <新配置>`：为新计算源运行两个tiny内核门；不复用不同计算源旧日志。多事件Relay另需对应masked门。
4. 新入口填新配置，默认`user/default/T8/freevideo-runtime-audio-v3.json`；也可用专用`T8_FREEVIDEO_AUDIO_RUNTIME_CONFIG`，不改变旧环境变量。

新表固定[OpenVDN revision b3a8dd8…f158](https://huggingface.co/OpenVDN/vdn-minimax-h3-edge/commit/b3a8dd8d4cf215b3a90bcc20b4362318f5f4f158)，目录`community-sigma3-v1-audio-20261007`、`sampling-presets-v1-audio-20261007`。只下载所需任务／档位，不重下主DiT、不删除旧表、不隐式恢复原投影、不升级共享环境。SDK Apache-2.0；权重／表沿用MiniMax H3 Community License，公开可取不等于无条件再分发。

## 资格边界

已通过新增合同测试及实际作者CPU微型Ref2VA-AV 8+3前向：参考audio行t=1、payload保持、HIGH返回LOW音频exact；各档表identity已核。新源两项微型GPU内核门通过；一张真实保存重开画布完成5秒／LOW448×256→learned1.2×HIGH512×288，实际8+3、120帧／24fps，完整RGB和原生音轨可解码，两完成Stage冷核音频lineage exact，无旧采样缓存。集中人工审核 A1／revision37 已通过，限定成片 SHA256 `fa9152ca28fe09612abe4f9c46466c2becc166a9f8896e197b9f2a0069ed2933`；单个案例不覆盖12／16／20、纯audio参考、长视频或未知LoRA组合。新源整体升级，不称严格仅t=1改变的A/B，也不以同人参考声明声纹锁定。外置EAV／Relay本例report_only，仅验证接线，非增强效果资格。

报告区分计划NFE与实际forward；HIGH3前置sigma=1初始化不计第4步。原件、原生声轨、LOW／HIGH和失败证据保留，不用TTS／静音／降噪掩盖失败。
