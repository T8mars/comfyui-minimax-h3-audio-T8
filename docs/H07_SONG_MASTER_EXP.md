# H07：真实录音演唱配方（EXP）

复用既有 Master Audio 与分离 4+4，不增加采样器、不改变旧工作流。
一次独立真实画布保存、关闭重开、UI Run 已完成：448×256 → learned 1.2× →
512×288，最终 120 帧、24fps、5 秒。技术完整性及集中人工审核 A2／revision37 通过；
不是所有歌曲、方言、多角色、声纹或预滚效果的资格证明。

## 来源与三种音频角色

代表素材为 [Derrick Coetzee / Dcoetzee 的清唱录音](https://commons.wikimedia.org/wiki/File:Twinkle_Twinkle_Little_Star_-_sung_with_full_lyrics.ogg)，
页面明确自录、a-cappella、CC0-1.0。原 OGG SHA256：
`0a9b05a0929699e30cf5f6f124c6e05251ed1505e2a5eb954746e7cd6a6c2f40`。
本地准备显式将 96kHz 单声道转为 32kHz float WAV；这不是原件字节或无损转封装。
实际录音不来自 TTS。人物参考是本项目合成角色，不是录音者肖像。

- DRIVE：完整同源清唱，用于表演时间；清唱 master 本身就是人声，不伪称分离 stem。
- VOICE：同一录音另取 16–19 秒作为声音参考，不是额外歌词时间线。
- FINAL：完整原录音，仅用于最终裁切和输出；不使用生成／解码的人声替代它。

共享源起点 4 秒，准备 136 帧 / 24fps。视频和 FINAL 都裁去前 12 帧（0.5 秒），
交付源 4.5–9.5 秒的整句“How I wonder what you are”。离线 ASR 仅辅助选区间，
不是口型、歌词或真人质量判定。只跑有预滚的一次，不声称 0 与 12 帧对照收益。

## 接线要点

保留既有独立 LOW/HIGH 几何与模型组合；外部原声驱动强度 0、完整音频 mask=0
是显式设置，不把未完成的一采生成音频冻结成最终原声。既有 Audio Audit 检查并
回填完整源 audio latent；最终 mux 仍直接选 FINAL 录音。EAV / Prompt Relay 可
外置接入，但本案例未启用，不冒称其效果。

`tools/prepare_h07_song_source.py` 准备已核来源；
`tools/prepare_h07_song_canvas.py` 从已接受 Master Audio 原生图复制到全新私有
目录，保留 39 节点、65 条原接线，仅替换明确源／提示词／seed／裁切／输出名。
`tools/audit_h07_song_canvas.py` 核原生保存图与实际导出 API；
`tools/collect_h07_song_canvas.py` 核实际 history、全部帧与原声区间。
工具不排队 API 采样，不自动下载其他歌曲或替用户接受许可。

## 已取得证据及限制

真实 UI job `40fc6c63-711b-4c5e-8226-b2843de0c02c` success，空缓存，91.784 秒。
两阶段实际调度各 4 步；未计每个底层去噪调用，不能扩大为任意补丁总 NFE。
实际 DRIVE/FINAL 的保存 FLAC 与 WAV 源仅有既定 16-bit 编码误差。
完整输出 120 帧 PTS 正确、有限且非黑屏；解码 AAC 与目标原录音相关性约
0.999935，AAC 尾 padding 明列，不声称 PCM 或文件字节 exact。

原生图 SHA256：`6b40469d501e9d36833f1a760d0dd95cb156a28104d13a00465e03d027869d30`。
成片 SHA256：`eb24e11335c7a62e9d8adb8b12172a1213d18ab2febd2dd5fb901adedccfe27a`。
所有原件、原 history、源音频预览和失败准备 epoch 本地保留。
用户已通过本片的集中人工审核；尚未发布推荐模板，不因此重采本已通过项目。
普通对白旧资格仍保留，不替代歌曲验收。多方言／多角色仍须实际同源角色与区间，
不承诺本代表自动获得这些用途的质量资格。
