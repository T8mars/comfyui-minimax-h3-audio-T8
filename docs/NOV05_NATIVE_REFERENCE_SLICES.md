# 使用现有节点准备原生参考视频片段

已有 Source Media Window 和 Audio Conditioning 可以表达逐段引用同一外片的窗口，
无需新增切片节点。这是手动明确各段来源时间的图，不会自动改写已有长视频任务。

以 24fps、每窗 124 帧、相邻窗重叠 22 帧为例，第二窗来源起点为
`(124 - 22) / 24 = 4.25` 秒。两个窗口都从原始来源取 RGB 和 PCM；第二窗必须有
足够素材覆盖至最后采样帧 `4.25 + 123 / 24 = 9.375` 秒，音轨覆盖至
`4.25 + 124 / 24` 秒。若指定上下文，其来源起点应包含上下文；节点不会猜测上下文
长度或把全局时间自动变成生成条件里的位置。

| 现有节点/连接 | 第一窗 | 第二窗 |
| --- | --- | --- |
| MiniMaxH3SourceMediaWindowT8.frames / source_audio | 原片完整 IMAGE / AUDIO | 同一原片 |
| source_fps | 实际来源 fps，例如 24 | 相同 |
| width / height / length | 736 / 416 / 124 | 相同 |
| start_seconds | 0 | 4.25 |
| short_video_policy / short_audio_policy | strict / strict | strict / strict |
| 输出 frames → AudioConditioning.ref_video_1 | 第一窗 RGB | 第二窗 RGB |
| 输出 audio → AudioConditioning.ref_video_audio_1 | 第一窗 PCM | 第二窗 PCM |
| 输出 frame_count → AudioConditioning.length | 124 | 124 |

Audio Conditioning 的 `audio_mode=native`、`task_type=auto`，普通提示可引用
`Video 1` 和 `Audio 1`；`media_map_json` 显示最终编号。视频与配对音轨必须使用同一
Autogrow 序号，稀疏槽位同样按原序号配对。设置输入长度一致，避免消费者把视频截成
更短窗口。`drive_audio` 是目标声音控制，`final_audio` 是后续封装选择，均不同于这里
的参考音轨；参考 soundtrack 不等于强制保留输出原声。

这里引用窗进入原生 `video_audio` 块，视频/音频 latent 共用该参考块的局部时间原点。
Qwen 视频抽帧时间也从窗口局部 0 开始，不表示全局生成时间锚定。来源非 24fps 时，
按真实时间用最近帧转换至 24fps；RGB 使用既有 Lanczos/量化路径，PCM 转为 32kHz
立体声。所有 IMAGE 已经驻留内存，不声称流式读取或降低完整来源的内存成本。

短于默认官方 2 秒范围的实验窗口需显式选择 `reference_video_policy=model_minimum`；
默认策略不变。Source Media Window 的长度向上对齐 17n+5；检查实际 `frame_count`
和报告，而不要只用请求长度计算结束时间。strict 音轨策略仍允许未连接 AUDIO 时明确
生成静音；要求配对原声时必须连接来源音轨，并检查报告 `audio_source=connected_audio`。

`ReferencePackage(kind=video, audio=...)` 的音频是独立 voice anchor，当前路由输出
`video` 与 `audio` 两个块，不能等同上述同轴 `video_audio`。参考包适合可认证的可移植
素材与角色选择；未知 VAE 可走普通引用路径，不伪造包的缓存身份。
`TimedVideoReference` 则只把指定时间的画面给 Qwen，不增加原生 VAE 引用槽或音频。
三种语义不要互相替代。

新增 `tests/test_nov05_reference_slices.py` 实跑两个 30fps 来源窗口（0.5 / 1 秒），
核 RGB 来源索引、PCM 起点 16000 / 32000、各 29333 样本，实际 tiny Core 编码后
分别送入完整两次前向的 AV 采样。观察到的原生块为 `video_audio`，两窗口的实际
视频/音频编码不同，来源未修改；缺视频或音轨范围会在编码前报错。3 项通过，CUDA
未初始化。Qwen 使用明确的测试 embedding 替身，VAE/DiT 是随机小模型，因此该证据
证明软件和时间轴接线，不证明语义理解、训练模型效果或长链质量。

后续补充全尺寸训练模型 GPU 请求：同一实际影片的 0、0.75、1.5 秒来源窗分别裁切
RGB 和配对 PCM，观察原生消费者收到三个不同编码的 video_audio 块，以及每块局部
零时间原点。完成 48 帧 AV 输出；不是将视频窗口条件冒充强制拷贝原声。完整影片的
画面和声音单独人审，见 [完整验收补充](NOV08_FULL_ACCEPTANCE.md)。
