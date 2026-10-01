# S25 Motion Recovery 分离式双采（EXP，仍未终验）

本目录保留旧公开 Fullclip／Windowed Stock20 工作流，另新增 26 张分离图。每种路线有无效果、仅外置 Enhance a Video（EAV）、仅外置 Prompt Relay、两者组合四种配置；每种配置提供完整运行、完整运行且显式保存首采、仅冷二采三张图，另外每条路线各有一张独立“只跑并冻结首采”图。没有迁移或覆盖旧图。

建议先导入 `none_full`，确认首采 `SamplerCustomAdvanced` 与二采 `MiniMaxH3StageSamplerEXPT8` 是两个独立节点。Relay 和 EAV 只挂在来源绑定后的二采，选择某个效果不会强迫同时启用另一个；默认 EAV `report_only` 只观察，不增强或承诺画质。原 Stock20、Motion Recover／AutoGate、Windowed 的 hot-ranges 语义和 `pass1_original` 音频策略保持。示例自行填写本机模型／素材与提示词。

`full_save` 使用非输出透传 Save 与 Motion／EAV Audit，让保存首采及审计成为最终视频的必经上游；`freeze_first` 只生成首采视频与冻结文件。两者都默认 `confirm_save=false`，须显式确认才会写检查点。把 Save 返回的相对路径、完整 manifest JSON 和整个文件 SHA-256 填到同一路线的 `cold_second` Load；冷图没有首采 sampler，但会读首采并重新执行 Motion 来源绑定和二采。空占位不能直接排队，也不自动接受冻结时或新生成的画质。不要混用 Fullclip／Windowed 的回执、变更已冻结首采的模型／参数后冒充同一次运行，或绕过来源审计。

当前 26/26 图已通过构图器与 Core 全输出静态校验、真实 Chromium 原生 Save As／重开语义审计；12 张效果保存／冷图的 Relay／EAV 控件做过可见编辑并保持。新旧相邻 Motion 测试与本图的真实权重探针合同合计 74 项 CPU 通过。Fullclip 与 Windowed 的**已保存公开** `combined/full`、`none/freeze_first`、`combined/cold_second` 已分别在三个独立 Core 以真实 FL2VA／Qwen／双 VAE 和128×64／22帧运行：完整图二采10步、冻结首采20步、冷图首采0步／二采10步，外置 Relay／EAV `report_only` 实调用，冷图与完整图的解码 RGB24／PCM 音频 SHA 各自相同。完整图有两个旧 VHS 输出端，当前 Core 会对相同种子的 Stock20 分支各执行一次，所以观察到40个首采进度事件；这不是40步首采。首轮 Fullclip 探针误以为只应有20事件而报失败，保留失败报告后依据旧图行为独立复核通过；Windowed 原探针与独立复核都通过。

`full_save→cold_second` 此后又分别以修订后的保存图完成 Fullclip、Windowed 的小画布真实权重实跑：各只生成一份首采检查点，冷图无首采采样，解码 RGB24／PCM 各自与完整保存图相同。Windowed 私有回执为 `artifacts/development/modular-formal-motion-s25-fullsave-real-gpu/20260927T154322Z-Windowed-combined-report_only/report.json`，九项检查全真；Fullclip 回执见 `roadmap.md`。两个旧 VHS 输出仍可能使 `full_save` 的首采分支重复求值；单次保存不等于单次计算。

这些是固定小画布、指定资产和手工 Motion 范围的**机械资格**，不等于26图全部运行，也不等于原尺寸／长片、其它资产／后端、EAV `apply_exp` 画质、旧新同设置画音或真人审片完成。浏览器检查本身未排队生成；真实生成只发生在隔离的 GPU 探针中。
