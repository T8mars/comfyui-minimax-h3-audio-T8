# H07：指定 Hybrid 姿态路线资源预检（EXP）

2026-10-10定向复审revision10，新的可见注液／关龙头／移杯拟音A6已通过。
保存[A6_Foley_Half.json](../examples/workflows/85-h07-reviewed/A6_Foley_Half.json)，
详见[拟音用法](H07_FOLEY_USAGE.md)。原姿态A4通过不变，旧空庭院拟音失败不推广；
下方新片“待听审”为过程历史，不能据此宣称所有动作、声纹或6×加速已验证。

本项完成资源预检、一个实际 CPU 姿态源及真实画布 5 秒采样，完整音画和
保存的 latent 检查通过，**指定姿态成片人工审核 A4／revision37 已通过**。旧模型、图和普通
Fun 控制默认值不改。既有 control 文件实际存在于 `models/controlnet`，
不是零字节的 Union2 占位，也不能拿普通 Ref2VA 代替指定 hybrid。

## 已核资源

| 实物 | 固定版本／完整 SHA256 | 当前资格 |
| --- | --- | --- |
| smhfacct b25–49 INT8 hybrid | revision `a36feb17fbd1f20ff4bdd509ccd07e2b7b585a38`；SHA `a629cfea8d89a071b140c6e1935dc9a23e72de6badc18975a2bb9e6d1423d76d` | 官方原字节 20,970,379,632 B；原生 ComfyUI 键，无转换 |
| 已有 pruned Fun Control | SHA `9c645c0a308c8af361efd43b409710f6f8fec0db297c29503e141a84991fed0c` | 本机完整 SHA；5 个控制块、49 输入通道，AdaLN 8-wide |
| hr16 YOLOX TorchScript | revision `a124b32c3b7c5cebda1c7cd96178f0f9d2050125`；SHA `80bc14b13c260c24b3014cd42c02994bf52296ab8fa2d80a60b6afe08c93ef42` | 官方原字节下载、217,697,649 B |
| 已有 hr16 DWPose TorchScript | revision `359d662a9b33b73f6d0f21732baf8845f17bb4be`；SHA `d86a0b2b59fddc0901a7076e9f59c9f8602602133ed72511c693fd11eea23d91` | 本机原件与官方 SHA 一致，135,059,124 B |

Hybrid 50 个块的实际 AdaLN 投影和 control 5 个块均为 `[96768,8]`；
曲线表 `[1025,8]`。本次只是实际文件头配对，不替代加载严格度、运行期
控制调用、动作遵循或音画质量。`tools/preflight_h07_pose_assets.py` 做
独立完整 SHA 与有限文件头检查，不 Queue／执行模型／下载。新增三个
header 门合同通过；没有重跑旧功能套件。

## 来源与许可边界

感谢 [smhfacct Hybrid 作者](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models)
与 [Smite79 指定路线](https://huggingface.co/Smite79/MiniMax-H3-Longvideos/tree/32b8ad122623007375048ac68767994036054748)。
Hybrid 继承 [MiniMax H3 许可](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)，
不是 MIT。Smite 固定来源使用 `h3-longvideos-no-redistribution`：可按其许可
本地使用与修改，不因此获得分发其插件的许可；本项目没有复制、安装或
执行其受限插件，也不打包权重。

[YOLOX](https://huggingface.co/hr16/yolox-onnx) 与
[hr16 DWPose 权重卡](https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5)
标 Apache-2.0；这不能自动证明本机外部预处理器每份借用的绘图源码同许可。
已装 DWPose 包部分文件有 CMU 非商用来源提示，不复制该实现入本项目，
不声称获得任意商业再分发权。后续采用明确外部依赖或独立绘图实现时，
仍分别核源码与权重许可。

## 下一步最低量

`tools/prepare_h07_pose_source.py` 只在显式 opt-in 后加载固定原件的
YOLOX / DWPose TorchScript，CPU 提取完整 120 帧：同源 512×288 缩至
256×144，没有时间插值、额外裁剪或隐藏人物补造。实际约 181 秒、CUDA
未初始化、0 H3 NFE；原始检测框、133 个关键点及分数逐帧保存。
自有图元绘制未导入 CMU OpenPose 绘图代码、整个外部节点或受限 Smite
插件。此示例的交叉遮挡会从两人检测减少到一人，不证明身份跟踪或全身
动作、面部、手指控制都可靠。九个源／姿态 contact 时间点已查看。

`tools/prepare_h07_pose_canvas.py` 准备独立 24 执行节点／38 边的原生图。
新文件链接指向 G 盘精确 hybrid 原件，现有 Core 模型列表实际可见，没有
重复复制 21GB。pose 124 输入使用 `0..119+[119]×4`，最后四帧只用于 H3
隐藏尾部；与原声 Conform 同时钟。两条 Apply 输出均已接入，最终音轨明确
旁路原声，不把这份配方当成生成声音质量测试。

仅当前 `H07_Pose_Draft.json`（SHA256
`746a7f93fc685c0795a765567e5ce0bc8ee8923d80f24ae1de0ae889ee30d078`）
准备通过实际 schema／combo／接线类型检查；随后实际另存
`H07_Pose_Native`、导出 API、关闭重开并从 UI Run。
初版错误 Qwen 文件名的准备件及拒绝回执保留，不用于采样。
新准备数学三个合同通过，没有重跑旧测试矩阵。

只验收这一份 5 秒草稿，再接既有 T8 Fun Apply 与指定 hybrid。
显式新配方为 512×288、无加速 LoRA、40 原生步、strength=1、end=0.6；
不是修改旧默认或声称逐值复现作者 recipe。early control 范围是显式新配方，
不能偷改旧控制默认；真实画布另存重开后做一次完整成片，最后集中人审。
真实 job `f2c569f7-8e20-4ff3-8ff5-42194840a615` 成功 210.221 秒；
只复用视频／音频 VAE loader 两项缓存，采样、checkpoint、解码、裁剪和保存
在本次执行。实际 Apply 报告为 `official_model_patch`、live AdaLN 8/8 配对；
未插桩统计控制塔／残差调用，不能声称其次数或所有动作控制均已证明。
全 120 帧、24fps、5 秒、音轨和完整原生 AV checkpoint 经只读检查；
浏览器实际 MP4 播放至 5 秒末帧、无解码错误。完整 contact 已查看，转身和
交叉遮挡可见，但源检测漏人、手部／杯子细节及身份跟踪不因此获通用资格。
原声音轨图上旁路不变，AAC 与源双声道相关约 0.99875／0.99800，并非 PCM
逐值相同。模型完整 SHA 在真实预检核验，收集后仅记录文件 stat，未假称再次
扫描 21GB 或证明运行期文件不可变。新收集器 `tools/collect_h07_pose_canvas.py`
不排队或执行模型。

## 独立无对白半尺寸拟音（真实画布完成，人工审核未通过）

无对白 half-size foley 是另一个用途，不由姿态成片推定。本次复用实际
完成、完整无声的庭院 plate，不用有对白的姿态片冒充。原视频 512×288，
仅内部采样视频 area 缩至 256×144，再上下各补 8 像素至 H3 网格 256×160；
不拉伸比例、不减时间帧率。124 帧仍为原 120 帧加四帧隐藏尾。视频 mask=0、
音频 mask=1；`require_lanpaint_sampler=false`，未执行外部 LanPaint 采样。
使用已核 b25–49 hybrid，40 原生步；联合 transformer 并非数学隔离的音频模型。

最终 IMAGE 显式旁路原完整 512×288 视频，只采用生成的风声／叶声音轨。
AVC 重新编码不是压缩视频字节不变，实际解码 RGB 对源 RMSE 约 2.287／255。
不替换或修改旧对白原声工作流，不声称作者约 6× 为本机收益。

`H07_Foley_Native` 实际另存、导出、关闭、保存目录重开、UI Run，job
`926ebeae-1854-419a-bc0d-88bda6bf7c0b` 成功 47.384 秒，25 执行节点／37 边。
完整原生 AV checkpoint 经 CPU loader SELF_VERIFIED，视频／音频及 mask
finite，最终全 120 帧／24fps／5 秒、160768 个解码音频 sample 经核验。
新片 SHA256：`c064005b7861a4b60584a0723b1e49249b33976ef184f64d3d162cddccde548f`。

音轨非零但非常轻：RMS 0.000757、peak 0.003632。没有偷偷放大或替换声音；
不能据非零音轨写“质量通过”。集中人工审核 A6／revision37 未通过：用户指出
空庭院只有树，没有明确可见的发声动作，无法判断动作拟音能力。此片不进推荐。
后续只换成合法、有明确可见发声动作且无对白的同源 5 秒素材，原画面按约
0.4MP 基础清晰度交付，半尺寸内部支路明示；不替入现成拟音或偷偷加增益。

### 可见动作定向复审（真实画布完成，待听审）

`tools/prepare_h07_foley_recheck.py` 另备一个25节点／37边的5秒图，
不改原失败图、runtime、默认值或对白链，不提交Queue。真实素材为
[Jane pouring beer／Angulidayaaluta](https://commons.wikimedia.org/w/index.php?title=File:Jane_pouring_beer.webm&oldid=1253219481)，
按该页面的 [CC-BY-SA4.0](https://creativecommons.org/licenses/by-sa/4.0/) 使用；
本机保留原件及归属记录，衍生片沿用相同许可并说明修改，不暗示作者背书。

选择原片23–28秒，画面可见龙头向杯中注液、约3.5秒关闭及随后移杯。
原1920×1080降采样至864×486，居中上下各裁3像素得864×480／0.41472MP；
转24fps完整120帧，源所有录音移除、另置32kHz双声道全零AAC。
实际全帧／PTS及解码音频全零已核，160768样本含768个AAC padding样本；
不是既有低清片放大，也不是将现场录音混进生成声音。

内部仍明确half-size：432×240四边各补8像素至448×256，原124时钟、
40步、seed2610091202、VIDEO mask0／AUDIO mask1及最终源IMAGE旁路不变。
只采用模型生成拟音，不增益或替SFX；它是音频专用路线，不额外塞入
learned1.2×来冒充二采。真正双采新候选仍按一采约0.4MP／learned1.2×。

两个新增准备合同与实际服务schema检查通过；新源和草稿均独立保存。
随后真实另存 `H07_R37_Foley`、导出 API、关闭并从保存树重开；UI job
`a56af5d1-15fc-4c04-9b94-67b2396c033d` 成功 285.179 秒，缓存仅原件 loader。
完整 864×480／120 帧／24fps／5 秒、全部音画 PTS 与原生 AV checkpoint
SELF_VERIFIED 通过，VIDEO mask0／AUDIO mask1。最终片 SHA256
`bfa8426514c2b7dafc90747009620eaf32e00ff30cc65e695185d3f6e68807bd`。
新声音 RMS 0.017382、peak 0.586576，未增益、未替入现场音或 SFX。
视频重新编码 RGB 对源 RMSE4.476／255，不称压缩字节相同。
完整120帧contact已检查，实际画布播放可见进度4.646秒，原生播放器循环，
不把 loop 的 ended=false 改写成播毕证明。动作同步、自然度、杂音与意外语音
仍须最后集中听审，A6尚未获人审通过，也没有6×加速证书。原空庭院失败片保留。
