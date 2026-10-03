# 1.89.0 — RADAR 集成与分离式扩展（EXP）

本次为向后兼容的新功能版本。原一体节点、采样默认、音频时钟、已发布工作流及完整时间HyperFlow路线保留；新增功能须显式接线，不自动迁移旧工程。

## 新入口与工作流

| 入口 | 交付与边界 |
| --- | --- |
| [导演台源证据／Skill](DIRECTOR_SOURCE_EVIDENCE_SKILLS_EXP.md) | 手动事实审核、创作意图独立、声明式镜头Skill快照、显式候选采用与工程保存／重开 |
| [外部Take](DIRECTOR_EXTERNAL_TAKES_EXP.md) | 外片登记、实际SHA绑定、手动采用与整数帧裁取，不伪造原生采样祖先 |
| [65 可见脸MASK](../examples/workflows/65-radar-visible-face-mask/README.md) | 16张Full／Cold模板；优先新逐帧MASK＋显式denoise0.05／原12步配对，外置MASK不是自动遮挡分割 |
| [66 无脸lazy](../examples/workflows/66-radar-no-face-lazy/README.md) | 6张模板；完整检测一次、unknown不冒称无脸，人工确认具体SHA后旁路，未选重分支实际不执行 |
| [67 外片续拍](../examples/workflows/67-radar-external-continuation/README.md) | 4张Full／Cold模板；原RGB／PCM明确绑定，EAV／Prompt Relay外置，完整Stage后零采样交付 |
| [68 HyperFlow曲线](../examples/workflows/68-radar-hyperflow-curves/README.md) | 3张HEAD4／TAIL4、冷TAIL4、完成TAIL零采样模板；15个专用fit、阶段及外置效果节点 |
| [69 原生配方](../examples/workflows/69-radar-native-recipes/README.md) | 4张Union2原生40步与A/B交替独唱完整／恢复图；不是同框双唱或移址缓存认证 |
| [70 模型对照](../examples/workflows/70-radar-model-compatibility/README.md) | 6张LMS、Orbit、Wallpaper R32单变量对照图，沿用现有加载器 |

65–70合计39张前端JSON。所有模板必须替换明确素材／模型占位；恢复图还须填真实path、manifest和完整SHA，不能把占位值直接Queue。完整[工作流目录](../examples/workflows/README.md)与旧[分离采样索引](../examples/workflows/SEPARATED_WORKFLOWS.md)均保留。

## 验证与限制

本轮本地55个完整技术范围896项通过、0跳过；另在实际发行树和官方打包产物中复核注册、旧schema、源码及工作流。该数字不是整仓全量或所有GPU组合通过声明。真实原生画布Save／重开／Queue成片与新进程冷恢复另有独立资格。

人工审核覆盖10个指定代表：LMS／Orbit／Wallpaper、导演台候选、Union2原生40步、A/B独唱、外片续拍、曲线双采，以及修复后的可见脸MASK和真实脚步／地面无脸旁路。原R07/R08失败片保留；通过结论不授所有素材、家族、时长或后续媒体。

- R07黑区保护原片，白区使用已有候选；MASK不能修复候选内已有错误头姿／手部幻觉。旧失败静态遮罩配方不推荐，旧默认不修改。
- Curve fit必须绑定准确底模、教师、原adapter及文件内容；调制残差降低不等于完整教师或画质等价。当前实跑显式关闭dynamic VRAM并预留4.5GiB，dynamic清理失败保留，不降低资源保护。
- Union2旧Turbo4+4彩色方块失败保留，不作推荐示例。原生40步也不保证黑区RGB或生成音频完全不变。
- Orbit内部仍是joint AV，仅输出静音视频，不冒称作者audio-off等价；Wallpaper仅R32／Ref2VA／Turbo4对照，不冒称R64／Taomate3／LMS多因素资格。
- 实测原Core缓存已避免同key重复Qwen编码，未证明额外LRU净收益，因此不叠第二缓存。原条件停止的来源、硬件和授权候选不伪称实现。
- 不恢复QuantFunc／Nunchaku，不升级共享Core、不重启用户服务，不把未验证的用户LoRA／Sage／Sol组合一刀切禁止。

本GitHub仅包含源码与通用模板，不含私人模型、视频、音频、实际Stage文件或本地交接。GitHub版本发布、Registry上传与Registry可安装是三个独立状态；Manager未显示新版本时可从GitHub安装，不能把上传成功写成Active。

English: This compatible EXP release adds explicit source-evidence/skills and external takes, Union2 and alternating solo cast, visible-face masks and confirmed lazy no-face bypass, and independently wired curve HEAD/TAIL with external EAV/Relay. The six new workflow directories contain39 reusable placeholder graphs. Human approval applies only to10 specific representative cases. Legacy workflows/defaults remain; no arbitrary-material, hardware, dynamic-VRAM, speed or teacher-equivalence guarantee, no private weights/media, and no Registry-installability claim.
