# v1.88.1 — 分离式音频复审修正

发布到 GitHub `main`，同时提供正式 Release。保留节点/模板本身的 EXP 边界；GitHub 正式版本不等于全部配置质量认证或 Comfy Registry 已可安装。

## 本次内容

- 新增 `MiniMaxH3RFRestartJointClockSetupEXPT8`：明确选择已经联合重噪的RF初始化策略，避免原通用路径第二次音频rebase。旧RF节点、默认策略、采样数学、原Core inpaint锚点/NOISE/masks以及联合AV参与不改；新策略进入独立StageContext身份。
- B8 Long Relay保留原4步、同源/种子和原时间投影，仅新增音频尾采强度0.35模板。它不使用RF修正，不是通用降噪或音轨替换。
- [六张修正版配对图](../examples/workflows/64-reviewed-audio-followup/README.md)：B8视频Freeze/音频Resume，C1/C2完整BASE＋RF/只读BASE后RF。模型、LoRA、EAV和Prompt Relay仍可外置独立编辑。
- 补齐此前漏收入GitHub的[四张S04 PDD分离图](../examples/workflows/19-pdd-acceleration/README.md)：FL2VA/Ref2VA各完整LOW4/HIGH4与冷HIGH4；不改变原八次前向配方。
- [全部S01–S29及Veda保存入口](../examples/workflows/SEPARATED_WORKFLOWS.md)。既有已发布JSON原样保留，不迁移旧图。

## 验收与限制

用户2026-10-02分别通过B8/C1/C2定向复审，连同先前19项构成22项代表审核闭环。C1/C2是在原生画布载入同源已存BASE、各仅原3步RF的完整124帧448×256/24fps5.167秒音画；B8是同源/同seed原4步/.35音频精修候选。通过只绑定这三份实际片，不把旧失败片、通用模板默认尺寸或任意资产组合改判为通过。

此前当前源码九个完整RF范围239CPU测试及六图保存/旧图回归三个完整范围47CPU通过，范围重叠不相加；旧581个实际Core接口/顺序保留，仅末尾追加新节点。发行候选另按实际新源码、官方CLI包、582节点和全部旧schema/JSON逐项校验，不沿用旧包资格。没有为发布再次生成已通过的片，也不恢复用户取消的矩阵。

恢复图路径/manifest/SHA故意留占位：先生成自己的完整首采并显式保存，再填写真实值。保留QualityGate/confirm手动决定与原音轨，不自动接受候选。审核用精确画布副本依赖本地隔离运行资产，不随GitHub/发行包上传。

QuantFunc/Nunchaku和Registry自动监控不恢复；不上传模型、检查点、反馈、媒体、roadmap或SKILL。Registry状态与安装可见性独立，不以GitHub推送或新版本号冒称已Active。

English: main + regular GitHub release; opt-in RF one-time joint-clock initialization, a separate B8 four-step denoise0.35 recipe and additive full/cold paired templates. Three specific native-canvas candidates passed human re-review; this does not certify arbitrary materials. Legacy RF behavior, all previously published workflows, original joint AV and manual acceptance remain. Registry installation is an independent state.
