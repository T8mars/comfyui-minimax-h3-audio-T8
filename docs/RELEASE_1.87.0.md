# v1.87.0 — Semantic Bridge 自动参数与显式多桥组合

本版追加两个EXP节点，保留既有节点ID、注册前缀、输入顺序与默认参数，不迁移旧工作流。训练器trans／可变宽MLP兼容继续保留。

## 接线

每个模型各接一个 **H3 Semantic Bridge / 自动模型参数 (T8 EXP)** 或原手动配置。
两个配置接 **H3 Semantic Bridge / 多桥组合 (T8 EXP)** 的 first／second，更多模型继续串接组合节点，最多8桥。
最终输出只接一次Apply，或Relay／内循环的既有语义桥插口；不要串多个Apply，也不要在Relay绑定后重复应用。

| 模型内容 | 自动alpha | 其余参数 |
| --- | --- | --- |
| 原六张量Original／BUNNY | 0.10 | per_token、all_tokens、chunk=256 |
| Comic-combat固定应用契约 | 1.0 | per_token、all_tokens、chunk=0 |
| 精确SHA识别的Wushu v1 | 0.12 | per_token、all_tokens、chunk=0 |

预设根据权重内容或固定metadata，不根据文件名、训练日志或“所有trans都一样”猜参数。
未知自训权重无可靠预设时使用手动入口及模型卡；固定契约不符提前报错，自由手动设置不同只提示。
Wushu v1预设不代表v2、重训或改包权重。模型位置与完整接线见[语义桥说明](SEMANTIC_BRIDGE_EXP.md)。

## 计算与兼容

- 多桥为有序串行，不是LoRA权重相加或并联平均。每桥独立强度，顺序影响结果。
- 全体先预检，同SHA重复选择拒绝；禁用配置完全旁路。取消不修改原条件。
- 完整组合及各阶段receipt保留；模型、顺序、参数变化会改变缓存身份，使用新chain_id。
- 训练器MLP的residual_skip底座修正为归一化输入；旧六张量数学不改。
- 权重和metadata读取同一份哈希绑定字节；Transformer构造不消耗调用者CPU随机数状态。
- 不改变采样步数、sigma、音频路径、学习型3D放大器或已接受的接缝配方。

## 证据与限制

已验证软件回归、两个新模型与训练器的CPU/CUDA数值对照、原版／BUNNY旧数学逐值一致，以及真实原生CPU画布配置／组合Queue、可见编辑、另存和刷新重开。
画布验证只执行配置／组合和报告，没有编码器、Apply或H3采样。新多桥完整视频、主观画音、多素材及8桥性能仍未验收；旧样片资格不自动继承。

GitHub版本与Registry安装状态分开。只有官方状态Active、公开latest更新并确认实际安装可见，才能报告Registry可安装；上传成功不等于该门禁通过。
本版不包含暂停中的Veda／分离采样开发，不包含模型、媒体、私有交接或验收工件。

English: this backward-compatible release adds content-bound Auto Config and explicit ordered Compose (up to eight individually configured bridges). Legacy manual defaults and old workflows remain unchanged. Trainer residual MLP uses normalized input. Numeric and configuration-canvas checks do not qualify new H3 video/voice quality, all materials or Registry installation availability.
