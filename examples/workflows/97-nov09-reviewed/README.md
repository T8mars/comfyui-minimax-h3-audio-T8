# 11 套已审工作流

这些图来自实际执行并保存、通过原生前端导入／API导出／保存／重开的配方；对应指定成片已获人工接受。只更新了标题和说明卡，模型、LoRA顺序／强度、采样参数和连线保持。失败的服装与旧返修实验不在此目录。

拖入下列 `.json` 打开画布；同名 API 请求保存在 [api](api)。[index.json](index.json) 保存工作流、API、对应成片及真实反馈的 SHA，供核对来源。效果验收只针对指定素材与配方，不是任意模型、硬件或新素材的保证。

| 工作流 | 用途与实际配方 |
| --- | --- |
| [Long_Garden_12s_4plus4.json](Long_Garden_12s_4plus4.json) | 花园三窗口长链，12秒；每窗口4+4，保留学习型3D放大与原续接 |
| [Living_4plus4_06MP_12s.json](Living_4plus4_06MP_12s.json) | 室内12秒、1024×576，三窗口共24次DiT |
| [PDMD4_06MP_5s.json](PDMD4_06MP_5s.json) | 指定pruned底模、4步、1024×576、5秒 |
| [DMAD8_06MP_5s.json](DMAD8_06MP_5s.json) | 指定DMAD配方、8步、1024×576、5秒 |
| [Control_PDD8_06MP_5s.json](Control_PDD8_06MP_5s.json) | 双参考、分支Guide、动态Canny与非零Control；两个PDD8分支，共16次DiT |
| [Cup_Blue_3Guides_5s.json](Cup_Blue_3Guides_5s.json) | 双人交叉遮挡中的杯子局修；3目标Guide、20外层／2内层、实际57次DiT |
| [Cardigan_Model_Refine_5s.json](Cardigan_Model_Refine_5s.json) | 原片设蓝色控制输入后低噪声模型细化；512×288、5秒、20次DiT |
| [Still_Direct1.json](Still_Direct1.json) | 既有H3 Ref2VA单帧编辑路径，不另建PH3图像栈 |
| [Still_Native22.json](Still_Native22.json) | 原生22帧条件的静帧编辑对照 |
| [Paired_Reference_Windows.json](Paired_Reference_Windows.json) | 两路真实参考片段、配对音轨及局部时间原点 |
| [Four_VAE_Compare.json](Four_VAE_Compare.json) | 同一完成latent的原生FP16／INT8、LightVAE、Sol完整解码对照 |

## 输入与依赖

图中保留实际测试的输入文件名，**媒体、逐帧选区、目标Guide、已完成Stage和模型权重不上传GitHub**。在每个加载节点选择自己的合法输入。工作流与输入副本另已保存在本地审片附件，旧素材不会被覆盖；仅下载仓库JSON不会获得这些私有素材。

模型／LoRA／VAE请使用各加载节点显示的配方，并自行选择本机已安装的文件。四VAE对照须先有自己的完成latent／Stage；局修需对应原片的完整逐帧MASK及匹配姿态的目标Guide。尺寸和时间必须实际匹配，不能用另一片的选区代替。外部Fun Control、VAE及ffmpeg依赖沿用相应节点原要求，不静默下载或换模型。

服装图使用 [局部颜色稳定节点](../../../docs/MASKED_COLOR_STABILIZE.md) 的显式 `source_recolour` 模式**制作采样前控制输入**，保留原片的褶皱和遮挡，再让原FL2VA模型细化。Karras rho=1、20步、sigma0.06→0.001→0；模型解码直接接Composite，生成后没有RGB校色。设色输入另存，不能当作模型生成结果；这是纯换色的指定配方，不是换款式／材质或通用原生去闪烁认证。

换长链参数、模型或素材时使用新的 `chain_id`。未知MODEL／多LoRA组合可按原节点执行，实际接口、维度和完成文件错误照常报告。复用完成阶段、解码失败重试及缓存成本见 [阶段复用](../../../docs/NOV05_STAGE_REUSE.md)。自动检查、GPU性能及人工画质分开记录，详见 [完整验收范围](../../../docs/NOV08_FULL_ACCEPTANCE.md)。
