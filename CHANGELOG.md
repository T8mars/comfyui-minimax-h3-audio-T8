# 更新日志 / Changelog

[首页](README.md) · [English](README_EN.md)

## 当前版本：1.95.0

### H05 已审工作流与独立节点

- 11项指定案例通过人工审核，保存短名公开图；旧一体／分离图、FreeVideo四档／8+2／4+4和原节点默认保留。
- 新增Qwen独立参考视图、Source Clock／Conform／Control Map、独立LanPaint音频时钟适配、RES历史／完成态与外置EAV／Relay、独立FreeVideo视频解码、连续坐标learned放大、显式外部参考导入。
- 导演台只读失效说明／参考槽／计划时长；配方卡区分未限定alpha；冷H3音频参考crop兼容和Windows自有输出flush修复。
- A1仅扩大可见杯子＋握杯手回贴，blend5；B3为tau1.5／10%–90%，实际gain全1，不声称非零增强。失败实验不作推荐。
- 无新增验收采样或重复旧CPU测试；不是H05全部许可／来源／硬件研究完成。详见[发布说明](docs/RELEASE_1.95.0.md)和[工作流索引](docs/H05_REVIEWED_WORKFLOWS.md)。

## 1.94.0

### FreeVideo 新四档独立入口（EXP）

- 新增8个Quality v2入口：四档加载／采样、Light独立HIGH3、Stage保存／冷载入、外置LoRA／EAV／Prompt Relay。保留旧663完整接口、默认、8+2、真正4+4和六张旧图。
- Light为LOW8→外置放大→HIGH3；Medium／High／Max分别联合单采12／16／20，不强制双采、不混用旧MID。复用旧rowwise主体，按实际任务补采样表，不转换为普通UNET、不升级主Comfy环境。
- Light 8+3与Medium 12两条5秒原生画布完整音画获人审通过；公开4张短名Full／Cold模板。16／20仅CPU结构支持，Cold无额外GPU，EAV本次仅report_only。详见[1.94.0说明](docs/RELEASE_1.94.0.md)。

## 1.93.0

### RADAR r6：参考包与分离交付（EXP）

- 新增六个可组合参考包节点和一个master后处理保存出口，当前663个ID，保留旧656的完整接口、默认与工作流。
- 模型配方卡提供header证据与明确差异确认；导演台新增两条声明式连续性／画外对白规则，不自动保存或排队。
- 普通VAELoader支持固定LightVAE解码图；公开10张原生保存来源的短名图及三张通用master模块，私有素材／缓存不随包。
- 五项指定代表已获人工通过；Cold、模板、EAV report_only、参考producer和Light仅decode的边界明确，不承诺所有素材／身份／速度。详见[1.93.0说明](docs/RELEASE_1.93.0.md)。

## 1.92.0

### Temporal Chunk 窗口对白作用域（EXP）

- 新增24个显式选择的入口：Global／Performance／Speech分离、首窗起原生文本重编码、v5及H16独立前缀合同、整合／分离、外置EAV／Relay、完成态音频门和literal Full／Cold。
- 固定v5原生4+4／12秒／187／34样片已获用户通过；保存4张原生画布公开模板和4张外置效果CPU接线模板。H16仍仅CPU资格，不泛化画质或硬时间隔离。
- 旧632接口、默认和工作流保持，缓存家族不混用，不提交私有交接／模型／媒体。详见[1.92.0说明](docs/RELEASE_1.92.0.md)。

## 1.91.0

### FreeVideo FP8 与真正分段4+4（EXP）

- 新增12个独立引擎节点：加载、在线LoRA、8+2 LOW/HIGH、Stage保存/冷载入、外置EAV/Relay、4+4 LOW/HIGH与MID保存/冷载入。旧620入口及默认不变。
- 三条固定5秒完整音画已获用户通过，保存六张可编辑Full/Cold模板；4+4继续实际未完成音频，8+2保留完成LOW音频，缓存类型不混用。
- 独立Python与官方rowwise FP8模型，不升级主环境、不冒称普通UNET转换；模型/配置/素材/私有交接不随包。详见[1.91.0说明](docs/RELEASE_1.91.0.md)。

## 1.90.0

### 2026-10-04 Kijai 九原件与分离双采（EXP）

- 新增 Kijai Acc-8Step 相对32头独立加载入口，区分FL2VA／Ref2VA、Full／Pruned，保留backbone、AdaLN、bias及绝对0:4／4:8联合音频。
- 保存九个家族18张Full_Save／Cold_HIGH；MODEL、Noise、条件、EAV及Prompt Relay独立外置，旧图不覆盖。
- DMAD、PDMD4、ELM、FlashGen与Ref Difference保持各自已验收配方；Ref Difference非加速、ELM非无限KV。只读Core Bypass身份与显式默认LCM识别，不清空用户补丁或替换旧数学。
- 九份指定完整音画已获真人通过；发布模板仅清理注释／私有元数据及本地LoRA别名，素材、Stage和编辑后质量不随包认证。详见[1.90.0说明](docs/RELEASE_1.90.0.md)。

## 1.89.0

### 2026-10-03 RADAR 集成与独立阶段（EXP）

- 新增导演台手动源证据／事实与意图分离、声明式Skill快照、显式候选采用，以及外片Take登记／采用／续拍；不自动猜台词、采用或触发队列。
- 新增Union2专用控制、A/B交替独唱、外置可见脸MASK、完整检测与显式确认的无脸lazy旁路；旧节点、旧默认和旧工作流保留。
- 新增独立HyperFlow pruned曲线fit／加载／HEAD／TAIL／存取／外置EAV与Prompt Relay入口；不替代完整教师路线或普通加载器，不认证Core dynamic VRAM模式。
- 保存65–70目录39张通用模板，附LMS／Orbit／Wallpaper R32现有加载器对照；指定10项代表用例已获人审通过，不泛化所有素材、硬件或作者完整配方。
- 技术资格与人审分别记录；原失败证据保留，不打包私有模型、媒体、实际检查点或本地交接。详见[1.89.0说明](docs/RELEASE_1.89.0.md)。GitHub与Registry可安装性独立。

## 1.88.1

### 2026-10-02 分离式音频复审修正

- 新增显式 JointClock RF Restart 节点：RF 联合重噪后各音频/视频时钟只初始化一次，避免再次音频 rebase；旧节点、默认数学、原 inpaint 锚点/噪声/遮罩、旧工作流保留。
- B8 Long Relay 新增独立4步音频尾采强度0.35配方；B8/C1/C2指定原生画布候选均已获用户复审通过，不泛化到任意素材或模型。
- 保存六张完整/冷恢复配对修正版，补齐上次未收入GitHub的四张PDD分离图；新增[所有分离图入口](examples/workflows/SEPARATED_WORKFLOWS.md)。恢复图需真实检查点路径/manifest/SHA，QualityGate/confirm不自动接受。
- 仅发布源码和通用模板，不打包私有实跑图、检查点、模型、媒体或本地交接；Registry可安装性独立。详见[1.88.1说明](docs/RELEASE_1.88.1.md)。

## 1.88.0

### 2026-10-02 Veda 与分离式采样（EXP）

- 追加独立阶段、交接、外置EAV／Prompt Relay及显式保存／恢复入口，旧一体节点／工作流和原采样配方保留。未来新增采样入口通过分离图准入检查，不能只增加一体调用而遗漏独立接线。
- Veda支持官方新版tile-score预测器及原生8步配方，Windows使用显式Flex动态编译；不把预测器当UNET，不承诺首次或普遍提速，Relay的Dense委托如实报告。
- 多人新增独立Source Save／Load与四份opt-in图；双人、三人原8步／新进程冷侧0采样完整124帧画音逐值一致，真实画布编辑保存／刷新重开审计通过。旧84份Face图与原审批默认不变。
- 当前完整CPU4416项及导演台429项通过（不叠加重叠范围），发行包按真实Core与线上基线另验；人工画质最后汇总，版本号不代表Registry已可安装。详见[1.88.0说明](docs/RELEASE_1.88.0.md)。不恢复QuantFunc／Nunchaku，不上传模型、媒体或私有交接。

## 1.87.0

### 2026-09-30 语义桥自动参数与显式多桥组合（EXP）

- 新增“自动模型参数”：按权重内容与固定应用契约配置，动漫战斗模型 alpha=1.0，精确识别的武术 v1 为0.12；原版／BUNNY旧默认0.10保留。未知自训权重仍走手动入口，固定契约错误在采样前报告。
- 新增“多桥组合”：最多8个桥按 first→second 有序串行，每桥独立参数；最终只接一次Apply或Relay／内循环插口。不是LoRA加法融合，顺序与内容身份进入缓存，原工作流不迁移。
- 修正训练器MLP残差底座为归一化输入，保留旧六张量数学及输入默认。软件回归、真实权重CPU/CUDA数值和原生配置／组合画布验证不等于新多桥视频质量验收。详见[1.87.0说明](docs/RELEASE_1.87.0.md)和[语义桥接线](docs/SEMANTIC_BRIDGE_EXP.md)。
- 本次不包含暂停中的Veda／分离采样开发、权重、媒体或私有交接。Registry上传、审核Active和实际可安装分别核实，不用新版本号绕过安全审核。


## 1.85.0

### 2026-09-21 导演台交互修正

- 修复启动环境 PATH 缺少 FFmpeg 时 SafeAVSave 在最终保存阶段失败：自动查找整合包与 imageio-ffmpeg 的现有程序，支持 T8_FFMPEG_PATH 指定路径；导演台提交前先检查编码器，失败窗口直接显示节点和原因。
- 紧凑预览完整等比显示图片／视频，不再裁切竖图；桌面设置栏加宽并分双列，页头、项目与工具按钮合并，减少占用剧本空间。

- 修复当前镜头生成误检整片、被其他首尾／录音草稿阻断的问题；当前镜头编译、D3 预检与导出使用相同范围。全部生成提交前检查全片，报错标注具体镜头。新建项目不再预置三个未完成示例镜头。

- 全局 D3 模式下可直接编辑右侧功能开关，共用镜头与新镜头同步；取消全局保留当前选择并改为本镜独立设置，修复旧本镜值覆盖显式全局的问题。
- 模型、多 LoRA 与分辨率移到顶部“模型设置”窗口，不再占据剧本区。
- 输入预览采用紧凑高度，宽屏下与素材卡并排，增加收起／展开预览；原图比例和生成尺寸不变。

Meridian 四节点与运镜／源时间编辑器、Avatar 渐进音频驱动、原生音色／情绪、一采动态预览及诊断功能已保存正式工作流。指定样片人审通过，原有采样和接缝配方不迁移。

### 2026-09-20 GitHub 源码更新

- 曜石导演台入口移入 ComfyUI 官方左侧栏，移除会遮挡画布工具栏的右下角悬浮按钮；节点内入口继续保留。
- 新增曜石导演台正式入口与干净启动工作流：统一图片／视频／音频素材、全片共享参考、首尾帧、Ref2VA、录音驱动、参考音色、新手／高级提示词和逐镜生成；高级面板可编译已验证的 Bridge、Relay、FastH3 V2 与低显存组合。
- 导演台新增全片 D3 默认配置与逐镜继承开关、按顺序生成全部镜头；模型、文本编码器、视频／音频 VAE 可从本机 H3 文件选择，LoRA 支持逐条叠加、独立强度、启用／移除与手动排序。
- 导演台新增 0.4／0.5／0.6／0.8 MP 总像素档位；后端按画幅计算并对齐 H3 要求的 32 像素网格，预检报告回显实际宽高与像素量。
- 新增 H16-3 `DeciiaChunkedPass2Sampler` 与正式 I2VA 4+4 模板；安全默认保留一采音频，`refined_exp` 才启用绝对时间轴、重叠交叉淡化和安静尾段保护。
- 修复新版 Topaz 官方 TensorRT 定义中 `[C]`／`[R]` 占位符解析，并保留旧版兼容；本机 `iris-3` 2×短片实际运行及原声保持已验收。
- 本次为同版本 GitHub 源码更新，不重复发布 Registry 1.85.0；安装 GitHub 最新源码后需完全重启 ComfyUI。

本次模型／文档整理：

- [TAEH3 模型](https://huggingface.co/t8star/Taeh3-Comfy)：时序及独立 2D tiny decoder，分别保留 MIT／Apache-2.0 许可。
- [Meridian 模型](https://huggingface.co/t8star/Meridian-Comfy)：合并 DMD 的原生 ConvRot INT8，附正确 Omega 1B512 原始 PT，目录／许可分别注明。
- 节点说明及模型参数提示新增下载入口；旧输入、默认值、采样逻辑和最终音频不变。
- [模型位置与源码准备](docs/MODEL_DOWNLOADS_1.85.md)，避免把权重文件夹误当完整程序或clone到非空目录。
- 中英文首页只保留安装、入口、模型与注意事项；历史内容独立保存，新增首页检查及 CI 防止再次堆积。

[完整 1.85.0 发布说明](docs/RELEASE_1.85.0.md)

## 历史版本

|版本|说明|
|---|---|
|1.84.0|[Semantic Bridge 与 H16 已完成部分](docs/RELEASE_1.84.0.md)|
|1.83.0|[FastH3 V2 与帧数合同](docs/RELEASE_1.83.0.md)|
|1.82.0|[版本说明](docs/RELEASE_1.82.0.md)|
|1.81.0|[版本说明](docs/RELEASE_1.81.0.md)|
|1.80.0|[版本说明](docs/RELEASE_1.80.0.md)|
|1.79.6|[版本说明](docs/RELEASE_1.79.6.md)|
|1.79.5|[版本说明](docs/RELEASE_1.79.5.md)|
|1.79.4|[版本说明](docs/RELEASE_1.79.4.md)|
|1.79.3|[版本说明](docs/RELEASE_1.79.3.md)|
|1.79.1|[版本说明](docs/RELEASE_1.79.1.md)|
|1.79.0|[版本说明](docs/RELEASE_1.79.0.md)|
|1.78.0|[版本说明](docs/RELEASE_1.78.0.md)|
|1.77.0|[版本说明](docs/RELEASE_1.77.0.md)|

旧首页逐条记录：[中文归档](docs/CHANGELOG_ARCHIVE_ZH.md) · [English archive](docs/CHANGELOG_ARCHIVE_EN.md)。
归档中的旧“未发布／待验收”状态仅供溯源，不覆盖当前版本说明。

详细功能、参数和安装合同：[中文](docs/README_DETAILS_ZH.md) · [English](docs/README_DETAILS_EN.md)。
