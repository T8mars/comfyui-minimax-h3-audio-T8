# RADAR r6 增量开发

v1.93.0交付N01–N05已完成实现、指定代表人工验收及原生工作流保存。旧工作流、原656个ID及默认采样方式不迁移；N05追加一个薄封装后处理保存节点，N01追加六个可组合参考包节点（含独立外置Relay适配），当前663个ID。独立真实Core对比确认旧656全部schema为精确有序前缀。QuantFunc／Nunchaku不恢复。

人工反馈分别通过N01指定5秒片、N05指定裁切对照、N03配方卡UI、N04导演台两规则UI，以及独立新N02 LightVAE同HIGH解码对照。LightVAE原许可由用户明确确认；独立新反馈精确绑定新片，不借旧四项批准它。十份实际原生保存图已原字节集中保存到默认用户工作流新目录，保留原参考producer和模板边界；不宣称其它素材／组合都获资格。

公开的10张去私有资产图在[75-radar-r6-reference](../examples/workflows/75-radar-r6-reference/README.md)；只改明确资产／缓存选择项、保存确认和非执行说明，数值与接线不变。使用自己的合法输入，不能把公开占位当本次已完成缓存。详见[发行范围](RELEASE_1.93.0.md)。

## P0 与推进顺序

先检查当前源、真实环境和各来源的结构／许可，再依次推进：配方差异卡（N03）、精确画幅与原片救援（N05）、LightVAE解码资格（N02）、可复用参考包（N01）、连续状态与画外声音模板（N04）。不是重跑全部既有路线，也不把条件研究算成承诺开发任务。

`tools/radar_r6_preflight.py` 只读取Git状态、保护文件SHA、旧示例字节集合、包版本及Core静态能力。只在新的本地证据目录写报告，不加载模型、不初始化CUDA、不修改环境或Git。静态支持不等于权重／GPU／成片资格。

## N03 配方卡与显式差异（指定UI已人审）

普通画布的模型／VAE／LoRA加载器及两种Bridge配置节点，右键有 **T8模型配方卡（证据／差异）**。选一个当前资源读取，不自动寻找相似文件；从Core配置的模型目录读取，不提供任意路径接口。源码需在独立验收服务或正常重启后加载，不强行重启用户实例。HyperVAE若有绝对路径覆盖，不用下拉项冒充实际权重；这条卡保持unknown，不修改原加载器。

- Header卡显示实际存储dtype、低秩矩阵配对rank、剪枝basis／量化命名空间，以及有来源标签的任务／alpha声明。缺证据显示unknown，不能按文件名推断Full、FP8、任务或像素倍率。
- 标量alpha的真实tensor值不在只读header卡中；metadata alpha仅是声明，不冒充读取tensor。训练alpha／rank、LoRA运行strength与Bridge alpha互不替代。
- Bridge复用已有按内容识别的推荐参数与固定合同，不新建预设。未知用户权重／补丁继续由原加载器执行，卡片不建立准入黑名单。
- VAE倍率仅在完整已知投影形状、24通道统计与packed元数据合同齐备时展示；是Core／T8既有header合同，不是解码成功或画质证明。sigma／clock有明确metadata时也仅标作者声明，不拿它覆盖现有采样表。
- **展开“导入／替换配方”**：编辑已有字段的声明式JSON，点击预览会列出字段旧值→新值；点击“我确认”才应用。不会保存或排队，不改连线／其它节点。可用ComfyUI撤销，保存和运行仍由用户决定。普通模型下拉操作保持原样，不拦截旧工作流。
- 修改暂限静态Core UNET／CLIP／VAE／LoRA加载器及Bridge配置。类型、枚举和范围按节点原schema核对；未知／重复／连接输入不迁移，未知第三方节点仍能原样运行。预览后画布／节点／JSON发生变化需重新预览；不执行配方里的脚本或回调。
- 手动Bridge的内容绑定推荐值只能先填入编辑框，仍须预览确认；自动Bridge节点已内部使用该profile，无第二套自动参数。
- 已实际画布验证VAE精确差异→明确确认→原生撤销／重做→另存新图→关闭后从工作流栏重开，指定UI已获用户通过。656旧节点完整finalized schema与公开main独立CPU捕获完全一致。旧API导出timeout未计成功，不借局部验证为全部第三方或画质资格。
- 手动Bridge同样已实际确认推荐差异、原生另存和重开。未知第三方加载器只显示只读卡；扩展的独立CPU前端检查保留原菜单回调、序列化、widget顺序和接线。这项使用实际扩展代码与模拟DOM，不冒充全部第三方浏览器或GPU兼容矩阵。
- 自有验收服务的只读端点已实际核对：配置目录内明确VAE读取成功、只读99680字节header，不加载tensor值；非资源目录类别、任意绝对路径、未安装名称均未成功读取。前后Queue为空，卡片不保存、不排队。请求大小边界另有CPU测试；端点拒绝错误输入不等于限制用户原加载器。

示例（在VAE节点本身打开编辑框；文件名必须是其当前可选列表里的明确值）：

```json
{"schema":"t8.widget-recipe.v1","node_type":"VAELoader","settings":{"vae_name":"minimax_h3_video_vae_int8_convrot.safetensors"}}
```

这不是整张工作流导入器：不会猜多个候选的替代品，也不会把训练alpha当运行strength。

## N05 精确画幅与原片救援（指定裁切对照已人审）

见[裁切与持久master合同](RADAR_R6_DELIVERY.md)。三份短名Full／Cold裁切／Cold留边模板与显式保存出口已整合；一个真实5秒冷裁切画布、独立新进程音频包／PCM／PTS绑定及持久失败／取消状态已技术验证，指定原片／裁切对照已获用户通过。沿用已有publisher，不改采样或把CPU测试信号当模型片；Full和Contain不借冷裁切代表声称任意组合质量通过。

## N01 参考包与全新条件（本地交付，六个EXP节点）

见[参考包格式与接入边界](RADAR_R6_REFERENCES.md)。已实现有界safetensors读写、显式新文件保存、配置目录精确选择、原生producer编码工厂、有序画内／画外声音路由和全新Qwen Apply；不修改已有embedding。真实素材编码／保存与fresh32B Qwen、LOW4＋HIGH4外置Relay已在原生画布完成，既有learned1.2实际输出512×288。当前Core CPU INT8视频VAE终末解码报错，复用完成HIGH到另一正常GPU解码画布，得到120帧／24fps／5秒原生音画，0额外采样；该指定片已获用户通过，不是原Full整图success。新Full／实际LOW绑定的HIGH-only Cold采样画布均保存重开并过Core validator，解码单独可复用；Cold未新增GPU跑，包仍要求匹配producer配置。最新19个完整受影响文件366项CPU通过不能代替其它素材质量；latent-only Apply尚未资格化，社区格式4／5互操作未认证。

## N04 连续状态与画外声音（指定UI已人审）

“镜头证据与规则”增加[两个声明式小配方](RADAR_R6_CONTINUITY.md)：角色道具连续账本、画内／画外对话。仅填写空规则表单；仍走已有Skill快照、候选编译／显式采用和保存，不造第二资产库，不自动推断事实或改变声音接线／事件时钟。完整受影响六文件58项CPU测试通过；真实独立QA页面已完成填表、拒覆草稿、添加两快照、编译、明确采用、回退、服务端保存及新标签页重开，指定UI已获用户通过。真实新入口曾遇到Core的/api前缀导入问题，仅修三处动态模块地址判断，旧入口仍保留；界面人审不等于模型身份／画质／声音资格。

## N02 LightVAE（指定实际解码对照已人审）

原作者[LightVAE](https://huggingface.co/stdstu123/LynnReal-Onmi-light-vae)的许可由用户自行理解并确认适用后使用。已验证[Kijai转换件](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/d8023be02fefbb3633b0cd335c3879f91177299d/minimax_h3_lynnreal_light_vae_int8_convrot.safetensors)：固定revision `d8023be02fefbb3633b0cd335c3879f91177299d`，文件2,137,493,384字节，SHA256 `f11f8b9b96c9dd22b36bc6c18464b6929c207d247e6d865d90fe31588e58e7e2`。公开非门控不免除原许可。

把该文件放入Core配置的 `models/vae/`（可用子目录），在**普通VAELoader**选择它，连接既有T8 AVDecode的视频VAE插口；音频继续接原音频VAE。无需新节点或升级Core。实际Core加载得到26层decoder／24通道，缺失和多余权重均为空。不是HyperVAE 2×，不要因此改变分辨率或放大器。

唯一新增的原生画布只解码已有完整HIGH：512×288／120帧／24fps／5秒，6节点／7接线／10执行参数与原生保存文件一致，0新增NFE。采用正常GPU-VAE配置，没有沿用已失败的INT8 CPU注意力分支；原失败证据保留。两边完整解码后的PCM及视频PTS完全一致；没有重跑采样、TTS／静音／剪对白或替换音轨。本轮任务约6.3秒，不是受控提速基准。

最终单项审核的独立真人导出明确“通过”，与两原始片及本次真实保存重开的解码图精确对应。**仅该指定decode对照已获画质通过，不扩展为encoder、所有素材、声纹或通用速度资格。** 实际原生图原字节保存，旧未评快照和失败证据不覆写；反馈前的JSON诊断标志不是最新人审状态。

## 条件研究边界

[Core PR16750](https://github.com/Comfy-Org/ComfyUI/pull/16750) 的guide／reference音频拼行变更先用真实Core微型CPU sentinel核对。手工丢失guide的payload不是T8当前生成链故障证明；没有本机反例不升级／回补Core。

[OrbitQuant模型](https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4) 的0.11融合布局和[Comfy pack](https://github.com/iamwavecut/ComfyUI-OrbitQuant) 的版本要求分别核对。未取得一致Windows栈和真实内核调用证据前，不安装到共享环境、不下载整套大模型或宣称T8 MODEL／双采支持。

LightVAE已按上方N02完成许可确认、固定权重、同一完成latent的decode及指定对照人审。不代其它用户接受原许可，不用作者速度或本次单次耗时宣称通用提速。

[RefMod PR59](https://github.com/pyros-projects/orrery/pull/59) 是互操作来源，不捆绑其角色素材、不动态导入其插件。先自建合法素材的安全Load／Save、有序角色路由和Apply；producer／归一化／Qwen grounding／媒体ordinal等合同另行验证。

人工审核最后集中提供绑定实际媒体SHA的可打开表，不自动接受、不用ASR／TTS／静音替代原生对白判断。
