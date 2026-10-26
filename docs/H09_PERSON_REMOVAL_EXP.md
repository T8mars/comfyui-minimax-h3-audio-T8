# Person Remover：分离式实验接入

MASK 的 CPU 方形 grow 采用先横后纵 max-pool；其它设备保持原二维路径。
float 转换、逐帧、feather、绿色阈值与报告语义不变，空遮挡帧不会传播成有遮罩。
已用软／边缘／非连续／正负零等输入验证 MASK、绿色预览和报告逐字节一致。
局部 CPU 测量不代表整片或 GPU 提速，也不改变以下人审／素材边界。

当前交付：[v1.98.0](RELEASE_1.98.0.md)包含分离节点及已审前缀续接图；逐帧MASK目录加载需ComfyUI-VideoHelperSuite。四份原生前缀、原素材和模型须自行匹配，历史验收边界保持，不是从零六窗图。

当前为本地开发：分离节点、原件准备和 CPU 合同已完成。三份完整5秒失败片保留：前两份曾出现替代人物，后续定向片仍残留目标眼镜／发丝和漂浮手／杯。最新仅修正外部晚段 MASK 漏罩，准确复用前四窗原生检查点、只重采最后两窗，完整120帧／5秒已从真实保存画布跑完；Agent查看全部帧，未再见上述残留。2026-10-11集中反馈revision15的A4已接受这份完整案例；[准确前缀续接短名图](../examples/workflows/86-h09-reviewed/A4_Person_Prefix_Resume.json)已保存，不是从零六窗生成图，尚未发布。兼容加载器实核416张量／208对全部应用；不承诺人物一定移除、旁人／背景像素锁定或提速。旧节点、旧默认和旧工作流不变。

## 模型

感谢 [Akatz Labs](https://huggingface.co/akatz-ai/MiniMax-H3-Person-Remover-LoRA) 的 Person Remover V1、示例和训练说明，底层为 MiniMax H3。

将 `H3-Person-Remover-V1.safetensors` 放入标准 `models/loras/`。固定原件 revision 为 `bbc23bf8d6ef833aef43afc0ae740b7cf937d763`；155110584 字节，SHA256 `b01dc9fe888c2f4e455a8eb3b4878dfd4038ee0e2320a7b186c7935e65c93ba5`。原件已有 ComfyUI direct LoRA 键，不改名冒转换，不另造底模。实际映射／应用仍由 LoRA 加载器检查。

另需原生 Ref2VA pruned INT8 ConvRot、Qwen3VL 32B NVFP4 AWQ、H3 视频 FP16／音频 FP32 VAE。节点接外部对象，不内部重复加载。作者设置为 LoRA1.0、video/audio shift12/3、CFG1、`er_sde`／`simple` 每窗12步；不额外加载 Turbo，不强行改成双采。原作者 `comfy kitchen attention`／CLIP GPU 配置需要适配环境，T8 不自动安装或替换注意力。

准备工具 `tools/prepare_h09_person_assets.py` 必须显式指定模型目录和新的回执目录，固定下载原件并验证完整 SHA，不运行第三方安装器。权重遵守 MiniMax H3 Community License；代码、模型、素材的许可分别处理。模型不随插件分发。

## 接线

菜单：`T8/MiniMax H3/Repair/Experimental/Person Removal`。

1. 外部准备 24fps RGB 原视频、同几何的干净首帧、每帧目标 MASK。宽高必须32整除。可用手工或已有分割节点，不强制安装 SAM。
2. `Person External MASK + Green Preview`：可接原 MASK 与实际 Source Conform map，共用源帧选择；否则直接接同尺寸／帧数 MASK，报告来源未验证。输出处理后 MASK 和绿色 RGB 参考；先看预览。
3. `Person Window Plan`：接原视频、绿色参考和一张干净首帧，输出实际窗口计划与总成本。
4. 每窗接 `Person Select Window` → `Person Native Conditioning` → 外置 Guider／Sampler → 原生 AV 分离／双 VAE 解码 → `Person Append AV`。
5. 下一窗的 `previous_frames`／`previous_audio` 都接上一窗 Append 的累计输出。Conditioning 输出的 MODEL 接该窗 Guider；不要误接原 MODEL，否则历史坐标适配不会执行。
6. 最终保存 Append 的 RGB。默认静音；如需原音轨，显式从原视频接到最终合成。Append 的生成音频只作为后续窗上下文，不自动作为交付音轨。

分离节点允许查看并更换外部采样器，但改变作者设置是新实验，不继承作者配方／本机未完成的质量资格。当前没有独立 Studio 包办全部操作；不要把分离入口说成完整 reroll／跨进程恢复平台。

本地定向重抽准备工具为 `tools/prepare_h09_person_reroll_canvas.py`：从已核全文件SHA的原生AV检查点加载前缀，经相同公开VAE重新解码生成历史，只采指定后续窗；原始提示词、LoRA、采样配方不变，仅显式改变重抽种子。它不从有损交付MP4恢复历史、不偷偷回贴原片，也不会自动运行或接受结果。本机先完成W2单窗64帧诊断，再续接W3；最新完整图 `H09_CupFix2` 已实际保存、API导出、关闭后从保存树重开，60执行节点输入／接线严格一致，再从真实画布运行。该次仅W4／W5各12步、seed904244／904245，前四窗复用此前准确原生加载／解码的同进程缓存，不冒称最新运行冷加载。完整案例已获上述A4人工接受，不代表从零新生成整套六窗都已获相同质量资格。

前缀Load的报告也在画布显式展示。`tools/collect_h09_person_reroll.py` 只收集已成功的真实UI job：严格重建准备图、核前缀全文件／内容身份与实际加载报告、每窗全局RGB／PCM追加、完整媒体／原声AAC相关性及新native checkpoint，并保持视觉结论未证明。它不提交队列或运行VAE／DiT。当前6个completed画布job、19个完成的非缓存Sampler窗口（含一个用户中断job里已完成的W3，不是19次DiT调用），共4份完整5秒片，其中3份失败保留、1份另获A4人工接受。中断job不算成功job，短前缀不冒称完整片；窗口基于新W2／W3历史续建，未沿用旧失败后缀。

## 三种遮罩语义不能混淆

绿色 RGB 是模型的视觉引导，不是 latent 采样 mask。处理后 MASK 是显式预览／外部准备产物，节点不会将它偷偷附到 `noise_mask`，也不回贴原 RGB 假装人物被模型正确移除。

MASK 必须一帧一张，禁止单帧广播。默认 grow5／feather0；feather 是额外显式实验，不是作者默认。逐帧面积与空帧会报告；空帧不凭时间插值补人，不推测正确跟踪、遮挡或旁人身份。

外部人像分割不一定覆盖眼镜、散发、手中道具或影子。先逐帧查看绿色参考，把属于要删除人物的可见区域一并标进 MASK；保留人物的杯子／手与目标相触时不要直接套宽矩形。可显式联合多份外部 MASK，节点不会猜测道具归属或自动补漏。漏罩可能让移除后的杯子／手漂浮；这仍须检查完整生成片，不能只看绿色预览就认定修复成功。

## 窗口／历史／音频对应

|步骤|合同|
|---|---|
|生成|每窗22帧，前向计划每窗12步；120帧源需6窗，不是整片总12步。|
|开始帧|120帧源为0、21、42、63、84、98；最后一窗重叠8帧，只追加未交付部分。|
|短尾|不足22帧只补绿色引导尾帧，交付裁回源帧数。|
|续接|从已生成交付结果取边界前17帧＋边界1帧，共18帧；同一全局时间范围裁生成 PCM。|
|原生条件|Qwen 接干净／生成边界 Picture1 与绿色 Video1；遵循作者配方，不额外编码成 DiT 参考块。绿色22帧作为原生 Guide 编码。|
|VAE历史|通过公开 VAE manager 编码17历史帧＋5重复边界，保留前5 token；边界单帧另编码。|
|目标 latent|续接原生39帧的12视频 token／65音频 token缩至22帧的7／37；历史作为独立条件，不混入被采样目标。|
|布局|仅输出 MODEL clone 的 `extra_conds` 委托链内调整本任务标记的时间坐标；不修改共享 Core／PackedLayout。参考、空间、update／modality行不改。|
|音频|按全局帧端点四舍五入到 PCM 样本；不靠重复叠加局部窗时长，生成重叠只丢弃一次。|

模型重生成全帧；干净首帧质量、漏手／发丝／影子和镜头切换可能影响结果。画面移除人物不意味着删去原音轨中该人的声音。保留原声、生成音频、视觉移除分别审核。

## 当前证据与余项

新合同实际调用 Core Ref2VA、AddGuide、MiniMaxH3.extra_conds、PackedLayout 和 ModelPatcher；VAE／CLIP 使用明确的 tiny fake，不是预训练质量证明。包含空帧／外置同源映射、6窗全局 RGB／PCM、短尾、不同分支历史、真实 delegate／异常、不可变源和追加注册。

CPU 初轮109项中102通过、7失败：一个44.1kHz夹具尾样本问题，六个历史测试的 tools 导入路径问题。修复夹具与 runner 后仅重跑这7项，全部通过；**109是去重覆盖，不累加重复项**。受保护旧工作流／运行时／Core／Git index2154项前后无变化，CUDA未初始化。失败记录保留，不删测试或削弱全样本断言。

实际LoRA加载已核416张量／208对全部应用。最新真实UI job `ae11a3a7-9c0d-46e3-be5f-b60c0e845d9e` 在693.398秒完成全120帧864×480／24fps／5秒，六个native latent检查点、全局追加与显式原声AAC完整性已核。全帧概览和原尺寸末帧未见目标／替代人物、目标眼镜／发丝或漂浮手／杯，左原人物及杯仍可见；不是逐像素保真。最终MP4 SHA256 `8edda90fca419be9470053317db5ecea3ede4323d2a21f330b60e0e8a7e9e862`，只改本案例外部晚段 MASK 覆盖，F0–95原MASK字节未变，不回贴生成输出。旧失败证据全部保留。绑定该SHA的revision15／A4已人工通过；上述短名图仅将MASK目录、四个prefix路径及完整SHA留空供自己的准确资源配置，预览清私有路径，原参数与接线不改，资源适配版未重采。未取得portable Stage或普适移除资格，不再盲扫种子，技术完整性与Agent观察不能自动接受其他案例。

[H3 Relay](https://github.com/akatz-ai/h3-relay) 是独立 GPL 项目，作者示例依赖它；本接入使用独立 T8 实现和原生 Core 接口，不安装／替换 Relay，不复制其全局补丁或复用其缓存／成绩。原作者示例仅保留为未执行来源材料。
