# 连续 dense 坐标传递（EXP）

新增两个独立节点，不改变普通 Learned Latent Upscale、旧双采接缝、音轨、
sigma、采样步数或旧工作流默认值。现有学习型3D放大 checkpoint 可继续使用，
不是下载另一套 DiT 底模，也不是给所有双采自动开启的质量修复。

## 分离式接线

`Dense Transfer Provider` 选择 `models/latent_upscale_models` 中的学习型3D权重、
精度与释放策略。将 provider 接 `Dense Learned Upscale`，一采 AV latent 接其
`av_latent`。输出宽高继续接现有 HIGH Conditioning，再经既有 Reconcile、
HIGH采样和完整 Decode。EAV／Prompt Relay仍沿原外置入口使用。

默认测试放大倍率1.2，实际尺寸沿用T8已有32像素对齐与1.05各轴比例保护。
倍数不是尺寸承诺；某些很小的输入无法在该倍率满足对齐／比例限制，会明确报错。
不自动换算法、提高比例阈值或缩小原输入。

## 坐标合同

原生H3为每个2×2 latent patch分配面积归一化的空间坐标。新路径将相邻
dense cell中心放在该patch中心两侧，以统一坐标重采样**完整encoder特征**，
再执行decoder。不是分别插值四种奇偶相位，不对最终生成视频作位移修补。
encoder及decoder各自完整处理全部时间单元；只插值scratch分批，目标约64MiB，
单个超大frame本身超过预算时不保证这个上限。没有时间分块或额外DiT NFE。

输出保留输入音频tensor对象，视频noise mask使用相同坐标变换，音频mask对象
保留。仅接受已归一化的5D视频mask／空间broadcast；像素域4D mask须先走
现有原生准备链，节点不猜像素／latent时间对应关系。

provider保留普通 `api_version=1`／`upscale_clean_video` 半像素路径，另显式提供
`h3_patch_lattice_api=2`／`upscale_clean_video_h3_patch_lattice`。普通调用不会被
偷偷切到物理坐标。API2外部provider可显式委托，其内部路由／整段时间执行
标为未认证，不因是陌生provider而禁止普通用户组合。

最终接口依据 [Upscaler-Plus PR16](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus/pull/16)
的 `fc58fb80246bc58383929d21e78f54cf87d6507a`，以及
[Flow PR93](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/93)
的 `35b7fc706270918b2f5bf1b555ed0324dd268635`；不是原草稿版本。
本实现从本机Core原生patch位置推导，复用T8自有网络层，不复制或安装上游
Upscaler代码。上游默认same-grid续接会绕过此checkpoint，完整Flow还依赖其
自身sampler／VDN／Sol等；接口可连接不等于整套Flow已在本机验收。

## 当前验证边界

8项新CPU测试通过：实际Core patch中心、两轴两相位连续边缘、dtype／B/C/T／
非连续输入、真实tiny网络encoder→decoder路由、重复几何与原权重不变、普通
半像素路径保留、音频与mask对象、错误前置与异常释放。684个旧节点完整schema
为686注册的精确前缀，365保护文件不变。不是旧CPU套件的重复执行。

另将已审的最终Apache Flow坐标函数与本实现逐值对照（上下采样、两方向及
实际448×256→512×288网格），不执行完整foreign模块或无明确许可的Upscaler代码。
六组必要坐标对照实际最大绝对差均为0；这是FP32 CPU算子对照，不是CUDA或
训练checkpoint连续性证明。私有helper第一次因缺包父路径在import前失败，
修正其搜索路径后完成；未执行外部模块、未安装或重跑旧测试。
首轮FP32常量场bit-exact断言失败被保留：双线性FP32加权和有舍入，不能承诺
重采样后的原字节一致；原输入、no-op与保留音频的精确约束没有放宽。

真实学习型checkpoint已完成一个固定原生画布案例：另存、保存、关闭、侧栏重开后
只点击一次Run，LOW4＋HIGH4、448×256→512×288、1.2×，实际368.806秒。
120帧／24fps／5秒完整视频逐帧及PTS通过，32kHz立体声完整音轨有限且非零；
LOW与HIGH的两个Stage均用标准StageLoad读取验证。现有权重SHA为
`043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2`。
没有额外采样、没有重跑旧CPU套件，也没有捕获实际CUDA encoder中间feature tensor；
不能用完整输出审计冒充该中间张量的数值对照。

当前固定案例使用PyTorch attention，旧获审N01使用xformers；其LOW结果不逐值一致。
源图、条件、seed、noise、sigma及stage context虽相同，也不能把本片当作同一代码／
后端下只有放大器变化的画质提升A/B对照；本次只证明新路径实际运行和完整交付。

**人工质量审核仍待。** 当前不是推荐工作流，不能宣称训练质量、异尺寸续接获审、
接缝无损、提速或任意Flow版本适配。旧放大器继续可用，EAV／Relay外置入口保留；
本片EAV仅report_only，Relay单事件透传，不等于增强效果通过。
