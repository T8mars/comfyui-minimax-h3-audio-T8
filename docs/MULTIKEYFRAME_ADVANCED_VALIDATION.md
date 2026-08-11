# MiniMax H3 Multi-Keyframe Advanced 验证报告

验证日期：2026-08-11

插件版本：1.13.0 EXP

ComfyUI：0.31.0 / `cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44`

硬件：RTX 4060 Ti 16GiB，Windows，SageAttention，DynamicVRAM

## 结论

多位置关键帧已通过当前ComfyUI核心下的机械、兼容和有限质量验证，可以作为隔离的Advanced
实验节点使用。稳定Conditioning、稳定采样数学、既有54个节点ID/顺序/默认值与旧工作流未改。

当前不能把逐帧 `visual_noise_aug` 称为线性参考强度，也不能承诺16GiB安全、1080p安全、
任意关键帧数量、Turbo四步质量或与Long Video/Motion Context叠加。推荐首版全部使用0.999、
736×416、124帧、Stock20；改变raw值和加速档都按A/B实验处理。

## 原始需求完成度审计

| 原始要求 | 当前证据 | 判定 |
|---|---|---|
| 首尾帧之外加入更多中间帧 | 链式plan支持1～7张中间图；frame/seconds/percent解析、排序、重复/端点/越界拒绝；0/1/3/5矩阵和最大7张实机通过 | 已完成，EXP |
| 每帧可单独设置参考影响 | 首帧、每张中间帧、尾帧分别保存raw `visual_noise_aug`；非统一值同时进入对应latent混噪和packed timestep rows；单帧五档实机证明其他锚点未被全局覆盖 | 机械完成；线性“强度百分比”明确否决 |
| 不破坏原工作流 | 新节点字面以Advanced结尾并追加在旧54节点之后；稳定Conditioning/采样未改；空plan返回原MODEL；稳定采样SHA-256保持不变 | 已完成 |
| 如有底层影响则隔离节点 | 只克隆MODEL并局部patch，未知core/外部patch/Long Video叠加fail closed，无全局monkey patch | 已完成 |
| 首尾、中间帧与音频/参考媒体共存 | 真实Hybrid image/video/video soundtrack/standalone audio探针及layout/payload回归通过 | 已完成，当前core |
| 设计节点状态和验收标准 | 节点标Experimental；512MiB显存门、位置/灾难帧/raw单调性和硬否决项已写入roadmap与本报告 | 已完成 |
| 参考GitHub现有方案与Wan22FMLF | roadmap固定到ComfyUI PR #15439、H3 Multishot、Motion Context、Wan22FMLF commit并记录可借鉴/不可迁移边界 | 已完成 |
| 示例工作流 | API和前端工作流均存在，默认raw全部0.999；前端源与ComfyUI用户工作流副本哈希一致 | 已完成 |

这里的“完成”只指功能契约与当前声明范围。下文列出的跨GPU、1080p、正式盲评、完整raw单调性
矩阵和Turbo质量属于明确保留的后续验证，不会被当作当前稳定能力。

## 隔离与失败保护

- 新增节点只在用户接入Advanced计划时克隆MODEL并局部修补 `extra_conds`；没有进程级
  `PackedLayout` monkey patch。
- 空plan且首尾/普通参考均为0.999时调用稳定Conditioning并返回原MODEL。
- 中间位置保留真实frame index；Advanced positive误接原MODEL会明确报错，不会静默落到首帧。
- 重复位置、首尾端点、越界、非finite位置、未知core/外部补丁以及Long Video双向叠加均拒绝。
- Hybrid已实测保留image、video、video soundtrack、standalone audio与keyframes的layout顺序。
- 当前core以后若原生支持任意位置，本版本仍会先拒绝，直到位置、refs/audio与逐帧raw混噪契约
  全部重新验证；不通过版本字符串猜兼容性。

## 真实机械与显存矩阵

模型统一为FL2VA pruned INT8、Qwen3-VL NVFP4、video FP16 VAE、audio FP32 VAE，无LoRA。

| 检查 | 结果 | 最低整卡余量 |
|---|---|---:|
| 736×416、124帧、0/1/3/5中间帧，各3冷+3暖、1步 | 24/24生成和媒体契约通过 | 672.95MiB |
| 最大7中间帧（首尾合计9张）、736×416、124帧、1步 | 124帧/24fps；32k stereo、162816 decoded samples、finite | 1819.42MiB |
| 非统一raw值、256×256、22帧、1步 | 真实per-condition路径通过 | 202.82MiB |
| Hybrid图/视频/视频音轨/独立音频、256×256、22帧、1步 | 组合条件与媒体契约通过 | 575.96MiB |
| 非双时钟 `res_multistep + simple`、256×256、22帧、1步 | 通过 | 227.37MiB |
| Block Cache CPU、统一raw、256×256、22帧、4步 | 通过，2/4 cache hits | 213.55MiB |
| Block Cache CPU、单帧raw 0.995、256×256、22帧、4步 | 通过，2/4 cache hits | 212.32MiB |
| KJ Memory Efficient Sage Attention、256×256、22帧、1步 | 通过 | 227.37MiB |

0/1/3/5矩阵中，PyTorch pool峰值随中间帧数从2470、2632、2944增到3174MiB；各组暖运行
基线波动不超过0.32MiB，最终全局卸载后torch used约65MiB。整卡峰值受offload时序影响并不
单调，因此不能据此宣称增加锚点会节省显存。

`added_rows_vs_target_video_rows_percent` 只描述新增的DiT packed视觉条件行与目标视频行的比例。
它排除了CLIP图像处理、普通refs、VAE峰值、allocator/offload行为和attention非线性交互，绝不是
VRAM百分比。由于若干已通过案例仍低于项目512MiB门槛，16GiB `memory_safe` 标签继续否决。

## Stock20有限质量矩阵

质量矩阵为736×416、124帧、Stock20、3类素材（人物舞蹈、快速幻想运动、室内人物运动）×
3个seed；每条使用首帧、25%、50%、75%、尾帧，共9条、45个锚点。

- 9/9生成成功，9/9视频/音频契约通过，最低余量1184.85MiB。
- 自动全局位置代理42/45在目标±2帧内，命中率93.33%；三次偏差均是舞蹈素材首帧的全局最大
  落到frame 3/4，全部27个中间锚点均命中目标±2帧。
- 9/9锚点全局顺序正确；目标帧full-resolution SSIM中位数0.890919，最小0.493190。
- 黑/白坏帧为0；灾难性邻接跳变代理为0/45。代表接触表人工查看未见四步样本中的融化。

这些是自动相似度代理与有限人工查看，不是DINO/face身份门槛、动作一致性、口型、音频主观质量
或正式盲评。它支持“当前Stock20路径能命中中间位置”的有限结论，不支持泛化到其他素材、分辨率、
时长、LoRA或GPU。

## 逐帧raw参数的科学边界

固定舞蹈素材、seed、Stock20和其余四个锚点，只改变frame 62的raw值
`0.999/0.995/0.990/0.980/0.950`：五次均完成，未改变的四个锚点SSIM范围仅
0.00044/0.00286/0.00397/0.00127，说明选定值没有被退化为一个全局值。

目标锚点SSIM分别为0.88598/0.88041/0.88831/0.87660/0.87595；Spearman `rho=0.70`、
`p=0.188`，没有达到预设 `rho>=0.8`，也不是严格单调。因此：

- UI和文档只称 `raw visual_noise_aug EXP`；
- 0.999是示例默认值；
- 0.950及以下继续视为激进实验；
- 不宣传“每帧参考强度百分比”或某个低值必然更接近/更清晰。

## 采样与加速边界

当前原生非双时钟、Block Cache和KJ Sage兼容机械探针通过。使用新的FL2V Turbo四步LoRA时，
空Advanced控制与三中间帧都能执行，但本次四步输出出现明显融化/涂抹，未通过质量门。旧EMA
Turbo LoRA在同一pruned模型的空Advanced控制上就发生shape错误，因此不是Advanced节点特有问题。
现阶段Advanced质量推荐Stock20，不推荐把任何四步LoRA标为已验证质量档。

## 仍未完成或明确否决

- 1080p、0.6M/362帧、更多分辨率/帧数组合与跨GPU三冷三暖；
- DINO/face身份指标、正式多人盲评、动作/口型和音频主观质量；
- 3素材×3seed×3位置的完整raw单调性矩阵；
- Long Video、第三方Motion Context或未知全局H3补丁叠加；
- Turbo四步稳定质量档；
- 16GiB `memory_safe`、never-OOM或“任意数量关键帧”声明。

本报告中的实机原始JSON、接触表和日志保存在本地忽略目录
`artifacts/multikeyframe-advanced-validation`，不随Git发布大体积生成物。
