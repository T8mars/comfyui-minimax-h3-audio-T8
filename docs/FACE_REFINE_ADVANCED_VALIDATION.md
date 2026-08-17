# MiniMax H3 Face Refine Advanced 验证记录

## 定位

这是一条隔离的实验性“局部二次生成”链，不是确定性超分或人脸复原。它把远景脸所在的
时序 crop 放大到 H3 画布，以低去噪第二遍生成补充结构，再只把脸区回贴到原视频。
任何输出都只是候选，不能自动覆盖原片或宣传为身份保持、通用修复、16GB安全。

## 1.26.0 作者机制对齐修正

固定上游`Carasibana/ComfyUI-H3-FaceRefine@79a97ce5`与T8逐帧对照发现，旧Parity仍有四项会改变
结果的差异：RGB/BGR检测输入、edge-clamp时的脸罩位置、逐帧denoise所用脸尺寸，以及colour match
执行坐标；此外旧T8在人为复制第90张crop后才编码，而作者直接编码Comfy返回的89帧。

1.26.0只修正隔离Parity模块：local Ultralytics改用BGR，mask使用作者居中的平滑脸框，denoise使用
`crop height / crop_factor`，feather按chunk中点倍率换算，colour match在warp回来源坐标后执行，
video VAE直接编码89个有效帧并要求latent shape自然吻合。固定机械探针中crop平均绝对差从
0.020292降到0.00000677，denoise曲线最大差0.00001423，完整stitch平均/最大差为
0.00000117/0.00053048。

真实prompt`1ed411fa-9b91-45c4-801d-7f45b3597fe5`在112.48秒完成，输出SHA-256
`0DD8C79F95B01647F3BF345B6503C83A5860BE99BA66D8D72114BD274E9A0884`；严格媒体仍为89帧
320×320@24fps和32kHz双声道，PCM MD5与来源/作者目标一致。相对作者目标全片SSIM由旧T8的
0.955273提高到0.967059；整卡粗峰值15,823/16,380MiB，约余557MiB。用户完整观看作者目标/T8 v2
双栏后确认“两边效果一样好”，因此该固定素材、seed和模型链的主观至少同等门通过。这个单例结论
不证明跨素材修复、身份保持或通用16GB安全。

## 1.25.0 人工验收 MANUAL512 REL 基线

推荐Parity示例已固定为本机完整视频人工审片选中的配置：`manual_512`、`crop_factor=2.5`、
`relative_to_clip`、逐帧强度`0.8/0.35`、21/51平滑，以及face-only、dilation24、feather24、
colour match1回贴。新增
`MiniMaxH3FaceRefineManual512RelativeBaselineT8Advanced`只做机械合同校验和IMAGE直通：它要求
Plan/latent注入/去噪/Stitch报告hash一致、crop内最小脸高至少200px、音频mask全零、无fallback且mask外逐位不变。
任何不匹配都直接报错，不会静默退回auto画布、absolute模式或旧Quality Gate。

固定89帧fixture的来源脸高为105～195px，512画布中的脸约205～312px，crop倍率1.60～1.95x；
用户在六路完整视频中选择MANUAL512 REL为最好。该结论只覆盖这条fixture和seed42；节点报告继续明确
`quality_guaranteed=false`、`identity_verified=false`和`universal_16gb_safe=false`。本次16GB运行最低
抽样余量161MiB，低于512MiB安全门。

该人工选择实际使用Ref2VA pruned INT8、Qwen3-VL NVFP4、官方双VAE、alpha8 T8转换的FL2V Turbo
LoRA@0.75、两张身份参考、来源音频锁定、face YOLO及`er_sde/simple/4步/denoise0.45/seed42`。Comfy
实际解码为89帧：Parity latent节点显式复制最后一张crop一次，形成合法90帧H3内部输入；Stitch仅丢弃
这1张内部对齐尾帧，最终回到原始89帧。复制与丢弃数都写入报告并由基线Guard绑定，不允许隐式VAE
时间拟合或裁掉任意有效源帧。旧Plan默认`require_h3_grid=true`不变，推荐示例才显式启用这一帧例外。

实现为四个只追加的节点：

1. `MiniMaxH3FaceRefinePlanT8Advanced`
2. `MiniMaxH3FaceRefineConditioningT8Advanced`
3. `MiniMaxH3FaceRefineSamplerT8Advanced`
4. `MiniMaxH3FaceRefineStitchAuditT8Advanced`

稳定`sampling.py`、既有90个Node ID及旧工作流均不修改。

## P0 已实现机械合同

- 输入必须是精确24fps，tensor路线最多362帧；默认仍要求`17n+5`。隔离Parity路线只额外允许
  “距离合法网格恰好1帧”的显式尾部对齐，并要求latent报告与Stitch报告证明复制/丢弃的都是这一张
  内部尾帧；不隐式重采样、裁掉源帧或容忍任意时间轴拟合。
- 使用低分辨率RGB帧差发现硬切，每个镜头单独重置跟踪和平滑；多镜头默认拒绝一次H3采样。
- 默认真人检测使用`models/face_detection/face_detection_yunet_2023mar.onnx`：固定OpenCV Zoo
  commit、MIT许可与SHA-256，只走CPU、不联网、不跨执行保留检测器对象。纯动漫可显式选择
  `local_anime_onnx_exp`，它只接受固定deepghs v1.4 nano YOLOv8 ONNX输出合同并用CPU Runtime；
  手工ROI仍为零依赖回退，本地Ultralytics仍接受用户自行授权的兼容模型。
- 检测器对象会在`finally`销毁并执行GC，但OpenCV/ONNX Runtime的进程全局CPU分配器可能保留
  热身页面；报告明确`process_global_allocator_release_guaranteed=false`，不能把对象释放写成
  进程RSS必然回到第一次运行前。
- 计划记录真实源脸框、边缘钳制后的crop框、crop内真实脸框、shot/state/贴回权重、源代理哈希
  和plan SHA-256；修改或错接输入会失败关闭。
- 视频VAE输出必须与目标H3视频latent在B/C/T/H/W上完全一致；不允许上游式静默trim/pad。
- `require_locked`要求音频noise mask全零；注入后复用同一音频tensor和同一个noise-mask对象。
- 低去噪节点复用稳定双时钟设置，只在新模块中按Comfy BasicScheduler同类语义截取sigma尾段；
  `denoise`未标定，不能理解为线性修复强度。
- 回贴使用真实浮点crop几何，默认CPU、ellipse、颜色均值/方差限幅；变化过大、lost状态及相邻
  帧回退原片。mask外像素在节点返回前做逐位一致校验。
- 最终音频不由Stitch处理；示例明确复用原Conditioning的`mux_audio`，丢弃第二遍H3音频。

## 当前证据

- 合成边缘脸ROI证明crop内脸框不是错误居中假设。
- 合成硬切证明shot边界被发现并触发单次采样默认阻断。
- 5/22帧H3时间合同、latent严格匹配、锁定音频tensor和mask对象复用均有单元测试。
- 回贴遮罩外逐位一致、过大变化整帧回退、陈旧源指纹拒绝均有单元测试。
- API与ComfyUI 0.4前端工作流均有节点/连线结构测试。
- RTX 4060 Ti完成5帧、384×384 crop的小型CUDA探针：GPU裁切、严格视频latent注入、同一音频
  tensor复用、GPU回贴和mask外逐位一致均通过。该探针不加载H3权重，不是16GB峰值或画质测试。
- ComfyUI`0.33.0@7fe8a61385`真实`/object_info`确认四节点类型，`/userdata`确认用户工作流可见。
- 当前ComfyUI完成一次真实FL2VA pruned INT8 + Qwen3-VL NVFP4 + 双H3 VAE端到端链：
  736×416、124帧、手工ROI、12步低去噪双时钟，12/12采样完成、无OOM、总耗时176.93秒；
  输出严格解码为124帧24fps H.264和32kHz双声道AAC。来源/输出解码音频均为164,864个双声道
  采样，相关系数0.997751、SNR 23.45dB；这是原`mux_audio`经过AAC重新编码的保留证据，不是
  PCM或压缩包逐位一致。测试来源来自已知被非等比横向拉宽的旧盲测素材，因此只计机械端到端
  证据，不计画质、身份或画幅正确性证据。
- 本机现有`models/SVFR/yoloface_v5m.pt`真实加载探针确认它是TorchScript/YOLOv5-face类文件，
  不能通过当前Ultralytics `YOLO.predict`运行；节点会明确拒绝并建议使用手工ROI或另装本地兼容
  检测器，不会自动下载、转换或把失败权重伪装为已验证检测路线。
- 默认YuNet模型精确SHA-256为
  `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`；本机真实图片探针命中
  正脸、清晰侧脸、嘴部遮挡及脸宽约占画面8%的两张全身远景图，置信度约0.918～0.941；纯动漫图
  0检测，证明必须隔离动漫域而不是假装一个模型通吃。
- 动漫模型精确SHA-256为
  `fd860b650a4377046842c3cd80d01b0b408bdfbdb4acee5759630f82c6ef04a9`；同一纯动漫样本通过
  `[1,5,8400]`YOLOv8原始输出解码得到1张脸框，置信度0.868。它没有landmark或身份能力。
- 自动YuNet在124帧真实来源上逐帧原始检测98帧；改良轨迹后Plan记录96个`detected`、2个
  `reacquired_unverified`和26个`lost`，lost只用于邻近crop时序上下文而不自动回贴，避免把错框
  当成确定脸。该素材本身已被上一轮非等比拉宽，因此不计precision/质量证据。
- 同一自动Plan接入完整FL2VA pruned INT8 + Qwen3-VL NVFP4 + 双VAE链：736×416×124、12步
  低去噪双时钟在171.855秒完成，整卡峰值15,814MiB/16,380MiB，约余566MiB；输出严格解码为
  124帧24fps及32kHz双声道音频，SHA-256
  `56df397c0789694ef9919da2593297785d8b0a2ca4e70439261154809b0526ca`。这关闭“自动检测不能接
  H3”的机械缺口，不关闭画质和通用16GB门。
- 六次124帧YuNet重复均在相同98帧检出。前3次OpenCV全局CPU分配器热身使post-run RSS累计约
  761MiB；最后3次增量为0.156、0.004、0.0MiB，说明暖态无继续阶梯，但不能声称RSS完全回落。
  六次动漫单图ONNX执行首末post-run差78.867MiB，最大暖步65.512MiB，未超过256MiB诊断线，
  仍只证明本机对象不持久缓存。
- 早期实现门为505项回归；完成扩展验证与盲评揭盲工具后，当前门更新为532项全回归、全仓Ruff、
  compileall、项目JSON解析和`git diff --check`。稳定`sampling.py`SHA-256仍为
  `111da5e52b28f2424f57b36f88db63e3ea02b538a8cdfdea1c8ad2f122ad7bb5`。

## 2026-08-16扩展验证

### 真实H3压力与重复运行

- 画幅安全的真人舞蹈来源使用736×416、24fps，不做非等比拉伸。124帧、512脸部画布、12步
  低去噪完整链完成3次冷启动和同进程3次暖运行，6/6成功。冷态最低整卡余量717.6MiB，暖态
  最低922.1MiB；冷态三次执行后private spread为3.2MiB，暖态为79.1MiB，均低于256MiB阶梯
  诊断线；六次执行后整卡占用均回到约1,648.7MiB。输出均为精确124帧、24fps、736×416和
  32kHz双声道。压缩后音频样本数与来源一致，相关系数均为0.993189；这不是PCM逐位声明。
- 同一画幅安全来源扩展为362帧、384脸部画布、12步，单次冷启动完整链在295.235秒完成，
  整卡峰值15,203.3MiB、按总显存减占用计算余量1,176.2MiB，进程private峰值41,311.6MiB；
  输出精确362帧、24fps、736×416及32kHz双声道。只完成了一次362帧，不把它写成三冷三暖、
  其他素材/分辨率或通用16GiB安全层级。
- Windows驱动的`nvidia-smi free`字段可能因系统保留显存而低于`total-used`余量；上述门统一沿用
  项目验证器的整卡`total-used`口径。结果证明受测配置可执行且124帧无暖态阶梯，不证明并发CUDA
  用户、其他GPU或所有插件组合安全。

### 公开标注集检测评估

固定YuNet 2023mar在WIDER FACE验证集3,226张、39,123个有效人脸上完成集成评估。验证图像与
标注压缩包SHA-256分别为
`f9efbd09f28c5d2d884be8c0eaef3967158c866a593fc36ab0413e4b2a58a17a`和
`c7561e4f5e7a118c249e0a5c5c902b0de90bbf120d7da9fa28d99041f68a8a5c`。数据只在本地按
CC-BY-NC-ND-4.0用于非商业验证，不进入插件仓库。以下是IoU≥0.5的固定阈值集成指标，不是官方
WIDER easy/medium/hard AP：

| YuNet阈值 | Precision | Recall | F1 | `<16px` Recall | `<2%短边` Recall |
|---:|---:|---:|---:|---:|---:|
| 0.35（节点默认） | 62.23% | 64.99% | 63.58% | 43.60% | 41.83% |
| 0.60（保守对照） | 86.10% | 56.94% | 68.55% | 32.25% | 30.40% |

0.60明显减少误检，但损失了节点最关心的远景微小脸召回，因此不能仅凭总体F1把默认值强改为
0.60。0.35也只有62.23%固定阈值precision，不能自动接受修复。可复现工具为
`tools/evaluate_face_refine_yunet_wider.py`；它不下载数据或模型。

### 多人轨迹与硬负例

- 一条真实竖屏群像被画幅安全转换为416×736、124帧、24fps；113/124帧有多张检测脸，Plan发现
  一次硬切并逐镜头重置，124帧全部有选中轨迹。8帧抽样中，切前跟随黄巾人物，切后持续跟随
  同一灰衣人物，没有看到抽样换脸。该素材没有受控交叉遮挡，跟踪器也没有外观/身份模型，故不能
  把它写成“多人身份安全”。
- 另一条362帧台球素材有149个multi-face帧，抽样中自动轨迹曾选择墙灯和台球上的卡通人物，
  形成明确高置信硬负例。提高到0.60虽可减少部分误检，但公开集结果证明它会显著漏掉微小脸。
- 另做5组、每组61帧的确定性双人交叉矩阵。稳定A优先、交叉后置信顺序翻转、交叉附近检测顺序
  交替和垂直分离四组没有换人；但目标A在交叉点遮挡3帧时，跟踪从第29帧切到B，最终31帧跟随
  错误身份B，仅1帧因两框完全重叠而不可判。该矩阵不依赖生物识别数据，只证明当前几何/速度
  跟踪器在短遮挡下可以持续换人；复现工具为`tools/audit_face_refine_tracker_crossing.py`。
- 结论是：`multi_face_frames>0`、大量lost/reacquired或异常尺度跳变都必须先人工检查Plan preview；
  当前默认不做身份验证、不能自动接受，也不能宣传跨人不串脸。

### 盲评状态

`tools/build_face_refine_blind_review.py`已生成6组随机A/B本地评审包，逐组比较保留原片与一个真实
二次生成候选，要求评分身份、表情/嘴型、时序、接缝、自然度和动作保留。该设计测的是实际创作
效用，不是同NFE算法优越性；同一来源重复出现也可能让评审者逐渐猜到对照。

首份人工评审已完整填写并在提交后通过`tools/analyze_face_refine_blind_review.py`严格校验、揭盲：

| 项目 | 原片 | 候选 | 平局 |
|---|---:|---:|---:|
| 总体偏好 | 6 | 0 | 0 |
| 身份偏好 | 6 | 0 | 0 |
| 动作偏好 | 0 | 0 | 6 |

身份、表情/嘴型、时序稳定、接缝和自然度的原片/候选均分都是5.0/1.0；动作保留都是5.0/5.0。
六条备注均指出候选出现鬼脸及脸部持续来回跳动。因此这不是“改善不明显”：当前固定来源、参数和
六个候选已经被该评审明确否决，不能自动接受或晋级稳定版。与此同时，这仍只有1名评审者、1个
重复来源和6次候选运行；预注册的至少5名独立评审者门没有达到，也可能存在评审逐渐识别重复对照
的偏差，故不能写成跨素材/跨评审的通用算法结论。原始导出和私有盲码只保存在忽略的本地证据目录，
汇总文件为`blind/blind_review_analysis.json`。

作为辅助而非替代，`tools/summarize_face_refine_candidates.py`对6个真实候选计算来源相似和时序
代理：全画面灰度SSIM均值为0.97368～0.97456，但扩展脸区SSIM均值只有0.47914～0.53921；候选/
来源脸区Laplacian方差中位比为0.54925～0.68355，脸区动作差分相关均值为0.35807～0.40712。
结果没有给出脸部细节或动作非劣信号，更不能证明身份；Laplacian也可能奖励噪声/振铃。它只强化
“不得自动接受”的结论，最终自然度、身份和偏好仍必须由盲评决定。

## 2026-08-17 固定上游代码与Parity质量门

- 已直接运行固定commit上游的4个原始节点。Face YOLO在124帧中直接命中116帧、插值8帧；
  `auto_capped_768`按约247px最大crop得到256×256真实画布，所以256不是T8强制压缩。
- 相同本机模型栈下，上游0.8/0.35的脸区SSIM/清晰度中位比/动作相关为
  0.49077/0.74321/0.42102；只改成0.45/0.15后为0.77776/0.56297/0.70423。两者仍有错误脸或
  清晰度回退，均未晋级。T8低强度对应值为0.76044/0.50535/0.69200，说明机械差距已经较小，
  继续照抄上游参数不能自动解决当前素材。
- 新增`MiniMaxH3FaceRefineQualityGateT8Advanced`，接在Parity Stitch之后。它要求候选在原tensor的
  change mask之外逐位不变，只保留结构、脸区变化、Laplacian清晰度和残差时序代理全部通过且至少
  连续3帧的区段；拒绝帧回到精确源tensor，边缘只在已接受区段内淡入淡出。
- 默认高强度真实候选的accepted-mask视频124帧全黑，即0帧被接受。质量门输出相对另一次源片直通
  编码的全画面/脸区SSIM为0.999885/0.996810、脸区MAE 0.000471；单元测试证明0接受时tensor逐位
  等于源片。剩余视频差异来自两次独立H.264编码。
- 这只证明质量门能阻止当前鬼脸进入最终输出。它不是人脸识别器，也没有真实生成帧通过增益门；
  `quality_validated=false`、`identity_verified=false`和`automatic_accept=false`必须保持。

## 尚未授予的声明

以下真实证据在完成前，功能必须保持Advanced EXP：

- 真人固定阈值precision/recall已有3,226张WIDER验证集证据，受控交叉矩阵也已证明短遮挡可换人；
  仍缺真实带身份标注的多人交叉/遮挡集，动漫也仍只有隔离后端和小样本机械探针。
- 当前六个H3第二遍候选已在单评审中明确失败；若以后改变算法/参数，需要使用新素材、新盲码和至少
  5名独立评审者重新验证清晰度、自然度、身份、表情、嘴型、时序稳定和接缝，不能复用本轮结论晋级。
- 真实多镜头拆窗、Selective Repair接受/回滚和Long Video文件级恢复。
- 124帧完整H3三冷三暖、连续三任务及一次362帧峰值已经关闭本机固定路线的机械缺口；362帧
  三冷三暖、多素材/分辨率、并发CUDA用户、其他GPU与其他插件组合仍未验证，因此不写通用
  `memory_safe`。
- 可选身份模型的许可、genuine/impostor门槛与至少5人盲评；当前实现不做身份验证。

## 发布否决门槛

- 错人、串脸、后脑勺生成正脸、cut前轨迹污染cut后镜头。
- 中心漂移超过源画面对角线2%，尺度漂移超过5%，或嘴型/A/V同步恶化超过1帧。
- mask外任一像素变化，音频PCM/样本数变化，或源/plan/hash不一致仍继续。
- 任一隐式裁帧、补帧、空间latent适配或静默降低canvas。
- 真实推荐档OOM、最低余量低于512MiB或连续任务出现超过256MiB阶梯增长。
- 盲评偏好95%置信区间下界未超过50%时宣传“稳定修复”。

## 示例

- API：`tests/fixtures/api/face_refine_advanced_api.json`
- 前端：`examples/workflows/H3_Face_Refine_Advanced_EXP.json`
- 动漫API：`tests/fixtures/api/face_refine_anime_advanced_api.json`
- 动漫前端：`examples/workflows/H3_Face_Refine_Anime_Advanced_EXP.json`
- Parity+MANUAL512 REL基线API：`tests/fixtures/api/face_refine_parity_advanced_api.json`
- Parity+MANUAL512 REL基线前端：`examples/workflows/H3_Face_Refine_Parity_Advanced_EXP.json`

示例要求用户先准备精确24fps、124帧、单镜头素材。真人示例默认YuNet，动漫示例必须显式使用
动漫EXP模型；二者都不做人脸识别。模型缺失时改用手工ROI，而不是运行时下载。先看Plan preview，
再运行第二遍H3；最后必须检查Stitch report和candidate，原片仍保留。
