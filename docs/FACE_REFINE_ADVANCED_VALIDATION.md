# MiniMax H3 Face Refine Advanced 验证记录

## 定位

这是一条隔离的实验性“局部二次生成”链，不是确定性超分或人脸复原。它把远景脸所在的
时序 crop 放大到 H3 画布，以低去噪第二遍生成补充结构，再只把脸区回贴到原视频。
任何输出都只是候选，不能自动覆盖原片或宣传为身份保持、通用修复、16GB安全。

实现为四个只追加的节点：

1. `MiniMaxH3FaceRefinePlanT8Advanced`
2. `MiniMaxH3FaceRefineConditioningT8Advanced`
3. `MiniMaxH3FaceRefineSamplerT8Advanced`
4. `MiniMaxH3FaceRefineStitchAuditT8Advanced`

稳定`sampling.py`、既有90个Node ID及旧工作流均不修改。

## P0 已实现机械合同

- 输入必须是精确24fps、帧数满足`17n+5`，tensor路线最多362帧；不隐式重采样、裁帧或补帧。
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
- 本轮最终门：505项全回归、全仓Ruff、compileall、101份JSON解析、`git diff --check`均通过；
  稳定`sampling.py`SHA-256仍为
  `111da5e52b28f2424f57b36f88db63e3ea02b538a8cdfdea1c8ad2f122ad7bb5`。

## 尚未授予的声明

以下真实证据在完成前，功能必须保持Advanced EXP：

- 更大授权集上的真人/动漫precision、recall以及多人交叉的轨迹交换率；当前真实样本只关闭
  正脸、侧脸、局部遮挡、远景和单张动漫的最小能力探针。
- H3第二遍对远景脸清晰度、自然度、身份、表情、嘴型、时序稳定和接缝的同NFE盲评。
- 真实多镜头拆窗、Selective Repair接受/回滚和Long Video文件级恢复。
- 124帧自动链单次峰值已过512MiB门槛约54MiB，但余量很窄；仍缺完整H3三冷三暖、连续三任务、
  362帧GPU峰值及主机private/commit，因此不写通用`memory_safe`。
- 可选身份模型的许可、genuine/impostor门槛与至少5人盲评；当前实现不做身份验证。

## 发布否决门槛

- 错人、串脸、后脑勺生成正脸、cut前轨迹污染cut后镜头。
- 中心漂移超过源画面对角线2%，尺度漂移超过5%，或嘴型/A/V同步恶化超过1帧。
- mask外任一像素变化，音频PCM/样本数变化，或源/plan/hash不一致仍继续。
- 任一隐式裁帧、补帧、空间latent适配或静默降低canvas。
- 真实推荐档OOM、最低余量低于512MiB或连续任务出现超过256MiB阶梯增长。
- 盲评偏好95%置信区间下界未超过50%时宣传“稳定修复”。

## 示例

- API：`examples/face_refine_advanced_api.json`
- 前端：`examples/workflows/H3_Face_Refine_Advanced_EXP.json`
- 动漫API：`examples/face_refine_anime_advanced_api.json`
- 动漫前端：`examples/workflows/H3_Face_Refine_Anime_Advanced_EXP.json`

示例要求用户先准备精确24fps、124帧、单镜头素材。真人示例默认YuNet，动漫示例必须显式使用
动漫EXP模型；二者都不做人脸识别。模型缺失时改用手工ROI，而不是运行时下载。先看Plan preview，
再运行第二遍H3；最后必须检查Stitch report和candidate，原片仍保留。
