# Source AV 同源帧映射（H05 N02，EXP）

这是新增的可选接口。旧 Source Media Window 的输入、五输出、默认帧选择、
空间 resize 和音频处理不变；不用新接口的旧工作流无需迁移。

## 接线

Load Video → **Source Decode + Clock** → **Source Conform Once**。
把同一次解码的 frames、audio、source_fps、source_clock 一起接入。
Conform 输出准备好的 RGB／32kHz stereo AUDIO 和完整 frame_map。
不要先在另一支独立裁视频、变 fps，再用相同尺寸宣称控制图同源。

全源 RGB → **Derive Control** → **Map Control**，共用上述 frame_map。

- `identity_rgb`：原 RGB，不另算时间。LOW／HIGH 可分别用 Map Control 的实际宽高。
- `grayscale`：确实由这份 RGB 执行逐像素通道平均；只是简单演示，不是 pose／depth 模型。
- `manual_mask`：用户提供与完整源 `N×H×W` 匹配的逐帧 MASK，显式绑定这份原片。
  白色允许重绘，黑色保留；不可见的目标应逐帧标零。不暗中广播单张 MASK。
- 外部 pose／depth／edge 可以直接接 Map Control。仍可运行，但报告
  `external_origin_unverified`；同尺寸、文件名或用户 JSON 不是预处理来源证明。

MASK 用 nearest-exact，RGB 用原 Core Lanczos，均不裁切；不是声称两者像素算法相同。
LOW／HIGH 共享源索引和 map SHA，各自输出内容 SHA 与 geometry，不能把 LOW 的画布
身份当作 HIGH 已验证。mask 像素到 latent 的扩张仍由所选 Fun／LanPaint 处理，另看实际报告。

## 实际证据与限制

完整映射记录的是**现有准备函数实际使用的** float64 nearest/ties-to-even/clamp
索引及时间，而不是在新节点重新实现重采样。仅调用一次既有 Source AV 函数。
源 RGB、准备后 RGB、输入与输出 PCM 内容、源索引、hold 和音频补零均可查。
输入／输出被修改或自有 lineage 不一致时拒绝过期回执；不是限制未知用户模型或第三方控制器。

Decode + Clock 在同一次原生 Core 解码处观察 PTS，Core 仍拥有 crop、trim、rotation、
浮点转换与音频选择。只有真实连续等间隔 PTS 与输出帧数匹配才标 `observed_CFR`。
VFR、量化到不等间隔 PTS、materialized 或未知 provider 标 `clock_unverified`，
保留运行但不冒称实际 CFR。未接 Clock 时是 `declared_fps_unverified`。
本版本不认证音频 PTS 严格对齐，不偷偷修正 Core 的源音频时间处理；视频首 PTS 与 trim
原点分别报告。缺失音轨产生的静默明确为 `generated_silence`，不是“保留原声”。

这里是完整 IMAGE 驻留的显式 4096 帧／每 tensor 2GiB 证据预算，不是流式解码或低内存承诺。
原生解码在累积前按完整源画幅保守检查 RGB／PCM 预算。取消与坏数据错误正常传出。
MASK 绑定只证明用户注释关联与内容完整性，不证明选对物体、遮挡者或动作。

已在独立 CPU Core 的真实画布执行一次机械代表：同一视频解码，LOW RGB 准备、
HIGH RGB 映射及 LOW／HIGH MASK 三路控制共用完整 map；120 帧／24fps／5 秒输出
完整解码通过。MASK 使用通道机械夹具，不是杯子／遮挡人工标注；没有新增采样 NFE。
旧用户工作流不变，尚未取得新局部编辑的 GPU／人工画质验收。
N01 已通过的单图 Qwen 样片不授予此新功能质量资格。

短名机械图另存为
[`H05_Source_Map_EXP.json`](../examples/workflows/77-h05-source-map/H05_Source_Map_EXP.json)。
它保留真实 NativeSaved 原字节；本机默认用户目录也有单独的新副本。
仓库不附输入媒体，Load Video 须选择实际素材。图内固定 120 帧／24fps／5 秒的
交付裁齐不是任意 fps 或时长的自动适配；改变素材时同步调整并检查报告。

## N03 局部编辑与遮挡

现有 LanPaint AV Prepare 仍用原来的 `adaptive_max_pool3d`（各轴降采时）或
nearest（需要升采时）生成 video_noise_mask；新增报告不改变 mask 或 sampler。
报告 `video_mask_transport` 显示实际源活动帧、实际 latent 活动时间单元以及每个单元
对应的源帧半开区间。例如 22 帧→7 单元，源第9帧的一次局部标记会影响
第2单元 `[6,10)` 和第3单元 `[9,13)`。这不是 VAE 感受野或最终像素精确保持的认证。

杯子／衣物编辑应保存原始采样解码与可选最终回贴两份片。遮挡时人工 MASK 为零，
但压缩后的邻近时间单元仍可能受前后可见帧影响；不要用最终 composite 隐藏原始候选失败。
生成的 audio mask 为零只表示采样保护意图；原 PCM 是否交付、是否重采样／补零另行报告。
现有 Union2 audio skip 也不等于联合音频逐值不变。
