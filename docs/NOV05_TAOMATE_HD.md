# TaoMate-HD：低分辨率 2 次＋高分辨率 1 次（实验）

`H3 TaoMate-HD · LOW2 + HIGH1` 实现 TaoMate-HD 的教师时间表耦合渐进采样。**它是新 TaoMate-HD 配方，与已有 TaoMate 三步节点分开保留。** 当前示例是未发布候选，最终代码已完成原生音频时钟、旧双时钟配置两份本机完整 GPU 音视频验证；默认配方指定样片的画面及声音于 2026-11-08 通过人审，旧双时钟仍仅具运行与媒体验证。

导入 [TaoMate_HD_2plus1_EXP.json](../examples/workflows/96-nov05-exp/TaoMate_HD_2plus1_EXP.json)。沿用普通提示词、现有 MODEL/LoRA 加载器和原生 AV 解码/保存；不要求 Prompt Relay。

## 配方和接线

- 使用现有 LoRA 兼容加载器加载 `TaoMate-HD-preview-comfy.safetensors`，示例强度 1。它不是旧 TaoMate LoRA，也不是通用加速 LoRA 的替代名称。
- 默认 `steps=3`、`cfg=1`、`low_scale=0.625`，即 2 次 LOW、1 次 HIGH 的求解器评估。**CFG=1 时对应 3 次实际网络前向；CFG>1 需要正负条件评估，实际前向可能翻倍，不能把求解器步数当成实际网络调用数。** 不强制用户使用 CFG=1。教师 50 点时间表使用索引 `0 / 16 / 33 / 49`，shift 读取所选 MODEL。更多步数为这些教师区间的实验细分，不宣称上游另训了对应配方。
- 高分尺寸由原生 Conditioning 的宽高决定；LOW 按比例和 H3 网格得到。示例为 512×288 → 832×480，输入 124 帧，最后按实际交付端点裁成 120 帧／5 秒。
- HIGH 噪声先生成，LOW 经正交低通投影得到关联噪声。阶段交接使用**双线性状态加 sigma×噪声残差**；这条算法不使用 learned latent upscaler。现有其他双采的学习型放大器不受影响。
- 音频保留 LOW 第二次后的演进状态，HIGH 继续原生音视频时钟；不重新抽取音频噪声、不重采样音轨。
- 不接 `high_model` 时共用输入 MODEL。需要分别设置 LoRA 时，可以从同一底模分支出两条现有 LoRA 链，将 HIGH 链接到可选输入；不要求用户重新选择两个底模。模型和 LoRA 选择、顺序、强度均保留，真实张量/接口要求仍校验。

首帧、多个原生 guide 和既有遮罩通过原生 AV 条件进入采样；LOW 的空间参考按该阶段尺寸调整，HIGH 保留原高尺寸参考与原音频遮罩。CPU 用例覆盖这些接线，不能由此宣称所有首尾帧、角色参考或音频驱动素材已经获得完整训练模型画质验收。

## 已验证和未验证

2026-11-08 指定反馈：默认配方样片画面及声音通过；旧双时钟样片不由此自动取得
画质资格。详细范围见[审片记录](NOV05_REVIEW_20261108.md)。

本轮最终代码选择完整 FL2VA INT8 convrot 底模、新 TaoMate-HD LoRA、Qwen3-VL 编码器、原生 AV VAE，以及分头低显存/FFN 分块处理。以下是最终原生时钟那次运行记录；新增的调用计数来自这次实际运行，不是倒填早期报告。

| 项目 | 本次结果 |
| --- | --- |
| LoRA 映射 | 208 个 targets 全部挂载，未匹配 0 |
| 求解器评估 / 实际网络前向 | 本例 CFG=1，LOW 2 次＋HIGH 1 次，两者均为 3 次 |
| 输出 | 832×480，120 帧，24fps，5 秒，有音频 |
| LOW / HIGH video latent | 37×18×32 / 37×30×52 |
| 进程内执行耗时 | 约 173.5 秒，含该节点图的加载与交付 |
| 该进程 CUDA 峰值已分配 | 8,652,432,896 字节 |
| 人工验收 | 2026-11-08，该指定样片画面及声音通过 |

文件 SHA256 为 `72636e6e6115284b1988f63cbe4eaa68f426d8d43087de9cc1f48ebc5a30e94d`。耗时与显存只描述这一设备和这一运行，不代表公平冷/热速度对比、最低显存或任意时长性能。文件可完整解码不等于手部、动作、对白或接缝画质已通过。

该文件与修复前默认配方的成片逐字节一致；仅音频 guide、CFG 调用计数与旧音频协议的修复没有改变这个 CFG=1 原生配方的输出。另一个最终代码运行使用原有 Dual-Clock 设置的 `audio_scale=1`、video shift 12、audio shift 3，各阶段报告确实进入 raw audio velocity 分支，实际网络调用仍为 3，完整交付 120 帧 AV；这证明该协议能够执行，不推广为任意双 MODEL 组合的质量保证。

现实现为**单窗口**推理；不宣称实现上游流式长历史、八 GPU 流水线或训练。现有长视频已通过的 4+4 成片尾部续接逻辑没有更换，也没有加入生成后的潜空间端点桥接。

模型文件已下载时直接使用，不需要重复下载。本次模型来源为 [slmonker/TaoMate-HD-preview-comfy](https://huggingface.co/slmonker/TaoMate-HD-preview-comfy)，固定 revision `c956fadf7c9bc1e26155a051f3d79bb4453d4e53`；本机文件 SHA256 `352a9ace5c5f4e99d52536267259a000eb514cbb62b745942bb3768cc0df6ec4`。

## 来源与许可

原项目：[TaoLiveAIGC/TaoMate-HD](https://github.com/TaoLiveAIGC/TaoMate-HD)。ComfyUI 适配研究来源：[slmonker/ComfyUI-TaoMateHD](https://github.com/slmonker/ComfyUI-TaoMateHD/tree/1e4fb7c76ed15918a89492f78dbf2250484d1c3f)，固定提交 `1e4fb7c76ed15918a89492f78dbf2250484d1c3f`。T8 复用教师配方与噪声耦合数学，通过现有 Core 采样消费者接入，不新增平行加载器或后台队列。

许可与声明分别保存在 [TAOMATE-HD-LICENSE.txt](../THIRD_PARTY_NOTICES/TAOMATE-HD-LICENSE.txt) 和 [TAOMATE-HD-NOTICE.txt](../THIRD_PARTY_NOTICES/TAOMATE-HD-NOTICE.txt)；MiniMax H3 模型许可及上游各组件许可仍适用。
