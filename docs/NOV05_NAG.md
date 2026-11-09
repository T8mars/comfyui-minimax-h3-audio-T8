# H3 NAG：CFG=1 负向注意力指导（实验）

`H3 NAG · Negative Attention / 负向注意力` 让 CFG=1 的 H3 采样也使用负提示词。它不加载模型、不改变 LoRA、采样器、步数或 sigma 表。默认关闭，原有工作流不变。

## 接线和参数

1. 按原工作流加载底模和 LoRA，保留注意力、FFN 等选项。
2. 用同一 H3 文本编码器编码负提示词，接到 NAG 的 `nag_negative`。正提示词照常接 Guider。
3. NAG 的 MODEL 输出接原来的采样准备或 Guider；先使用 CFG=1。
4. 采样结果和 NAG 的 `runtime` 接 `H3 NAG · Actual Calls / 实际调用`，可读取本配置的真实调用统计，LATENT 原样传给解码。

初始参数为 `scale=5`、`tau=2.5`、`alpha=0.25`、`sigma_end=0`，来自上游节点的默认值，不代表适合所有模型。`scale` 控制负向指导强度，`tau` 限制指导向量相对正向向量的范数比例，`alpha` 混合原正向结果，`sigma_end` 以下停止应用。`scale<=1` 或 `alpha=0` 旁路本节点。

**空负提示词与关闭 NAG 不同。** 空文本仍由编码器产生向量，启用后仍执行负向注意力。要关闭此处新增的 NAG，请取消 `enabled`；若输入 MODEL 已在上游应用过 NAG，旁路会保留它，并在报告中标注 `upstream_nag_retained`。

当前负向输入为一项、batch=1 的纯文本 CONDITIONING。多段文本先合并后编码；不会静默截掉多 batch 或伪造缺失的模态标签。NAG 仅作用于原生正向分支，实际 batch/CFG 调用没有进入可用分支时，报告记录 bypass，不冒充已应用。

## 与其他节点组合

保留所选 MODEL 的 QKV、RoPE、输出投影、分头注意力和 FFN 路径。正向注意力仍委托原有 `optimized_attention_override`；额外负向流使用 dense 注意力，因为其文本行不同，不能沿用正向稀疏布局。

分头计算完成后，才在真实输出投影之前对**全部头拼合的隐藏维度**计算 NAG 范数。按每个头组分别归一化会改变算法，已用独立 SDPA/公式对照复现并修复。它保留分头内核，但增加正向结果缓冲和负向注意力，不能据此承诺显存更低或更快。

未知模型类不会被白名单拦截。实际接口必须提供完整 packed token 行、可委托的 optimized attention 以及输出投影；完全绕过这些接口的私有 fused producer 会报告具体缺失接口，不会静默删除其补丁或换成 dense 冒充组合成功。当前 Core 的直接 chunked sparse producer 在进入该私有路径时不满足此接口；本文不宣称该 CUDA 组合通过。普通正向 override 与原生低显存分头组合已有实际消费者测试。

重复应用 NAG 会更新其自身配置，保留前面的其他 block patch。采样中断、负向预处理报错和注意力报错会清理临时投影 hook 与本次负向流；原 MODEL 不被持久修改。统计范围是 `since_this_configuration`，可跨同一配置的多次采样累计；不是所有平台、所有模型的画质认证。

## 验证范围

`tests/test_nag_h3.py` 使用真实 Core 的小型随机权重 H3、原生 Guider/Sampler、注意力与输出投影：

- 关闭精确旁路、空负向与正常负向产生不同结果；替代 MODEL 类可执行。
- 1/2/3 头组与独立全维公式一致；正向第三方 override、所选 preferred backend 确实被调用。
- Semantic Bridge、真实 image reference、EAV 正向 override 与 NAG 组合可执行。
- 重复配置、克隆、sigma 截止、异常清理、未暴露内核的真实接口错误。
- 当前安装 Core 0.44.0 与本地固定 0.41.1、0.43.0 的 CPU 消费者验证分开记录；CPU 验证禁止初始化 CUDA。

这些 CPU 测试证明数值与接线合同，不证明预训练模型画质、稀疏 CUDA 内核、任意编译器或任意第三方组合。

本轮另以实际 Pika FL2VA 权重完成关闭、空负面文字、有负面文字三份 GPU 成片：每份 6 次网络前向，832×480、120 帧、有音频。两份启用组各观察到 6 次 active、300 个 guided blocks、2400 次注意力内核调用，bypass 和 failure 均为 0；空文字为 1 个负向 token，实际负面文字为 23 个。关闭组没有附加 NAG runtime。三份均完整解码，三份指定画面已通过人审、声音未审；这些数字不证明所有第三方注意力后端可用。

2026-11-08 反馈更新：三份指定画面通过，声音未审。
范围见[审片记录](NOV05_REVIEW_20261108.md)，不扩展为其它组合质量通过。

## 来源

基于 [BenderBlender/ComfyUI-NAG-MiniMaxH3](https://github.com/BenderBlender/ComfyUI-NAG-MiniMaxH3/tree/c63a04674047cfaa3c538286ad2041d6375735d2)，固定提交 `c63a04674047cfaa3c538286ad2041d6375735d2`，MIT；许可见 [NAG-H3-LICENSE.txt](../THIRD_PARTY_NOTICES/NAG-H3-LICENSE.txt)。T8 改动包括按实际接口检查、关闭旁路、完整头维度归一化、原调用路径委托、临时 hook 清理和真实调用报告。
