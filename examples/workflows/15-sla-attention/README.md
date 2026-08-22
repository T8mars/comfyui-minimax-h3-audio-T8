# LightX2V SLA Attention 工作流

本目录提供 MiniMax H3 FL2VA 的 LightX2V Turbo-SLA 实验工作流：基础版由SLA节点独占attention；组合版允许保留KJNodes的MiniMax H3 Sage完整forward，并由专用Composer逐调用选择唯一后端。普通SLA节点仍不接受KJ、Sol-Attn、FETA、Prompt Relay、BlockCache、STG或其他attention/block patch。

## 两份入口

- `2026-08-22_H3_LightX2V_SLA_FL2VA_4Step_Advanced_EXP.json`：基础严格版，Dual-Clock后直接接SLA。
- `2026-08-22_H3_LightX2V_SLA_KJ_Sage_Composer_FL2VA_4Step_Advanced_EXP.json`：KJ组合版，连接顺序为 `Dual-Clock → KJ MiniMax H3 Sage → SLA + KJ Composer`。不要再插入 `ModelAttentionBackend` 或 Sol-Attn。

## 使用方法

1. 准备 `minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors`，放入 `ComfyUI/models/loras`。
2. 使用 FL2VA 基座，同时接入首帧与尾帧。上游 LoRA 明确标注的正式基座是 BF16 FL2VA；INT8 ConvRot 只属于本机兼容实验。
3. Dual-Clock 固定为 `native_flow`、4 NFE、视频 shift 6、音频 shift 3，并把同一组 `sigmas` 接入 SLA 节点。
4. 生成时选 `apply_lightx2v_sla`；做同 LoRA 科学对照时选 `dense_lora_control`；完全旁路时选 `disabled_identity`。
5. 采样后必须连接 Runtime Audit。成功的稀疏运行应看到 4 次前向、每次 50 次主 attention、共 200 次稀疏内核、0 fallback、0 kernel failure。

组合版中，`apply_lightx2v_sla`的200次主调用必须全部进入SLA block-sparse Sage2，KJ调用数为0；`dense_lora_control`则必须有200次KJ Sage调用和0次SLA稀疏调用。每次Attention只计算一个后端，并不是把两个加速结果相乘。

## 已完成验证

- 固定 LightX2V 代码 revision：`f8aee98b5462cca8d7288888146ebd95592bf266`。
- 固定模型 revision：`10ade67cd15ff7a135fa35c2a0673ea96c839247`。
- LoRA SHA-256：`5CAE6DF40A06EA825F85FC8876C9EA1C9692C833A9AF07BB8B3BAC9CE2A71BAC`；208 个 LoRA patch 全部映射并加载。
- RTX 4060 Ti / sm89 上，一条 256×256×22、4 NFE 的 FL2VA INT8 兼容机械运行完成：4×50 次稀疏调用，失败和回退均为 0。

## 科学边界

这里实现的是 LightX2V 发布版的 learned dynamic block router 数学与 Sage2 block-sparse 内核接入，不是一个可脱离 SLA LoRA 单独使用的通用“加速开关”，也不代表已经复现通用 SLA 论文中的全部 sparse+linear 分支。KJ组合器解决的是完整forward与ComfyUI attention override的hook冲突，并不让同一调用重复运行两种kernel。当前验证只证明节点、LoRA、路由和稀疏内核真实生效；尚未证明相对稠密对照的画质更好、速度一定更快、音频不劣或所有 16GB 工作流安全。公开工作流默认 736×416×124，请按显卡余量逐步试用。
