# 16 · RAVEN Streaming

本目录只包含 MiniMax H3 RAVEN 分块流式 T2VA 实验工作流。它不会替换项目中的稳定双时钟、
长视频或普通 H3 采样路线。

## 依赖

- 单独安装 `YanzuoLu/ComfyUI-MiniMax-H3-RAVEN-Streaming` 0.1.0 或更高兼容版本。
- 使用完整、非裁剪、非 INT8/FP8/NVFP4 的 MiniMax H3 BF16 diffusion model。
- 安装对应的 MiniMax H3 RAVEN Streaming LoRA；强度固定为 1.0。
- 继续使用官方 MiniMax H3 CLIP、视频 VAE 和音频 VAE。

本节点包不复制 RAVEN 运行时，也不下载或分发上述模型。真正执行分块采样和节点内预览的是外部
`RAVENStreamingSampler`；T8 节点只提供统一参数、加载前保护和请求合同审计。

## 当前工作流

- `2026-08-23_H3_RAVEN_Streaming_T2VA_Guarded_Advanced_EXP.json`

工作流默认使用论文/插件发布配置：4 NFE、视频/音频 shift 12/3、sink/window 2/2、
`cpu_pinned` KV。Profile 的六个输出同时连接 Request Audit 与外部 sampler，避免两处参数不一致。

首版只接受 T2VA：`first_frame`、`last_frame`、参考图/视频/音频、mask、已采样latent、CFG、negative、
普通KSampler和其他attention/model wrapper都不在已验证合同中。不要把Sage、Sol、SLA、FETA、
Prompt Relay、BlockCache、STG或其他模型补丁接到这条图上，除非以后有独立实测证据。

## 资源边界

默认 Guarded Loader 在模型加载前要求约24GiB显存、192GiB物理内存、160GiB当时可用内存，并检查
CUDA/BF16。该门槛来自外部插件的审阅证据；上游是在H200上把可用显存限制到24.1GiB验证，不是
物理24GB消费卡，且记录的主机RSS超过129GiB。因此通过预检也不是OOM保证。

当前本机RTX 4060 Ti 16GB、约128GB内存不在审阅范围，默认会在加载巨大权重之前拒绝，这是正确行为。
`report_only`仍会继续加载，可能造成系统交换、OOM或卡死，不建议使用。

工作流保守采用768×448×90帧，画布必须32整除且面积不超过1376×768；长度使用`17k+5`，22～362帧，
超过192帧需要显式确认实验风险。短帧不会显著降低完整模型与文本编码器所需的主机内存。
