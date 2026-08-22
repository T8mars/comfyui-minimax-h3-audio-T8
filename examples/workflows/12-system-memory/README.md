# 系统、显存与通用工具

这一组用于运行前审计以及内存、计算与采样轨迹诊断。

## 工作流

- `Environment_Audit`：报告ComfyUI、Torch、GPU、依赖和能力探针。
- `Activation_Chunk`：实验性MLP激活分块，降低部分中间激活峰值。
- `Qwen_Prefix_Cache`：复用重复参考前缀，减少重复编码。
- `Trajectory_Probe`：记录采样轨迹，定位sigma/时钟/坏帧问题。
- `2026-08-22_H3_External_BlockSwap_Stock20_Advanced_EXP.json`：把 T8 的保守 BlockSwap 参数桥接到独立 MiniMax H3 流式运行时。

## 当前成果

环境审计、前缀缓存、激活分块和轨迹报告都有独立回归。普通32整除放大与学习型二阶段放大已移动到相邻的`13-latent-upscale`目录。

## 使用方法与注意事项

排错先运行Environment Audit和Trajectory Probe。Activation Chunk、Prefix Cache、VBAR和Block Cache解决的是不同内存/计算问题，不应混称。

外部 BlockSwap 工作流需要单独安装 `xiaolibai-sys/ComfyUI-MiniMaxH3`；本机核对 revision 为 `099aa38c122cea030ce45a51eb1d83208b16a363`。该上游当前未声明源码许可证，因此本项目不复制或再分发其源码，只输出兼容的 `MINIMAX_H3_SWAP` 参数对象。桥只能连接外部 `MiniMaxH3KSampler`，不能连接 ComfyUI 官方 `MODEL`。示例从736×416、5秒、Stock20、SDPA和上游自动显存策略开始；未做16GB压力认证，不承诺不会OOM。
