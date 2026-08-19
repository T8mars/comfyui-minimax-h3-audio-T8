# 系统、显存与通用工具

这一组用于运行前审计以及内存、计算与采样轨迹诊断。

## 工作流

- `Environment_Audit`：报告ComfyUI、Torch、GPU、依赖和能力探针。
- `Activation_Chunk`：实验性MLP激活分块，降低部分中间激活峰值。
- `Qwen_Prefix_Cache`：复用重复参考前缀，减少重复编码。
- `Trajectory_Probe`：记录采样轨迹，定位sigma/时钟/坏帧问题。

## 当前成果

环境审计、前缀缓存、激活分块和轨迹报告都有独立回归。普通32整除放大与学习型二阶段放大已移动到相邻的`13-latent-upscale`目录。

## 使用方法与注意事项

排错先运行Environment Audit和Trajectory Probe。Activation Chunk、Prefix Cache、VBAR和Block Cache解决的是不同内存/计算问题，不应混称。
