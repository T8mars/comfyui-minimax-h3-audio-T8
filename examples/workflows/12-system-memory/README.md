# 系统、显存与通用工具

这一组用于运行前审计、内存/计算诊断和不会改变H3语义的通用尺寸处理。

## 工作流

- `Environment_Audit`：报告ComfyUI、Torch、GPU、依赖和能力探针。
- `Activation_Chunk`：实验性MLP激活分块，降低部分中间激活峰值。
- `Qwen_Prefix_Cache`：复用重复参考前缀，减少重复编码。
- `Trajectory_Probe`：记录采样轨迹，定位sigma/时钟/坏帧问题。
- `Latent_Upscale_By32`：按目标比例选择宽高都能被32整除的latent尺寸。

## 当前成果

环境审计、前缀缓存、激活分块和轨迹报告都有独立回归；Latent Upscale By32在普通latent及H3联合AV latent上保持音频值和时钟不变，并报告剩余宽高比误差。

## 使用方法与注意事项

排错先运行Environment Audit和Trajectory Probe。Activation Chunk、Prefix Cache、VBAR和Block Cache解决的是不同内存/计算问题，不应混称。Latent放大不会创造细节，也不保证目标像素数完全等于理论倍数；它优先保证32整除和尽量小的画幅误差。
