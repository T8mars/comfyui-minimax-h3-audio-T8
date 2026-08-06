# Example

`audio_lock_api.json` 是 API prompt 格式示例，展示完整链路：音频窗口、统一条件、H3
采样、联合解码、同步裁切和 MP4 输出。

`dual_clock_4step_api.json` 展示 H3 Turbo LoRA 的四步原生音画生成。双时钟节点同时
替代 Sigma Shift、sampler 选择器和 scheduler，输出直接接入 `SamplerCustomAdvanced`。

`multirate_exp_api.json` 展示独立 EXP 节点的 4 个视频宏步 / 8 个音频微步设置。
`audio_steps` 是完整联合 H3 DiT 的实际调用次数；测试 4/10 时直接把它改成 10。

`still_image_edit_api.json` 展示 Ref2VA 实验性单图编辑：主图作为 `<Picture 1>`，直接
生成一个视频 latent 帧并用 Still Decode 输出 IMAGE。当前示例使用本机已有的 pruned
Ref2VA INT8，不加载与其不完整兼容的 Turbo LoRA，并以 20 步运行。需要把 LoadImage 的
占位文件名替换为输入图。

`workflows/` 内是可直接拖入 ComfyUI 画布的完整示例：

- `H3_Turbo_Stable_4V4A.json`：稳定双时钟 4/4；
- `H3_Turbo_EXP_4V8A.json`：EXP 视频 4 / 音频 8；
- `H3_Turbo_EXP_4V10A.json`：EXP 视频 4 / 音频 10。

三份画布工作流使用相同 seed、prompt、EMA LoRA 和输出设置，便于直接比较。它们已写入
当前 T8 安装中实际存在的非裁剪 H3 INT8 基模、NVFP4 H3 文本编码器和两套 VAE 文件名，
并使用 `LoraLoaderBypassModelOnly`，适配 INT8 基模。要比较非 EMA LoRA，只需在 LoRA
节点中切换到 `minimax_h3_turbo_4步加速_comfyui.safetensors`。

API prompt 内的占位模型名需要手动替换；`workflows/` 中的前端工作流已经使用当前机器的
实际文件名。示例最后使用 VideoHelperSuite 的 `VHS_VideoCombine`；核心 T8 节点本身不依赖
VideoHelperSuite。
