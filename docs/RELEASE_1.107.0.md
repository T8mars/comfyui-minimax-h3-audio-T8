# v1.107.0 · Sulphur-3 Corrector

新增 **H3 Sulphur · Corrector加载**和 **H3 Sulphur · Corrector应用**两个节点。将 `s3_corrector.safetensors` 放入 `ComfyUI/models/h3_correctors/`，把加载节点接到应用节点，再将应用后的 `MODEL` 接入所用 Guider。双采可以共用一份 Corrector 权重，并分别设置两路的强度与步骤范围。[节点接线与参数](SULPHUR3_CORRECTOR.md)

随版提供 20 步 Euler、CFG 1 的普通提示词候选示例和 API 工作流。安装包注册 806 个节点。工作流附件包含新增示例和原 22 套已审工作流。[UI 候选工作流](../examples/workflows/99-sulphur3/S3_Corrector_T2VA_EXP.json) · [API 工作流](../examples/workflows/99-sulphur3/S3_Corrector_T2VA_EXP_API.json)

完整 Corrector 权重通过了与上游实现的 CPU 和 CUDA 固定输入对照，两组最大绝对差均为 0。GPU 双采用小型 H3 网络验证了两路实际修正调用和音频遮罩。Core 0.46.0 定向回归 211 项通过，0.43.0 为 88 项，0.41.1 为 49 项。两个旧 Core 环境各有 2 项因缺少历史源码快照跳过。

完整 S3 成片画质尚未验收。带 Corrector 的 Stage 支持采样和保存，跨进程缓存恢复暂未开放。模型权重请从上游获取。[Sulphur-3 模型与权重](https://huggingface.co/SulphurAI/Sulphur-3)
