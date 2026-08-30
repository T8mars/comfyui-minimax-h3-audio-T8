# FlashVSR v1.1 视频超分

这三份工作流处理已经生成并解码的视频，不会加速 MiniMax H3 采样。

## 安装

1. 下载官方 [JunhaoZhuang/FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1)。
2. 保持文件名，把整个目录放到 `ComfyUI/models/FlashVSR-v1.1`。
3. HF 模型仓库没有 `posi_prompt.pth`；从[官方 FlashVSR 的 prompt_tensor 目录](https://github.com/OpenImagingLab/FlashVSR/tree/main/examples/WanVSR/prompt_tensor)下载并放入同一目录。
4. 从 [SpargeAttn](https://github.com/thu-ml/SpargeAttn) 或 [Windows wheels](https://github.com/woct0rdho/SpargeAttn/releases)安装与 ComfyUI 当前 Torch、CUDA、Python 和显卡架构匹配的 `spas_sage_attn`。
5. 重启 ComfyUI，拖入本目录任一 JSON。

模型目录需要包含 DiT、`LQ_proj_in.ckpt`、`TCDecoder.ckpt`、`Wan2.1_VAE.pth` 和
`posi_prompt.pth`。节点只检查必要结构和是否能加载，不按哈希、文件大小或像素数拦截。

## 三个工作流

| 工作流 | 用途 |
| --- | --- |
| `Quality_Locked` | 推荐起点；固定公开 LCSA 参数 `2.0 / 3.0 / 11` |
| `Balanced_Dynamic_EXP` | 只降低内部低运动块预算；首尾和高运动块保持基线，必须看完整成片 |
| `Memory_Safe` | 保持固定质量预算，使用同一 seed 的空间分块和阶段卸载；更省显存但通常更慢 |

默认使用 Tiny、BF16、2×。官方主要验证4×；2×是本项目的保守实验路线。音频对象原样返回，
不重采样、不降噪、不调整响度。超分不能补回源片已经丢失的身份、口型或真实纹理。
