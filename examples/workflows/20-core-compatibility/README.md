# 官方核心兼容

这组节点把仍在 PR 阶段的 ComfyUI H3 能力以可选方式提供给当前版本；官方核心已经支持时会自动透传。

- `AV Latent Builder`：组合独立的视频与音频 latent，并检查 batch、通道和时间长度。
- `Attention Hooks`：让旧 H3 核心支持标准 `attn1_patch` / `attn1_output_patch`。
- `Forward Sync Optimization`：每步只读取一次 sigma 标量，并缓存文本 tag 列表；不改采样 schedule。
- `Tiled VAE Coordinate Audit`：默认只报告。当前 fp16 VAE 真实复核中，上游全局坐标候选反而产生更强规则网格，因此未作为修复启用；实验模式仅供复现。
- PDD 原生支持已内置到 `19-pdd-acceleration` 的原节点：新版核心优先，旧核心回退。

以上均为追加节点，不会更改旧节点字段和默认值。请按用途接入，不需要全部串联。
