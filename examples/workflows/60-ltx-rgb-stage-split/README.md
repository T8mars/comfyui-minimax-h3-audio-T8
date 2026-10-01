# S27 RGB → LTX 外置 Stage（实验图）

两张图从 `22-sol-engine-h3-super` 中对应的旧图逐字节固定来源生成；旧图不改。新的 `LTXRGBStageBind` 明确绑定输入 RGB、原始 H3 音轨、LTX 准备结果及二采控制，后接独立的 `SamplerCustomAdvanced`，再经 `LTXRGBStageAudit` 审核候选 latent 后解码。Identity Preserve 与普通 Refiner 各一张。

这里的 RGB → LTX 是已生成 H3 视频后的独立精修阶段，并非把 H3 生成首采也放在同一张图内；不能以单张图中只有一个 sampler 误称 H3 双采已经完整验证。当前有确定性 CPU 合同、保存 JSON 结构和两张公开图的原生浏览器另存重开资格，但没有跨进程冷检查点、真实权重视频／音频或人审。外置 EAV／Prompt Relay 与其它双采路线的完整矩阵仍需单独实现和验证。本目录不替代旧图，也不推荐无审核直接用于成片。
