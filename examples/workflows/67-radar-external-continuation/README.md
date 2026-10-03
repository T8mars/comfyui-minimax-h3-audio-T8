# 外片续拍：外置 EAV／Relay（EXP）

四张新增图，旧工作流不变：EAV Only、Relay＋EAV 各一张 Full Save 与 Cold Delivery。

Full 先填写已保存且手动采用外片的 project／shot／take ID，检查本机模型。默认单个完整4步、448×256、124渲染帧／22上下文，交付显式裁前22帧。Relay／EAV 默认 report_only，不自动采用、拼回、接受或改变原音频。

Cold 填写实际完成 Stage 的 manifest 路径和 SHA，只解码固定选中结果，无扩散 MODEL、CLIP 或采样器。它不是自动确认当前改过的源、提示或配方仍与旧结果相同。

真实采样／冷交付机械资格与主观接缝／身份／音频质量独立。当前默认可编辑示例不代表所有素材已通过人审。详见 [外片续拍](../../../docs/EXTERNAL_CONTINUATION_EXP.md) 与 [外置效果](../../../docs/EXTERNAL_CONTINUATION_EFFECTS_EXP.md)。
