# S23 H16 七窗口分离式双采（EXP）

本目录提供固定 124 帧、LOW 736×416 → learned3D HIGH 1472×832、34 帧窗口／17 帧重叠的 224 张可导入研究图。LOW partial4 和七个独立 PASS2 窗口是显式节点；EAV、Prompt Relay 或两者组合可只作用 LOW、某一窗口或全部活动阶段。每种配置有完整不保存、完整逐窗保存、从窗口 1–6 任一位置冷恢复三类入口。旧 H16 一体图仍在 `13-latent-upscale`，未迁移或替换。

先看 `none_none_full_no_save`，再看 `combined_all_full_save`。LOW／HIGH 各有独立模型加载；在选中对应效果的图中，各活动窗口才有自己的 Relay Plan 或 EAV Config。逐窗 Relay 先从全片计划投影到 guarded-overlap 局部 AV 布局；EAV 对同窗实际调用做审计。默认 EAV 为 `report_only`，不表示默认画质增强。原 H16 `refined_exp` 音频历史、原时钟和遮罩、一次全局噪声准备与逐窗 learned lift 保持。

`full_save` 的 learned handoff 原生保存和七个 Window Save 均默认 `confirm_save=false`。冷图要填入同一完整运行的 handoff 路径、外部 manifest、文件 SHA256，以及所选前一窗口的 manifest 相对路径和 SHA256。例如 `cold_3` 从 window2 回执只运行 window3–6；`cold_6` 从 window5 只运行最后一窗。冷图不会重采 LOW 或已冻结窗口，但仍重建当前 H16 计划、来源和音频上下文以核验回执；它不证明今天的模型、LoRA、提示词或效果设置与冻结时自动等价。改变冻结阶段设置后应重新生成与保存。

导入后必须先替换首帧占位图，核对本机 FL2VA INT8、Turbo LoRA、Qwen、双 VAE 与 trained learned3D 权重；冷图还必须填真实回执，否则不可排队。当前 Core 静态连接、七窗 tiny 实际效果调用及每个冷窗的 AV 一致性已验证，但这不代表新图原尺寸真实权重、多资产／后端、EAV `apply_exp` 画质、旧新画音对照或人工接缝验收完成。

本批224张图均通过当前Core静态全输出校验和隔离Chromium原生Save As／刷新重开；三张`combined/all`代表图已实际编辑外置Relay与EAV后回读。相关新旧H16及双模型接缝CPU回归595通过。前端验收只检查序列化，未排队生成；旧一体图继续可用。
