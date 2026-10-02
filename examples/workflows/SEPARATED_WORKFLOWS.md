# 分离式工作流总索引

全部 S01–S29 已有前端 JSON 文件保存在下列目录；打开目录说明选择“完整生成/保存”或“冷恢复/只跑后采”。这里说明保存位置，不代表任意素材、模型和效果组合均已通过画质验收。

最新 B8 / C1 / C2 音频修正版的完整/恢复配对入口在 [64-reviewed-audio-followup](64-reviewed-audio-followup/README.md)。三份用户通过的原生实跑副本另有本地交接记录；通用示例不带私有检查点。旧 S26 / S29 保留以兼容旧工作流，新 RF 修正需要本地新增 JointClock 节点，不在既有 v1.88.0 中。

| 路线 | 已保存的分离式图目录 |
|---|---|
| S01 Base Flow / S02 LBH / S03 完整首采 | [40-modular-two-pass](40-modular-two-pass/README.md) |
| S04 PDD | [19-pdd-acceleration](19-pdd-acceleration/README.md)，选择 `PDD_Split_*` |
| S05 VDN | [41-vdn-two-pass](41-vdn-two-pass/README.md) |
| S06 / S07 Dual MODEL | [42-dual-model-split](42-dual-model-split/README.md) |
| S08 FastH3 V2 | [34-fasth3-v2](34-fasth3-v2/README.md)，选择 `FastH3_V2_Split_*` |
| S09 手动第二采 | [43-manual-second-pass](43-manual-second-pass/README.md) |
| S10 Progressive | [44-progressive-split](44-progressive-split/README.md) |
| S11 Progressive 续段 | [45-progressive-continuation-split](45-progressive-continuation-split/README.md) |
| S12 Avatar Progressive | [46-avatar-progressive-split](46-avatar-progressive-split/README.md) |
| S13 HyperFlow Continuous | [47-hyperflow-continuous-split](47-hyperflow-continuous-split/README.md) |
| S14 / S15 HyperFlow Fresh | [48-hyperflow-fresh-split](48-hyperflow-fresh-split/README.md) |
| S16 HyperFlow P7 | [49-hyperflow-p7-split](49-hyperflow-p7-split/README.md) |
| S17 SPEED / 多模态 | [50-speed-split](50-speed-split/README.md)、[51-speed-multimodal-split](51-speed-multimodal-split/README.md) |
| S18 Chunked v1 | [52-chunked-v1-split](52-chunked-v1-split/README.md) |
| S19 / S20 / S21 Chunked v2/v3/v4 | [53-chunked-v234-split](53-chunked-v234-split/README.md) |
| S22 Chunked v5 | [54-chunked-v5-split](54-chunked-v5-split/README.md) |
| S23 H16 | [55-h16-split](55-h16-split/README.md) |
| S24 Face / 多人 / 动漫 | [56-face-refine-split](56-face-refine-split/README.md)，含独立 Source / 冷交付子目录 |
| S25 Motion Recovery | [57-motion-recovery-split](57-motion-recovery-split/README.md) |
| S26 Audio Refine | [59-audio-refine-split](59-audio-refine-split/README.md)，含 `effects` 外置 EAV / Relay |
| S27 H3 RGB → LTX | [60-ltx-rgb-stage-split](60-ltx-rgb-stage-split/README.md)，含原生加载/冷交付子目录 |
| S28 Prepared LTX | [61-prepared-ltx-split](61-prepared-ltx-split/README.md) |
| S29 RF Restart | [62-rf-restart-split](62-rf-restart-split/README.md)，新 JointClock 入口见上方 64 目录 |
| Veda 专用8步 T2VA | [63-veda-t2va](63-veda-t2va/README.md) |

注意：JSON 已保存不等于权重、素材和检查点已随图打包。恢复图需填写对应完整图实际 Save 的路径、manifest 和 SHA，不能使用占位值；独立 Relay/EAV 依具体路线说明接入。完成结果读取图只解码/交付，不能在其中重新施加采样效果。原一体图、默认值和审核失败记录不覆盖。
