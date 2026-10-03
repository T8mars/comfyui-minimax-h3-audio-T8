# 外置 Visible Face MASK（EXP）

七个人脸路线各一对 Full Save / Cold Delivery，共十四张新增图。旧图、原采样、外置效果和原人工确认开关不修改。

- 在 Load Image 中选择完整原片坐标的可见脸 MASK；红通道白色允许已有脸部变化，黑色保护原画面。图中**明确广播同一静态单帧 MASK**。动态逐帧 MASK 应替换成外部完整序列并关闭广播。
- Full Save 保留原采样与独立 Stage Save。Cold Delivery 没有 StageSampler，须手动粘贴所选完成阶段的准确 manifest path + SHA；多人逐角色有独立 receipt。仅调整 MASK 不需重跑已保存的采样。
- 原 Standard / Parity / 动漫 / 手动窗 / Studio / 两人 / 三人的接受或确认逻辑保持原值；新 MASK 不自动接受候选。
- 窗口图仍用完整结果及完整接受 mask，不将局部时间窗 MASK 自动铺到全片。多人使用最终完整 composite state。
- 这组通用模板保留明确素材占位，不包含私人视频、模型或已接受素材。可见脸语义及最终效果仍需人工看听。

详细约束：[VISIBLE_FACE_MASK_EXP.md](../../../docs/VISIBLE_FACE_MASK_EXP.md)。本地未发布；新图机械资格与人审分别记录。

另有两张独立可选 `Temporal_MASK_Light005_Relay` Full Save / Cold Delivery：真实逐帧MASK视频红通道、广播关闭，明确denoise0.05／原12步／原外置Relay。它们不覆盖上述十四图或任何旧默认。MASK序列必须与完整原源同帧／同坐标／24fps；请选lossless RGB，不将运动脸用静态半幅0.5遮罩混合。黑区保护原片，但无法修复候选中本已错位的头姿／手部幻觉；原失败示例不推荐。2026-10-03用户定向复审第2版已通过该新代表；人工标注外置MASK并非自动遮挡分割认证，配方不作任意素材保证。

这张ColdDelivery是原阶段的零扩散交付；原AV／Relay身份审核依赖的CLIP、MODEL、VAE仍保留，并非完全不加载重量的RGB旁路。不得删除原审核以省加载。
