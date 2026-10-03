# Visible Face MASK / 可见脸像素外置保护（EXP）

这是独立的最终合成约束，不改原 Face / Parity / 多人修复、检测器、采样器、羽化或音频。

2026-10-03人工反馈：原半幅0.5示例未通过（原Face候选有贴脸手，限制后重影）。MASK只限制变化，不能校正候选头姿或清除其中的幻觉；不要将位置不同的脸用大片半透明遮罩直接叠在一起。失败不因像素／PCM机械通过而算画质通过。

同日定向复审第2版：用户明确通过新的R07轻修／逐帧外置MASK完整代表片。真实画布full12及新进程StageLoad0的完整RGB/PCM一致，原音轨保留。此结论只接受该新素材／配方，不将旧失败改为通过，不认证所有七路线、任意遮挡分割或自动XSeg；旧节点默认及原十四图不变。

新增**独立可选**轻修／逐帧MASK配对图（不改原十四图）：明确denoise0.05、原12步及外置Relay；MASK由Load Video→Get Video Components→Image To Mask红通道输入完整序列，关闭单帧广播。动态人物需要同源、同帧、同几何的逐帧可见MASK；白色内区1、遮挡／不可信区域0，仅必要边缘羽化。MASK生成与语义判断仍外置，没有默认安装XSeg或把几何脸框当成真实遮挡模型。该低噪设置是新配方，不是自动改变旧节点默认，也不保证任意素材修复质量。

ColdDelivery沿原source-bound audit接线：零扩散采样不等于不加载任何模型。原AV／Relay身份核对仍依赖原CLIP、MODEL与VAE，不能为省加载而删去审核支路；新图保留这些原依赖及手动填写Stage path＋SHA，不将它们伪称完全无重量的RGB旁路。

新增两个节点：`MiniMaxH3VisibleFaceMaskBindEXPT8` 与 `MiniMaxH3VisibleFaceCompositeEXPT8`。

## 接线

1. 将完整原始 RGB 和对应完整帧坐标的可见脸 MASK 接入 Bind。白色允许已有脸部变化，黑色保留原始像素。单帧 MASK 默认不广播；需要固定遮罩时明确开启 `broadcast_single_mask`。
2. 将原 Stitch / Gate 的**已混合最终候选**和对应 `changed_mask` 接入 Composite。不能接未贴回的小脸裁剪、fallback mask，或只包含局部时间窗的 mask。
3. 多人路线将最终 `composited_frames` 和最终 `composite_state` 接入，使用它已有的完整 `applied_mask`。`changed_alpha` 与 `multiface_composite` 必须二选一，不自动采用候选或改变旧的接受开关。
4. 原音轨接入可选 `audio`，返回同一输入对象；输出图像交由原保存节点处理。

设原片为 B，原 Stitch / Gate 已混合候选为 C，可见 MASK 为 V，已有 changed alpha 为 A：新图像为 **B + (C − B) × V**，有效变化区域为 A × V。C 已经包含原混合 A，不能再乘 A，否则会把羽化和混合强度平方。V=1 区域精确保留 C；有效 mask=0 区域精确保留 B。很小的正 mask 不以 epsilon 当作零；半精度 RGB 的 alpha / V 在至少 float32 中计算。

## 边界

- 绑定完整 RGB / MASK 张量字节、shape / dtype、声明的 24fps / 完整帧区间、广播策略及实现代码。使用时重新核对；源、mask、receipt、时间或代码变化须重新绑定。
- 要求原片、最终候选、已有变化 mask 的完整几何和帧数一致；不静默 resize、裁切、补帧、retime、clamp，RGBA 应在接入前显式转换。
- 24fps / `source_start_frame` 是调用方对完整 RGB 缓冲区的声明，不冒称已经测量媒体 PTS。
- 默认不缓存不透明 mask binding；修改 mask 只重新执行合成 / 交付，不使已完成采样 Stage 文件自动失效，也不把新 RGB 偷塞入旧采样条件。
- 不自动下载 SAM / XSeg 或宣称遮挡分割正确。遮罩来自用户或外部节点，其“确实只含可见脸”的语义质量及最终整片看听仍需人工确认。
- 此功能不降低扩散 NFE，不证明速度或人脸身份 / 接缝 / 口型质量通过。实际资格范围见本地交接；尚未发布。

新增通用配对图见 [65-radar-visible-face-mask](../examples/workflows/65-radar-visible-face-mask/README.md)：七路线十四图的接线及当前 Core 校验、十一完整 CPU 范围203测试已通过。另一个真实原生画布代表复用原 Standard Face 完成 Stage，Save/reopen/Queue 零采样，完整124帧736×416；新进程同路零采样完整 RGB/PCM 精确且原音轨解码保持。本例使用明确的合成静态几何限制 MASK，保护了321539个原会变化像素；不是自动遮挡分割或真人质量批准，也不冒称七路线均做过新画布/媒体验收。14图本身仍保留素材/检查点占位，与私有实跑副本分开。
