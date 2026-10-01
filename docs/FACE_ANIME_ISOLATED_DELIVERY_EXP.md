# 动漫 Face 分离采样：显式隔离导出（EXP）

新增两张独立工作流，只将交付端换成已有的 `Save Video · Isolated H.264`，不改采样、裁剪、外置 EAV／Prompt Relay、来源审计或音轨接线。原有八十四张 Face 图及普通 `SaveVideo` 默认路径不变。

- [完整采样并保存 Stage](../examples/workflows/56-face-refine-split/isolated-delivery-exp/S24_Face_anime_combined_full_save_Isolated_Delivery_EXP.json)
- [加载指定 Stage 后交付，不重采样](../examples/workflows/56-face-refine-split/isolated-delivery-exp/S24_Face_anime_combined_cold_delivery_Isolated_Delivery_EXP.json)

导入后先设置源视频、模型及素材。完整图运行后，将该 job 返回的 `artifact_path` 和 `artifact_sha256` 粘贴到同一源片、配方与效果的冷图；空回执不能排队。冷图重新核验当前来源，但没有 Stage Sampler，不重新执行增强或冻结 job 的 Relay。EAV 仍默认 `report_only`，不是自动接受修复。

两侧都使用同一隔离导出节点；报告单独接到 PreviewAny。导出保留所选 VIDEO 的帧率、色彩／位深与原输入音频，独立目录写入并严格完整解码后才发布文件。默认期限 600 秒、暂存预算 16 GiB、最低空闲磁盘 2 GiB，不降低资源保护，不全局修改编码器或旧节点。

## 已验证的范围

一条明确选择的虚构单人动漫测试素材，已在真实原生 Chromium 画布完成原配方十二步 full-save，再在新 Core 进程中零采样恢复。两侧均显式使用上述已有隔离导出节点，完整 124 帧、736×416、24 fps、5.167 秒 H.264/AAC 严格解码通过。原 VIDEO 输入身份、量化 RGB、转换 YUV、完整解码 RGB 与 PCM 均逐值相同；源文件、六份选定权重、隔离运行时、控制器及服务收尾守卫通过。

这是该素材的机械执行和恢复证据，不是所有动漫素材、画质或人工接受认证。两张新图的可见编辑重开另经下述独立验收，不直接继承旧执行副本的全部资格。两个 MP4 容器字节并不相同，也未证明普通 SaveVideo 与隔离编码逐值等价。

两张新增图的五项完整图结构／实际节点契约／保存文件验证已通过，并包含在二十三个完整文件范围的 448 项 CPU 定向回归中，原八十四张图字节保持。该回归绑定其实际 4736 文件源码版本，之后 v5 兼容修改另有独立回归；不能当成此后全仓结果。

两张新图随后在独立 CPU 服务的真实原生画布完成可见编辑、Save As、刷新与重开：完整图仅在私有 QA 副本编辑外置 Relay 文本及 EAV 的 `report_only` → `apply_exp`，冷图不编辑；两份真实界面保存文件再逐字节复制到新 CPU 进程并从工作流侧栏重新打开。完整图 26 节点／63 条具名边／80 个控件，冷图 17 节点／31 条具名边／67 个控件；两轮独立文件审计确认显式编辑保留，其余仅原已审查的界面默认控件追加，公开 JSON 不变。截图确认 Relay 文本、EAV 模式和节点数量；实际队列与历史始终为空，未填占位媒体或冷恢复回执，不排队、不新增视频质量资格。两轮公开源码、私有控制器与隔离依赖稳定，所有自有服务子进程及端口已收尾。当前文档／元数据版本仍需最终回归，不将这项界面资格写成全仓或人工验收。

原 Face 联合采样确实改变了音频 latent；交付仍使用原源片音频路径，`automatic_accept=false`。不能把 full／cold PCM 相同写成“音频 latent 不变”，也不能声称与重新编码前的 AAC 字节完全相同。

首轮缺少 ONNX Runtime 的失败，以及普通 SaveVideo 的冷输出被严格 H.264 解码拒绝的失败，均保留为失败。新的显式方案不会改判旧坏片、不换动漫检测器或其 0.35 阈值，不升级共享 Python 环境。本轮实际 ONNX Runtime／OpenCV 为任务隔离安装；一般用户仍需安装原 Face 功能所需的可选依赖。

## 构图器

`python -m tools.build_formal_face_anime_isolated_delivery_workflows` 只生成这两张新增图；已存在的不同内容拒绝覆盖。完整图与冷图的其余 API、采样步数、sigma、模型、音频及审图控件必须逐值保持原公开配方。需要新增功能不能靠修改旧图默认值来完成。
