# 独立 Qwen 参考视图（EXP）

接线：原生参考包 Create／Load → Reference Route → **Qwen View** → Apply 或 Relay Apply。
仍可使用普通／分离 LOW、learned upscaler、HIGH；EAV／Prompt Relay 继续外置。
此功能只处理选中的图片参考，不处理视频、声音、首尾帧或已编码 CONDITIONING。

`mode=original` 缺省精确走旧路径。明确选择 `image_short_edge` 才派生 Qwen 输入：

- `short_edge` 是短边目标，不是长边；默认512。
- `max_long_edge` 默认1024，`max_pixels` 默认524288，是每张图片输入像素上限。
- 三者同时限制，取不放大的最小缩放比例，整数向下取整、不裁切。长边／像素限制可能使短边低于目标。
- 小图原样保留，不再量化。真正缩图使用现有 Core RGB24 Lanczos，可能损失细节，不是无损。
- 预算会使极端长宽比任一边低于32时明确报错；请放宽预算，不静默拉伸。

原参考包的 RGB、VAE latent、producer、角色与声音保持。Apply 校验原包和实际 VAE，
只全新编码一次 Qwen；新原生 text recipe 保留该视图供逐窗重新编码，不切旧 embedding。
原包可在 LOW／HIGH 使用；不同 Qwen 选择显式分别接线，不改变旧节点默认。

报告绑定实际原包／原像素、策略、派生像素 SHA 和真实输入尺寸。`routed_picture_ordinal`
是 Route 内编号；Apply 的 `native_picture_ordinals` 包含首尾帧带来的偏移。
已有前缀缓存仍按真实输入像素内容区分；不新增 LRU，不给 opaque CLIP 伪造可移植身份。
Stage／text recipe 元数据绑定独立视图 receipt，原来的 VAE producer 不改成缩图 producer。

像素预算不是任意 CLIP 内部 processor 的最终 token 保证。Core 可能再次网格对齐，
或放大小图；不能把输入尺寸冒称 processor grid。当前 API 不暴露每图最终 grid／token 时
报告为 `null`。若编码结果返回真实 `minimax_token_tags`，另报告整段 tag 数，包含视觉块
分隔符，不能当每图 patch 数、加速收益或 encoder 身份证明。

普通用户加载器与未知补丁组合仍允许运行／委托；实际包损坏、producer 不符、视图
输入或 receipt 在编码中改变仍报错。`original` 可恢复旧行为，原包不会被改写。

当前属于本地开发：CPU 数学／契约核验与 GPU／原生画布／最终整片人工审核分开记录。
恢复验收已完成一条原生画布5秒4+4成片：原VAE参考768×960，Qwen独立输入512×640，
learned二采1.2×实际512×288；120帧、24fps、完整视频和原生音轨均可解码。
Full／Cold图已经原生另存、重开并验证接线，真实参考包和LOW完成缓存可冷读；Cold未额外跑GPU。
此前193项CPU不重复；2026-10-06本项A1完整成片已获用户人工审核通过，不借其他项目的人审。
人工通过仅覆盖此固定单图、独立Qwen视图、5秒4+4案例；Cold仅有真实缓存冷读和接线资格，
没有额外GPU成片，不扩大为所有素材、多人物或多窗对白资格。该N01增量本地完成，尚未发布。
没有普遍速度、显存、画质、音频保持或多图小脸／文字资格；缩图也可能改变生成声音。
