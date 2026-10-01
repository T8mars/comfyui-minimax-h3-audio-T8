# 导演台双采样与分阶段多 LoRA（2026-09-21）

## 使用

在导演台右上角打开“模型与采样”，选择全片默认或当前镜头独立，再选“双采样 · 高清细化”。扩散底模、文本编码器和视频／音频 VAE 只选择一次。一采与二采分别维护有序 LoRA 列表；每条可搜索文件、调强度、启用、上下移动、移除。可一次性“复制一采列表到二采”，但之后两列独立。最终输出选择 MP；实际 LOW/HIGH 宽高以预检回显为准。点“应用”才修改项目；“取消”或 Esc 丢弃窗口草稿。旧项目仍默认单采，不会自动套双采或隐藏加载 Turbo。

全片默认只影响跟随全片的镜头。进入“当前镜头独立”会复制当时的有效配置，此后不随全片改变。共用底模始终属于项目全局，不在两阶段重复选择。导出/保存项目格式从 version 1 升为 2；v0/v1 自动迁移为单采，未知版本拒绝执行。两个采样模式的窗口草稿分别保留。

生成完成后，当前镜头自动切到“镜头成片”页签并在编辑区直接播放；每张已完成镜头卡显示“▶ 有成片”，点卡片即可回看。刷新／重开项目通过服务端持久请求回执恢复输出，Core 重启前已经缓存完成任务的媒体元数据；更早没有回执的 MP4 按导演台实际使用的 `T8_Director/<项目 UUID 前 8 位>/<镜头 UUID 前 8 位>` 前缀只读找回。不把输出视频混进输入素材、首尾帧或提示词引用。生成失败保留已有成片，不自动重采样。

## 编译合同

双采编译为一个 Core DAG：共用一个 UNETLoader，两条从原始 MODEL 分叉的 LoRA 链；LOW 和 HIGH 独立条件，LOW 8 步 DualClock 配 4 步 parity sigma，取一采 denoised output 经学习型 3D latent upscaler；HIGH 条件用放大器的实际尺寸重建，再 Reconcile 与 DetailMixer 完成 4 步细化。只有二采输出进入 AVDecode、Trim、SafeAVSave。两阶段共用同一镜头 seed 数值但分别有 RandomNoise 实例；一采声音不直接交付。首尾图片均从原素材分别准备低、高尺寸，不把低图再放大。音频驱动复用同一裁剪后的录音条件，交付仍取 HIGH 条件的原音轨；短录音为满足最小时长而左补的上下文，在交付时使用 AudioWindow 的精确偏移裁切音画，不能从 0 秒截取。默认 Tail/Bias/STG/Restart 关闭；FastH3 V2 与此标准 4+4 路线互斥。

最终尺寸由项目画幅、目标总像素和现有 `learned_upscale_geometry` 算出 32 对齐的 LOW/HIGH 计划。启用的 LoRA 缺文件、强度越界、放大模型缺失会明确失败；禁用条目不加载，重复条目按列表顺序执行，0 强度仍读取文件。未识别自训 LoRA 不按文件名一律禁止，兼容加载器的真实 patch 回执／日志仍是判断其实际映射的依据，界面计数只表示“已配置”，不表示质量已验收。

“生成全部”在提交前冻结整个 project 快照并逐镜生成；每次提交带 UUID 请求 ID。服务端将 UUID、快照指纹、Core prompt ID 写入用户目录下 `t8_director/requests/`；相同请求重试返回原 prompt，不会重复排队。若提交时的状态无法判定，会保留 reservation 并要求查原 prompt，不冒险重发。这不是跨 Core 重启的 latent 阶段续算。

## 验证范围

- 原单采／项目迁移／图编译、真实 Core schema、两阶段 LoRA 独立分叉、录音绑定、0.4／0.5 MP 常用画幅几何、幂等单测和 Chromium 弹窗事务交互均有自动回归。
- 本机 8189 Core 的实际 GPU 样例：T2VA、4 秒、最终 832×480，LOW 448×256，两阶段各配置 2 条 LoRA（EMA B 强度 1 + 另一条强度 0），prompt `39fc84e3-ad95-4005-ac8e-ba4b702eb2bb` 已成功，文件 SHA256 `c5f7a5ec30d8bf2682dc0df0bb07deaa032b9a2f534def22283ae2eca1c4bf2c`，H.264 96 帧／24fps、AAC 32kHz、时长 4.000 秒。正式 `/generate` 的相同配置 `fdda08a6-fd57-4097-bb3b-66fc0f0dee18` 亦成功，重试相同 UUID 未重复排队。
- **重要反证：**该两次样例明确选择 `minimax_h3_fl2va_pruned_int8_convrot.safetensors`，Core 日志出现 `ERROR lora diffusion_model.blocks.*.adaln_proj.linear.weight shape ... invalid`。成功的视频只证明图执行与媒体交付，**不能证明这组裁剪底模与 EMA LoRA 的所有 patch 生效**；强度 0 条目同样不能被当作画质改善。用户须查看真实 LoRA patch 日志／回执。已验收长片示例使用的是非 pruned `minimax_h3_fl2va_int8_convrot.safetensors`，但其资格不能直接转移到本次新导演台；该组合另复测。
- 对照：非 pruned 完整 INT8 FL2VA 底模 `minimax_h3_fl2va_int8_convrot.safetensors` 配 EMA B LoRA 强度 1、另一条强度 0，正式 Director prompt `1de221ee-eaac-4799-86d5-1f4be850b248` 成功，LOW/HIGH 各有 `259 patches attached`，该次模型加载后的日志无 LoRA 形状错误，输出 H.264 96 帧／AAC 32kHz／4.000 秒。第二条 LoRA 在 LOW/HIGH 分别设为 **0.1/0.2 非零强度**的对照 prompt `e2e5a57c-5539-4f06-9261-6ed85b3dfb10` 也成功，日志未见 LoRA 形状错误；这证明独立参数的两条 LoRA 可以真实排队执行，但不替代画质人审，也不能概括任意用户 LoRA。
- 真实 Core GPU 八路线覆盖：首帧 I2VA `f581f424-f84b-465d-b31f-beb9c4b870db`、首尾 FL2VA `808c0292-c8b0-4661-938d-1af642fb8780`、参考 Ref2VA `eb7f6021-d546-4111-b677-03727ac8d599`、混合 `03613756-385c-4363-8647-d5a7fca5e872`、原音驱动 `fd630d65-e7f1-4732-acee-2d48e513f6f5`、参考音色 `a696c4cf-2632-4d3d-8ee9-33ed98cebacb`、Relay `316378f7-18b0-456d-bc84-ac089e141fc7`、Bridge `99611747-87fc-474b-98c7-d168c22f1600`，均返回成片。**这八项使用的是上述有形状错误的裁剪版 FL2VA 底模**，仅证明路由、AV 编译和输出可执行，不给 LoRA 生效或画质背书。
- 原音驱动的额外反证与修复：旧成片 `fd630d65-e7f1-4732-acee-2d48e513f6f5` 从 0 秒裁切了带左补的音频上下文，与上传录音零偏移相关系数仅约 0.0048。修正交付 Trim 从 AudioWindow 输出的 `final_trim_start_seconds` 开始后，以完整 INT8 底模重跑 prompt `a3d4c884-b908-419c-80b0-c4dbbd30d515`，成片 SHA256 `d5949dcbf0a33fafe634ba790248845d62a56b08b5159ea8fb1782b4881363e9`，H.264 96 帧、AAC 16kHz、4.000 秒；与本次上传的 64,000 个原录音 PCM 样本在零延迟相关系数 **0.9999868**，最佳延迟 0 样本。AAC 尾包解码多出 512 样本是容器填充，未把它误认成新增语音；机器相关性不等于真人口型审片。
- 结果找回：正式 8189 Core 重启前补记 11 条成功任务的媒体回执，重启后 `/results/{project_id}` 实际返回对应 MP4；新增成片页签与镜头卡在真实 Chromium 里可打开旧任务并播放。修复后的第 12 条任务也已缓存终态，持久回执不含视频字节，只保存 Core 输出元数据。
- 用户原保存项目 `e6340d39-4f66-4b9c-a20e-364ee943e925` 是无回执旧成片的实测用例：3 镜中 2 镜在磁盘有 MP4，正式结果接口仅找回这 2 镜；真实 Chromium 打开此项目后两张卡显示“有成片”，点击第一镜在本页播放器加载 `eeee6795_00001_.mp4`，页面错误 0。未完成的第二镜没有假结果。
- 以上证明该配置的真实执行与媒体完整性，不等于全部任务类型的 GPU 画质、人物一致性、口型、连续长片接缝或任意 LoRA 组合获得真人验收。其他配置变更后仍需单独审片。

旧单采 API 图、旧项目默认和保存的原生示例不迁移；本次未改 Long Video 接缝、条件音频时钟、既有 3D 放大器算法或 Core 模型权重。仅修正导演台原音驱动的最终交付裁切起点。

## 跨 Core 批次恢复实测（2026-09-22）

开发树通过 `tools/director_batch_restart_probe.py` 完成独占 GPU 两镜验收：两个自有 Core 共用隔离的 user/output 目录，首镜完整交付后硬终止进程，再启动新 Core。PID 从 2616 变为 35072，Core epoch 同时变化；新 Core 无首镜历史，读取持久批次仍只恢复状态，不自动提交。用户显式继续的等价 API 操作只提交第二镜，新 Core 历史仅含第二个 prompt，首镜输出 SHA 保持不变、未重算，最终两镜完成且队列为空。

- 批次：`cfeb5801-3185-47e5-bb3a-b25bc3b7770c`；首镜 prompt `fbadd7a8-b734-4cb8-b707-aeab7b9f92f3`，次镜 prompt `0966d558-eb6b-4b4a-a5b4-64be2f9a1a7b`。
- 反例：仅篡改自有首镜回执指纹或媒体字节，状态均变为 `needs_review`，继续接口返回 HTTP 409，队列仍为空；恢复原始字节后才能继续。没有改用户项目或历史媒体。
- 两镜使用完整 INT8 FL2VA、EMA B 单采基线，各输出 448×448、24 帧／1 秒视频；完整解码音频各 32,768 样本／1.024 秒（含 AAC 尾包填充）。首镜 SHA256 `0853ef1721280e3ba3ce0e7c833c3b737bae481e7cac8b8dd47d10b7a2ee3898`，次镜 `0dd3c9d00a1de7a680c601cabc18562a316b1be958a783190fe79fe2a0e5fd35`。
- 本地证据目录 `artifacts/development/director-batch-restart-453e9b6570/` 保留结果、坏回执／媒体拒绝响应、源码身份、日志与停机回执；两个自有进程均已退出、无剩余子进程，端口 8871 无监听。控制器退出码 0；Core 停机码 1 来自验收要求的 Windows 强制终止，不是生成失败。

结论仅为该固定配置的跨进程批次恢复与完整音画交付机械验收通过（`machine_pass_human_review_pending`），不是中途 latent 续算、任意模型／双采组合资格，也不代表画质、声音、口型或连续长片接缝已获人审。证据与媒体仅本地保存，不随源码发布。
