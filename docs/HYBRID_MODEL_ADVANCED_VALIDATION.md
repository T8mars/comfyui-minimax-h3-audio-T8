# MiniMax H3 Hybrid Model Advanced 验证报告

验证日期：2026-08-12
插件版本：1.14.0
ComfyUI：`cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44` / 0.31.0
GPU：RTX 4060 Ti 16GiB，DynamicVRAM，2GiB headroom

## 结论

P0 机械路线已经成立：项目可以在不修改原有56个节点、不全局patch H3 core、不生成完整融合
checkpoint、运行时不加载第二个完整MODEL的前提下，把精确Ref2VA pruned checkpoint的选定
AdaLN模态行曲线重基后，以小型artifact应用到ComfyUI stock FL2VA loader输出的MODEL。

这不证明“两个模型优点兼得”。当前ComfyUI的参考与目标共享video/audio AdaLN tag，静态行替换
也会改变目标流。后续视觉、音频与混合参考共15路Stock20真实生成全部成功，并筛出值得继续扩大
样本的候选；但仍只有单素材/单seed，不能命名为最佳、去油、高参考遵循或16GB安全档。精确矩阵
最差显存余量只有41.34MiB，明确未过512MiB门槛。

## 精确输入与artifact

| 角色 | 文件 | Bytes | SHA-256 |
|---|---|---:|---|
| quality base | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20,970,379,616 | `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a` |
| reference overlay | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 20,970,379,616 | `9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779` |

两者均为932 tensors，key/shape/dtype合同一致。FL2VA与Ref2VA的`adaln_t_table` SHA-256分别为
`ac8727cdec52137c73878d004de5bd2a0e19227e8311e29ab3b68f328310e34e`和
`c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7`。

默认profile：`blocks_25_49_video_audio_exp`。

- artifact：`models/h3_hybrid_artifacts/h3_t8_blocks_25_49_video_audio_exp_191cf2162f26c4cb.safetensors`；
- artifact SHA-256：`fcb4cdcc5dbb9742af6163654da7246783dbffb045c6dd32d99a6c42772cd0ab`；
- payload：29,030,400 bytes（27.69MiB）；
- 25 blocks × video/audio × weight/bias = 100个offset-set操作；
- 完整pair检查约33.59秒，首次artifact构建约0.255秒，重复构建命中内容寻址缓存；
- 曲线仿射基底rank 9、condition number约15500.75；
- table相对误差`4.9342639e-5`，最大绝对误差`4.9467484e-5`；
- 所有保存slice的最大有效调制重构误差`2.3020626e-5`，低于`1e-4`门槛。

artifact和sidecar使用临时文件、`fsync`与原子替换；同路径存在孤立artifact/sidecar时拒绝覆盖，
manifest的source hash、recipe、block、modality、offset、shape、dtype、操作类型和payload大小均再次
校验。未知full、W4A8、混合量化、自由glob与final/output-head替换不属于P0。

## MODEL / DynamicVRAM 契约

真实加载默认artifact时：

- stock `comfy.sd.load_diffusion_model()`继续负责quality base；
- DynamicVRAM输出为原生`ModelPatcherDynamic`，保留`cached_patcher_init`；
- 模型记录50个patched tensor key、100个patch entry；
- 普通clone、non-dynamic delegate与同设备deepclone均保留100个entry及同一provenance attachment；
- legacy加载进程只映射一份base checkpoint；DynamicVRAM进程加载后RSS约从798.85MiB增至
  1004.01MiB，不能把RSS等同于完整系统内存上界；
- 对block 25的weight/bias逐张量复算确认：video/audio行逐位等于artifact target，text行逐位等于
  原FL2VA base，且video/audio两行都确实发生变化。

`base_only`直接调用相同stock loader且不应用artifact。Loader架构不允许把已有MODEL传入后偷偷
覆盖；LoRA只能在Hybrid Loader后连接。当前没有证实任何Turbo LoRA已经在Hybrid语义上校准。

## 真实GPU机械链

固定参考图`input/10A.jpg`、256×256、22帧、1步、Ref2VA conditioning、双时钟Euler、无LoRA：

- 完整Inspector → Builder → Hybrid Loader → Conditioning → Sampler → AV Decode → MP4成功；
- 服务日志明确报告`MiniMaxH3 ... 50 patches attached`；
- 输出22帧、256×256、24fps；AAC为32kHz双声道；
- 整卡基线约1156.5MiB、峰值12872.8MiB、峰值增量11716.3MiB、最低余量约3507.2MiB；
- VideoHelperSuite输出音频流约0.896秒，视频约0.916667秒，这是现有保存链的AAC边界差异，
  不是Hybrid latent时钟通过sample级精确性的证据。

## 首个Stock20质量pilot

控制量：同一`10A.jpg`、prompt、seed `2608125201`、736×416、124帧、Stock20、
`dual_clock_euler + native_flow`、shift 12/3、无LoRA、每路运行前显式free。

| Treatment | 成功 | 时长 | 整卡峰值 | 最低余量 |
|---|---:|---:|---:|---:|
| FL2VA pruned base_only | 是 | 233.86s | 12932.8MiB | 3447.2MiB |
| Hybrid blocks25–49 video+audio | 是 | 228.34s | 12684.9MiB | 3695.1MiB |
| stock Ref2VA pruned | 是 | 230.58s | 12883.2MiB | 3496.8MiB |

Hybrid比FL2VA单次峰值低约247.9MiB，小于项目256MiB“实质差异”阈值；不能称为省显存。
这次早期pilot的三路都超过512MiB，但后续独立顺序矩阵在同一16GiB设备测得41～352MiB的
Hybrid余量，说明加载/VBAR状态会显著影响整卡峰值。安全结论必须采用后续更差值，不能沿用这次
pilot的3.4GiB余量。

无参考对齐的像素/频谱代理：

| 指标 | FL2VA | Hybrid | Ref2VA |
|---|---:|---:|---:|
| video high-pass mean | 0.008135 | 0.008182 | 0.008362 |
| median Laplacian variance | 0.000291 | 0.000279 | 0.000180 |
| median temporal MAD | 0.01111 | 0.01168 | 0.01818 |
| mean saturation | 0.1769 | 0.1699 | 0.3359 |
| audio RMS | 0.1987 | 0.1982 | 0.1263 |
| clipping fraction | 0 | 0 | 0 |

逐帧RGB差异为：FL2VA?Hybrid `13.31/255`，FL2VA?Ref2VA `91.08/255`，
Hybrid?Ref2VA `90.58/255`。接触表中FL2VA与Hybrid保持相近的中近景侧脸运动，stock Ref2VA
变为极近景皮肤/耳部镜头；这只说明默认Hybrid在本例更接近FL2VA输出分布。

InsightFace在12个抽样帧中只检测到FL2VA 4帧、Hybrid 4帧、Ref2VA 2帧，覆盖率不足，不能作为
身份验收。可比较的低覆盖信号里Hybrid来源余弦中位数约0.317、FL2VA约0.250；不能外推。
faster-whisper在FL2VA音频中报告一次非要求的“Thank you for watching!”，Hybrid与Ref2VA没有
返回语音段；这是单次ASR研究信号，不证明Hybrid普遍抑制多余台词。

## Conditioning感知的最小模态路由

Inspector新增`auto_match_reference_modalities_exp`，可选连接稳定Conditioning的`positive`：

| 实际额外reference | 解析profile |
|---|---|
| image或video | `blocks_25_49_video_exp` |
| 独立audio | `blocks_25_49_audio_exp` |
| video_audio，或visual+audio混合 | `blocks_25_49_video_audio_exp` |

首尾/中间keyframe的`t8_keyframe_latent`不是Ref2VA额外reference，不会误触发路由；没有额外
reference、未知kind或损坏Conditioning时直接拒绝。这个功能只减少不相关模态行的修改范围，
不自动选择block区间，也不声称选中了质量最佳profile。显式profile的旧输入与默认值不变。

## 可恢复顺序矩阵与质量报告

`tools/run_hybrid_model_matrix.py`固定验证一个Inspector/Builder/Loader、一个稳定Conditioning、
一个Stock20双时钟采样器、一个seed和一个输出节点；检测LoRA、第二UNET、无reference或非20步
会在排队前拒绝。每个treatment前调用专用Comfy服务的全局free，严格顺序运行FL2VA control、
stock Ref2VA control和显式Hybrid recipes，不在GPU同时常驻两个完整MODEL。

manifest用目录锁、原子写、workflow/output SHA-256和control fingerprint支持`--resume`；恢复时输出
hash不一致会重跑。工具生成随机盲化MP4、五帧接触表、评价CSV与`matrix_summary.json/csv`。
可选ASR、InsightFace、WavLM都只读取用户显式提供的本地目录，不自动下载；说话人/人脸余弦没有
通用阈值。总结文件固定写`not_ranked_requires_blind_review_and_broader_matrix`，不会按代理自动选冠军。

## 后续真实Stock20矩阵

以下矩阵均为RTX 4060 Ti 16GiB、124帧、`dual_clock_euler + native_flow`、无LoRA，每路前free。
整卡总量按同一服务报告的16379.5MiB计算。不同顺序中的FL2VA/Ref2VA MP4容器hash不同，但
解码BGR与PCM SHA-256逐位相同，证明control可复现。

### 视觉参考profile筛选

同一`10A.jpg`、prompt与seed `2608125201`：

| Treatment | 峰值MiB | 余量MiB | Face cosine中位数 | Audio HF≥8kHz dB | 观察 |
|---|---:|---:|---:|---:|---|
| FL2VA control | 15426.21 | 953.29 | 0.416 | -27.61 | ASR出现非要求短句 |
| Ref2VA control | 16318.72 | 60.78 | 0.032（覆盖25%） | -73.77 | 极近景，身份代理不可用 |
| 25–49 video+audio | 16253.39 | 126.11 | 0.479 | -50.47 | 接近FL构图，音频高频明显下降 |
| 25–49 all modalities | 16337.62 | 41.88 | 0.495 | -50.79 | face代理最高，余量最低 |
| 0–49 video+audio | 16027.92 | 351.58 | 0.440 | -35.49 | 高频/时序代理更高，身份低于25–49 |
| 25–49 video-only | 16337.62 | 41.88 | 0.476 | -28.09 | 身份接近video+audio，音频频谱更接近FL |

单案例中`video-only`是更合理的视觉参考Pareto候选：face cosine与video+audio接近，同时少改变
音频频谱；这正是最小模态路由的产品依据。它仍未经过多素材、多seed和人工盲评，不能升稳定。

### 独立音频参考

许可LibriSpeech speaker 61参考、目标句、seed `2608126201`、32×32暗画布：

| Treatment | 峰值MiB | 余量MiB | WavLM cosine | ASR | Audio HF≥8kHz dB |
|---|---:|---:|---:|---|---:|
| FL2VA control | 16250.95 | 128.55 | 0.914 | 完整目标 | -53.77 |
| Ref2VA control | 16250.95 | 128.55 | 0.945 | 完整目标 | -48.00 |
| 25–49 audio-only | 16266.61 | 112.89 | 0.928 | 完整目标 | -49.11 |
| 25–49 video+audio | 16250.95 | 128.55 | 0.927 | 完整目标 | -51.53 |

四路归一化词序列均与目标一致。audio-only把单参考余弦从FL2VA的0.914提高到0.928，但仍低于
stock Ref2VA的0.945；它与video+audio身份信号相近而少改一个无关视觉模态，因此保留为更小的
audio参考候选。单说话人/单句余弦不能证明高保真克隆。

### 图像+音频混合参考

同一`10A.jpg`、speaker 61、目标句和seed `2608127201`、736×416：

| Treatment | 峰值MiB | 余量MiB | Face cosine | WavLM cosine | ASR |
|---|---:|---:|---:|---:|---|
| FL2VA control | 16310.09 | 69.41 | 0.449 | 0.467 | 完整目标 |
| Ref2VA control | 16338.16 | 41.34 | 0.443 | 0.945 | 完整目标 |
| 25–49 video+audio | 16338.16 | 41.34 | 0.523 | 0.868 | 完整目标 |

Hybrid在这个样本的12/12抽样帧都检测到脸，face cosine高于两个control；WavLM身份显著高于
FL2VA但低于Ref2VA，接触表保持自然中景而非Ref2VA对照的黑衣远景。这是当前最有价值的Pareto
信号，但只有一个素材/seed，且最低余量41.34MiB；结论仍是“扩大验证”，不是“默认最佳”。

### 资源结论

后续15条run record全部完成、124帧、5.152秒32kHz双声道、无clipping/OOM；但除视觉矩阵中
一次FL2VA control外，绝大多数精确测量低于512MiB余量，最差41.34MiB。加载顺序/VBAR状态能让
同一control峰值明显变化，因此不得以早期pilot的较低峰值覆盖后续坏值。当前所有profile继续
标`Advanced/EXP`且禁止宣称16GB安全、省显存或绝不OOM。

## 自动化与兼容回归

- 新增严格pair、曲线重基、artifact原子性/缓存/篡改拒绝、offset-set、patch冲突、base_only与
  API/前端工作流结构测试；
- Hybrid模型与矩阵工具定向33 tests通过；全项目305 tests通过；
- Ruff使用系统可用二进制运行并通过；项目嵌入式Python的Ruff wrapper仍指向已删除的用户目录，
  这是既有环境问题；
- `compileall`、63个示例JSON与meta/features解析和`git diff --check`通过；
- 隔离ComfyUI白名单实例导入成功，`/object_info`确认59个T8节点和3个新Advanced节点的真实契约；
- 画布工作流与`user/default/workflows/MiniMax H3 T8/`中的安装副本SHA-256一致；
- 旧56个节点ID/顺序保持原样，新3个Advanced节点只追加在末尾；稳定`sampling.py`未修改。

## 尚未完成与否决门

- 3素材类×3seed、多图、参考视频、有声参考视频、首尾+refs和完成人工盲评；
- 0～49 all-modal及更多block范围尚未筛选；现有video-only/audio-only/混合参考都只有单素材单seed；
- stable双时钟之外的非双时钟、Turbo LoRA、Block Cache、Sage、Long Video与MultiKeyframe组合；
- 3冷3暖、连续三任务、profile交替缓存、结束15秒回落和跨GPU；
- 真正按segment kind区分target/reference的双AdaLN前向路由。

没有同时通过参考遵循、主观画质/油感、音频、副作用和资源矩阵的profile前，示例保持
`Advanced/EXP`。当前自动功能只做Conditioning模态匹配，不设置“最佳混合”或自动质量路由。
