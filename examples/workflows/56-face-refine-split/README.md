# S24 Face Refine 分离 Stage（八十四张 EXP，路线未完成）

最新 Face／Stage／存储／浏览器语义及原 Studio 状态机等16文件联合 CPU 回归468通过／0失败跳过；M5采样调用门此前103/103，本批未重跑。此数仅为机械回归，不是原尺寸真实权重或人工画音验收。

双／三人物无效果保存图已分别在 tiny CPU Core 中完成各自独立 Stage 保存、原 AV 解码／Parity 回贴、默认 `reject` 顺序合成及9帧原生 MP4；全新 Python/Core 冷图只读每个 job 的正确回执，解码画音与完整图一致。两／三人 ROI 特意空间分离，错接回执和重复合成遮罩会拒绝；不能据此认定 Studio、全部效果图、真实权重原尺寸或人审通过。

Studio Serial 无效果保存图的**默认预览**路径亦已在 tiny CPU Core 完成 Start→Commit→Compose 与原生 MP4，冷图在全新进程只读 Stage 而不重采。`review_only`／`preview_only`／未确认时，持久清单保持 pending、0接受，最终画面仍是原片；未确认却请求接受会拒绝。另有两窗口显式决策测试：窗口0确认后接受，窗口1独立 Stage 冷预览后拒绝，错误窗口回执拒绝；最终在全新进程用原有 Compose-only 保存图仅合成，画音与决策完成时一致。测试仅为无外置效果的小尺寸 CPU 替身；没有改公开图默认值，也不覆盖任意多窗口、其它效果图或真人审片。

标准 Face 的外置 EAV `full_save→cold_delivery` 保存图另有小尺寸 Core 完整媒体测试：`report_only` 与显式 `apply_exp` 两模式均实际经过独立 EAV Config／Apply／Audit 并保存 Stage；后者实测增益大于1，但审计仍标明未获画质接受。各自冷图只读 Stage、不重采也不重新运行 EAV，跨新 Python/Core 的解码画音与对应完整图一致。极小随机测试模型在公开默认 `tau=4` 下超过 EAV 原有 `g_hard_limit=1.5` 而安全拒绝；测试执行副本只把 `tau` 改为 0.2，公开 JSON 未改。这不代表真实权重下 `apply_exp` 画质更好，或其它 Face 配方／效果组合已经整图验收。

本目录七种配方（标准／动漫、Parity、SAM3.1 双人／三人、Manual Window、Studio Serial）×四种效果（无、外置 EAV、外置 Prompt Relay、组合）各有三张图：原独立 Stage、`full_save` 完整采样并保存每个修复 job、`cold_delivery` 按各自路径＋SHA 加载冻结结果后交付，共八十四张。全部是旧公开工作流的新增副本；原素材输入、逐帧遮罩、窗口审图／提交或多人顺序合成及源视频音轨交付路径不变，旧图没有被替换。

所有实时采样图的 EAV 是独立 Config→Apply→实际调用 Audit，默认 `report_only`，不代表开启增强或画质验收。标准／动漫的外置 Relay Plan、Conditioning、FaceRelayBind 严格校验完整源片计划、成对 MODEL／CONDITIONING、裁剪 AV 和锁定源音频。Parity、多人和窗口的每个修复 job 有独立局部 Ref2VA Relay Plan／Conditioning／FaceLocalRelayBind；**其 Plan 时间轴只针对该局部修复窗口，不能直接当作整条父片的全局 Plan 自动投影**。小尺寸定向采样观察到真实调用，但未以公开图跑原尺寸成片。当前精确 Core `er_sde` solver／首 sigma 函数源码、默认采样器选项、实际回调与前向数均匹配时，StageResult 才能获得可移植完成身份；源码或选项变化退回未验证。三类局部 Face 已分别通过小尺寸保存→全新 CPU 进程 Load→当前来源审计；这**不是**原尺寸、最终画质或通用 Core 版本资格。

使用 `full_save` 时，每个 Stage Save 都会新建文件；请把每个 job 返回的 `artifact_path` 和 `artifact_sha256` 分别粘贴到同配方同效果 `cold_delivery` 的对应 Stage Load。冷图没有 Stage Sampler，不重采冻结 job；它仍重新构建 Face 来源／裁剪 AV 并逐 job 审计，Relay 配方可能因此仍加载条件侧模型，但不会执行扩散采样。冷图不重新执行 EAV Audit；实际 EAV/Relay 证据留在已冻结的 StageResult 回执。路径和 SHA 的空占位不能直接排队。多人 job 及窗口审图／显式接受顺序须保留，冷恢复不等于自动接受。旧 Parity／多人／窗口示例引用当前未注册的 `MiniMaxH3SigmaShift`；新增图用当前 DualClock MODEL 设置、外置 `er_sde`／`simple`，**不声称与缺失旧节点数值等价**。导入后须替换源视频、身份图片、模型等素材。新增56图 Core 静态校验和构图器／布局核对113项通过，真实 Chromium 原生 Save As／重开56/56通过（21张完整效果保存图可见编辑）；此前28张独立图同样通过且保持原字节。标准 Face 无效果的保存／冷图已有小尺寸 CPU Core 原生视频尾链验证：全新 Core 冷图只读 Stage，5帧 H.264/AAC 解码画音与完整图一致。Manual Window 无效果图亦在严格 MANUAL512 REL 基线门和默认 `preview_only` 审图下完成22帧原生 MP4 保存／新 Core 冷交付，未确认时最终画面严格保持原片，解码画音相等。Parity／双人物／Manual 的 Stage 与来源审计子图也实跑，双人物回执交叉接错及窗口音轨变化拒绝；这些替身测试并非全部84张整图执行。浏览器没有排队采样。真实权重原尺寸画音、旧新数值对照与人审仍待。
