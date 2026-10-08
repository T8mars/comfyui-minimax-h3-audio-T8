# 外部 CFR 视频 → 独立 LTX 精修（H05 Q05，EXP）

这是新增的独立模板，不替换旧 RGB／learned → LTX 工作流或采样默认。
来源可以不是 H3 生成的视频；它不构成原生 H3 ancestor、teacher latent
或可移植 Cold 缓存。当前仅一个固定 CFR／立体声素材完成机械验证，待集中人审。

## 使用

导入 [独立短名工作流](../examples/workflows/81-h05-ltx-source/H05_LTX_Source_EXP.json)，
把 LoadVideo 的占位名换成有权使用、长度超过 6.2 秒的 CFR 立体声素材。
模板使用现有 LTX-2.5 transformer、Gemma、distilled LoRA、full video VAE、
x2 latent upscaler 和 TAEHV；按画布各加载器选择本机文件。不自动安装依赖，
当前默认输出需要已安装的 VideoHelperSuite。

接线分别可见：读取／有界切片 → SourceClock → SourceConform → RGB 准备
→ full LTX VAE Encode → x2 latent lift → Stage Bind → 三步 Sampler
→ Stage Audit → EAV／Relay Audit → TAEHV Decode → Trim → 独立输出。
EAV 与 Prompt Relay 外置，缺省均 `report_only`，可独立检查报告。
本例 Relay 只有一个事件，不是多事件偏置或 EAV 实际施加资格。

图中 Video Slice 在 resident decode 前选择 0–6.2 秒；SourceClock 观察真实
PTS，Conform 从原片 1 秒映射为 24fps／124 帧，LTX 按既有 8n+1 规则
保留 121 帧，最终裁为 120 帧／5 秒。不是 VFR、任意原点或所有音频 PTS
对齐的资格。空间准备可能按比例中心裁切，需人工检查内容。

选定 full LTX VAE 的编码网格为 32 像素，模板先编码半尺寸再 x2；
所以目标宽高须为 **64 的倍数**，当前为 256×448，编码 128×224。
生成工具在创建图前拒绝错误尺寸，不静默取整，也不更改旧准备节点合同。

`tools/prepare_h05_ltx_source_workflow.py --server http://127.0.0.1:<port>
--output <不存在的新文件.json>` 可生成同一模板；它不 Queue、不覆盖文件。
可显式选择现有 `--load-mode`，旧加载默认仍 `legacy_default`。

## 音频与输出边界

原音频走相同 1 秒原点的 TrimAudioDuration → Stage Audit → OutputTrim
→ 输出节点，保留其输入采样率和实际立体声，不使用 Conform 的 32k 音频、
LTX audio VAE 或生成的人声替换。输入源与输出编码的精确性须分开。

默认新模板明确选择 **VHS 外置 H.264／AAC 输出**；VHS 接受 float32 PCM，
不会由本项目悄悄归一化源音量。AAC 解码 PCM 不等于原始 PCM，且可能有
包边界尾差。本次 5 秒源音频 220500 samples，实际 AAC 解码 220160，
少 340 samples（约 7.71ms）；视频严格为 120 帧／5 秒。不声明音轨精确
5 秒、无损保留、声纹锁定或广义同步资格。

原素材解码后有 89 个样本超过单位幅度，峰值 1.05774；原 H3 Safe AV
严格出口正确拒绝。该失败原样保留，没有放宽守卫。生成器的
`--output-backend safe_av` 是显式选择，仍会拒绝超范围输入，不自动降级。
Core 的 FLAC 预览也不能完整表达此 float 源的超幅度样本；不要用预览
声称实际送给 VHS 的 float PCM 精确相同。对这类源必要时由用户明确
选择音量处理；本模板不代作决定。

VHS 原生控件使用 **按名称保存的字典**，避免它的旧列表迁移把 pix_fmt／
crf 错装为 pingpong／save_output。只改变新模板序列化，不改 VHS 或共享
工作流转换器。

## 本地验证范围

修正网格并使用上述显式出口后，真实画布另存、重开、Run 一次成功：
42 执行节点／86 连线，三步 LTX，24.753 秒；120 帧256×448／24fps
完整视频解码、全部视频 PTS、44.1k 立体声有限非零音频与来源身份已核。
实际 latent `[1,128,16,14,8]`，48 个视频块各观察三次，EAV／Relay
零施加。尝试复用旧候选时实际 Core 未缓存 sampler，确实新增三步，
不是零采样出口恢复；之后完整媒体审计没有新增 GPU。

该例不是 NVIDIA 原 benchmark、普遍无损精修或性能承诺。旧尺寸失败、
Safe AV 拒绝与旧 VHS 参数错位均保留。指定人像／动作／音乐质量尚未
获人工通过；不据此批准所有外部来源、长片、VFR 或全部双采路线。
