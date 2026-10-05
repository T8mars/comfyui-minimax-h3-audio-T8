# Temporal Chunk 窗口对白作用域（EXP，固定v5案例已通过人审）

这是显式选择的新分支。旧632节点、默认值、单窗／旧Chunk及LongVideo执行器不自动换成新策略；旧工作流继续使用原合同。新增节点在 `T8/MiniMax H3/Temporal Dialogue Experimental`。

## 处理什么

多窗联合二采重复使用整段对白时，后面才应出现的句子可能在首窗提前触发，后窗也可能重新说已完成的句子。新条件计划从首窗起按实际snap后的时间区间筛选，而不是继续增加禁止词或删除全部HIGH条件。此解释与反馈对照相符，但没有第三方失败媒体，不能称唯一根因已证明。

输入分三类：

- `global_prompt`：人物／衣着／物品／位置／音色／持续视觉约束，贯穿全片。
- `performance_events_json`：`event_id, cue, start_seconds, end_seconds`，局部动作与表演。按整个render窗口保留，包括只读重叠区。
- `dialogue_events_json`：`event_id, speaker, utterance, start_seconds, end_seconds`。按新可写区间筛选；相同文字的两次合法台词必须使用不同ID。

Global／Performance不要再次夹带整段对白或重复发声命令。系统不会regex猜测、拆解或静默删掉自由文本。时间为全局24fps时间，原请求时间和量化结果同时保留。合法跨缝句保留同ID、原句和全局区间，标为continuation，不重新制造一个句首。

示例对白：

```json
[
  {"event_id":"line1","speaker":"the woman","utterance":"我没看过。","start_seconds":6.2,"end_seconds":7.4},
  {"event_id":"line2","speaker":"the same woman","utterance":"一次都没有？","start_seconds":8.1,"end_seconds":9.2},
  {"event_id":"line3","speaker":"the same woman","utterance":"不敢。","start_seconds":10.1,"end_seconds":10.9}
]
```

窗口编译、continuation文字与有限Relay偏置不是数学上的硬时间隔离，也不保证模型逐字只说一次。仍需完整听看。

## 接线：v5标准联合4+4

`Global + Speech + Performance Plan` 的 `full_prompt_for_LOW` 接原LOW Conditioning，保持一采完整剧情。HIGH的 `Native HIGH Media Recipe` 使用相同媒体、实际HIGH尺寸、完整原生帧数；其新recipe输出接 `Precompile ALL Native Window Text`。Bank同时接实际一采 `denoised_output`、Chunk Plan和对白Plan。

Bank在HIGH前一次性重新进行各窗的native prepare_prompt→tokenize→encode。已编码媒体复用，不每窗重新VAE编码，不裁切旧Qwen contextual embedding；Bridge每份条件只应用一次。原生标签／音频ordinal规范化后重建文字跨度。

分离接线使用原 `Global Learned Lift` 和 `Prepare Full AV Noise`，每项仅一次；各新 `One Joint AV Window` 共用该准备、Bank、NOISE、SAMPLER、refine SIGMAS。窗口0没有previous，后窗接上一窗新 `scoped_window_result`。原v5 `base_window_result`仍可接旧外置EAV／Audit。

整合接线可用新 `Integrated Joint PASS2`，它顺序调用相同的分离窗口，不另开一套数值算法。旧all-in-one节点不变。

一采partial4的音频没有完成，必须联合续采。新分支继续用原v5实际已接受AV前缀、overlap mask=0、全局噪声绝对切片、只读roundoff恢复和exact append；不会冻结半成品音频。

## H16独立新合同

H16新 `H16 One Refined AV Window / Real Prefix` 需要原v4 full-frame joint plan。下一窗实际接回上一窗已精修的音视频，并将完整已接受重叠设为只读。此新EXP合同采用exact append，保留末尾未采样的原source audio padding。

它**不等同于旧H16最后统一crossfade／energy gate**。旧H16默认、`preserve_first_pass`及旧缓存读取源pin没有修改。新H16使用独立typed result、独立保存／恢复格式；旧无ownership缓存不能冒充新接受前缀。

整合H16调用相同分离窗口；旧每segment learned lift与一次全局噪声准备保持。只读前缀这一数值改变仅在新显式合同中发生。

## 已完成音频与外置效果

已经完成且正确的LOW音轨可使用既有video-only／preserve-first-pass HIGH路线。新 `Verified Completed LOW Audio Passthrough` 增加显式交付检查：实际Stage Result完整性、真实回调／NFE、轨迹终点与terminal zero均须成立，原音频tensor精确透传。partial4、只写“completed”的普通latent或未观测完成的receipt不能通过。该交付节点不运行HIGH、不修口型，也不把音轨透传称为画质通过。

外置EAV与Relay都是可选，作用域不依赖开启Bridge／Relay。新Relay根据本窗新native token／规范化span／实际局部AV layout配对MODEL＋CONDITIONING，时钟仅投影一次；不要把全文旧Relay binding贴到新短条件。0／1对白事件窗口明确bypass，有限bias不称硬音频隔离。H16新外置Effects节点可单用EAV、Relay或组合，并提供实际调用审计。

未知LoRA、Sage、Sol、hook／callable继续保留、委托或警告。真实输入形状、自有配对／receipt／SHA错误仍检查；“未经质量资格”不等于禁止组合运行。

## Full、Cold与边界

v5新Window Save的sidecar环绕原v5 literal cache；H16新Window Save使用独立格式。两者均默认 `confirm_save=false`；明确保存才创建文件，恢复必须填写实际path＋SHA及正确窗口，不自动找缓存。

v5 `Freeze Literal V5 Restart Capsule` 额外保存真实partial4、全局learned结果、全局AV噪声、全部已编码原生条件和实际接受的AV前缀。`Load Literal V5 Capsule` 字面加载选定值，**不重跑LOW、CLIP、媒体VAE、放大器或噪声**；输出接剩余窗口，用原HIGH seed、sampler、refine sigmas及匹配MODEL。不是宣称今天的MODEL／provider与历史运行等价，也不是质量批准。

普通Window Load需要当前source／lift／noise／bank精确匹配。异进程重新CLIP／放大得到近似而非相同值时，不能放松SHA硬拼旧前缀，应使用literal capsule。含外部可执行metadata的bank仍可在Full运行，但不能将未知callback pickle成安全Cold值；保持live-only警告，不伪造provider认证。

H16也有独立的whole capsule。`H16 Prepare Literal Bundle` 保留原先每piece的learned计算，但在HIGH前准备全部piece，不改成v5全局放大。`H16 Select Literal Piece` 输出该窗的source／lift／spec、同一全局context／plan／bank及完整source。窗口0的结果和bundle接 `H16 Freeze Whole Literal Restart Capsule`；默认不保存，明确 `confirm_save=true` 才保存全部原生值和真实AV前缀。Cold用 `H16 Load Whole Capsule`，接Selector的 `window_index=已保存index+1` 和下一HIGH，上一result来自Load。无需重跑LOW、文本／媒体编码、piece放大或噪声；重新连接原HIGH MODEL、sampler、seed和SIGMAS。H16／v5格式互不混用，MODEL和可移植provider等价不由此认证。额外保存全部piece会增加缓存大小，是显式可选分支；未知live-only metadata仍可走普通Full。

所有格式绑定策略版本、实际窗口ownership、原生编码／媒体、source／lift／噪声、接受音频SHA、前窗receipt和实现pin。已结束对白的receipt描述发布区间，不表示已经ASR确认逐字说完。

## 验收状态

当前24个新增入口append在旧632之后，共656唯一ID；真实Core读取的旧632全部schema／顺序／默认值exact。受影响24文件CPU为321pass／无skip或deselect／CUDA未初始化，包括H16／v5 literal恢复与完成态音频门；不是全仓或训练权重画质认证。

唯一新增GPU定向案例已从原生画布完整执行成功：原图派生12秒／187／overlap34、448×256→learned1.2实际512×288，native4+4，677.491秒。原三句对白、原模型／LoRA／seed／FFN／KJ／Core／Sol链保留。全RGB和PCM解码有限、非空，原片H.264／yuv420p／AAC可直接播放，无预览转码或替换音轨。实际W0 whole capsule已在新CPU进程从验证目录及本机默认output分别字面恢复。原图本身是5秒／136／scale2／单句／partial4；不是第三方原始12秒8+4失败严格复现。

原生294帧用于合法H3网格，最终trim到288帧／24fps／12秒。v5 Full／Cold两图已实际SaveAs、重开并核对；Cold只W1。H16 Full／Cold为另一路完成LOW8→低sigma HIGH4的CPU接线模板，512×288、不额外跑GPU，不能借v5样片批准H16听感。H16 Cold path／SHA须填写自己的H16 Full保存结果，空占位不是伪造可用缓存。四图均另存，不覆盖旧工作流。

2026-10-05，用户看完上述唯一12秒v5样片后在本聊天明确确认“审核完成，没问题！完美”，固定案例已通过人工审核。结论绑定原片SHA `cda3325081cfcf6018cd4556c20a96caa5042106d0ccecea86025f685b509ada`，不是助手自动审核或播放器可播检查。网页表当时仍未填写，聊天结论独立留档，不伪造用户表单操作。没有TTS／静音／替换声音／后期删除重复，也不以ASR或中间x0解码代替人审。

本次只批准固定v5原生4+4／12秒／chunk187／overlap34／512×288样片，不外推H16听感、任意模型／素材／尺寸或硬时间隔离。H16两图保持CPU接线模板资格。v1.92.0提供[73目录](../examples/workflows/73-temporal-dialogue-scope/README.md)的四张原生画布公开模板，另附四张外置EAV／Relay CPU接线模板。GitHub正式发布不等于Comfy Registry已激活或可安装，状态分别检查。
