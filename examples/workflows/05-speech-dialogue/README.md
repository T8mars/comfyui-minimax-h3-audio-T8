# 语音、音色与对白

这一组把H3联合AV模型作为生成式语音/对白工具，包含描述音色、参考音色、双人对白、长文本控制、ADR和本地音色库实验。

## 推荐顺序

1. `Speech_Described_Stock20`：先验证单人描述音色。
2. `Speech_Reference_Clone_Stock20`：再接授权参考音频，检查实际转录和身份相似度。
3. `Speech_Dialogue_Two_Speaker_Stock20`：双人逐turn生成与混音。
4. `Speech_LongForm_Resume_Stock20`：长文本分段、accepted和恢复。
5. ADR、Joint Dialogue、Voice Library和背景床路线均为更高风险EXP。
6. `H3_Audio_Integrity_Audit_Advanced`：生成或混音完成后检查精确采样数、A/V边界、开头突变、DC、削波和疑似尾首回卷。
7. `H3_Speaker_Routing_Audit_Advanced`：多人逐turn生成前检查每个人物到参考音频的唯一映射。
8. `H3_Audio_Perceptual_Drift_Audit_Advanced`：把已接受的一采音轨和同内容、同时间线候选做参考相对漂移检查。

## 当前成果

单人、参考音色、逐turn对白、长文本计划/合成/恢复、字幕与报告已有节点和回归；上游所谓TTS本质也是H3生成式音频，不是确定性音素TTS。Joint多人、严格高保真克隆、精确情绪/语速/音高和真正实时流式尚未得到充分证据。

转录计分已采用Unicode NFKC+casefold：中日韩按字符CER，重音拉丁、西里尔、阿拉伯等脚本按Unicode词单元WER，混合中英按真实Unicode字母数字字符计算CER，不再把不认识的脚本静默丢掉。这里只证明计分器合同；每语言30条真实H3生成、人工复核和多说话人听感矩阵仍未完成。

## 多语言验证清单

`tools/validate_speech_multilingual.py`的严格模式要求每个case填写`case_id`、`language_code`、`generation_mode`（`described`或`clone`）、`utterance_id`、整数`seed`、`audio_path`和`expected_text`；clone还要填写`speaker_id`，described填写`voice_profile_id`。同一utterance的期望文本必须一致，同一utterance/mode/音色条件下才允许把不同seed计为重复实验。默认门槛是每语言30份不重复音频、10条utterance、两种模式，以及每个实验条件三组不同seed。

正式中英矩阵先由只写计划、不提交ComfyUI任务的构建器生成。它会从两份已审阅API模板派生120个独立提示：中英各10条utterance、`described/clone`两种模式、每格3个固定seed。clone轮换10名有明确许可且内容SHA唯一的LibriSpeech参考说话人；中文clone因此是跨语言参考实验，不能单独证明身份保持。当前文本集是词汇多样的人工审阅集，不是音素平衡语料，也不等于“每名说话人10句”的正式身份矩阵。

```powershell
python tools/build_speech_multilingual_formal_matrix.py `
  --clone-sources artifacts/speech-reliability-check/librispeech-10-speaker-sources.json `
  --output artifacts/speech-multilingual-formal-en-zh-v1
```

构建器可重复执行，计划内容和120份API提示必须逐字节保持不变；已存在文件发生漂移时会拒绝覆盖。它不连接`8188`、不排队、不加载模型。生成完成后再以只读方式收集指定输出目录：

```powershell
python tools/build_speech_multilingual_formal_matrix.py `
  --clone-sources artifacts/speech-reliability-check/librispeech-10-speaker-sources.json `
  --output artifacts/speech-multilingual-formal-en-zh-v1 `
  --collect-from F:/AI-T8-video-onekey/ComfyUI/output
```

只有120个case都恰好匹配一份可解码、至少2秒且内容SHA互不重复的音频时，才会生成`multilingual_manifest.json`；缺失、重名、损坏、过短或重复内容都会失败关闭，并删除过期的派生manifest。随后才能对该manifest运行下面的严格设计门和固定ASR。当前本机只是完成了计划，SHA-256为`119735A82B59ED5F1EDDBBD68A74B5A19B8742CBECCCCCB49F32DF6338C9CBE8`；收集状态为120项全部`PENDING_MISSING_OUTPUT`，没有manifest，稳定门仍为false。

需要实际生成时使用独立的有界执行器，先运行默认预检：

```powershell
python tools/run_speech_multilingual_formal_batch.py
```

它只允许回环地址和非8188的私有端口，默认要求至少12000MiB空闲显存、选择1个未完成case，且不会启动服务。所有门通过后，用户再显式确认执行；单次最多6项并严格串行，推荐保持1项：

```powershell
python tools/run_speech_multilingual_formal_batch.py --confirm-run --max-cases 1
```

执行器启动自己拥有的8197 ComfyUI进程，使用独立user/temp/内存数据库和矩阵专用输出目录；不会向8188排队、interrupt、unload或terminate。每次尝试后原子保存`execution_state.json`，成功音频由同一严格收集器验证，已收集case自动跳过，失败立即停止。锁文件存在、端口被占、显存不足、模型/参考文件缺失、计划或提示哈希漂移、输出重名/损坏都会在启动前拒绝。2026-08-23本机真实预检除显存外全部通过；8188占用时只余4628MiB，因此状态为`ABSTAIN_INSUFFICIENT_FREE_VRAM`，没有启动8197或生成任务。

先只检查实验设计，不加载ASR：

```powershell
python tools/validate_speech_multilingual.py manifest.json --output manifest_audit.json --validate-only
```

通过设计门后再连接固定revision的多语言Faster Whisper执行正式评估。设计门通过只说明样本结构合格；正式ASR通过也只说明文字准确，不代表音色、自然度、情绪或口型通过。

现有14条历史真实H3语音已用固定多语言Faster Whisper small、CPU INT8、2线程、beam 1全部复查。
2条英文描述音色WER为0，1条中文描述音色CER为1/14；11条英文clone的WER中位数虽为0，但均值
0.7841，只有6/11不超过0.15，另有1条0.25和4条1.625～2.5的严重额外/非目标语音。工具现在会
单独输出语言×生成模式、音色条件和超阈值case，避免总中位数掩盖失败。该历史集不平衡、中文仅1条、
没有三seed重复条件，所以正式多语言门仍为失败，不能把这批结果宣传成克隆或多语言通过。

## 音色克隆 ABX 盲评

`tools/build_voice_clone_abx_review.py`把真人目标参考、真人冒充者参考和生成音频打成A/B/X匿名网页。输入必须是音频-only文件，且三路codec、采样率、声道和容器一致；工具不会自动重采样、混音或响度归一。页面只显示不透明案例号，真实case、speaker、target/impostor映射和每个媒体SHA只保存在`blind_key.json`。

正式身份矩阵与上面的120条多语言矩阵相互独立。以下命令只生成10名目标×每人10句×3seed的
300份clone API提示，以及每目标3名同LibriSpeech元数据标签impostor×3seed的90组ABX预注册；不会
连接8188或提交生成：

```powershell
python tools/build_voice_clone_identity_formal_matrix.py `
  --clone-sources artifacts/speech-reliability-check/librispeech-10-speaker-sources.json `
  --output artifacts/speech-voice-clone-identity-en-v1
```

需要生成时仍复用同一个安全串行入口，显式指定该计划目录；默认只预检，推荐每次1条：

```powershell
python tools/run_speech_multilingual_formal_batch.py `
  --plan-root artifacts/speech-voice-clone-identity-en-v1 `
  --language en --mode clone --max-cases 1
```

当前计划/设计SHA-256分别为`3D9814B2...1C33F335`和`7AC44DCA...BFF224FD`，收集为0/300。
预检发现空闲显存4250MiB，低于12000MiB门，因此没有启动私有8197。全部300条唯一收齐后，身份工具也
只会写32kHz mono FLAC、关闭响度归一的标准化作业合同；标准化实际完成并由ABX打包器再次确认A/B/X
媒体合同一致之前，不会产生可评审manifest。LibriSpeech的F/M标签只用于阻断明显跨标签音高线索，
不代表本项目推断生理属性。

```powershell
python tools/materialize_voice_clone_abx_standardized.py `
  artifacts/speech-voice-clone-identity-en-v1/abx_standardization_jobs.json `
  --output-root artifacts/speech-voice-clone-identity-en-v1/abx-standardized

# 只有上一步报告READY后才显式执行；默认每次最多10个文件，严格串行并可恢复
python tools/materialize_voice_clone_abx_standardized.py `
  artifacts/speech-voice-clone-identity-en-v1/abx_standardization_jobs.json `
  --output-root artifacts/speech-voice-clone-identity-en-v1/abx-standardized `
  --confirm-run --max-files 10

python tools/build_voice_clone_abx_review.py `
  artifacts/speech-voice-clone-identity-en-v1/abx-standardized/abx_manifest.json `
  --output abx-review --random-seed 260823
python tools/analyze_voice_clone_abx_review.py --review reviewer-01.json --review reviewer-02.json --review reviewer-03.json --blind-key abx-review/blind_key.json --output abx-analysis.json
```

标准化器不会改变原件、不会做响度归一，也不会在默认预检中调用FFmpeg。实际执行会完整核对输入SHA，
原子写入独立输出目录，并在每个文件后保存状态；已有未登记文件、内容漂移或并发锁都会停止。正式
manifest让目标参考在A/B位置按目标人物及全局平衡，避免单侧偏好成为线索；旧pilot仍使用原来的独立随机策略。

默认正式门要求至少10名目标说话人、每人至少3名不同冒充者、3个已知seed和3名独立评审，并检查准确率、Wilson 95%下界、弃权率和无效率。即使该身份辨别门通过，报告仍固定输出`high_fidelity_clone_claim=NOT_ESTABLISHED`：自然度、演绎、授权、安全和陌生说话人泛化必须另验。

旧10人ABX包经新工具复核时发现真人参考为16kHz mono、生成音频为32kHz stereo，存在可作弊线索。原件未改，现有本地pilot仅另存32kHz mono副本并生成新匿名页；它每人只有1名冒充者、1句、未知seed且无人填写，所以只能作页面/流程试用，不能通过正式门。

## 使用方法与注意事项

参考音频优先选择单人、干净、无音乐污染的5～14秒片段。中文与英文应分别检查实际transcript；不要只靠ASR通过就宣称音色一致。Voice Library会持久化本地资料，保存前确认路径和隐私策略；普通创作优先使用临时profile。

三个审计节点都只返回`PASS/ABSTAIN`和报告，不自动修音、换声或重写对白。尾首高度相似也可能来自有意循环的音乐。音色漂移审计必须比较同台词、同起点、同采样率的同步A/B：默认500ms窗、100ms步进、-50dBFS有效音频门和连续3窗；不同表演或错位也会触发。此前实听异常的`pass2_recovered_exp`在1.4～3.6秒被正确标记，正常一采和本例80%混音通过，但单组校准不能外推为通用听感判定。
