# 语音、音色与对白

这一组把H3联合AV模型作为生成式语音/对白工具，包含描述音色、参考音色、双人对白、长文本控制、ADR和本地音色库实验。

## 推荐顺序

1. `Speech_Described_Stock20`：先验证单人描述音色。
2. `Speech_Reference_Clone_Stock20`：再接授权参考音频，检查实际转录和身份相似度。
3. `Speech_Dialogue_Two_Speaker_Stock20`：双人逐turn生成与混音。
4. `Speech_LongForm_Resume_Stock20`：长文本分段、accepted和恢复。
5. ADR、Joint Dialogue、Voice Library和背景床路线均为更高风险EXP。

## 当前成果

单人、参考音色、逐turn对白、长文本计划/合成/恢复、字幕与报告已有节点和回归；上游所谓TTS本质也是H3生成式音频，不是确定性音素TTS。Joint多人、严格高保真克隆、精确情绪/语速/音高和真正实时流式尚未得到充分证据。

## 使用方法与注意事项

参考音频优先选择单人、干净、无音乐污染的5～14秒片段。中文与英文应分别检查实际transcript；不要只靠ASR通过就宣称音色一致。Voice Library会持久化本地资料，保存前确认路径和隐私策略；普通创作优先使用临时profile。
