# 外部视频接 LTX（EXP）

`H05_LTX_Source_EXP.json`：独立外源 RGB／立体声模板，原生保存重开和一条
5 秒机械实跑已核，待人审。替换 LoadVideo 素材占位名，并选择本机模型。
目标须为64倍数，当前256×448；可能中心裁切，不认证任意来源。

默认 EAV／Relay 仅报告，默认新模板输出为显式 VHS H.264／AAC，需要
VideoHelperSuite。原音频不由本项目归一化；AAC非PCM exact且有包边界尾差。
不替换旧图、不认证 H3 Safe AV、不构成原生H3祖先／Cold缓存。
详见 [完整用法及边界](../../../docs/H05_LTX_SOURCE_EXP.md)。
