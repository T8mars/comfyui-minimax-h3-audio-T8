# H07 · 七项已审配方

下列原生画布对应集中审核 revision37 的四项通过案例和定向复审 revision10
的三项接受结果。接线、seed、
采样步数、原始尺寸、声音路径和外置效果保留；仅本机资源路径改为用户入口，
并补画布说明。路径适配已审计，不冒充重新运行或新的质量验收。
原字节画布本机另存，旧工作流不迁移、不覆盖；不捆绑权重或素材。

| 审核项 | 工作流 | 运行前配置 |
| --- | --- | --- |
| A1 | [新音频时钟8+3](A1_AudioClock_8plus3.json) | 节点4选择已准备的 audio-v3 runtime 配置；输入 `H07_RefPerson.png` 和 `H07_RefVoice.mp4` |
| A2 | [真实清唱4+4](A2_Song_4plus4.json) | 输入 `Song_Performer.png` 和真实清唱 `Song_Master.wav`，两音频加载器选同一源 |
| A4 | [指定Hybrid姿态](A4_Hybrid_Pose.json) | 输入 `Background_Source.mp4`；节点5选择同源 `pose-124` PNG目录；使用匹配8-wide Hybrid／Control |
| A5 | [同完成Stage的Native／X2](A5_SameStage_X2.json) | 节点1填写自己的已完成 audio-v3 HIGH `stage.json` 路径和真实SHA256；HyperVAE放 `models/vae` 并选择 |
| A3 | [独立背景plate＋原人物](A3_Background_Plate.json) | 输入 `H07_R37_BG_Source.mp4`；节点4填同源白背景MASK PNG目录，节点32填干净前景RGB PNG目录，节点14选6秒32k双声道静音WAV；仅接受当前有限效果 |
| A6 | [半尺寸内部动作拟音](A6_Foley_Half.json) | 节点1选静音动作视频，节点14按动作及时间重写提示词；匹配b25–49 Hybrid；[拟音用法](../../../docs/H07_FOLEY_USAGE.md) |
| A7 | [专用PDD 7＋1](A7_PDD_7plus1.json) | 完整FL2VA＋专用PDD动态AV头，864×480→learned1.2×实际1024×576；不是普通Turbo或PDD baked |

A1空配置使用既有 audio-v3 专用默认／环境变量，不自动下载或改旧配置。
A2保留已审图的 `ComfyUI-KJNodes`／`MiniMaxH3MemoryEfficientSageAttentionPatch`
依赖及对应 Sage 环境；不静默换后端。当前独立8991未加载KJ，保存审计只按
本机KJ实际源码的单MODEL输入／输出声明核序列化，未冒称该依赖已在8991运行。
A4的姿态目录、A5的Stage路径和SHA故意留空，选好真实资源后才能运行；
不自动复用别人或本机私有的Stage，不用MP4冒充原生完成latent。

说明：[音频时钟](../../../docs/FREEVIDEO_AUDIO_CLOCK_EXP.md)、
[合法清唱及DRIVE／VOICE／FINAL](../../../docs/H07_SONG_MASTER_EXP.md)、
[姿态资源及许可](../../../docs/H07_POSE_ASSETS_EXP.md)、
[同Stage可选X2](../../../docs/H07_X2_DECODE_EXP.md)。

只批准对应已审原片，不是所有模型或任意输入的保证。A1的EAV／Relay为
`report_only`；A2最终采用原录音；A4原声旁路，不证明拟音；A5零新增采样，
不证明独立7+1或高清接缝。新A7仅证明独立专用PDD配方；旧普通Turbo闪光片不收入。
A3用户明确接受当前效果但仍认为不佳，保留软浅边／遮挡限制，不当LanPaint算法修复。
A3需自行准备完整同源MASK与干净前景，本图不自动分割任意人物；最终合成采用原声。
A6采用新可见注液动作＋生成声，原空庭院失败片不收入。

已通过案例不追溯重跑，故保留实际旧尺寸。后续**新**视觉验收按一采约
0.4MP、learned放大1.2倍；不会将本例旧片插值成新验收。三项新原件另存不覆盖旧图。
