# RADAR r6 参考包与全新条件（EXP）

N01已追加Create／Save／Load／Route／Apply及独立外置Relay适配六个EXP节点，实现实际原生VAE编码身份绑定及重新构建Qwen条件。真实画布已完成素材包编码保存、fresh32B Qwen条件和LOW4＋HIGH4采样；该完成HIGH在另一个正常GPU解码画布恢复成5秒原生音画，指定片已获用户通过。原Full终末CPU INT8 VAE解码报错，不能把原history称为success；Cold仅真实缓存绑定、保存重开与CPU资格，没有新增HIGH GPU实跑。N02同HIGH新Light独立对照也已获真人通过，五项交付纳入v1.93.0。没有训练模型、身份锁定或声音克隆承诺，不迁移旧refs／加载器／采样默认路径。

## 分开的节点

| 节点 | 作用 |
| --- | --- |
| `MiniMaxH3ReferenceCreateEXPT8` | 明确角色ID与image／video／audio类型；实际VAE编码一次，保留完整RGB grounding及可选独立声音锚 |
| `MiniMaxH3ReferenceSaveEXPT8` | 明确确认后在models/refmods发布新短名文件；不覆盖既有包；未确认只透传，不写目录或文件 |
| `MiniMaxH3ReferenceLoadEXPT8` | 从配置目录精确选择已安装包，可绑定文件SHA；不猜模型或导入任意路径 |
| `MiniMaxH3ReferenceRouteEXPT8` | 多包按显式角色顺序组合；分别控制画内视觉与声音锚，报告媒体编号 |
| `MiniMaxH3ReferenceConditioningEXPT8` | 为当前镜头重新编码Qwen；复用包内原生VAE latent，输出标准positive／AV_LATENT及原生文本配方 |
| `MiniMaxH3ReferenceRelayConditioningEXPT8` | 独立接入ReferenceSet与外置Relay Plan；输出成对MODEL／positive／AV_LATENT，LOW／HIGH各自绑定 |

Create需要连接真实素材和对应VAE，Apply需要当前CLIP与VAE。资产生成前后、Apply开始和全新Qwen编码后均重新核实际producer及内容；选错VAE或输入在编码期间变化时明确失败，不把文件名当身份。新可移植格式不能认证不透明encoder；原有普通refs仍可使用它们，不增加用户加载器黑名单。声音锚不是drive_audio或final_audio，不用它替换源配音。

## 格式与保存

自有格式为 `t8.h3.reference-package/v1`：safetensors tensor＋有界JSON metadata。不是社区RefMod格式4／5的静默兼容导入器。不pickle、动态导入插件、读取包内外部路径或捆绑角色素材。

- 文件tensor总量最多256MiB，header最多1MiB，最多15个成员；分配前检查shape、dtype、byte count、offset及成员集合，随后检查有限值与实际内容SHA。
- 视频VAE的原生未denoiser缩放latent为24通道；音频为 `[1,32,2,T]`。视觉时间／空间网格和合法RGB grounding须对应，保存不做pooling、JPEG压缩、强度曲线或重复引用。
- producer／source字段是内容声明，不是签名。底层格式检查本身不能证明调用者填的SHA来自真实模型；Create／Apply额外绑定实际原生权重、配置和实现内容身份。CPU模拟声明和微型Core类实例不是预训练权重资格。
- 保存要明确确认，只发布新文件；在独立CPU快照上校验后写自有临时文件、fsync、原子不覆盖发布。既有文件及失败证据不覆盖。读取核文件SHA及tensor实际内容，损坏不能自动命中。
- `models/refmods` 与Core配置的额外目录一起列出；加载必须选择精确已安装名称，不猜替代、接受任意绝对路径或目录穿越。这个新格式的安全校验不限制用户原有普通参考输入。

## 人物 presence 与画外声音

每个包角色明确填写 `visual`／`voice`，按显式有序角色列表路由。人物离场时，同时撤掉其VAE视觉block与Qwen图像grounding，不用视觉strength=0冒充撤除；明确画外说话人仍可保留音频参考。映射报告记录媒体type／ordinal及实际packed reference rows。

路由结果仍是**重新构建条件之前的输入**。Apply重新执行一次Qwen编码，只接收选中媒体；不从旧CONDITIONING里删除embedding或直接追加latent冒充撤图。视频包保留完整RGB，送Qwen时按原生2fps选帧并给实际时间戳；VAE latent不重复编码。当前新包路径不与旧raw refs混接，避免编号和producer含义不明确。

Apply复用原条件构建器的prompt、keyframe、drive／final audio、Bridge和文本配方规则。新节点primary audio ordinal默认0，不自动把第一个声音锚当驱动配音；明确映射驱动音轨时仍按实际编号处理。Bridge只在原位置调用一次；EAV／Prompt Relay继续用下游外置节点，不藏进资产包。标准输出适配普通和分离式接口，但完整LOW／HIGH画布及成片仍须实际验证，不借CPU接口检查宣传效果。

只有Qwen grounding缺失、其余格式和producer合同都合格时，底层允许单独明确的latent-only EXP。它返回缺grounding标志，不伪造图或token。当前Apply明确拒绝该路径，因为其独立token／ordinal适配尚未资格化；完整RGB包不受影响。producer／归一化缺失没有豁免。社区格式4／5互操作仍未认证。

## 当前证据与未完成项

5项格式测试及8项运行时测试覆盖有界读写、不覆盖、配置目录、实际编码器身份来源、VAE只编码一次、Qwen只重新编码一次、A画外声音与B画内参考、视频2fps、源配音与声音锚区分、producer／内容中途变化。运行时测试使用显式模拟编码器和微型Core原生类实例，不含预训练模型推理或画质认证。

完整受影响11文件CPU最终范围168 passed，0 skip／deselect，CUDA未初始化；4986源在运行期间冻结无变化，包含原条件、producer、Temporal Dialogue、双采上下文和长视频缓存门禁。真实Core加载把继承的空weight/bias回调列表写入实例，首次画布编码因此身份误报；已限定规范化真实Core无效空回调与绑定的cast驻留别名，不豁免权重、非空回调、未知hook或选定cast策略。真实失败、加强回归后的红与最终绿均保留；不改Core或旧缓存身份。独立真实Core finalized schema对比仍是旧656接口、hidden、默认与顺序为662精确前缀，新增只有N05一个＋N01五个，不声称整个新旧列表相同。

原生画布实际创建五种节点，A音频包经未确认Save透传、B图像包直接进入Route，A visual=false／voice=true、B visual=true／voice=false，然后接Apply；实际另存短名新工作流、关闭标签并从工作流栏重开，角色参数和四条typed接线保持。Load无已安装包时空选项显示null，未排队；这只验证创建及序列化，不当真实文件加载或音画效果验收。

随后实际原生Create／Save画布一次17.053秒完成，使用已安装音频FP32与视频INT8 convrot VAE；声音参考和首帧图像参考来自同一份已通过5秒素材。这证明资产链，不证明两个不同真人身份／声纹控制。真实Load／Route画布按两文件实际SHA加载，未确认的Save只透传，映射A声音400行／B图像64行，另存、关闭和重开保持参数与typed接线；没有采样、Qwen或驱动音频替换。两独立新CPU进程实际加载对应VAE与包，各0身份差异／CUDAfalse，确认跨进程producer匹配；没有encode或完整Cold采样。

新增外置Relay适配不改原Relay节点schema；共享工厂只增加kw-only `prepared_reference_set=None`，None时保持原调用，非None才走上述包的fresh条件路径。Bridge沿原位置一次调用，不再编码包内VAE。独立LOW／HIGH绑定、坏包/Qwen前拒绝、声音锚不是master、原接口未变均有CPU实证。最新19个完整受影响文件366 passed／0skip-deselect／CUDAfalse；不是171+168等历史数量累加。两独立实际Core过程对公开main证明旧656完整schema为663精确有序前缀，新增七项ID明确列出，whole list不同属预期。

`tools/build_reference_sampling_workflows.py` 构建普通8步、分离Full4+4和HIGH-only Cold三个小候选，不排队、不覆写工作流。LOW448×256／124原生帧，现有learned1.2×的实际输出接HIGH尺寸，最终原生音画Trim120帧／5秒。Full分别有两Relay Plan、EAV配置/审计及Stage Save；EAV默认report_only，不声称实际增强。Cold真正没有LOW模型/采样/效果，仅显式读取真实旧LOW path/SHA；占位不能当可用缓存。CPU检查包含真实finalized schema、typed边与Core validator，不等于实际fresh Qwen/GPU通过。

当前producer绑定包含选定设备、精度和cast策略。CPU创建包不能默认为GPU VAE生成的等价包；实际使用须选择匹配的VAE配置或明确重新创建包，不忽略身份字段。现有普通raw refs不受此新资产合同限制。

普通／Full／Cold三个候选已从实际服务生成，并通过原生文件选择器导入、另存新短名、关闭画布标签和工作流栏重开。磁盘核对19／40／27节点、25／70／42条named typed edges、40／109／72个widgets全部原样保持，没有允许数值修改、Queue或模型加载。Cold的LOW缓存path/SHA仍是明确待填占位，不借保存重开证明Cold采样通过。

## 最新真实5秒验收与解码边界

一个真实画布代表已完成：448×256 LOW4→既有learned1.2实际512×288→HIGH4，124原生帧后Trim120帧／24fps／5秒。LOW／HIGH都执行原32B Qwen新条件与独立外置Relay，实际attention审计各800次；EAV仍report_only，记录测量而不施加增强。原生联合音频没有TTS／静音／源配音替换，不冻结LOW partial4。两份素材来自同一已通过片的声音及图像，不证明两个真人声纹锁定。

两实际完成Stage均通过新CPU进程的严格SHA、receipt／portable identity与有限tensor冷读取。但新Cold HIGH并未再采样，不用冷读取证明完整Cold生成。

当前Core的INT8视频VAE decoder分支在CPU误调用CUDA注意力，导致原Full及纯CPU解码尝试失败。Core和旧采样器未改，也没有放宽producer身份。复用同一个完成HIGH，在**正常GPU VAE配置**的独立真实解码画布成功出片，新增采样NFE为0；120帧及全部PCM完整解码、非静音只是媒体完整性，不是对白／口型／人物质量批准。

本机交付因此明确分为采样保存Stage画布和独立GPU解码画布。采样侧参考资产producer对应GPU服务的`--cpu-vae --fp32-vae`；必须使用匹配的设备／精度配置，或从自己的合法素材重新Create／Save，不能忽略身份来迁移。新的Full／Cold采样图保留实际所有采样参数及外置效果，仅移走有问题的最终解码尾部，均实际另存、关闭标签、工作流栏重开，Core validator与named typed edges／widgets核对保持。不能宣称任意默认GPU／CPU服务都可直接复用这些包。

指定N01最终片已通过，N02许可及同HIGH新Light独立对照也已通过。十份正确原生图已集中另存到默认用户新目录，原字节及既有工作流保持；不会将失败整图保存成推荐图。普通8步只有结构／画布资格，未为扩大矩阵另生成。后续latent-only和社区来源另过门；互操作缺许可、producer或归一化证据时只停该来源，不禁旧普通路径。
