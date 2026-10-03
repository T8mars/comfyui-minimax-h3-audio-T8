# H3 Fun Union 2.0：独立局部重绘入口（本地 EXP）

这是新入口，不改变旧 Fun 5 块节点、控件或已有工作流。CPU 结构/委托/内容身份与完整权重严格加载，以及一例原生 LOW→保存→独立 HIGH→fresh HIGH 重跑的机械检查已通过；**该 Turbo 4+4 实验片存在彩色方块，不是通过的质量样例，也不推荐这份组合。** 另一例无 Turbo 的原生40步已输出并完整解码5秒音视频，质量未人审。不作通用质量或提速承诺，未发布。

## 加载与接线

权重放 `models/controlnet` 或配置的 `model_patches` 搜索目录。使用 `MiniMaxH3FunUnion2LoaderEXPT8`，接到 `MiniMaxH3FunUnion2ApplyEXPT8` 的独立 `union_control` 口；旧 `H3_T8_FUN_CONTROL` 口不通用。文件名不是训练身份凭证，加载器核真实结构并严格加载全部参数。

官方 [固定 Union 2.0 权重](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/tree/0cb77d08bd6902d54eefede0d74b3450ae4840eb) 使用 Diffusers 键名，节点在内存中复用 T8 转换：Q/K/V 拼接、FF 半块顺序转换、norm/output 名称映射；不导出重复的大模型文件、不下载时自动执行上游脚本、不修改共享 Core。此配置为 10 块、位置 0/5/…/45、49 通道、post_norm。冲突元数据、漏块、错误维度正常报错。

原权重 AdaLN 输入宽度 2688，需要完整形式底模。常见剪枝底模输入宽度 8 不能直接搭配；节点看实际 live tensor，不猜文件名，不补零“兼容”。若选择显式 curve 形式权重，其真实宽度和元数据也必须一致；本地下载的原权重并不是 curve 形式。

## 源视频与外置 MASK

`source_video`、可选 `control_video`、`regen_mask` 必须与本阶段的画布和 `17n+5` 帧格完全对应。先显式完成同一 resize/crop/trim，再接入节点。节点还在实际采样时核目标 latent 与原生 H3 16×编码 VAE，防止只改 UI 宽高但实际采样仍使用另一画布。仅张量形状相同不证明 VFR/原 PTS 或语义来源相同。

MASK 白色重绘，黑色保留，阈值及 keep=`1-regen` 由当前 Core 原生路径计算。49 通道为控制 latent 24 + keep 1 + masked-source latent 24。post_norm 填洞使用 Core 的 `IMAGENET_MEAN`，不是裸黑或自行固定 0.5。单帧静态 MASK 默认不能自动复制，必须明确打开 `broadcast_single_mask`。输入不得含 NaN、越界值、错误帧数或错尺寸。

两采的 LOW/HIGH 可以各自接一个 Apply，分别提供对应分辨率的源、控制和 MASK；EAV／Prompt Relay 保持外置。添加的 Core wrapper 保留前置 block delegate，不清用户 LoRA、hook 或注意力选项；未验证组合只报告风险，不恢复准入禁令。`strength=0` 是原对象旁路，不编码 VAE，也不算非零控制资格。

## 分离阶段与音频边界

Union stage 内容投影仅在检查用 MODEL clone 上去除已识别的 Union wrapper，原执行对象不变。身份绑定实际控制权重、VAE 生产者、源/控制/MASK 张量、native sigma、配置和代码；派生控制 latent cache 核内容后才可不计入独立输入。未识别对象仍可新采样，但不能借一个 JSON 报告获跨进程缓存认证。多次 Union Apply 叠在同一阶段目前保持执行而不授予便携资格。

内容身份不等于已完成 StageResult；完成、StageSave/Load、fresh-process 数值和完整媒体需要单独资格。旧存档的实现内容校验不被关闭或重签。

## 当前实测边界与配方

通用原生40步 Full Save／Completed Cold Delivery 图在 [69-radar-native-recipes](../examples/workflows/69-radar-native-recipes/README.md)，采用明确源文件占位与外置MASK，不包含私有素材。它不是前述失败Turbo4+4的推荐发布；选好源、完成阶段path／SHA后再运行。

[官方固定配方](https://github.com/aigc-apps/VideoX-Fun/blob/4b7b6402a1e0f0406bd6801fb66c0a00bd922621/examples/minimax_h3_fun/predict_v2v_control_inpaint.py) 使用无LoRA、40步、CFG=1、control scale=1。Union2是guidance蒸馏，不是Turbo8步模型；不能把分离阶段接口可接通等同于任意4+4组合画质通过。节点仍保留用户LoRA/效果，不硬禁未知组合，也不会自动替用户改步数或采样器。

固定本地实物的首例原生4+4 LOW/HIGH都保存了便携stage；独立fresh HIGH的请求、状态与完整解码AV逐值一致，最终120帧448×256/24fps/5秒。外置Relay/EAV均有实际委托观察，但此例EAV增益全为1，不是非单位增强资格。它的彩色方块失败片单独保留，不发成推荐模板。

无Turbo的定向40步对照使用同一原始源、448×256画布、native双时钟12/3、原生CFG=1、全程Union strength1，不经过低分辨率双采放大或EAV/Relay。真实Save As/重开/Queue、40回调、stage保存和完整5秒AV解码通过；独立fresh进程原生画布显式Load完成stage，零采样，完整解码RGB/PCM与full逐值相同，原存档保持。抽查帧未见前例方块，但不代表整片人审。量化底模、小画布、fp16 VAE与官方上游设置不同，且同时更改了多个实验因素，**不是逐值上游复现或“已证明Turbo是唯一根因”**。人工质量仍独列；不以自测认证所有分辨率/底模。

Core 只将 control skip 的音频位置置零；联合网络仍可能改变生成音频，这不是音轨逐样本保持保证。需保留原声时沿已有显式音频锁和最终 mux 政策。控制 source/MASK 也不保证黑区输出 RGB 精确不变；需要精确贴回保护时是独立 composite 功能，不以本控制节点冒充。
