# DiT 参考跨步缓存（实验）

这是显式开启的**近似算法**，不是 Qwen 视觉前缀缓存，也不是重复请求的精确记忆化。
不改变采样器、sigma 表、NFE、噪声种子、音频时钟、参考位置或 learned 3D upscaler。
默认 `off` 返回原 MODEL / positive 对象，旧图不受影响。

## 接线与选择

1. 先用原流程选择底模、按用户顺序加载 LoRA、构建对应条件。
2. 将 MODEL 和 positive 一起接入 `H3 参考跨步缓存 · Reference Step Cache (EXP/T8)`。
3. 将节点返回的 MODEL 和 positive **成对**接入 Guider / 原采样器；不可混用另一阶段的 positive。
4. Audit 接同一个返回 MODEL，`after_sampling` 接实际采样输出，以建立执行顺序。

正常 CFG 的 negative 不需额外接此节点：Core 明确标为 negative 且没有配对标记时保留原计算、全算并
单独计数。正条件仍须匹配配对标记；带着另一个 MODEL 的错误非空标记仍报错。

选项：

- `mode=off`：直通，不安装缓存。
- `mode=approximate`：显式允许参考输入的跨步近似复用。
- `refresh_every=1`：每次完全重算，作为执行基线；不会命中近似路径。
- `refresh_every=2`：在真实 sigma 网格的 0、2、4…步刷新，1、3、5…步尝试复用。
  其它正整数同理，没有人为的小步数上限；大间隔可能增加画面偏差。
- `include_text=false`：只选择 keyframe、参考图像和参考音频行；生成视频/音频始终属于活动行。
  `true` 还选择原文本行，属于额外近似，默认不启用。
- `phase`：显式标签，常用 `SINGLE` / `LOW` / `HIGH`；每阶段单独配置、单独缓存。

更改已有自有节点配置时，只在克隆 MODEL 上替换其精确自有槽位，保留前后外部 owner 和 callback
顺序。外部已覆盖 MODEL function wrapper 时保留它并警告，不删除用户补丁来伪装重新启用。
从原底模分支重新配置最容易辨认阶段归属。`off` 直通输入对象，不会主动清掉前面节点安装的功能。

## 实际省略了什么

刷新时，运行原 DiT block，并将该 block 的参考**输入**逐位复制到 CPU。
复用时，用冻结输入计算当前 timestep 的参考归一化/调制及 QKV；生成行仍用当前输入。
K/V 保留完整物理行顺序和原 RoPE，只有活动行执行 attention query、out projection 与 MLP。
不改写共享 network 的 forward、不替换全局 kernel、不截断参考 latent、不修改输入条件或生成 latent。

因为参考状态本来会与目标交互，冻结旧输入一般不等价于全重算，不能承诺人物、文字、运动或声音不变。
单层微型模型的参考输入可能恰好恒定，观察到零误差不构成普遍的精确性证明。

## 时钟、组合与失败

缓存按当前真实 `sample_sigmas` 网格和 `sigmas` 识别逻辑步，不用模型调用次数猜步号。
同 sigma 的重复实际评估（例如强/弱分支、重复前驱调用）完整重算，不推进刷新索引，也不覆盖旧刷新输入。
网格无法识别或非连续调用时保留全算。记录来自实际 DIFFUSION_MODEL 入口，而非外层包装器调用数。

选择到未知 network/attention、已有 DiT block owner、实例 hook、非已知逐行算子或不透明权重存储时，
不因类名或“未验证”禁止生成：保留原 callable，相关部分全算，Audit 解释旁路。
Relay 和 EAV 的路由/偏置属于已有 attention owner，当前保留其原算法，不在它们的 mask 上猜测裁剪缓存。
本轮实际 tiny Core 已覆盖 Relay/EAV 的正常执行和全算旁路，**不是**这些组合的缓存收益资格。
其它未知组合保留执行并报告实际覆盖，不据此宣称完整兼容、GPU 或训练画质通过。

每请求重新建缓存；LOW/HIGH 缓存不共享。同一次请求按实际 context、refs/keyframes、provider 内容、
原布局与位置 SHA 分桶。配置、已知条件、MODEL/LoRA 有序补丁与 owner 选择、实际参数版本会复核。
首/末逻辑评估还对实际 dense 模型权重做完整 SHA，避免 `.data` 改动被当成成功缓存；这有额外成本，
不是免费的性能测量。不透明对象仅核进程内选择，不能认证其内部状态。

真实缺失 API、错形状、NaN/Inf、配对错误、已知内容/自有缓存 SHA 损坏仍直接报错。
失败或取消不重试 kernel、不换算法；CPU 缓存会清除，下一请求重新刷新。
没有持久化缓存或跨进程 portable identity；接入 Stage 时仍可正常执行，但不能用本节点伪造便携完成记录。

## Audit 与证据边界

Audit 显示实际模型评估、各 block 完全重算/近似命中、重复 sigma、实际跳过的参考行、CPU 缓存字节、
清理状态、旁路原因及错误。报告是独立副本，不允许编辑报告来改变 live 缓存。
`approximate_block_hits=0` 不能写成“缓存已加速”；计划 `refresh_every` 也不代表实际 NFE。
实际评估总数包括 CFG negative 全算，并另列 paired / unpaired-negative 数，不把外层 4 步冒充 8 次模型评估。
始终明确 `bit_exact_claimed=false`、`portable_cache_identity=false`、`GPU_performance_qualified=false`
及 `quality_accepted=false`。实际 tiny 数值误差记录在测试证据，不外推到训练权重。

CPU 自动验证使用真实 Core 的随机小参数 DiT、原 attention/LoRA parser/CPU sampler。
当前 Core 的 VBAR 驻留运输字段曾被模块结构校验误当成执行结构变更：窄修仅对身份对应的原 Core cast/resolver 忽略其写入的三个运输字段，权重、forward、条件、已知内容与完成身份检查保留。原 45 项加新 15 项实际 CPU 回归通过；未知诊断接口不授予该豁免，也不成为禁止用户生成的模型名单。

修复后固定 Ref2VA 原生 20 步、`refresh_every=2`／`include_text=false` 已实际生成成功：20 次 paired 模型评估、每次 50 个 block，实际 500 次完整刷新、500 次近似命中，共跳过 5,738,500 行参考 query／输出投影／MLP 工作；请求结束缓存清零、cleanup 被观察。完整 120 帧 736×416／24fps 原生 stereo32k 音视频机械通过。原失败 R35 记录仍保留，新的单例有自己的末尾文件校验及进程退出证明，不借新结果把旧失败批次改绿。

该记录只证明选定训练模型上的实际缓存消费与完整媒体，不包括 GPU 显存收益、端到端提速、人审、数值或声音一致性、真实长视频接缝或所有第三方组合。源码、原始报告及完整媒体边界见本地 `h22-R5-fresh-postflight-v2/actual-one-index-v1/merge.json`。
使用时先与同素材、同 seed、同 sigma 的 `refresh_every=1` 完整 AV 对照，再决定是否采用；不要用
CPU 通过或一个中间 latent 预览代替最终画面和声音验收。已接受的 4+4 / 3D / 接缝配方不改变。

## 来源与许可证

设计问题来自固定上游 [22e719f 参考跨步缓存提交](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder/commit/22e719fe89336c628383363182a4da3cb58d4f9b)。
该提交仓库的 [MIT 许可证](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder/blob/22e719fe89336c628383363182a4da3cb58d4f9b/LICENSE)
与本项目许可各自保留。本实现独立编写，没有复制或引入上游 `ref_cache.py`。
上游个别性能记录不是本项目性能承诺；这里的实际逻辑 sigma、成对条件/阶段、原物理行顺序、
异常隔离和保留外部 owner 均单独实现与验证。
