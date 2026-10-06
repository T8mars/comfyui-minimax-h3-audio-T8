# FreeVideo 新四档（独立增量 EXP）

原 8+2、真正前4＋后4、原十二节点、旧默认配置和六张已通过工作流保留。新功能使用独立 Quality v2 模型／阶段类型，不自动迁移旧图或旧缓存。

| 选择 | 实际执行 | 出口 |
| --- | --- | --- |
| Light／轻量 | LOW8 → 外置放大 → Community HIGH3 | LOW8完成不等于整个Light完成；HIGH保留LOW完成音频 |
| Medium／标准 | 单采12 | 联合音画终态，直接解码 |
| High／精细 | 单采16 | 同上 |
| Max／极致 | 单采20 | 同上 |

档名代表采样投入，不保证任意场景画质、速度或对白更好。Community HIGH3 是独立 `[0.9035,0.6316,0.3158]→0` video sigma 表，不是原八步表的尾3步。12／16／20都使用完整原生网格，不重复八步或插值调制输出。

20步有明确新路线路径修正：固定作者表的raw index9为FP32(.55)，部分本机CPU `torch.linspace`给出其下一邻居，导致audio t相差1.19209e-7、严格表身份失配。只在新worker中显式绑定该发布时钟，收据写`published_20_FP32_raw_index9_v1`，仍精确匹配全task／50层表；不放宽比较、不回写主Diffusers、不称本机未修正20步逐位parity，旧8／4+4不变。

八个新增入口：Quality Loader、Quality Sampler、Community HIGH3、Quality Stage Save／Load，以及独立 Quality LoRA、EAV、Prompt Relay。LOW／HIGH分别接各自的效果模型和条件；Relay必须成对接线。原本家族的三个效果节点保留，不因混用模型类型默默改旧执行器。EAV默认report_only，不承诺可见增强；Relay为原VDN实验性文本种子机制，不宣称论文softmax等价。

## 独立环境与表

固定作者正式v0.2.3 `e7eb66326a038344ba241fa31355c15b258b98cb`；与旧路线主体rowwise模型相同，可只读复用现有22.9GB主体。新增的是匹配模式及实际任务的50层AdaLN表。T2VA HIGH3／12／16／20约58／223／300／377MB；带视觉或音频参考会更大。无需把独立FP8引擎伪转换成普通UNET。模型许可仍按MiniMax H3适用范围，不由代码Apache许可改变。

1. `tools/prepare_freevideo_quality_source.py --help`：下载固定源码并核官方Git blob；不执行作者安装器。
2. `tools/prepare_freevideo_quality_runtime.py --help`：借用已验证旧主体/VDN/Python，按 `--profiles` 和实际 `--task` 准备缺失表，创建全新v2配置。缺表明确失败，无隐式26GB原投影恢复；运行时表只读，LoRA派生也绑定真实modulation weight identity。
3. `tools/check_freevideo_quality_kernels.py <新配置>`：新源码做两个tiny检查，不重标签旧日志；应用多事件Relay再显式加 `--masked`。
4. Quality Loader填实际新配置，或专用 `user/default/T8/freevideo-runtime-v2.json`。不读取旧默认环境变量；主Comfy Python不升级。

如果只改变新UI适配类、已有同v0.2.3计算源的tiny结果，可显式用`tools/bind_freevideo_quality_probes.py <新配置> --original-config <原配置>`复用。先完整库存校验、当前实际环境/GPU及计算源identity精确一致、原日志/数值门/实现SHA一致，才create-only复制原字节并记录原执行出处；不声称重新执行过tiny。计算身份变化必须重新执行，不能靠此工具改写旧证据。

新Stage保存完整实际NFE、clock/task/表50层SHA、源及tensor身份和LOW音频lineage。HIGH3只接新Light LOW8和外置放大的同音轨AV；拒绝旧MID、旧Stage、SINGLE终态、错帧数、坏SHA及非有限数据。Cold填写真实path/SHA，不用占位，也不重跑LOW。取消、异常或清理失败不发布完成标记。

## 验收边界

当前受影响CPU范围121项通过，旧664个完整schema保持、新8个入口末尾追加。Light LOW8→外置learned1.2实际512×288→独立HIGH3，以及Medium12单采已各从原生画布完整生成5秒音画；实际history与NativeSaved全部执行参数及接线精确核对，真实完成Stage独立CPU冷加载、Light LOW／HIGH完成音轨一致。两原片均120帧／24fps／H264 yuv420p、完整RGB／PCM及PTS核验，并在浏览器播放至结尾。2026-10-06用户对当前两项审核直接确认“2个都通过了”，只批准这两条固定成片，不扩大为任意素材／效果组合或16／20档画质认证。

四张实际NativeSaved完整／Cold图已逐字节另存本机`user/default/workflows/FreeVideo_四档_已通过_20261006`，不改348张原用户图，也保留先前待审副本。Light Cold只继续HIGH3；单采Cold直接解码，均真实缓存path／SHA。Cold只做原生保存重开、CPU冷加载与真实Core结构校验，0新增Cold GPU；其执行投影明确记录为CPU projection而非未落盘的浏览器API导出。真人聊天批准单独绑定审核ID及两原片SHA；网页表仍revision0未填，助手没有代填表单。新专用v2默认配置与旧v1完全分离。本轮本地交付，不代表GitHub或Registry已发布。

当前实现／CPU结构验证与GPU完整片／人审分开记录。本轮最低仅Light8+3及标准12两条5秒正常中文对白“你好，今天真不错。”，448×256→外置1.2实际512×288／单采512×288；不是作者默认2×几何parity或同算量速度对照。16／20结构支持，不借前两片声明16／20指定GPU画质通过。后续发布需用户授权，再按正式线上基线白名单补齐发行包含的准备工具／文档和脱敏示例；不上传私有交接、绝对路径配置、模型、Stage缓存、媒体或审核反馈。
