# 社区问题核对与 Core 0.45 兼容

2026-11-17，逐项检查仓库的15个开放Issue和2个PR。以下区分代码修复、使用说明和外部依赖，不把尚未验证的画质或外部审核写成完成。

## 新发现并修复：FinalLayer显式网格

本机实际Core已更新为v0.45.0（`989d93d0d968e704c589ce0146bb18c77f6b50cc`）。它把原生FinalLayer的`shifts`输入改为`grids`。旧适配器将这个真实接口判为不完整，影响多关键帧、使用旧外部接口的block cache，以及相关HyperFlow路径。

现在根据实际调用接口分派：旧四参数接口保持原调用；旧`sigma/sample_sigmas/shifts`接口原样传入；新`grids`接口使用已安装Core自己的`pdd_grid`及实际输出头数，显式给出的网格直接保留。调用仍进入原layer或用户delegate一次，真实内部错误不捕获重试。未改模型、LoRA、步数、采样sigma、声音或缓存完整性要求。

本次当前Core相关用例按测试ID去重后834项通过、6项有明确前提缺失而跳过；这些用例跨数个定向批次核对，初次失败保留，不宣称整个历史测试树全绿。v0.43.0同组226项通过/30项跳过，v0.41.1为223项通过/33项跳过。旧Core环境的跳过包括未装外部BlockCache/KJ/Sol、缺少历史源码、旧Core不存在的网格/PDD接口，以及实际CPython3.13编译器元数据用例；元数据注入和负例在Python3.12执行。当前Core另跳过2.8GB可选原权重及两个用户运行服务的live入口，使用独立CPU/浏览器测试验证相应基础交互。

实际原生FinalLayer普通头与PDD多头均与直接调用逐位对照；分块投影仍与对应原生调用比较并保留原容差。Topaz183项独立回归通过。列宽刷新初次测试有一次时序失败，独立诊断快照确认65%正确保存，原始测试随后连续两次通过，未据此修改UI。未重新生成GPU成片，不把CPU数学/接口验证称为新完整模型质量验收。

## PR结论

- [#31](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/31)：问题已由现有`_sampling_class_members`覆盖，保留其更严格的元数据类型、非执行读取和真实方法检查。Core SigmaShift的整文件误拒也已在`96f0381`修复。该贡献作为已被现有实现覆盖处理，未重复合并较弱实现。
- [#29](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/29)：固定head `71a99d6a681aeb3d81e1f229aaf2ad78af5633db`仍会错误规整真实VFR。实际运行其两段纯计算函数，`[(900,512),(50,256),(50,768)]`会返回512；虽然总时长变化为0，各段真实速度仍被改变。此前要求的完整时序/音轨审计、受控取消worker和注册顺序等也未更新，因此拒绝合并此实现。感谢yangala的提案；不会把拒绝此PR写成已经交付时间轴修复节点。

## Issue处理结论

|Issue|结论与使用边界|
|---|---|
|[#32](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/32)|已修复匹配原生状态导出hook的非持久buffer误判；真实Dynamic加载、原始backup还原、伪造导出拒绝均有回归。仅换run_name不能修复旧代码。|
|[#30](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/30)、[#24](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/24)|已有v1.98.2的显式software/CUVID选择，视频/音频探测、PCM、增强和最终解码均传递；保留Main10帧级位深审计。CPU模拟及原有官方程序验证不冒充报告者机器的新完整增强实测。|
|[#23](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/23)|Topaz已有独立NVML读取器，不再套用DLSS的单GPU门禁；模拟两卡选择及越界清理覆盖，不声明实际双NVIDIA硬件验收。|
|[#28](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/28)|已有`/api/`前缀及`ui-modules`路由修复，更新T8后需重启后端并刷新页面。|
|[#22](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/22)|LoRA行选择、LOW/HIGH有序链、取消/应用及项目保存恢复已有修复和独立浏览器回归。时间轴等建议与选择器bug分开；定制分发遵循LICENSE及第三方NOTICE。|
|[#27](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/27)|已缩短示例名并保留JSON内容；旧名见filename-map.tsv。相对路径140、工作流名96的门禁继续执行。|
|[#25](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/25)|相同MP不是固定尺寸承诺。把放大器width/height连到HIGH Conditioning，再接Reconcile；精确尺寸用target_dimensions。保留旧图默认，不让HIGH独立重选另一套宽高。|
|[#21](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/21)|已有独立HyperFlow Loader/Plan/Sampler及工作流；Flash也是独立消费者，不当普通LoRA加载。组合资格按各示例验收范围。|
|[#20](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/20)|已有native慢/快ByteLevel兼容；仍核对实际token与完整提示词字节，不替换成任意tokenizer。|
|[#19](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/19)|Registry激活由官方处理，上传与Active/installable分开核验；保留公开问题继续跟踪当前目标版本。|
|[#12](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/12)|上游前端仍未正确归一化Registry CUDA分类值，跟踪ComfyUI_frontend#15364及已有PR#15468。保持合法classifier，不修改用户安装的前端资产。|
|[#17](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/17)|MPS/Ref2VA声音前导仍需原工作流、参考和输出FLAC确认；trim_exact_target是有完整ASR匹配条件的生成后裁切，不是模型级预防。|
|[#18](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/18)|原生joint双采和时间分块已有实现；空间tile的细节/接缝质量仍未获得通用资格，不能凭可执行就宣称达到全画幅质量。保留特性跟踪。|
|[#26](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/26)|原云端部署、DeepCleaner合法原包和私人参考素材仍需来源方提供，不能从本地节点代码补造。|

更新应针对T8自定义节点源码并重启对应后端；仅刷新浏览器或只更新ComfyUI本体不会载入Python修复。本次维护不修改或重启用户正在使用的服务。
