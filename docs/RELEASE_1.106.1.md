# v1.106.1：Core 0.45 FinalLayer兼容

修复新Core把FinalLayer的`shifts`改为`grids`后，旧适配器误报不完整接口的问题。根据真实接口保留旧四参数/shift调用；新接口直接使用Core的PDD网格及实际头数，显式网格和用户delegate保留。真实内部异常只执行一次并继续传出。

保留804个节点、现有参数和22套已审UI/API工作流。此次没有改模型、LoRA、采样步数/sigma、声音或原工作流JSON，也未重新生成或借用新的画质验收。

当前Core0.45的相关用例去重后834项通过、6项前提缺失跳过；Core0.43为226通过/30跳过，Core0.41为223通过/33跳过。跳过原因、首次失败、两份PR和15项Issue的处理边界见[社区核对说明](COMMUNITY_ISSUES_20261117.md)。原有通过视频/工作流仍按[v1.106.0验收范围](NOV12_ACCEPTANCE.md)使用。

[#31](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/31)的Python3.13问题已由更完整实现覆盖；[#29](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/pull/29)因真实VFR误改等风险不合并。Registry激活、上游CUDA误报、MPS声音前导、空间tile质量及外部云端原材料仍分别跟踪，不用关闭条目冒充解决。

安装更新后重启对应ComfyUI后端。本次发布不替换或重启用户正在运行的服务。GitHub发布、Registry上传和Registry可安装状态分别核验。
