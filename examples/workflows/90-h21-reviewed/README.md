# H21：已审来源工作流

80份按功能命名的画布工作流，另有6份普通提示词副本。可直接拖入ComfyUI；[完整机器清单](index.json)记录每份来源成片及JSON的SHA。

## 先选这些

| 要做什么 | 工作流 |
| --- | --- |
| 原生8步控制 | [Native8_full](01-control/Native8_full.json) |
| 标准4+4控制 | [Standard4plus4_full](01-control/Standard4plus4_full.json) |
| 8秒竖屏标准双采 | [real8_Standard4plus4_full](01-control/real8_Standard4plus4_full.json) |
| 8秒竖屏HyperFlow放大双采8+4 | [real8_HF_Upscale8plus4_full](01-control/real8_HF_Upscale8plus4_full.json) |
| 8秒竖屏HyperFlow放大双采4+4 | [real8_HF_Upscale4plus4_full](01-control/real8_HF_Upscale4plus4_full.json) |
| 区域裁剪＋分离4+4＋中间帧 | [middle_HF_Split4plus4_RegionCrop](01-control/middle_HF_Split4plus4_RegionCrop.json) |
| 安静声音＋HyperFlow中间帧 | [HF_Continuous4plus4_middle_ON](02-keyframes/HF_Continuous4plus4_middle_ON.json) |
| PTD完整4步／连续2+2 | [Full](06-ptd/PTD4_Full_T2VA_EXP.json) · [2+2](06-ptd/PTD4_2plus2_T2VA_EXP.json) |
| PTD完整6步／连续3+3 | [Full](06-ptd/PTD6_Full_T2VA_EXP.json) · [3+3](06-ptd/PTD6_3plus3_T2VA_EXP.json) |
| FreeVideo参考音频High修正版 | [FreeVideo_ref2va_audio_high](05-freevideo/FreeVideo_ref2va_audio_high.json) |

## 分类

- [区域／裁剪／双控制／阶段保存](01-control)：原生、标准双采、HyperFlow单采／连续／分离／放大各自保留参数。
- [首中尾关键帧](02-keyframes)：OFF／ON来源分别保存，不把中间帧当首帧或复用失配的高分条件。
- [低显存与Core回归](03-memory)：投影行分块、INT8行分块与组合；普通→PDD→普通的来源各自保留。
- [参考优化与Veda](04-reference)：参考时序打包、逐步缓存、独立tile预算，不是统一速度排行。
- [独立FreeVideo H21 v4](05-freevideo)：T2VA／参考音频各Light、Medium、High、Max，旧FreeVideo家族不替换。
- [独立PTD专家](06-ptd)：总4／6步和同画布连续2+2／3+3，不是普通加速LoRA。
- [收尾事件](07-ending)：显式两事件原生8步来源。

程序用的API图独立放在[workflow_api/h21-reviewed](../../../workflow_api/h21-reviewed)，不混进前端工作流菜单。普通提示词副本以`_普通提示词`结尾，移除时间路由但保留采样控制；这些文字改写副本不继承原片画质资格。

## 使用前配置

模型、个人首尾图、控制视频及音频不随包分发。每图首个说明框列出资源要求，请在对应加载节点改为自己的文件；公开路径`H21/...`只是素材槽位，不会自动下载。换素材、模型或LoRA后效果由实际结果判断。

竖屏8秒放大双采使用本次通过的LOW448×800→原学习型3D→576×1024，保留标准4+4、HyperFlow8+4或4+4的各自时钟。不要换回旧288×512失败配方或把放大节点旁路。

FreeVideo须按[准备说明](../../../docs/H21_FREEVIDEO_RUNTIME_EXP.md)生成自己的v4配置，再填`runtime_config`。High是20次调用的近似缓存，并非20次完整计算或偷偷切换Max。

PTD先准备原精度底模与安全转换的4／6步专家；分别放`models/diffusion_models`与`models/ptd`，其它模型按图选择。冷TAIL示例的HEAD路径和SHA故意留空，必须填入自己同源完成HEAD的真实文件与完整SHA。

2026-10-23原84组与2026-10-24替换10组（9条不同AV）均获人工通过。80张图是可移植的生产节点来源图；直接算子测试、私有MC集成和额外机制探针只保留本地，不伪造可用工作流。MC公开独立节点见[算子说明](../../../docs/H21_MC_SEAM_EXP.md)。

此次导出保留执行数值、模型与LoRA顺序及接线，完成真实Core的API／序列化往返CPU校验，没有为保存重复采样。路径适配、任意新素材、全设备、性能及未验证组合不继承固定样片资格。旧工作流完全保留，详见[版本范围](../../../docs/RELEASE_1.102.0.md)。
