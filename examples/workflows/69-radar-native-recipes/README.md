# RADAR 原生配方通用模板（本地 EXP）

四张图均可导入，素材是明确占位；先选择自己的文件及正确模型，再执行。文件／模型／已保存阶段不随图打包，不自动下载或接受质量，不覆盖旧图。

| 工作流 | 用途 |
| --- | --- |
| Union2_Native40_Full_Save_EXP.json | 原生40步、CFG1、无Turbo；外置源与白色重绘MASK，完成阶段显式保存 |
| Union2_Native40_Completed_Cold_Delivery_EXP.json | 填该完成阶段的真实path／完整SHA，仅两VAE解码，零采样 |
| MV_CastSolo_AB_Full_EXP.json | 独立A/B参考、明确逐镜角色／歌词，原歌末尾只mux一次 |
| MV_CastSolo_AB_Same_Output_Resume_EXP.json | 与Full同执行节点／设置，保持同output根、chain、素材和producer；已完成零采样返回master |

Union2模板源须显式448×256、124帧、24fps；改变画布需同步条件、Apply、源与MASK。音频及黑区RGB不保证逐值保持。此前彩色方块失败的Turbo4+4不作推荐图。

Cast模板边界／分配只示意两镜10⅓秒。换歌曲要改实际scene_plan、完整覆盖assignments；空歌词表示未提供，不猜原文。首次用新chain，Resume复制原Full设置，不能改素材后继续借同一chain，也不是任意移址解码。

相关：[Union2边界](../../../docs/H3_FUN_UNION2_EXP.md)、[CastSolo边界](../../../docs/MV_CAST_SOLO_EXP.md)。原生实际材料的机械资格与这些通用占位图的接线／序列化资格分列，均不代表人审通过。
