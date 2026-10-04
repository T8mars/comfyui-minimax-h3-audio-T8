# Kijai 九原件分离工作流（v1.90.0 EXP）

九个家族，每个都有完整生成／保存 LOW 的 `Full_Save` 和只执行 HIGH 的 `Cold_HIGH`，共18张。详细模型配对、下载来源、音频策略与限制见[使用说明](../../../docs/KIJAI_EXPERIMENTAL_LORAS_EXP.md)。旧图不覆盖。

| 家族 | 完整生成／保存 | 冻结 LOW／只跑 HIGH |
|---|---|---|
| DMAD | [Full_Save](DMAD_Full_Save_Reviewed_EXP.json) | [Cold_HIGH](DMAD_Cold_HIGH_Reviewed_EXP.json) |
| PDMD4 | [Full_Save](PDMD4_Full_Save_Reviewed_EXP.json) | [Cold_HIGH](PDMD4_Cold_HIGH_Reviewed_EXP.json) |
| ELM LongLive | [Full_Save](ELM_Full_Save_Reviewed_EXP.json) | [Cold_HIGH](ELM_Cold_HIGH_Reviewed_EXP.json) |
| FlashGen | [Full_Save](FlashGen_Full_Save_Reviewed_EXP.json) | [Cold_HIGH](FlashGen_Cold_HIGH_Reviewed_EXP.json) |
| FL2VA Full Acc8 | [Full_Save](FL2VA_Full_Full_Save_Reviewed_EXP.json) | [Cold_HIGH](FL2VA_Full_Cold_HIGH_Reviewed_EXP.json) |
| FL2VA Pruned Acc8 | [Full_Save](FL2VA_Pruned_Full_Save_Reviewed_EXP.json) | [Cold_HIGH](FL2VA_Pruned_Cold_HIGH_Reviewed_EXP.json) |
| Ref2VA Full Acc8 | [Full_Save](Ref2VA_Full_Full_Save_Reviewed_EXP.json) | [Cold_HIGH](Ref2VA_Full_Cold_HIGH_Reviewed_EXP.json) |
| Ref2VA Pruned Acc8 | [Full_Save](Ref2VA_Pruned_Full_Save_Reviewed_EXP.json) | [Cold_HIGH](Ref2VA_Pruned_Cold_HIGH_Reviewed_EXP.json) |
| Ref Difference（非加速） | [Full_Save](Ref_Difference_Full_Save_Reviewed_EXP.json) | [Cold_HIGH](Ref_Difference_Cold_HIGH_Reviewed_EXP.json) |

先选择自己的模型与素材；LoRA 原件放 `models/loras/`，图片 `10A.jpg` 不随包提供，需重新选参考图。模板不包含私有模型、媒体或 Stage 缓存。

完整运行 `Full_Save` 后，将 LOW StageSave 返回的相对路径和64位 SHA 填入同家族 `Cold_HIGH`。占位路径与64个0不可直接运行，不要填 HIGH 或其他家族的 Stage。Cold 没有 LOW 执行闭包，但更改 LOW 必须重新生成并保存。

两阶段 Relay Plan 与 EAV 独立。Relay 初始文本相同；EAV 初始 `report_only`，不是已应用增强。learned scale1.2；PDMD为960×544 → 1152×640，其余448×256 → 512×288，124帧／24fps。四Acc8继续绝对0:4／4:8联合音频，不能锁未完成 LOW 音频；普通完成 LOW 才锁音。

对应九份固定成片与初始配方已获真人通过，不认证新素材、任意参数、Cold编辑、无限KV、作者完整pipeline或性能。文件名的 Reviewed 仅指这一限定验收。
