# 参考编号、预览续跑与道具检查（EXP）

三个可选只读入口追加到旧节点列表末尾。旧工作流无需迁移，不改提示词、Stage、采样或导演台草稿，不自动采用／排队。

## 参考编号／实际执行摘要

连接条件节点输出的 `media_map_json`，可声明角色对应的实际 Picture／Video／Audio 标签。角色多图不会被合并成新编号；视频音轨占用Audio槽时以实际media_map为准，不按图像序号猜音频序号。

角色声明示例：

```json
[{"role_id":"A","tags":["Picture 1","Picture 2","Audio 2"],"expected_labels":{"Audio 2":"ref_audio_0"}}]
```

Subject／S无显式绑定时显示unknown。绑定声明不是身份或声纹证明。输入map标注为调用方提供的原生map，不假称本节点重新编码核实了素材。

可接SIGMAS或完成Quality Stage：前者报告实际输入表／SHA／间隔数，不把间隔数当forward计数；后者沿用完整tensor、receipt、producer、clock校验后报告实际完成NFE。计划、实际、unknown分开。

## 预览选片·继续／恢复卡

只选择一个真实 typed Stage/MID，或现有RES历史的相对路径和SHA。卡片显示恢复节点、下一节点、还需的输入与额外NFE，不代替恢复节点的当前MODEL／条件校验。

| 选中结果 | 读取代价 | 下一动作 |
| --- | --- | --- |
| 四质量Light完成LOW8 | 0 NFE | 独立Community HIGH3，额外3 NFE |
| 四质量HIGH／单采完成态 | 0 NFE | 直接解码，0 NFE |
| 原FreeVideo完成LOW8 | 0 NFE | 保留原尾2默认；显式尾1–7另选 |
| 原Split MID4 | 0 NFE | 真实含噪音频继续后4，非冻结完成LOW音轨 |
| 完成的RES Stage | 0 NFE | 读完成结果；不是POST4历史恢复 |
| RES POST4历史文件（原8步） | 0 NFE | 同SIGMAS轨迹余4，保持原solver history |

RES文件仍位于 `output/MiniMaxH3/res_checkpoints`。文件完整性不等于当前权重／有序LoRA／条件／compiled assets相符，原恢复器在执行时核实。不从MP4或x0反造原始latent，不让未知补丁获得可持久复用认证；普通旧组合仍可运行。

## 道具ID／左右手连续性检查

输入显式全局秒数与道具状态，输出手工可采用的文本和冲突报告。相同kind的两个杯子用不同ID，不自动认作同一物体。

```json
[{"t":0,"props":[{"id":"cup_a","kind":"cup","holder":"A","hand":"left"},{"id":"cup_b","kind":"cup","holder":"A","hand":"right"}]},{"t":3,"props":[{"id":"cup_a","kind":"cup","holder":"B","hand":"left"}]}]
```

若交接是预期动作，可显式声明 `[{"t":3,"id":"cup_a","from":"A","to":"B"}]`。kind变化、未声明交接、同手多物、holder／hand缺失分别报告；不会因文本检查就声称画面检测、生成几何锁定或已确认片尾事实。

新增CPU合同检查与真实画布执行分别记录。低成本只读验证不需要重新采样旧通过素材；人工看听验收仍由用户完成。

当前指定只读画布已保存、关闭重开并执行：真实完成LOW8／HIGH3与已有匹配producer的RES POST4分别读出余3／0／4，实际参考摘要读取HIGH3完成NFE，显式道具交接正常。0新采样。旧producer SHA不符的RES历史仍被原恢复门拒绝，不覆盖旧文件或修改门来消除失败。指定map与已执行两源图对应；同角色多图及视频自带音轨的报告合同不意味着本次又完整生成了多视频／多声音案例。
