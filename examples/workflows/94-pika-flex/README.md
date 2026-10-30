# Pika H3 Flex · 6步已审工作流

三图均使用新节点`H3 Pika Flex · 6步采样准备 (EXP/T8)`，输出原生LCM、作者sigma和12/3时钟。
先更新本项目源码、重启对应Core，才能使用新节点；旧Core不因导入JSON自动获得Python节点。
2026-10-30三套指定样片的画面均通过人工审查，用户已授权保存和发布。
导出反馈中声音仍为未审核，原状态保留；不扩展为其它分辨率、时长或组合的人工验收。
三份JSON保留原浏览器保存字节、接线和参数，图内旧候选说明的最新状态以本说明和[index.json](index.json)为准。

| 工作流 | 输入与用途 | 交付 |
| --- | --- | --- |
| [PikaFlex_t2va_6step_EXP.json](PikaFlex_t2va_6step_EXP.json) | 文字生成、FL2VA底模与FL2VA Pika LoRA | 832×480，120帧/5秒 |
| [PikaFlex_fl2va_6step_EXP.json](PikaFlex_fl2va_6step_EXP.json) | 首尾图、FL2VA底模与FL2VA Pika LoRA | 832×480，完整124帧/约5.17秒 |
| [PikaFlex_ref2va_6step_EXP.json](PikaFlex_ref2va_6step_EXP.json) | 参考图、Ref2VA底模与Ref2VA Pika LoRA | 832×480，120帧/5秒 |

两模型已下载时直接选择现有文件，普通LoRA加载器强度1.0，不需要再转换或下载。
首尾/参考图请在LoadImage里选择自己的图片，画布预设文件名仅对应本机测试素材，未捆绑人物图片。
图中8组注意力与8块FFN是已选低显存配置；用户可按自己的配置调整，不作组合白名单。

每图单次联合采样6步，BasicGuider对应CFG1，保留本次模型生成声音。
首尾帧路线保留全部原生124帧，避免为凑整5秒丢失最后条件对应的画面。
不会自动叠加其它加速LoRA或开启2K细化；已有普通/风格LoRA可由用户自行连接和判断效果。

接线、数值配方、上游地址及验证范围见[使用说明](../../../docs/PIKA_H3_FLEX.md)。
