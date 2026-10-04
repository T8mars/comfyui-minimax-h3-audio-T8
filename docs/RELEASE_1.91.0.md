# v1.91.0 · FreeVideo FP8 与独立双采（EXP）

正式版本新增12个FreeVideo节点与6张可编辑前端工作流。保留旧620个入口、默认值与既有工作流；功能仍为EXP，不代表任意参数或素材的画质保证。

## 两种独立采样

- 原8+2：完整LOW八步→外置learned放大→HIGH原时钟末尾两步；保留已完成LOW音频，支持Stage保存与冷载入。
- 真正4+4：原八步表前四步→外置learned放大→后四步，总8NFE。独立MID保存video_x4、partial video x0及未完成audio_a4；HIGH视频重新加噪，音频从实际a4继续联合更新，不能冻结半成品。
- EAV与Prompt Relay均有本家族外置节点，可分别连接两端。Relay输出模型与条件须成对；不是把普通Core MODEL补丁套到自定义引擎。

## 已完成验收

两条8+2和一条真正4+4的指定完整5秒音画已获用户通过。六张Full/Cold来自实际原生画布保存；公开模板仅清理私有缓存路径、审核信息与参考音频文件名，采样参数、接线和布局不变。Cold模板必须填写同家族Full实际产生的manifest路径和SHA，空值不是有效缓存。

测试配方为448×256→原learned 1.2×实际512×288、124帧采样同步裁到120帧、24fps。没有重跑已通过片或追加Cold GPU矩阵，不将CPU恢复检查当作新增画质验收。开发受影响范围141项CPU通过、CUDA未初始化；固定四种layout的小方程对照不等于学习权重验收。4+4实际LOW EAV增益1..1.0507023、HIGH为1，不能当作一般可见增强承诺。

## 安装与边界

使用独立、内容身份绑定的FreeVideo Python引擎与官方rowwise FP8权重，接原生H3条件及标准AV LATENT；不是普通UNET格式转换，也不升级主ComfyUI环境。依赖、模型、配置、素材及Stage不随包，首次使用须按[环境准备](FREEVIDEO_EXP.md)完成配置及内核检查。

派生音频参考表与非线性文本种子Relay均明确为EXP，不承诺原投影重新GEMM逐位相同、论文softmax等价、任意LoRA/参考画质、作者2×配方、RNG或速度等价。在线线性LoRA检查实际目标；进程取消只管理本任务，不卸载全局模型。QuantFunc、Nunchaku和已关闭自动任务不恢复。

[六张通过版工作流](../examples/workflows/72-freevideo-split/README.md) · [详细说明与许可](FREEVIDEO_EXP.md)。GitHub正式发布与Comfy Registry激活/可安装性为不同状态，Registry不作为本次发布等待条件。
