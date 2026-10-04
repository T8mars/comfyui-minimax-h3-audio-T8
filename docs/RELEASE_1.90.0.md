# v1.90.0 · Kijai 九原件与独立双采

本次正式 GitHub 版本新增一个 EXP 节点和18张分离式工作流；旧节点、默认值和旧图保留。功能仍标 EXP，正式 Release 不代表所有素材／参数质量认证。

- 四个 Kijai Acc-8Step Full／Pruned、FL2VA／Ref2VA 原件使用独立相对32头入口，保留 backbone、AdaLN 与 bias，执行绝对 LOW0:4／HIGH4:8联合音频。
- DMAD、PDMD4、ELM、FlashGen 和 Ref Difference 使用各自对应加载器与采样配方，不混用；Ref Difference 不是加速 LoRA。
- 九个家族各有 Full_Save／Cold_HIGH，MODEL、条件、Noise、EAV与Prompt Relay外置独立。保留正常中文短句、learned1.2及完成态音频策略。
- 只读 Core Bypass 内容身份及默认 LCM 的显式识别支持上述已验收路线；不清空用户补丁、不替换旧采样数学。

九份指定完整双采音画已由用户逐项通过，原生画布保存／重开核对完成。公开模板仅清理私有注释与元数据，并将本地 LoRA 别名换为同一原件文件名；不包含参考图、模型、媒体或实际 Stage。Cold 要填写用户自己的 LOW 路径／SHA，编辑后的质量不自动获批。

发布副本受影响的八个完整 CPU 测试范围共145项通过，无跳过、排除或源码变更，CUDA未初始化；另行核对18张模板执行图与原生保存来源。此回归不是全仓或全部参数组合的质量认证，发布时不重跑已通过视频。

操作入口：[18张工作流](../examples/workflows/71-kijai-experimental-split/README.md) · [模型配对和使用说明](KIJAI_EXPERIMENTAL_LORAS_EXP.md)。二步PDMD、QuantFunc、Nunchaku不在本次范围内。GitHub发布与Comfy Registry审核／可安装性独立，不能将上传成功当作Registry已激活。
