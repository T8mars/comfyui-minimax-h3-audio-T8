# H05 外部 Encode 图像参考导入（EXP）

新增独立 `MiniMaxH3ExternalReferenceImageImportEXPT8`，不改变旧 Reference Load、
Create、Save、Route、Apply、外置 EAV／Prompt Relay 或任何采样器。

## 接线与限制

1. 将合法可用的外部 `.safetensors` 放进配置的 `models/refmods` 目录，选择文件，
   填入**整个文件的 SHA256**及从 1 开始的成员序号。
2. 提供作者编码时的实际 RGB 原图，尺寸必须正好是该 latent 网格的 16 倍。
   不会自动缩放、裁切、猜底模或归一化。必须连接对应原生 H3 视频 VAE。
3. 明确确认导入与来源使用。节点使用当前 VAE 重新编码原图作验证，
   比较成功后保留的是**外部原 latent 字节**，不是偷偷替换成重建结果。
4. 输出接现有 Route → Apply；需要落盘时接既有 Save 并明确确认新文件。
   Save 保持 create-only，不覆盖原外部文件或旧用户参考包。

只有 v4 单体或 v5／RefLoRA 参考半部里的一个 `mode=encode` 图像成员。
不加载其他成员或 LoRA 权重，不同时 attach 与 Apply。包内路径、配置、曲线、
声纹／作者／编码器声明均不执行或采信。本图像入口不接视频或音频；新独立入口
及其指定真实权重验收范围见 [AV 导入说明](H05_EXTERNAL_AV_IMPORT_EXP.md)。
训练／池化／未知模式仍未资格。
选定张量和原 RGB 使用现有 256 MiB 包预算；文件头使用现有 1 MiB／8192 描述符预算。

## 两种显式精度

- `native_exact`（默认）：外部 latent 与当前 VAE 原输出在 shape、dtype 和原字节 SHA
  上完全一致。没有容差匹配或隐式类型转换。
- `author_encode_fp16`：固定作者 Encode 路径实际执行 `z.to(torch.float16)`。
  将当前 VAE 输出作同一个显式转换后再逐字节比较。若原输出为 FP32，则这是
  精度转换，不宣称 FP32 无损、相同生成结果或借用已通过的 N01 画质审核。

依据固定[作者源码](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/blob/6131c1f62c612a074d6aac8435c26ffa19c00563/nodes.py)
及 [v5 格式](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/blob/6131c1f62c612a074d6aac8435c26ffa19c00563/BUNDLE_FORMAT.md)。
本实现不打包或调用第三方执行代码。

## 证据边界

匹配证明选定外部张量对应于这一次实际 RGB／原生 VAE／明确精度的输出；
不认证外部作者或编码器历史，不证明素材许可、不接管 LoRA 权重半部，
也不证明整个多成员文件可用于推理。来源确认是使用者声明，不是法律证书。
不匹配直接报错；重新编码不同原素材属于重建，不会被称作本次成功导入。

8 项新的定向单元边界通过；684 节点注册的原683完整 finalized schema 精确前缀不变。
已使用固定作者 H3RefMod artifact class 与 bundle writer，生成一个已有真实原图／原生
编码结果的 v5 FP16 样本，再经**原生 SaveAs→Save→关闭→侧栏重开→Run**完成验收。
唯一运行 47.61 秒，15 执行节点／15 边、无缓存节点、**0 DiT NFE**；当前真实 VAE
重新编码结果与已有 FP32 原件逐字节一致，外部原 FP16 latent、完整 RGB 和实际
producer 进入新 T8 包，create-only 保存、重新读取验证、Route 与 fresh Apply 均通过。
输入原件、既有包、365保护文件与旧工作流未改。

这是真实编码／导入／条件接线资格，不是新视频成片、人审、第三方任意原件或
FP16与FP32相同画质资格。测试使用的原图、模型、writer执行回执、画布和私有报告
不随插件发行；不借用原 N01 人审来批准这个精度变化后的成片。
原来缺原 RGB／PCM／producer 的社区视频＋音频样本继续 unsupported，原件和失败证据保留。
