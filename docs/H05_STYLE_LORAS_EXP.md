# H05 新风格 LoRA（本地资格中）

沿用现有 MiniMax H3 LoRA compatibility loader，不新增另一套权重应用器，
不改变旧 LoRA、Bridge、采样器或工作流默认值。权重不随插件发行。

## 当前核验

| 来源／固定版本 | 文件与结构 | 当前边界 |
| --- | --- | --- |
| [Animatediff-style](https://huggingface.co/Charlietooth/Animatediff_style_Minimax_H3/tree/303ad23a7854b4f04fe1464237fb27248465196b) | `animatediff_style_h3.safetensors`，416张量／208对／rank16 | 实际下载SHA与原生解析、目标尺寸、meta模型登记通过；来源未提供明确许可证，尚未取得此风格GPU／人审资格 |
| [PixelTune stable v3](https://huggingface.co/ij/PixelTune-MiniMax-H3-LoRA/tree/c048680232a2cde9bad9fbfbc556024b00e89525) | `pixeltune_h3_full_canvas4_v3_step_01000.safetensors`，416张量／208对／rank32 | 原件原生解析及固定 INT8/ConvRot 底模的真实画布50步／5秒、208目标实际应用通过；完整音画和Stage已核，待人审。不是v12或作者BF16数值复现 |
| [SOLRICKS 3D](https://huggingface.co/SOLRICKS/3D-Animation-Style-MiniMax-H3/tree/974246c6b2d03769fa6c820d02c1d0f370e46583) | 作者列出rank16 step1250权重 | 官方manual gate，需要用户自行取得访问资格；未绕过／接受条款，未取得权重，不冒称适配通过 |

前两份原件都能被本机原生 Core LoRA parser 完整消费，不需要另造
ComfyUI转换件。核对基于实际已下载的A/B张量及本机FL2VA pruned INT8/ConvRot
底模的真实header尺寸，原生meta模型只用于结构和登记。

后续已用 PixelTune 原件的一对真实 rank32 因子，与 INT8/ConvRot 底模
`blocks.0.attn.out_proj` 的实际矩阵执行 CPU 原生补丁：FP32 增量和真实
Core GEMM 逐值一致，ModelPatcher 的 INT8/ConvRot 重量化有限且确有变化；
原矩阵、尺度和文件未改。**仅覆盖208个目标中的1个，不是完整模型应用、
逐目标数值验收**；重量化存在舍入差，不能称与FP32结果逐值相同。

随后 PixelTune 指定真实画布已完成 512×288、120帧、24fps、5秒及50次去噪，
208个补丁目标实际应用、0遗漏，原生保存关闭重开与完整音画／Stage均有独立证据。
该整片资格与上面单矩阵CPU对照分开记录，不扩大为208个目标各自FP32数值对照，
也不证明作者BF16／DiffSynth路径或任意Turbo双采逐值相同。像素风、声音和口型
仍待集中人审；不重复采样。

Animatediff-style 是环境／主体连续形变风格，不是 AnimateDiff 运动模块。
作者给出 `animatediff_style` trigger，但未给出可核实的底模任务、训练alpha或
唯一推荐strength，不从名称或rank猜配方。

PixelTune作者声明训练rank／alpha为32／32，runtime strength为1.0，
Comfy QKV排列，训练使用FL2VA pruned BF16底模。该声明不证明任意Ref2VA或
Turbo双采已经具有像素风质量。4×像素格、50步、静音样片等是作者配方条件，
不得把短双采测试叫作者数值复现；原生音频生成也需单独验收。

## 不混淆三种alpha

训练LoRA的alpha/rank、用户选择的LoRA strength、语义Bridge alpha三者独立。
配方卡显示已知训练字段、配对矩阵观察到的rank，以及文件内未限定的 `alpha`
元数据；未限定字段只显示，不自动解释或套入strength／Bridge。

PixelTune原件的通用metadata `alpha="32"` 不是每个模块的 `.alpha` 标量张量。
本机原生parser对这份无标量alpha的普通LoRA使用原来的因子尺度；不得把32填成
运行strength，也不能为了配方卡把额外 `.alpha` 张量注入原件。

既有配方卡仍是“预览差异 → 明确确认 → 可撤销 → 用户另行保存”，不自动选底模、
升级v12、改步数、排队或允许许可证访问。SOLRICKS访问和Animatediff许可缺口
只限制本项目将该来源认证成推荐配方，不封禁用户普通loader或其他合法来源。

2026-10-08 官方公开元数据复查：Animatediff 固定revision未变，模型卡仍没有
明确许可字段或LICENSE文件；公开可下载不当作许可已解决。SOLRICKS 当前revision
为 `84208cdfded98835eb3b41af545371bc02b50975`，列出的同一权重LFS SHA仍是
`28b56d2240990436fba8d9a1fa06310c971a7ef95d3046059118445e0c0a4ef0`。
公开README／LICENSE可读，但API仍为manual gate，NOTICE返回401，权重尚未获取。
没有接受条款、提交国家／联系信息、借用其他账号或绕过门禁；新revision不自动
替换原固定配方。需要用户自行处理访问资格后再继续这一来源的实际权重资格。
