# Prepared Tao / LTX：来源、环境与分发边界

这是未发布候选的复现说明，不是许可授权或任意设备兼容承诺。新入口不会自动
下载安装模型、源码或Python依赖，也不会修改用户主Comfy环境。

## 本地实测来源

| 外部组件 | 本次固定源码/模型修订 | 用途 |
| --- | --- | --- |
| TaoLiveAIGC/TaoMate-H3（GitHub） | ccc1a70adbf7f552a84a0cd7eeac0a6f3d461cad | 原生teacher/clean KV/streaming算法 |
| Lightricks/LTX-2（GitHub） | d151147788a9284cca791edc6ce898007e727fe6 | 原生AV类型、packing、3次Euler及解码 |
| NVlabs/Sana（GitHub） | 144085566a866f9784f3798d4c8d1603f3adbccf | models/minimax_h3/Sol-H3-Spark下转换实现 |
| MiniMaxAI/MiniMax-H3（Hugging Face） | 42ed227ee7df40d41602854ae760620d6eb651fe | 官方底模 |
| TaoLiveAIGC/TaoMate-H3（Hugging Face） | cfa5150cc7b41974a6b657ec6479adc5cbabc611 | Tao适配器 |
| Efficient-Large-Model/H3-to-LTX-Latent-Adapter（Hugging Face） | 1792c42689a0f22de880eaf57a187c6a373a636d | 194M潜空间适配器 |

上述Git源码由用户本机的显式目录提供，不随节点包复制整个仓库。节点包中的
`prepared_backend`为本项目的单卡调度、身份校验、权重搬运和媒体封装代码；
它调用指定上游原生实现，不用普通LoRA＋少量步数冒充完整Tao算法。
旧GPU原型中的8个Tao与5个LTX数学/输入helper已在迁移时逐文件哈希比较。
四个新封装worker作了导入、参数、输入预检等改动，未重新跑完整GPU，不能据
helper相同就宣称新封装在所有配置下已经实测。

当前本地Tao源码的LICENSE标题为“MiniMax H3 COMMUNITY LICENSE AGREEMENT”；
LTX源码的LICENSE.md标题为“LTX-2.x Community License Agreement”。上游代码、
权重和所用模型条款分别保留，不因本节点仓库的许可证而被重新授权。
当前H3→LTX适配器模型卡没有单独明确新的权重许可；不要自行补写为GPL或镜像
上传权重。Sana使用稀疏源码目录，本说明不从缺失的根LICENSE推断无条件授权。
分发前应核对相应固定版本的完整原始条款；本候选不分发任何上述外部权重、
隔离Python运行目录或外部完整源码。

## 实测环境和依赖

Prepared 的资源观察器需要 NVIDIA 的 [`nvidia-ml-py`](https://pypi.org/project/nvidia-ml-py/) 分发包，导入名为 `pynvml`；基础节点不依赖它，故不由基础 `requirements.txt` 自动安装。请在所选 Prepared 运行环境中显式准备该可选依赖，或在独立依赖目录中准备并让自有测试进程使用，不为此替换用户的 Torch/CUDA。缺失时资源观察器必须在启动 worker 前失败；不能改成虚假显存数、取消保护或拿 CPU 输入资格冒充实际推理通过。开发中的分离 Prepared LTX 原生画布已实际暴露过这一缺失，原失败记录保留，补齐依赖后的 worker/完整媒体资格需重新验证。

隔离 import 路径有优先顺序，旧目录中的 NumPy 可能覆盖主环境里的版本，并与主环境 SciPy 冲突。不要通过增加属性别名或吞掉导入异常伪装兼容；在独立目录准备匹配依赖，重新构造清单和验证实际导入模块位置。准备工具会将 Git 忽略且没有任何被跟踪文件的依赖目录作完整资产清单，不把它误标为外层节点仓库的 Git 源码。被跟踪的真实源码仍走原 Git pin/脏源码拒绝门。全依赖导入可能触发上游 CUDA 能力探测；仅符号导入、零已分配 tensor 不能声称 CPU-only、模型推理或音视频通过。

分离 Prepared LTX 的独立 Relay Encode 候选使用原生 INT8 Gemma 和 AV connector 的串行权重搬运，不改变 INT8 权重、FP32 scales 或原始 forward。它不是修改上游 resident cache-builder 的 SM121 限制来冒称该路线支持 SM89，而是另一个显式 worker。固定 provider 对实际文件名、尺寸、原生层数量、缓存 recipe／prompt／特征形状作校验；所有输入权重和依赖目录内容绑定到任务身份。其 private 两事件真实编码和后续实际权重 Relay／组合数值控制已通过，公开节点的原生画布、完整媒体、其它设备和主观效果仍须分别验收。模型、外部完整源码与隔离依赖目录不随节点包分发。

本次为Windows、单张RTX4060Ti16GB、128GB RAM、Python3.12、PyTorch2.10.0+cu130。
这不是最低系统配置或其他显卡/操作系统认证。Tao全底模CPU加载会使用大量内存；
生成和VAE分进程串行，继续保留资源保护线，不以降低保护换取“跑通”。

共同使用现有Comfy Core、Torch/torchaudio、safetensors、numpy、psutil、
nvidia-ml-py及FFmpeg/ffprobe；具体上游导入另依赖其原环境。LTX本次在隔离路径
使用Transformers5.14.1、OpenImageIO3.1.17.0等已准备运行文件，未升级主环境。
环境指纹包含Python/可执行文件、系统，以及相关已安装分发包的版本/位置；
LTX隔离运行目录还单独按全文件清单核对。这不是每个主环境依赖文件都已做
对抗性完整性证明，源码Git pin也只覆盖跟踪源码，不包含无关未跟踪项目。

项目旧入口的`requires-python>=3.10`不能解释成新Prepared流程已在3.10实测。
Prepared实际资格仅为上述Python3.12组合；缺少3.11 `add_note` 的异常兼容测试
只是模拟该API缺失，不构成完整3.10执行认证。

## 包内准备工具

仅以下三个准备CLI随本候选交付，不包含测试控制器、研究服务器、私有请求、
本地交接或样片：

- `tools/prepare_generation_bundle.py`：构造全身份输入清单。
- `tools/qualify_prepared_inputs_cpu.py`：原生CPU输入/提示词/教师一致性预检。
- `tools/migrate_prepared_checkpoint.py`：从原始回执显式迁移已证明来源的样片缓存。

三者的`--help`不加载模型。完整命令和工作流接线见
`examples/workflows/32-prepared-generation/README.md`。准备清单中的路径必须属于
本机实际输入；复制节点目录、改Python/依赖/代码后旧指纹会失效，重新构造/
迁移清单，不手改fingerprint。输入准备、缓存复用和新GPU生成在报告中分别记录。
