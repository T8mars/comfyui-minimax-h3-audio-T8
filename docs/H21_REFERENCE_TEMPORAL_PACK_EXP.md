# 参考视频时间抽样（实验）

这是独立、显式选择的参考条件变换，不是新采样器。输入是原条件节点输出的
`MODEL` 和 `positive`；两个新输出必须一起接 scheduler／guider。生成 `latent`
仍直接接原采样器，原步数、sigma、音轨和 learned 3D 放大器不变。

默认 `0/5` 原对象直通。其它档位只对 `minimax_refs` 中 video／video_audio 的
VAE latent 时间轴取真实索引；图片、参考音频、首尾图、Qwen embedding／token tags
保留原对象，不重新编码 Qwen、不改变上游取图或重采样。

## 档位与真实时间

每五个原 VAE 时间 token 为一组，保留如下原索引相位；不足五个的一组取交集。

| 选择 | 每组保留相位 |
| --- | --- |
| 0/5 | 0、1、2、3、4（不安装 owner） |
| 1/5 | 0、1、2、3 |
| 2/5 | 0、1、3 |
| 3/5 | 0、3 |

例如原 `T=12`，2/5 实际保留 `[0,1,3,5,6,8,10,11]`，不是取前八帧或均匀
改成八个新时间点。原 H3 每 token 覆盖像素帧数按 `(1,4,4,4,4)` 循环；报告列出
每个参考独立的原索引、原像素帧起点、按 Core 24fps 时钟的秒数和原 RoPE 偏移。
这些秒数不是容器 PTS；若原始视频被上游重采样、裁切或变速，须另保留其源映射，
不能凭本报告声称恢复了 VFR 的原 PTS。

物理布局由原完整参考时间轴建立后选取对应行。后续参考、参考音频、首尾图及
生成 video／audio 的原时间坐标保留，不能用缩短后的 latent_t 把它们提前。
所有原空间位置和未抽样行保持；兼容的既有位置修复只保留其实际增量。

参考条件变少可能影响生成画面和声音。这里的“音频不变”仅指参考音频和采样输入／
时钟未被替换或抽样，不是保证联合 DiT 最终输出的音频逐位一致。

## 双采、Relay 与用户组合

LOW／HIGH 从各自原 `positive` 和所选 MODEL 分别接一次，并分别标记 phase；不要把
LOW 的已压缩条件再接一次压缩，或把 LOW 的 layout／行号直接搬到 HIGH。
`phase` 是审计标签，不自动切模型、尺寸、LoRA、强度或采样阶段。

已认证的普通 Relay owner 可以重新绑定实际压缩布局，保留原事件、文字 key spans、
query route 和物理 query times。原 Relay MODEL／条件不修改。未知的 Relay／位置
composer 或 MODEL 计算路径保留原对象并警告，报告“不应用压缩”，而不是阻断原工作流。
现有 model-function wrapper 保留并调用一次；如果它忽略输入／绕过实际网络，只能
报告委托前已准备布局，不能据此声称模型实际消费或 GPU 压缩已成功。

后续用户 wrapper 可以覆盖此入口，审计明确显示 wrapper 缺失或覆盖不足，不自动清除
后加补丁。真实形状、NaN、配对条件、内容 SHA 或已选执行源码被修改仍报告具体错误。

## 内容身份与隔离边界

报告绑定真实原／抽样 latent SHA、索引、条件 embedding／metadata、phase、实现源及
实际布局。已有条件 digester 也能读取这些显式字段。已知条件在实际调用前后核内容；
opaque 对象保留执行，但不能被声明为已认证 producer。没有重新加载或冒称认证原 VAE。

当前 owner 明确 `portable_cache_identity=false`；现有 Stage 身份读取实际回调／owner
后同样回落到 nonportable。不可借描述性 producer 字段、完成计数或 audit 的执行顺序
复用跨进程 Stage／HEAD 缓存。没有新增持久化压缩缓存。

Apply 不修改共享模型 forward、不改全局 PackedLayout 构造器或源 payload。实际调用
复制 payload 和 layout；异常／取消原样传出、不重试采样；请求回调重置统计，cleanup
保留最后观察。多个重叠生命周期标为 ambiguous，不当作单请求可靠计数。

内容核验包含 tensor 扫描及可能的设备到主存传输，布局准备也有 CPU 分配成本；
不能宣称这些开销免费。Audit 的 model-call 计数不是 NFE、CUDA kernel 数、allocator
峰值或已被人审接受的质量。GPU 公平性能、跨进程缓存及人工质量仍未认证；选定的实际媒体范围如下。

## 实际验证与 GPU 接续计划

CPU 回归使用真实 Core PackedLayout、随机权重 tiny H3、BasicGuider 和原 dual-clock
Euler，覆盖多参考／音频前缀、每参考时间映射、0/5 完全恒等、各实际 LOW/HIGH 尺寸、
Relay 两种 query route、原事件偏置、原生节点 schema／validate_prompt、普通下一请求、
异常／取消、后继 wrapper、真实坏 SHA／NaN／形状及现有 nonportable 身份。
合成 Qwen／VAE 夹具只负责输入准备，不代表训练模型、完整 VAE、CUDA 或画质资格。
最终通过数和冻结 SHA 以本地独立验证收据为准，不累计历史轮次。

选定当前 Core 原生 20 步、2/5 参考压缩已实际观察 20 次 embedding 与 1000 次 block 返回，参考时间 token 为 37→23，保留原 mod5 相位和位置；完整 120 帧 736×416／24fps 原生音视频机械通过。原 R35 三例批次因另一个 R5 错误失败，缺少该旧批次 whole postscan；后失败 CPU 的同模型／Core／Python 文件完整性补证不是补造旧 GPU whole。两个成功个案及这个边界保存在本地 `h22-R35-split-review-v1/actual-two-index-v1/merge.json`，没有将原失败批次改绿。

后续未验证组合仍须独立固定模型／有序 LoRA／参考 RGB 和音轨 SHA、原条件 provider、
分辨率、seed、步数和 sigma；保留原 0/5 对照，再分别验证仅 LOW、仅 HIGH、双阶段及 Relay。
每个 case 在 POST 前订阅实际事件，记录真实调用／进度／终态、原及压缩布局 SHA／
索引，核生成 latent 尺寸、音频输入／时钟和实际 learned 3D 路径；不从声明步数算 NFE。
分别测 DiT 段和整片时间、CUDA allocated/reserved 峰值、host RSS／传输及 VAE 时间，
控制预热和缓存，保留原片，不擦线或换音轨。最终完整解码／逐 PTS／听音及人工审片，
独立核人物／动作／尾段和接缝；不可只看 x0 或单个均分。性能无收益或质量失败如实保留。

固定设计参考是 [VPack ee13c64](https://github.com/zechengtang/VPack/commit/ee13c64aec8124c2a9fa64e421d36eed3b7dfb3f)，
代码许可为 Apache-2.0。本实现针对本项目 Core／owner／Relay 合同独立编写，未复制外部源码。
作者单个 SM120 例子的 DiT 时间约 24% 变化只属于其测试，不是本项目结果，更非整片承诺。
