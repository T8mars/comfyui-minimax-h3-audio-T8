# 渐进采样的高低分关联噪声（实验）

`H3 Progressive · Paired Noise` 给已有 LOW Sampler 和 HIGH Handoff 提供两路
NOISE。节点不加载模型、不采样、不放大，也不替换两阶段的独立多 LoRA 链。

连接 Stage Plan 的 `low_source` 与原始 `high_source`，把 `low_noise` 接 LOW
Sampler，`high_video_noise` 接 HIGH Handoff。HIGH 噪声只有视频，不能直接接需要
联合音视频噪声的普通 SamplerCustomAdvanced。中间继续使用现有学习型 3D 潜空间
放大器；本节点不能代替放大模型。

| 模式 | 行为 |
| --- | --- |
| independent（默认） | 保留旧路线：LOW 使用 seed，HIGH 视频使用 seed+1。 |
| haar | HIGH 空间噪声每 2×2 块求和除以 2，形成 LOW 噪声；LOW 宽高须各为 HIGH 一半。 |
| bilinear | 显式实验插值；其方差会缩小，不等同于 Haar 的单位方差投影。 |

关联模式先抽取 HIGH 视频噪声，再抽取音频噪声。音频没有缩放，但 RNG 位置和
independent 不同，因此同一个 seed 的跨模式对比，声音也可能变化。不能把它当成
“画面改变、声音严格不变”的实验。内部使用私有 CPU 随机生成器，不修改全局 RNG；
不在 LOW 采样期间保留整份 HIGH 噪声。

已验证原生 CPU RNG 的多种精度、batch_index、最大 seed、Haar 方差与相关性，以及
真实小 Core LOW→外部放大输入→HIGH 两阶段消费者。CPU 测试中的放大器是明确的
测试替身，不能证明学习型放大效果。独立 GPU 对照另行使用实际已安装的 H3 3D
放大权重、EMA B、原生 Euler 6+2、1024×576、120 帧成片；两份指定画面已通过人审，声音未审。

上述独立/Haar 两组 GPU 对照已各自执行，均观察到 6 次 LOW 和 2 次 HIGH 前向，
实际学习型放大节点参与计算，最终视频与音频完整解码。并未仅凭方差或相关性
指标把 Haar 判为更好；指定反馈中用户偏好 Haar，两组声音仍未审。

2026-11-08 反馈更新：两份画面通过，用户偏好 Haar，两份声音未审。
该反馈取代上面的待审状态，范围见[审片记录](NOV05_REVIEW_20261108.md)。

这是渐进首采中的噪声选择，不是完成一采后的 2K Refiner，也不修改已接受的
4+4 长视频续接路线。不保证更清晰、更快或更省显存；默认仍为 independent。
