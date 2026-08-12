# MiniMax H3 Advanced VRAM Policy 验证报告

验证日期：2026-08-12
插件版本：1.15.1
ComfyUI：`cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44`
GPU：RTX 4060 Ti 16GiB
DynamicVRAM：启动日志明确显示已启用，`comfy-aimdo 0.4.13`

## 结论

保留显存策略对该工作流有效，但“有VBAR且虚拟内存足够就不会OOM”不成立。

- 无策略虽然成功，最低余量只有41.879MiB；
- 固定2GiB把余量提高到343.086MiB，仍未达到512MiB门槛；
- 固定3GiB达到528.828MiB，只比门槛高16.828MiB，不适合作为推荐值；
- 固定4GiB的三冷三暖全部成功，最差冷态余量1028.117MiB、最差暖态余量1401.415MiB；
- 同seed无策略、2GiB、3GiB和4GiB的解码视频帧与PCM哈希逐位一致；
- 因此示例采用`fixed_total_reserved_exp=4.0`且`clean_before_load=false`，但仍保持EXP，
  不标`memory_safe`或`never_oom`。

## 严格控制变量

- FL2VA pruned INT8基座；
- 27.69MiB `blocks_25_49_video_audio_exp` Hybrid artifact；
- Qwen3-VL NVFP4文本编码器；
- H3 video VAE FP16和audio VAE FP32；
- `10A.jpg`视觉参考；
- 736×416、124帧、20步；
- `dual_clock_euler + native_flow`；
- 基线/2GiB/3GiB/4GiB sweep固定seed `2608125201`；
- 关闭预览，验证器以0.1秒轮询`/system_stats`；
- 每个冷态处理使用新的ComfyUI进程，不继承策略和模型缓存。

`tools/validate_h3_vram.py make-policy-pair`自动保证除有类型的Loader policy依赖外，模型、
artifact、Conditioning、采样、seed、latent和输出控制一致。项目比较器确认基线与处理控制输入相等。

## 真实显存结果

| 运行 | 状态 | 耗时 | 设备峰值 | 最低余量 | 峰值节点 |
|---|---|---:|---:|---:|---|
| 无policy cold1 | success | 282.546s | 16337.621MiB | 41.879MiB | AV Decode |
| 固定2GiB cold1 | success | 281.610s | 16036.414MiB | 343.086MiB | AV Decode |
| 固定3GiB cold1 | success | 279.406s | 15850.672MiB | 528.828MiB | Conditioning |
| 固定4GiB cold1 | success | 272.891s | 15351.383MiB | 1028.117MiB | Conditioning |
| 固定4GiB cold2 | success | 270.875s | 14992.379MiB | 1387.121MiB | Conditioning |
| 固定4GiB cold3 | success | 289.265s | 15195.613MiB | 1183.887MiB | Conditioning |
| 固定4GiB warm1 | success | 241.562s | 14978.085MiB | 1401.415MiB | AV Decode |
| 固定4GiB warm2 | success | 236.985s | 14956.367MiB | 1423.133MiB | polling peak |
| 固定4GiB warm3 | success | 237.235s | 14779.450MiB | 1600.050MiB | AV Decode |

4GiB相对无策略冷态最差值至少增加986.238MiB余量。三次暖态运行前基线的最大正向连续增加
只有12.25MiB，低于项目预设的256MiB/2%阶梯门槛；暖态峰值范围198.635MiB。

## 输出与数值一致性

三份暖态输出均通过：

- 736×416 H.264；
- 124帧、24fps、视频5.166667秒；
- 32kHz双声道AAC、音频5.152秒；
- 无执行错误或OOM。

固定seed的无policy、2GiB、3GiB和4GiB输出解码后只有一个视频BGR SHA-256和一个PCM
SHA-256；四个处理的像素和音频样本逐位一致。容器文件大小可因封装元数据不同而变化，不能替代
解码内容哈希。

## 主机内存与科学边界

矩阵结束后的本机快照：127.834GiB RAM、252.834GiB commit limit、209.651GiB commit
headroom。说明本机有足够的权重换页空间，不代表任何配置都不会OOM。

VBAR主要管理模型权重页，不能保证以下分配成功：

- Transformer activation与attention workspace；
- VAE/CLIP临时分配；
- CUDA上下文、分配器碎片和其他GPU进程；
- pinned host memory与系统commit；
- 更高分辨率、更长帧数、多参考、长视频或语音链。

尚未覆盖0.6M/362帧、1080p、Long Video、Speech、其他GPU、跨GPU、并发CUDA程序和低commit
环境。因此4GiB是当前精确工作流的保守起点，不是全局安全承诺。机器可读汇总保存在本地忽略目录
`artifacts/vram-policy-validation/summary.json`；原始报告、启动日志和工作流pair位于同目录。
