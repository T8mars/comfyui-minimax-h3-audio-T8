# Solar 结构重渲染候选入口

本页对应隔离开发版新增的三个实验节点；原771节点顺序完整保留，候选总数774。Solar静音作用域默认关闭，现有工作流不自动改变。2026-10-27，指定同步对照与Solar放大片已获完整人工看听接受，画面、声音通过；尚未部署或发布。

候选只封装既有 `solar_structural.py` 与 `solar_structural_sampling.py`。继续使用用户的模型、CLIP、VAE、LoRA 顺序和强度；未知组合给出未验证提示，不按类名、来源或固定权重 SHA 作准入。文件损坏、真实接口缺失、形状不符和非有限数仍明确报错。

## 三个候选节点

| 节点 | 输入 | 输出与用途 |
| --- | --- | --- |
| H3 Solar · 结构与原声区间 | CLIP、video_vae、结构 IMAGE、提示词、尺寸、长度、source_start；可选首帧、structure_vae、原声 | positive、av_latent、actual_frames、final_audio、报告。结构编码采用已有完整片段路径。 |
| H3 Solar · 静音采样作用域 | MODEL、enabled、silence_file、tensor_key、seed；可选 av_latent | MODEL、报告。默认关闭；开启时在每次预测前为完全可改音频行提供编码静音与同请求固定噪声。 |
| H3 Solar · 实际运行报告 | 最终 MODEL、after_sampling | 最近一次实际请求的调用次数、音画时钟、替换/保留行数及清理状态。排序输入不证明结果来源。 |

静音节点关闭时返回同一个 MODEL，不读取静音文件、不检查可选 latent，也不安装作用域。它不会撤销输入模型上已经存在的 LoRA、Solar 或其他作用域；关闭单个节点不能冒充整条 Solar 分支全部禁用。

开启时 `silence_file` 接受 `models/solar` 内相对文件名或用户明确选择的绝对路径。默认不选文件，不自动下载。`tensor_key` 默认 `silence_170f`；真实 safetensors 张量须为 `[2,32,T]`，由现有读取器转换为 `[1,32,2,T]`。不以零张量代替文件、不循环或补足短静音。可选 `av_latent` 提前检查实际形状和所需长度；真实采样时仍由运行时检查。每次开启执行都会重读文件，避免同路径替换后继续使用旧编码静音。

## 完整接线

1. 使用原有模型、CLIP、VAE 加载节点。用户已有 LoRA 与注意力/控制补丁保持原选择。
2. 串联两次已有 **H3 LoRA Compatibility Loader**：先 fewstep，再 control。两个文件和两个强度分别可选；强度不联动。推荐配方不代表其他选择被禁止。加载器输出报告用于检查实际映射，不由 Solar 审计猜测哪些 LoRA 已加载。
3. 将24fps结构帧、提示词和所选 VAE 接结构编码节点。可选外观首帧接 `first_frame`；其声明坐标须等于 `source_start`。未声明时使用 `-1`，报告只提示坐标未验证，不验证视觉身份。
4. 结构节点的 `av_latent` 和上游 MODEL 接已有 **H3 Dual-Clock Sampler**。短片配方选择 `lcm`、`native_flow`、4步、`shift_video=12`、`shift_audio=3`。它输出 MODEL、SAMPLER、SIGMAS；Solar 不增加采样器实现，也不暗改步数。
5. 双时钟 MODEL 接 Solar 静音作用域，明确开启并选择编码静音文件。可选连接同一 `av_latent` 提前检查长度。用同一个 seed 数值分别连接 RandomNoise 与静音节点。
6. 静音节点的 MODEL 与结构节点的 `positive` 接 **BasicGuider**；RandomNoise、Guider、上述 SAMPLER/SIGMAS 和原 `av_latent` 接 **SamplerCustomAdvanced**。
7. 采样输出接已有 **H3 AV Decode** 和实际视频/音频 VAE。需要原声交付时，解码画面与结构节点的 `final_audio` 接已有 **H3 Safe AV Save**；不把解码出的生成音轨冒充原声。Save 本身已有视频预览，不需要重复保存。
8. 最终 MODEL 与采样结果连接 Solar 实际运行报告；观察最近一次请求，不把节点配置报告当成执行或质量回执。

输入没有原声时 `final_audio` 为 `None`，不会制造音轨。不要把这个空输出接到要求 AUDIO 的 Safe AV Save；可改用已有无音轨视频保存链，或明确提供所选音轨。

静音作用域读取实际 Core DiT 的音画 shift，并尊重 `transformer_options` 覆盖。采样器的 `audio_shift=None` 不自动意味着没有可用时钟。实际消费接口没有可解释时钟时明确报错，不用默认12/3猜测。用户换采样器或网格不会被静默改回推荐配方，其效果也不由该候选认证。

## 源区间与原声

结构时钟要求24fps；非24fps素材须先明确转换。宽高为32的倍数。实际长度不超过请求与源剩余帧数，并向下对齐 `17n+5`，不会复制尾帧补够请求长度。以 `actual_frames` 和报告的 `source_range=[start,end)` 为准。158帧是短片训练配方提示，不是其他长度的准入白名单；超过362帧仍是完整片段计算，不代表因果流式或固定显存。

原声输入的样本零点须对应输入结构 IMAGE 的第0帧。节点只在已选源区间内切片，不推断两份媒体原本是否同步。端点为：

```text
start_sample = sample_at(source_start, sample_rate)
end_sample   = sample_at(source_start + actual_frames, sample_rate)
```

`sample_at` 复用已有全局24fps有理数时钟。44100Hz、源区间 `[3,25)` 的端点为5512和45938，切出40426样本；不能用分别取整的起点加22帧局部时长替代。保持原采样率、声道和 PCM 值，输出独立切片；原声不足明确报错，不补声或重采样。它是最终音轨，不是声音驱动条件。

Safe AV Save 会做其既有声道/长度规范化和 AAC 编码；源切片逐样本保留不等于最终 MP4 音频字节不变。尤其非整除采样率可能出现一个样本的交付长度规范化，查看保存器报告，不宣称无损音频复用。

源音轨若在解码后超过正常幅值范围，保存器会报告具体问题。需要降低音量时，在 `final_audio` 与保存器之间显式连接 Core **Audio Adjust Volume**，保留用户可见的 dB 数值。本次源片的峰值约为1.09023，使用−1 dB后为0.97167；44100 Hz、双声道和290325样本区间保持不变。这只影响最终交付音轨，不改变结构条件、种子、采样或原始音频文件，也不是所有素材必须套用的默认增益。

## 查看实际报告

设置后的 `selected_not_executed` 只表示已安装。执行报告中的行数按 **batch × stereo × audio tick** 计数，并累加成功预测次数；不是32个潜通道的元素数。

- mask 为1的完整可改行采用静音输入；0或部分 mask 行保留进入 wrapper 的原音频。
- `silence_input_partially_delegated_unqualified` 表示只替换了部分行，不是全静音成功。
- `delegated_without_silence_consumer` 表示实际委托路径没有消费该作用域，不能据输出存在声称已生效。
- 失败会保留真实错误，清理请求状态；用户 wrapper、hook 和权重补丁继续归用户所有。
- 外来同名 attachment 保留并提示未安装，不能把兼容旁路冒充成功。

报告不认证输入 latent 归属、LoRA 文件选择、生成音轨质量、真人接受或任意 GPU 组合收益。普通请求、取消、实际整片看听和资源比较仍分别验证。

## 来源与验证边界

结构与静音协议来源：`Enderfga/SolarRenderer@00b89093234a194bb06e12b7df0d2ed354769a8d`，源码为 Apache-2.0；权重有单独 Community License。固定权重参考版本为 `c5512be825f297ed8db390cbeda0289fb0c24779`，仅作来源说明，不强制用户选同一文件。许可文本与 NOTICE 见 [H24 来源记录](../THIRD_PARTY_NOTICES/H24-SOURCES.md)、[Solar LICENSE](../THIRD_PARTY_NOTICES/H24-SOLAR-LICENSE.txt)、[Solar NOTICE](../THIRD_PARTY_NOTICES/H24-SOLAR-NOTICE.txt)。

新增 `tests/test_h24_solar_nodes.py` 的9项用例已在统一冻结后的235项CPU回归中通过，覆盖关闭直通、实际 Core 编码/LCM/双VAE、全局原声端点、未知模型名称与用户补丁、部分 mask 运行报告、失败恢复及静音文件重选。学习式文本编码采用明确的小型测试替身；不以该测试证明训练模型画质。复用底层既有 helper，不复制完整底层矩阵。训练成片的质量和收益结论需另外记录，本页不预判。


## 指定完整训练样片

已完成480×832、158帧、24fps、seed26102641的完整Ref2VA INT8测试。两份指定Solar权重先fewstep后control、强度各1；LCM4步与视频/音频shift12/3，编码静音选`silence_158f`。实际4次前向中观察到目标时间结构位置、首帧位置及4次静音输入消费，两份LoRA各映射208个补丁；不声称每个投影的运行覆盖都已独立测量。

本例Solar采样427.234秒，普通Euler40步3908.625秒。二者协议、步数和条件形式不同，这只是该样例顺序运行时间，不是公平单变量或通用加速倍数。前后普通请求的音画latent逐位一致，模型作用域恢复。完整158帧、声画时间戳和三路同一−1dB原声音轨均验证；原40步的末尾原声保存失败保留原记录，另用不重编码视频的方式恢复交付。

九个固定时刻抽帧中人物、服装、场景和主要动作维持，Solar后段动作位置比普通对照更贴近源片；这支持接入可选实验入口；2026-10-27已另行收到指定成片的完整人工看听接受。源视频本来含短暂白色圆圈图形，不把它误报为新算法造成。没有验证因果长视频、Reactor、所有用户模型或该训练方案的其它分辨率/时长。原声是最后合入的音轨，不是驱动条件。
