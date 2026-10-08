# H05 Q01：已有 RES 的联合音画资格边界

原生 RES8 代表复用 ComfyUI Core 的 `res_multistep`。后续另追加独立的
[RES 历史步边界 EXP](H05_RES_HISTORY_EXP.md)，不修改 Core 选择器或 solver，
不更改旧 `dual_clock_euler`、蒸馏配方、调度或工作流默认值。

## 已检查

实际 Core CPU 小模型检查使用现有 Dual-Clock Setup：
`res_multistep / native_flow / 8 steps / video shift 12 / audio shift 3`。
它选择 `ModelSamplingAV`，输出有限、形状一致的 video/audio latent，且不改写
输入 latent 或 noise mask。这个资格不是完整权重的 GPU 成片或画质批准。

实际 Core 的确定性 RES 在一个共同 sigma 序列上可对打包流运行。
解析 CPU 见证中打包与分流计算逐值一致，但这不是 vLLM 两个独立 solver／物理时钟
的数值复现，更不是本机速度收益证明。

## 不能将 latent-only 4+4 当成连续八步

Core RES 每次调用初始化 `old_denoised` 和 `old_sigma_down`。
即使第一次返回的是真实 `x_sigma`、并将剩余 sigma 正确传给第二次调用，
前四步的多步历史仍然丢失。非线性 CPU 见证明确观察到它与连续八步不同。

因此现有 Trajectory／NFE Resume 的 Euler-only 守卫保持不变；不能删守卫或将
RES 放入旧 latent-only 恢复流程。完整 LOW 后另启 HIGH 是独立阶段，不是这个
同轨恢复问题，也不能以阶段独立为由宣称恢复历史完整。

同轨 RES 恢复需要独立保存并验证实际 solver history、原条件、坐标、sigma
边界及 carry state。现已新增独立 EXP 实现并通过12项新 CPU 检查：第4步后保存，
新进程只运行余下4步与连续8步逐值一致；实际 Core 的 packed AV／MASK 包装也一致。
这12项属于 `test_res_history_exp.py` 和 `test_res_history_setup.py`，不是重复下方
旧12项。上述CPU证据本身不代替真实完整权重的画布恢复。后续独立两进程
GPU恢复已验证，见下节和历史步边界文档；通用Stage与集中人审仍待。

## 当前验证范围

`tests/test_h05_res_qualification.py` 的三项新检查，与现有
`tests/test_nfe_resume_advanced.py` 的九项守卫／恢复检查，单一 CPU epoch
共 12 项通过，零跳过、CUDA 未初始化。旧恢复代码、Core solver 和采样数学未改。
以上12项没有重复执行。它们仍不能单独证明完整模型成片或恢复历史。

## 本地完整模型画布代表

新增独立 [RES8 实验图](../examples/workflows/82-h05-res/H05_RES8_EXP.json)，
复用现有节点，不增加注册 ID。原生另存、关闭标签、从工作流侧栏重开后，
只运行一次完整八步：Ref2VA INT8 ConvRot 底模、不加载任何 Turbo／蒸馏 LoRA，
`res_multistep / native_flow / video12 / audio3`；124准备帧，明确裁切为
120帧、512×288、24fps、5秒。场景是单人正常中文对白。

本地任务 `dbe9f666-1524-4ef1-8923-f13ca6103db1` 成功，56.555秒；
20个执行节点／27条连线及完整输入与实际原生请求一致。采样器、解码节点
均未命中缓存，实际日志完成8/8；没有另挂全模型 forward 计数器。
整片RGB有限非黑、视频PTS逐帧正确，原片SHA256为
`0e423db76618b025ce3b3b7f58b950d9b6b9980a75677ae0724c716770779d95`。

声音来自实际联合生成 latent 的原生 AVDecode，不用源音频、TTS或其他片段
替换。32kHz双声道音频有限非零；5秒FLAC预览160000样本，MP4的AAC解码
160768样本，包含编码尾部，不能声称AAC与FLAC逐样本相同。浏览器已播放到
5秒结尾且有可见画面；听感、对白、口型与画质仍待集中真人审核。

这证明一个完整模型／完整RES调用的机械成片，**不证明同轨4+4历史恢复**，
也不是LOW→HIGH双采配方、可移植Stage认证、vLLM数值复现或提速保证。
外置EAV／Prompt Relay接口保留，本例没有应用它们，不据此认证其效果。
旧Euler-only恢复守卫、Core solver、生产采样数学和旧工作流未改。

## 后续固定计算链的新进程恢复

当前源码另用两个独立进程完成原生画布验收：完整RES8在POST-step4保存
实际solver history及编译资产，新的进程重建同一原始权重／FFN／KJ SM89／
原版Sol／实际条件位置，再只执行剩余4步。约161.559／108.803秒，两边均无
节点执行缓存。完整5秒RGB、解码PCM和两最终输出槽的四个AV tensor与原
连续8步逐值一致，检查点SHA未改。没有将旧latent-only入口改成RES。

该结果只适用于这一个固定来源绑定的计算链，不是任意GPU／补丁或通用Stage
证书；新EAV／Relay恢复组合和人审尚待。第一进程的临时FLAC在共用临时目录
启动清理后缺失，已单列并保留当时审计，不宣称原bytes仍在；持久完整MP4、
历史边界和最终AV数据保持。当前预览已另存，后续验收实例使用独立临时目录。
