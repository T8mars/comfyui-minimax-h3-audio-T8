# FreeVideo H21：独立 v4 实验运行时

v1.102.0发布状态：固定T2VA／参考音频的八档来源成片均已获人工通过，其中High参考音频采用2026-10-24复审的新缓存历史修正版，不采用旧失败片。见[已审工作流](../examples/workflows/90-h21-reviewed/05-freevideo)。仅这些固定配方获验收，不是任意素材、效果组合或性能资格；下文较早的“等待人审”是修复时的历史状态。

这是显式选择的新生产者，源码固定为 `f2c25e6e4f82d5ad9ed966998423762503d98e5c`。
不改已有 FreeVideo Loader、Quality 节点、工作流、默认配置或旧阶段记录。
旧 Medium12／High16／Max20 仍保持原样，不能将旧阶段改标签充当新版完成。

| 新档位 | 原算法 | 完成回调 | 说明 |
| --- | --- | --- | --- |
| Light | LOW8 → learned 3D → HIGH3 | 8+3 | 请求 2×；小画布可能同尺寸细化 |
| Medium | LOW8 → learned 3D → HIGH3 | 8+3 | 请求 1.5×；短边不足512和小画布按作者真实计划回落 |
| High | EasyCache-AV 单采 | 20 | 近似缓存；20次回调不等于20次完整DiT |
| Max | 全算单采 | 20 | 不因用户环境变量自动启用缓存 |

### High 缓存历史的 FP32 存储修正

固定来源的参考采样器会原地更新 FP32 的生成行；其 EasyCache 原实现把
`tensor[stride].float()` 留作上一轮输入，FP32 时这仍是同一存储的视图。
下一步覆盖生成行后，历史也被改写，可能把实际变化误算成零、连续误命中。
当前独立 H21 worker 只对缓存的小型抽样输入／输出历史建立独立副本，保留原
0.2 阈值、15–95% 窗口、速率与残差公式、20 步和全部原条件／权重。
仍是 EasyCache 近似计算，不是关闭缓存或换用 Max；不能承诺与全算逐位相同。

该适配只在自有子进程范围内安装并在异常时恢复，不修改固定上游文件、主进程
旧 FreeVideo 家族或其它缓存类型。High 的计划与实际统计带
`owned_FP32_history_snapshot_v1`，旧阶段记录保留原身份，不改标签冒充新结果。
更新自有适配源码后，请用准备工具生成新的配置与 home；旧配置仍绑定旧源码，
不会放松真实文件校验来运行变更后的代码。CPU 存储与实际微型采样器回归通过
不代表训练模型画质已经通过，最终成片仍须单独验证。

本机已用一条原先人审失败的固定参考音频请求完成训练模型对照：原条件张量、
种子、资产、Engine 选项与 20 步保持不变，实测 20 次调用、每层 13 次完整计算、
7 次非零变化下的缓存复用。独立 CPU 回读核对了原始条件的字段及全部 tensor
字节；序列化时槽号重排不视为条件变化。原视频和失败输入未覆盖。新结果经原
视频／音频 VAE、原 5 秒裁切与安全 AV 输出，完整解码 120 帧、1024×576、24fps、
stereo32k 通过。抽查帧清晰不等于整片质量通过，当前仍等待该新片的人审；
不据此宣称任意素材、其它缓存、速度收益或所有组合已经通过。

实际计划直接调用固定源码 `tiers`、`two_pass`，保留32网格、真实首采尺寸、
放大后裁边、独立三步时钟和种子增量。收到预编码条件时沿用作者
`conditioned=True` 分支，不偷偷扩大后再缩回。HIGH返回完成LOW音频原值；
新参考音频固定在 t=1，与旧 t=0 家族分开。新阶段记录分别绑定配置、实际资产、
计划、输出、前驱及有序LoRA，不把两个阶段换权重仅因“未经验证”禁止执行。

## 准备及调用

`tools/prepare_h21_freevideo_runtime.py` 只读借用已有权重和VDN，校验固定源码Git
blob、真实文件SHA及所需调制表。必须使用新的 `--home`、`--output`。
不下载、安装、升级、编译探针或启动推理。缺少可选预计算表时，只有真实投影文件
已经全部存在、大小与SHA均匹配，才沿作者原动态AdaLN计算委托；报告优化未启用，
不伪造表身份。完整slim文件以新的自有硬链接视图如实描述实际全投影存储，不改
原缓存。真实投影缺失时报告所需文件与字节数，不用零表、不悄悄下载26GB投影。
模型声明已安装的表或投影损坏仍报错，不把损坏当作普通优化旁路；
官方目录仅列出但尚未安装的可选表，属于正常缺失，不冒称文件损坏。

```text
python -B -X utf8 tools/prepare_h21_freevideo_runtime.py
  --base-config <已有完整资源配置.json>
  --source-root <固定新版源码目录>
  --home <全新自有缓存目录> --sampling-root <已准备调制表目录>
  --output <全新v4配置.json> --upscaler <实际learned3D权重>
```

如需调整放置／计算分块，可用 `--engine-options <明确Engine参数.json>`。
步数、任务、画布、step_cache及fetch_tables由实际计划与只读资产接点控制，不能
混在放置参数中覆盖。没有性能保证；默认不修改旧配置的任何内核选择。
有序LoRA使用作者原准备算法：显式开启有界内核且head分块可用时走在线分支，
否则走原fused准备；不为使用LoRA静默开启内核。LoRA改变实际调制权重后，
保留新权重并在实投影齐全时使用动态计算，不复用旧常量表或仅因换权重拒绝。

Python薄入口是 `freevideo_h21.runtime.load_model`、`with_lora`、`sample`。
`sample(..., tier=..., role='FULL'|'LOW'|'HIGH')` 返回独立Stage。
LOW/HIGH只适用于真实两阶段档位；HIGH必须提供真实完成LOW和已放大／裁边video。
`seed`均表示根种子；FULL和独立HIGH遵循作者同一`restart_seed_offset=1`，
实际HIGH使用`(seed+1) mod 2^64`，不是再抽一个隐藏随机种子。
`stages.save/load` 支持内容绑定、create-only目录和完整tensor SHA核验。
画布入口使用独立H21八节点，按项目尾部追加，不替换原注册。
独立EAV／Relay入口复用共享的原生条件编码和只读数据绑定；不调用旧采样器、
配置加载器或Stage格式。Relay仍需本次实际成对MODEL与CONDITIONING，原始JSON
不能冒充token绑定。效果工作端复用serial cuDNN有界head公式，生命周期为新v4薄适配：
按真实global/window后端识别`dense`等价别名，按实际hybrid层逐层安装和报告，不以旧50层
数目限制准入。不满足serial／已启用kernel／head分块真实API时，保留原计算并明确
报告partial、效果未消费，不静默切换内核；实际安装／kernel错误仍正常报出。
默认无效果不导入或安装这条路径。
原生条件按实际所选`transformer/config.json`中的`text_dim`导出和配对，不把5120
当作替代合法模型的准入条件。旧5120默认保留原函数和身份；非5120只在自有worker
范围内适配四个原生校验引用，finally恢复，不改Source文件、不补零／截断或重编码。
非5120校验会产生CPU／BF16验证副本，可能有同步和复制成本，不宣称零开销。
维度不符、非有限值、错误token／首尾形状仍明确报错。小模型CPU实际forward通过
不等于任意其它权重或CUDA训练成片已经验证。
新源码上的 EAV／Relay 完整效果组合成片尚未验证，不能把默认无效果的八例验证移作这些组合的资格；旧固定样片的人审不转授新版资格。

工作进程由已有自有进程树监督器启动。主进程不导入作者初始化器或CUDA，
不重设旧 `freevideo_engine` 全局模块。每次子进程清除继承的
`FREEVIDEO_STEP_CACHE`，显式High/Max选择不受外部环境静默改写；取消只关闭本次
自有树，不停止用户ComfyUI。所有错误原样留日志，没有TypeError重跑或失败降档。
实际Engine逐pass重新读取的自有`conditioning.pt`也核初始内容SHA，任一pass前后
变化不发完成记录；不能只核原始safetensors而漏掉真正消费的派生文件。
FULL阶段逐项核8+3，不以总数11接受7+4；HIGH前驱必须有完整64位SHA。
自有投影视图在故障后只凭明确producer和内容身份续建，保留已写partial；未知目录
不覆盖。准备配置先写独立partial再create-only发布，真实失败不会留下伪完成配置。

## 观测与验证边界

每一pass分别记录实际采样回调、transformer调用和各DiT block调用。只有各block
覆盖一致才给出完整DiT计算次数，非均匀／外部绕过明确partial；不使用
`callbacks - cache_hits` 冒充实测。表读取也记录实际每层内容身份。

作者原Engine明确仅支持CUDA。`tiny_worker.py` 是另一个明确标注的CPU合同工具：
实际VDN Diffusers两层随机权重、原采样器、原时钟、原packing及EasyCache，
不是伪造CUDA后端，也不能生成生产Stage或训练画质／GPU资格。
它避免旧采样器CUDA-only计时路径，用实际module hooks收CPU调用证据，不改采样数学。
tiny 工具不能授予训练模型、完整 learned3D 成片、声音／画质、人审或 GPU 收益资格。

当前固定来源的 T2VA、物理参考音频 Ref2VA 各四档已完成八次真实训练模型生成。Light 的 LOW8→实际 learned 2×3D→HIGH3、Medium 的 LOW8→实际 learned 1.5×3D→裁边→HIGH3 均保留完成 LOW 的生成音频，不是复制参考录音；High／Max 沿各自原 20 回调配方，High 的完整 DiT 次数按实际 block 观察记录，不能写成 20 次完整计算。四个音频参考 case 各实际 402 个物理音频 token，原表的 t=1 行已经检查。

八片各完整 120 帧、1024×576、24fps／原生 stereo32k 的 RGB、PCM、PTS 和 strict 解码通过，共 960 帧；原独立运行时 2061 文件前后完整 SHA、原模型末尾核验、实际 worker 及自有进程退出证据齐。后置整份类型和值已独立重推，记录在本地 `h22-R6-actual-postflight-v2/actual-eight-index-v1/merge.json`。这是这些固定配方的机械资格，不是其它后端的数值 parity、公平速度／显存／内核收益、冷 Stage 恢复、任意模型／LoRA 或人工观看／听感通过。

代码许可为作者Apache-2.0及文件内保留的第三方声明；下载的H3权重另受MiniMax
Community许可约束。准备工具不将代码许可当作权重许可，也不运行包内安装脚本。
