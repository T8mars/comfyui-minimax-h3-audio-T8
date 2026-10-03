# Pruned H3 HyperFlow 曲线近似（本地 EXP）

这是独立、可选择的新路线，不替换完整时间结构的 HyperFlow，也不修改旧工作流。它将匹配的完整时间教师及原 HyperFlow 的两时间调制，明确拟合到原生 pruned H3 的各个八维 AdaLN 基；近似不等于完整教师或完整 HyperFlow 的数值、画质等价。

## 准备与防选错

用 `Curve Build Explicit Fit` 单独选择 pruned 底模、匹配的完整时间教师及**原始** HyperFlow adapter，明确 CPU 或 CUDA。它生成新的独立 asset，不改模型文件。所有 50 个 block 和 final 共 51 投影分别拟合，保留训练 video/audio 时钟、固定 pins、每投影残差、全部文件及实现 SHA。拟合残差只评价调制近似，不替代成片评价。

已有 asset 放在 `models/hyperflow/curve_fits`，由 `Curve Load Selected Fit` 直接选择；也可用明确绝对路径读取本地输出。加载 asset 不加载或认证 MODEL。`Curve Apply to Pruned MODEL` 再核对精确 fit、所选 pruned 文件、原 adapter、实际 MODEL 原 FP32 曲线基和全部 208 backbone LoRA 目标。完整时间 MODEL、错 fit、修改后的基或非原 HyperFlow 会明确报错，不默默降为单时间。所选文件 SHA 不意味着所有实际 MODEL 参数都被证明来自该文件。

曲线图使用独立 `Curve Load Native FP32 Curve Base` 首次加载底模，再向两路分支。当前 Core 的 mixed-precision Linear 构造器会忽略声明的 FP32，把文件中的 FP16 AdaLN 基舍入到 BF16；这会被严格基校验拒绝。新入口在加载前保留 Core 声明的 51 个 AdaLN 及四个输入／输出 FP32 层，其他量化主干、实际层类、forward 和 LoRA 生命周期仍由 Core 处理。它不改共享 Core、普通 UNETLoader、模型文件或 fit，也不对已经 warm 的 MODEL 做浅复制修补。Core 原生重建工厂仍重新执行此入口并核文件 SHA；动态 dtype hints 参与分离阶段身份。两路普通内容 LoRA 和自定义委托的保留政策不变。

## 独立 HEAD / TAIL

两路从同一 pruned base 分支，分别插普通内容 LoRA、分别 Apply 相同 fit/adapter，条件和提示可以独立。HEAD 仅走 `0:split`，输出强类型的 model-space `x_sigma` 与独立 Core scaffold；TAIL 仅继续 `split:8`。TAIL 不重新抽噪声、不执行 HEAD、不放大、不将音频第二次 rebase。边界不是 clean x0，不能接普通 LATENT 放大器或原完整 HyperFlow HEAD。

当前支持原训练八区间，HEAD split 为 1–7、同分辨率 native joint AV；持久分离阶段不接受 restart mask。完整八区间也可用独立 Full Setup 接原 SamplerCustomAdvanced，但部分续跑必须使用专用 HEAD/TAIL。

外置 Prompt Relay Conditioning 的 MODEL/CONDITIONING 必须配对，接到各阶段 `Effects Bind`；其返回的 template/sigmas/context 接原 `Stage EAV Apply`，MODEL 再进对应采样节点。两阶段各有独立 Relay Plan、EAV Config、审计。效果开启时 CFG1；EAV 默认 report_only 不改变数学，apply_exp 才施加。审计读取已完成采样冻结的证据，不拿可变 UI 计数或 wrapper 经过次数冒充 51 个实际原生投影。

## 显式保存与恢复

HEAD Save 和 TAIL Save 各写新的唯一 safetensors，不覆盖原文件，位于 `output/MiniMaxH3/curve_stage_artifacts`。记录各自输出的相对路径及完整 SHA。

- HEAD Load → 独立 TAIL：只读已选 HEAD，再跑剩余区间。同实际 base、adapter、fit 和采样实现必须匹配；冻结 HEAD 后可独立修改 TAIL 内容 LoRA、提示和外置效果。
- TAIL Load → 两 VAE 解码：只读已完成 AV，零采样，无需 diffusion MODEL、CLIP 或 fit。更改今天的设置不会改写冻结结果。

缺失、坏 SHA、错类型、忙碌文件或未知持久 owner 不会隐藏重跑。未知用户 callable/后端保留原执行权，但不能据此获得 portable/cache 认证。

三个可导入工作流在 [68-radar-hyperflow-curves](../examples/workflows/68-radar-hyperflow-curves/README.md)。占位 fit/path/SHA 不是可直接队列运行的素材包；旧目录不迁移。

## 当前验证边界

真实完整 INT8/ConvRot 教师、pruned base 及原 adapter 已生成 51 投影 fit，生成／pinned 平均调制残差下降。Tiny 实际 Core 测试包括连续八步与 1+7／4+4／7+1 逐位一致、TAIL 不抽新噪声、专用存取、外置效果实际调用、委托保留和取消恢复。公开节点的独立 tiny 入口与存取测试通过。

十八个完整 CPU 文件 281 项通过，包含实际混合精度构造差异、独立原生重建、非零 AdaLN／主干补丁的 load-unpatch-reload、原分支不变、单值基扰动拒绝和 dtype-hint 身份变化；另含旧完整 HyperFlow、真实原 adapter 与五接缝门。初次真实大 MODEL 在采样前因原普通加载器的基舍入失败，证据保留。

新增专用入口已用实际大 MODEL 从画布 Save／重开／Queue：full HEAD4＋TAIL4、新进程 HEAD Load 后只 TAIL4、再新进程 TAIL Load 零采样解码。三份完整124帧512×512／24fps H264/AAC 的全画面 RGB／PCM 精确一致，阶段可移植身份通过；HEAD、TAIL 的外置 Relay／EAV 各200次真实调用。此材料 EAV 测得 gain=1，不宣称画质增益。六份完整权重／fit、源文件、Core 和原工程文件均稳定。

此次资格显式关闭 Core dynamic VRAM，并预留4.5GiB；不改共享 Core 或生产资源阈值。此前 dynamic 模式在首个 forward 的资源保护中止后，Core 清理发生 CUDA illegal-access，原失败保留，dynamic 模式尚未认证。机械恢复不代表完整时间教师等价、速度优势或人审画质。未发布、未自动接受；两路提示／内容 LoRA 可编辑机制的 CPU 资格也不等于所有素材已验。
