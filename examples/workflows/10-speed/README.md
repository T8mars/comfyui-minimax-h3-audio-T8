# SPEED 与 FastH3 VSA（研究/EXP）

这一组研究SPEED论文的空间分辨率逐级增长、频谱高频噪声扩展和时间重对齐，并为H3建立独立频谱标定数据集。

## 当前入口

- `2026-08-30_H3_FastH3_VSA_T2VA_4Step_0p4MP_Advanced_EXP.json`：FastH3 Preview v1 的普通T2VA 4步工作流；优先使用真实learned-gate VSA，能力不完整时明确回退Dense。
- `2026-09-03_H3_OpenVDN_DMD8_T2VA_0p5MP_Advanced_EXP.json`：OpenVDN MiniMax H3独立混合注意力路线；默认DMD 8步、960×512×73，只支持普通T2VA。
- `2026-08-18_H3_SPEED_T2VA_Stock20_Advanced_EXP.json`：绑定正式100条T2VA标定数据与profile的最新前端工作流。
- `2026-08-19_H3_SPEED_Spectrum_Dataset_Calibration_Advanced_EXP.json`：加载、累积、定稿和验证频谱数据集。
- 其他T2VA/FL2VA/L2VA/Ref2VA/Hybrid/Turbo8文件是历史机械示例，不代表已通过各任务质量门。

## 当前成果

100条固定语料标定得到A=29.96418670445687、beta=2.3183720623777164、R²=0.9951511913433466，数学与数据合同通过。但正式同输入Stock20对照中，SPEED为248.688秒/16175.8MiB，弱于基线243.203秒/12504.6MiB，用户盲评也选择基线；因此当前没有通过加速、质量、音频非劣或通用16GB安全门。

## 使用方法与注意事项

仅用于研究和复现，不作为日常推荐采样器。T2VA正式工作流使用 `delta_optimal` 自动确定阶段，手工0.85转场字段在该模式下不生效。不同任务、checkpoint和VAE需要独立profile；禁止把T2VA profile直接套到Ref/Hybrid。发现花屏、坏帧、音频异常或显存余量不足时立即回到全分辨率基线。

FastH3 VSA 需下载官方`vsa-datafree/adapter_model.safetensors`到
`models/loras/FastH3-VSA/vsa-datafree/`。它只覆盖普通T2VA、4 NFE、12/3双时钟；FL2VA、Ref2VA和混合参考不在本预览模型合同内。

VSA运行时要求Comfy Kitchen的`sol_attn`同时提供`topk_ratio`、`tail`、`block_len`和`coarse_gate`。截至本次实现，本机验证使用[Comfy Kitchen PR #117](https://github.com/Comfy-Org/comfy-kitchen/pull/117)源码构建的兼容wheel；请用启动ComfyUI的同一Python和匹配的Torch/CUDA编译安装。节点会结构检查50层gate和接口，不检查模型文件名、大小或哈希。缺失时报告原因并回退Dense，不会把Sage、SLA或普通Sol-Attn冒充VSA。

## OpenVDN MiniMax H3

从[OpenVDN/vdn-minimax-h3](https://huggingface.co/OpenVDN/vdn-minimax-h3/tree/main)固定revision `18be6bcc4ee72585eee322ba28b5ccac2cf85ef0`下载以下内容到`ComfyUI/models/diffusion_models/OpenVDN/vdn-minimax-h3/`：

- `stage-dmd-step-250/linear_branch/model.safetensors`：4,279,428,112字节，SHA-256 `DEC6981C7874F5B3BC92D1A02E256B673A3B3499DC1A124714BB3B19DA602855`
- `stage-dmd-step-250/adapters/default/adapter_model.safetensors`：334,026,912字节，SHA-256 `58558FEF506F88BB41649242DE9B9B3A365DA806B51B2E96AFBBE1625222058A`
- `stage-dmd-step-250/adapters/turbo/adapter_model.safetensors`：851,452,696字节，SHA-256 `24FC93C82FE84DC45D0627F4E72C637BC387D282BA18F60ED3B7F8C81089392C`
- 两个stage的`model_spec.json`、`metadata.json`以及对应adapter/branch config。Stage B发布的branch/default与DMD相同，本地可复用DMD的单份大文件。
- 模型仓库顶层`README.md`、`NOTICE`、`LICENSE`和`licenses/MiniMax-H3-Community-License-Agreement.txt`。权重许可的Applicable Territory排除欧盟、英国、韩国和美国；下载或运行前必须自行确认许可适用。

推荐工作流使用`MiniMaxH3VDNModelComposerT8Advanced → MiniMaxH3VDNExecutionPlanT8Advanced`。DMD会自动应用104个default和259个turbo patch target并固定8 NFE；切到`stage_b_50nfe`时自动只用default并固定50 NFE。不要在它前后叠加任何LoRA或Attention接管节点。v1只接受`[text|audio|video]`普通T2VA布局，其他参考任务全部fail closed。
