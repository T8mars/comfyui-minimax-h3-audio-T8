# Kijai 九原件：加载入口与分离双采（EXP）

v1.90.0 新增四个 Acc-8Step 原件的专用加载入口，以及九个家族共18张分离工作流。旧 PDD 节点、旧默认值和已有工作流不替换。九份指定完整双采片已由用户逐项通过；这只覆盖对应固定片与初始配方，不保证新素材、任意参数、冷 HIGH 编辑或作者 pipeline 等价。

原件取自 [Kijai 的 LoRA 目录](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/d8023be02fefbb3633b0cd335c3879f91177299d/loras)，固定修订 `d8023be02fefbb3633b0cd335c3879f91177299d`；请先阅读[上游说明与许可](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/d8023be02fefbb3633b0cd335c3879f91177299d/README.md)。这些原件已经是 ComfyUI 可解析布局，T8 不再转换、改写或再分发模型。二步 PDMD 不在本次范围内。

## 选择对应文件和底模

文件放在 `ComfyUI/models/loras/`，可放子目录并在工作流里重新选择。下面是实际验收路线，不是对任意同形底模的认证。

| 家族／原件文件 | 底模与加载入口 | LOW → HIGH／音频 |
|---|---|---|
| DMAD · `minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors` | FL2VA pruned；Core Bypass LoRA | 完整4 → 4，shift12/2；锁完成 LOW 音频 |
| PDMD4 · `minimax_h3_pdmd_4step_lora_avg_rank_57_bf16.safetensors` | FL2VA pruned；Core Bypass LoRA | 完整4 → 4，960×544 → 1152×640；锁完成 LOW 音频 |
| ELM · `minimax_h3_ELM_longlive_plug_4step_lora_bf16.safetensors` | FL2VA pruned；Core Bypass LoRA | LOW LCM4 → HIGH Euler4，shift12/3；锁完成 LOW 音频 |
| FlashGen · `minimax_h3_4step_lora_flashgen_v1.0_768p_fl2va_pruned_avg_rank_13_bf16.safetensors` | FL2VA pruned；Core Bypass LoRA，保留 bias | 原手动四步表 → HIGH4；锁完成 LOW 音频 |
| FL2VA Full · `MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors` | FL2VA full；新 Kijai PDD Setup，选择 FL2VA | 绝对0:4 → 4:8，联合音频 |
| FL2VA Pruned · `MiniMax-H3-FL2VA-Acc-8Step_pruned_comfy.safetensors` | FL2VA pruned；新 Kijai PDD Setup，选择 FL2VA | 绝对0:4 → 4:8，联合音频 |
| Ref2VA Full · `MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors` | Ref2VA full；新 Kijai PDD Setup，选择 Ref2VA | 绝对0:4 → 4:8，参考条件／联合音频 |
| Ref2VA Pruned · `MiniMax-H3-Ref2VA-Acc-8Step_pruned_comfy.safetensors` | Ref2VA pruned；新 Kijai PDD Setup，选择 Ref2VA | 绝对0:4 → 4:8，参考条件／联合音频 |
| Ref Difference · `minimax_h3_ref_lora_rank_256_bf16.safetensors` | FL2VA pruned＋参考图；T8 完整 LoRA 兼容加载器 | 完整8 → 4；锁完成 LOW 音频；不是加速 LoRA |

全部模板 LoRA strength 为1；不是语义桥的 alpha 控件。四个 Acc8 的 FL/Ref 和 Full/Pruned 不可互换。形状相同不证明训练基模相同，`base_variant` 是使用者声明；文件、真实形状、A/B 配对、有限值及实际目标映射继续校验。

## 新 Acc8 入口

节点 **MiniMax H3 Kijai PDD 8-Step · Full / Pruned (T8 EXP)**，ID `MiniMaxH3KijaiPDD8StepSetupEXPT8`。保留32组相对视频／音频输出头以及 backbone、AdaLN、bias；再接已有 PDD Stage Setup 的 `pdd_low_0_4` 和 `pdd_high_4_8`。

这些相对头文件不要接旧 T8 PDD Setup 的绝对 bank 入口。Acc8 LOW 只完成完整八步表前四步，音频尚未完成；HIGH 继续绝对4:8，**不能复制普通四步图的 LOW 锁音策略**。

Core Bypass 的身份审计只读、绑定实际代码与完整权重内容，不卸载或清空用户 MODEL。已有用户 LoRA、注意力或 hook 保留／委托；未知组合警告并限制跨进程缓存认证，不因未验证效果而硬禁用。旧采样数学保持；ELM 的默认 Core LCM 路线是显式识别，不代表无限历史 KV 或作者 RNG/checkpoint 等价。

## 保存 LOW，再单独编辑 HIGH

在 [71-kijai-experimental-split](../examples/workflows/71-kijai-experimental-split/README.md) 选择同一家族的 `Full_Save`／`Cold_HIGH`。18张公开模板来自已保存的原生画布图；发布时仅清理注释与私有元数据，并将本地 LoRA 菜单别名改为同一原件的文件名，执行接线、数值参数、ID和布局保持。

1. 替换自己的模型、LoRA和图片。模板中的 `10A.jpg` 是参考素材文件名占位，图片不随包提供；HIGH 尺寸有连接时由 upscaler 输出决定，未激活的 widget 数字不是实际运行尺寸。
2. 跑 `Full_Save`，记录 LOW StageSave 返回的相对 `artifact_path` 与完整64位 `artifact_sha256`。
3. 打开同家族 `Cold_HIGH`，填入自己的实际 LOW 路径与 SHA。`REPLACE_WITH_YOUR_SAVED_LOW/manifest.json` 和64个0故意不可直接运行；不能用 HIGH、别的家族或伪造 SHA 替代。
4. Stage 文件保留在当前服务 `output/MiniMaxH3/stage_artifacts`。JSON 不包含缓存；换电脑须转移实际 Stage，或重新完整生成 LOW。
5. Cold 只采样 HIGH。改变 HIGH 模型、LoRA、seed、提示词及效果可以保留选定 LOW；要让 LOW 修改生效须重新运行 LOW。修改后的效果需自行检查。

两路 MODEL、Noise、条件、Prompt Relay Plan 与 EAV 均独立外置。Relay 初始同文，可分别编辑；EAV 初始 `report_only`，不代表增强已经生效，显式改 `apply_exp` 是新配置。learned3D scale 为1.2；除 PDMD4 外，固定测试为448×256 → 512×288，共124帧／24fps，约5.17秒。

## 验收边界

九份完整音画固定片已经人审通过，原生另存／刷新／重开及参数、接线核对完成。原失败、旧图和私有证据保留，本发行包不包含它们或模型、媒体、Stage。PDMD4 的旧256p彩块不作为推荐配方；544p通过不证明544p为强制最低分辨率。普通蓝底配方通过也不证明已定位所有闪烁共同根因。

Ref Difference 保留全部264个适配器与266个普通差分，共794键／530目标；不是四 Acc8 的相对32头路线。机器检查、CPU回归与真人质量验收分别记录，不将文件能加载或声音有限值当作质量通过。GitHub 发布和 Registry 可安装性独立。
