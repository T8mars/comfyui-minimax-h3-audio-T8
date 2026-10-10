# VM01 · 视觉标记空间引导（EXP）

2026-10-08，用户已接受 A1/A2/A3 三条指定的完整5秒原生音画。
下列前三张是对应原生画布的公开模板；第四张为已验证的独立框选编辑器。
原生保存图的原字节另存本机，旧工作流不覆盖。

| 入口 | 工作流 | 用途 |
| --- | --- | --- |
| A1 | [VM01_Clean_4plus4.json](VM01_Clean_4plus4.json) | 干净图＋普通空间描述，作为无框对照 |
| A2 | [VM01_Marked_4plus4.json](VM01_Marked_4plus4.json) | Prepare＋Prompt，整张框图同时进入 VAE/Qwen，LOW/HIGH各自外置 Relay |
| A3 | [VM01_Split_4plus4.json](VM01_Split_4plus4.json) | 显式 clean-VAE／marked-Qwen 分流，其他配方与 A2 保持 |
| 编辑 | [VM01_Editor.json](VM01_Editor.json) | LoadImage→Prepare→PreviewImage，仅准备/拖框，不采样 |

## 导入后先做什么

1. 在 LoadImage 选择自己的合法 RGB 底图，替换 `YOUR_REFERENCE_IMAGE.png` 占位入口。
   模板不打包测试参考图、视频、模型或私人反馈；换图不等于复现已通过样片。
2. 有 Prepare 时点击“框选编辑（不运行）”，选择人物框与目标框并拖动，修改角色、颜色、描述、关系，应用后另行保存。
   随测试图保存的坐标只是示例，不能直接套到新图。编辑器模板已清空旧图专属的 `expected_source_sha256`；
   真正应用当前 RGB 预览后会写入新源 SHA，防止陈旧预览误接。
3. 修改外置 Relay Plan 中的人物/场景、局部动作和对白。静态图例由 Marker adapter 新鲜绑定实际 Picture；
   一次动作仍由局部事件拥有，不反复追加 global。手绘图可在 Prepare 选择 `provided_marked`，不自动识框或擦框。
4. 检查两阶段加载器里的实际模型文件与本机环境；另存工作流，重新打开，再运行。不要改旧工作流的默认设置。

最小普通条件接法也受支持：`LoadImage → Prepare.marked_image → 原 Conditioning.ref_images`，
`Prepare.marker_plan → Prompt.full_prompt → 原 Conditioning.prompt`。这种纯预处理接法不声称已核对下游编号。
有其他图片/首尾帧时，用独立 `H3 Marker · Bind` 的 `prompt_recipe` 入口；有 Relay 时用 `External Relay`，
不要将编码后 media_map 回接上游制造循环。详见[完整说明](../../../docs/VISUAL_MARKER_EXP.md)。

## 已接受配方与依赖

- LOW4：448×256，HIGH4：经原 learned 1.2×得到512×288；裁切交付120帧／24fps／5秒。
- LOW/HIGH seed分别为2610080101／2610080102；保留实际8步表的LOW前4步与原HIGH4配方，不改sigma。
- 模型：`minimax_h3_ref2va_int8_convrot.safetensors`；Qwen：`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`。
- 两阶段Turbo：`minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors`，强度1；原低显存注意力/前馈分块保留。
- 视频/音频VAE：`minimax_h3_video_vae_fp16.safetensors`／`minimax_h3_audio_vae_fp32.safetensors`。
- 3D放大器：`minimax_h3_latent_upscaler_3d_fp16.safetensors`，放在原加载器扫描的 `models/latent_upscale_models`。
- 需要支持这些 H3 模型的 ComfyUI Core 与本地 VM01 新节点；本次无新增权重、LoRA或SOLRICKS依赖。
- 原生音频，没有TTS/静音/换音轨。外置EAV为 `report_only`，Relay本例单事件直通，不能当增强效果证明。

## 验收边界与来源

整图和分流样片均观察到青/黄框残留，用户仍接受了这组结果；这不是“去框成功”。
只批准指定三片，不保证新素材到位率、硬坐标锁、多人身份、跨窗动作隔离、路径或物理接触。
不能把本机4+4代表当作作者原16步严格复现或成功率统计。

参考方法：[RK-BoilingPoint/H3-Visual-Marker-Control](https://github.com/RK-BoilingPoint/H3-Visual-Marker-Control)，
固定 commit `0f0274f5f07c0e8dda0bb8cd0286256587f4e959`；T8提示词/测试素材独立编写，不打包作者素材。
特别感谢原作者 **RK-BoilingPoint** 开源并分享参考图视觉标记的空间引导思路；这些节点、编辑器和示例是 T8 独立适配，不代表原作者官方版本或背书。
[source_map.tsv](source_map.tsv) 记录原生文件名、原SHA、模板SHA与变更范围。
模板只改图片占位入口、非执行UUID/标题/NOTE和普通控件插槽序列化；编辑器额外清空旧源图绑定。
实际具名/类型接线和所有其他执行控件逐项保留；收录于GitHub v1.96.0，Comfy Registry可安装状态独立核对。
