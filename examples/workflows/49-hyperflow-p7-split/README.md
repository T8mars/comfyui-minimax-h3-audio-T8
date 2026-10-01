# S16 HyperFlow P7 分离式长视频（EXP）

本目录按 segment0／segment1、无效果或外置 EAV／Prompt Relay／组合在 LOW／HIGH／两段的10种作用配置，以及四种保存／恢复方式提供80张可编辑实验图。原 P7 一体工作流不迁移。新图是通用示例，不是旧公开图原尺寸、seed或画音的逐帧替代。

- `Full_NoStageSave`：LOW 0:4 → 原 learned3D → 原音频 reconcile → HIGH fresh 4:8，并保存**未接受**候选，不保存阶段回执。
- `Full_SaveStages`：再保存 LOW 和 HIGH 的实际 manifest 路径及 SHA，供后续显式恢复。
- `Cold_HIGH`：从真实 LOW 回执只恢复 lift 与 HIGH；不加载或重跑 LOW 模型、噪声、采样或效果。冻结 LOW 曾用 Relay 时，保留原 LOW Plan／投影／条件以核验来源。
- `Load_Completed_HIGH`：同时填写真实 LOW 和已完成 HIGH 的回执；不跑 HIGH 扩散，但仍核验父片、LOW lift、HIGH handoff，并重新保存未接受候选。若 HIGH 曾用 Relay，来源重建仍可能加载 HIGH 模型；这不是无模型的简单解码图。

两段必须使用同一实际唯一 `chain_id`。先运行 segment0，预览候选后显式将专用 P7 Candidate Accept 从默认 `false` 改为 `true`；再将其 candidate ID、revision 和 job SHA 填到 segment1 的 Accepted Parent。segment1 渲染124帧、只交付新增68帧；两个候选都经显式接受后，才可使用现有 `MiniMaxH3LongVideoComposeAcceptedT8` 节点按同一 chain ID 合成192帧／8秒。未接受候选不得冒充已接受父片。

示例里的 chain ID、parent ID／revision／job SHA、LOW/HIGH manifest 路径／64位零 SHA，以及模型、素材和提示词均须核实并按实际任务替换，不能直接排队。外置 Relay Plan 每段单独可编辑；EAV 默认 `report_only`，改 `apply_exp` 才施加增益。读取完成 HIGH 时，效果标签只表示预期冻结来源，不会重新对结果施加效果。

现有 P7 tiny CPU 来源链与冷恢复证据，加上本目录80图的当前 Core 静态验证及真实 Chromium 原生保存重开，仅证明结构与前端可编辑性。它不等于这些通用图已在原尺寸真实权重 GPU 运行，也不认证多素材／LoRA／后端、`apply_exp` 画质、完整长片接缝或人工音画。
