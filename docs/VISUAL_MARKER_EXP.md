# H3 视觉标记空间引导（EXP）

在参考图上标出“谁”和“去哪”，再由原生 H3/Qwen 理解图例。基于
[H3-Visual-Marker-Control](https://github.com/RK-BoilingPoint/H3-Visual-Marker-Control)
的实验思路；这不是新模型、LoRA、ControlNet 或坐标硬锁，不依赖 SOLRICKS 动画风格 LoRA。
上游固定参考 commit：`0f0274f5f07c0e8dda0bb8cd0286256587f4e959`。
本地模板独立编写；不打包作者的参考图和视频。

特别感谢原作者 **RK-BoilingPoint** 开源 [H3-Visual-Marker-Control](https://github.com/RK-BoilingPoint/H3-Visual-Marker-Control)，
分享参考图视觉标记与空间语义引导的实验思路。T8 在此基础上独立实现 ComfyUI 节点、框选编辑器和外置 Relay 接线；
不是原作者官方版本或合作背书，原方法来源与 T8 适配分别署名。

## 独立节点

- `H3 Marker · Prepare / 标记图准备`：`render_rectangles` 从干净 IMAGE 绘制明确矩形；
  `provided_marked` 保留用户已有手绘图，后者可不填 xyxy。不修改输入 tensor，不自动检测、去框或保存。
- `H3 Marker · Prompt / 空间提示词`：编排图例及动作，原 base_prompt/对白不翻译、不删改。
  `manual_original` 输出原文。图例与一次性动作分别输出，便于外置计划。
- `H3 Marker · Bind / 原生空间条件`：插入真实标记参考图，在 tokenize 之前绑定实际 Picture 编号。
  保留其他 raw 图片、视频、音频以及首尾帧；首尾帧会占用前面的 Picture 编号。
- `H3 Marker · External Relay / 外置分段条件`：接用户的外置 Relay Plan。
  Plan 拥有事件、对白和时间范围；仅追加静态图例，不把 Prompt 节点的动作再次追加到 global。
  构建单独校验的派生计划及全新 token span/layout，不修改原 Plan。

旧节点、旧工作流无需换节点；新入口完全显式。EAV 仍接原阶段配置/应用节点，Bridge 接对应条件的可选输入，
不隐藏启用效果、不更改 alpha、步数、音频、sigma、放大器或接缝默认值。

## 标记数据与编辑

```json
{"markers":[
  {"marker_id":"actor_A","kind":"actor","role_id":"A","color":"cyan",
   "description":"the woman in a red sweater","xyxy":[0.18,0.06,0.33,0.93]},
  {"marker_id":"target_A","kind":"target","color":"yellow",
   "description":"the floor immediately in front of the door on the right","xyxy":[0.76,0.77,0.93,0.97]}
],"relations":[
  {"actor_marker_id":"actor_A","target_marker_id":"target_A",
   "action":"walk to that floor area and stop with both feet inside it"}
]}
```

xyxy 是源图归一化坐标，不是生成视频逐帧坐标。先按自己的底图改框。
同色人物/目标合法，多个人物共享目标也合法；重叠/相同人物颜色只警告。
自动绘框需要有限、非零、0–1 坐标；真实错误拒绝，不静默截断。
当前每次一张 float32 RGB 图（≤4MP）、2–64 标记、1–128 关系、JSON ≤64KiB。

Prepare 上的“框选编辑（不运行）”可读取直接连接的 LoadImage，或选同一张 RGB 图预览。
选标记后拖动矩形，JSON 可编辑颜色/描述/关系。点击“应用”只写真实工作流控件，另行保存画布。
加载未完成不能应用，换预览有并发保护；不上传远程素材、不替换 IMAGE 接线、不排队。
预览绑定实际 RGB SHA；执行时输入不一致明确拒绝。
EXIF/色彩管理/透明图合成或中间图像处理可能使浏览器预览与实际 IMAGE 不同：
先显式转换/加载相同 RGB；手动 authoring 可明确清空高级 `expected_source_sha256`，不要冒称预览已核对。

## 基本接线与防误接

最小 P1：LoadImage → Prepare → 原 Conditioning 的 ref_images；Prepare 的 MarkerPlan → Prompt → 原 prompt。
仅这种预处理接线时，报告 `binding_verified=false`：节点看不到下游真实图编号。
有其他图片/首尾帧时，建议使用 Bind 或 External Relay 的 `prompt_recipe`，
`marker_position` 是在**其他 raw 图片**中的实际插入位置（1 表示最前），不是手填 Picture 标签。
最多八张其他图片＋一张标记图；节点会把生成图例改成实际编号，manual_original 不改用户原标签。
原文中的任意标签、角色身份、声纹仍由用户负责；编号绑定不是语义身份认证。

本入口不隐式混合存储 ReferenceSet。要使用旧 ReferenceSet 路线，仍可接 Prepare 的图和 Prompt 的文本，
但它属于 P1 声明式接线而非本次真实绑定资格。不要把后编码 media_map 接回上游 Prompt，避免图循环。

## 显式 clean / marked 分流实验

`marked_vae_and_qwen`：默认两编码器接相同实际缩放后的标记图，沿作者完整参考图思路。
`clean_vae_marked_qwen`：同一逻辑 Picture 槽，干净图供 VAE，标记图供 Qwen。
自动绘框可使用保留的真实底图；手绘图需额外提供实际干净图，不自动擦框。
记录实际 VAE/Qwen RGB、尺寸、源 hash、编号及 native recipe；只编码一次参考 VAE。
手绘 clean/marked 同尺寸不证明同裁切或配准，报告明确标为用户声明。
分流可能减少残框，也可能削弱控制，必须另外匹配验证，不默认替换整图路线或保证去框。
未知 processor grid/token 观测为 null，不用源码推测充作运行测量。

## 双采、Relay 与验收边界

LOW/HIGH 分别构建自己的实际条件，沿原分离双采接线。
连续身份/静态图例可贯穿全片；一次动作写进明确的局部事件，不在每窗反复发送。
NativeTextRecipe 保存原生媒体的 fresh-encode 输入，不能单独替代与 Relay MODEL 配对的 token/layout 合同；
新的窗口/文本必须由相应执行器重新绑定，不能使用旧全文 binding 或截取旧 embedding。

最小方法对照：同场景/seed，两条完整 5 秒原生音画，干净图＋普通空间描述 vs 标记图＋等价目标动作图例。
采用小画布分离4+4，LOW448×256 → 原 learned1.2 → HIGH512×288；不改原生音轨或后处理擦框。
必须真实画布保存、重开、Run；仅 API 接受、CPU 通过或 LOW 预览不算最终验收。
分流另做定向对照。人工检查目标到位、动作、人物、框残留、闪烁/瞬移及对白口型。
一个样例不是成功率证明，当前工程不保证硬路径、物体接触、战斗物理或多窗动作硬隔离。

## 本机代表验收状态

已从真实画布另存、关闭、重开并运行干净图、整张框图、显式分流三条完整5秒原生音画；
框选编辑器也完成真实拖框、应用、保存、关闭重开与仅CPU准备运行。
实际六份LOW/HIGH Stage通过原有加载器和结果内容校验；三路每阶段的模型、噪声、seed、sigma、AV时序一致。
分流路线的实际Qwen RGB、文本token与条件向量和整框路线相同，实际VAE参考latent与干净图路线相同。
浏览器三片均实际播放到结尾，HTTP完整文件与原片逐字节一致；这些是工程/交付检查，不是人工质量通过。

当前这组小画布4+4样片中，整张框图和显式分流两路均观察到色框残留。
不以擦框、静音或替换音轨掩盖结果，也不把分流宣称为已成功去框。
2026-10-08，用户已完成A1干净图、A2整框图、A3分流三项正式审核并全部通过；指定残框样片保留原样。
已保存[三张已审4+4公开模板及独立框选编辑器](../examples/workflows/84-visual-marker/README.md)，
原生画布原字节另存本机，旧图不覆盖。公开模板使用自己的底图入口，保留配方和实际具名/类型接线；
换图必须重选框、改角色/动作并自行验收。编辑器模板清空旧图专属SHA，真正应用预览后重新绑定。
工程验证和用户接受只针对指定样例，不把“全部通过”扩写成任意新素材质量保证；发布版本见[v1.96.0说明](RELEASE_1.96.0.md)。
本组不能外推作者16步方案、任意模型/参考图、成功率、多窗动作隔离或物理交互效果。
