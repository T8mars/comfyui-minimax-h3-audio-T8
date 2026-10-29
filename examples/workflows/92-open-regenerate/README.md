# Open Regenerate 2K 实验接线

## 作者完整配方

- [已有视频 → 完整配方3步2K](CompleteRecipe_ExistingVideo_2K_EXP.json)：选择输入视频及对应提示词，学习型3D潜空间放大后细化。合并LoRA与独立Sharp曲线均加载，Euler、CFG1、原始sigma；原音独立保存。已完成所选121帧/2560×1440训练模型实跑。
- [HyperFlow8步 → 完整配方3步2K](CompleteRecipe_HyperFlow8_Refine3_2K_EXP.json)：一个共用底模，两套独立MODEL分支。一采1344×768的完成潜空间直接放大后二采，最终保留一采声音。所选121帧一采及2K二采均已完整实跑，保存前的原声音频逐样本一致。

这两图已在独立原生前端完成加载、保存、重开、导出及全部显式参数和连线核对。它们依赖本次源码更新新增的完整配方加载、采样准备和运行报告节点。2026-10-29用户确认当前三组完整配方样片“没问题”并要求推送GitHub；不扩大为任意输入或底模的画质保证。

合并权重放在`models/loras/OpenRegenerateAuthor/`，另选含作者Sharp曲线表的Ref2VA来源。加载器只读取该表，不再加载第二套Ref2VA到GPU。两种权重及接线详见[接入说明](../../../docs/OPEN_REGENERATE_2K.md)。

- [同源8步／起点0.7对照](CompleteRecipe_Custom8_LowNoise_2K_EXP.json)：已实跑并通过所选样片人工审查，属于低重绘实验；不是作者默认，也不宣称优于3步。

## 仅Kijai纯Refiner

**不需要更新节点的版本：** [RefinerOnly_ExistingNodes_2K.json](RefinerOnly_ExistingNodes_2K.json)。只使用已发布既有节点和原生Load LoRA Model Only，默认肖像4秒；已完成浏览器保存、重开和导出核对，以及576×1024/1440×2560各96帧、3步的完整训练模型实测。原声音轨独立接保存，ManualSigmas提供原始采样表，不重复套12/3时钟。导入后选择自己的视频和原镜头提示词。

**可选简化接线版本：**

导入 [RefinerOnly_2K_EXP.json](RefinerOnly_2K_EXP.json)，选择自己的视频、底模、VAE、CLIP及Kijai refiner，提示词换为原镜头提示词。

默认只处理前5秒，学习型潜空间放大到2560×1440；原声音轨从输入视频独立接保存。新增细化准备节点提供原始sigma及锁定的编码静音上下文。

这是纯refiner接入示例，不包含官方完整配方中的Turbo/Sharp及曲线分支。已做原生schema和接线核对，尚无完整2K成片或浏览器往返验收。该节点还在开发候选，未自动部署或发布。

详见[接入说明](../../../docs/OPEN_REGENERATE_2K.md)。
