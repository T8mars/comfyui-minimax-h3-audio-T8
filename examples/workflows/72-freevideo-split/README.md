# FreeVideo · 已通过的独立双采模板（EXP）

三条固定5秒完整音画已获用户通过。六张图来自实际原生画布保存，不覆盖旧图；换素材、参考、LoRA或参数不自动继承质量结论。

|路线|完整生成并保存|冷恢复，只执行HIGH|
|---|---|---|
|基础8+2|[Basic Full](FreeVideo_Basic_Full_8plus2.json)|[Basic Cold](FreeVideo_Basic_Cold_HIGH.json)|
|外置EAV/Relay与音频参考8+2|[Effects Full](FreeVideo_Effects_Full_8plus2.json)|[Effects Cold](FreeVideo_Effects_Cold_HIGH.json)|
|真正前4＋后4|[Split Full](FreeVideo_Split_Full_4plus4.json)|[Split Cold](FreeVideo_Split_Cold_HIGH.json)|

## 使用前

1. 按[独立环境说明](../../../docs/FREEVIDEO_EXP.md)准备固定FreeVideo/VDN、独立Python、rowwise权重与内核收据。Loader默认读取`user/default/T8/freevideo-runtime.json`，也可填写自己的配置；配置和依赖不随包。
2. CLIP、VAE、learned upscaler使用常规Comfy模型目录。FreeVideo rowwise权重不是普通UNET，勿直接交给普通模型加载器。
3. 带参考的图需把`YOUR_REFERENCE_AUDIO.wav`换成自己的音频并生成匹配任务表。公开模板已清除私有预览与缓存路径，未改变采样数学或连线。
4. 先跑同路线Full保存缓存，再把实际manifest路径与SHA填到Cold的加载节点。空值不是可用缓存；改LOW输入/模型/效果后须重新生成，不能沿用旧MID。仅改HIGH也须符合真实缓存身份规则。

## 4+4不是8+2

4+4的MID是半成品：partial video x0用于原外置learned放大，实际audio_a4继续后四次联合更新，不冻结未完成音频。MID Load用于继续HIGH，完成HIGH Stage用Stage Load且选择HIGH；两类缓存不能互换。旧8+2完成LOW音频保留策略不变。

示例LOW448×256、原learned1.2×实际512×288；124帧生成同步裁到120帧/24fps/5秒。不是作者2×配方或性能对照。新4+4图LOW EAV显式start=0，旧节点默认不变；Relay模型与条件必须成对接入同一端。语义桥在条件编码后按实际规则应用一次，不重复套用。自定义引擎不用Core MODEL补丁冒充效果兼容；任意LoRA/参考仍需自己的片验收。
