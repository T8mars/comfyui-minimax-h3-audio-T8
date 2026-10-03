# RADAR 模型兼容与对照边界（本地 EXP）

只复用现有加载器，不创建“所有模型通用加速”入口，也不以SHA白名单限制用户执行。选定实物完整SHA用于本地资格和交接，真正加载以原生层／结构映射为准。旧默认、接口、工作流保持。

## LMS

LMS是与现有LBH同结构的完整3D放大器，FP16优先，不是LoRA。实际网络322个tensor严格加载，tiny CPU前向保留原音频和遮罩对象。真实画布同一完成124帧512方AV，仅替换原LBH／LMS，两独立冷进程各Save／重开／Queue→1024方完整H264/AAC。完整PCM精确，清CPU缓存／卸载GPU权重及所有所选文件保护通过。两次单冷Queue含load/upscale/decode/write约60.7秒，**不宣称加速、统计结论或画质优势**；人审和16MP独立。

来源：[固定LMS说明](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/e25a489717c7067ed1813bbd884e64fa68b76d58/docs/latent-upscaler.md)。它是实验细节增强权重，不保证文字／logo重建。

## Orbit

权重已是原生Comfy LoRA命名，rank16／208模块，走现有LoRA Compatibility Loader。作者建议pruned FL2VA、同首尾人物参考、768方／73帧／28步、strength1，无Turbo；不是Ref2VA或任意时长相机模型。T8对照图保持joint AV计算但只输出静音视频，**静音交付不等于作者audio-off推理复现**；使用同一独立简短描述，也不宣称逐字prompt／上游数值等价。当前实际成片资格另记，不以tensor库存通过冒充相机／冻结世界／闭环质量。

本地实际画布Save／重开／Queue两份单变量成片通过：同源同种子28步、73帧768方／24fps／3.042秒，完整H264静音解码；候选416个tensor／208个目标实际加载、零遗漏。内部仍是joint AV，完整轨道、冻结世界、首尾循环由人工判断；不是作者audio-off数值复现。

来源：[固定Orbit模型卡](https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA/blob/5ddbc2dbbe95edbbdaf5017c3e934b1d01791697/README.md)。作者训练域仅28个人物方形短片，外域不保证。

## Live Wallpaper R32

权重已是原生Comfy LoRA命名，rank32／208模块／Ref2VA。一张明确参考使用 `live_wallpaper:` 与Picture1；多图顺序和tags须真正连接，不能仅写进提示词。本次T8单变量兼容小样固定既有Ref2V Turbo4，其它参数相同，只插R32 strength1，不冒充作者R64＋Taomate3＋LMS完整双采配方，也不偷偷下载R64或自动换采样器。R32未专门训练相机标签，固定相机／脸保持／循环需实际人审。

本地实际画布两份单变量成片各4步、124帧512方／24fps／5.167秒，完整H264/AAC严格解码通过；候选416个tensor／208个目标实际加载、零遗漏。音轨是各自联合生成，不承诺与基线PCM相同。全所选模型和源素材字节保护通过；映射和解码不等于画质通过。

来源：[固定Wallpaper说明](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/e25a489717c7067ed1813bbd884e64fa68b76d58/docs/live-wallpaper.md)。R64的相机倾向不是R32的保证，模型授权沿各自模型卡，不随本插件重新许可权重。

六张通用对照图：[70-radar-model-compatibility](../examples/workflows/70-radar-model-compatibility/README.md)。明确占位尚需用户选输入；完整媒体、人审、运行环境、作者配方和加载兼容五项资格分开。

## 放置与选择

使用现有加载器：两份LoRA放在 `models/loras/zz_radar/`，LMS放在 `models/latent_upscale_models/zz_radar/`。候选示例明确选择这些子目录，基线保持原权重／disabled；文件分隔符沿操作系统菜单。不改原首选项或默认参数，也不自动下载／转换或替用户接受效果。曲线fit另放 `models/hyperflow/curve_fits/`，只用于其绑定的base／teacher／adapter，不能当通用模型权重。

当前本机为四个精确链接，未复制或改写权重；完整619个schema、22个指定菜单字段、旧首选／默认、2029份旧JSON、index与旧稳定采样器均核对。已有ComfyUI进程需要正常重新载入后才能看到新增文件；没有擅自重启用户服务或发布。
