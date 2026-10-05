# RADAR r6：参考包、独立解码与正确保存图

v1.93.0的10张短名前端JSON来自实际原生另存、关闭、重开的图。公开版只去除私有资产／缓存／诊断extra，调整明确文件选择项和保存确认，不改变接线、采样数值、几何或音视频时间。不是捆绑模型或参考素材的即开即跑包。

## 选择工作流

|文件|用途与运行边界|
|---|---|
|[Ref_Encode.json](Ref_Encode.json)|选自己的完整5秒master，实际编码A声音／B首帧参考，显式确认保存新短包名；初始确认关闭，已有文件不覆盖|
|[Ref_Full_Sample.json](Ref_Full_Sample.json)|选实际两包、VAE与模型后LOW4＋现有learned1.2＋HIGH4；两路外置Relay／EAV，保存完成Stage，不包含已知失败的CPU解码尾部|
|[Ref_Cold_Sample.json](Ref_Cold_Sample.json)|只读取自己的完成LOW manifest及其实际SHA，继续HIGH，真正没有LOW模型／采样；未追加Cold GPU质量验证|
|[Decode_FullVAE.json](Decode_FullVAE.json)|选自己的完成HIGH、普通视频VAE及原音频VAE，正常GPU解码，不重跑采样|
|[Decode_LightVAE.json](Decode_LightVAE.json)|同HIGH，普通VAELoader选Light转换件，只替换视频解码VAE；按本机菜单选择精确名称|
|[Master_Cold_Crop.json](Master_Cold_Crop.json)|选自己的完整master，示例512×288→512×280/y4；按真实尺寸调裁切框，明确确认后只写新文件，音频从master逐包保留|
|[Master_Full_Template.json](Master_Full_Template.json)|接全部回贴后的最终RGB／AUDIO，先保存master再裁切1920×1088→1920×1080；附加模块，不是完整生成图|
|[Master_Contain_Template.json](Master_Contain_Template.json)|选完整master，明确等比例缩放加黑边；未将裁切片的通过借给该模板所有素材|
|[Recipe_Confirmed.json](Recipe_Confirmed.json)|模型配方卡UI示例，不是生成图；选本机可用VAE，右键预览差异，明确确认，可撤销，再自己保存|
|[Bridge_Confirmed.json](Bridge_Confirmed.json)|Bridge配方卡UI示例，文件须自己安装／选择；内容推荐值与alpha合同沿原实现，先预览再明确确认|

## 最短操作顺序

1. 使用自己的合法素材，`Ref_Encode`选视频、VAE和新的`role_A.safetensors`／`role_B.safetensors`包名，核对后开启保存确认。A声音、B图像来自同一素材的例子不证明两个独立人物。
2. `Ref_Full_Sample`选择实际包、精确SHA和对应producer配置，再采样保存LOW/HIGH。示例测试producer为`--cpu-vae --fp32-vae`，包名不证明设备；默认GPU VAE服务须自己重建匹配包，不能关闭身份检查。
3. 正常GPU VAE服务用`Decode_FullVAE`或`Decode_LightVAE`选择该完成HIGH；LOW/HIGH缓存选自己的真实path/SHA，目录占位不可运行。Cold则选Full产生的完成LOW，只继续HIGH。

声音锚是reference_only，不是drive_audio或final_audio；没有替换源配音。未完成LOW partial4声音不锁定。EAV保持report_only，不声称增强；Relay独立于参考包。示例采样尺寸448×256→既有learned1.2实际512×288，124原生帧后Trim120帧／24fps／5秒。新素材、事件、参数必须重新准备自己的完整缓存，不能借本次通过证明任意任务质量。

## LightVAE

原作者：[LynnReal-Onmi-light-vae](https://huggingface.co/stdstu123/LynnReal-Onmi-light-vae)。使用前自行理解并符合其地域／商业等许可；不沿用其他人的个人确认。

固定[Kijai转换件](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/d8023be02fefbb3633b0cd335c3879f91177299d/minimax_h3_lynnreal_light_vae_int8_convrot.safetensors)放入Core配置的`models/vae/t8-light`或其它已配置VAE目录；普通VAELoader选择其实际菜单值，连接T8 AVDecode视频VAE。音频VAE不变；不是HyperVAE2×，不因此改倍率／分辨率。

实际26层／24通道严格Core加载及同完成HIGH对照已通过，0新NFE。只证明指定decode，不保证encoder、全部素材或通用提速；本仓库不分发权重。

导演台两条规则在“更多→镜头证据与规则”，不是本目录中的另一个QA工程。仍须填写实际参数、编译待审、明确采用／回退与保存；不会自动改镜头时长或声音路由。

五项指定代表均已人工通过；这些UI图、Cold及Full／Contain模板的边界单独列明。详细说明：[总览](../../../docs/RADAR_R6.md)、[参考包](../../../docs/RADAR_R6_REFERENCES.md)、[后处理](../../../docs/RADAR_R6_DELIVERY.md)、[连续规则](../../../docs/RADAR_R6_CONTINUITY.md)。旧图保持，新文件名遵循Windows短路径预算。
