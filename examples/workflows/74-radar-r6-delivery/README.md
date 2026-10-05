# RADAR r6：完整原片后处理（EXP）

开发中，未发布；需本轮T8后处理保存节点及支持BoundingBox的Core。

- [Full_Master_Crop_EXP.json](Full_Master_Crop_EXP.json)：附加到已有完成工作流。将**全部MASK／Face回贴之后**的完整RGB和AUDIO接左侧Safe AV Save，它返回的已保存VIDEO同时连到读帧与后处理保存，保证master先落盘。它不是独立的模型生成图，不接partial LOW或中间x0。
- [Cold_Master_Crop_EXP.json](Cold_Master_Crop_EXP.json)：重开已有完整master，默认1920×1088→1920×1080，x0/y4。LoadVideo文件占位必须换成自己的视频，按实际尺寸修改Core裁切框；会裁掉上下内容。
- [Cold_Master_Contain_EXP.json](Cold_Master_Contain_EXP.json)：不愿裁内容时明确选此图，等比例缩放到目标范围后黑色留边。不是无损缩放，也不会偷偷替代裁切。

后处理确认默认false。只支持完整零起点24fps SDR、相同帧数和偶数宽高；不偷偷处理VFR/HDR或改音视频时间。Cold无需重跑模型／一采／二采。完整RGB仍会占RAM，不是流式解码。

声音从已保存master逐包复制，并核对音频包内容／时间线、解码PCM／时间线；不会再编码AAC或生成新声音。后处理只写新文件。失败状态`postprocess_failed`：成片出口阻断，原片在`master_video`独立出口，不冒充增强成功。状态文件和成片回执必须与实际媒体SHA匹配，不能只凭状态JSON当作完成。

指定5秒、512×288→512×280冷裁切真实画布及原片／裁切片对照已获用户通过，其他模板真实打开并另存／CPU schema通过；不宣传全部素材或后处理组合已通过。独立LightVAE对照随后也已通过，不重复此裁切审核。详见[合同与边界](../../../docs/RADAR_R6_DELIVERY.md)。
