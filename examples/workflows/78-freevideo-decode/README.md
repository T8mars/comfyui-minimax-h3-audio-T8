# FreeVideo 独立视频解码 EXP

- `FVQ_Decode_EXP.json`：Quality完整HIGH／SINGLE。
- `FV_Decode_EXP.json`：旧FreeVideo 8+2／真4+4完整HIGH；不能接LOW或MID。

这是新增入口，旧工作流与Core VAE Decode保留。先按[说明](../../../docs/FREEVIDEO_DECODER_EXP.md)准备官方三分片视频VAE和独立decoder配置，再填写Stage manifest路径、SHA和decoder_config路径。公开示例故意留空这些私人路径，不会直接运行，不自动下载或安装。

默认eager，compile可显式选；编译可能更慢并有舍入差，不保证提速或逐位无损。第二输出为原音频latent，使用正常音频VAE解码，不向视频子进程传音频。示例裁齐为0起点、24fps、5秒，其他时长需按真实Stage调整。

本机只有一份已完成Quality HIGH做过真实画布eager／compile对照：124准备帧裁至120帧，两路完整音画已核，0新增采样。旧版入口及其他配方不借此获得GPU质量资格；最终人审尚未完成。
