# 高动态、尾段细节与混合采样

这一组研究高速运动、小脸稳定和采样尾段细节，包括Dynamic Guidance、额外尾段NFE、Model-Time Bias、联合AV Restart、H3 STG、时域后处理、Mixer，以及实验性的 Enhance-A-Video / FETA 时序注意力增强。

## 推荐入口

- `Motion_Quality_Advanced_8Step`：基础高动态控制。
- `Hanfu_Tail_Detail_3Step`：尾部追加3个逐渐趋近0的细化步。
- `Hanfu_Detail_Mixer`：显式组合已支持的细节策略，不要手工串多个独立采样器。
- `H3_Enhance_A_Video_FETA_Stock20`：Plain T2VA / Stock20 基准模板。
- `H3_Enhance_A_Video_FETA_I2VA_Stock20`：首帧生音视频；替换首帧后再做同 seed A/B。
- `H3_Enhance_A_Video_FETA_FL2VA_Stock20`：首尾帧生音视频；首尾图必须分别接入正确插槽。
- `H3_Enhance_A_Video_FETA_L2VA_Stock20`：尾帧生音视频；只接尾帧，不伪装成 I2VA。
- `H3_Enhance_A_Video_FETA_T2VA_Turbo8`：仅接受修正 Alpha8、208个 bypass hooks、strength 1.0 的严格 Turbo8 实验模板。
- `H3_Enhance_A_Video_FETA_Ref2VA_Stock20`：独立 Reference Composer；参考图进入原生参考块，FETA 只处理目标视频行。
- `H3_Enhance_A_Video_FETA_Hybrid_Stock20`：首帧 + 独立参考图的任务型 Hybrid；不要与混合模型权重节点混淆。
- 其他单路线工作流用于A/B诊断。

## 当前成果

五条路线均有同素材实测、自动指标和盲测工作流；Tail、Restart/STG、Model-Time Bias和Temporal Detail的作用机制不同。用户此前认为部分路线观感接近，项目没有把任何单路线强制设为全局最佳。

FETA 路线已完成 736×416 与 1152×640 两档、124帧、20步、同 seed 基线/增强对照。0.7MP档20次前向和50个主块全部命中，`g=1.0000～1.0466`、平均约 `1.00034`，新增工作区约7.90MiB，总耗时约增加7.2%；两路均为1152×640、124帧、24fps、32kHz双声道并通过三轮严格解码。自动代理显示运动轨迹确有变化，但清晰度没有明确提升，声音也不是bit-exact，因此当前只证明机械可用，不证明稳定提质。

追加的0.7MP单对照覆盖 I2VA、FL2VA、L2VA 和严格 Alpha8 Turbo8。四组增强端均完成预期的`20×50`或`8×50`审计，八条成片均为1152×640、124帧、24fps、32kHz双声道，并各通过三轮严格解码。FL2VA 本组变化很小；I2VA、L2VA、Turbo8 的轨迹和音频变化明显，但自动锐度没有提升证据。后续移除了完整packed输出的额外复制，复跑与旧输出逐帧/逐样本一致；但整卡最低余量仍曾降到271MiB，因此不宣称通用16GB安全。

## 使用方法与注意事项

先一次只启用一种方法，固定图像、提示词、seed、分辨率和NFE进行对比。Restart会联合迁移AV状态；STG会增加额外模型前向并可能明显改变声音；Temporal Detail属于生成后像素处理。需要组合时只用Mixer的明确参数和冲突检查。

FETA 必须按工作流中的顺序连接，并保留 Runtime Audit。`disabled` 才是严格关闭；`tau=0` 不是关闭。普通 EAV 节点仍只接受无参考块的 T2VA / I2VA / FL2VA / L2VA；Turbo8 仅接受模板中的修正 Alpha8 bypass LoRA。Ref2VA / Hybrid 必须改用独立 Reference Composer，并且当前只开放原生 Stock20布局。两条参考任务模板已通过精确PackedLayout和导入接线回归，但真实0.7MP A/B仍未完成。Prompt Relay、BlockCache、Sage、STG、Long Video、其他 LoRA、模型权重 Hybrid、任意中间关键帧和 denoise mask 仍会主动拒绝，不能把报错节点绕开继续跑。
