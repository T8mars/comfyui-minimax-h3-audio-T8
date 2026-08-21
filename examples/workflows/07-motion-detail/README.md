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
- `H3_Enhance_A_Video_FETA_Strict_Sage_T2VA_Stock20`：由单一组合节点同时绑定FETA与严格Sage HND后端；不要再串第三方H3 Sage节点。
- `H3_Enhance_A_Video_FETA_Prompt_Relay_T2VA_Stock20`：由单一组合节点先执行Prompt Relay局部事件路由，再只对目标视频输出行施加FETA；不要串两个独立Attention拥有者。
- 其他单路线工作流用于A/B诊断。

## 当前成果

五条路线均有同素材实测、自动指标和盲测工作流；Tail、Restart/STG、Model-Time Bias和Temporal Detail的作用机制不同。用户此前认为部分路线观感接近，项目没有把任何单路线强制设为全局最佳。

FETA 路线已完成 736×416 与 1152×640 两档、124帧、20步、同 seed 基线/增强对照。0.7MP档20次前向和50个主块全部命中，`g=1.0000～1.0466`、平均约 `1.00034`，新增工作区约7.90MiB，总耗时约增加7.2%；两路均为1152×640、124帧、24fps、32kHz双声道并通过三轮严格解码。自动代理显示运动轨迹确有变化，但清晰度没有明确提升，声音也不是bit-exact，因此当前只证明机械可用，不证明稳定提质。

追加的0.7MP单对照覆盖 I2VA、FL2VA、L2VA 和严格 Alpha8 Turbo8。四组增强端均完成预期的`20×50`或`8×50`审计，八条成片均为1152×640、124帧、24fps、32kHz双声道，并各通过三轮严格解码。FL2VA 本组变化很小；I2VA、L2VA、Turbo8 的轨迹和音频变化明显，但自动锐度没有提升证据。后续移除了完整packed输出的额外复制，复跑与旧输出逐帧/逐样本一致；但整卡最低余量仍曾降到271MiB，因此不宣称通用16GB安全。

Ref2VA 和任务型 Hybrid 也各完成一组 1152×640、124帧、Stock20 的 disabled/apply 实测。两条增强端都命中20次前向、每次50个主块；接受成片均通过视频、音频和联合解码门。Ref2VA 的自动音频相关约0.8695且最低显存余量仅约417MiB，Hybrid约0.9867；这些值只说明差异大小，不能替代盲看、盲听或参考遵循审核。Hybrid在这里指“首帧条件 + 独立参考图”的任务类型，不是混合模型权重节点。

## 使用方法与注意事项

先一次只启用一种方法，固定图像、提示词、seed、分辨率和NFE进行对比。Restart会联合迁移AV状态；STG会增加额外模型前向并可能明显改变声音；Temporal Detail属于生成后像素处理。需要组合时只用Mixer的明确参数和冲突检查。

FETA 必须按工作流中的顺序连接，并保留 Runtime Audit。`disabled` 才是严格关闭；`tau=0` 不是关闭。普通 EAV 节点仍只接受无参考块的 T2VA / I2VA / FL2VA / L2VA；Turbo8 仅接受模板中的修正 Alpha8 bypass LoRA。Ref2VA / Hybrid 必须改用独立 Reference Composer，并且当前只开放原生 Stock20布局。两条参考任务已通过精确PackedLayout、导入接线和单组真实0.7MP A/B机械门，但尚未完成用户盲评。Prompt Relay、BlockCache、Sage、STG、Long Video、其他 LoRA、模型权重 Hybrid、任意中间关键帧和 denoise mask 仍会主动拒绝，不能把报错节点绕开继续跑。

Strict Sage模板是一个独立追加路线：组合节点直接调用本机 `sageattention.sageattn` 的HND内核，并由Runtime Audit要求每次模型前向50个成功Sage调用、零失败、零静默回退。KJNodes的MiniMax H3 Sage节点会整块替换`Attention.forward`并绕过FETA观测入口，因此不能与普通EAV节点串联。本机单条1152×640×124、Stock20实测完成20次前向、1000次FETA测量和1000次Sage调用；H.264/AAC成片通过视频、音频和联合三轮严格解码。与同seed原生attention增强端相比，视频SSIM约0.8641、音频相关约0.9145，只证明结果发生变化；尚无人工盲评，也不授予画质、速度、音频非劣或通用16GB安全结论。

Prompt Relay组合模板同样是隔离追加路线。普通EAV和普通Prompt Relay都需要拥有同一个diffusion wrapper与`optimized_attention_override`，所以不能直接串联；组合节点会验证当前Relay绑定、至少两个事件、PackedLayout与wrapper归属，然后在一次attention调用中依次执行Relay路由和FETA目标视频增益。`disabled`仅关闭FETA并保留原Relay MODEL不变，适合作为单变量对照。当前只开放Stock20 T2VA；确定性合同、注册顺序、导入接线和Runtime Audit交接已通过，但真实0.7MP生成、听感、重复显存和质量门仍待完成，因此不得宣传提质、音频非劣或16GB安全。
