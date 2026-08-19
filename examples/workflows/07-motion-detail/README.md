# 高动态、尾段细节与混合采样

这一组研究高速运动、小脸稳定和采样尾段细节，包括Dynamic Guidance、额外尾段NFE、Model-Time Bias、联合AV Restart、H3 STG、时域后处理和Mixer。

## 推荐入口

- `Motion_Quality_Advanced_8Step`：基础高动态控制。
- `Hanfu_Tail_Detail_3Step`：尾部追加3个逐渐趋近0的细化步。
- `Hanfu_Detail_Mixer`：显式组合已支持的细节策略，不要手工串多个独立采样器。
- 其他单路线工作流用于A/B诊断。

## 当前成果

五条路线均有同素材实测、自动指标和盲测工作流；Tail、Restart/STG、Model-Time Bias和Temporal Detail的作用机制不同。用户此前认为部分路线观感接近，项目没有把任何单路线强制设为全局最佳。

## 使用方法与注意事项

先一次只启用一种方法，固定图像、提示词、seed、分辨率和NFE进行对比。Restart会联合迁移AV状态；STG会增加额外模型前向并可能明显改变声音；Temporal Detail属于生成后像素处理。需要组合时只用Mixer的明确参数和冲突检查。
