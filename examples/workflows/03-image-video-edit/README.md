# 图像与视频编辑

这一组用于把H3的参考能力用于单帧语义编辑、源视频重绘和Ref2VA视觉参考强度实验。

## 推荐入口

- `2026-08-07_H3_Still_Edit_22Frames_EXP.json`：用22帧短窗口生成并提取单帧，适合图像编辑实验。
- `2026-08-09_H3_Source_Video_Repaint_Stock20_EXP.json`：对源视频做联合AV重绘。
- `2026-08-10_H3_Ref2VA_Visual_Reference_Strength_EXP.json`：研究统一视觉参考噪声强度。
- `2026-08-28_H3_CADS_Visual_Reference_Annealing_Advanced_EXP.json`：按CADS论文公式在采样早期扰动视觉参考，并在后期恢复干净条件；音频条件和目标音频不变。
- `2026-08-22_H3_LanPaint_AV_Local_Repair_Advanced_EXP.json`：用独立 LanPaint 采样器局部重绘画面和指定音频区间，并把未声明区域回贴为源素材。

## 当前成果

22帧链、源视频latent/mask和音频复用均有机械回归；参考强度是全局条件参数，不应误解为每张参考图的独立精确权重。

## 使用方法与注意事项

输入图像必须按目标画幅做等比裁切或填充，禁止强行拉伸。图像编辑仍由视频模型完成，不等同于专用文生图模型。源视频重绘时保留原声请使用锁音频策略，并核对帧率、时长和最终mux。

LanPaint 路线需要单独安装 `scraed/LanPaint`；本机核对 revision 为 `32cf848e93971da380d868936e007f5611218bee`。图片蒙版白色表示重绘、黑色表示保留；音频用秒区间 JSON 单独声明。Prepare 的 `report_json` 应直接连接 Composite 的 `audio_intervals`，让采样和回贴共享同一事实源。推荐从 736×416、124帧、Stock20、LanPaint 5/5.0/0.2 开始；当前只有结构、掩码和回贴合同验证，不宣称提质或16GB安全。

CADS工作流建议从`noise_scale=0.10 / tau1=0.60 / tau2=0.90 / rescale_mix=1.0`开始固定seed对比。`paper_independent`每步使用独立且可复现的噪声；`stable_fixed_path`沿同一噪声方向退火，通常更平滑，但属于H3适配实验。强度升高可能增加构图多样性，也可能降低人物身份、动作、构图与首尾帧遵循度，因此不能默认开启。
