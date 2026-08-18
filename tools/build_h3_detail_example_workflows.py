#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .api_to_frontend_workflow import _get_json, convert
except ImportError:
    from api_to_frontend_workflow import _get_json, convert  # type: ignore[no-redef]


EXAMPLES = {
    "tail3": (
        "2026-08-18_H3_Hanfu_Tail_Detail_3Step_Advanced_EXP.json",
        "红色汉服：尾段3步联合AV细化",
        """## 用途：最后区间减小步长，不是升噪或修脸

本例固定1152×640、124帧、Turbo双时钟8步、同一10A首帧/提示词/seed，并把`extra_tail_steps`设为3。最后非零视频sigma按75%、50%、25%插入后降到精确0；音频sigma由shift12→3映射。实际为11次联合AV DiT前向，约比8步多37.5%计算。

节点自身默认是+1；改为0才是精确透传。末尾0只是积分终点，不会再调用模型。请完整检查人物、小脸、纱衣、闪烁、非要求语音、风声/布料声和音画同步；不能仅凭单帧锐度宣传质量提升。""",
    ),
    "time_bias": (
        "2026-08-18_H3_Hanfu_Model_Time_Bias_Advanced_EXP.json",
        "红色汉服：平滑共享AV模型时间偏置",
        """## 用途：同8 NFE的模型时间实验

`bias=-0.05`只在70%～100%进度内使用真正的`sin²`包络，让共享AV Transformer看到略低的模型sigma；Euler积分仍使用原始双时钟sigmas，总NFE仍是8。它没有随机加噪、没有改mask，也不是CFG或锐化。

H3音画在同一个Transformer中，偏置会同时改变画面与声音预测。必须与原8步同seed完整试听/观看；若出现运动幅度下降、身份漂移、额外语音或音效退化，应直接否决。""",
    ),
    "rf_restart": (
        "2026-08-18_H3_Hanfu_RF_Restart_Advanced_EXP.json",
        "红色汉服：联合AV Rectified-Flow Restart",
        """## 用途：真正重新加噪后再下降，高风险EXP

本例先完成8步，再以video sigma 0.15重新加噪；audio sigma按双时钟映射，seed固定为2608183001，然后增加3次联合AV下降。公式是`x_sigma=(1-sigma)*x_clean+sigma*epsilon`，总计11 NFE。

这和Navyblue/model-time bias完全不同，是真随机Restart。首版故意不提供video-only模式，因为共享Transformer下冻结audio并不保证联合分布正确。重点检查画面结构是否被重写、声音是否变化、是否新增说话/噪声及A/V同步。""",
    ),
    "stg": (
        "2026-08-18_H3_Hanfu_STG_Advanced_EXP.json",
        "红色汉服：H3时空引导",
        """## 用途：H3专用skip-block时空引导

本例`scale=0.60`、block 25、进度25%～85%。每个生效步额外运行一条跳过该H3 double block的正条件联合AV分支，再将差异加回主预测。它不是通用图像SLG的无脑复制，也不是脸部修复。

同block已有Block Cache或其他replacement时节点会fail closed。STG会显著增加计算，并可能同时改变声音；先单独运行，不要和Restart/时间偏置叠加。观察高速旋转、纱衣连续性、小脸、闪烁和完整音轨。""",
    ),
    "temporal_detail": (
        "2026-08-18_H3_Hanfu_Temporal_Detail_Advanced_EXP.json",
        "红色汉服：解码帧时序保护细节增强",
        """## 用途：后期亮度细节增强，不是生成式修复

本例在原8步解码帧后使用strength 0.35、motion threshold 0.04、temporal guard 0.85。它只增强亮度高频，并在相邻帧变化大的区域自动减弱，以降低闪烁；`upscale_factor=1.0`不改尺寸。大于1时使用保比例优先的32对齐，绝不缩小输入。默认每8帧处理并带一帧halo，输出超过2.1MP会fail closed，避免长视频高倍率临时张量失控。

该节点不接触音频，但也不能补回本来不存在的人脸身份、五官或布料几何。若只产生边缘过冲、颗粒、光晕或时序抖动，应降低strength或停用。""",
    ),
    "mixer": (
        "2026-08-18_H3_Hanfu_Detail_Mixer_Advanced_EXP.json",
        "红色汉服：混合细节采样（Tail + Bias + STG）",
        """## 用途：在一个节点中组合生成阶段的细节实验

本例启用Tail +1、Model-Time Bias -0.025（70%～95%）和STG 0.35/block25（25%～85%）；RF Restart默认关闭。节点报告会分别列出积分NFE、STG额外弱分支前向和计划中的联合AV Transformer总调用数。不要只看`steps=8`误判真实成本。

四个开关默认全部关闭；Restart是真随机联合音画重新加噪，风险和成本最高，只建议单独受控测试。旧五个节点和旧工作流未被替换。解码后的Temporal Detail仍是独立IMAGE节点，音频从AV Decode直接旁路到保存节点。完整检查人物结构、运动幅度、闪烁、额外语音、音效、音乐和音画同步；本例仍是EXP，不承诺质量提升或16GB安全。""",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build six importable H3 detail workflows with notes.")
    parser.add_argument("prompt_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--route", choices=sorted(EXAMPLES), help="Build only one route.")
    args = parser.parse_args()

    object_info = _get_json(f"{args.server.rstrip('/')}/object_info")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = EXAMPLES.items() if args.route is None else [(args.route, EXAMPLES[args.route])]
    for route, (filename, title, note) in selected:
        prompt_path = args.prompt_dir / f"{route}.json"
        if route == "mixer" and not prompt_path.exists():
            prompt_path = (
                Path(__file__).resolve().parents[1]
                / "tests"
                / "fixtures"
                / "api"
                / "detail_mixer_advanced_api.json"
            )
        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
        workflow = convert(prompt, object_info, title)
        note_id = int(workflow["last_node_id"]) + 1
        workflow["nodes"].append(
            {
                "id": note_id,
                "type": "MarkdownNote",
                "title": "参数、成本、科学边界与审片重点",
                "pos": [1320, -680],
                "size": [1500, 560],
                "flags": {},
                "order": len(workflow["nodes"]),
                "mode": 0,
                "color": "#2d3f66",
                "bgcolor": "#111827",
                "inputs": [],
                "outputs": [],
                "properties": {},
                "widgets_values": [note],
            }
        )
        workflow["last_node_id"] = note_id
        if route == "mixer":
            extra_notes = [
                (
                    "混合采样器参数建议",
                    [850, 1120],
                    [860, 390],
                    """## 建议起点（不是质量保证）

- `Tail`: 开，+1；只细分最后区间。
- `Model-Time Bias`: 开，-0.025，70%～95%。
- `STG`: 开，0.35，block 25，25%～85%。
- `RF Restart`: 默认关；它会对联合音画真正重新加噪。

报告中的`actual_nfe`是积分调用数；`planned_joint_av_forwards`还包含STG弱分支，才更接近真实Transformer成本。已有post-CFG或double-block替换冲突会直接报错，不静默覆盖。""",
                ),
                (
                    "解码后细节与音频旁路",
                    [2600, 520],
                    [900, 330],
                    """## Temporal Detail必须位于AV Decode之后

它只处理解码后的IMAGE，示例使用`strength=0.20 / guard=0.90 / upscale=1.0`。它不是生成采样器，也不能恢复不存在的身份或五官。音频从AV Decode直接连到保存节点，完全绕过此后期节点。""",
                ),
                (
                    "最终审片清单",
                    [3080, 460],
                    [900, 360],
                    """## 保存前后都要完整看与听

检查：小脸与手脚结构、快速旋转是否被压慢、纱衣边缘/闪烁、额外说话、风声/布料声/音乐、削波和音画同步。任一声音退化都不能用“只是画面参数”解释；H3生成阶段四种机制都经过同一个联合AV Transformer。""",
                ),
            ]
            for extra_title, position, size, extra_text in extra_notes:
                note_id += 1
                workflow["nodes"].append(
                    {
                        "id": note_id,
                        "type": "MarkdownNote",
                        "title": extra_title,
                        "pos": position,
                        "size": size,
                        "flags": {},
                        "order": len(workflow["nodes"]),
                        "mode": 0,
                        "color": "#5c3b1e",
                        "bgcolor": "#1f1710",
                        "inputs": [],
                        "outputs": [],
                        "properties": {},
                        "widgets_values": [extra_text],
                    }
                )
            workflow["last_node_id"] = note_id
        destination = args.output_dir / filename
        destination.write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
