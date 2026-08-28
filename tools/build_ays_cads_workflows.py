from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "examples" / "workflows"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _note(node_id: int, title: str, text: str, pos: list[int]) -> dict:
    return {
        "id": node_id,
        "type": "MarkdownNote",
        "title": title,
        "pos": pos,
        "size": [680, 290],
        "flags": {},
        "order": 99,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "properties": {
            "Node name for S&R": "MarkdownNote",
            "cnr_id": "comfy-core",
        },
        "widgets_values": text,
    }


def build_ays() -> None:
    source = WORKFLOWS / "01-basic-generation" / "2026-08-06_H3_Turbo_Stable_4V4A.json"
    workflow = copy.deepcopy(_read(source))
    sampler = next(node for node in workflow["nodes"] if node["id"] == 7)
    sampler.update(
        {
            "type": "MiniMaxH3DualClockAYSScheduleT8Advanced",
            "title": "AYS research contract: native baseline or externally calibrated H3 knots",
            "size": [650, 360],
            "outputs": [
                {"name": "model", "type": "MODEL", "links": [7]},
                {"name": "sampler", "type": "SAMPLER", "links": [11]},
                {"name": "sigmas", "type": "SIGMAS", "links": [12]},
                {"name": "report_json", "type": "STRING", "links": []},
            ],
            "properties": {
                "Node name for S&R": "MiniMaxH3DualClockAYSScheduleT8Advanced"
            },
            "widgets_values": [
                8,
                12.0,
                3.0,
                "native_flow_baseline",
                "[1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]",
                "unvalidated H3 calibration",
                "dual_clock_euler",
            ],
        }
    )
    save = next(node for node in workflow["nodes"] if node["id"] == 12)
    save["widgets_values"]["filename_prefix"] = "MiniMaxH3_T8/AYS_contract_8step_EXP"
    note_id = max(node["id"] for node in workflow["nodes"]) + 1
    workflow["nodes"].append(
        _note(
            note_id,
            "AYS 科学边界与用法",
            "## AYS Schedule Contract\n\n"
            "- 默认 `native_flow_baseline` 与原 H3 8 步时间表一致，不声称提升质量。\n"
            "- 只有拿到针对 **MiniMax H3 + 当前求解器 + 当前任务数据** 校准的 `steps+1` 个 base sigma 后，才切换 `manual_h3_calibrated`。\n"
            "- 列表必须从 1.0 严格递减到 0.0；节点会分别用 video/audio shift 映射同一组 base knots。\n"
            "- 不要把 SD、SDXL 或 SVD 的 AYS 数组直接复制到 H3。",
            [1020, -360],
        )
    )
    workflow["last_node_id"] = note_id
    _write(
        WORKFLOWS
        / "07-motion-detail"
        / "2026-08-28_H3_Dual_Clock_AYS_Schedule_Contract_Advanced_EXP.json",
        workflow,
    )


def build_cads() -> None:
    source = (
        WORKFLOWS
        / "03-image-video-edit"
        / "2026-08-10_H3_Ref2VA_Visual_Reference_Strength_EXP.json"
    )
    workflow = copy.deepcopy(_read(source))
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] != 7]
    workflow["links"] = [link for link in workflow["links"] if link[0] not in {6, 7}]
    conditioning = next(node for node in workflow["nodes"] if node["id"] == 6)
    conditioning["outputs"][0]["links"] = [20]
    guider = next(node for node in workflow["nodes"] if node["id"] == 10)
    guider["inputs"][1]["link"] = 20
    workflow["links"].append([20, 6, 0, 10, 1, "CONDITIONING"])

    # Route the source model through CADS before the existing Dual-Clock setup.
    model_link = next(link for link in workflow["links"] if link[0] == 1)
    model_link[3] = 14
    model_link[4] = 0
    dual_clock = next(node for node in workflow["nodes"] if node["id"] == 8)
    dual_clock["inputs"][0]["link"] = 21
    dual_clock["title"] = "Existing 20-step Dual-Clock sampler; CADS changes visual condition only"
    workflow["links"].append([21, 14, 0, 8, 0, "MODEL"])
    workflow["nodes"].append(
        {
            "id": 14,
            "type": "MiniMaxH3CADSVisualReferenceT8Advanced",
            "title": "CADS visual-reference annealing; audio stays native",
            "pos": [1120, 0],
            "size": [610, 270],
            "flags": {},
            "order": 13,
            "mode": 0,
            "inputs": [{"name": "model", "type": "MODEL", "link": 1}],
            "outputs": [
                {"name": "model", "type": "MODEL", "links": [21]},
                {"name": "report_json", "type": "STRING", "links": []},
            ],
            "properties": {
                "Node name for S&R": "MiniMaxH3CADSVisualReferenceT8Advanced"
            },
            "widgets_values": [0.1, 0.6, 0.9, 1.0, "paper_independent", 26082801],
        }
    )
    workflow["nodes"].append(
        _note(
            15,
            "CADS 使用说明",
            "## CADS Visual Reference Annealing\n\n"
            "- 只退火图像/视频参考与首尾关键帧，**不改音频条件和目标音频 latent**。\n"
            "- 从 `noise_scale=0.10, tau1=0.60, tau2=0.90, rescale_mix=1.0` 开始；先固定 seed 做 A/B。\n"
            "- `paper_independent` 每步使用独立确定性噪声；`stable_fixed_path` 使用同一噪声方向，通常更平滑但不是论文默认解释。\n"
            "- 强度越高，多样性可能增加，但身份、动作、构图和首尾帧约束可能下降；当前仍是 EXP。",
            [1120, -360],
        )
    )
    save = next(node for node in workflow["nodes"] if node["id"] == 13)
    save["widgets_values"]["filename_prefix"] = "MiniMaxH3_T8/Ref2VA_CADS_EXP"
    workflow["last_node_id"] = 15
    workflow["last_link_id"] = 21
    _write(
        WORKFLOWS
        / "03-image-video-edit"
        / "2026-08-28_H3_CADS_Visual_Reference_Annealing_Advanced_EXP.json",
        workflow,
    )


if __name__ == "__main__":
    build_ays()
    build_cads()
