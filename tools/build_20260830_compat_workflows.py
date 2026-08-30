from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
WORKFLOWS = ROOT / "examples" / "workflows"
USER_WORKFLOWS = (
    COMFY_ROOT / "user" / "default" / "workflows" / "MiniMax H3 T8"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(relative_path: Path, workflow: dict) -> None:
    text = json.dumps(workflow, ensure_ascii=False, indent=2) + "\n"
    source = WORKFLOWS / relative_path
    mirror = USER_WORKFLOWS / relative_path
    source.parent.mkdir(parents=True, exist_ok=True)
    mirror.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(text, encoding="utf-8")
    mirror.write_text(text, encoding="utf-8")


def _node(workflow: dict, node_id: int) -> dict:
    return next(node for node in workflow["nodes"] if int(node["id"]) == node_id)


def build_fast_h3_vsa() -> None:
    base = _read(
        WORKFLOWS
        / "01-basic-generation"
        / "2026-08-06_H3_Turbo_Stable_4V4A.json"
    )
    workflow = copy.deepcopy(base)

    lora = _node(workflow, 2)
    lora["type"] = "MiniMaxH3LoRACompatibilityLoaderT8Advanced"
    lora["title"] = "2. Load official FastH3 VSA adapter (LoRA + exact deltas + 50 gates)"
    lora["properties"] = {
        "Node name for S&R": "MiniMaxH3LoRACompatibilityLoaderT8Advanced"
    }
    lora["widgets_values"] = [
        "FastH3-VSA\\vsa-datafree\\adapter_model.safetensors",
        1.0,
    ]
    lora["outputs"] = [
        lora["outputs"][0],
        {"name": "report_json", "type": "STRING", "links": []},
    ]

    conditioning = _node(workflow, 6)
    conditioning["title"] = "3. Plain T2VA, 832x480, 124 frames (about 0.4MP / 5.17s)"
    conditioning["widgets_values"][0] = (
        "A cinematic medium shot of a woman in flowing red Hanfu beneath moonlit "
        "clouds. She turns toward the camera, waves naturally and says in Mandarin: "
        "‘你在干嘛呢，我在这里呀，看看效果如何。’ Stable identity, coherent hands, "
        "detailed gauze fabric, synchronized clear speech, wind and cloth ambience."
    )
    conditioning["widgets_values"][1:4] = [832, 480, 124]

    setup = _node(workflow, 7)
    setup["type"] = "MiniMaxH3FastH34StepSetupT8Advanced"
    setup["title"] = "4. FastH3 4-step + real learned-gate VSA 90% / tile 64"
    setup["properties"] = {
        "Node name for S&R": "MiniMaxH3FastH34StepSetupT8Advanced"
    }
    setup["widgets_values"] = ["t2va_only", "external_vsa_if_available"]
    setup["outputs"].append(
        {"name": "report_json", "type": "STRING", "links": []}
    )

    save = _node(workflow, 12)
    save["title"] = "Save FastH3 VSA candidate"
    if isinstance(save["widgets_values"], dict):
        save["widgets_values"]["filename_prefix"] = (
            "MiniMaxH3/FastH3_VSA_T2VA_4Step/fast_h3_vsa_0p4mp_5s"
        )

    workflow["nodes"].append(
        {
            "id": 13,
            "type": "MarkdownNote",
            "title": "FastH3 VSA 使用边界",
            "pos": [820, -620],
            "size": [650, 340],
            "flags": {},
            "order": 12,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": {"Node name for S&R": "MarkdownNote"},
            "widgets_values": [
                "## FastH3 Preview v1：只用于 T2VA 4步\n\n"
                "模型：`FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA` 的 "
                "`vsa-datafree/adapter_model.safetensors`。放入 "
                "`ComfyUI/models/loras/FastH3-VSA/vsa-datafree/`。\n\n"
                "VSA 需要带 `topk_ratio / block_len / coarse_gate` 的 Comfy Kitchen。"
                "如果内核或 learned gate 不完整，Setup 会明确报告并回退到有效的 Dense 4步，"
                "不会把 Dense 冒充 VSA。FL2VA/Ref2VA 不属于本预览模型的训练范围。"
            ],
            "color": "#3f3a1f",
            "bgcolor": "#242112",
        }
    )
    workflow["last_node_id"] = 13
    _write(
        Path("10-speed")
        / "2026-08-30_H3_FastH3_VSA_T2VA_4Step_0p4MP_Advanced_EXP.json",
        workflow,
    )


def build_long_video_second_pass() -> None:
    base = _read(
        WORKFLOWS
        / "04-long-video"
        / "2026-08-27_H3_In_Node_Long_Video_Prompt_Relay_EAV_Stock20_Advanced_EXP.json"
    )
    workflow = copy.deepcopy(base)
    runner = _node(workflow, 6)
    runner["title"] = (
        "7. Long Video + Prompt Relay + EAV + optional low-sigma second pass"
    )
    runner["inputs"].append(
        {
            "name": "long_video_sampling_plan",
            "type": "H3_T8_LONG_VIDEO_SAMPLING_PLAN",
            "link": 6,
        }
    )
    runner["widgets_values"][0] = "h3_long_video_relay_eav_manual_second_pass_demo"

    workflow["nodes"].append(
        {
            "id": 11,
            "type": "MiniMaxH3LongVideoSamplingPlanT8Advanced",
            "title": "6. Manual low-sigma second pass / 手动低Sigma二次采样",
            "pos": [1040, -330],
            "size": [650, 230],
            "flags": {},
            "order": 5,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {
                    "name": "sampling_plan",
                    "type": "H3_T8_LONG_VIDEO_SAMPLING_PLAN",
                    "links": [6],
                },
                {"name": "report_json", "type": "STRING", "links": []},
            ],
            "properties": {
                "Node name for S&R": "MiniMaxH3LongVideoSamplingPlanT8Advanced"
            },
            "widgets_values": [
                "manual_second_pass",
                1,
                "video_sigma_linear",
                "0.5, 0.412, 0.350, 0",
            ],
            "color": "#614a1f",
            "bgcolor": "#332712",
        }
    )
    workflow["nodes"].append(
        {
            "id": 12,
            "type": "MarkdownNote",
            "title": "Sampling Plan 使用说明",
            "pos": [1730, -330],
            "size": [600, 300],
            "flags": {},
            "order": 11,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": {"Node name for S&R": "MarkdownNote"},
            "widgets_values": [
                "## 不破坏旧长视频\n\n"
                "断开此节点，或把 mode 设为 `disabled`，旧采样路径逐值不变。\n\n"
                "`tail_subdivide`：只细分原采样尾段；`manual_second_pass`：每段完成主采样后，"
                "再按手动 Sigma 独立采样。两遍共用同一份有界 preview cache 状态；音频仍由原 "
                "H3 双时钟联合采样，不做独立加噪/冻结。Prompt Relay 继续按全局时间线工作。"
            ],
            "color": "#3f3a1f",
            "bgcolor": "#242112",
        }
    )
    workflow["links"].append(
        [
            6,
            11,
            0,
            6,
            len(runner["inputs"]) - 1,
            "H3_T8_LONG_VIDEO_SAMPLING_PLAN",
        ]
    )
    workflow["last_node_id"] = 12
    workflow["last_link_id"] = 6
    _write(
        Path("04-long-video")
        / "2026-08-30_H3_In_Node_Long_Video_Prompt_Relay_EAV_Manual_Second_Pass_Advanced_EXP.json",
        workflow,
    )


if __name__ == "__main__":
    build_fast_h3_vsa()
    build_long_video_second_pass()
