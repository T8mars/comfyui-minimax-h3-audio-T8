from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "examples" / "workflows" / "04-long-video"
SOURCE = WORKFLOW_DIR / "2026-08-27_H3_In_Node_Long_Video_Prompt_Relay_EAV_Stock20_Advanced_EXP.json"
TARGET = WORKFLOW_DIR / "2026-08-28_H3_FreeNoise_Prompt_Relay_EAV_Long_Video_Advanced_EXP.json"


def main() -> None:
    workflow = json.loads(SOURCE.read_text(encoding="utf-8"))
    workflow["id"] = "b76a8d5b-19fd-46ec-aeae-55946389f955"
    workflow["last_node_id"] = 12
    workflow["last_link_id"] = 6
    nodes = {int(node["id"]): node for node in workflow["nodes"]}

    runner = nodes[6]
    runner["title"] = "7. One queue run · FreeNoise + Prompt Relay + EAV Stock20"
    runner["inputs"][0]["link"] = 6
    runner["widgets_values"][0] = "h3_freenoise_relay_eav_stock20_demo"
    runner["widgets_values"][-2] = 18
    runner["widgets_values"][-1] = "minimax_h3_freenoise_stock20_relay_eav"

    for node in workflow["nodes"]:
        if int(node["order"]) >= 5:
            node["order"] = int(node["order"]) + 1

    free_noise = {
        "id": 11,
        "type": "MiniMaxH3FreeNoiseLongVideoT8Advanced",
        "title": "6. FreeNoise video-noise rescheduling · no attention owner",
        "pos": [440, 570],
        "size": [540, 260],
        "flags": {},
        "order": 5,
        "mode": 0,
        "inputs": [{"name": "model", "type": "MODEL", "link": 1}],
        "outputs": [
            {"name": "model", "type": "MODEL", "links": [6]},
            {"name": "report_json", "type": "STRING", "links": None},
        ],
        "properties": {
            "cnr_id": "minimax-h3-audio-T8",
            "Node name for S&R": "MiniMaxH3FreeNoiseLongVideoT8Advanced",
        },
        "widgets_values": ["paper_permutation", 123456789, 0.65],
        "color": "#23506b",
        "bgcolor": "#142c3a",
    }
    note = {
        "id": 12,
        "type": "MarkdownNote",
        "title": "NOTE 5 · FreeNoise的H3适配边界",
        "pos": [1870, 1340],
        "size": [620, 340],
        "flags": {},
        "order": 11,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "properties": {"cnr_id": "comfy-core", "Node name for S&R": "MarkdownNote"},
        "widgets_values": [
            "## 只重排视频噪声，不抢占Attention\n`paper_permutation`让每段从同一视频噪声池做确定性时间置换；音频噪声保持原生独立，因此可以放在Prompt Relay/EAV之前。原论文还在一个长latent中做滑窗时序Attention融合；H3这里仍是独立124帧续段，所以属于噪声重排适配，不等于论文完整复现。改变mode、seed或比例后必须换新的`chain_id`。"
        ],
        "color": "#23506b",
        "bgcolor": "#142c3a",
    }
    workflow["nodes"].extend((free_noise, note))
    workflow["nodes"].sort(key=lambda node: int(node["order"]))
    workflow["links"][0] = [1, 1, 0, 11, 0, "MODEL"]
    workflow["links"].append([6, 11, 0, 6, 0, "MODEL"])
    TARGET.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(TARGET)


if __name__ == "__main__":
    main()
