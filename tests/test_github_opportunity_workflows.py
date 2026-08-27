from __future__ import annotations

import json
from pathlib import Path

from h3_audio_t8_pkg.nodes_prompt_rewriter_8b_advanced import (
    MiniMaxH3PromptRewriter8BT8Advanced,
)
from helpers import plugin_widget_map


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = ROOT / "examples" / "workflows"


def _load(category: str, filename: str) -> dict:
    return json.loads((WORKFLOW_ROOT / category / filename).read_text(encoding="utf-8"))


def _by_type(workflow: dict) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for node in workflow["nodes"]:
        result.setdefault(node["type"], []).append(node)
    return result


def _assert_importable_contract(workflow: dict) -> None:
    node_ids = {node["id"] for node in workflow["nodes"]}
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(node_ids)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert len([node for node in workflow["nodes"] if node["type"] == "MarkdownNote"]) == 3
    for link in workflow["links"]:
        assert link[1] in node_ids
        assert link[3] in node_ids


def test_prompt_rewriter_workflow_has_safe_local_unload_defaults():
    workflow = _load(
        "14-prompt-relay",
        "2026-08-22_H3_Prompt_Rewriter_8B_Advanced_EXP.json",
    )
    _assert_importable_contract(workflow)
    node = _by_type(workflow)["MiniMaxH3PromptRewriter8BT8Advanced"][0]
    values = plugin_widget_map(node, MiniMaxH3PromptRewriter8BT8Advanced)
    assert values["base_model_path"] == "Qwen3-VL-8B-Instruct"
    assert values["adapter_path"] == "MiniMax-H3-Prompt-Rewriter-LoRA-8B"
    assert values["load_policy"] == "auto_cpu_offload"
    assert values["max_new_tokens"] == 1024
    assert values["unload_after_generate"] is True
    assert values["allow_hub_download"] is False


def test_lanpaint_workflow_uses_external_sampler_and_one_interval_source():
    workflow = _load(
        "03-image-video-edit",
        "2026-08-22_H3_LanPaint_AV_Local_Repair_Advanced_EXP.json",
    )
    _assert_importable_contract(workflow)
    types = _by_type(workflow)
    assert len(types["MiniMaxH3LanPaintAVPrepareT8Advanced"]) == 1
    assert len(types["LanPaint_SamplerCustomAdvanced"]) == 1
    assert len(types["MiniMaxH3LanPaintAVCompositeT8Advanced"]) == 1
    assert types["MiniMaxH3DualClockSamplerT8"][0]["widgets_values"][:3] == [20, 12.0, 3.0]

    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    prepare = types["MiniMaxH3LanPaintAVPrepareT8Advanced"][0]
    composite = types["MiniMaxH3LanPaintAVCompositeT8Advanced"][0]
    interval_input = next(item for item in composite["inputs"] if item["name"] == "audio_intervals")
    source = links[interval_input["link"]]
    assert nodes[source[1]] is prepare
    assert source[2] == 3


def test_blockswap_workflow_is_isolated_from_comfy_model_and_turbo_lora():
    workflow = _load(
        "12-system-memory",
        "2026-08-22_H3_External_BlockSwap_Stock20_Advanced_EXP.json",
    )
    _assert_importable_contract(workflow)
    types = _by_type(workflow)
    assert "UNETLoader" not in types
    assert "LoraLoaderModelOnly" not in types
    assert len(types["MiniMaxH3ExternalBlockSwapBridgeT8Advanced"]) == 1
    assert len(types["MiniMaxH3KSampler"]) == 1
    bridge = types["MiniMaxH3ExternalBlockSwapBridgeT8Advanced"][0]
    sampler = types["MiniMaxH3KSampler"][0]
    assert bridge["widgets_values"][0] == "upstream_default_auto"
    assert sampler["widgets_values"][2:4] == [20, 1.0]
