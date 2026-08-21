from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import re

from h3_audio_t8_pkg.nodes import comfy_entrypoint
from tools.api_to_frontend_workflow import convert
from tools.repair_frontend_workflow_order import node_needs_repair, repair_workflow


def test_all_frontend_workflows_have_publication_date_prefix():
    root = Path(__file__).resolve().parents[1] / "examples" / "workflows"
    paths = sorted(root.rglob("*.json"))
    categories = sorted(path for path in root.iterdir() if path.is_dir())
    publication_name = re.compile(r"^\d{4}-\d{2}-\d{2}_.+\.json$")
    assert len(paths) == 98
    assert [path.name for path in categories] == [
        "01-basic-generation",
        "02-audio-control",
        "03-image-video-edit",
        "04-long-video",
        "05-speech-dialogue",
        "06-face-refine",
        "07-motion-detail",
        "08-multi-keyframe",
        "09-hybrid-model",
        "10-speed",
        "11-studio-production",
        "12-system-memory",
        "13-latent-upscale",
        "14-prompt-relay",
    ]
    assert (root / "README.md").is_file()
    assert all((category / "README.md").is_file() for category in categories)
    assert list(root.glob("*.json")) == []
    assert [path.name for path in paths if not publication_name.fullmatch(path.name)] == []


def test_prompt_relay_long_video_workflow_has_absolute_window_and_safe_model_order():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "examples"
        / "workflows"
        / "04-long-video"
        / "2026-08-20_H3_Prompt_Relay_Long_Video_Turbo8_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    by_type = {node["type"]: node for node in workflow["nodes"]}
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 3

    plan = by_type["MiniMaxH3PromptRelayPlanT8Advanced"]
    window = by_type["MiniMaxH3PromptRelayLongVideoPlanT8Advanced"]
    conditioning = by_type["MiniMaxH3PromptRelayLongVideoConditioningT8Advanced"]
    lora = by_type["LoraLoaderBypassModelOnly"]
    sampler = by_type["MiniMaxH3DualClockSamplerT8"]
    unet = by_type["UNETLoader"]

    links = {link[0]: link for link in workflow["links"]}

    def source_for_input(node, name):
        item = next(value for value in node["inputs"] if value["name"] == name)
        link = links[item["link"]]
        return nodes[link[1]], link[2]

    assert source_for_input(window, "prompt_relay_plan") == (plan, 0)
    assert source_for_input(conditioning, "prompt_relay_plan") == (window, 0)
    assert source_for_input(conditioning, "model") == (unet, 0)
    assert source_for_input(lora, "model") == (conditioning, 0)
    assert source_for_input(sampler, "model") == (lora, 0)
    assert lora["widgets_values"] == [
        "minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8-T8-convert.safetensors",
        1.0,
    ]
    assert sampler["widgets_values"][:3] == [8, 12.0, 3.0]
    assert "video_only_paper" == workflow["extra"]["t8_prompt_relay_long_video"][
        "default_route"
    ]

    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def _plugin_object_info() -> dict:
    extension = comfy_entrypoint()
    classes = asyncio.run(extension.get_node_list())
    result = {}
    for node_class in classes:
        info = node_class.define_schema().get_v1_info(node_class)
        result[info.name] = json.loads(json.dumps(info.__dict__))
    return result


def test_repair_restores_widget_names_values_and_link_slots():
    object_info = {
        "Source": {
            "input": {"required": {}},
            "output": ["INT"],
            "output_name": ["value"],
        },
        "Target": {
            "input": {
                "required": {
                    "text": ["STRING", {"default": ""}],
                    "width": ["INT", {"default": 1344}],
                    "mode": [["native", "lock"], {"default": "native"}],
                },
                "optional": {"image": ["IMAGE", {}]},
            },
            "output": [],
            "output_name": [],
        },
    }
    workflow = {
        "nodes": [
            {
                "id": 1,
                "type": "Source",
                "inputs": [],
                "outputs": [{"name": "value", "type": "INT", "links": [1]}],
                "widgets_values": [],
            },
            {
                "id": 2,
                "type": "Target",
                "inputs": [
                    {"name": "mode", "type": "COMBO", "widget": {"name": "mode"}, "link": None},
                    {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None},
                    {"name": "width", "type": "INT", "link": 1},
                ],
                "outputs": [],
                "widgets_values": ["native", "prompt"],
            },
        ],
        "links": [[1, 1, 0, 2, 2, "INT"]],
    }
    result = repair_workflow(workflow, object_info)
    assert result == {"repaired": ["2:Target"], "skipped": []}
    target = workflow["nodes"][1]
    assert [item["name"] for item in target["inputs"]] == [
        "text",
        "width",
        "mode",
        "image",
    ]
    assert target["widgets_values"] == ["prompt", 1344, "native"]
    assert workflow["links"][0][4:] == [1, "INT"]
    frozen = deepcopy(workflow)
    assert repair_workflow(workflow, object_info) == {"repaired": [], "skipped": []}
    assert workflow == frozen


def test_api_converter_uses_schema_order_not_prompt_dictionary_order():
    prompt = {
        "1": {
            "class_type": "Target",
            "inputs": {"mode": "native", "width": 640, "text": "hello"},
        }
    }
    object_info = {
        "Target": {
            "input": {
                "required": {
                    "text": ["STRING", {"default": ""}],
                    "width": ["INT", {"default": 1344}],
                    "mode": [["native", "lock"], {"default": "native"}],
                }
            },
            "output": [],
            "output_name": [],
            "display_name": "Target",
        }
    }
    workflow = convert(prompt, object_info, "schema order")
    node = workflow["nodes"][0]
    assert [item["name"] for item in node["inputs"]] == ["text", "width", "mode"]
    assert node["widgets_values"] == ["hello", 640, "native"]


def test_api_converter_keeps_full_slots_before_a_later_optional_link():
    prompt = {
        "1": {"class_type": "Source", "inputs": {}},
        "2": {
            "class_type": "Target",
            "inputs": {"text": "hello", "late_image": ["1", 0]},
        },
    }
    object_info = {
        "Source": {
            "input": {"required": {}},
            "output": ["IMAGE"],
            "output_name": ["image"],
            "display_name": "Source",
        },
        "Target": {
            "input": {
                "required": {"text": ["STRING", {"default": ""}]},
                "optional": {
                    "early_mask": ["MASK", {}],
                    "late_image": ["IMAGE", {}],
                },
            },
            "output": [],
            "output_name": [],
            "display_name": "Target",
        },
    }
    workflow = convert(prompt, object_info, "full optional slots")
    target = next(node for node in workflow["nodes"] if node["type"] == "Target")
    assert [item["name"] for item in target["inputs"]] == [
        "text",
        "early_mask",
        "late_image",
    ]
    assert target["inputs"][2]["link"] == 1
    assert workflow["links"] == [[1, 1, 0, 2, 2, "IMAGE"]]


def test_autogrow_repair_does_not_cross_match_overlapping_prefixes():
    object_info = {
        "Target": {
            "input": {
                "required": {},
                "optional": {
                    "ref_videos": [
                        "COMFY_AUTOGROW_V3",
                        {
                            "template": {
                                "input": {"required": {"ref_video": ["IMAGE", {}]}},
                                "prefix": "ref_video_",
                            }
                        },
                    ],
                    "ref_video_audios": [
                        "COMFY_AUTOGROW_V3",
                        {
                            "template": {
                                "input": {"required": {"ref_video_audio": ["AUDIO", {}]}},
                                "prefix": "ref_video_audio_",
                            }
                        },
                    ],
                },
            },
            "output": [],
            "output_name": [],
        }
    }
    workflow = {
        "nodes": [
            {
                "id": 1,
                "type": "Target",
                "inputs": [
                    {
                        "name": "ref_videos.ref_video_0",
                        "type": "IMAGE",
                        "link": 1,
                    },
                    {
                        "name": "ref_video_audios.ref_video_audio_0",
                        "type": "AUDIO",
                        "link": 2,
                    },
                ],
                "outputs": [],
                "widgets_values": [],
            }
        ],
        "links": [],
    }

    frozen = deepcopy(workflow)
    assert repair_workflow(workflow, object_info) == {"repaired": [], "skipped": []}
    assert workflow == frozen


def test_all_t8_frontend_workflows_match_current_schema_order():
    root = Path(__file__).resolve().parents[1] / "examples" / "workflows"
    object_info = _plugin_object_info()
    failures = []
    for path in sorted(root.rglob("*.json")):
        workflow = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list):
            continue
        nodes = {node["id"]: node for node in workflow["nodes"]}
        for node in nodes.values():
            info = object_info.get(node.get("type"))
            if info is not None and node_needs_repair(node, info):
                failures.append(f"{path.name}:{node['id']}:{node['type']}")
        for link in workflow.get("links", []):
            target = nodes[link[3]]
            if target.get("type") not in object_info:
                continue
            if link[4] >= len(target.get("inputs", [])):
                failures.append(f"{path.name}:link{link[0]}:target slot out of range")
                continue
            target_input = target["inputs"][link[4]]
            if target_input.get("link") != link[0]:
                failures.append(f"{path.name}:link{link[0]}:wrong target slot")
    assert failures == []
