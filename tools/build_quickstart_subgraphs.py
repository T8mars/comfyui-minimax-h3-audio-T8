from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBGRAPH_NAMESPACE = uuid.UUID("cbf6f46f-d78c-4ca5-a822-75d1b450e56c")


# Native ComfyUI 0.4 saves omit unlinked widget descriptors from node.inputs.
# Quick-start proxy inputs still need a real inner input socket, so keep the
# audited schema order only for node types whose controls are promoted below.
# This is deliberately not a runtime compatibility gate: it is an offline
# frontend serialization map covered by current-schema workflow tests.
INPUT_LAYOUTS = {
    "UNETLoader": [("unet_name", "COMBO", True), ("weight_dtype", "COMBO", True)],
    "LoraLoaderBypassModelOnly": [
        ("model", "MODEL", False),
        ("lora_name", "COMBO", True),
        ("strength_model", "FLOAT", True),
    ],
    "LoraLoaderModelOnly": [
        ("model", "MODEL", False),
        ("lora_name", "COMBO", True),
        ("strength_model", "FLOAT", True),
    ],
    "CLIPLoader": [
        ("clip_name", "COMBO", True),
        ("type", "COMBO", True),
        ("device", "COMBO", True),
    ],
    "VAELoader": [("vae_name", "COMBO", True)],
    "MiniMaxH3AudioConditioningT8": [
        ("clip", "CLIP", False),
        ("video_vae", "VAE", False),
        ("audio_vae", "VAE", False),
        ("prompt", "STRING", True),
        ("width", "INT", True),
        ("height", "INT", True),
        ("length", "INT", True),
        ("task_type", "COMBO", True),
        ("audio_mode", "COMBO", True),
        ("audio_denoise_strength", "FLOAT", True),
        ("add_source_as_reference", "BOOLEAN", True),
        ("prompt_primary_audio_ordinal", "INT", True),
        ("strict_prompt_tags", "BOOLEAN", True),
        ("ref_image_size", "COMBO", True),
        ("reference_video_policy", "COMBO", True),
        ("drive_audio", "AUDIO", False),
        ("final_audio", "AUDIO", False),
        ("first_frame", "IMAGE", False),
        ("last_frame", "IMAGE", False),
    ],
    "RandomNoise": [("noise_seed", "INT", True)],
    "MiniMaxH3DualClockSamplerT8": [
        ("model", "MODEL", False),
        ("av_latent", "LATENT", False),
        ("steps", "INT", True),
        ("shift_video", "FLOAT", True),
        ("shift_audio", "FLOAT", True),
        ("sampler_name", "COMBO", True),
        ("scheduler", "COMBO", True),
    ],
    "BasicScheduler": [
        ("model", "MODEL", False),
        ("scheduler", "COMBO", True),
        ("steps", "INT", True),
        ("denoise", "FLOAT", True),
    ],
    "KSamplerSelect": [("sampler_name", "COMBO", True)],
    "LoadAudio": [("audio", "COMBO", True)],
    "CreateVideo": [
        ("images", "IMAGE", False),
        ("fps", "FLOAT", True),
        ("audio", "AUDIO", False),
        ("bit_depth", "INT", True),
    ],
    "MiniMaxH3AudioPerceptualDriftAuditT8Advanced": [
        ("reference_audio", "AUDIO", False),
        ("candidate_audio", "AUDIO", False),
        ("analysis_window_ms", "FLOAT", True),
        ("hop_ms", "FLOAT", True),
        ("active_rms_floor_dbfs", "FLOAT", True),
        ("spectral_drift_threshold", "FLOAT", True),
        ("level_delta_threshold_db", "FLOAT", True),
        ("persistent_window_count", "INT", True),
        ("max_duration_delta_ms", "FLOAT", True),
    ],
    "MiniMaxH3AudioWindowT8": [
        ("audio", "AUDIO", False),
        ("scene_start_seconds", "FLOAT", True),
        ("scene_duration_seconds", "FLOAT", True),
        ("warmup_seconds", "FLOAT", True),
        ("cooldown_seconds", "FLOAT", True),
        ("ensure_minimum_context", "BOOLEAN", True),
    ],
    "MiniMaxH3LongVideoOrchestratorT8": [
        ("chain_id", "STRING", True),
        ("total_duration_seconds", "FLOAT", True),
        ("render_window_frames", "INT", True),
        ("context_frames", "COMBO", True),
        ("global_prompt", "STRING", True),
        ("segment_prompts_json", "STRING", True),
        ("base_seed", "INT", True),
        ("seed_policy", "COMBO", True),
        ("steps", "INT", True),
        ("shift_video", "FLOAT", True),
        ("shift_audio", "FLOAT", True),
        ("sampler_name", "COMBO", True),
        ("scheduler", "COMBO", True),
    ],
    "MiniMaxH3LongVideoConditioningT8": [
        ("model", "MODEL", False),
        ("clip", "CLIP", False),
        ("video_vae", "VAE", False),
        ("audio_vae", "VAE", False),
        ("context", "H3_T8_CONTEXT", False),
        ("segment_index", "INT", False),
        ("context_frames", "INT", False),
        ("context_audio", "COMBO", True),
        ("prompt", "STRING", True),
        ("width", "INT", True),
        ("height", "INT", True),
        ("length", "INT", False),
        ("task_type", "COMBO", True),
        ("audio_mode", "COMBO", True),
        ("audio_denoise_strength", "FLOAT", True),
        ("add_source_as_reference", "BOOLEAN", True),
        ("prompt_primary_audio_ordinal", "INT", True),
        ("strict_prompt_tags", "BOOLEAN", True),
        ("ref_image_size", "COMBO", True),
        ("reference_video_policy", "COMBO", True),
        ("drive_audio", "AUDIO", False),
        ("final_audio", "AUDIO", False),
        ("first_frame", "IMAGE", False),
        ("last_frame", "IMAGE", False),
    ],
    "MiniMaxH3LongVideoAcceptCandidateT8": [
        ("candidate_json_path", "STRING", False),
        ("accept_candidate", "BOOLEAN", True),
        ("replace_policy", "COMBO", True),
        ("strict_chain_identity", "BOOLEAN", True),
    ],
    "MiniMaxH3LongVideoCandidateSaveT8": [
        ("frames", "IMAGE", False),
        ("audio", "AUDIO", False),
        ("av_latent", "LATENT", False),
        ("chain_id", "STRING", False),
        ("segment_index", "INT", False),
        ("timeline_start_seconds", "FLOAT", False),
        ("save_context", "BOOLEAN", False),
        ("parent_candidate_id", "STRING", False),
        ("parent_manifest_revision", "INT", False),
        ("candidate_id", "STRING", True),
        ("model_id", "STRING", True),
        ("sampling_summary", "STRING", True),
        ("prompt", "STRING", False),
        ("seed", "INT", False),
        ("bit_depth", "COMBO", True),
        ("crf", "INT", True),
    ],
    "MiniMaxH3OutputTrimT8": [
        ("frames", "IMAGE", False),
        ("start_seconds", "FLOAT", True),
        ("duration_seconds", "FLOAT", True),
        ("fps", "FLOAT", True),
        ("audio", "AUDIO", False),
    ],
    "MiniMaxH3SigmaShift": [
        ("model", "MODEL", False),
        ("shift_video", "FLOAT", True),
        ("shift_audio", "FLOAT", True),
    ],
    "LoadVideo": [("file", "COMBO", True)],
    "LoadImage": [("image", "COMBO", True)],
    "MiniMaxH3FaceRefineParityPlanT8Advanced": [
        ("frames", "IMAGE", False),
        ("fps", "FLOAT", True),
        ("detector_mode", "COMBO", True),
        ("detector_model", "COMBO", True),
        ("detector_device", "COMBO", True),
        ("confidence", "FLOAT", True),
        ("manual_roi_x", "FLOAT", True),
        ("manual_roi_y", "FLOAT", True),
        ("manual_roi_width", "FLOAT", True),
        ("manual_roi_height", "FLOAT", True),
        ("scene_cut_threshold", "FLOAT", True),
        ("max_track_jump", "FLOAT", True),
        ("max_gap_frames", "INT", True),
        ("center_smooth_window", "INT", True),
        ("size_smooth_window", "INT", True),
        ("crop_factor", "FLOAT", True),
        ("canvas_mode", "COMBO", True),
        ("require_h3_grid", "BOOLEAN", True),
        ("analysis_chunk_frames", "INT", True),
    ],
    "MiniMaxH3FaceRefineParityLatentT8Advanced": [
        ("positive", "CONDITIONING", False),
        ("av_latent", "LATENT", False),
        ("crops", "IMAGE", False),
        ("video_vae", "VAE", False),
        ("face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN", False),
        ("audio_policy", "COMBO", True),
        ("allow_multi_shot_exp", "BOOLEAN", True),
    ],
    "MiniMaxH3FaceRefinePerFrameDenoiseT8Advanced": [
        ("av_latent", "LATENT", False),
        ("face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN", False),
        ("strength_small_face", "FLOAT", True),
        ("strength_large_face", "FLOAT", True),
        ("scale_mode", "COMBO", True),
        ("face_px_small", "FLOAT", True),
        ("face_px_large", "FLOAT", True),
        ("gamma", "FLOAT", True),
        ("smooth_frames", "INT", True),
        ("video_mask_mode", "COMBO", True),
        ("require_locked_audio", "BOOLEAN", True),
    ],
    "MiniMaxH3FaceRefineParityStitchT8Advanced": [
        ("base_frames", "IMAGE", False),
        ("refined_crops", "IMAGE", False),
        ("face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN", False),
        ("paste_region", "COMBO", True),
        ("mask_dilation", "INT", True),
        ("feather_source_px", "FLOAT", True),
        ("colour_match", "FLOAT", True),
        ("blend", "FLOAT", True),
        ("undetected_frames", "COMBO", True),
        ("max_face_mean_abs_delta", "FLOAT", True),
        ("processing_device", "COMBO", True),
    ],
    "MiniMaxH3FaceRefineManual512RelativeBaselineT8Advanced": [
        ("candidate_frames", "IMAGE", False),
        ("face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN", False),
        ("latent_report_json", "STRING", False),
        ("denoise_report_json", "STRING", False),
        ("stitch_report_json", "STRING", False),
        ("profile", "COMBO", True),
        ("minimum_crop_face_height_px", "FLOAT", True),
    ],
    "MiniMaxH3CreatorSynchronizedCompareT8Advanced": [
        ("frames_a", "IMAGE", False),
        ("frames_b", "IMAGE", False),
        ("label_a", "STRING", True),
        ("label_b", "STRING", True),
        ("seed_a", "INT", True),
        ("seed_b", "INT", True),
        ("winner", "COMBO", True),
        ("reviewer_notes", "STRING", True),
        ("require_equal_geometry", "BOOLEAN", True),
    ],
    "SaveVideo": [
        ("video", "VIDEO", False),
        ("filename_prefix", "STRING", True),
        ("format", "COMBO", True),
        ("codec", "COMFY_DYNAMICCOMBO_V3", False),
    ],
}


def _uuid(name: str) -> str:
    return str(uuid.uuid5(SUBGRAPH_NAMESPACE, name))


def _node(workflow: dict, node_id: int) -> dict:
    for item in workflow["nodes"]:
        if int(item["id"]) == int(node_id):
            return item
    raise ValueError(f"workflow node {node_id} was not found")


def _input_index(node: dict, input_name: str) -> int:
    for index, item in enumerate(node.get("inputs", [])):
        if item.get("name") == input_name:
            return index
    raise ValueError(f"node {node['id']} input {input_name!r} was not found")


def _layout_entry(node: dict, input_name: str) -> tuple[str, str, bool]:
    for entry in INPUT_LAYOUTS.get(str(node.get("type", "")), []):
        if entry[0] == input_name:
            return entry
    raise ValueError(
        f"node {node['id']} ({node.get('type')}) has no audited layout entry "
        f"for input {input_name!r}"
    )


def _ensure_input(
    node: dict,
    links: list[dict],
    input_name: str,
    *,
    input_type: str | None = None,
    widget: bool = False,
) -> int:
    for index, item in enumerate(node.get("inputs", [])):
        if item.get("name") == input_name:
            if widget:
                item["widget"] = {"name": input_name}
            return index

    layout = INPUT_LAYOUTS.get(str(node.get("type", "")), [])
    layout_name, layout_type, layout_widget = _layout_entry(node, input_name)
    order = {name: index for index, (name, _type, _widget) in enumerate(layout)}
    target_order = order[layout_name]
    inputs = node.setdefault("inputs", [])
    slot = len(inputs)
    for index, item in enumerate(inputs):
        existing_order = order.get(str(item.get("name", "")), len(order))
        if existing_order > target_order:
            slot = index
            break

    for link in links:
        if int(link["target_id"]) == int(node["id"]) and int(link["target_slot"]) >= slot:
            link["target_slot"] = int(link["target_slot"]) + 1

    item = {
        "name": input_name,
        "type": input_type or layout_type,
    }
    if widget or layout_widget:
        item["widget"] = {"name": input_name}
    item["link"] = None
    inputs.insert(slot, item)
    return slot


def _output_index(node: dict, output_name: str) -> int:
    for index, item in enumerate(node.get("outputs", [])):
        if item.get("name") == output_name:
            return index
    raise ValueError(f"node {node['id']} output {output_name!r} was not found")


def _object_links(workflow: dict) -> list[dict]:
    result = []
    for raw in workflow.get("links", []):
        if isinstance(raw, dict):
            result.append(deepcopy(raw))
            continue
        if not isinstance(raw, list) or len(raw) < 6:
            raise ValueError(f"unsupported workflow link: {raw!r}")
        result.append(
            {
                "id": int(raw[0]),
                "origin_id": int(raw[1]),
                "origin_slot": int(raw[2]),
                "target_id": int(raw[3]),
                "target_slot": int(raw[4]),
                "type": raw[5],
            }
        )
    return result


def _remove_link(workflow: dict, links: list[dict], link_id: int) -> None:
    matched = [item for item in links if int(item["id"]) == int(link_id)]
    if len(matched) != 1:
        raise ValueError(f"expected exactly one workflow link {link_id}, found {len(matched)}")
    link = matched[0]
    links.remove(link)
    origin = _node(workflow, int(link["origin_id"]))
    output = origin["outputs"][int(link["origin_slot"])]
    output["links"] = [
        item for item in (output.get("links") or []) if int(item) != int(link_id)
    ]
    target = _node(workflow, int(link["target_id"]))
    target["inputs"][int(link["target_slot"])]["link"] = None


def _widget_index(node: dict, input_name: str) -> int:
    layout = INPUT_LAYOUTS.get(str(node.get("type", "")), [])
    cursor = 0
    for name, _input_type, widget in layout:
        if not widget:
            continue
        if name == input_name:
            return cursor
        cursor += 2 if name in {"seed", "noise_seed"} else 1
    raise ValueError(
        f"node {node['id']} ({node.get('type')}) input {input_name!r} is not "
        "an audited widget"
    )


def _set_widget(workflow: dict, node_id: int, input_name: str, value) -> None:
    node = _node(workflow, node_id)
    index = _widget_index(node, input_name)
    values = node.get("widgets_values")
    if isinstance(values, list):
        values[index] = value
    elif index == 0:
        node["widgets_values"] = value
    else:
        raise ValueError(f"node {node_id} has a scalar widgets_values payload")


def _add_optional_input(node: dict, name: str, input_type: str) -> None:
    if any(item.get("name") == name for item in node.get("inputs", [])):
        return
    node.setdefault("inputs", []).append({"name": name, "type": input_type, "link": None})


def _restore_proxy_source_inputs(workflow: dict, links: list[dict]) -> None:
    """Rehydrate schema widget inputs required inside a subgraph definition.

    Top-level native workflows correctly omit unlinked widgets.  A subgraph
    definition is different: proxy widgets link to inner widget sockets, so its
    inner nodes must carry those descriptors.  Rehydrating from the audited
    schema order preserves the long-standing subgraph interface without putting
    obsolete unlinked widgets back into the user-facing source workflows.
    """

    for node in workflow.get("nodes", []):
        layout = INPUT_LAYOUTS.get(str(node.get("type", "")), [])
        widget_values = node.get("widgets_values")
        for name, input_type, widget in layout:
            if (
                widget
                and isinstance(widget_values, list)
                and _widget_index(node, name) < len(widget_values)
            ):
                _ensure_input(
                    node,
                    links,
                    name,
                    input_type=input_type,
                    widget=True,
                )


def build_subgraph(spec: dict) -> dict:
    source = PROJECT_ROOT / spec["source"]
    workflow = json.loads(source.read_text(encoding="utf-8"))
    workflow = deepcopy(workflow)
    links = _object_links(workflow)
    _restore_proxy_source_inputs(workflow, links)
    for node_id, input_name, value in spec.get("widget_overrides", []):
        _set_widget(workflow, node_id, input_name, value)
    for node_id, input_name, input_type in spec.get("optional_inputs", []):
        _add_optional_input(_node(workflow, node_id), input_name, input_type)

    next_link = max([int(item["id"]) for item in links] + [0]) + 1
    definition_inputs = []
    top_inputs = []
    proxy_widgets = []
    for slot, entry in enumerate(spec.get("inputs", [])):
        node = _node(workflow, entry["node"])
        if entry.get("widget", False):
            layout_type = _layout_entry(node, entry["input"])[1]
            target_slot = _ensure_input(
                node,
                links,
                entry["input"],
                input_type=entry.get("type", layout_type),
                widget=True,
            )
        else:
            target_slot = _input_index(node, entry["input"])
        target = node["inputs"][target_slot]
        old_link = target.get("link")
        if old_link is not None:
            _remove_link(workflow, links, int(old_link))
        link_id = next_link
        next_link += 1
        target["link"] = link_id
        input_type = entry.get("type", target.get("type", "*"))
        public_name = entry["name"]
        item = {
            "id": _uuid(f"{spec['id']}:input:{public_name}"),
            "name": public_name,
            "type": input_type,
            "linkIds": [link_id],
            "label": entry.get("label", public_name),
            "pos": [20, 40 + slot * 20],
        }
        definition_inputs.append(item)
        top_input = {
            "name": public_name,
            "label": entry.get("label", public_name),
            "type": input_type,
            "link": None,
        }
        if entry.get("widget", False):
            top_input["widget"] = {"name": public_name}
            proxy_widgets.append([str(entry["node"]), entry["input"]])
        top_inputs.append(top_input)
        links.append(
            {
                "id": link_id,
                "origin_id": -10,
                "origin_slot": slot,
                "target_id": int(entry["node"]),
                "target_slot": target_slot,
                "type": input_type,
            }
        )

    definition_outputs = []
    top_outputs = []
    for slot, entry in enumerate(spec.get("outputs", [])):
        node = _node(workflow, entry["node"])
        origin_slot = _output_index(node, entry["output"])
        output = node["outputs"][origin_slot]
        link_id = next_link
        next_link += 1
        output.setdefault("links", [])
        if output["links"] is None:
            output["links"] = []
        output["links"].append(link_id)
        output_type = entry.get("type", output.get("type", "*"))
        public_name = entry["name"]
        definition_outputs.append(
            {
                "id": _uuid(f"{spec['id']}:output:{public_name}"),
                "name": public_name,
                "type": output_type,
                "linkIds": [link_id],
                "localized_name": entry.get("label", public_name),
                "pos": [3900, 40 + slot * 20],
            }
        )
        top_outputs.append(
            {
                "name": public_name,
                "localized_name": entry.get("label", public_name),
                "type": output_type,
                "links": [],
            }
        )
        links.append(
            {
                "id": link_id,
                "origin_id": int(entry["node"]),
                "origin_slot": origin_slot,
                "target_id": -20,
                "target_slot": slot,
                "type": output_type,
            }
        )

    graph_id = _uuid(f"subgraph:{spec['id']}")
    max_node = max(int(item["id"]) for item in workflow["nodes"])
    groups = deepcopy(workflow.get("groups", []))
    definition = {
        "id": graph_id,
        "version": 1,
        "state": {
            "lastGroupId": max([int(item.get("id", 0)) for item in groups] + [0]),
            "lastNodeId": max_node,
            "lastLinkId": next_link - 1,
            "lastRerouteId": 0,
        },
        "revision": 0,
        "config": {},
        "name": spec["name"],
        "inputNode": {"id": -10, "bounding": [0, 0, 120, max(80, 40 + len(top_inputs) * 20)]},
        "outputNode": {"id": -20, "bounding": [3900, 0, 120, max(60, 40 + len(top_outputs) * 20)]},
        "inputs": definition_inputs,
        "outputs": definition_outputs,
        "widgets": [],
        "nodes": workflow["nodes"],
        "groups": groups,
        "links": links,
        "extra": deepcopy(workflow.get("extra", {})),
        "category": "MiniMax H3 T8/Quick Start",
        "description": spec["description"],
    }
    top_node = {
        "id": 1,
        "type": graph_id,
        "pos": [100, 100],
        "size": [560, 0],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": top_inputs,
        "outputs": top_outputs,
        "title": spec["name"],
        "properties": {
            "proxyWidgets": proxy_widgets,
            "cnr_id": "minimax-h3-audio-t8",
            "ver": spec.get("version", "1.43.0"),
            "ue_properties": {
                "widget_ue_connectable": {},
                "input_ue_unconnectable": {},
                "version": "7.7",
            },
        },
        "widgets_values": [],
    }
    return {
        "revision": 0,
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [top_node],
        "links": [],
        "version": 0.4,
        "definitions": {"subgraphs": [definition]},
        "extra": {"ue_links": []},
    }


COMMON_MODEL_INPUTS = [
    {"node": 1, "input": "unet_name", "name": "model", "label": "H3模型", "widget": True},
    {"node": 2, "input": "lora_name", "name": "turbo_lora", "label": "Turbo LoRA", "widget": True},
    {"node": 2, "input": "strength_model", "name": "lora_strength", "label": "LoRA强度", "widget": True},
    {"node": 3, "input": "clip_name", "name": "text_encoder", "label": "H3文本编码器", "widget": True},
    {"node": 4, "input": "vae_name", "name": "video_vae", "label": "视频VAE", "widget": True},
    {"node": 5, "input": "vae_name", "name": "audio_vae", "label": "音频VAE", "widget": True},
]


def _basic_inputs(extra: list[dict] | None = None) -> list[dict]:
    values = deepcopy(COMMON_MODEL_INPUTS)
    values.extend(
        [
            {"node": 6, "input": "prompt", "name": "prompt", "label": "提示词", "widget": True},
            {"node": 6, "input": "width", "name": "width", "label": "宽度", "widget": True},
            {"node": 6, "input": "height", "name": "height", "label": "高度", "widget": True},
            {"node": 6, "input": "length", "name": "length", "label": "帧数", "widget": True},
            {"node": 9, "input": "noise_seed", "name": "seed", "label": "随机种子", "widget": True},
        ]
    )
    values.extend(extra or [])
    return values


SPECS = [
    {
        "id": "quick_t2va",
        "filename": "2026-08-22_H3_Quick_T2VA.json",
        "name": "MiniMax H3 Quick T2VA / 快速文生音视频",
        "source": "examples/workflows/01-basic-generation/2026-08-06_H3_Turbo_Stable_4V4A.json",
        "description": "Stable T2VA quick-start that reuses the original conditioning, dual-clock sampler and AV decoder without changing their schemas.",
        "inputs": _basic_inputs(),
        "outputs": [
            {"node": 11, "output": "frames", "name": "frames", "label": "生成画面"},
            {"node": 11, "output": "generated_audio", "name": "audio", "label": "生成音频"},
            {"node": 6, "output": "report", "name": "report", "label": "条件报告"},
        ],
    },
    {
        "id": "quick_i2va_fl2va",
        "filename": "2026-08-22_H3_Quick_I2VA_FL2VA.json",
        "name": "MiniMax H3 Quick I2VA-FL2VA / 快速首尾帧",
        "source": "examples/workflows/01-basic-generation/2026-08-06_H3_Turbo_Stable_4V4A.json",
        "description": "One compatible quick-start for first-frame and first-last-frame generation; connect first_frame and optionally last_frame.",
        "widget_overrides": [(6, "task_type", "auto")],
        "inputs": _basic_inputs(
            [
                {"node": 6, "input": "first_frame", "name": "first_frame", "label": "首帧", "type": "IMAGE"},
                {"node": 6, "input": "last_frame", "name": "last_frame", "label": "尾帧（可选）", "type": "IMAGE"},
            ]
        ),
        "outputs": [
            {"node": 11, "output": "frames", "name": "frames", "label": "生成画面"},
            {"node": 11, "output": "generated_audio", "name": "audio", "label": "生成音频"},
            {"node": 6, "output": "report", "name": "report", "label": "条件报告"},
        ],
    },
    {
        "id": "quick_ref2va",
        "filename": "2026-08-22_H3_Quick_Ref2VA.json",
        "name": "MiniMax H3 Quick Ref2VA / 快速参考生音视频",
        "source": "examples/workflows/01-basic-generation/2026-08-06_H3_Turbo_Stable_4V4A.json",
        "description": "Reference-image quick-start with the stable H3 conditioning and joint AV path. Use a task-compatible model/LoRA combination.",
        "widget_overrides": [(6, "task_type", "Ref2VA")],
        "optional_inputs": [(6, "ref_images.ref_image_0", "IMAGE")],
        "inputs": _basic_inputs(
            [
                {"node": 6, "input": "ref_images.ref_image_0", "name": "reference_image", "label": "参考图", "type": "IMAGE"},
            ]
        ),
        "outputs": [
            {"node": 11, "output": "frames", "name": "frames", "label": "生成画面"},
            {"node": 11, "output": "generated_audio", "name": "audio", "label": "生成音频"},
            {"node": 6, "output": "report", "name": "report", "label": "条件报告"},
        ],
    },
    {
        "id": "quick_audio_drive",
        "filename": "2026-08-22_H3_Quick_Audio_Drive.json",
        "name": "MiniMax H3 Quick Audio Drive / 快速原音轨驱动",
        "source": "examples/workflows/02-audio-control/2026-08-06_H3_Audio_Lock_Source_Stable_4V4A.json",
        "description": "Audio-drive quick-start that locks the aligned source latent and uses mux_audio for the final soundtrack.",
        "inputs": _basic_inputs(
            [
                {"node": 14, "input": "audio", "name": "source_audio", "label": "输入音频", "type": "AUDIO"},
                {"node": 14, "input": "scene_duration_seconds", "name": "duration", "label": "时长（秒）", "widget": True},
            ]
        ),
        "outputs": [
            {"node": 15, "output": "frames", "name": "frames", "label": "生成画面"},
            {"node": 15, "output": "audio", "name": "audio", "label": "原音轨"},
            {"node": 15, "output": "report_json", "name": "report", "label": "裁切报告"},
        ],
    },
    {
        "id": "quick_long_video",
        "filename": "2026-08-22_H3_Quick_Long_Video.json",
        "name": "MiniMax H3 Quick Long Video / 快速长视频",
        "source": "examples/workflows/04-long-video/2026-08-09_H3_Long_Video_Auto_Resume_22F_EXP.json",
        "description": "Review-first long-video quick-start. It keeps candidate acceptance false by default and preserves the existing resumable manifest contract.",
        "inputs": [
            *deepcopy(COMMON_MODEL_INPUTS),
            {"node": 6, "input": "chain_id", "name": "chain_id", "label": "任务ID", "widget": True},
            {"node": 6, "input": "total_duration_seconds", "name": "duration", "label": "总时长（秒）", "widget": True},
            {"node": 6, "input": "global_prompt", "name": "prompt", "label": "全局提示词", "widget": True},
            {"node": 6, "input": "base_seed", "name": "seed", "label": "随机种子", "widget": True},
            {"node": 8, "input": "width", "name": "width", "label": "宽度", "widget": True},
            {"node": 8, "input": "height", "name": "height", "label": "高度", "widget": True},
            {"node": 17, "input": "accept_candidate", "name": "accept_after_review", "label": "审片后接受", "widget": True},
        ],
        "outputs": [
            {"node": 13, "output": "candidate_json_path", "name": "candidate_json", "label": "候选记录"},
            {"node": 13, "output": "candidate_video_path", "name": "candidate_video", "label": "候选视频"},
            {"node": 17, "output": "report_json", "name": "report", "label": "接受报告"},
        ],
    },
    {
        "id": "quick_repair",
        "filename": "2026-08-22_H3_Quick_Face_Repair.json",
        "name": "MiniMax H3 Quick Repair / 快速单人五官修复",
        "source": "examples/workflows/06-face-refine/2026-08-09_H3_Face_Refine_Parity_Advanced_EXP.json",
        "description": "Human-reviewed MANUAL512 relative-to-clip single-person repair candidate; it preserves the original soundtrack and is not a deblur/upscale tool.",
        "inputs": [
            {"node": 1, "input": "file", "name": "source_video", "label": "源视频", "widget": True},
            {"node": 3, "input": "image", "name": "identity_image_1", "label": "清晰参考图1", "widget": True},
            {"node": 24, "input": "image", "name": "identity_image_2", "label": "清晰参考图2", "widget": True},
            {"node": 8, "input": "unet_name", "name": "model", "label": "Ref2VA模型", "widget": True},
            {"node": 9, "input": "lora_name", "name": "turbo_lora", "label": "修复LoRA", "widget": True},
            {"node": 7, "input": "clip_name", "name": "text_encoder", "label": "H3文本编码器", "widget": True},
            {"node": 5, "input": "vae_name", "name": "video_vae", "label": "视频VAE", "widget": True},
            {"node": 6, "input": "vae_name", "name": "audio_vae", "label": "音频VAE", "widget": True},
            {"node": 11, "input": "prompt", "name": "prompt", "label": "原视频提示词", "widget": True},
            {"node": 4, "input": "detector_model", "name": "face_detector", "label": "人脸检测模型", "widget": True},
            {"node": 14, "input": "noise_seed", "name": "seed", "label": "随机种子", "widget": True},
        ],
        "outputs": [
            {"node": 23, "output": "candidate_frames", "name": "frames", "label": "修复候选画面"},
            {"node": 21, "output": "VIDEO", "name": "video", "label": "带原音轨视频"},
            {"node": 23, "output": "baseline_report_json", "name": "report", "label": "修复报告"},
        ],
    },
    {
        "id": "quick_creator_av_review",
        "filename": "2026-08-23_H3_Quick_Creator_AV_Review.json",
        "name": "MiniMax H3 Quick Creator AV Review / 快速音画审片",
        "version": "1.45.0",
        "source": (
            "examples/workflows/11-studio-production/"
            "2026-08-22_H3_Creator_Synchronized_AV_AB_Advanced.json"
        ),
        "description": (
            "Human-reviewed synchronized A/B workspace for two aligned videos: it preserves "
            "source pixels, keeps A/B audio separate, saves a silent comparison video and "
            "reports reference-relative audio drift without choosing a winner automatically."
        ),
        "inputs": [
            {
                "node": 1,
                "input": "file",
                "name": "baseline_video",
                "label": "A基准视频",
                "widget": True,
            },
            {
                "node": 3,
                "input": "file",
                "name": "candidate_video",
                "label": "B候选视频",
                "widget": True,
            },
            {
                "node": 5,
                "input": "label_a",
                "name": "label_a",
                "label": "A标签",
                "widget": True,
            },
            {
                "node": 5,
                "input": "label_b",
                "name": "label_b",
                "label": "B标签",
                "widget": True,
            },
            {
                "node": 5,
                "input": "seed_a",
                "name": "seed_a",
                "label": "A种子",
                "widget": True,
            },
            {
                "node": 5,
                "input": "seed_b",
                "name": "seed_b",
                "label": "B种子",
                "widget": True,
            },
            {
                "node": 5,
                "input": "winner",
                "name": "winner_after_review",
                "label": "人工结论",
                "widget": True,
            },
            {
                "node": 5,
                "input": "reviewer_notes",
                "name": "reviewer_notes",
                "label": "审片备注",
                "widget": True,
            },
            {
                "node": 5,
                "input": "require_equal_geometry",
                "name": "require_equal_geometry",
                "label": "要求相同画布",
                "widget": True,
            },
            {
                "node": 7,
                "input": "filename_prefix",
                "name": "output_prefix",
                "label": "无声对比输出前缀",
                "widget": True,
            },
        ],
        "outputs": [
            {
                "node": 5,
                "output": "comparison_frames",
                "name": "comparison_frames",
                "label": "并排对比画面",
            },
            {"node": 2, "output": "audio", "name": "audio_a", "label": "A独立音轨"},
            {"node": 4, "output": "audio", "name": "audio_b", "label": "B独立音轨"},
            {
                "node": 7,
                "output": "video",
                "name": "silent_comparison_video",
                "label": "无声对比视频",
            },
            {"node": 5, "output": "winner", "name": "winner", "label": "人工结论"},
            {
                "node": 5,
                "output": "selected_seed",
                "name": "selected_seed",
                "label": "选中种子",
            },
            {
                "node": 5,
                "output": "review_json",
                "name": "visual_review_json",
                "label": "画面审片报告",
            },
            {
                "node": 10,
                "output": "decision",
                "name": "audio_drift_decision",
                "label": "声音漂移结论",
            },
            {
                "node": 10,
                "output": "report_json",
                "name": "audio_drift_report",
                "label": "声音漂移报告",
            },
        ],
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build native ComfyUI H3 T8 quick-start subgraphs")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "subgraphs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        payload = build_subgraph(spec)
        target = args.output / spec["filename"]
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
