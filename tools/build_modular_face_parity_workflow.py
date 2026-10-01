"""Private Parity/Per-Frame/Sampler-Mask Face stage candidate.

The old sample names an unavailable MiniMaxH3SigmaShift. This copy uses the
current native dual-clock MODEL setup for er_sde/simple while retaining the
old standalone KSamplerSelect, BasicScheduler and optional mask patch. Exact
equivalence to that missing historical node is not asserted.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402
from tools.build_modular_face_workflow import ROOT, WIDGETS as STANDARD_WIDGETS  # noqa: E402

SOURCE = ROOT / "examples/workflows/06-face-refine/2026-08-09_H3_Face_Refine_Parity_Advanced_EXP.json"
TARGET = ROOT / "artifacts/development/modular-sampling-m4-face-parity-20260923/candidate-v1"

WIDGETS = {**STANDARD_WIDGETS,
    "LoadImage": ("image",), "LoraLoaderModelOnly": ("lora_name", "strength_model"),
    "MiniMaxH3DualClockSamplerT8": ("steps", "shift_video", "shift_audio", "sampler_name", "scheduler"),
    "MiniMaxH3FaceRefineParityPlanT8Advanced": (
        "fps", "detector_mode", "detector_model", "detector_device", "confidence",
        "manual_roi_x", "manual_roi_y", "manual_roi_width", "manual_roi_height",
        "scene_cut_threshold", "max_track_jump", "max_gap_frames", "center_smooth_window",
        "size_smooth_window", "crop_factor", "canvas_mode", "require_h3_grid", "analysis_chunk_frames"),
    "MiniMaxH3FaceRefineParityLatentT8Advanced": ("audio_policy", "allow_multi_shot_exp"),
    "MiniMaxH3FaceRefinePerFrameDenoiseT8Advanced": (
        "strength_small_face", "strength_large_face", "scale_mode", "face_px_small",
        "face_px_large", "gamma", "smooth_frames", "video_mask_mode", "require_locked_audio"),
    "MiniMaxH3FaceRefineSamplerMaskPatchV11T8Advanced": ("enabled",),
    "KSamplerSelect": ("sampler_name",), "BasicScheduler": ("scheduler", "steps", "denoise"),
    "MiniMaxH3FaceRefineParityStitchT8Advanced": (
        "paste_region", "mask_dilation", "feather_source_px", "colour_match", "blend",
        "undetected_frames", "max_face_mean_abs_delta", "processing_device"),
    "MiniMaxH3FaceRefineManual512RelativeBaselineT8Advanced": (
        "profile", "minimum_crop_face_height_px"),
    "MiniMaxH3FaceParityStageBindEXPT8": ("audio_policy",),
}


def split_frontend(source):
    draft = Draft(source)
    nodes = draft.nodes
    shift = nodes[10]
    if shift["type"] != "MiniMaxH3SigmaShift":
        raise ValueError("Expected the historical parity shift placeholder")
    shift["type"] = "MiniMaxH3DualClockSamplerT8"
    shift["title"] = "Current native FLOW_AV model shift; er_sde/simple remain external"
    shift["properties"]["Node name for S&R"] = shift["type"]
    shift["inputs"].append({"name": "av_latent", "type": "LATENT", "link": None})
    shift["outputs"].extend([
        {"name": "sampler", "type": "SAMPLER", "links": []},
        {"name": "sigmas", "type": "SIGMAS", "links": []},
        {"name": "report_json", "type": "STRING", "links": []}])
    shift["widgets_values"] = [8, 12., 3., "er_sde", "simple"]
    draft.connect((12, 1), shift, "av_latent", "LATENT")
    bind = draft.make("MiniMaxH3FaceParityStageBindEXPT8", "Bind Parity plan and actual er_sde stage",
        [("face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN"), ("source_frames", "IMAGE"),
         ("model", "MODEL"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
         ("av_latent", "LATENT")],
        [("model", "MODEL"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
         ("stage_context", "T8_STAGE_CONTEXT"), ("report_json", "STRING")],
        ["require_locked"], (3180, 530))
    audit = draft.make("MiniMaxH3FaceParityStageAuditEXPT8", "Audit Parity source and sampled candidate",
        [("stage_result", "T8_STAGE_RESULT"), ("face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN"),
         ("source_frames", "IMAGE"), ("av_latent", "LATENT")],
        [("candidate_av", "LATENT"), ("report_json", "STRING")], [], (4050, 450))
    stage = nodes[18]
    stage["type"] = "MiniMaxH3StageSamplerEXPT8"
    stage["title"] = "ONE editable Parity er_sde stage; completion unverified"
    stage["properties"]["Node name for S&R"] = stage["type"]
    stage["properties"]["cnr_id"] = "minimax-h3-audio-T8"
    stage["inputs"].append({"name": "stage_context", "type": "T8_STAGE_CONTEXT", "link": None})
    stage["outputs"].extend([
        {"name": "stage_result", "type": "T8_STAGE_RESULT", "links": []},
        {"name": "report_json", "type": "STRING", "links": []}])
    for target, name in ((nodes[15], "model"), (stage, "sampler"),
                         (stage, "sigmas"), (nodes[19], "av_latent")):
        draft.disconnect(target, name)
    for source_slot, name, dtype in [((4, 0), "face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN"),
                                     ((2, 0), "source_frames", "IMAGE"),
                                     ((32, 0), "model", "MODEL"),
                                     ((16, 0), "sampler", "SAMPLER"),
                                     ((17, 0), "sigmas", "SIGMAS"),
                                     ((32, 1), "av_latent", "LATENT")]:
        draft.connect(source_slot, bind, name, dtype)
    for source_slot, target, name, dtype in [((bind["id"], 0), nodes[15], "model", "MODEL"),
                                            ((bind["id"], 1), stage, "sampler", "SAMPLER"),
                                            ((bind["id"], 2), stage, "sigmas", "SIGMAS"),
                                            ((bind["id"], 3), stage, "stage_context", "T8_STAGE_CONTEXT")]:
        draft.connect(source_slot, target, name, dtype)
    for source_slot, name, dtype in [((18, 2), "stage_result", "T8_STAGE_RESULT"),
                                     ((4, 0), "face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN"),
                                     ((2, 0), "source_frames", "IMAGE"),
                                     ((32, 1), "av_latent", "LATENT")]:
        draft.connect(source_slot, audit, name, dtype)
    draft.connect((audit["id"], 0), nodes[19], "av_latent", "LATENT")
    graph = draft.graph
    graph["last_node_id"] = max(draft.nodes)
    graph["last_link_id"] = max(draft.links)
    seen = set()
    ordered = []
    while len(ordered) < len(draft.nodes):
        ready = [node for node in graph["nodes"] if node["id"] not in seen and
                 all(item.get("link") is None or draft.links[item["link"]][1] in seen
                     for item in node.get("inputs", []))]
        if not ready:
            raise ValueError("Face Parity candidate has a dependency cycle")
        for node in ready:
            seen.add(node["id"])
            ordered.append(node)
    for order, node in enumerate(ordered):
        node["order"] = order
    return graph


def split_api(frontend):
    graph = {}
    nodes = {node["id"]: node for node in frontend["nodes"]}
    for node in frontend["nodes"]:
        if node["type"] == "MarkdownNote":
            continue
        names = WIDGETS.get(node["type"], ())
        values = node.get("widgets_values", [])
        if len(values) < len(names):
            raise ValueError(f"Missing widgets for {node['type']}")
        graph[str(node["id"])] = {"class_type": node["type"],
                                  "inputs": dict(zip(names, values[:len(names)], strict=True))}
    for _, source, source_slot, target, target_slot, _ in frontend["links"]:
        if str(target) in graph:
            name = nodes[target]["inputs"][target_slot]["name"]
            graph[str(target)]["inputs"][name] = [str(source), source_slot]
    return graph


def main():
    graph = split_frontend(deepcopy(json.loads(SOURCE.read_text(encoding="utf-8"))))
    api = split_api(graph)
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "Face_Parity_Separate_Stage_EXP.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (TARGET / "Face_Parity_Separate_Stage_EXP.api.json").write_text(
        json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"nodes": len(graph["nodes"]), "links": len(graph["links"]),
                      "stage_samplers": sum(n["type"] == "MiniMaxH3StageSamplerEXPT8" for n in graph["nodes"])}))


if __name__ == "__main__":
    main()
