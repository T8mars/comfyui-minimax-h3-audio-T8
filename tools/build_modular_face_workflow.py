"""Generate a private source-audited standard Face Refine stage candidate.

The legacy example is read-only. The copied graph keeps its source audio on
CreateVideo and inserts Bind -> Stage Sampler -> Audit before crop decoding.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402

SOURCE = ROOT / "examples/workflows/06-face-refine/2026-08-16_H3_Face_Refine_Advanced_EXP.json"
TARGET = ROOT / "artifacts/development/modular-sampling-m4-face-standard-20260923/candidate-v1"

WIDGETS = {
    "LoadVideo": ("file",), "VAELoader": ("vae_name",),
    "CLIPLoader": ("clip_name", "type", "device"),
    "UNETLoader": ("unet_name", "weight_dtype"),
    "MiniMaxH3FaceRefinePlanT8Advanced": (
        "fps", "detector_mode", "detector_model", "detector_device", "confidence",
        "manual_roi_x", "manual_roi_y", "manual_roi_width", "manual_roi_height",
        "scene_cut_threshold", "max_track_jump", "max_gap_frames", "smoothing_radius",
        "crop_context_scale", "canvas_size", "require_h3_grid", "analysis_chunk_frames"),
    "MiniMaxH3AudioConditioningT8": (
        "prompt", "width", "height", "length", "task_type", "audio_mode",
        "audio_denoise_strength", "add_source_as_reference", "prompt_primary_audio_ordinal",
        "strict_prompt_tags", "ref_image_size", "reference_video_policy"),
    "MiniMaxH3FaceRefineConditioningT8Advanced": ("audio_policy", "allow_multi_shot_exp"),
    "MiniMaxH3FaceRefineSamplerT8Advanced": (
        "steps", "denoise", "shift_video", "shift_audio", "sampler_name", "scheduler"),
    "RandomNoise": ("noise_seed",),
    "MiniMaxH3FaceRefineStitchAuditT8Advanced": (
        "paste_region", "feather_source_px", "blend_strength", "color_match_strength",
        "max_face_mean_abs_delta", "fallback_neighbor_frames", "processing_device"),
    "CreateVideo": ("fps", "bit_depth"),
    "SaveVideo": ("filename_prefix", "format", "codec"),
    "MiniMaxH3FaceStageBindEXPT8": ("audio_policy",),
}


def split_frontend(source):
    draft = Draft(source)
    nodes = draft.nodes
    bind = draft.make("MiniMaxH3FaceStageBindEXPT8", "Bind source + one separate Face stage",
        [("face_plan", "H3_T8_FACE_REFINE_PLAN"), ("source_frames", "IMAGE"),
         ("model", "MODEL"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
         ("av_latent", "LATENT")],
        [("model", "MODEL"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
         ("stage_context", "T8_STAGE_CONTEXT"), ("report_json", "STRING")],
        ["require_locked"], (2910, 460))
    audit = draft.make("MiniMaxH3FaceStageAuditEXPT8", "Audit source before candidate crop decode",
        [("stage_result", "T8_STAGE_RESULT"), ("face_plan", "H3_T8_FACE_REFINE_PLAN"),
         ("source_frames", "IMAGE"), ("av_latent", "LATENT")],
        [("candidate_av", "LATENT"), ("report_json", "STRING")], [], (3800, 450))
    sampler = nodes[13]
    sampler["type"] = "MiniMaxH3StageSamplerEXPT8"
    sampler["title"] = "ONE source-bound Face refinement stage"
    sampler["properties"]["Node name for S&R"] = sampler["type"]
    sampler["properties"]["cnr_id"] = "minimax-h3-audio-T8"
    sampler["inputs"].append({"name": "stage_context", "type": "T8_STAGE_CONTEXT", "link": None})
    sampler["outputs"].extend([
        {"name": "stage_result", "type": "T8_STAGE_RESULT", "links": []},
        {"name": "report_json", "type": "STRING", "links": []}])
    nodes[8]["title"] = "Face crop latent; input audio mask locked (candidate audio not delivered)"
    draft.disconnect(nodes[12], "model")
    draft.disconnect(sampler, "sampler")
    draft.disconnect(sampler, "sigmas")
    draft.disconnect(nodes[14], "av_latent")
    for source_slot, target_name, dtype in [((3, 0), "face_plan", "H3_T8_FACE_REFINE_PLAN"),
                                            ((2, 0), "source_frames", "IMAGE"),
                                            ((10, 0), "model", "MODEL"),
                                            ((10, 1), "sampler", "SAMPLER"),
                                            ((10, 2), "sigmas", "SIGMAS"),
                                            ((8, 1), "av_latent", "LATENT")]:
        draft.connect(source_slot, bind, target_name, dtype)
    draft.connect((bind["id"], 0), nodes[12], "model", "MODEL")
    draft.connect((bind["id"], 1), sampler, "sampler", "SAMPLER")
    draft.connect((bind["id"], 2), sampler, "sigmas", "SIGMAS")
    draft.connect((bind["id"], 3), sampler, "stage_context", "T8_STAGE_CONTEXT")
    for source_slot, target_name, dtype in [((13, 2), "stage_result", "T8_STAGE_RESULT"),
                                            ((3, 0), "face_plan", "H3_T8_FACE_REFINE_PLAN"),
                                            ((2, 0), "source_frames", "IMAGE"),
                                            ((8, 1), "av_latent", "LATENT")]:
        draft.connect(source_slot, audit, target_name, dtype)
    draft.connect((audit["id"], 0), nodes[14], "av_latent", "LATENT")
    graph = draft.graph
    graph["last_node_id"] = max(draft.nodes)
    graph["last_link_id"] = max(draft.links)
    # LiteGraph order is a display hint; keep it consistent with dependencies.
    seen = set()
    ordered = []
    while len(ordered) < len(draft.nodes):
        ready = [node for node in graph["nodes"] if node["id"] not in seen and
                 all(item.get("link") is None or draft.links[item["link"]][1] in seen
                     for item in node.get("inputs", []))]
        if not ready:
            raise ValueError("Face candidate has a dependency cycle")
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
        names = WIDGETS.get(node["type"], ())
        values = node.get("widgets_values", [])
        if len(values) < len(names):
            raise ValueError(f"Missing widgets for {node['type']}")
        graph[str(node["id"])] = {"class_type": node["type"],
                                  "inputs": dict(zip(names, values[:len(names)], strict=True))}
    for _, source, source_slot, target, target_slot, _ in frontend["links"]:
        name = nodes[target]["inputs"][target_slot]["name"]
        graph[str(target)]["inputs"][name] = [str(source), source_slot]
    return graph


def main():
    if TARGET.exists():
        raise FileExistsError("Private Face candidate already exists")
    graph = split_frontend(deepcopy(json.loads(SOURCE.read_text(encoding="utf-8"))))
    api = split_api(graph)
    TARGET.mkdir(parents=True)
    (TARGET / "Face_Standard_Separate_Stage_EXP.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (TARGET / "Face_Standard_Separate_Stage_EXP.api.json").write_text(
        json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"nodes": len(graph["nodes"]), "links": len(graph["links"]),
                      "stage_samplers": sum(n["type"] == "MiniMaxH3StageSamplerEXPT8" for n in graph["nodes"])}))


if __name__ == "__main__":
    main()
