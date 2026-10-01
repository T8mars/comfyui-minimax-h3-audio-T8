"""Build private Motion first-pass freeze and cold pass-2 graph pairs.

The source Relay/EAV candidate is SHA-pinned. Old/accepted workflows are never
edited. The cold graph visibly has no first-pass sampler, and its Load demands
an independently retained exact manifest and whole-file hash before pass 2.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from tools.build_modular_h16_storage_workflow import Draft
from tools.build_modular_motion_effect_workflows import WIDGETS
from tools.build_modular_motion_recovery_workflows import ROOT, split_api

SOURCE = ROOT / "artifacts/development/modular-sampling-m4-motion-effects-20260925/candidate-v2"
TARGET = ROOT / "artifacts/development/modular-sampling-m4-motion-storage-20260925/candidate-v1"
SOURCE_SHA = {
    "Fullclip": "bc923bdcdf1aaa0fb0a878f76f67d8e197b71a8eec8eac322af1207c237fa122",
    "Windowed": "7ad57ab1d8b427d1481f531084d38abe05466eb75f049a40e24d59a8cde09e7c",
}
CHECKPOINT_ID = "motion_recovery_firstpass"
SAVE = "MiniMaxH3NativeLatentCheckpointSaveT8Advanced"
PASS_THROUGH_SAVE = "MiniMaxH3NativeLatentCheckpointPassThroughSaveEXPT8"
LOAD = "MiniMaxH3MotionFrozenFirstPassLoadEXPT8"
WIDGETS.update({
    SAVE: ("filename_prefix", "checkpoint_id", "confirm_save", "verify_after_write",
           "hash_chunk_megabytes"),
    PASS_THROUGH_SAVE: ("filename_prefix", "checkpoint_id", "confirm_save",
                        "verify_after_write", "hash_chunk_megabytes"),
    LOAD: ("checkpoint_path", "expected_manifest_json", "expected_file_sha256",
           "expected_frame_count", "expected_width", "expected_height", "hash_chunk_megabytes"),
})


def source(variant: str) -> dict:
    if variant not in SOURCE_SHA:
        raise ValueError("Unknown Motion Recovery variant")
    path = SOURCE / f"Motion_Recovery_{variant}_Relay_EAV_Separate_Pass2_EXP.json"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA[variant]:
        raise ValueError("Pinned Motion Relay/EAV frontend changed")
    return json.loads(raw)


def _api(frontend: dict, variant: str) -> dict:
    graph = deepcopy(frontend)
    if variant == "Windowed":
        segment = next((item for item in graph["nodes"] if item["id"] == 13), None)
        if segment is not None:
            if segment["widgets_values"] != [209, 0, "fixed", 12, "hot_ranges_only"]:
                raise ValueError("Windowed seed control or original semantics changed")
            segment["widgets_values"].pop(2)
    return split_api(graph)


def freeze_graph(variant: str) -> tuple[dict, dict]:
    draft = Draft(source(variant))
    save = draft.make(SAVE, "STEP 1 · freeze exact original first-pass AV",
        [("av_latent", "LATENT")],
        [("av_latent", "LATENT"), ("status", "STRING"),
         ("checkpoint_path", "STRING"), ("file_sha256", "STRING"),
         ("manifest_json", "STRING"), ("report_json", "STRING")],
        [f"motion_{variant.lower()}_firstpass", CHECKPOINT_ID, False, True, 8],
        (2650, 150))
    draft.connect((9, 0), save, "av_latent", "LATENT")
    frontend = draft.prune((save["id"],))
    api = _api(frontend, variant)
    kinds = [node["class_type"] for node in api.values()]
    if kinds.count("SamplerCustomAdvanced") != 1 or kinds.count(SAVE) != 1 or any(
            kind in kinds for kind in ("MiniMaxH3StageSamplerEXPT8", LOAD)):
        raise ValueError("Motion freeze graph is not first-pass only")
    return frontend, api


def resume_graph(variant: str) -> tuple[dict, dict]:
    draft = Draft(source(variant))
    source_api = _api(draft.graph, variant)
    conditioning = source_api["5"]["inputs"]
    dimensions = (conditioning["length"], conditioning["width"], conditioning["height"])
    if any(type(item) is not int or item <= 0 for item in dimensions):
        raise ValueError("Motion source geometry is not a fixed native H3 canvas")
    load = draft.make(LOAD, "STEP 2 · paste frozen first-pass path, manifest and file SHA",
        [], [("av_latent", "LATENT"), ("report_json", "STRING")],
        ["", "", "", *dimensions, 8], (0, 250))
    consumers = [(node, item["name"], item["type"])
                 for node in tuple(draft.nodes.values()) for item in node.get("inputs", [])
                 if item.get("link") is not None and draft.links[item["link"]][1:3] == [9, 0]]
    if {node["id"] for node, _, _ in consumers} != {10, 12}:
        raise ValueError("Motion source pass-1 AV consumers changed")
    for node, name, dtype in consumers:
        draft.disconnect(node, name)
        draft.connect((load["id"], 0), node, name, dtype)
    delivery = 24 if variant == "Windowed" else 22
    frontend = draft.prune((delivery,))
    api = _api(frontend, variant)
    kinds = [node["class_type"] for node in api.values()]
    if ("9" in api or kinds.count("SamplerCustomAdvanced") != 0
            or kinds.count("MiniMaxH3StageSamplerEXPT8") != 1
            or kinds.count(LOAD) != 1
            or kinds.count("MiniMaxH3PromptRelayPlanT8Advanced") != 1
            or kinds.count("MiniMaxH3StageEAVApplyEXPT8") != 1):
        raise ValueError("Motion cold graph still executes pass 1 or lost pass-2 effects")
    return frontend, api


def main() -> None:
    if TARGET.exists():
        raise FileExistsError("Private Motion storage candidate already exists")
    pairs = {variant: (freeze_graph(variant), resume_graph(variant)) for variant in SOURCE_SHA}
    TARGET.mkdir(parents=True)
    for variant, (freeze, resume) in pairs.items():
        for phase, (frontend, api) in (("Freeze_Pass1", freeze), ("Resume_Only_Pass2", resume)):
            stem = f"Motion_Recovery_{variant}_{phase}_Relay_EAV_EXP"
            (TARGET / f"{stem}.json").write_text(
                json.dumps(frontend, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (TARGET / f"{stem}.api.json").write_text(
                json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"variant": variant, "phase": phase,
                              "frontend_nodes": len(frontend["nodes"]),
                              "api_nodes": len(api)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
