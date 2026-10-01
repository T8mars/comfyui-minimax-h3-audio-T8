"""Four additive Multi-Face full/fresh graphs; preserve all existing S24 files.

Each character has independently selected source-data and completed-stage
artifacts. Cold bypasses SAM inference, VAE encoding and all H3 samplers,
without weakening the original source-bound audit, stitch or review controls.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_modular_sampling_compat import write_new  # noqa: E402
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402
from tools import build_formal_face_refine_storage_workflows as storage  # noqa: E402
from tools.modular_frontend_layout import spread_frontend_columns  # noqa: E402

SAVE = "MiniMaxH3MultiFaceSourceSaveEXPT8"
LOAD = "MiniMaxH3MultiFaceSourceLoadEXPT8"
DESTINATION = storage.face.DESTINATION / "frozen-multiface-source-exp"
FILES = {(variant, mode): f"S24_Face_{variant}_combined_{mode}_Frozen_Source_EXP.json"
         for variant in ("multiface2", "multiface3") for mode in storage.MODES}
DATA_OUTPUTS = [("face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN"), ("source_frames", "IMAGE"),
                ("av_latent", "LATENT")]


def _rewire(draft, node, name, edge, dtype):
    draft.disconnect(node, name)
    draft.connect(edge, node, name, dtype)


def graph_for(variant, mode):
    if (variant, mode) not in FILES:
        raise ValueError("Only the two explicit Multi-Face combined full/cold pairs are supported")
    source = storage.face.DESTINATION / storage.FILES[variant, "combined", mode]
    original_bytes = source.read_bytes()
    original, _ = storage.graph_for(variant, "combined", mode)
    if json.loads(original_bytes) != original:
        raise ValueError("Original Multi-Face graph differs from its unchanged builder")
    draft = Draft(original)
    audits = sorted((node for node in draft.nodes.values()
                     if node["type"] == "MiniMaxH3MultiFaceStageAuditEXPT8"), key=lambda node: node["id"])
    if len(audits) != (2 if variant == "multiface2" else 3):
        raise ValueError("Multi-Face character inventory changed")
    creates = [node for node in draft.nodes.values() if node["type"] == "CreateVideo"]
    if len(creates) != 1:
        raise ValueError("One original source-audio delivery required")
    audio_edge = storage._source(draft, creates[0], "audio")
    roots = [node["id"] for node in draft.nodes.values()
             if node["type"] in {"SaveVideo", "MiniMaxH3StageSaveEXPT8"}]
    for index, audit in enumerate(audits):
        stage_edge = storage._source(draft, audit, "stage_result")
        parent_edge = storage._source(draft, audit, "parent_frames")
        plan_edge = storage._source(draft, audit, "face_plan")
        source_edge = storage._source(draft, audit, "source_frames")
        av_edge = storage._source(draft, audit, "av_latent")
        if mode == "full_save":
            node = draft.make(SAVE, f"Character {index + 1} · enable explicit source save; copy path + SHA",
                [("stage_result", "T8_STAGE_RESULT"), ("face_plan", DATA_OUTPUTS[0][1]),
                 ("source_frames", "IMAGE"), ("parent_frames", "IMAGE"), ("av_latent", "LATENT"),
                 ("source_audio", "AUDIO")],
                [*DATA_OUTPUTS, ("artifact_path", "STRING"), ("artifact_sha256", "STRING"),
                 ("report_json", "STRING")], [f"{variant}_character_{index + 1}", False], (5700, index * 550))
            for name, edge, dtype in (("stage_result", stage_edge, "T8_STAGE_RESULT"),
                    ("face_plan", plan_edge, DATA_OUTPUTS[0][1]), ("source_frames", source_edge, "IMAGE"),
                    ("parent_frames", parent_edge, "IMAGE"), ("av_latent", av_edge, "LATENT"),
                    ("source_audio", audio_edge, "AUDIO")):
                draft.connect(edge, node, name, dtype)
            roots.append(node["id"])
        else:
            node = draft.make(LOAD, f"Character {index + 1} · paste matching source-data path + SHA",
                [("stage_result", "T8_STAGE_RESULT"), ("parent_frames", "IMAGE"), ("source_audio", "AUDIO")],
                [*DATA_OUTPUTS, ("report_json", "STRING")], ["", ""], (3500, index * 550))
            draft.connect(stage_edge, node, "stage_result", "T8_STAGE_RESULT")
            draft.connect(parent_edge, node, "parent_frames", "IMAGE")
            draft.connect(audio_edge, node, "source_audio", "AUDIO")
            # Reuse the exact saved plan/window for audit, stitch and composite.
            # Never rerun SAM or silently change its hash/identity assignments.
            for consumer in list(draft.nodes.values()):
                if consumer["id"] == node["id"]:
                    continue
                for item in list(consumer.get("inputs", [])):
                    if item.get("link") is None:
                        continue
                    old = storage._source(draft, consumer, item["name"])
                    if old in (plan_edge, source_edge):
                        slot = 0 if old == plan_edge else 1
                        _rewire(draft, consumer, item["name"], (node["id"], slot), item["type"])
            _rewire(draft, audit, "av_latent", (node["id"], 2), "LATENT")
    frontend = draft.prune(roots)
    # These are new explicit delivery examples; the 84 old stock writers stay intact.
    writer, = (node for node in frontend["nodes"] if node["type"] == "SaveVideo")
    prefix = writer["widgets_values"][0]
    writer.update(type="MiniMaxH3SaveVideoIsolatedEXPT8",
        title="Explicit isolated H264/AAC · original sampling/audio unchanged",
        widgets_values=[prefix, 600, 16., 2.], size=[470, 300],
        inputs=[deepcopy(writer["inputs"][0])],
        outputs=[{"name": name, "type": dtype, "links": []}
                 for name, dtype in (("video", "VIDEO"), ("video_path", "STRING"), ("report_json", "STRING"))],
        properties={"Node name for S&R": "MiniMaxH3SaveVideoIsolatedEXPT8", "cnr_id": "minimax-h3-audio-T8"})
    api = storage.face._base(variant)[1](frontend)
    widget_fields = {SAVE: ("filename_prefix", "confirm_save"), LOAD: ("artifact_path", "artifact_sha256"),
        "MiniMaxH3StageSaveEXPT8": ("prefix",),
        "MiniMaxH3StageLoadEXPT8": ("artifact_path", "artifact_sha256", "expected_stage"),
        "MiniMaxH3SaveVideoIsolatedEXPT8": ("filename_prefix", "timeout_seconds", "max_staging_gib", "min_free_disk_gib"),
        "MiniMaxH3PromptRelayPlanT8Advanced": storage.face.RELAY_PLAN_WIDGETS,
        "MiniMaxH3PromptRelayConditioningT8Advanced": storage.face.RELAY_CONDITIONING_WIDGETS}
    for node in frontend["nodes"]:
        if node["type"] in widget_fields:
            storage.face._add_widget_inputs(api, node, zip(widget_fields[node["type"]], node["widgets_values"], strict=True))
        elif node["type"] == "MiniMaxH3StageEAVConfigEXPT8":
            storage.face._add_widget_inputs(api, node, storage.face.CONFIG_WIDGETS)
    # The historical converter retains old SaveVideo widget mappings by ID.
    # Replace that one API contract, not merely overlay the new writer fields.
    writer_key = str(writer["id"])
    api[writer_key] = {"class_type": "MiniMaxH3SaveVideoIsolatedEXPT8", "inputs": {
        "video": api[writer_key]["inputs"]["video"],
        **dict(zip(widget_fields["MiniMaxH3SaveVideoIsolatedEXPT8"], writer["widgets_values"], strict=True))}}
    frontend["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL,
        "t8/S24/frozen-multiface/" + variant + "/" + mode + "/" + hashlib.sha256(original_bytes).hexdigest()))
    frontend["extra"]["t8_split_example"].update(source_inputs="explicit_frozen_multiface",
        delivery="explicit_isolated_h264", status="experimental_source_data_not_quality_accepted")
    spread_frontend_columns(frontend)
    if mode == "cold_delivery" and any(node["class_type"] in {
            "MiniMaxH3SAM31MultiPersonTrackT8Advanced", "MiniMaxH3MultiFaceRepairJobT8Advanced",
            "MiniMaxH3StageSamplerEXPT8", "MiniMaxH3PromptRelayConditioningT8Advanced"} for node in api.values()):
        raise ValueError("Frozen cold graph unexpectedly reruns source preparation or sampling")
    if source.read_bytes() != original_bytes:
        raise ValueError("Original Multi-Face workflow changed during export")
    return frontend, api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    output = args.output.resolve()
    if output != DESTINATION.resolve() and not output.is_relative_to(ROOT / "artifacts/development"):
        raise ValueError("Only additive examples or owned private drafts are supported")
    for (variant, mode), filename in FILES.items():
        frontend, _ = graph_for(variant, mode)
        write_new(output / filename, frontend)
        print(json.dumps(dict(file=filename, nodes=len(frontend["nodes"]), links=len(frontend["links"]))))


if __name__ == "__main__":
    main()
