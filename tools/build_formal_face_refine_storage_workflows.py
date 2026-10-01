"""Append S24 full-save and explicit cold-delivery graphs without changing old Face graphs.

Each frozen Face job has its own path and SHA. A cold graph does not resample
that job; the variant-specific source audit still checks the current plan,
source frames, input AV, and (where present) parent/window mapping and audio.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_modular_sampling_compat import write_new  # noqa: E402
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402
from tools import build_formal_face_refine_eav_workflows as face  # noqa: E402
from tools.modular_frontend_layout import spread_frontend_columns  # noqa: E402

MODES = ("full_save", "cold_delivery")
FILES = {
    (variant, effect, mode): f"S24_Face_{variant}_{effect}_{mode}_Separate_Stage_EXP.json"
    for variant, effect in face.FILES for mode in MODES
}
LOAD_OUTPUTS = [("output", "LATENT"), ("denoised_output", "LATENT"),
                ("stage_context", "T8_STAGE_CONTEXT"),
                ("stage_result", "T8_STAGE_RESULT"), ("report_json", "STRING")]
SAVE_OUTPUTS = [("output", "LATENT"), ("denoised_output", "LATENT"),
                ("artifact_path", "STRING"), ("artifact_sha256", "STRING"),
                ("report_json", "STRING")]


def _source(draft: Draft, node: dict, name: str) -> tuple[int, int]:
    item = next(value for value in node["inputs"] if value["name"] == name)
    return tuple(draft.links[item["link"]][1:3])


def _save_graph(frontend: dict, variant: str, effect: str) -> dict:
    draft = Draft(frontend)
    stages = sorted((node for node in draft.nodes.values()
                     if node["type"] == "MiniMaxH3StageSamplerEXPT8"), key=lambda node: node["id"])
    delivery = [node for node in draft.nodes.values() if node["type"] == "SaveVideo"]
    if not stages or len(delivery) != 1:
        raise ValueError("S24 full-save needs separate jobs and one delivery")
    roots = [delivery[0]["id"]]
    for index, stage in enumerate(stages):
        saver = draft.make("MiniMaxH3StageSaveEXPT8",
            f"Face job {index + 1} · freeze exact completed stage (copy path + SHA)",
            [("stage_result", "T8_STAGE_RESULT"), ("prefix", "STRING")], SAVE_OUTPUTS,
            [f"S24_Face_{variant}_{effect}_job{index + 1}"], (5550, index * 480))
        draft.connect((stage["id"], 2), saver, "stage_result", "T8_STAGE_RESULT")
        roots.append(saver["id"])
    return draft.prune(roots)


def _cold_graph(frontend: dict) -> dict:
    draft = Draft(frontend)
    stages = sorted((node for node in draft.nodes.values()
                     if node["type"] == "MiniMaxH3StageSamplerEXPT8"), key=lambda node: node["id"])
    delivery = [node for node in draft.nodes.values() if node["type"] == "SaveVideo"]
    if not stages or len(delivery) != 1:
        raise ValueError("S24 cold delivery needs separate jobs and one delivery")
    for index, stage in enumerate(stages):
        audits = [node for node in draft.nodes.values() if node["type"] in face.FACE_AUDITS
                  and _source(draft, node, "stage_result") == (stage["id"], 2)]
        if len(audits) != 1:
            raise ValueError("S24 cold job has no unique source-bound Face audit")
        audit = audits[0]
        loader = draft.make("MiniMaxH3StageLoadEXPT8",
            f"Face job {index + 1} · paste frozen manifest path + exact SHA",
            [("artifact_path", "STRING"), ("artifact_sha256", "STRING"),
             ("expected_stage", "COMBO")], LOAD_OUTPUTS,
            ["", "", "native_high"], (3700, index * 480))
        draft.disconnect(audit, "stage_result")
        draft.connect((loader["id"], 3), audit, "stage_result", "T8_STAGE_RESULT")
        eav_audits = [node for node in draft.nodes.values()
                      if node["type"] == "MiniMaxH3StageEAVAuditEXPT8" and
                      _source(draft, node, "av_latent") == (audit["id"], 0)]
        if eav_audits:
            if len(eav_audits) != 1:
                raise ValueError("S24 cold job has ambiguous EAV audits")
            decoders = [node for node in draft.nodes.values()
                        if node["type"] == "MiniMaxH3AVDecodeT8" and
                        _source(draft, node, "av_latent") == (eav_audits[0]["id"], 0)]
            if len(decoders) != 1:
                raise ValueError("S24 cold EAV job has no unique crop decoder")
            draft.disconnect(decoders[0], "av_latent")
            draft.connect((audit["id"], 0), decoders[0], "av_latent", "LATENT")
    return draft.prune((delivery[0]["id"],))


def graph_for(variant: str, effect: str, mode: str) -> tuple[dict, dict]:
    if (variant, effect, mode) not in FILES:
        raise ValueError("Unknown S24 Face storage graph")
    frontend, _ = face.graph_for(variant, effect)
    frontend = (_save_graph(frontend, variant, effect) if mode == "full_save"
                else _cold_graph(frontend))
    convert = face._base(variant)[1]
    api = convert(frontend)
    for node in frontend["nodes"]:
        kind = node["type"]
        if kind == "MiniMaxH3StageEAVConfigEXPT8":
            face._add_widget_inputs(api, node, face.CONFIG_WIDGETS)
        elif kind == "MiniMaxH3PromptRelayPlanT8Advanced":
            face._add_widget_inputs(api, node, zip(
                face.RELAY_PLAN_WIDGETS, node["widgets_values"], strict=True))
        elif kind == "MiniMaxH3PromptRelayConditioningT8Advanced":
            face._add_widget_inputs(api, node, zip(
                face.RELAY_CONDITIONING_WIDGETS, node["widgets_values"], strict=True))
        elif kind == "MiniMaxH3StageSaveEXPT8":
            face._add_widget_inputs(api, node, {"prefix": node["widgets_values"][0]})
        elif kind == "MiniMaxH3StageLoadEXPT8":
            face._add_widget_inputs(api, node, zip(
                ("artifact_path", "artifact_sha256", "expected_stage"),
                node["widgets_values"], strict=True))
    frontend["extra"]["t8_split_example"].update(storage=mode,
        status="experimental_frozen_stage_not_quality_accepted")
    spread_frontend_columns(frontend)
    return frontend, api


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=face.VARIANTS)
    parser.add_argument("--effect", choices=face.EFFECTS_BY_VARIANT[face.VARIANTS[0]])
    parser.add_argument("--mode", choices=MODES)
    options = parser.parse_args()
    for key, filename in FILES.items():
        variant, effect, mode = key
        if ((options.variant and variant != options.variant) or
                (options.effect and effect != options.effect) or
                (options.mode and mode != options.mode)):
            continue
        frontend, _api = graph_for(*key)
        write_new(face.DESTINATION / filename, frontend)
        print(json.dumps({"file": filename, "nodes": len(frontend["nodes"]),
                          "links": len(frontend["links"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
