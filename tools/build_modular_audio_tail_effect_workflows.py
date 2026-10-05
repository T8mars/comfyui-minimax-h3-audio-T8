"""Private S26 tail EAV / independent Relay drafts; old pairs stay unchanged."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_formal_audio_refine_split_workflows import generated as old_graphs  # noqa: E402
from tools.build_modular_audio_refine_resume_all import _single, _source  # noqa: E402
from tools.build_modular_audio_refine_workflows import FAMILIES  # noqa: E402
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402

BIND = "MiniMaxH3AudioRefineEffectsBindEXPT8"
GUIDER = "MiniMaxH3AudioRefineEffectsGuiderEXPT8"
CONFIG = "MiniMaxH3StageEAVConfigEXPT8"
APPLY = "MiniMaxH3StageEAVApplyEXPT8"
AUDIT = "MiniMaxH3StageEAVAuditEXPT8"
RELAY_PLAN = "MiniMaxH3PromptRelayPlanT8Advanced"
RELAY_ROUTE = "MiniMaxH3PromptRelayQueryRouteT8Advanced"
RELAY_COND = "MiniMaxH3PromptRelayConditioningT8Advanced"
PLAN_TYPE = "H3_T8_PROMPT_RELAY_PLAN"
CONDITIONING_WIDGETS = ("prompt", "width", "height", "length", "task_type", "audio_mode",
    "audio_denoise_strength", "add_source_as_reference", "prompt_primary_audio_ordinal",
    "strict_prompt_tags", "ref_image_size", "reference_video_policy", "allow_above_reference_area")


def _fresh_relay(draft, bind):
    """Encode the tail's independent text against the exact original media.

    The output empty AV is intentionally unconnected. Geometry comes from the
    existing checkpoint guard; the original audit/Setup graph is not rewired.
    """
    if any(node["type"] in {RELAY_COND, "MiniMaxH3PromptRelayLongVideoConditioningT8Advanced"}
           for node in draft.nodes.values()):
        raise ValueError("Use the existing independent tail Relay Plan, not a second owner")
    setup = _single(draft, set(FAMILIES))
    old, slot, dtype = _source(draft, setup, "positive")
    if (old["type"], slot, dtype) != ("MiniMaxH3AudioConditioningT8", 0, "CONDITIONING"):
        raise ValueError("Fresh tail Relay requires the original native conditioner")
    values = old["widgets_values"]
    if not isinstance(values, list) or len(values) not in (12, 13):
        raise ValueError("Original native conditioning widget contract changed")
    widgets = dict(zip(CONDITIONING_WIDGETS, values))
    plan = draft.make(RELAY_PLAN, "TAIL ONLY · independent Relay text / timeline",
        [("length", "INT")], [("prompt_relay_plan", PLAN_TYPE), ("compiled_prompt", "STRING"),
        ("length", "INT"), ("timeline_json", "STRING"), ("report_json", "STRING")],
        [widgets["prompt"], "", widgets["length"], "auto_equal", "", "paper_v1", .1, False, False],
        (1580, -1420))
    plan["inputs"][0]["widget"] = {"name": "length"}
    source, source_slot, port_type = _source(draft, old, "length")
    if port_type != "INT":
        raise ValueError("Fresh tail Relay length must come from authenticated frozen AV")
    draft.connect((source["id"], source_slot), plan, "length", "INT")
    route = draft.make(RELAY_ROUTE, "TAIL ONLY · optional joint AV query route",
        [("prompt_relay_plan", PLAN_TYPE)], [("prompt_relay_plan", PLAN_TYPE), ("report_json", "STRING")],
        ["video_only_paper"], (2100, -1420))
    draft.connect((plan["id"], 0), route, "prompt_relay_plan", PLAN_TYPE)
    # allow_above_reference_area is a legacy warning-only flag, not an input on
    # Relay Conditioning. Do not invent that port or change the native node.
    copied = [item for item in old["inputs"]
              if item["name"] not in {"prompt", "length", "allow_above_reference_area"}]
    cond = draft.make(RELAY_COND, "TAIL ONLY · Relay MODEL + positive (AV unused)",
        [("model", "MODEL"), ("prompt_relay_plan", PLAN_TYPE)] +
        [(item["name"], item["type"]) for item in copied],
        [("model", "MODEL"), ("positive", "CONDITIONING"), ("av_latent", "LATENT"),
         ("mux_audio", "AUDIO"), ("conditioned_prompt", "STRING"), ("media_map_json", "STRING"),
         ("report_json", "STRING")],
        [widgets[name] for name in CONDITIONING_WIDGETS[1:3] + CONDITIONING_WIDGETS[4:12]] +
        ["report_only", 256], (2620, -1420))
    draft.connect((bind["id"], 0), cond, "model", "MODEL")
    draft.connect((route["id"], 0), cond, "prompt_relay_plan", PLAN_TYPE)
    for item in old["inputs"]:
        if item["name"] == "prompt" and item.get("link") is not None:
            plan["inputs"].append({"name": "global_prompt", "type": "STRING", "link": None,
                                   "widget": {"name": "global_prompt"}})
            source, source_slot, port_type = _source(draft, old, "prompt")
            draft.connect((source["id"], source_slot), plan, "global_prompt", port_type)
    for item, target_item in zip(copied, cond["inputs"][2:], strict=True):
        for key in ("widget", "shape", "label"):
            if key in item:
                target_item[key] = deepcopy(item[key])
        if item.get("link") is not None:
            source, source_slot, port_type = _source(draft, old, item["name"])
            draft.connect((source["id"], source_slot), cond, item["name"], port_type)
    return cond


def add_tail_effects(graph, *, effect="eav"):
    if effect not in {"eav", "relay", "combined"}:
        raise ValueError("Unknown tail effect variant")
    draft = Draft(graph)
    bound = _single(draft, {pair[0] for pair in FAMILIES.values()})
    candidate_audit = _single(draft, {pair[1] for pair in FAMILIES.values()})
    sampler = _single(draft, {"SamplerCustomAdvanced"})
    origin, slot, dtype = _source(draft, sampler, "guider")
    if (origin["id"], slot, dtype) != (bound["id"], 2, "GUIDER"):
        raise ValueError("S26 tail guider is no longer the original bound Setup guider")
    bind = draft.make(BIND, "TAIL ONLY · source-bound external effects",
        [("stage_boundary", "T8_AUDIO_REFINE_STAGE_BOUNDARY"), ("model", "MODEL"), ("noise", "NOISE"),
         ("guider", "GUIDER"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"), ("stage_latent", "LATENT")],
        [("model", "MODEL"), ("stage_context", "T8_STAGE_CONTEXT"), ("report_json", "STRING")],
        [0, 0], (2100, -720))
    for name, source_slot, port_type in (("model", 0, "MODEL"), ("noise", 1, "NOISE"),
            ("guider", 2, "GUIDER"), ("sampler", 3, "SAMPLER"), ("sigmas", 4, "SIGMAS"),
            ("stage_latent", 5, "LATENT"), ("stage_boundary", 6, "T8_AUDIO_REFINE_STAGE_BOUNDARY")):
        draft.connect((bound["id"], source_slot), bind, name, port_type)
    long_conds = [node for node in draft.nodes.values()
                  if node["type"] == "MiniMaxH3PromptRelayLongVideoConditioningT8Advanced"]
    if long_conds:
        if len(long_conds) != 1:
            raise ValueError("S26 cold long tail must have one independent Relay conditioner")
        for field in ("segment_index", "context_frames"):
            source, source_slot, port_type = _source(draft, long_conds[0], field)
            if port_type != "INT":
                raise ValueError("Long tail window is no longer an INT input")
            bind["inputs"].append({"name": field, "type": "INT", "link": None, "widget": {"name": field}})
            draft.connect((source["id"], source_slot), bind, field, "INT")
    relay_cond = _fresh_relay(draft, bind) if effect != "eav" else None
    effect_model = ((relay_cond or bind)["id"], 0)
    config = draft.make(CONFIG, "TAIL ONLY · EAV controls (not audio quality approval)", [],
        [("eav_config", "T8_STAGE_EAV_CONFIG")], ["report_only", 4., .15, .90, 32, 1.5], (2620, -720))
    apply = draft.make(APPLY, "TAIL ONLY · apply external EAV",
        [("model", "MODEL"), ("sigmas", "SIGMAS"), ("av_latent", "LATENT"),
         ("stage_context", "T8_STAGE_CONTEXT"), ("eav_config", "T8_STAGE_EAV_CONFIG")],
        [("model", "MODEL"), ("runtime", "T8_STAGE_EAV_RUNTIME"), ("report_json", "STRING")], [], (3140, -720))
    for field, source, port_type in (("model", effect_model, "MODEL"),
            ("sigmas", (bound["id"], 4), "SIGMAS"), ("av_latent", (bound["id"], 5), "LATENT"),
            ("stage_context", (bind["id"], 1), "T8_STAGE_CONTEXT"),
            ("eav_config", (config["id"], 0), "T8_STAGE_EAV_CONFIG")):
        draft.connect(source, apply, field, port_type)
    guider = draft.make(GUIDER, "TAIL ONLY · guider (optional paired Relay positive)",
        [("model", "MODEL"), ("stage_context", "T8_STAGE_CONTEXT"), ("sigmas", "SIGMAS"),
         ("stage_latent", "LATENT"), ("positive", "CONDITIONING")],
        [("guider", "GUIDER"), ("report_json", "STRING")], [], (3660, -720))
    for field, source, port_type in (("model", (apply["id"], 0), "MODEL"),
            ("stage_context", (bind["id"], 1), "T8_STAGE_CONTEXT"),
            ("sigmas", (bound["id"], 4), "SIGMAS"), ("stage_latent", (bound["id"], 5), "LATENT")):
        draft.connect(source, guider, field, port_type)
    if relay_cond is not None:
        draft.connect((relay_cond["id"], 1), guider, "positive", "CONDITIONING")
    draft.disconnect(sampler, "guider")
    draft.connect((guider["id"], 0), sampler, "guider", "GUIDER")
    audit = draft.make(AUDIT, "TAIL ONLY · actual EAV / combined Relay calls",
        [("av_latent", "LATENT"), ("runtime", "T8_STAGE_EAV_RUNTIME")],
        [("av_latent", "LATENT"), ("report_json", "STRING")], [], (4180, -720))
    original, original_slot, port_type = _source(draft, candidate_audit, "candidate_av_latent")
    if (original["id"], original_slot, port_type) != (sampler["id"], 0, "LATENT"):
        raise ValueError("S26 candidate audit no longer consumes the final tail output")
    draft.connect((sampler["id"], 0), audit, "av_latent", "LATENT")
    draft.connect((apply["id"], 1), audit, "runtime", "T8_STAGE_EAV_RUNTIME")
    draft.disconnect(candidate_audit, "candidate_av_latent")
    draft.connect((audit["id"], 0), candidate_audit, "candidate_av_latent", "LATENT")
    if effect == "relay":
        # Relay-only really omits EAV, rather than requiring an unused wrapper.
        draft.disconnect(guider, "model")
        draft.connect(effect_model, guider, "model", "MODEL")
        draft.disconnect(candidate_audit, "candidate_av_latent")
        draft.connect((sampler["id"], 0), candidate_audit, "candidate_av_latent", "LATENT")
        removed = {config["id"], apply["id"], audit["id"]}
        for node_id in removed:
            for item in list(draft.nodes[node_id]["inputs"]):
                draft.disconnect(draft.nodes[node_id], item["name"])
        for node_id in removed:
            del draft.nodes[node_id]
        draft.graph["nodes"] = [node for node in draft.graph["nodes"] if node["id"] not in removed]
    draft.graph["last_node_id"] = draft.next_node - 1
    draft.graph["last_link_id"] = draft.next_link - 1
    return draft.graph


def generated(effect="eav"):
    originals = old_graphs()  # Pins all ten old examples before creating anything.
    result = {}
    for path, graph in originals.items():
        if "_resume_audio_" in path.name:
            if json.loads(path.read_text(encoding="utf8")) != graph:
                raise ValueError("Public S26 base graph changed: " + str(path))
            if effect != "eav" and any(node["type"] == RELAY_PLAN for node in graph["nodes"]):
                continue  # These two recipes already have independent Relay plans.
            result[path] = add_tail_effects(graph, effect=effect)
    if len(result) != (10 if effect == "eav" else 8):
        raise ValueError("S26 effects draft recipe inventory changed")
    return result


def generated_matrix():
    from tools.workflow_paths import legacy_workflow_path
    return {legacy_workflow_path(path).with_name(legacy_workflow_path(path).stem + "_" + effect + ".json"): (path, graph)
            for effect in ("eav", "relay", "combined") for path, graph in generated(effect).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix", action="store_true", help="All 26 additive effect drafts")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifacts directory; old graphs are never overwritten")
    originals = old_graphs()
    original_sha = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in originals}
    graphs = generated_matrix() if args.matrix else {path: (path, graph) for path, graph in generated().items()}
    from tools.validate_modular_audio_refine_resume_all import validate_pairs
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTORCH_NVML_BASED_CUDA_CHECK="0")
    sys.path[:0] = [str(ROOT.parents[1]), str(ROOT)]
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    pairs = {}
    for path, (original_path, graph) in graphs.items():
        freeze = next(value for key, value in originals.items()
                      if key.name == original_path.name.replace("_resume_audio_", "_freeze_video_"))
        pairs[path] = (freeze, graph)
    results, candidates = asyncio.run(validate_pairs(pairs=pairs))
    unchanged = original_sha == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in originals}
    success = len(results) == 2 * len(graphs) and all(item["core_valid"] for item in results.values()) and unchanged
    output.mkdir(parents=True)
    for key, (graph, api) in candidates.items():
        if not key.endswith("/resume_audio"):
            continue
        filename = key.split("/")[0] + "_TailEffects"
        (output / (filename + ".json")).write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        (output / (filename + ".api.json")).write_text(json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    report = {"status": "pass" if success else "fail", "results": results,
        "original_public_sha256": original_sha, "originals_unchanged": unchanged,
        "queued": False, "browser_used": False, "cuda_initialized": torch.cuda.is_initialized(),
        "boundary": "Static CPU Core validation only. Independent Relay text starts with no events "
                    "and report_only; user opts in. Browser/cold/GPU/quality gates are separate."}
    (output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": report["status"], "cases": len(results), "drafts": len(graphs),
                      "failed": [key for key, item in results.items() if not item["core_valid"]]}))
    return 0 if success and not torch.cuda.is_initialized() else 1


if __name__ == "__main__":
    raise SystemExit(main())
