"""Eight opt-in LTX Relay full/cold drafts, with optional independent EAV.

The existing RGB source, sampling, media and audio routes are copied, never
rewritten in place. Relay defaults to report-only until explicitly enabled.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import build_modular_ltx_eav_workflows as eav  # noqa: E402
from tools import build_modular_ltx_rgb_source_workflows as source  # noqa: E402
from tools.build_modular_ltx_rgb_workflows import _source, _unique, SETUPS, BIND, AUDIT  # noqa: E402
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402

PLAN = "MiniMaxH3LTXPromptRelayPlanEXPT8"
ENCODE = "MiniMaxH3LTXPromptRelayEncodeEXPT8"
APPLY = "MiniMaxH3LTXPromptRelayApplyEXPT8"
RELAY_AUDIT = "MiniMaxH3LTXPromptRelayAuditEXPT8"


def build(graph):
    draft = Draft(graph)
    setup = next(node for node in draft.nodes.values() if node["type"] in SETUPS)
    bind, stage_audit = _unique(draft, BIND), _unique(draft, AUDIT)
    guider = _unique(draft, "CFGGuider")
    conditioning = _unique(draft, "LTXVConditioning")
    decoder = _unique(draft, "MiniMaxH3SolEngineTAEHVDecodeT8Advanced")
    exporter = _unique(draft, source.ISOLATED_WRITER)
    store = _unique(draft, source.LOAD if any(n["type"] == source.LOAD for n in draft.nodes.values()) else source.SAVE)
    positive_slot, _ = _source(draft, conditioning, "positive")
    positive = draft.nodes[positive_slot[0]]
    negative_slot, _ = _source(draft, conditioning, "negative")
    negative = draft.nodes[negative_slot[0]]
    clip_slot, _ = _source(draft, positive, "clip")
    combined = any(node["type"] == eav.APPLY for node in draft.nodes.values())
    prior_model = (_unique(draft, eav.APPLY)["id"], 0) if combined else (setup["id"], 0)
    prior_candidate = (_unique(draft, eav.EFFECT_AUDIT)["id"], 0) if combined else (stage_audit["id"], 0)
    if (positive["type"] != "CLIPTextEncode" or negative["type"] != "CLIPTextEncode"
            or positive["id"] == negative["id"] or _source(draft, negative, "clip")[0] != clip_slot
            or _source(draft, guider, "model")[0] != prior_model
            or _source(draft, bind, "model")[0] != prior_model
            or _source(draft, stage_audit, "model")[0] != prior_model
            or _source(draft, decoder, "latent")[0] != prior_candidate):
        raise ValueError("Saved LTX text/model/candidate route changed")
    if combined and _source(draft, _unique(draft, eav.EFFECT_AUDIT), "model")[0] != prior_model:
        raise ValueError("Saved LTX EAV audit model route changed")

    global_prompt = positive["widgets_values"][0]
    plan = draft.make(PLAN, "Plan exact 8n+1 LTX events from this stage latent",
        [("ltx_latent", "LATENT"), ("prompt_relay_events", "H3_T8_PROMPT_RELAY_EVENTS")],
        [("ltx_relay_plan", "T8_LTX_PROMPT_RELAY_PLAN"), ("compiled_prompt", "STRING"),
         ("frame_count", "INT"), ("timeline_json", "STRING"), ("report_json", "STRING")],
        [global_prompt, "The subject enters the scene.\nThe subject turns toward the camera.",
         "auto_equal", "", 24., .1, False, False], (1760, -800))
    encode = draft.make(ENCODE, "Encode global/local segments with selected LTX CLIP",
        [("clip", "CLIP"), ("ltx_relay_plan", "T8_LTX_PROMPT_RELAY_PLAN")],
        [("positive", "CONDITIONING"), ("text_binding", "T8_LTX_RELAY_TEXT_BINDING"),
         ("report_json", "STRING")], [2048], (2180, -800))
    apply = draft.make(APPLY, "Video text-attention Relay; report only by default",
        [("model", "MODEL"), ("ltx_latent", "LATENT"), ("sigmas", "SIGMAS"),
         ("positive", "CONDITIONING"), ("text_binding", "T8_LTX_RELAY_TEXT_BINDING")],
        [("model", "MODEL"), ("positive", "CONDITIONING"),
         ("runtime", "T8_LTX_RELAY_RUNTIME"), ("report_json", "STRING")],
        ["report_only", 32], (2580, -550))
    audit = draft.make(RELAY_AUDIT, "Observe Relay coverage; do not certify media",
        [("model", "MODEL"), ("candidate_latent", "LATENT"),
         ("runtime", "T8_LTX_RELAY_RUNTIME")],
        [("candidate_latent", "LATENT"), ("report_json", "STRING")], [], (3400, 500))
    for target, field, slot, dtype in (
        (plan, "ltx_latent", (store["id"], 4), "LATENT"),
        (encode, "clip", clip_slot, "CLIP"),
        (encode, "ltx_relay_plan", (plan["id"], 0), "T8_LTX_PROMPT_RELAY_PLAN"),
        (apply, "model", prior_model, "MODEL"),
        (apply, "ltx_latent", (store["id"], 4), "LATENT"),
        (apply, "sigmas", (setup["id"], 2), "SIGMAS"),
        (apply, "positive", (encode["id"], 0), "CONDITIONING"),
        (apply, "text_binding", (encode["id"], 1), "T8_LTX_RELAY_TEXT_BINDING"),
        (audit, "model", (apply["id"], 0), "MODEL"),
        (audit, "candidate_latent", prior_candidate, "LATENT"),
        (audit, "runtime", (apply["id"], 2), "T8_LTX_RELAY_RUNTIME"),
    ):
        draft.connect(slot, target, field, dtype)
    for target in (guider, bind, stage_audit, *([_unique(draft, eav.EFFECT_AUDIT)] if combined else [])):
        draft.disconnect(target, "model")
        draft.connect((apply["id"], 0), target, "model", "MODEL")
    draft.disconnect(conditioning, "positive")
    draft.connect((apply["id"], 1), conditioning, "positive", "CONDITIONING")
    draft.disconnect(decoder, "latent")
    draft.connect((audit["id"], 0), decoder, "latent", "LATENT")
    return draft.prune((exporter["id"], audit["id"]))


def generated():
    base = {name + "_external_relay": graph for name, graph in
            source.generated(isolated_media=True).items() if not name.endswith("freeze_source")}
    combined = {name + "_relay": graph for name, graph in eav.generated().items()}
    return {name: build(graph) for name, graph in {**base, **combined}.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists() or not destination.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("A new private artifacts directory is required")
    graphs = generated()
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2")
    sys.path[:0] = [str(ROOT.parents[1]), str(ROOT)]
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    torch.set_num_threads(2)
    results, apis = asyncio.run(source.validate(graphs))
    if torch.cuda.is_initialized():
        raise RuntimeError("Static validation initialized CUDA")
    destination.mkdir(parents=True)
    for name, graph in graphs.items():
        (destination / (name + ".json")).write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf8")
        (destination / (name + ".api.json")).write_text(json.dumps(apis[name], ensure_ascii=False, indent=2), encoding="utf8")
    report = {"cases": results, "status": "pass" if all(row["valid"] for row in results.values()) else "fail",
              "queued": False, "cuda_initialized": False, "effect_default": "report_only",
              "source_media": "isolated_candidate_not_full_media_qualified"}
    (destination / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps({"status": report["status"], "cases": len(results), "directory": str(destination)}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
