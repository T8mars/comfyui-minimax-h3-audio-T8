"""Four opt-in RGB→LTX EAV full/cold graphs; originals stay byte-exact."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import build_modular_ltx_rgb_source_workflows as source  # noqa: E402
from tools.build_modular_ltx_rgb_workflows import _source, _unique, SETUPS, BIND, AUDIT  # noqa: E402
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402

CONFIG = "MiniMaxH3StageEAVConfigEXPT8"
APPLY = "MiniMaxH3LTXEAVApplyEXPT8"
EFFECT_AUDIT = "MiniMaxH3LTXEAVAuditEXPT8"


def build(graph):
    draft = Draft(graph)
    setup = next(node for node in draft.nodes.values() if node["type"] in SETUPS)
    bind, stage_audit = _unique(draft, BIND), _unique(draft, AUDIT)
    guider, decoder = _unique(draft, "CFGGuider"), _unique(draft, "MiniMaxH3SolEngineTAEHVDecodeT8Advanced")
    exporter = _unique(draft, source.ISOLATED_WRITER)
    store = _unique(draft, source.LOAD if any(n["type"] == source.LOAD for n in draft.nodes.values()) else source.SAVE)
    if (_source(draft, bind, "model")[0] != (setup["id"], 0) or
            _source(draft, guider, "model")[0] != (setup["id"], 0) or
            _source(draft, stage_audit, "model")[0] != (setup["id"], 0) or
            _source(draft, decoder, "latent")[0] != (stage_audit["id"], 0)):
        raise ValueError("Saved LTX external model/candidate contract changed")
    config = draft.make(CONFIG, "LTX video EAV controls; report only until explicitly enabled", [],
        [("eav_config", "T8_STAGE_EAV_CONFIG")], ["report_only", .25, 0., 1., 32, 1.5], (1780, -250))
    effect = draft.make(APPLY, "Apply LTX EAV after Setup/LoRA, before Guider and Bind",
        [("model", "MODEL"), ("ltx_latent", "LATENT"), ("sigmas", "SIGMAS"),
         ("eav_config", "T8_STAGE_EAV_CONFIG")],
        [("model", "MODEL"), ("runtime", "T8_LTX_EAV_RUNTIME"), ("report_json", "STRING")],
        [], (2190, -180))
    audit = draft.make(EFFECT_AUDIT, "Report measured/applied LTX EAV; never certify media",
        [("model", "MODEL"), ("candidate_latent", "LATENT"), ("runtime", "T8_LTX_EAV_RUNTIME")],
        [("candidate_latent", "LATENT"), ("report_json", "STRING")], [], (2920, 500))
    for field, slot, dtype in (("model", (setup["id"], 0), "MODEL"),
                               ("ltx_latent", (store["id"], 4), "LATENT"),
                               ("sigmas", (setup["id"], 2), "SIGMAS"),
                               ("eav_config", (config["id"], 0), "T8_STAGE_EAV_CONFIG")):
        draft.connect(slot, effect, field, dtype)
    for target in (bind, guider, stage_audit):
        draft.disconnect(target, "model")
        draft.connect((effect["id"], 0), target, "model", "MODEL")
    draft.connect((effect["id"], 0), audit, "model", "MODEL")
    draft.connect((effect["id"], 1), audit, "runtime", "T8_LTX_EAV_RUNTIME")
    draft.connect((stage_audit["id"], 0), audit, "candidate_latent", "LATENT")
    draft.disconnect(decoder, "latent")
    draft.connect((audit["id"], 0), decoder, "latent", "LATENT")
    return draft.prune((exporter["id"], audit["id"]))


def generated():
    originals = source.generated(isolated_media=True)
    return {name + "_external_eav": build(graph) for name, graph in originals.items()
            if not name.endswith("freeze_source")}


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
