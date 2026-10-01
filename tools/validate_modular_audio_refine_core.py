"""CPU-only Core validation of all ten private S26 frontend/API candidates.

No PromptServer, queue, model execution, GPU sampling or old example edits.
The optional VHS terminal is omitted; other existing output branches remain.
"""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_modular_audio_refine_workflows import sources, split_api, split_frontend  # noqa: E402

CORE = ROOT.parents[1]
TARGET = ROOT / "artifacts/development/modular-sampling-m4-audio-refine-20260923/candidate-v2"
REPORT = ROOT / "artifacts/development/modular-sampling-m4-audio-refine-20260923/core-validation-v1.json"


async def validate(verify_candidate_dir=None, *, abstain_safe=False,
                   external_refine_relay=False, independent_refine_model=False,
                   only_source_stem=None):
    import nodes
    import execution

    for name in ("nodes_custom_sampler.py", "nodes_video.py", "nodes_preview_any.py",
                 "nodes_lora_debug.py"):
        if not await nodes.load_custom_node(str(CORE / "comfy_extras" / name), module_parent="comfy_extras"):
            raise RuntimeError("Required Core node module failed: " + name)
    if not await nodes.load_custom_node(str(CORE / "custom_nodes/ComfyUI-ClipProj")):
        raise RuntimeError("Required installed ClipProj plugin failed import")
    spec = importlib.util.spec_from_file_location(
        "_t8_modular_audio_refine_core", ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    for cls in await package.comfy_entrypoint().get_node_list():
        nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
    frontend = []
    needed = set()
    source_paths = [path for path in sources()
                    if only_source_stem is None or path.stem == only_source_stem]
    if not source_paths:
        raise ValueError("No Audio Refine source matches the requested stem")
    for path in source_paths:
        graph = split_frontend(json.loads(path.read_text(encoding="utf-8")),
                               abstain_safe=abstain_safe,
                               external_refine_relay=external_refine_relay,
                               independent_refine_model=independent_refine_model)
        frontend.append((path, graph))
        needed.update(node["type"] for node in graph["nodes"])
    node_info = {}
    for kind in needed - {"MarkdownNote", "VHS_VideoCombine"}:
        cls = nodes.NODE_CLASS_MAPPINGS.get(kind)
        if cls is None:
            raise ValueError("Current Core lacks " + kind)
        node_info[kind] = cls.GET_NODE_INFO_V1() if hasattr(cls, "GET_NODE_INFO_V1") else cls.INPUT_TYPES()
    results = {}
    candidates = []
    for path, graph in frontend:
        try:
            api = split_api(graph, node_info)
            if verify_candidate_dir is not None:
                stem = path.stem + ("_StageBound_RefineRelay_IndependentModel_EXP"
                                    if independent_refine_model else
                                    "_StageBound_RefineRelay_EXP" if external_refine_relay
                                    else "_StageBound_EXP")
                stored_frontend = json.loads((verify_candidate_dir / (stem + ".json")).read_text(encoding="utf-8"))
                stored_api = json.loads((verify_candidate_dir / (stem + ".api.json")).read_text(encoding="utf-8"))
                if stored_frontend != graph or stored_api != api:
                    raise ValueError("Saved candidate differs from current source and converter")
                api = stored_api
            validated = await execution.validate_prompt("audio-refine-" + path.stem, api, None)
            types = {str(node["id"]): node["type"] for node in graph["nodes"]}
            output_types = {types[node_id] for node_id in validated[2]}
            audit_included = any(kind.endswith("StageAuditEXPT8") for kind in output_types)
            quality_gate_included = "MiniMaxH3AudioRefineQualityGateT8Advanced" in output_types
            results[path.name] = {"core_valid": bool(validated[0]) and not bool(validated[3])
                                  and audit_included and quality_gate_included,
                                  "result": validated, "api_nodes": len(api),
                                  "frontend_nodes": len(graph["nodes"]),
                                  "validated_output_count": len(validated[2]),
                                  "stage_audit_included": audit_included,
                                  "quality_gate_included": quality_gate_included}
            candidates.append((path, graph, api))
        except (KeyError, TypeError, ValueError) as error:
            results[path.name] = {"core_valid": False, "conversion_error": str(error)}
    return results, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--candidate-dir", type=Path, default=TARGET)
    parser.add_argument("--verify-candidate-dir", type=Path)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--abstain-safe", action="store_true",
                        help="Build new split candidates with explicit ABSTAIN-safe tail audit")
    parser.add_argument("--external-refine-relay", action="store_true",
                        help="Add a separate tail-only Relay Plan and paired MODEL/CONDITIONING")
    parser.add_argument("--independent-refine-model", action="store_true",
                        help="Give the single-segment Relay tail its own checkpoint and optional LoRA")
    parser.add_argument("--only-source-stem",
                        help="Limit validation to one exact existing source workflow stem")
    command = parser.parse_args()
    report_path = command.report.resolve()
    candidate_dir = command.candidate_dir.resolve()
    verify_candidate_dir = command.verify_candidate_dir.resolve() if command.verify_candidate_dir else None
    if (not report_path.is_relative_to(ROOT / "artifacts") or
            not candidate_dir.is_relative_to(ROOT / "artifacts") or
            (verify_candidate_dir is not None and
             not verify_candidate_dir.is_relative_to(ROOT / "artifacts"))):
        parser.error("Private evidence must stay under project artifacts")
    if report_path.exists() or (candidate_dir.exists() and not command.report_only):
        raise FileExistsError("Use a new private report/candidate version; do not overwrite evidence")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PYTORCH_NVML_BASED_CUDA_CHECK="0", OMP_NUM_THREADS="2")
    sys.path[:0] = [str(CORE), str(ROOT)]
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    torch.set_num_threads(2)
    results, candidates = asyncio.run(validate(verify_candidate_dir,
                                               abstain_safe=command.abstain_safe,
                                               external_refine_relay=command.external_refine_relay,
                                               independent_refine_model=command.independent_refine_model,
                                               only_source_stem=command.only_source_stem))
    success = len(results) == (1 if command.only_source_stem else 10) and all(
        item["core_valid"] for item in results.values())
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU Core validation unexpectedly initialized CUDA")
    report = {"schema": "t8.modular-sampling.audio-refine-core-validation.v1",
              "status": "pass" if success else "fail", "cases": results,
              "queued": False, "browser_used": False, "cuda_initialized": False,
              "saved_candidate_verified": verify_candidate_dir is not None,
              "abstain_safe_tail": command.abstain_safe,
              "external_refine_relay": command.external_refine_relay,
              "independent_refine_model": command.independent_refine_model,
              "terminal_boundary": "VHS media terminal excluded; no graph execution or media decoding"}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if success and not command.report_only:
        candidate_dir.mkdir(parents=True)
        for source, frontend, api in candidates:
            stem = source.stem + ("_StageBound_RefineRelay_IndependentModel_EXP"
                                  if command.independent_refine_model else
                                  "_StageBound_RefineRelay_EXP" if command.external_refine_relay
                                  else "_StageBound_EXP")
            (candidate_dir / (stem + ".json")).write_text(
                json.dumps(frontend, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (candidate_dir / (stem + ".api.json")).write_text(
                json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": {
        name: {key: value for key, value in item.items() if key != "result"}
        for name, item in results.items()}}, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
