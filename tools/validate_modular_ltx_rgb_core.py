"""Validate the two S27 RGB/LTX candidate API paths in an isolated CPU Core.

Only an existing input-video filename replaces the old placeholder for schema
validation. No file is decoded, node is executed, prompt is queued or GPU used.
"""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_modular_audio_refine_workflows import split_api  # noqa: E402
from tools.build_modular_ltx_rgb_workflows import sources, split_frontend  # noqa: E402

TARGET = ROOT / "artifacts/development/modular-sampling-m4-ltx-rgb-20260923/candidate-v2"
REPORT = ROOT / "artifacts/development/modular-sampling-m4-ltx-rgb-20260923/core-validation-v1.json"
MODULES = ("nodes_custom_sampler.py", "nodes_video.py", "nodes_lt.py",
           "nodes_lt_upsampler.py", "nodes_hunyuan.py")


async def validate(verify_candidate_dir=None):
    import nodes
    import execution

    for name in MODULES:
        if not await nodes.load_custom_node(str(CORE / "comfy_extras" / name), module_parent="comfy_extras"):
            raise RuntimeError("Required Core node module failed: " + name)
    spec = importlib.util.spec_from_file_location(
        "_t8_modular_ltx_rgb_core", ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    for cls in await package.comfy_entrypoint().get_node_list():
        nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
    candidates = []
    needed = set()
    for path in sources():
        graph = split_frontend(json.loads(path.read_text(encoding="utf-8")))
        candidates.append((path, graph))
        needed.update(node["type"] for node in graph["nodes"])
    info = {}
    for kind in needed - {"MarkdownNote"}:
        cls = nodes.NODE_CLASS_MAPPINGS.get(kind)
        if cls is None:
            raise ValueError("Current Core lacks " + kind)
        info[kind] = cls.GET_NODE_INFO_V1() if hasattr(cls, "GET_NODE_INFO_V1") else cls.INPUT_TYPES()
    results = {}
    built = []
    for path, frontend in candidates:
        try:
            api = split_api(frontend, info)
            load = next(node for node in api.values() if node["class_type"] == "LoadVideo")
            if load["inputs"].get("file") != "replace_with_h3_draft_24fps.mp4":
                raise ValueError("Old Sol source video placeholder changed")
            # This validates the original graph's schema and dependencies, not
            # the absent placeholder media or the meaning of its pixels.
            load["inputs"]["file"] = "0.6.mp4"
            if verify_candidate_dir is not None:
                stem = path.stem + "_StageBound_EXP"
                stored_frontend = json.loads((verify_candidate_dir / (stem + ".json")).read_text(encoding="utf-8"))
                stored_api = json.loads((verify_candidate_dir / (stem + ".api.json")).read_text(encoding="utf-8"))
                if stored_frontend != frontend or stored_api != api:
                    raise ValueError("Saved LTX RGB candidate differs from current source and converter")
                api = stored_api
            validated = await execution.validate_prompt("ltx-rgb-" + path.stem, api, None)
            types = {str(node["id"]): node["type"] for node in frontend["nodes"]}
            output_types = {types[node_id] for node_id in validated[2]}
            audit_present = "MiniMaxH3LTXRGBStageAuditEXPT8" in output_types
            save_present = "SaveVideo" in output_types
            results[path.name] = {"core_valid": bool(validated[0]) and not bool(validated[3])
                                  and audit_present and save_present,
                                  "result": validated, "api_nodes": len(api),
                                  "frontend_nodes": len(frontend["nodes"]),
                                  "stage_audit_in_validated_outputs": audit_present,
                                  "save_video_in_validated_outputs": save_present}
            built.append((path, frontend, api))
        except (KeyError, TypeError, ValueError) as error:
            results[path.name] = {"core_valid": False, "conversion_error": str(error)}
    return results, built


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--candidate-dir", type=Path, default=TARGET)
    parser.add_argument("--verify-candidate-dir", type=Path)
    parser.add_argument("--report-only", action="store_true")
    options = parser.parse_args()
    report_path = options.report.resolve()
    candidate_dir = options.candidate_dir.resolve()
    verify_dir = options.verify_candidate_dir.resolve() if options.verify_candidate_dir else None
    if (not report_path.is_relative_to(ROOT / "artifacts") or
            not candidate_dir.is_relative_to(ROOT / "artifacts") or
            (verify_dir is not None and not verify_dir.is_relative_to(ROOT / "artifacts"))):
        parser.error("Private evidence must stay under project artifacts")
    if report_path.exists() or (candidate_dir.exists() and not options.report_only):
        raise FileExistsError("Use a new private report/candidate version")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PYTORCH_NVML_BASED_CUDA_CHECK="0", OMP_NUM_THREADS="2")
    sys.path[:0] = [str(CORE), str(ROOT)]
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    torch.set_num_threads(2)
    results, built = asyncio.run(validate(verify_dir))
    success = len(results) == 2 and all(row["core_valid"] for row in results.values())
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU Core validation unexpectedly initialized CUDA")
    report = {"schema": "t8.modular-sampling.ltx-rgb-core-validation.v1",
              "status": "pass" if success else "fail", "cases": results,
              "source_media_substituted_for_schema_only": "0.6.mp4",
              "queued": False, "decoded": False, "browser_used": False,
              "cuda_initialized": False, "saved_candidate_verified": verify_dir is not None}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if success and not options.report_only:
        candidate_dir.mkdir(parents=True)
        for path, frontend, api in built:
            stem = path.stem + "_StageBound_EXP"
            (candidate_dir / (stem + ".json")).write_text(
                json.dumps(frontend, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (candidate_dir / (stem + ".api.json")).write_text(
                json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": {
        name: {key: value for key, value in row.items() if key != "result"}
        for name, row in results.items()}}, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
