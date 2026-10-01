"""Validate the saved S28 split-stage candidates in an isolated CPU Core.

No prepared bundle is opened and no worker, GPU or prompt queue is started.
"""

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parents[1]
TARGET = ROOT / "artifacts/development/modular-sampling-m4-prepared-ltx-20260923"


async def validate():
    import execution
    import nodes
    from tools.build_modular_prepared_ltx_workflows import build_prompt, build_workflow

    spec = importlib.util.spec_from_file_location("_t8_modular_prepared_ltx_core", ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    classes = await package.comfy_entrypoint().get_node_list()
    for cls in classes:
        nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
    needed = ("MiniMaxH3PreparedGenerationBundleEXPT8", "MiniMaxH3PreparedLTXGenerateEXPT8",
              "MiniMaxH3PreparedLTXLoadGenerationEXPT8", "MiniMaxH3PreparedLTXDecodeEXPT8")
    info = {name: nodes.NODE_CLASS_MAPPINGS[name].GET_NODE_INFO_V1() for name in needed}
    for value in info.values():
        value["cnr_id"] = "minimax-h3-audio-T8"
    result = {}
    candidate = TARGET / "candidate-v1"
    for resume, stem in ((False, "Prepared_LTX_Generate_Decode_EXP"),
                         (True, "Prepared_LTX_DecodeOnly_EXP")):
        api = json.loads((candidate / (stem + ".api.json")).read_text(encoding="utf8"))
        frontend = json.loads((candidate / (stem + ".json")).read_text(encoding="utf8"))
        if api != build_prompt(resume=resume) or frontend != build_workflow(info, resume=resume):
            raise ValueError("Saved S28 candidate differs from current builder or node schema")
        checked = await execution.validate_prompt("prepared-ltx-" + stem, api, None)
        result[stem] = {"valid": bool(checked[0]) and not bool(checked[3]) and "3" in checked[2],
                        "api_nodes": len(api), "frontend_nodes": len(frontend["nodes"]),
                        "output_nodes": checked[2], "errors": checked[3]}
    return result


def main():
    report_path = TARGET / "core-validation-v1.json"
    if report_path.exists():
        raise FileExistsError("Use a new report version; do not overwrite evidence")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PYTORCH_NVML_BASED_CUDA_CHECK="0", OMP_NUM_THREADS="2")
    sys.path[:0] = [str(CORE), str(ROOT)]
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    torch.set_num_threads(2)
    cases = asyncio.run(validate())
    success = len(cases) == 2 and all(case["valid"] for case in cases.values()) and not torch.cuda.is_initialized()
    report = {"schema": "t8.modular-sampling.prepared-ltx-core-validation.v1",
              "status": "pass" if success else "fail", "cases": cases,
              "cuda_initialized": torch.cuda.is_initialized(), "queued": False,
              "bundle_opened": False, "worker_run": False, "browser_used": False}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
