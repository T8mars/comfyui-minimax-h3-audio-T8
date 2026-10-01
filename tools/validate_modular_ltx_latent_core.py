"""Check saved learned-LATENT LTX candidate with current CPU Core schemas.

Validation does not load a checkpoint, model, VAE, CLIP or queue a prompt.
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
EVIDENCE = ROOT / "artifacts/development/modular-sampling-m4-ltx-latent-20260924"
STEM = "H3_Learned_Latent_to_LTX_Separate_Refiner_EXP"


async def validate(candidate_version="v1"):
    import nodes
    import execution
    from tools.build_modular_ltx_latent_workflow import build_prompt

    for name in ("nodes_custom_sampler.py", "nodes_video.py", "nodes_lt.py"):
        if not await nodes.load_custom_node(str(CORE / "comfy_extras" / name), module_parent="comfy_extras"):
            raise RuntimeError("Required Core node module failed: " + name)
    spec = importlib.util.spec_from_file_location("_t8_ltx_learned_validator", ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    for cls in await package.comfy_entrypoint().get_node_list():
        nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
    candidate = EVIDENCE / f"candidate-{candidate_version}"
    frontend = json.loads((candidate / (STEM + ".json")).read_text(encoding="utf8"))
    api = json.loads((candidate / (STEM + ".api.json")).read_text(encoding="utf8"))
    if api != build_prompt(certified_sample=candidate_version == "v3"):
        raise ValueError("Saved API differs from current learned LTX builder")
    if len(frontend["nodes"]) != len(api) + 1 or len(frontend["links"]) < 30:
        raise ValueError("Saved frontend is not the complete external sampler graph")
    checked = await execution.validate_prompt("ltx-learned-s27", api, None)
    output_ids = set(checked[2])
    success = bool(checked[0]) and not bool(checked[3]) and {"14", "21"} <= output_ids
    return {"valid": success, "api_nodes": len(api),
            "frontend_nodes": len(frontend["nodes"]),
            "frontend_links": len(frontend["links"]),
            "output_nodes": checked[2], "errors": checked[3]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("v1", "v3"), default="v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = (args.output or EVIDENCE / "core-validation-v1.json").resolve()
    if not target.is_relative_to(EVIDENCE.resolve()):
        parser.error("Use a private S27 evidence report path")
    if target.exists():
        raise FileExistsError("Use a new report version")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PYTORCH_NVML_BASED_CUDA_CHECK="0", OMP_NUM_THREADS="2")
    sys.path[:0] = [str(CORE), str(ROOT)]
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    torch.set_num_threads(2)
    case = asyncio.run(validate(args.candidate))
    success = case["valid"] and not torch.cuda.is_initialized()
    report = {"schema": "t8.modular-sampling.ltx-learned-core-validation.v1",
              "status": "pass" if success else "fail", "case": case,
              "candidate_version": args.candidate,
              "cuda_initialized": torch.cuda.is_initialized(), "queued": False,
              "checkpoint_loaded": False, "sampled": False, "media_decoded": False,
              "browser_used": False}
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
