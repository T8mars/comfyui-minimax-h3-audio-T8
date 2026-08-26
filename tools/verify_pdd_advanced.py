#!/usr/bin/env python3
"""Serial, CPU/meta integration verifier for the T8 MiniMax-H3 PDD node.

This deliberately does not render or allocate the H3 base on CUDA.  It proves
that a converted adapter maps to all 258 current-Comfy modules, creates all
dynamic bypass hooks, installs the dynamic PDD final head and returns the exact
8-step native sampler contract.  Run FL2VA and Ref2VA one at a time.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys

import torch


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMFY_ROOT = DEFAULT_PROJECT_ROOT.parents[1]


def load_project(project_root: Path):
    name = "h3_audio_t8_pdd_validation"
    spec = importlib.util.spec_from_file_location(
        name,
        project_root / "__init__.py",
        submodule_search_locations=[str(project_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load project package: {project_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("--variant", choices=("FL2VA", "Ref2VA"), required=True)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.adapter, args.comfy_root, args.project_root):
        if not path.exists():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(args.comfy_root.resolve()))

    import comfy.model_patcher
    import comfy.nested_tensor
    import comfy.supported_models

    load_project(args.project_root.resolve())
    pdd = sys.modules["h3_audio_t8_pdd_validation.pdd_advanced"]

    config = comfy.supported_models.MiniMaxH3({"image_model": "minimax_h3"})
    with torch.device("meta"):
        base = config.get_model({}, device=torch.device("meta"))
    patcher = comfy.model_patcher.ModelPatcher(
        base, torch.device("cpu"), torch.device("cpu")
    )
    video = torch.zeros((1, 24, 2, 2, 2))
    audio = torch.zeros((1, 32, 2, 8))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    patched = None
    try:
        patched, sampler, sigmas, report_json = pdd.build_pdd_8step_setup(
            patcher,
            latent,
            args.adapter,
            base_variant=args.variant,
            strength=1.0,
        )
        report = json.loads(report_json)
        final = patched.get_model_object("diffusion_model.final_layer")
        wrappers = patched.get_wrappers("diffusion_model", pdd.PDD_WRAPPER_KEY)
        injections = patched.get_injections(pdd.PDD_INJECTION_KEY)
        assert type(final).__name__ == "PDDHeadFinalLayer"
        assert len(wrappers) == 1
        assert len(injections) == 1
        assert report["lora"]["mapped_adapters"] == 258
        assert report["lora"]["bypass_hooks"] == 258
        assert report["lora"]["rank_counts"] == {"64": 206, "192": 52}
        assert report["sampling"]["block_indices"] == list(range(8))
        assert len(sigmas) == 9
        assert sampler is not None
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        print(
            f"T8_PDD_META_INTEGRATION=PASS variant={args.variant} "
            "dynamic_hooks=258 blocks=0..7",
            flush=True,
        )
    finally:
        del patched
        del patcher
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
