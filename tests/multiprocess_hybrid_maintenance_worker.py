from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
PACKAGE_NAME = "h3_audio_t8_pkg"


def _load_package() -> None:
    sys.path.insert(0, str(COMFY_ROOT))
    if PACKAGE_NAME in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ready", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--hold-seconds", type=float, default=60.0)
    args = parser.parse_args()
    _load_package()

    from h3_audio_t8_pkg import hybrid_model as hybrid

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    hybrid.PROFILE_SPECS = contract["profile_specs"]
    hybrid.CURVE_SHAPE = tuple(contract["curve_shape"])
    hybrid.MODALITY_ROWS = int(contract["modality_rows"])
    hybrid.MODALITY_INDEX = contract["modality_index"]
    hybrid.KNOWN_QUALITY_BASE_SHA256 = contract["base_sha256"]
    hybrid.KNOWN_REFERENCE_OVERLAY_SHA256 = contract["overlay_sha256"]
    hybrid.KNOWN_QUALITY_CURVE_SHA256 = contract["base_curve_sha256"]
    hybrid.KNOWN_REFERENCE_CURVE_SHA256 = contract["overlay_curve_sha256"]

    output_root = Path(args.output_root).resolve()
    artifact_path = hybrid.artifact_path_for_plan(plan, output_root).resolve()
    sidecar_path = artifact_path.with_suffix(artifact_path.suffix + ".json")
    ready = Path(args.ready)
    real_replace = hybrid.os.replace

    def replace_then_pause(source, destination):
        source_path = Path(source).resolve()
        destination_path = Path(destination).resolve()
        if source_path == sidecar_path and "_recycle" in destination_path.parts:
            ready.write_text(str(os.getpid()), encoding="utf-8")
            time.sleep(args.hold_seconds)
        return real_replace(source, destination)

    hybrid.os.replace = replace_then_pause
    hybrid.maintain_hybrid_artifact(
        plan,
        output_root,
        "quarantine_artifact_exp",
        True,
        args.epoch,
    )
    raise RuntimeError("parent did not terminate hybrid maintenance worker")


if __name__ == "__main__":
    raise SystemExit(main())
