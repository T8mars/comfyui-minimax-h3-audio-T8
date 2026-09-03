#!/usr/bin/env python3
"""Audit pinned OpenVDN assets and optionally compose them onto a real Comfy H3 base."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time
import types

import psutil


SCHEMA = "t8.minimax_h3.openvdn.runtime_audit.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--comfy-root", type=Path, default=Path(r"F:\AI-T8-video-onekey\ComfyUI")
    )
    parser.add_argument(
        "--base-model", default="minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    )
    parser.add_argument("--vdn-root", default="OpenVDN/vdn-minimax-h3")
    parser.add_argument(
        "--stage",
        choices=("stage_dmd_8nfe", "stage_b_50nfe"),
        default="stage_dmd_8nfe",
    )
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/openvdn-h3-runtime-audit-20260903/report.json"),
    )
    return parser


def _install_project_package(project_root: Path) -> None:
    package = types.ModuleType("h3_audio_t8_pkg")
    package.__path__ = [str(project_root)]
    package.__package__ = "h3_audio_t8_pkg"
    sys.modules["h3_audio_t8_pkg"] = package


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    comfy_root = args.comfy_root.resolve()
    output = args.output if args.output.is_absolute() else project_root / args.output
    base_path = comfy_root / "models" / "diffusion_models" / args.base_model

    sys.path.insert(0, str(comfy_root))
    sys.argv = ["openvdn-h3-runtime-audit", "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    import comfy.model_management as model_management
    import comfy.sd

    _install_project_package(project_root)
    from h3_audio_t8_pkg import vdn_h3_advanced as vdn

    process = psutil.Process()
    started = time.monotonic()
    report = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "stage": args.stage,
        "base_model": str(base_path),
        "assets_only": bool(args.assets_only),
        "rss_before_bytes": process.memory_info().rss,
    }
    model = composed = None
    try:
        root = vdn.resolve_vdn_root(args.vdn_root)
        assets, asset_errors = vdn._asset_report(
            root, args.stage, verify_hashes=not args.skip_hashes
        )
        report["asset_report"] = assets
        report["asset_errors"] = asset_errors
        if asset_errors:
            raise RuntimeError("; ".join(asset_errors))
        if not args.assets_only:
            if not base_path.is_file():
                raise FileNotFoundError(base_path)
            model = comfy.sd.load_diffusion_model(str(base_path), model_options={})
            report["rss_after_base_bytes"] = process.memory_info().rss
            ready, compatibility = vdn.audit_vdn_runtime(
                model,
                root,
                args.stage,
                verify_hashes=False,
                allow_structural_base=True,
            )
            report["compatibility"] = compatibility
            if not ready:
                raise RuntimeError("real-base compatibility audit blocked")
            composed, receipt_json = vdn.compose_vdn_model(
                model,
                root,
                args.stage,
                verify_hashes=False,
                allow_structural_base=True,
            )
            receipt = json.loads(receipt_json)
            report["composition"] = receipt
            report["rss_after_composition_bytes"] = process.memory_info().rss
            report["checks"] = {
                "all_800_branch_tensors_loaded": receipt["branch"]["tensor_count"]
                == 800,
                "all_adapters_fully_applied": all(
                    item["patch_targets"] == item["applied_targets"]
                    for item in receipt["adapters"]
                ),
                "all_50_blocks_replaced": receipt["main_block_count"] == 50,
                "additional_model_lifecycle": receipt["additional_model_lifecycle"]
                is True,
                "plain_t2va_fail_closed": receipt["task_scope"]
                == "plain_t2va_only_fail_closed",
            }
            if not all(report["checks"].values()):
                raise RuntimeError("one or more composition checks failed")
        report["status"] = "PASS"
    except Exception as exc:  # noqa: BLE001 - the durable report must retain failures
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["rss_final_bytes"] = process.memory_info().rss
        _write_report(output.resolve(), report)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        del composed, model
        gc.collect()
        model_management.soft_empty_cache()
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
