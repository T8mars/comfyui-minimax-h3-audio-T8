#!/usr/bin/env python3
"""Run one CPU-only real-model loader probe for the INT8 SLA bypass profile.

This validates real 208-target mapping and ComfyUI bypass injection without
sampling, CLIP, VAE, media decode, repeated runs or a GPU model load. It is not
a visual-quality test.
"""

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


SCHEMA = "t8.minimax_h3.sla_int8_bypass_loader_probe.v1"
PROFILE = "sla_4step_int8_bypass_exp"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=Path(r"F:\AI-T8-video-onekey\ComfyUI"),
    )
    parser.add_argument(
        "--base-model",
        default="minimax_h3_fl2va_int8_convrot.safetensors",
    )
    parser.add_argument(
        "--sla-lora",
        default=(
            "minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/sla-int8-bypass-loader-probe-20260826/report.json"),
    )
    return parser


def _install_project_package(project_root: Path) -> None:
    package = types.ModuleType("h3_audio_t8_pkg")
    package.__path__ = [str(project_root)]
    package.__package__ = "h3_audio_t8_pkg"
    sys.modules["h3_audio_t8_pkg"] = package


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    comfy_root = args.comfy_root.resolve()
    output = args.output if args.output.is_absolute() else project_root / args.output
    output = output.resolve()
    base_path = comfy_root / "models" / "diffusion_models" / args.base_model
    lora_path = comfy_root / "models" / "loras" / args.sla_lora
    missing = [str(path) for path in (base_path, lora_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing SLA loader-probe assets: " + ", ".join(missing))

    # ComfyUI ignores process argv unless argument parsing is explicitly enabled.
    # Force CPU before importing model_management so this probe cannot load the DiT
    # onto CUDA. Torch still sees CUDA for the already-audited sparse-kernel contract.
    sys.path.insert(0, str(comfy_root))
    sys.argv = ["sla-int8-bypass-loader-probe", "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    import comfy.model_management as model_management
    import comfy.sd

    _install_project_package(project_root)
    from h3_audio_t8_pkg.sampling import native_flow_sigmas
    from h3_audio_t8_pkg.sla_profile_router_advanced import (
        build_turbo_sla_profile_model,
    )

    process = psutil.Process()
    started = time.monotonic()
    before = process.memory_info().rss
    report = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "status": "RUNNING",
        "scope": "one CPU-only real-model loader and 208-hook injection; no sampling",
        "assets": {
            "base": str(base_path),
            "base_bytes": base_path.stat().st_size,
            "sla_lora": str(lora_path),
            "sla_lora_bytes": lora_path.stat().st_size,
        },
        "cpu_mode": str(model_management.cpu_state),
        "load_device": str(model_management.get_torch_device()),
        "offload_device": str(model_management.unet_offload_device()),
        "rss_before_bytes": before,
    }
    model = patched = runtime = None
    try:
        if not model_management.is_device_cpu(model_management.get_torch_device()):
            raise RuntimeError("loader probe failed to enter ComfyUI CPU mode")
        model = comfy.sd.load_diffusion_model(str(base_path), model_options={})
        transformer = dict(model.model_options.get("transformer_options", {}))
        transformer["minimax_h3_sigma_shift_video"] = 6.0
        transformer["minimax_h3_sigma_shift_audio"] = 3.0
        model.model_options["transformer_options"] = transformer
        after_base = process.memory_info().rss

        patched, runtime, profile_json = build_turbo_sla_profile_model(
            model,
            native_flow_sigmas(4, 6.0),
            turbo_lora_path="",
            sla_lora_path=str(lora_path),
            profile=PROFILE,
            base_policy="auto_detect_exp",
            max_router_workspace_mib=512,
        )
        after_bypass = process.memory_info().rss
        profile_report = json.loads(profile_json)
        loader_report = profile_report["sla_loader"]
        base_contract = loader_report["core_contract"]["base"]
        lora_contract = loader_report["lora_contract"]
        target_contract = base_contract["lora_target_quantization"]
        injections = dict(getattr(patched, "injections", {}) or {})
        patches = dict(getattr(patched, "patches", {}) or {})
        checks = {
            "cpu_load_device": model_management.is_device_cpu(
                model_management.get_torch_device()
            ),
            "profile_ready": profile_report["status"]
            == "sla_int8_bypass_percent_window_ready_for_runtime_audit",
            "base_family_exact": profile_report["base_family"]
            == "comfyui_int8_convrot_bypass_experiment",
            "main_200_int8_convrot": target_contract
            == {
                "main_target_count": 200,
                "main_int8_convrot_count": 200,
                "main_unquantized_count": 0,
                "token_refiner_target_count": 8,
                "token_refiner_int8_convrot_count": 0,
                "token_refiner_unquantized_count": 8,
            },
            "all_208_lora_targets_mapped": int(lora_contract["mapped_patch_count"])
            == 208,
            "all_208_bypass_hooks_created": int(lora_contract["bypass_hook_count"])
            == 208,
            "dynamic_model_only_application": lora_contract["application_mode"]
            == "comfyui_bypass_model_only",
            "base_weight_mutation_false": lora_contract["base_weight_mutation"]
            is False,
            "no_standard_weight_patches": not patches,
            "bypass_injection_registered": "bypass_lora" in injections,
            "runtime_profile_bound": runtime.config["mode"]
            == "apply_lightx2v_sla_upstream_exact_exp",
            "default_percent_window_bound": runtime.config[
                "sparse_percent_window"
            ]
            == {
                "start_percent": 0.15,
                "end_percent": 0.9,
                "full_range": False,
                "boundary_semantics": "inclusive_current_model_forward_sigma",
            },
        }
        report.update(
            {
                "status": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks,
                "profile_report": profile_report,
                "rss_after_base_bytes": after_base,
                "rss_after_bypass_bytes": after_bypass,
                "rss_base_delta_bytes": after_base - before,
                "rss_bypass_delta_bytes": after_bypass - after_base,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "quality_claim": (
                    "none; loader mechanics only, full-duration render and human review pending"
                ),
            }
        )
    except BaseException as error:
        report.update(
            {
                "status": "FAIL_EXCEPTION",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
    finally:
        del runtime, patched, model
        gc.collect()
        report["rss_after_release_bytes"] = process.memory_info().rss
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
