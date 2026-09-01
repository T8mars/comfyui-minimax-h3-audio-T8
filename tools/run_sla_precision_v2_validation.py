#!/usr/bin/env python3
"""Run one guarded 124-frame SLA Precision V2 FP8 dialogue validation."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_sla_profile_router_probe as prior  # noqa: E402


SCHEMA = "t8.minimax_h3.sla_precision_v2.real_validation.v1"
WIDTH = prior.WIDTH
HEIGHT = prior.HEIGHT
FRAME_COUNT = prior.FRAME_COUNT
STEPS = 8
SHIFT_VIDEO = 12.0
SHIFT_AUDIO = 3.0
SEED = prior.SEED
FP8_BASE = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
SLA_LORA = prior.SLA_LORA_NAME
LOADER_NODE = "MiniMaxH3SLADynamicLoRABypassV2T8Advanced"
ATTENTION_NODE = "MiniMaxH3SLAPrecisionV2T8Advanced"
AUDIT_NODE = "MiniMaxH3SLAPrecisionV2AuditT8Advanced"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _required_paths(args: argparse.Namespace) -> dict[str, Path]:
    models = args.comfy_root / "models"
    project = args.project_root
    return {
        "comfy_main": args.comfy_root / "main.py",
        "python": args.python,
        "project": project,
        "vhs": args.comfy_root / "custom_nodes" / "ComfyUI-VideoHelperSuite",
        "source_prompt_png": prior._source_prompt_path(project),
        "first_frame": args.comfy_root / "input" / "codex_prompt_relay_fl2va_first.png",
        "base": models / "diffusion_models" / FP8_BASE,
        "clip": models / "text_encoders" / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae": models / "vae" / "minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": models / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
        "sla_lora": models / "loras" / SLA_LORA,
        "vendor_block_map": project / "sla_precision_v2_vendor" / "block_map.py",
        "vendor_kernel": project / "sla_precision_v2_vendor" / "kernel.py",
        "vendor_patch": project / "sla_precision_v2_vendor" / "patch.py",
    }


def _load_prompt(
    source_png: Path, run_id: str, route: str = "precision_v2"
) -> dict[str, Any]:
    prompt = prior._load_exact_prompt(
        source_png,
        run_id,
        prior.CONSUMER_PROFILE,
        transition=False,
    )
    prompt["4"]["inputs"].update(
        {"unet_name": FP8_BASE, "weight_dtype": "default"}
    )
    prompt["8"]["inputs"].update(
        {"steps": STEPS, "shift_video": SHIFT_VIDEO, "shift_audio": SHIFT_AUDIO}
    )
    prompt["9"] = {
        "class_type": LOADER_NODE,
        "inputs": {"lora_name": SLA_LORA, "model": ["8", 0]},
    }
    if route == "precision_v2":
        prompt["16"] = {
            "class_type": ATTENTION_NODE,
            "inputs": {
                "schedule_policy": "recommended_8nfe_12v_3a",
                "sparsity_ratio": 0.90,
                "block_size": "32",
                "min_seq_len": 8192,
                "dense_last_steps": 1,
                "protect_audio": True,
                "dense_steps": "0",
                "dense_backend": "comfy_kitchen",
                "disable_fp16_accum": True,
                "stabilize_motion": False,
                "reference_protection": False,
                "model": ["9", 0],
                "sigmas": ["8", 2],
            },
        }
        prompt["11"]["inputs"]["model"] = ["16", 0]
        prompt["13"] = {
            "class_type": AUDIT_NODE,
            "inputs": {"av_latent": ["12", 0], "runtime": ["16", 1]},
        }
        prompt["14"]["inputs"]["av_latent"] = ["13", 0]
        suffix = "precision_v2_32block_firstlastdense"
    elif route == "dense_control":
        prompt.pop("16", None)
        prompt.pop("13", None)
        prompt["11"]["inputs"]["model"] = ["9", 0]
        prompt["14"]["inputs"]["av_latent"] = ["12", 0]
        suffix = "dense_control_xformers"
    else:
        raise ValueError(f"unknown validation route {route!r}")
    prompt["15"]["inputs"]["filename_prefix"] = (
        f"MiniMaxH3_SLA_PrecisionV2/{run_id}_fp8_736x416_124f_"
        f"8nfe_12v3a_{suffix}_dialogue"
    )
    return prompt


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_strings(child)


def _extract_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    matches = []
    for raw in _walk_strings(payload):
        if "precision_v2_mechanically_verified" not in raw:
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema", "").startswith(
            "t8.minimax_h3.sla_precision_v2"
        ):
            matches.append(value)
    if not matches:
        raise ValueError("history did not contain SLA Precision V2 audit JSON")
    return matches[-1]


def _audit_checks(audit: Mapping[str, Any]) -> dict[str, bool]:
    config = dict(audit.get("config") or {})
    attention = dict(config.get("attention") or {})
    observed = dict(audit.get("observed") or {})
    per_step = {
        int(index): dict(record)
        for index, record in dict(observed.get("per_logical_step") or {}).items()
    }
    checks = dict(audit.get("checks") or {})
    return {
        "audit_status_pass": audit.get("status") == "precision_v2_mechanically_verified",
        "all_internal_checks_pass": bool(checks) and all(bool(value) for value in checks.values()),
        "eight_logical_nfe": int(observed.get("n_steps") or 0) == STEPS,
        "six_sparse_forwards_x_50": int(observed.get("sparse_calls") or 0) == 300,
        "first_and_last_dense_x_50": int(observed.get("dense_calls") or 0) >= 100,
        "block_size_32": attention.get("block_size_q") == 32
        and attention.get("block_size_k") == 32,
        "sparse_steps_1_through_6": attention.get("sparse_step_indices")
        == [1, 2, 3, 4, 5, 6],
        "dense_steps_0_and_7": attention.get("dense_step_indices") == [0, 7],
        "per_step_dense_0_and_7_exactly_50": all(
            int(per_step.get(index, {}).get("dense_calls") or 0) == 50
            and int(per_step.get(index, {}).get("sparse_calls") or 0) == 0
            and per_step.get(index, {}).get("expected_dense") is True
            for index in (0, 7)
        ),
        "per_step_sparse_1_through_6_exactly_50": all(
            int(per_step.get(index, {}).get("sparse_calls") or 0) == 50
            and int(per_step.get(index, {}).get("dense_calls") or 0) == 0
            and int(per_step.get(index, {}).get("kernel_fallbacks") or 0) == 0
            and per_step.get(index, {}).get("expected_dense") is False
            for index in range(1, 7)
        ),
        "protect_audio_enabled": attention.get("protect_audio") is True,
        "protected_blocks_observed": int(observed.get("pinned_key_blocks") or 0) > 0,
        "no_kernel_failure": observed.get("first_kernel_failure") is None,
        "direct_triton_fp32_route": attention.get("router_precision")
        == "fp32_pool_and_fp32_scores"
        and attention.get("sparse_kernel") == "direct_triton_fp32_online_softmax",
    }


def _gate_result(
    route_checks: Mapping[str, bool],
    media_checks: Mapping[str, bool],
    resource_checks: Mapping[str, bool],
) -> dict[str, Any]:
    mechanical_av_pass = (
        bool(route_checks)
        and all(bool(value) for value in route_checks.values())
        and bool(media_checks)
        and all(bool(value) for value in media_checks.values())
    )
    resource_pass = bool(resource_checks) and all(
        bool(value) for value in resource_checks.values()
    )
    if not mechanical_av_pass:
        status = "FAIL_MECHANICAL_AV"
    elif not resource_pass:
        status = "MECHANICAL_AV_PASS_RESOURCE_GATE_FAIL_HUMAN_REVIEW_PENDING"
    else:
        status = "MECHANICAL_AV_AND_RESOURCE_PASS_HUMAN_REVIEW_PENDING"
    return {
        "status": status,
        "mechanical_av_pass": mechanical_av_pass,
        "resource_pass": resource_pass,
        "human_review_pass": False,
        "human_review_pending": mechanical_av_pass,
        "exit_success": mechanical_av_pass and resource_pass,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--comfy-root", type=Path, default=Path(r"F:\AI-T8-video-onekey\ComfyUI")
    )
    parser.add_argument(
        "--python", type=Path, default=Path(r"F:\AI-T8-video-onekey\python\python.exe")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--min-free-vram-mib", type=int, default=14_500)
    parser.add_argument(
        "--route",
        choices=("precision_v2", "dense_control"),
        default="precision_v2",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/sla-precision-v2-real-validation-20260902"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.project_root = args.project_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    args.artifact_root = (
        args.artifact_root
        if args.artifact_root.is_absolute()
        else args.project_root / args.artifact_root
    ).resolve()
    paths = _required_paths(args)
    missing = [str(path) for path in paths.values() if not path.exists()]
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    gpu = prior.shared.gpu_memory_mib()
    checks = {
        "required_paths_present": not missing,
        "ffmpeg_present": bool(ffmpeg),
        "ffprobe_present": bool(ffprobe),
        "normal_user_service_8188_stopped": not prior.shared.port_is_listening(
            args.host, 8188
        ),
        "isolated_port_free": not prior.shared.port_is_listening(args.host, args.port),
        "gpu_query_available": bool(gpu.get("available")),
        "free_vram_at_least_14500_mib": bool(
            gpu.get("available")
            and int(gpu.get("free_mib") or 0) >= int(args.min_free_vram_mib)
        ),
    }
    preflight = {
        "schema": f"{SCHEMA}.preflight",
        "created_at": _utc_now(),
        "checks": checks,
        "missing": missing,
        "gpu": gpu,
        "ready": all(checks.values()),
    }
    print(_json(preflight), flush=True)
    if not preflight["ready"]:
        return 2

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = args.artifact_root / run_id
    prompt = _load_prompt(paths["source_prompt_png"], run_id, args.route)
    assets = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": prior.shared._sha256_file(path)
            if name in {
                "source_prompt_png",
                "first_frame",
                "vendor_block_map",
                "vendor_kernel",
                "vendor_patch",
            }
            else None,
        }
        for name, path in paths.items()
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "run_id": run_id,
        "contract": {
            "base": FP8_BASE,
            "lora": SLA_LORA,
            "lora_application": "dynamic_model_only_bypass",
            "width": WIDTH,
            "height": HEIGHT,
            "frame_count": FRAME_COUNT,
            "steps": STEPS,
            "shift_video": SHIFT_VIDEO,
            "shift_audio": SHIFT_AUDIO,
            "seed": SEED,
            "same_first_last_frame": True,
            "clear_mandarin_dialogue": True,
            "route": args.route,
            "single_render_only": True,
        },
        "preflight": preflight,
        "assets": assets,
    }

    monitor = prior.clipprobe.GpuPeakMonitor(interval_seconds=0.25)
    phase = None
    try:
        with prior.shared.IsolatedServer(args, run_root, "sla_precision_v2"):
            monitor.start()
            phase = asyncio.run(
                prior._submit_prompt_capture(
                    server=f"http://{args.host}:{args.port}",
                    prompt=prompt,
                    timeout_seconds=args.timeout_seconds,
                )
            )
    finally:
        report["gpu_monitor"] = monitor.stop()

    report["phase"] = phase
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "phase.json").write_text(
        json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not phase or phase.get("terminal", {}).get("type") != "execution_success":
        report["status"] = "FAIL_EXECUTION"
        (run_root / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(_json(report), flush=True)
        return 1

    audit_capture_error = None
    if args.route == "precision_v2":
        try:
            audit = _extract_audit(
                {
                    "history": phase.get("history"),
                    "executed_outputs": phase.get("executed_outputs"),
                }
            )
        except (TypeError, ValueError) as error:
            audit = None
            audit_capture_error = f"{type(error).__name__}: {error}"
    else:
        audit = None
    output_dir = run_root / "output" / "MiniMaxH3_SLA_PrecisionV2"
    video = prior.shared._latest_file(output_dir, f"{run_id}*-audio.mp4")
    media = prior.shared.media_report(video, ffmpeg=str(ffmpeg), ffprobe=str(ffprobe))
    contact = run_root / "contact_0s_to_5s.png"
    prior._contact_sheet(video, contact, str(ffmpeg))
    report.update(
        {
            "audit": audit,
            "audit_capture_error": audit_capture_error,
            "audit_checks": _audit_checks(audit) if audit is not None else {},
            "dense_control_topology_checks": (
                {
                    "precision_attention_node_absent": "16" not in prompt,
                    "precision_audit_node_absent": "13" not in prompt,
                    "guider_uses_dynamic_lora_model_directly": prompt["11"]["inputs"][
                        "model"
                    ]
                    == ["9", 0],
                }
                if args.route == "dense_control"
                else {}
            ),
            "media": media,
            "media_checks": prior._media_checks(media),
            "output_video": str(video.resolve()),
            "contact_sheet": str(contact.resolve()),
        }
    )
    report["resource_checks"] = {
        "minimum_free_vram_at_least_512_mib": int(
            report["gpu_monitor"].get("minimum_free_mib") or 0
        )
        >= 512
    }
    route_checks = (
        report["audit_checks"]
        if args.route == "precision_v2"
        else report["dense_control_topology_checks"]
    )
    gate_result = _gate_result(
        route_checks,
        report["media_checks"],
        report["resource_checks"],
    )
    report["gates"] = gate_result
    report["status"] = gate_result["status"]
    report["quality_claim"] = (
        "Mechanical evidence only. Full-speed face/motion review, clear-dialogue "
        "listening and lip-sync inspection remain required."
    )
    (run_root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(_json(report), flush=True)
    return 0 if gate_result["exit_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
