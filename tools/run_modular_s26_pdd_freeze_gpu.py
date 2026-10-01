"""Run one saved S26 PDD freeze graph with the installed full Ref2VA/PDD assets.

The default is read-only preflight. ``--confirm-run`` starts one owned, isolated
Core, reduces only the copied graph's test canvas/length, and saves a private
native AV checkpoint. It does not alter candidates, old workflows or user Core.
This is a mechanical graph/asset check, not media or perceptual acceptance.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from safetensors import safe_open

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as pdd  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s26-pdd-saved-freeze-gpu.v1"
CANDIDATES = PROJECT / "artifacts/development/modular-sampling-m4-audio-refine-all-resume-20260924/candidate-v7"
ROUTES = {
    "pdd8": ("2026-08-29_H3_Audio_Refine_PDD_Ref2VA8_Advanced_EXP", "31", "6", 1),
    "pdd4plus4": ("2026-08-29_H3_Audio_Refine_PDD_Ref2VA_4Plus4_Advanced_EXP", "41", "7", 2),
}
METADATA_KEY = "t8_native_latent_checkpoint_json"


def _sha(path: Path) -> str:
    return shared._sha256_file(path)


def _candidate(route: str) -> Path:
    return CANDIDATES / ROUTES[route][0] / "freeze_video.api.json"


def build_probe_graph(route: str, *, width: int, height: int, frames: int) -> tuple[dict, str]:
    if route not in ROUTES or min(width, height, frames) < 1 or width % 32 or height % 32:
        raise ValueError("Use a known PDD route and positive 32-aligned canvas/frames")
    source = _candidate(route)
    original = json.loads(source.read_text(encoding="utf-8"))
    graph = deepcopy(original)
    _family, save_id, conditioning_id, sampler_count = ROUTES[route]
    if sum(node["class_type"] == "SamplerCustomAdvanced" for node in graph.values()) != sampler_count:
        raise ValueError("Saved PDD first-pass sampler count changed")
    if graph["8"]["class_type"] != "MiniMaxH3PDD8StepSetupT8Advanced":
        raise ValueError("Saved PDD graph no longer uses the real adapter loader")
    if graph["8"]["inputs"]["strength"] != 1.0 or graph["8"]["inputs"]["base_variant"] != "Ref2VA":
        raise ValueError("Saved PDD adapter contract changed")
    if graph[save_id]["class_type"] != "MiniMaxH3NativeLatentCheckpointSaveT8Advanced":
        raise ValueError("Saved PDD freeze output changed")
    final_sampler = "11" if route == "pdd8" else "19"
    if graph[save_id]["inputs"]["av_latent"] != [final_sampler, 0]:
        raise ValueError("Checkpoint does not receive the final video sampler")
    graph[conditioning_id]["inputs"].update(width=width, height=height, length=frames)
    if route == "pdd4plus4":
        if (graph["13"]["class_type"] != "MiniMaxH3LearnedLatentUpscaleT8Advanced"
                or graph["13"]["inputs"]["model_name"] !=
                "minimax_h3_latent_upscaler_3d_fp16.safetensors"):
            raise ValueError("Saved 4+4 graph no longer uses the trained 3D upscaler")
        graph["14"]["inputs"]["length"] = frames
    graph[save_id]["inputs"].update(
        confirm_save=True, filename_prefix=f"s26_{route}_frozen_firstpass")
    graph["200"] = {"class_type": "PreviewAny", "inputs": {"source": ["8", 3]}}
    graph["201"] = {"class_type": "PreviewAny", "inputs": {"source": [save_id, 5]}}
    return graph, _sha(source)


def _required_assets(comfy_root: Path, route: str, graph: dict) -> dict[str, Path]:
    models = comfy_root / "models"
    inputs = comfy_root / "input"
    result = {
        "base": models / "diffusion_models" / graph["1" if route == "pdd8" else "4"]["inputs"]["unet_name"],
        "pdd_adapter": models / "loras" / graph["8"]["inputs"]["pdd_lora_name"],
        "clip": models / "text_encoders" / graph["2" if route == "pdd8" else "3"]["inputs"]["clip_name"],
        "video_vae": models / "vae" / "minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": models / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
        "reference_image": inputs / graph["5" if route == "pdd8" else "6"]["inputs"]["image"],
    }
    if route == "pdd4plus4":
        result["trained_3d_upscaler"] = models / "latent_upscale_models" / graph["13"]["inputs"]["model_name"]
    return result


def preflight(args: argparse.Namespace, graph: dict, source_sha: str) -> dict:
    assets = _required_assets(args.comfy_root, args.route, graph)
    gpu = shared.gpu_memory_mib()
    checks = {
        "saved_candidate_exists_and_sha_bound": bool(source_sha),
        "all_required_assets_exist": all(path.is_file() for path in assets.values()),
        "isolated_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "gpu_available": bool(gpu.get("available")),
        "free_vram_gate": bool(gpu.get("available") and gpu["free_mib"] >= args.min_free_vram_mib),
    }
    return {"schema": SCHEMA + ".preflight", "route": args.route,
            "candidate_sha256": source_sha, "assets": {key: str(value) for key, value in assets.items()},
            "checks": checks, "gpu": gpu, "ready": all(checks.values())}


def _verify_checkpoint(run_root: Path, save_report: dict) -> dict:
    if save_report.get("status") != "SAVED_VERIFIED" or save_report.get("checkpoint_id") != "audio_refine_firstpass":
        raise ValueError("Native PDD freeze checkpoint was not verified")
    store = (run_root / "output" / "MiniMaxH3" / "latent_checkpoints").resolve()
    relative = save_report["checkpoint_path"]
    path = (store / relative).resolve()
    if not path.is_relative_to(store) or not path.is_file():
        raise ValueError("PDD freeze checkpoint escaped or is missing")
    if _sha(path) != save_report["file_sha256"].upper():
        raise ValueError("PDD freeze checkpoint file SHA differs from its Save receipt")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = json.loads(handle.metadata()[METADATA_KEY])
        names = sorted(handle.keys())
    if metadata.get("checkpoint_id") != "audio_refine_firstpass" or not names:
        raise ValueError("PDD checkpoint metadata or tensor payload is incomplete")
    return {"relative_path": relative, "file_sha256": _sha(path),
            "file_bytes": path.stat().st_size, "tensor_names": names,
            "embedded_manifest": metadata.get("manifest")}


def stage_execution_checks(route: str, phase: dict) -> dict[str, bool]:
    events = phase.get("events") or []
    progress = [str(event.get("node")) for event in events if event.get("type") == "progress"]
    executing = [str(event.get("node")) for event in events if event.get("type") == "executing"]
    if route == "pdd8":
        expected_progress = ["11"] * 8
        sequence = ["8", "11", "31"]
    elif route == "pdd4plus4":
        expected_progress = ["12"] * 4 + ["19"] * 4
        sequence = ["8", "12", "13", "16", "19", "41"]
    else:
        raise ValueError("Unknown saved PDD route")
    actual_sequence = [node for node in executing if node in sequence]
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        "exact_eight_sampler_progress_events": progress == expected_progress,
        "actual_pdd_upscaler_sampler_save_order": actual_sequence == sequence,
    }


def audit_existing(run_root: Path, route: str) -> dict:
    report = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    phase = json.loads((run_root / "phase.json").read_text(encoding="utf-8"))
    if report.get("route") != route or report.get("status") != \
            "saved_pdd_freeze_real_assets_mechanical_pass_not_quality_acceptance":
        raise ValueError("Only a completed, matching owned S26 PDD run can be audited")
    checks = stage_execution_checks(route, phase)
    checks["original_candidate_sha_unchanged"] = _sha(_candidate(route)) == report["source_sha256"]
    checks["previous_mechanical_checks_pass"] = all(report["checks"].values())
    result = {"schema": SCHEMA + ".stage-audit", "route": route,
              "run_root": str(run_root), "checks": checks,
              "status": "stage_execution_pass" if all(checks.values()) else "stage_execution_fail",
              "boundary": "Native Core progress and ordering, not independent pretrained AV quality."}
    output = run_root / "stage-audit.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing stage audit")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=tuple(ROUTES), required=True)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8862)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--server-start-timeout", type=float, default=240.0)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--audit-run-root", type=Path,
                        help="Offline stage audit of an owned run; writes a new stage-audit.json but starts no Core/GPU")
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    if args.audit_run_root is not None:
        result = audit_existing(args.audit_run_root.resolve(), args.route)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "stage_execution_pass" else 1
    graph, source_sha = build_probe_graph(args.route, width=args.width, height=args.height, frames=args.frames)
    readiness = preflight(args, graph, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run:
        return 0 if readiness["ready"] else 2
    if not readiness["ready"]:
        return 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "artifacts/development/modular-sampling-s26-pdd-freeze-20260924" / f"{run_id}-{args.route}"
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "route": args.route, "source_sha256": source_sha,
              "preflight": readiness, "test_dimensions": [args.width, args.height, args.frames],
              "run_root": str(run_root), "status": "started"}
    try:
        with shared.IsolatedServer(args, run_root, f"s26-{args.route}") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        report["terminal"] = (phase.get("terminal") or {}).get("type")
        if report["terminal"] != "execution_success":
            raise RuntimeError("Saved PDD freeze graph did not complete; inspect phase.json")
        setup = json.loads(pdd._phase_text(phase, "200"))
        save = json.loads(pdd._phase_text(phase, "201"))
        report["setup"] = setup
        report["checkpoint"] = _verify_checkpoint(run_root, save)
        report["checks"] = {
            "adapter_258_backbone_targets": setup.get("lora", {}).get("mapped_adapters") == 258,
            "pdd_eight_steps": setup.get("sampling", {}).get("nfe") == 8,
            "reference_variant": setup.get("adapter", {}).get("base_variant") == "Ref2VA",
            "native_checkpoint_verified": save.get("status") == "SAVED_VERIFIED",
            "candidate_unchanged": _sha(_candidate(args.route)) == source_sha,
            "server_stopped": not shared.port_is_listening("127.0.0.1", args.port),
        }
        if not all(report["checks"].values()):
            raise ValueError("PDD saved-freeze evidence failed mechanical contract checks")
        report["status"] = "saved_pdd_freeze_real_assets_mechanical_pass_not_quality_acceptance"
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
