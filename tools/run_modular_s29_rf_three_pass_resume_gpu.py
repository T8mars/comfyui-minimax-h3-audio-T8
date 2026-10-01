"""Cold RF third-stage RESTART from a verified real three-pass BASE.

Default checks the exact successful owned source and current resources only.
``--confirm-run`` copies its RF BASE manifest/state to a new isolated Core;
LOW, learned 3D and RF BASE must not execute in that fresh process.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_modular_s29_rf_three_pass_gpu as full  # noqa: E402
import run_modular_s29_rf_resume_gpu as two_stage_resume  # noqa: E402
import run_modular_s04_pdd_gpu as common  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as core_capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s29-rf-three-pass-real-cold-restart-gpu.v1"
FULL_PASS = "real_rf_three_pass_small_media_mechanical_pass_not_quality_acceptance"


def verified_base(full_root: Path) -> tuple[dict, dict, Path, Path]:
    report = json.loads((full_root / "report.json").read_text(encoding="utf-8"))
    if report.get("schema") != full.SCHEMA or report.get("status") != FULL_PASS or \
            report.get("run_root") != str(full_root) or \
            report.get("source_candidate_sha256") != shared._sha256_file(full.candidate()) or \
            report.get("test_dimensions") != [128, 64, 22] or \
            not all(report.get("checks", {}).values()):
        raise ValueError("Source must be the exact successful owned RF three-pass small graph")
    phase = json.loads((full_root / "phase.json").read_text(encoding="utf-8"))
    if not all(full.stage_checks(phase).values()) or \
            not all(full.media_checks(full_root, 128, 64, 22).values()):
        raise ValueError("Source RF three-pass execution or media changed after its report")
    receipts = {name: report.get(f"{name}_save") for name in ("low", "base", "restart")}
    if any(not isinstance(value, dict) or not common._artifact_sha_matches(full_root, value)
           for value in receipts.values()):
        raise ValueError("Source RF three-pass stage receipt is missing or changed")
    receipt = receipts["base"]
    store = (full_root / "output/MiniMaxH3/stage_artifacts").resolve()
    manifest = (store / receipt["path"]).resolve()
    if not manifest.is_relative_to(store) or manifest.name != "manifest.json":
        raise ValueError("Source RF BASE path escaped its owned stage store")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema") != "t8.modular-sampling.frozen-stage.v1" or \
            data.get("portable_identity") is not True or \
            data.get("stage_context", {}).get("stage") != "rf_base" or \
            data.get("state_file") != "state.safetensors":
        raise ValueError("Source RF BASE is not an exact portable third-stage anchor")
    state = manifest.parent / "state.safetensors"
    if not state.is_file() or state.stat().st_size != data["state_bytes"] or \
            shared._sha256_file(state) != data["state_sha256"].upper():
        raise ValueError("Source RF BASE tensor bytes changed")
    return report, receipt, manifest, state


def build_resume_graph(receipt: dict, *, frames: int = 22) -> tuple[dict, str]:
    if frames != 22:
        raise ValueError("Current cold RF three-pass probe is bound to the 22-frame full graph")
    source = full.candidate("resume_effects")
    source_sha = shared._sha256_file(source)
    graph = deepcopy(json.loads(source.read_text(encoding="utf-8")))
    required = {"93": "UNETLoader", "94": "MiniMaxH3LoRACompatibilityLoaderT8Advanced",
                "95": "MiniMaxH3PromptRelayConditioningT8Advanced",
                "96": "MiniMaxH3RFRestartStageSetupEXPT8", "99": "MiniMaxH3StageSamplerEXPT8",
                "100": "MiniMaxH3StageEAVConfigEXPT8", "101": "MiniMaxH3StageEAVApplyEXPT8",
                "102": "MiniMaxH3StageEAVAuditEXPT8", "103": "MiniMaxH3StageSaveEXPT8",
                "104": "MiniMaxH3PromptRelayPlanT8Advanced",
                "120": "MiniMaxH3ModelTimeBiasSamplerT8Advanced",
                "121": "MiniMaxH3SpatioTemporalGuidanceT8Advanced",
                "60": "MiniMaxH3StageLoadEXPT8", "90": "MiniMaxH3RFHandoffEXPT8"}
    forbidden = {"1", "9", "10", "11", "12", "13", "22", "23", "24", "25", "26", "29",
                 "40", "41", "42", "43", "44", "45", "46", "47", "50", "51", "70", "71",
                 "88", "89", "110", "111"}
    if any(graph.get(key, {}).get("class_type") != kind for key, kind in required.items()) or \
            forbidden & graph.keys() or \
            graph["93"]["inputs"].get("unet_name") != full.MODEL or \
            "completed_stage" in graph["93"]["inputs"] or \
            graph["94"]["inputs"].get("lora_name") != full.LORA or \
            graph["94"]["inputs"].get("strength_model") != 1. or \
            graph["94"]["inputs"].get("model") != ["93", 0] or \
            graph["95"]["inputs"].get("prompt_relay_plan") != ["104", 0] or \
            graph["60"]["inputs"].get("expected_stage") != "rf_base" or \
            graph["90"]["inputs"].get("completed_av") != ["60", 0] or \
            graph["96"]["inputs"].get("model") != ["121", 0] or \
            graph["96"]["inputs"].get("restart_steps") != 3 or \
            graph["96"]["inputs"].get("restart_video_sigma") != .15 or \
            graph["100"]["inputs"].get("mode") != "report_only" or \
            graph["103"]["inputs"].get("stage_result") != ["99", 2] or \
            graph["14"]["inputs"].get("av_latent") != ["102", 0]:
        raise ValueError("Saved three-pass RF cold graph lost its RESTART-only boundary")
    graph["60"]["inputs"].update(artifact_path=receipt["path"], artifact_sha256=receipt["sha256"])
    graph["104"]["inputs"]["length"] = frames
    graph["100"]["inputs"].update(tau=.2, start_video_progress=0.,
                                   end_video_progress=1., g_hard_limit=3.)
    graph["16"]["inputs"]["filename_prefix"] = f"{full.OUTPUT}/cold_selected"
    return graph, source_sha


def preflight(args: argparse.Namespace, source_sha: str) -> dict:
    models = args.comfy_root / "models"
    assets = {
        "base": models / "diffusion_models" / full.MODEL,
        "ema_b_lora": models / "loras" / full.LORA,
        "clip": models / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae": models / "vae/minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": models / "vae/minimax_h3_audio_vae_fp32.safetensors",
    }
    gpu = shared.gpu_memory_mib()
    free_ram = full.native_gpu._free_physical_mib()
    checks = {
        "saved_candidate_sha_bound": bool(source_sha),
        "all_assets_installed": all(path.is_file() for path in assets.values()),
        "isolated_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "gpu_free_vram_gate": bool(gpu.get("available") and gpu["free_mib"] >= args.min_free_vram_mib),
        "system_free_ram_gate": free_ram is not None and free_ram >= args.min_free_ram_mib,
    }
    return {"schema": SCHEMA + ".preflight", "candidate_sha256": source_sha,
            "assets": {name: str(path) for name, path in assets.items()},
            "gpu": gpu, "physical_free_mib": free_ram,
            "checks": checks, "ready": all(checks.values())}


def stage_checks(phase: dict) -> dict[str, bool]:
    events = phase.get("events") or []
    progress = [str(item.get("node")) for item in events if item.get("type") == "progress"]
    executing = [str(item.get("node")) for item in events if item.get("type") == "executing"]
    required = ("60", "90", "93", "94", "95", "120", "121", "96", "99", "103", "102")
    present = all(node in executing for node in required)
    ordered = present and executing.index("60") < executing.index("90") < \
        executing.index("96") < executing.index("99") < executing.index("103") and \
        executing.index("93") < executing.index("94") < executing.index("95") < \
        executing.index("120") < executing.index("121") < executing.index("96")
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        "only_rf_restart_three_progress": progress == ["99"] * 3,
        "frozen_rf_base_and_effects_before_restart": ordered,
        "no_low_lift_or_rf_base_execution": not any(key in executing for key in
                                                ("1", "9", "10", "11", "12", "13", "22", "23", "24", "25",
                                                 "26", "29", "40", "41", "42", "43", "44", "45", "46", "47",
                                                 "50", "51", "70", "71", "88", "89", "110", "111")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-root", required=True, type=Path)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8868)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=85000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.full_run_root = args.full_run_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    source_report, receipt, manifest, state = verified_base(args.full_run_root)
    graph, source_sha = build_resume_graph(receipt)
    readiness = preflight(args, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "artifacts/development/modular-sampling-m2-rf-real-gpu-20260925" / \
        f"{run_id}-three-pass-cold-restart"
    run_root.mkdir(parents=True, exist_ok=False)
    two_stage_resume.copy_base(receipt, manifest, state, run_root)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "full_run_root": str(args.full_run_root), "port": args.port,
              "candidate_sha256": source_sha, "base_artifact_path": receipt["path"],
              "base_artifact_sha256": receipt["sha256"], "preflight": readiness}
    try:
        with shared.IsolatedServer(args, run_root, "s29-rf-three-pass-cold-restart") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(core_capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        report["checks"] = stage_checks(phase)
        if report["checks"]["terminal_success"]:
            report["restart_save"] = common._save_receipt(phase, "103")
            report["checks"]["restart_artifact_sha"] = common._artifact_sha_matches(run_root, report["restart_save"])
            report["checks"]["copied_base_sha"] = common._artifact_sha_matches(run_root, receipt)
            report["checks"]["source_base_sha"] = shared._sha256_file(manifest) == receipt["sha256"].upper()
            report["checks"].update(full.media_checks(run_root, 128, 64, 22, prefix="cold_selected"))
            report["checks"]["restart_request_matches_uninterrupted"] = (
                report["restart_save"]["report"]["request_sha256"] ==
                source_report["restart_save"]["report"]["request_sha256"])
            report["checks"]["restart_state_matches_uninterrupted"] = (
                report["restart_save"]["report"]["state_sha256"] ==
                source_report["restart_save"]["report"]["state_sha256"])
            complete_media = full.media_path(args.full_run_root)
            resumed_media = full.media_path(run_root, "cold_selected")
            report["decoded_sha256"] = {
                stream: {"full": full.native_gpu.decoded_stream_hash(complete_media, stream),
                         "cold": full.native_gpu.decoded_stream_hash(resumed_media, stream)}
                for stream in ("video", "audio")}
            for stream in ("video", "audio"):
                report["checks"][f"decoded_{stream}_matches_uninterrupted"] = (
                    report["decoded_sha256"][stream]["full"] == report["decoded_sha256"][stream]["cold"])
        report["checks"]["source_candidate_unchanged"] = \
            shared._sha256_file(full.candidate("resume_effects")) == source_sha
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("cold_rf_three_pass_restart_request_state_decoded_av_parity_mechanical_pass"
                            if all(report["checks"].values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("RF three-pass cold RESTART-only graph failed mechanical parity")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
