"""Real-weight native S01–S03 split/cold probes from sealed graphs.

The default validates the sealed candidate and resources only. Explicit
``--confirm-run`` uses a fresh owned Core; ``--full-run-root`` selects a cold
HIGH-only run from a verified prior full-run LOW manifest.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_modular_s02_native_gpu as native  # noqa: E402
import run_modular_s04_pdd_gpu as common  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as core_capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s01-base-flow-real-split-gpu.v1"
FULL_PASS = "real_s01_base_flow_small_media_mechanical_pass_not_quality_acceptance"
COLD_PASS = "cold_s01_high_request_state_decoded_av_parity_mechanical_pass"
CANDIDATES = PROJECT / "artifacts/development/modular-sampling-m2-native-explicit-20260923/candidates-v4"
OUTPUT = "MiniMaxH3/S01_BaseFlow_RealSplit"
DIMENSIONS = (128, 64, 22)


def _route(recipe: str, coarse: int, refine: int) -> tuple[str, str, str, str]:
    if recipe == "base_flow" and coarse == 4 and refine == 4:
        return SCHEMA, FULL_PASS, COLD_PASS, OUTPUT
    if recipe == "lbh" and coarse == 4 and refine in (3, 4, 5):
        return ("t8.modular-sampling.s02-lbh-real-split-gpu.v1",
                "real_s02_lbh_small_media_mechanical_pass_not_quality_acceptance",
                "cold_s02_lbh_high_request_state_decoded_av_parity_mechanical_pass",
                "MiniMaxH3/S02_LBH_RealSplit")
    if recipe == "complete" and coarse in (8, 20) and refine in (3, 4, 5):
        return ("t8.modular-sampling.s03-complete-native-real-split-gpu.v1",
                "real_s03_complete_native_small_media_mechanical_pass_not_quality_acceptance",
                "cold_s03_high_request_state_decoded_av_parity_mechanical_pass",
                "MiniMaxH3/S03_CompleteNative_RealSplit")
    raise ValueError("Expected base_flow 4+4, LBH 4+3/4/5, or complete 8/20+3/4/5")


def candidate(cold: bool = False, *, recipe: str = "base_flow",
              coarse: int = 4, refine: int = 4) -> Path:
    _route(recipe, coarse, refine)
    variant = "resume_effects" if cold else "save_effects"
    return CANDIDATES / f"Native_{recipe}_{coarse}plus{refine}_{variant}_EXP.api.json"


def build_graph(*, cold: bool = False, receipt: dict | None = None,
                width: int = 128, height: int = 64, frames: int = 22,
                recipe: str = "base_flow", coarse: int = 4, refine: int = 4) -> tuple[dict, str]:
    if min(width, height, frames) < 1 or width % 32 or height % 32:
        raise ValueError("Use positive 32-aligned native test geometry")
    _, _, _, output = _route(recipe, coarse, refine)
    plan_type = ("MiniMaxH3TwoPassSigmaPlanT8Advanced" if recipe == "base_flow"
                 else "MiniMaxH3LearnedTwoPassParityPlanT8Advanced")
    source = candidate(cold, recipe=recipe, coarse=coarse, refine=refine)
    source_sha = shared._sha256_file(source)
    graph = deepcopy(json.loads(source.read_text(encoding="utf-8")))
    common_nodes = {"22": "UNETLoader" if cold else "MiniMaxH3StageUNETLoaderAfterEXPT8",
                    "23": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
                    "25": "MiniMaxH3TwoPassLatentReconcileT8Advanced",
                    "26": "MiniMaxH3NativeStageBindEXPT8",
                    "29": "MiniMaxH3StageSamplerEXPT8",
                    "42": "MiniMaxH3StageEAVConfigEXPT8",
                    "47": "MiniMaxH3PromptRelayPlanT8Advanced",
                    "51": "MiniMaxH3StageSaveEXPT8",
                    "90" if not cold else "92": "MiniMaxH3DualClockSamplerT8",
                    "93": plan_type}
    if any(graph.get(key, {}).get("class_type") != kind for key, kind in common_nodes.items()):
        raise ValueError("Saved native graph lost stages/effects")
    plan_fields = ({"coarse_steps": 4, "refine_steps": 4}
                   if recipe == "base_flow" else
                   {"base_steps": 8, "coarse_steps": 4, "refine_steps": refine})
    if any(graph["93"]["inputs"].get(key) != value for key, value in plan_fields.items()):
        raise ValueError("Saved native HIGH schedule changed")
    if graph["22"]["inputs"].get("unet_name") != native.MODEL or \
            graph["23"]["inputs"].get("model_name") != native.UPSCALE or \
            graph["71"]["inputs"].get("lora_name") != native.LORA or \
            graph["71"]["inputs"].get("strength_model") != 1. or \
            graph["26"]["inputs"].get("stage") != "native_high" or \
            graph["26"]["inputs"].get("sampler") != ["92", 1] or \
            graph["26"]["inputs"].get("sigmas") != ["93", 1] or \
            graph["29"]["inputs"].get("latent_image") != ["25", 0] or \
            graph["92"]["inputs"].get("steps") != 8 or \
            graph["25"]["inputs"].get("second_pass_audio_source") != \
            ("first_pass" if recipe == "complete" else "legacy_policy") or \
            graph["25"]["inputs"].get("second_pass_audio_strength") != 0. or \
            graph["42"]["inputs"].get("mode") != "report_only" or \
            graph["51"]["inputs"].get("stage_result") != ["29", 2]:
        raise ValueError("Saved native high recipe/audio/learned contract changed")
    if cold:
        forbidden = {"1", "9", "10", "11", "12", "13", "40", "41", "43", "45", "50", "70", "90", "91"}
        if forbidden & graph.keys() or graph.get("60", {}).get("class_type") != \
                "MiniMaxH3StageLoadEXPT8" or \
                graph["60"]["inputs"].get("expected_stage") != "native_low" or \
                "completed_stage" in graph["22"]["inputs"] or \
                graph["23"]["inputs"].get("av_latent") != ["60", 1] or \
                not isinstance(receipt, dict):
            raise ValueError("Saved native cold graph lost its exact LOW boundary")
        if recipe == "complete" and not _complete_audio_audit(graph):
            raise ValueError("Saved complete-first cold graph lost its audio audit")
        graph["60"]["inputs"].update(artifact_path=receipt["path"],
                                       artifact_sha256=receipt["sha256"])
    else:
        required = {"1": "UNETLoader", "10": "MiniMaxH3NativeStageBindEXPT8",
                    "13": "MiniMaxH3StageSamplerEXPT8",
                    "40": "MiniMaxH3PromptRelayPlanT8Advanced",
                    "41": "MiniMaxH3StageEAVConfigEXPT8",
                    "50": "MiniMaxH3StageSaveEXPT8"}
        if recipe != "complete":
            required["91"] = plan_type
            if any(graph["91"]["inputs"].get(key) != value for key, value in plan_fields.items()):
                raise ValueError("Saved native LOW schedule changed")
        if recipe == "complete" and ("91" in graph or not _complete_audio_audit(graph)):
            raise ValueError("Saved complete-first graph lost full LOW or audio audit")
        if any(graph.get(key, {}).get("class_type") != kind for key, kind in required.items()) or \
                graph["1"]["inputs"].get("unet_name") != native.MODEL or \
                graph["22"]["inputs"].get("completed_stage") != ["13", 2] or \
                graph["10"]["inputs"].get("stage") != "native_low" or \
                graph["10"]["inputs"].get("sampler") != ["90", 1] or \
                graph["10"]["inputs"].get("sigmas") != \
                (["90", 2] if recipe == "complete" else ["91", 0]) or \
                graph["90"]["inputs"].get("steps") != (coarse if recipe == "complete" else 8) or \
                graph["23"]["inputs"].get("av_latent") != ["45", 0] or \
                graph["41"]["inputs"].get("mode") != "report_only" or \
                graph["50"]["inputs"].get("stage_result") != ["13", 2] or \
                graph["40"]["inputs"].get("length") != graph["47"]["inputs"].get("length"):
            raise ValueError("Saved native LOW recipe/independent HIGH contract changed")
        if (recipe == "complete" and coarse == 20 and "70" in graph) or \
                (recipe != "complete" or coarse == 8) and \
                (graph.get("70", {}).get("class_type") != "MiniMaxH3LoRACompatibilityLoaderT8Advanced" or
                 graph["70"]["inputs"].get("lora_name") != native.LORA or
                 graph["70"]["inputs"].get("strength_model") != 1.):
            raise ValueError("Saved native LOW LoRA branch changed")
        graph["9"]["inputs"].update(width=width, height=height)
        graph["40"]["inputs"]["length"] = frames
        graph["41"]["inputs"].update(tau=.2, start_video_progress=0.,
                                       end_video_progress=1., g_hard_limit=3.)
    graph["47"]["inputs"]["length"] = frames
    graph["42"]["inputs"].update(tau=.2, start_video_progress=0.,
                                   end_video_progress=1., g_hard_limit=3.)
    graph["16"]["inputs"]["filename_prefix"] = f"{output}/{'high_only' if cold else 'full'}_selected"
    return graph, source_sha


def _complete_audio_audit(graph: dict) -> bool:
    return graph.get("94", {}).get("class_type") == "MiniMaxH3TwoPassAudioAuditT8Advanced" and \
        graph["94"]["inputs"].get("second_pass_input") == ["25", 0] and \
        graph["94"]["inputs"].get("second_pass_output") == ["46", 0] and \
        graph["94"]["inputs"].get("expected_audio_strength") == 0. and \
        graph.get("14", {}).get("inputs", {}).get("av_latent") == ["94", 0]


def media_path(root: Path, *, cold: bool = False, recipe: str = "base_flow",
               coarse: int = 4, refine: int = 4) -> Path:
    _, _, _, output = _route(recipe, coarse, refine)
    stem = "high_only_selected" if cold else "full_selected"
    paths = list((root / "output" / output).glob(f"{stem}*.mp4"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one owned S01 MP4")
    return paths[0]


def media_checks(root: Path, width: int, height: int, frames: int,
                 *, cold: bool = False, recipe: str = "base_flow",
                 coarse: int = 4, refine: int = 4) -> dict[str, bool]:
    video = media_path(root, cold=cold, recipe=recipe, coarse=coarse, refine=refine)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)],
                           capture_output=True, text=True, timeout=30, check=True)
    streams = json.loads(probe.stdout)["streams"]
    picture = [item for item in streams if item.get("codec_type") == "video"]
    sound = [item for item in streams if item.get("codec_type") == "audio"]
    decode = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video),
                             "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                            capture_output=True, timeout=90, check=False)
    return {
        "h264_expected_frames_geometry": len(picture) == 1 and picture[0].get("codec_name") == "h264"
        and int(picture[0].get("nb_frames", 0)) == frames
        and (picture[0].get("width"), picture[0].get("height")) == (width * 2, height * 2),
        "one_aac_audio": len(sound) == 1 and sound[0].get("codec_name") == "aac",
        "full_av_decode": decode.returncode == 0,
        "private_file_nonempty": video.is_relative_to(root) and video.stat().st_size > 0,
    }


def stage_checks(phase: dict, *, cold: bool = False, recipe: str = "base_flow",
                 coarse: int = 4, refine: int = 4) -> dict[str, bool]:
    _route(recipe, coarse, refine)
    events = phase.get("events") or []
    progress = [str(item.get("node")) for item in events if item.get("type") == "progress"]
    executing = [str(item.get("node")) for item in events if item.get("type") == "executing"]
    if cold:
        required = ("60", "23", "22", "92", "93", "26", "29", "51", "46")
        if recipe == "complete":
            required += ("94",)
        present = all(node in executing for node in required)
        ordered = present and executing.index("60") < executing.index("23") < \
            executing.index("29") < executing.index("51") < executing.index("46") and \
            executing.index("22") < executing.index("92") < executing.index("26") < executing.index("29")
        if ordered and recipe == "complete":
            ordered = executing.index("46") < executing.index("94")
        return {"terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
                "only_high_refine_progress": progress == ["29"] * refine,
                "frozen_low_learned_independent_high_order": ordered,
                "no_low_execution": not any(key in executing for key in
                                        ("1", "9", "10", "11", "12", "13", "40", "41",
                                         "43", "45", "50", "70", "90", "91"))}
    required = ("90", "13", "50", "45", "23", "22", "92", "93", "26", "29", "51", "46")
    required += ("94",) if recipe == "complete" else ("91",)
    present = all(node in executing for node in required)
    ordered = present and executing.index("13") < executing.index("45") < \
        executing.index("23") < executing.index("29") < executing.index("51") < \
        executing.index("46") and executing.index("13") < executing.index("50") and \
        executing.index("13") < executing.index("22") < executing.index("92") < \
        executing.index("26") < executing.index("29")
    if ordered and recipe == "complete":
        ordered = executing.index("46") < executing.index("94")
    first_steps = coarse if recipe == "complete" else 4
    return {"terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
            "exact_low_high_refine_progress": progress == ["13"] * first_steps + ["29"] * refine,
            "native_schedule_lift_independent_high_order": ordered,
            "both_stage_saves_and_eav_audits_executed": present}


def verified_low(full_root: Path, *, recipe: str = "base_flow",
                 coarse: int = 4, refine: int = 4) -> tuple[dict, dict, Path, Path]:
    schema, full_pass, _, _ = _route(recipe, coarse, refine)
    report = json.loads((full_root / "report.json").read_text(encoding="utf-8"))
    phase = json.loads((full_root / "phase.json").read_text(encoding="utf-8"))
    if report.get("schema") != schema or report.get("status") != full_pass or \
            report.get("run_root") != str(full_root) or \
            report.get("recipe", recipe) != recipe or report.get("refine", refine) != refine or \
            report.get("coarse", coarse) != coarse or \
            report.get("candidate_sha256") != shared._sha256_file(candidate(
                recipe=recipe, coarse=coarse, refine=refine)) or \
            report.get("test_dimensions") != list(DIMENSIONS) or \
            not all(report.get("checks", {}).values()) or \
            not all(stage_checks(phase, recipe=recipe, coarse=coarse, refine=refine).values()) or \
            not all(media_checks(full_root, *DIMENSIONS, recipe=recipe,
                                 coarse=coarse, refine=refine).values()):
        raise ValueError("Source must be exact successful owned native full graph")
    receipt = report.get("low_save")
    if not isinstance(receipt, dict) or not common._artifact_sha_matches(full_root, receipt) or \
            not isinstance(report.get("high_save"), dict) or \
            not common._artifact_sha_matches(full_root, report["high_save"]):
        raise ValueError("Source S01 stages are missing or changed")
    store = (full_root / "output/MiniMaxH3/stage_artifacts").resolve()
    manifest = (store / receipt["path"]).resolve()
    if not manifest.is_relative_to(store) or manifest.name != "manifest.json":
        raise ValueError("Source S01 LOW escaped owned stage store")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema") != "t8.modular-sampling.frozen-stage.v1" or \
            data.get("portable_identity") is not True or \
            data.get("stage_context", {}).get("stage") != "native_low" or \
            data.get("state_file") != "state.safetensors":
        raise ValueError("Source S01 LOW is not portable native stage")
    state = manifest.parent / "state.safetensors"
    if not state.is_file() or state.stat().st_size != data["state_bytes"] or \
            shared._sha256_file(state) != data["state_sha256"].upper():
        raise ValueError("Source S01 LOW tensor changed")
    return report, receipt, manifest, state


def copy_low(receipt: dict, manifest: Path, state: Path, run_root: Path) -> None:
    store = (run_root / "output/MiniMaxH3/stage_artifacts").resolve()
    target = (store / receipt["path"]).resolve()
    if not target.is_relative_to(store) or target.name != "manifest.json" or target.exists():
        raise ValueError("Refusing ambiguous S01 LOW destination")
    target.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(state, target.parent / "state.safetensors")
    if shared._sha256_file(target.parent / "state.safetensors") != \
            receipt["report"]["state_sha256"].upper():
        raise ValueError("Copied S01 LOW state changed")
    shutil.copy2(manifest, target)
    if shared._sha256_file(target) != receipt["sha256"].upper():
        raise ValueError("Copied S01 LOW manifest changed")


def preflight(args: argparse.Namespace, candidate_sha: str) -> dict:
    result = native.preflight(args, candidate_sha)
    result["schema"] = _route(args.recipe, args.coarse, args.refine)[0] + ".preflight"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("base_flow", "lbh", "complete"), default="base_flow")
    parser.add_argument("--coarse", type=int, choices=(4, 8, 20), default=4)
    parser.add_argument("--refine", type=int, choices=(3, 4, 5), default=4)
    parser.add_argument("--full-run-root", type=Path,
                        help="Verified full root; enables new-Core HIGH-only resume")
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8875)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        schema, full_pass, cold_pass, _ = _route(args.recipe, args.coarse, args.refine)
    except ValueError as error:
        parser.error(str(error))
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    cold = args.full_run_root is not None
    args.min_free_ram_mib = args.min_free_ram_mib or (
        70000 if cold else (90000 if args.recipe == "complete" else 85000))
    source_report = receipt = manifest = state = None
    if cold:
        args.full_run_root = args.full_run_root.resolve()
        source_report, receipt, manifest, state = verified_low(
            args.full_run_root, recipe=args.recipe, coarse=args.coarse, refine=args.refine)
    graph, source_sha = build_graph(cold=cold, receipt=receipt,
                                    recipe=args.recipe, coarse=args.coarse, refine=args.refine)
    readiness = preflight(args, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    route_id = {"base_flow": "S01", "lbh": "S02", "complete": "S03"}[args.recipe]
    run_root = PROJECT / "artifacts/development/modular-sampling-m2-native-explicit-real-gpu-20260925" / \
        f"{stamp}-{route_id}-{args.recipe}-{args.coarse}plus{args.refine}-{'cold-high' if cold else 'full'}"
    run_root.mkdir(parents=True, exist_ok=False)
    if cold:
        copy_low(receipt, manifest, state, run_root)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": schema, "status": "started", "run_root": str(run_root),
              "mode": "cold_high" if cold else "full", "port": args.port,
              "recipe": args.recipe, "coarse": args.coarse, "refine": args.refine,
              "candidate_sha256": source_sha, "preflight": readiness,
              "test_dimensions": list(DIMENSIONS)}
    if cold:
        report.update(full_run_root=str(args.full_run_root), low_artifact_path=receipt["path"],
                      low_artifact_sha256=receipt["sha256"])
    try:
        with shared.IsolatedServer(args, run_root,
                                   f"{route_id}-{args.recipe}-{'cold' if cold else 'full'}") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(core_capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        report["checks"] = stage_checks(phase, cold=cold, recipe=args.recipe,
                                         coarse=args.coarse, refine=args.refine)
        if report["checks"]["terminal_success"]:
            if not cold:
                report["low_save"] = common._save_receipt(phase, "50")
                report["checks"]["low_artifact_sha"] = common._artifact_sha_matches(run_root, report["low_save"])
            report["high_save"] = common._save_receipt(phase, "51")
            report["checks"]["high_artifact_sha"] = common._artifact_sha_matches(run_root, report["high_save"])
            report["checks"].update(media_checks(run_root, *DIMENSIONS, cold=cold,
                                                   recipe=args.recipe, coarse=args.coarse,
                                                   refine=args.refine))
            if cold:
                report["checks"]["copied_low_sha"] = common._artifact_sha_matches(run_root, receipt)
                report["checks"]["source_low_sha"] = shared._sha256_file(manifest) == receipt["sha256"].upper()
                report["checks"]["high_request_matches_full"] = \
                    report["high_save"]["report"]["request_sha256"] == \
                    source_report["high_save"]["report"]["request_sha256"]
                report["checks"]["high_state_matches_full"] = \
                    report["high_save"]["report"]["state_sha256"] == \
                    source_report["high_save"]["report"]["state_sha256"]
                report["decoded_sha256"] = {
                    stream: {"full": native.decoded_stream_hash(
                                 media_path(args.full_run_root, recipe=args.recipe,
                                            coarse=args.coarse, refine=args.refine), stream),
                             "cold": native.decoded_stream_hash(
                                 media_path(run_root, cold=True, recipe=args.recipe,
                                            coarse=args.coarse, refine=args.refine), stream)}
                    for stream in ("video", "audio")}
                for stream in ("video", "audio"):
                    report["checks"][f"decoded_{stream}_matches_full"] = \
                        report["decoded_sha256"][stream]["full"] == \
                        report["decoded_sha256"][stream]["cold"]
        report["checks"]["source_candidate_unchanged"] = \
            shared._sha256_file(candidate(cold, recipe=args.recipe,
                                          coarse=args.coarse, refine=args.refine)) == source_sha
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = (cold_pass if cold else full_pass) if all(report["checks"].values()) else "fail"
        if report["status"] == "fail":
            raise RuntimeError("S01 base-flow graph failed mechanical checks")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
