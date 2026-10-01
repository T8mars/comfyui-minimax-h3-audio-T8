"""Cold HIGH-only DMD8/B50 -> native3/4/5 from an exact owned LOW manifest.

This can recover a completed LOW even if its prior full graph failed later.
The LOW manifest and tensor SHA are checked and copied; no LOW model, VDN
branch, conditioning or sampler exists in the resumed prompt.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_modular_s04_pdd_gpu as common  # noqa: E402
import run_modular_s05_vdn_gpu as full  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as core_capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s05-vdn-dmd8-native4-real-cold-high-gpu.v1"
CANDIDATE = (PROJECT / "artifacts/development/modular-sampling-m2-vdn-relay-20260923/candidates-v2/"
             "VDNRelay_stage_dmd_8nfe_native4_resume_relay_eav_EXP.api.json")


def candidate(refine: int, training: str = "stage_dmd_8nfe") -> Path:
    full.candidate(refine, training)
    return CANDIDATE.with_name(
        f"VDNRelay_{training}_native{refine}_resume_relay_eav_EXP.api.json")


def schema(refine: int, training: str = "stage_dmd_8nfe") -> str:
    candidate(refine, training)
    if training == "stage_b_50nfe":
        return f"t8.modular-sampling.s03-vdn-b50-native{refine}-real-cold-high-gpu.v1"
    return SCHEMA if refine == 4 else SCHEMA.replace("s05-vdn", "s03-vdn").replace(
        "native4", f"native{refine}")


def verified_low(full_root: Path, refine: int = 4,
                 training: str = "stage_dmd_8nfe") -> tuple[dict, Path, Path]:
    report = json.loads((full_root / "report.json").read_text(encoding="utf-8"))
    if report.get("run_root") != str(full_root) or \
            report.get("source_candidate_sha256") != \
            shared._sha256_file(full.candidate(refine, training)) or \
            report.get("refine", 4) != refine or \
            report.get("training", "stage_dmd_8nfe") != training or \
            report.get("test_dimensions") != [128, 64, 22]:
        raise ValueError("Source must be the matching owned DMD8/native small full graph")
    store = (full_root / "output/MiniMaxH3/stage_artifacts").resolve()
    manifests = list(store.glob(f"VDNRelay/{training}/native{refine}/LOW-*/manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Expected exactly one committed source LOW manifest")
    manifest = manifests[0].resolve()
    if not manifest.is_relative_to(store) or manifest.name != "manifest.json":
        raise ValueError("Source LOW manifest escaped the owned stage store")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema") != "t8.modular-sampling.frozen-stage.v1" or \
            data.get("portable_identity") is not True or \
            data.get("stage_context", {}).get("stage") != "vdn_complete" or \
            data.get("state_file") != "state.safetensors":
        raise ValueError("Source LOW is not an exact portable completed VDN stage")
    state = manifest.parent / "state.safetensors"
    if not state.is_file() or state.stat().st_size != data["state_bytes"] or \
            shared._sha256_file(state) != data["state_sha256"].upper():
        raise ValueError("Source LOW tensor bytes changed")
    receipt = {"path": manifest.relative_to(store).as_posix(),
               "sha256": shared._sha256_file(manifest).lower(), "report": data}
    return receipt, manifest, state


def build_resume_graph(receipt: dict, refine: int = 4,
                       training: str = "stage_dmd_8nfe") -> tuple[dict, str]:
    source = candidate(refine, training)
    path, sha = receipt.get("path"), receipt.get("sha256")
    parts = PurePosixPath(path).parts if isinstance(path, str) else ()
    if (len(parts) != 5 or parts[:3] != ("VDNRelay", training, f"native{refine}")
            or not parts[3].startswith("LOW-") or parts[4] != "manifest.json"
            or not isinstance(sha, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha) is None):
        raise ValueError("Frozen LOW receipt does not match the selected VDN training/refine route")
    source_sha = shared._sha256_file(source)
    graph = deepcopy(json.loads(source.read_text(encoding="utf-8")))
    required = {"22": "UNETLoader", "23": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
                "26": "MiniMaxH3NativeStageBindEXPT8", "29": "MiniMaxH3StageSamplerEXPT8",
                "51": "MiniMaxH3StageSaveEXPT8", "60": "MiniMaxH3StageLoadEXPT8"}
    if any(graph.get(key, {}).get("class_type") != name for key, name in required.items()) or \
            any(key in graph for key in ("1", "9", "10", "11", "12", "13", "40", "41", "43",
                                         "45", "50", "70", "110", "112")):
        raise ValueError("Saved VDN resume graph lost its HIGH-only boundary")
    if graph["22"]["inputs"]["unet_name"] != full.MODEL or \
            graph["71"]["inputs"]["lora_name"] != full.HIGH_LORA or \
            graph["23"]["inputs"]["av_latent"] != ["60", 0] or \
            graph["26"]["inputs"]["stage"] != "native_high" or \
            graph["60"]["inputs"]["expected_stage"] != "vdn_complete" or \
            graph["93"]["inputs"]["refine_steps"] != refine:
        raise ValueError("Saved VDN-to-native HIGH model/handoff contract changed")
    graph["60"]["inputs"].update(artifact_path=path, artifact_sha256=sha)
    graph["140"]["inputs"]["length"] = 22
    graph["42"]["inputs"].update(tau=.2, start_video_progress=0.,
                                 end_video_progress=1., g_hard_limit=3.)
    graph["16"]["inputs"]["filename_prefix"] = \
        f"MiniMaxH3/{full.media_dir(refine, training)}/high_only_selected"
    return graph, source_sha


def copy_low(receipt: dict, manifest: Path, state: Path, run_root: Path) -> None:
    store = (run_root / "output/MiniMaxH3/stage_artifacts").resolve()
    target = (store / receipt["path"]).resolve()
    if not target.is_relative_to(store) or target.exists() or \
            (target.parent / "state.safetensors").exists():
        raise ValueError("Refusing ambiguous destination for frozen VDN LOW")
    target.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(state, target.parent / "state.safetensors")
    if shared._sha256_file(target.parent / "state.safetensors") != \
            receipt["report"]["state_sha256"].upper():
        raise ValueError("Copied VDN LOW tensor SHA changed")
    shutil.copy2(manifest, target)
    if shared._sha256_file(target) != receipt["sha256"].upper():
        raise ValueError("Copied VDN LOW manifest SHA changed")


def stage_checks(phase: dict, refine: int = 4) -> dict[str, bool]:
    events = phase.get("events") or []
    progress = [str(event.get("node")) for event in events if event.get("type") == "progress"]
    executing = [str(event.get("node")) for event in events if event.get("type") == "executing"]
    required = ("60", "23", "22", "29", "51")
    ordered = all(node in executing for node in required) and \
        executing.index("60") < executing.index("23") < executing.index("29") < executing.index("51") and \
        executing.index("22") < executing.index("29")
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        ("only_native_high4" if refine == 4 else "only_native_high_steps"):
            progress == ["29"] * refine,
        "frozen_low_and_independent_high_dependencies": ordered,
        "no_vdn_low_execution": not any(node in executing for node in
                                        ("1", "9", "10", "11", "12", "13", "40", "41", "43",
                                         "45", "50", "70", "110", "112")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-root", required=True, type=Path)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8870)
    parser.add_argument("--refine", type=int, choices=(3, 4, 5), default=4)
    parser.add_argument("--training", choices=full.TRAINING, default="stage_dmd_8nfe")
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    # Native HIGH alone still staged >60 GiB in the observed cold run.
    parser.add_argument("--min-free-ram-mib", type=int, default=80000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.full_run_root = args.full_run_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    receipt, manifest, state = verified_low(args.full_run_root, args.refine, args.training)
    source_report = json.loads((args.full_run_root / "report.json").read_text(encoding="utf-8"))
    source_complete = source_report.get("status") == \
        "real_vdn_split_small_media_mechanical_pass_not_quality_acceptance"
    if source_complete and (not isinstance(source_report.get("high_save"), dict) or
                            not common._artifact_sha_matches(args.full_run_root,
                                                             source_report["high_save"])):
        raise ValueError("Completed source HIGH receipt is missing or changed")
    graph, source_sha = build_resume_graph(receipt, args.refine, args.training)
    # A verified frozen LOW is the only VDN input to this graph. Requiring
    # removed/offloaded LOW weights here would prevent genuine HIGH-only retry.
    readiness = full.preflight(args, source_sha, include_low_assets=False)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "artifacts/development/modular-sampling-m2-vdn-real-gpu-20260925" / \
        f"{run_id}-{'DMD8' if args.training == 'stage_dmd_8nfe' else 'B50'}-native{args.refine}-high-only"
    run_root.mkdir(parents=True, exist_ok=False)
    copy_low(receipt, manifest, state, run_root)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": schema(args.refine, args.training), "status": "started", "run_root": str(run_root),
              "full_run_root": str(args.full_run_root), "port": args.port,
              "candidate_sha256": source_sha, "low_artifact_path": receipt["path"],
              "low_artifact_sha256": receipt["sha256"], "preflight": readiness,
              "source_full_graph_complete": source_complete, "refine": args.refine,
              "training": args.training}
    try:
        with shared.IsolatedServer(
                args, run_root,
                f"s{'05' if args.training == 'stage_dmd_8nfe' and args.refine == 4 else '03'}-vdn-{'dmd8' if args.training == 'stage_dmd_8nfe' else 'b50'}-native{args.refine}-high-only"
        ) as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(core_capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        report["checks"] = stage_checks(phase, args.refine)
        if report["checks"]["terminal_success"]:
            report["high_save"] = common._save_receipt(phase, "51")
            report["checks"]["high_artifact_sha"] = common._artifact_sha_matches(run_root, report["high_save"])
            report["checks"]["low_artifact_sha"] = common._artifact_sha_matches(run_root, receipt)
            report["checks"]["source_low_sha"] = shared._sha256_file(manifest) == receipt["sha256"].upper()
            report["checks"]["source_candidate_unchanged"] = \
                shared._sha256_file(candidate(args.refine, args.training)) == source_sha
            report["checks"].update(full._media_checks(
                run_root, 128, 64, 22, prefix="high_only_selected", refine=args.refine,
                training=args.training))
            if source_complete:
                report["checks"]["high_request_matches_uninterrupted"] = (
                    report["high_save"]["report"]["request_sha256"] ==
                    source_report["high_save"]["report"]["request_sha256"])
                report["checks"]["high_state_matches_uninterrupted"] = (
                    report["high_save"]["report"]["state_sha256"] ==
                    source_report["high_save"]["report"]["state_sha256"])
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        if all(report["checks"].values()):
            report["status"] = (
                "cold_high_only_vdn_request_state_parity_mechanical_pass_not_quality_acceptance"
                if source_complete else
                "cold_high_only_vdn_media_mechanical_pass_not_parity_or_quality_acceptance")
        else:
            report["status"] = "fail"
        if report["status"] == "fail":
            raise RuntimeError("VDN cold HIGH-only graph failed a mechanical stage check")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
