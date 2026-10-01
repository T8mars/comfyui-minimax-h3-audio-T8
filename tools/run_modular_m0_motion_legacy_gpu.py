"""Compare frozen legacy Motion graphs with source-bound split graphs on real weights.

Read-only preflight by default. The opt-in run uses two serial, owned Cores and
private reduced copies. It never edits frozen workflows or the user's Core.
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

from tools.build_modular_motion_recovery_workflows import source_path, split_api  # noqa: E402
from tools.run_modular_motion_storage_gpu import _stream_hash  # noqa: E402
import run_modular_motion_effect_gpu as effect  # noqa: E402
import run_modular_s02_native_gpu as native_gpu  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402

SCHEMA = "t8.modular-m0-motion-legacy-split-real-gpu.v1"
PRIVATE = PROJECT / "artifacts/development/modular-sampling-m0-motion-legacy-20260925"
SPLIT = PROJECT / "artifacts/development/modular-sampling-m4-motion-20260923/candidate-v2"
SOURCE_SHA = {
    "Fullclip": {
        "legacy": "0455a78a8a6d1b8769371422b2a6401a84f72213679c8117fd2f7c40a75775c5",
        "split": "b2600d937ba8b2fdee87f95890949a03f20beed7127ee59ba4e5e921138000fd",
    },
    "Windowed": {
        "legacy": "414db234c7cc4246b0b4e8c2ac77ab4822a3e1aead7c2406164ec57f2450a989",
        "split": "ca18a44d1d7f6e88e2bf5a8130f27def8e14f3140d765379116717fe6378fb3a",
    },
}


def source_files(variant: str) -> tuple[Path, Path]:
    if variant not in SOURCE_SHA:
        raise ValueError("Unknown frozen Motion variant")
    return (source_path(variant), SPLIT /
            f"Motion_Recovery_{variant}_Source_Bound_Separate_Stage_EXP.api.json")


def probe_graphs(variant: str) -> tuple[dict, dict]:
    legacy_path, split_path = source_files(variant)
    for phase, path in (("legacy", legacy_path), ("split", split_path)):
        if shared._sha256_file(path).lower() != SOURCE_SHA[variant][phase]:
            raise ValueError(f"Frozen {phase} Motion source changed")
    legacy = split_api(json.loads(legacy_path.read_bytes()))
    separated = json.loads(split_path.read_bytes())
    windowed = variant == "Windowed"
    stage = "19" if windowed else "18"
    affected = {"17", "19", "20"} if windowed else {"16", "18", "19"}
    added = {"30", "31"} if windowed else {"28", "29"}
    changed = {key for key in legacy if legacy[key] != separated.get(key)}
    if (changed != affected or set(separated) - set(legacy) != added
            or legacy["9"]["class_type"] != "SamplerCustomAdvanced"
            or legacy[stage]["class_type"] != "SamplerCustomAdvanced"
            or separated[stage]["class_type"] != "MiniMaxH3StageSamplerEXPT8"
            or legacy["6"]["inputs"]["steps"] != 20
            or legacy["16" if windowed else "15"]["inputs"]["steps"] != 20):
        raise ValueError("Legacy/split Motion structural contrast changed")
    graphs = []
    for graph in (legacy, separated):
        result = deepcopy(graph)
        result["5"]["inputs"].update(prompt=effect.PROMPT, width=128, height=64, length=22)
        result["12"]["inputs"].update(mode="manual_ranges", manual_ranges="5-14:3")
        result["11"]["inputs"]["filename_prefix"] = (
            f"MiniMaxH3/S25_Motion_{variant}/pass1")
        result["24" if windowed else "22"]["inputs"]["filename_prefix"] = (
            f"MiniMaxH3/S25_Motion_{variant}/recovered")
        graphs.append(result)
    return graphs[0], graphs[1]


def preflight(args: argparse.Namespace) -> dict:
    models = args.comfy_root / "models"
    assets = {
        "base": models / "diffusion_models" / effect.MODEL,
        "clip": models / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae": models / "vae/minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": models / "vae/minimax_h3_audio_vae_fp32.safetensors",
    }
    gpu = shared.gpu_memory_mib()
    ram = native_gpu._free_physical_mib()
    checks = {
        "assets_installed": all(path.is_file() for path in assets.values()),
        "ports_free": all(not shared.port_is_listening("127.0.0.1", port)
                          for port in (args.port, args.port + 1)),
        "gpu_free_vram_gate": bool(gpu.get("available") and
                                   gpu["free_mib"] >= args.min_free_vram_mib),
        "system_free_ram_gate": ram is not None and ram >= args.min_free_ram_mib,
    }
    return {"schema": SCHEMA + ".preflight", "variant": args.variant,
            "source_sha256": SOURCE_SHA[args.variant],
            "assets": {key: str(path) for key, path in assets.items()},
            "gpu": gpu, "physical_free_mib": ram, "checks": checks,
            "ready": all(checks.values())}


def _sampled(phase: dict, variant: str) -> dict:
    events = phase.get("events") or []
    progress = [str(row.get("node")) for row in events if row.get("type") == "progress"]
    stage = "19" if variant == "Windowed" else "18"
    return {"pass1": progress.count("9"), "pass2": progress.count(stage)}


def _media(root: Path, variant: str) -> dict:
    checks = effect.media_checks(root, variant)
    if not all(value for key, value in checks.items() if not key.endswith("_sha256")):
        raise ValueError(f"Legacy/split Motion media failed strict stream checks: {checks}")
    directory = root / f"output/MiniMaxH3/S25_Motion_{variant}"
    result = {}
    for name in ("pass1", "recovered"):
        files = list(directory.glob(f"{name}_*-audio.mp4"))
        if len(files) != 1:
            raise ValueError(f"Expected exactly one {name} joint media")
        path = files[0]
        result[name] = {"path": str(path), "container_sha256": shared._sha256_file(path),
                        "rgb24_sha256": _stream_hash(path, "video"),
                        "pcm_s16le_sha256": _stream_hash(path, "audio")}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(SOURCE_SHA), required=True)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=85000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.comfy_root, args.python = args.comfy_root.resolve(), args.python.resolve()
    args.host = "127.0.0.1"
    old, new = probe_graphs(args.variant)
    readiness = preflight(args)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = PRIVATE / f"{run_id}-{args.variant}-pair"
    root.mkdir(parents=True, exist_ok=False)
    paths = root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2),
                     encoding="utf-8")
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "status": "started", "variant": args.variant,
              "source_sha256": SOURCE_SHA[args.variant], "preflight": readiness,
              "run_root": str(root), "test_dimensions": [128, 64, 22],
              "manual_motion_range": "5-14:3", "phases": {}}
    try:
        for index, (label, graph) in enumerate((("legacy", old), ("split", new))):
            phase_root = root / label
            phase_root.mkdir()
            (root / f"{label}-prompt.json").write_text(
                json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
            args.port = args.port + (1 if index else 0)
            with shared.IsolatedServer(args, phase_root, f"m0-motion-{args.variant}-{label}") as server:
                pid = server.process.pid
                phase = asyncio.run(capture._submit_prompt_capture(
                    server=f"http://127.0.0.1:{args.port}", prompt=graph,
                    timeout_seconds=args.timeout_seconds))
            (root / f"{label}-phase.json").write_text(
                json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
            report["phases"][label] = {
                "core_pid": pid, "terminal": (phase.get("terminal") or {}).get("type"),
                "sampled": _sampled(phase, args.variant),
                "server_stopped": not shared.port_is_listening(args.host, args.port),
            }
            if report["phases"][label]["terminal"] != "execution_success":
                raise RuntimeError(f"{label} Motion graph did not finish")
            report["phases"][label]["media"] = _media(phase_root, args.variant)
        legacy, split = report["phases"]["legacy"], report["phases"]["split"]
        checks = {
            "both_terminal_success": all(row["terminal"] == "execution_success"
                                         for row in (legacy, split)),
            "stock20_and_pass2_observed": all(row["sampled"]["pass1"] in (20, 40)
                                               and row["sampled"]["pass2"] == 10
                                               for row in (legacy, split)),
            "both_servers_stopped": legacy["server_stopped"] and split["server_stopped"],
            "frozen_sources_unchanged": all(
                shared._sha256_file(path).lower() == SOURCE_SHA[args.variant][label]
                for label, path in zip(("legacy", "split"), source_files(args.variant), strict=True)),
        }
        for name in ("pass1", "recovered"):
            for stream in ("rgb24_sha256", "pcm_s16le_sha256"):
                checks[f"{name}_{stream}_equal"] = (
                    legacy["media"][name][stream] == split["media"][name][stream])
        report["checks"] = checks
        report["status"] = ("pass_fixed_small_real_weight_legacy_split_stream_parity_not_quality_acceptance"
                            if all(checks.values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("Frozen old Motion and split Motion differ mechanically")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
