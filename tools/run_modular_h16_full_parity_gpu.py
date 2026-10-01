"""Compare live seven-window H16 Relay/EAV with a new-Core cold continuation.

Only private 124-frame 224->448 derived graphs are used. Source candidates,
published workflows, production sampling nodes, and the user's Core are not
modified. Without --confirm-run this tool performs structural preflight only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))
import run_progressive_pilot as transport  # noqa: E402
from progressive_probe_control import (  # noqa: E402
    NvmlResourceReader, SerialProbeLease, file_identity,
)
from audit_modular_h16_media_gpu import _effect_report, _media  # noqa: E402
from build_modular_h16_media_api import PAIRS  # noqa: E402
from run_modular_h16_media_gpu import (  # noqa: E402
    CORE, _phase, _reachable, _server_command_factory, build_probe_graphs,
)
from vdn_probe_environment import verify_core_source  # noqa: E402

SOURCE_SHA = {
    "01_freeze_after_window_2_relay_eav.api.json":
        "00ed46f56a7c8d79540bb14c8b69d28d5e2cf38d8967c63572105cc977c2ab7b",
    "02_resume_windows_3_to_6_relay_eav_DRAFT.api.json":
        "0d98dda36bebac5a70f7df00b5ec76f69d5c1f6c69a6845590ea198eace077ef",
}
FINAL_ID = "h16_124f_full_parity_final"
SAVE_CLASS = "MiniMaxH3NativeLatentCheckpointSaveT8Advanced"
WINDOW_CLASS = "MiniMaxH3H16Pass2WindowEXPT8"


def _sha(path: Path) -> str:
    return file_identity(path)["sha256"].lower()


def _links(node: dict) -> set[str]:
    return {value[0] for value in node["inputs"].values()
            if isinstance(value, list) and len(value) == 2
            and isinstance(value[0], str)}


def _shared_nodes(freeze: dict, resume: dict) -> set[str]:
    shared = {key for key in freeze.keys() & resume.keys()
              if freeze[key] == resume[key]}
    while True:
        narrowed = {key for key in shared if _links(freeze[key]) <= shared}
        if narrowed == shared:
            return shared
        shared = narrowed


def build_graphs() -> tuple[dict, dict, dict]:
    source = PAIRS["effects"]["target"]
    actual = {name: _sha(source / name) for name in SOURCE_SHA}
    if actual != SOURCE_SHA:
        raise ValueError("Pinned H16 effect media candidate changed")
    freeze, resume = build_probe_graphs("effects")
    shared = _shared_nodes(freeze, resume)
    mapping = {key: key if key in shared else str(300 + int(key)) for key in freeze}
    if not {"13", "38", "88", "89"} <= freeze.keys():
        raise ValueError("H16 freeze handoff node contract changed")
    full = deepcopy(resume)
    for key in ("88", "89", "90"):
        del full[key]
    for key, node in freeze.items():
        if key in shared:
            continue
        moved = deepcopy(node)
        for input_name, value in moved["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                moved["inputs"][input_name] = [mapping[value[0]], value[1]]
        full[mapping[key]] = moved
    full["15"]["inputs"]["learned_latent"] = [mapping["13"], 0]
    for node in full.values():
        for input_name, value in node["inputs"].items():
            if value == ["89", 1]:
                node["inputs"][input_name] = [mapping["38"], 1]
    full["21"]["inputs"]["filename_prefix"] = "MiniMaxH3/ModularH16EffectFullParity"
    full["250"] = {
        "class_type": SAVE_CLASS,
        "inputs": {"av_latent": ["80", 0], "filename_prefix": FINAL_ID,
                   "checkpoint_id": FINAL_ID, "confirm_save": True,
                   "verify_after_write": True, "hash_chunk_megabytes": 8},
    }
    cold = deepcopy(resume)
    cold["21"]["inputs"]["filename_prefix"] = "MiniMaxH3/ModularH16EffectColdParity"
    cold["250"] = deepcopy(full["250"])
    # Resume-only source window 2 was needed by WindowLoad, which the live
    # handoff removes. Prune it before execution: Core legitimately skips an
    # unreachable orphan, but the probe's complete-uncached audit must not.
    full = _reachable(full, ("21", "250", "h16_report", "362", "365", "368",
                             "381", "382", "383", "388", "389", "71", "74",
                             "77", "80", "84", "85", "86", "87"))
    for graph in (full, cold):
        for key, node in graph.items():
            missing = _links(node) - graph.keys()
            if missing:
                raise ValueError(f"{key} has missing H16 sources: {sorted(missing)}")
    def kinds(graph: dict, kind: str) -> int:
        return sum(node["class_type"] == kind for node in graph.values())
    if (kinds(full, "SamplerCustomAdvanced") != 1
            or kinds(full, WINDOW_CLASS) != 7
            or kinds(cold, "SamplerCustomAdvanced") != 0
            or kinds(cold, WINDOW_CLASS) != 4
            or full["15"]["inputs"]["learned_latent"] != [mapping["13"], 0]
            or full["41"]["inputs"]["previous_result"] != [mapping["38"], 1]
            or any(node["class_type"] in {"MiniMaxH3NativeLatentCheckpointLoadT8Advanced",
                                                 "MiniMaxH3H16WindowLoadEXPT8",
                                                 "MiniMaxH3H16VerifiedNativeSourceEXPT8"}
                   for node in full.values())):
        raise ValueError("H16 full/cold seven-window boundary changed")
    return full, freeze, cold


def _checkpoint(output: Path, checkpoint_id: str) -> dict:
    from safetensors import safe_open

    root = output / "MiniMaxH3/latent_checkpoints"
    matches = []
    for path in root.rglob("*.h3latent.safetensors"):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            payload = json.loads(handle.metadata()["t8_native_latent_checkpoint_json"])
        if payload["checkpoint_id"] == checkpoint_id:
            matches.append((path, payload))
    if len(matches) != 1:
        raise ValueError(f"Expected one H16 {checkpoint_id} checkpoint, got {len(matches)}")
    path, payload = matches[0]
    return {"path": str(path), "file_sha256": _sha(path),
            "content_sha256": payload["manifest"]["content_sha256"],
            "components": payload["manifest"]["components"],
            "manifest": payload["manifest"]}


def _handoff_receipt(output: Path) -> dict:
    handoff = _checkpoint(output, "h16_124f_learned_handoff")
    native_root = output / "MiniMaxH3/latent_checkpoints"
    native = Path(handoff["path"])
    window_root = output / "MiniMaxH3/h16_window_artifacts"
    manifests = list(window_root.rglob("manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Expected one H16 window-2 handoff manifest")
    window = manifests[0]
    window_metadata = json.loads(window.read_bytes())
    if window_metadata["binding"]["index"] != 2:
        raise ValueError("H16 frozen window index differs")
    return {
        "native_path": native.relative_to(native_root).as_posix(),
        "native_file_sha256": handoff["file_sha256"],
        "native_manifest_json": json.dumps(handoff["manifest"], ensure_ascii=False,
                                            sort_keys=True, separators=(",", ":")),
        "window_path": window.relative_to(window_root).as_posix(),
        "window_manifest_sha256": _sha(window),
        "window_state_sha256": window_metadata["state_sha256"],
    }


def _decoded_rgb(path: Path) -> str:
    process = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, check=True)
    expected_bytes = 124 * 448 * 448 * 3
    if len(process.stdout) != expected_bytes:
        raise ValueError("H16 decoded RGB frame count/geometry differs")
    return hashlib.sha256(process.stdout).hexdigest()


def _history(root: Path, name: str) -> dict:
    return json.loads((root / name / "generation/history.json").read_bytes())


def _phase_checks(root: Path, name: str, expected_nodes: set[str]) -> dict:
    phase = json.loads((root / name / "terminal.json").read_bytes())
    nodes = {str(row["node"]) for row in phase["timing"]["node_intervals"]}
    if (phase["status"] != "mechanical_execution_pass"
            or not phase["history_status"]["completed"]
            or phase["timing"]["terminal"] != "execution_success"
            or not phase["timing"]["complete_uncached_graph"]
            or phase["server_stop"]["owned_children_remaining"]
            or phase["resources"]["status"] != "observations_within_policy"
            or not expected_nodes <= nodes):
        raise ValueError(f"H16 {name} phase execution/resource contract failed")
    return {"elapsed_seconds": phase["elapsed_seconds"], "executed_nodes": sorted(nodes)}


def _copy_handoff(full_output: Path, cold_output: Path, receipt: dict) -> dict:
    bases = (
        ("native", "MiniMaxH3/latent_checkpoints", receipt["native_path"],
         receipt["native_file_sha256"]),
        ("window", "MiniMaxH3/h16_window_artifacts", receipt["window_path"],
         receipt["window_manifest_sha256"]),
    )
    copied = {}
    for label, base, relative, expected in bases:
        source = (full_output / base / relative).resolve(strict=True)
        if not source.is_relative_to((full_output / base).resolve()) or _sha(source) != expected.lower():
            raise ValueError(f"H16 {label} handoff source differs")
        destination = cold_output / base / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if _sha(destination) != expected.lower():
            raise ValueError(f"H16 {label} handoff copy differs")
        copied[label] = str(destination)
    window_source = (full_output / "MiniMaxH3/h16_window_artifacts" /
                     receipt["window_path"]).resolve(strict=True)
    window_metadata = json.loads(window_source.read_bytes())
    state_name = window_metadata["state_file"]
    state_source = (window_source.parent / state_name).resolve(strict=True)
    if not state_source.is_relative_to(window_source.parent) or (
            _sha(state_source) != receipt["window_state_sha256"].lower()):
        raise ValueError("H16 handoff state SHA/path differs")
    state_destination = (cold_output / "MiniMaxH3/h16_window_artifacts" /
                         receipt["window_path"]).parent / state_name
    shutil.copyfile(state_source, state_destination)
    if _sha(state_destination) != receipt["window_state_sha256"].lower():
        raise ValueError("H16 copied state differs")
    copied["state"] = str(state_destination)
    return copied


def audit_pair(root: Path, receipt: dict) -> dict:
    full_history, cold_history = _history(root, "full"), _history(root, "cold")
    phase = {
        "full": _phase_checks(root, "full", {"312", "332", "335", "338", "41", "44",
                                                 "47", "50", "20", "21", "250", "388", "389"}),
        "cold": _phase_checks(root, "cold", {"41", "44", "47", "50", "20", "21", "250"}),
    }
    if "12" in phase["cold"]["executed_nodes"]:
        raise ValueError("Cold H16 route unexpectedly executed LOW sampler")
    reports = {
        "full": [_effect_report(full_history, node, index,
                                "apply_exp" if index == 6 else "report_only")
                 for index, node in enumerate(("362", "365", "368", "71", "74", "77", "80"))],
        "cold": [_effect_report(cold_history, node, index,
                                "apply_exp" if index == 6 else "report_only")
                 for index, node in enumerate(("71", "74", "77", "80"), start=3)],
    }
    full_output, cold_output = root / "full-output", root / "cold-output"
    final = {"full": _checkpoint(full_output, FINAL_ID),
             "cold": _checkpoint(cold_output, FINAL_ID)}
    media = {"full": _media(root / "full", full_history, output=full_output),
             "cold": _media(root / "cold", cold_history, output=cold_output)}
    rgb = {name: _decoded_rgb(Path(row["path"])) for name, row in media.items()}
    comparison = {
        "final_av_content": final["full"]["content_sha256"] == final["cold"]["content_sha256"],
        "final_av_components": final["full"]["components"] == final["cold"]["components"],
        "decoded_rgb": rgb["full"] == rgb["cold"],
        "decoded_pcm": media["full"]["decoded_pcm_sha256"] == media["cold"]["decoded_pcm_sha256"],
        "effect_plan": len({row["plan_sha256"] for values in reports.values()
                            for row in values}) == 1,
        "frozen_native_sha": _sha(full_output / "MiniMaxH3/latent_checkpoints" /
                                  receipt["native_path"]) == receipt["native_file_sha256"].lower(),
        "frozen_window_sha": _sha(full_output / "MiniMaxH3/h16_window_artifacts" /
                                  receipt["window_path"]) == receipt["window_manifest_sha256"].lower(),
    }
    return {"schema": "t8.modular-h16-effect-full-cold-parity.v1",
            "status": "mechanical_parity_pass_human_pending" if all(comparison.values()) else "parity_failed",
            "comparison": comparison, "phase": phase, "effects": reports,
            "final": final, "media": media, "decoded_rgb_sha256": rgb,
            "qualification": "Only 124f 224->448 real-model derived H16 Relay/EAV, last window apply_exp; "
                             "not original canvas, broader inputs/backends, or human AV quality."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8280)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    private = (PROJECT / "artifacts/development").resolve()
    if (root.exists() or not root.is_relative_to(private) or
            not 1024 <= args.port <= 65533 or {args.port, args.port + 1} & {8189, 8247}):
        parser.error("Use a fresh private root and isolated non-user ports")
    full, freeze, cold = build_graphs()
    preflight = {"schema": "t8.modular-h16-effect-full-cold-preflight.v1",
                 "status": "ready_to_run" if args.confirm_run else "prepared_not_executed",
                 "source_sha256": SOURCE_SHA, "graph_nodes": {"full": len(full), "freeze": len(freeze),
                                                          "cold": len(cold)},
                 "geometry": [224, 224, 448, 448], "frames": 124}
    if not args.confirm_run:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    root.mkdir(parents=True)
    full_output, cold_output = root / "full-output", root / "cold-output"
    full_output.mkdir()
    cold_output.mkdir()
    transport.CORE, transport.PROJECT = CORE, PROJECT
    transport.write_json(root / "preflight.json", preflight)
    transport.write_json(root / "full-prompt.json", full)
    transport.write_json(root / "cold-template.json", cold)
    result = {"status": "incomplete", "started_utc": datetime.now(timezone.utc).isoformat()}
    original_command = transport.server_command
    lease = PROJECT / "artifacts/acceleration-research-20260909/serial-gpu.lock"
    try:
        with SerialProbeLease(lease), NvmlResourceReader() as reader:
            result["core"] = verify_core_source(CORE)
            transport.server_command = _server_command_factory(full_output)
            result["full"] = _phase("full", full, root, full_output, args.port, reader)
            receipt = _handoff_receipt(full_output)
            transport.write_json(root / "handoff-receipt.json", receipt)
            result["handoff_copy"] = _copy_handoff(full_output, cold_output, receipt)
            cold["88"]["inputs"].update(checkpoint_path=receipt["native_path"],
                                         expected_file_sha256=receipt["native_file_sha256"],
                                         expected_manifest_json=receipt["native_manifest_json"])
            cold["89"]["inputs"].update(artifact_path=receipt["window_path"],
                                         artifact_sha256=receipt["window_manifest_sha256"])
            transport.write_json(root / "cold-prompt.json", cold)
            transport.server_command = original_command
            transport.server_command = _server_command_factory(cold_output)
            result["cold"] = _phase("cold", cold, root, cold_output, args.port + 1, reader)
            result["audit"] = audit_pair(root, receipt)
            result["status"] = result["audit"]["status"]
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        transport.server_command = original_command
        transport.write_json(root / "terminal.json", result)
        print(json.dumps({"status": result["status"], "root": str(root)}, ensure_ascii=False),
              flush=True)
    return 0 if result["status"] == "mechanical_parity_pass_human_pending" else 1


if __name__ == "__main__":
    raise SystemExit(main())
