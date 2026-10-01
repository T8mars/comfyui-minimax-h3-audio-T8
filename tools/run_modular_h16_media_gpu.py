"""Isolated resource-gated real-model H16 freeze -> cold HIGH/media probe.

This is a 124-frame 224->448 square *derived probe*, never a qualification of
the unchanged 736->1472 workflow. It retains the saved candidate's real
samplers, learned 3D upscaler, H16 windows, native/H16 Save/Load and AV/VHS
terminal. The controller owns only its isolated Core processes and outputs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
CORE = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / "tools"))
import run_progressive_pilot as transport  # noqa: E402
from progressive_probe_control import (  # noqa: E402
    NvmlResourceReader, ResourceGuard, SerialProbeLease, file_identity,
)
from audit_modular_h16_media_gpu import _media  # noqa: E402
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402
from build_modular_h16_media_api import PAIRS  # noqa: E402


def _reachable(graph: dict, outputs: tuple[str, ...]) -> dict:
    keep = set(outputs)
    pending = list(outputs)
    while pending:
        node = graph[pending.pop()]
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                source = value[0]
                if source not in graph:
                    raise ValueError(f"H16 API has missing source {source}")
                if source not in keep:
                    keep.add(source)
                    pending.append(source)
    return {key: deepcopy(value) for key, value in graph.items() if key in keep}


def build_probe_graphs(route: str = "plain") -> tuple[dict, dict]:
    if route not in PAIRS:
        raise ValueError("Expected plain or effects H16 route")
    config = PAIRS[route]
    freeze, resume = config["names"]
    original_freeze = json.loads((config["target"] / f"{freeze}.api.json").read_bytes())
    original_resume = json.loads((config["target"] / f"{resume}.api.json").read_bytes())
    frozen = deepcopy(original_freeze)
    resumed = deepcopy(original_resume)
    frozen["7"]["inputs"].update(width=224, height=224)
    frozen["13"]["inputs"].update(target_width=448, target_height=448,
                                  target_megapixels=0.2)
    save_nodes = ("51", "52") if route == "plain" else ("88", "89")
    for node_id in save_nodes:
        frozen[node_id]["inputs"]["confirm_save"] = True
    resumed["14"]["inputs"].update(width=448, height=448)
    resumed["21"]["inputs"]["filename_prefix"] = (
        "MiniMaxH3/ModularH16ColdResumeProbe" if route == "plain"
        else "MiniMaxH3/ModularH16EffectColdResumeProbe")
    freeze_outputs = save_nodes
    resume_outputs = ("21", "h16_report")
    if route == "effects":
        # The Relay conditioning is external to HIGH's ordinary conditioning.
        # It must use the same reduced HIGH geometry, not the saved full canvas.
        resumed["52"]["inputs"].update(width=448, height=448)
        # Change only the final HIGH window so the frozen/report-only control
        # and an independently editable active EAV window share one cold run.
        resumed["78"]["inputs"]["mode"] = "apply_exp"
        freeze_outputs += ("81", "82", "83")
        resume_outputs += ("84", "85", "86", "87")
    frozen = _reachable(frozen, freeze_outputs)
    resumed = _reachable(resumed, resume_outputs)
    if (frozen["7"]["inputs"]["length"] != 124
            or resumed["14"]["inputs"]["length"] != 124
            or frozen["13"]["inputs"]["scale_by"] != 2.0
            or resumed["20"]["inputs"]["av_latent"] != [str(config["final_av"][0]), 0]
            or (route == "effects" and (
                resumed["52"]["inputs"]["width"] != 448
                or resumed["52"]["inputs"]["height"] != 448
                or resumed["78"]["inputs"]["mode"] != "apply_exp"))
            or any(node["class_type"] == "SamplerCustomAdvanced"
                   for node in resumed.values())):
        raise ValueError("H16 fixed seven-window GPU probe contract changed")
    return frozen, resumed


def _receipt(output: Path) -> dict:
    from safetensors import safe_open

    native_root = output / "MiniMaxH3/latent_checkpoints"
    window_root = output / "MiniMaxH3/h16_window_artifacts"
    native_files = list(native_root.rglob("*.h3latent.safetensors"))
    window_files = list(window_root.rglob("manifest.json"))
    if len(native_files) != 1 or len(window_files) != 1:
        raise RuntimeError("Expected exactly one verified native and one H16 window artifact")
    native, window = native_files[0], window_files[0]
    with safe_open(str(native), framework="pt", device="cpu") as handle:
        payload = json.loads(handle.metadata()["t8_native_latent_checkpoint_json"])
    manifest = payload["manifest"]
    if payload["checkpoint_id"] != "h16_124f_learned_handoff":
        raise RuntimeError("Unexpected H16 native checkpoint identity")
    window_meta = json.loads(window.read_bytes())
    if window_meta["binding"]["index"] != 2:
        raise RuntimeError("Expected only H16 window 2 to be frozen")
    return {
        "native_path": native.relative_to(native_root).as_posix(),
        "native_file_sha256": file_identity(native)["sha256"],
        "native_manifest_json": json.dumps(manifest, ensure_ascii=False,
                                            sort_keys=True, separators=(",", ":")),
        "window_path": window.relative_to(window_root).as_posix(),
        "window_manifest_sha256": file_identity(window)["sha256"],
        "window_state_sha256": window_meta["state_sha256"],
    }


def _server_command_factory(output: Path):
    original = transport.server_command

    def command(*values):
        argv = original(*values)
        argv[argv.index("progressive_probe_extension")] = "ComfyUI-VideoHelperSuite"
        argv[argv.index("--output-directory") + 1] = str(output)
        return argv

    return command


def _phase(name: str, graph: dict, run_root: Path, output: Path, port: int,
           reader: NvmlResourceReader) -> dict:
    phase_root = run_root / name
    phase_root.mkdir()
    transport.write_json(phase_root / "paths.json", probe_resource_config(CORE, PROJECT))
    guard = ResourceGuard()
    server = transport.OwnedServer(phase_root, port, False, 2)
    monitor = None
    result = {"status": "incomplete", "graph_sha256": hashlib.sha256(
        json.dumps(graph, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    try:
        reason = guard.observe(reader.sample(), startup=True)
        if reason:
            raise RuntimeError("Startup resource guard: " + reason)
        with ExitStack() as cleanup:
            server.start()
            monitor = transport.ContinuousGuard(reader, guard,
                                                phase_root / "resources.jsonl", server)
            cleanup.callback(monitor.close)
            monitor.start()
            transport.wait_ready(server, monitor.check)
            info = server.request("GET", "/object_info")
            missing = {node["class_type"] for node in graph.values()} - set(info)
            if missing:
                raise RuntimeError("Missing H16 node registrations: " + repr(sorted(missing)))
            started = time.perf_counter()
            history, timing = transport.execute_graph(
                server, graph, phase_root / "generation", monitor.check, timeout=3600)
            result.update(status="mechanical_execution_pass",
                          elapsed_seconds=time.perf_counter() - started,
                          history_status=history.get("status"), timing=timing)
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if monitor:
            monitor.close()
        server.stop()
        result.update(server_stop=server.stop_receipt, resources=guard.report())
        transport.write_json(phase_root / "terminal.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8250)
    parser.add_argument("--route", choices=tuple(PAIRS), default="plain")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if (root.exists() or not root.is_relative_to((PROJECT / "artifacts/development").resolve())
            or not 1024 <= args.port <= 65533
            or {args.port, args.port + 1} & {8189, 8247}):
        parser.error("Use a new private artifact root and isolated non-user ports")
    freeze, resume = build_probe_graphs(args.route)
    root.mkdir(parents=True)
    transport.write_json(root / "freeze-prompt.json", freeze)
    transport.write_json(root / "resume-template.json", resume)
    if args.dry_run:
        transport.write_json(root / "terminal.json", {
            "status": "prepared_not_executed", "route": args.route,
            "freeze_nodes": len(freeze),
            "resume_nodes": len(resume), "geometry": [224, 224, 448, 448],
            "frames": 124, "qualification": "private reduced-geometry graph derivation only"})
        return
    transport.CORE, transport.PROJECT = CORE, PROJECT
    output = root / "output"
    output.mkdir()
    transport.server_command = _server_command_factory(output)
    result = {"status": "incomplete", "route": args.route,
              "qualification": "reduced_124f_224_to_448_real_model_probe"}
    lease = PROJECT / "artifacts/acceleration-research-20260909/serial-gpu.lock"
    try:
        with SerialProbeLease(lease), NvmlResourceReader() as reader:
            result["core"] = verify_core_source(CORE)
            result["freeze"] = _phase("freeze", freeze, root, output, args.port, reader)
            receipt = _receipt(output)
            transport.write_json(root / "receipt.json", receipt)
            native_load, window_load = (("51", "52") if args.route == "plain"
                                        else ("88", "89"))
            resume[native_load]["inputs"].update(
                checkpoint_path=receipt["native_path"],
                expected_file_sha256=receipt["native_file_sha256"],
                expected_manifest_json=receipt["native_manifest_json"])
            resume[window_load]["inputs"].update(
                artifact_path=receipt["window_path"],
                artifact_sha256=receipt["window_manifest_sha256"])
            result["resume"] = _phase("resume", resume, root, output, args.port + 1, reader)
            history = json.loads((root / "resume/generation/history.json").read_bytes())
            media = _media(root, history)
            if (media["width"], media["height"], media["frames"], media["fps"]) != (
                    448, 448, 124, "24/1"):
                raise RuntimeError("H16 reduced probe media geometry or timeline mismatch")
            result.update(status="mechanical_av_pass_human_pending", media=media,
                          receipt=receipt)
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        transport.write_json(root / "terminal.json", result)
        print(json.dumps({"status": result["status"], "root": str(root)},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
