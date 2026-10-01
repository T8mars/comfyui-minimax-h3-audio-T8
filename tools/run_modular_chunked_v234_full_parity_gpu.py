"""Compare uninterrupted Chunked v2/v3/v4 with fresh-Core HIGH-only resume.

Default mode is read-only preflight. --confirm-run submits private reduced-canvas
copies of SHA-pinned saved candidates; neither published nor original graphs move.
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

import run_modular_chunked_v2_gpu as v2  # noqa: E402
import run_modular_chunked_v34_gpu as v34  # noqa: E402
import run_modular_s02_native_gpu as native_gpu  # noqa: E402
import run_modular_s18_chunked_gpu as s18  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.chunked-v234-full-cold-parity.v1"
RUNS = PROJECT / "artifacts/development/modular-sampling-m4-chunked-v234-full-parity-20260925"


def _spec(variant: str) -> dict:
    if variant == "v2":
        return {"adapter": v2, "save": "35", "load": "35", "high": "26",
                "audit": "34", "video": "19", "image": v2.IMAGE, "mask": None,
                "low_steps": 4, "high_steps": 4, "hashes": v2.HASHES}
    if variant in v34.CONFIG:
        current = v34.CONFIG[variant]
        return {"adapter": v34, "save": current["save"], "load": current["load"],
                "high": current["high"], "audit": current["eav_audit"],
                "video": current["video"], "image": v34.IMAGE,
                "mask": v34.MASK if variant == "v4" else None,
                "low_steps": 8, "high_steps": 3,
                "hashes": {v34._path(variant, "freeze"): current["hashes"][0],
                           v34._path(variant, "resume"): current["hashes"][1]}}
    raise ValueError("Use v2, v3 or v4")


def _base_graph(variant: str, kind: str, width: int, height: int) -> dict:
    if variant == "v2":
        return v2.build_graph(kind, width=width, height=height)
    return v34.build_graph(variant, kind, width=width, height=height)


def _remap_freeze(freeze: dict) -> dict:
    result = {}
    for key, node in freeze.items():
        if int(key) >= 200:
            continue
        mapped = deepcopy(node)
        for name, value in mapped["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                mapped["inputs"][name] = [str(300 + int(value[0])), value[1]]
        result[str(300 + int(key))] = mapped
    return result


def _final_save(variant: str, audit: str) -> dict:
    return {"class_type": "MiniMaxH3NativeLatentCheckpointSaveT8Advanced", "inputs": {
        "av_latent": [audit, 0], "filename_prefix": f"chunked_{variant}_full_parity_final",
        "checkpoint_id": f"chunked_{variant}_full_parity_final",
        "confirm_save": True, "verify_after_write": True, "hash_chunk_megabytes": 8}}


def build_graph(variant: str, kind: str, *, width: int = 128,
                height: int = 128, receipt: dict | None = None) -> dict:
    spec = _spec(variant)
    if kind not in {"full", "cold"}:
        raise ValueError("Use full or cold")
    graph = _base_graph(variant, "resume", width, height)
    load_id = spec["load"]
    if kind == "full":
        freeze = _base_graph(variant, "freeze", width, height)
        graph.pop(load_id)
        graph.pop("202", None)
        for node in graph.values():
            for name, value in node["inputs"].items():
                if value == [load_id, 0]:
                    node["inputs"][name] = ["312", 1]
                elif isinstance(value, list) and len(value) == 2 and value[0] == load_id:
                    raise ValueError("Unexpected native Load output in complete graph")
        graph.update(_remap_freeze(freeze))
        graph["200"] = {"class_type": "PreviewAny", "inputs": {
            "source": [str(300 + int(spec["save"])), 4]}}
        graph["201"] = {"class_type": "PreviewAny", "inputs": {
            "source": [str(300 + int(spec["save"])), 5]}}
    elif receipt is not None:
        graph[load_id]["inputs"].update(
            checkpoint_path=receipt["path"],
            expected_manifest_json=receipt["manifest_json"],
            expected_file_sha256=receipt["file_sha256"],
        )
    graph[spec["video"]]["inputs"]["filename_prefix"] = (
        f"MiniMaxH3/Chunked_v234_Full_Parity/{variant}/{kind}")
    graph["250"] = _final_save(variant, spec["audit"])
    for key, source in {"220": ["250", 4], "221": ["250", 5]}.items():
        graph[key] = {"class_type": "PreviewAny", "inputs": {"source": source}}
    if kind == "full" and (load_id in graph or "312" not in graph):
        raise ValueError("Uninterrupted graph still loads an earlier checkpoint")
    if kind == "cold" and ("12" in graph or "312" in graph):
        raise ValueError("Cold graph unexpectedly contains first-pass sampling")
    return graph


def preflight(args: argparse.Namespace) -> dict:
    spec = _spec(args.variant)
    if args.variant == "v2":
        base = v2.preflight(args)
    else:
        base = v34.preflight(args)
    checks = dict(base["checks"])
    checks.update({
        "cold_port_free": not shared.port_is_listening("127.0.0.1", args.port + 1),
        "system_free_ram_gate": (native_gpu._free_physical_mib() or 0) >= args.min_free_ram_mib,
        "all_source_candidates_sha_pinned": all(
            spec["adapter"]._sha(path) == digest for path, digest in spec["hashes"].items()),
    })
    return {**base, "schema": SCHEMA + ".preflight", "checks": checks,
            "ready": all(checks.values())}


def _copy_receipt(full_root: Path, cold_root: Path, receipt: dict) -> None:
    base = (full_root / "output/MiniMaxH3/latent_checkpoints").resolve()
    source = (base / receipt["path"]).resolve()
    target_base = (cold_root / "output/MiniMaxH3/latent_checkpoints").resolve()
    if not source.is_relative_to(base) or not source.is_file():
        raise ValueError("First-pass checkpoint escaped owned output")
    target = target_base / source.relative_to(base)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if spec_sha(target) != receipt["file_sha256"].lower():
        raise ValueError("Copied first-pass checkpoint SHA differs from source receipt")


def spec_sha(path: Path) -> str:
    return v2._sha(path).lower()


def _final_receipt(root: Path, phase: dict) -> dict:
    manifest = json.loads(capture._phase_text(phase, "220"))
    saved = json.loads(capture._phase_text(phase, "221"))
    if saved.get("status") != "SAVED_VERIFIED":
        raise ValueError("Final native AV checkpoint did not verify")
    base = (root / "output/MiniMaxH3/latent_checkpoints").resolve()
    file = (base / saved["checkpoint_path"]).resolve()
    if (not file.is_relative_to(base) or not file.is_file()
            or spec_sha(file) != saved["file_sha256"].lower()
            or not isinstance(manifest.get("components"), list)):
        raise ValueError("Final AV checkpoint differs from its SHA/manifest")
    return {"path": str(file), "sha256": spec_sha(file), "manifest": manifest}


def _media(root: Path, variant: str, kind: str, width: int, height: int) -> dict:
    folder = root / f"output/MiniMaxH3/Chunked_v234_Full_Parity/{variant}"
    matches = list(folder.glob(f"{kind}_*-audio.mp4"))
    if len(matches) != 1:
        raise ValueError("Expected one private complete H.264/AAC media output")
    file = matches[0]
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(file)],
                           capture_output=True, text=True, timeout=30, check=True)
    streams = json.loads(probe.stdout)["streams"]
    video = [item for item in streams if item.get("codec_type") == "video"]
    audio = [item for item in streams if item.get("codec_type") == "audio"]
    decode = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(file),
                             "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                            capture_output=True, timeout=120, check=False)
    checks = {
        "h264_22_frames": len(video) == 1 and video[0].get("codec_name") == "h264"
                          and int(video[0].get("nb_frames", 0)) == 22
                          and (video[0].get("width"), video[0].get("height")) ==
                          (width * 2, height * 2),
        "aac_audio": len(audio) == 1 and audio[0].get("codec_name") == "aac",
        "full_av_decode": decode.returncode == 0,
        "private_nonempty": file.is_relative_to(root) and file.stat().st_size > 0,
    }
    return {"path": str(file), "sha256": spec_sha(file), "checks": checks,
            "decoded_md5": {stream: s18._decoded_md5(file, stream)
                            for stream in ("video", "audio")}}


def _progress(phase: dict, low: str, high: str) -> tuple[int, int]:
    nodes = [str(event.get("node")) for event in phase.get("events") or []
             if event.get("type") == "progress"]
    return nodes.count(low), nodes.count(high)


def _effect(phase: dict, high: str, node: str, expected_steps: int) -> bool:
    report = json.loads(capture._phase_text(phase, node))
    return all((
        report.get("status") == "observed_report_only",
        report.get("completed_forwards") == expected_steps,
        report.get("planned_forwards") == expected_steps,
        report.get("selector_calls") == expected_steps * 50,
        report.get("relay_attention_calls") == expected_steps * 50,
        report.get("relay_required") is True,
        report.get("clock_match") is True,
        report.get("quality_accepted") is False,
        report.get("cache_reuse_authorized") is False,
        _progress(phase, "12", high)[1] == expected_steps,
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("v2", "v3", "v4"), required=True)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8880)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=95000)
    parser.add_argument("--server-start-timeout", type=float, default=240.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.comfy_root, args.python = args.comfy_root.resolve(), args.python.resolve()
    spec = _spec(args.variant)
    full = build_graph(args.variant, "full", width=args.width, height=args.height)
    build_graph(args.variant, "cold", width=args.width, height=args.height)
    ready = preflight(args)
    print(json.dumps(ready, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not ready["ready"]:
        return 0 if ready["ready"] else 2
    run_root = RUNS / args.variant / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-pair")
    full_root, cold_root = run_root / "full", run_root / "cold"
    full_root.mkdir(parents=True, exist_ok=False)
    cold_root.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "status": "started", "variant": args.variant,
              "run_root": str(run_root), "preflight": ready,
              "test_dimensions": [args.width, args.height, 22]}
    try:
        names = [spec["image"]] + ([spec["mask"]] if spec["mask"] else [])
        inputs = {name: spec_sha(args.comfy_root / "input" / name) for name in names}
        report["input_sha256"] = inputs
        for root in (full_root, cold_root):
            (root / "input").mkdir()
            for name, digest in inputs.items():
                target = root / "input" / name
                shutil.copy2(args.comfy_root / "input" / name, target)
                if spec_sha(target) != digest:
                    raise ValueError("Private input changed while copying")
            (root / "paths.json").write_text(json.dumps(
                probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
        (full_root / "prompt.json").write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
        args.host, args.input_directory = "127.0.0.1", full_root / "input"
        args.extra_model_paths_config = full_root / "paths.json"
        with shared.IsolatedServer(args, full_root, f"chunked-{args.variant}-full") as server:
            report["full_core_pid"] = server.process.pid
            full_phase = asyncio.run(capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=full,
                timeout_seconds=args.timeout_seconds))
        (full_root / "phase.json").write_text(json.dumps(full_phase, ensure_ascii=False, indent=2), encoding="utf-8")
        if (full_phase.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("Uninterrupted complete graph did not succeed")
        receipt = spec["adapter"]._receipt(full_root, full_phase)
        final_full = _final_receipt(full_root, full_phase)
        media_full = _media(full_root, args.variant, "full", args.width, args.height)
        report["full"] = {"receipt": receipt, "final": final_full, "media": media_full}
        _copy_receipt(full_root, cold_root, receipt)
        cold = build_graph(args.variant, "cold", width=args.width, height=args.height,
                           receipt=receipt)
        (cold_root / "prompt.json").write_text(json.dumps(cold, ensure_ascii=False, indent=2), encoding="utf-8")
        args.port += 1
        args.input_directory, args.extra_model_paths_config = cold_root / "input", cold_root / "paths.json"
        with shared.IsolatedServer(args, cold_root, f"chunked-{args.variant}-cold") as server:
            report["cold_core_pid"] = server.process.pid
            cold_phase = asyncio.run(capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=cold,
                timeout_seconds=args.timeout_seconds))
        (cold_root / "phase.json").write_text(json.dumps(cold_phase, ensure_ascii=False, indent=2), encoding="utf-8")
        if (cold_phase.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("Cold HIGH-only graph did not succeed")
        final_cold = _final_receipt(cold_root, cold_phase)
        media_cold = _media(cold_root, args.variant, "cold", args.width, args.height)
        report["cold"] = {"final": final_cold, "media": media_cold}
        checks = {
            "both_core_success": True,
            "full_low_high_steps": _progress(full_phase, "312", spec["high"]) ==
                (spec["low_steps"], spec["high_steps"]),
            "cold_only_high_steps": _progress(cold_phase, "12", spec["high"]) ==
                (0, spec["high_steps"]),
            "cold_graph_has_no_low": "12" not in cold and "312" not in cold,
            "full_effect_calls": _effect(full_phase, spec["high"], "203", spec["high_steps"]),
            "cold_effect_calls": _effect(cold_phase, spec["high"], "203", spec["high_steps"]),
            "final_av_content_equal": final_full["manifest"]["content_sha256"] ==
                final_cold["manifest"]["content_sha256"],
            "decoded_video_audio_equal": media_full["decoded_md5"] == media_cold["decoded_md5"],
            "both_media_valid": all(media_full["checks"].values()) and
                                all(media_cold["checks"].values()),
            "receipt_and_sources_unchanged": spec_sha(Path(receipt["absolute_path"])) ==
                receipt["file_sha256"].lower() and all(
                    spec["adapter"]._sha(path) == digest
                    for path, digest in spec["hashes"].items()) and all(
                    spec_sha(args.comfy_root / "input" / name) == digest
                    for name, digest in inputs.items()),
            "both_owned_cores_stopped": not shared.port_is_listening("127.0.0.1", args.port - 1)
                                       and not shared.port_is_listening("127.0.0.1", args.port),
        }
        report["checks"] = checks
        report["status"] = ("real_chunked_v234_small_full_cold_exact_parity_not_quality_acceptance"
                            if all(checks.values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("Chunked full/cold pair failed a mechanical gate")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
