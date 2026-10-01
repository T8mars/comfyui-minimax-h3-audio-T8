"""Real-weight S22 four-window v5 full versus final-window-only cold resume.

The saved drafts are immutable. Default mode checks resources only; an explicit
``--confirm-run`` submits reduced-canvas private copies to two owned Cores.
This is mechanical parity evidence, never image/audio quality acceptance.
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

import run_modular_s02_native_gpu as native_gpu  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as core_capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s22-v5-four-window-real-gpu.v1"
BASE = (PROJECT / "artifacts/development/"
        "modular-sampling-m4-chunked-v5-storage-20260924/candidate-v6-four-windows")
FREEZE = BASE / "01_freeze_window_2.api.json"
RESUME = BASE / "02_resume_window_3_DRAFT.api.json"
HASHES = {
    FREEZE: "ec81a29932884e7bc6a556dde02c6afe07850462adad023209b17d78bbe47c37",
    RESUME: "6f79adfa04a598c71b5978805b447d6371f68e8912782e106b9cd5650018aa19",
}
IMAGE = "10A.jpg"
OUTPUT = "MiniMaxH3/S22_ChunkedV5_Real"
RUNS = PROJECT / "artifacts/development/modular-sampling-m4-chunked-v5-real-gpu-20260925"


def _candidate(path: Path) -> dict:
    if shared._sha256_file(path).lower() != HASHES[path]:
        raise ValueError(f"Saved S22 candidate changed: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _small_common(graph: dict, *, width: int, height: int) -> None:
    if width < 128 or height < 128 or width % 32 or height % 32:
        raise ValueError("S22 source canvas must be at least 128 and 32-aligned")
    if (graph["7"]["inputs"].get("length") != 192
            or graph["14"]["inputs"].get("temporal_chunk_frames") != 85
            or graph["14"]["inputs"].get("temporal_overlap_frames") != 34
            or graph["14"]["inputs"].get("sampling_contract") != "standard_joint_4plus4_exp"
            or graph["14"]["inputs"].get("spatial_strategy") != "full_frame_safe"
            or graph["9"]["inputs"].get("coarse_steps") != 4
            or graph["9"]["inputs"].get("refine_steps") != 4
            or graph["34"]["inputs"].get("length") != 192):
        raise ValueError("Saved S22 four-window standard4+4 contract changed")
    graph["6"]["inputs"].update(width=width, height=height)
    graph["7"]["inputs"].update(width=width, height=height)
    graph["14"]["inputs"].update(
        target_width=width * 2, target_height=height * 2,
        tile_width=width * 2, tile_height=height * 2,
        minimum_tile_size=128,
    )
    # Only the private execution copies increase the explicit guard: the
    # stored 1.5 default and all old workflows remain untouched.
    for key in ("40", "43", "46", "49"):
        if key in graph:
            graph[key]["inputs"].update(mode="report_only", g_hard_limit=3.0)


def _final_window_save() -> dict:
    return {"class_type": "MiniMaxH3ChunkedV5WindowSaveEXPT8",
            "inputs": {"window_result": ["33", 1],
                       "partial4_denoised_output": ["12", 1],
                       "lifted_full_av": ["28", 0], "prepared": ["29", 0],
                       "plan": ["14", 0], "confirm_save": True}}


def build_graph(kind: str, *, width: int = 128, height: int = 128,
                receipts: dict | None = None) -> dict:
    if kind not in ("full", "cold"):
        raise ValueError("Expected full or cold S22 graph")
    graph = deepcopy(_candidate(FREEZE if kind == "full" else RESUME))
    if any(graph.get(key, {}).get("class_type") != node for key, node in {
        "14": "MiniMaxH3ChunkedTwoPassPlanT8Advanced",
        "28": "MiniMaxH3ChunkedV5GlobalLiftEXPT8",
        "29": "MiniMaxH3ChunkedV5PrepareEXPT8",
        "34": "MiniMaxH3PromptRelayPlanT8Advanced",
        "35": "MiniMaxH3PromptRelayConditioningT8Advanced",
    }.items()):
        raise ValueError("S22 candidate lost plan/lift/Relay contracts")
    _small_common(graph, width=width, height=height)
    if kind == "full":
        expected = {"12": "SamplerCustomAdvanced",
                    "30": "MiniMaxH3ChunkedV5PASS2WindowEXPT8",
                    "31": "MiniMaxH3ChunkedV5PASS2WindowEXPT8",
                    "32": "MiniMaxH3ChunkedV5PASS2WindowEXPT8",
                    "52": "MiniMaxH3NativeLatentCheckpointSaveT8Advanced",
                    "53": "MiniMaxH3ChunkedV5WindowSaveEXPT8"}
        if any(graph.get(key, {}).get("class_type") != node for key, node in expected.items()):
            raise ValueError("S22 freeze candidate lost a partial/window stage")
        graph["52"]["inputs"].update(confirm_save=True,
                                     filename_prefix="s22_real_partial4_source")
        graph["53"]["inputs"]["confirm_save"] = True
        tail = deepcopy(_candidate(RESUME))
        for key in ("18", "19", "25", "33", "39", "49", "50", "51"):
            graph[key] = tail[key]
        graph["49"]["inputs"].update(mode="report_only", g_hard_limit=3.0)
        for key in ("33", "39", "50"):
            for name, value in list(graph[key]["inputs"].items()):
                if value == ["54", 0]:
                    graph[key]["inputs"][name] = ["12", 1]
                elif value == ["53", 1]:
                    graph[key]["inputs"][name] = ["53", 1]
        graph["33"]["inputs"]["previous_result"] = ["53", 1]
        graph["39"]["inputs"]["previous_result"] = ["53", 1]
        graph["50"]["inputs"]["previous_result"] = ["53", 1]
        graph["51"]["inputs"]["window_result"] = ["33", 1]
        graph["55"] = _final_window_save()
        graph["18"]["inputs"]["av_latent"] = ["51", 0]
        previews = {
            "200": ["52", 2], "201": ["52", 3], "202": ["52", 4],
            "203": ["52", 5], "204": ["53", 2], "205": ["53", 3],
            "206": ["53", 4], "207": ["42", 1], "208": ["45", 1],
            "209": ["48", 1], "210": ["51", 1],
            "211": ["55", 2], "212": ["55", 3], "213": ["55", 4],
        }
    else:
        if any(key in graph for key in ("11", "12", "30", "31", "32")):
            raise ValueError("Saved S22 cold candidate unexpectedly includes earlier samplers")
        if (graph["33"]["inputs"].get("previous_result") != ["53", 1]
                or graph["53"]["inputs"].get("expected_window_index") != 2
                or graph["49"]["inputs"].get("mode") != "report_only"):
            raise ValueError("S22 cold final-window boundary changed")
        if receipts is not None:
            graph["52"]["inputs"].update(
                checkpoint_path=receipts["native"]["path"],
                expected_file_sha256=receipts["native"]["sha256"],
                expected_manifest_json=receipts["native"]["manifest_json"],
            )
            graph["53"]["inputs"].update(
                artifact_path=receipts["window2"]["path"],
                artifact_sha256=receipts["window2"]["sha256"],
            )
        graph["55"] = _final_window_save()
        graph["55"]["inputs"]["partial4_denoised_output"] = ["54", 0]
        graph["18"]["inputs"]["av_latent"] = ["51", 0]
        previews = {"200": ["52", 7], "201": ["53", 2],
                    "210": ["51", 1], "211": ["55", 2],
                    "212": ["55", 3], "213": ["55", 4]}
    graph["25"]["inputs"].update(
        filename_prefix=f"{OUTPUT}/{kind}_selected",
        format="mp4",
        **{"format.codec": "auto"},
    )
    for key, source in previews.items():
        graph[key] = {"class_type": "PreviewAny", "inputs": {"source": source}}
    return graph


def preflight(args: argparse.Namespace) -> dict:
    freeze, resume = _candidate(FREEZE), _candidate(RESUME)
    models = args.comfy_root / "models"
    assets = {
        "base": models / "diffusion_models" / freeze["4"]["inputs"]["unet_name"],
        "clip": models / "text_encoders" / freeze["3"]["inputs"]["clip_name"],
        "video_vae": models / "vae" / freeze["1"]["inputs"]["vae_name"],
        "audio_vae": models / "vae" / freeze["2"]["inputs"]["vae_name"],
        "turbo_lora": models / "loras" / freeze["23"]["inputs"]["lora_name"],
        "trained_3d": models / "latent_upscale_models" / freeze["14"]["inputs"]["model_name"],
        "source_image": args.comfy_root / "input" / freeze["5"]["inputs"]["image"],
    }
    gpu, free_ram = shared.gpu_memory_mib(), native_gpu._free_physical_mib()
    checks = {
        "both_saved_candidates_sha_bound": bool(freeze and resume),
        "all_assets_installed": all(path.is_file() for path in assets.values()),
        "full_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "cold_port_free": not shared.port_is_listening("127.0.0.1", args.port + 1),
        "gpu_free_vram_gate": bool(gpu.get("available") and gpu["free_mib"] >= args.min_free_vram_mib),
        "system_free_ram_gate": free_ram is not None and free_ram >= args.min_free_ram_mib,
    }
    return {"schema": SCHEMA + ".preflight", "candidate_sha256":
            {path.name: digest for path, digest in HASHES.items()},
            "assets": {key: str(path) for key, path in assets.items()},
            "gpu": gpu, "physical_free_mib": free_ram,
            "checks": checks, "ready": all(checks.values())}


def _phase_text(phase: dict, key: str) -> str:
    return core_capture._phase_text(phase, key)


def _native_receipt(full_root: Path, phase: dict) -> dict:
    path = _phase_text(phase, "200")
    sha = _phase_text(phase, "201").lower()
    manifest = _phase_text(phase, "202")
    report = json.loads(_phase_text(phase, "203"))
    root = (full_root / "output/MiniMaxH3/latent_checkpoints").resolve()
    file = (root / path).resolve()
    if (not file.is_relative_to(root) or not file.is_file()
            or shared._sha256_file(file).lower() != sha
            or report.get("status") != "SAVED_VERIFIED"
            or report.get("checkpoint_path") != path
            or report.get("file_sha256", "").lower() != sha
            or json.loads(manifest).get("schema") is None):
        raise ValueError("S22 native partial4 receipt/file mismatch")
    return {"path": path, "sha256": sha, "manifest_json": manifest,
            "report": report, "bytes": file.stat().st_size}


def _window_receipt(root: Path, phase: dict, *, stage: int,
                    path_node: str, sha_node: str, manifest_node: str) -> dict:
    path, sha = _phase_text(phase, path_node), _phase_text(phase, sha_node).lower()
    manifest = json.loads(_phase_text(phase, manifest_node))
    store = (root / "output/MiniMaxH3/chunked_v5_window_artifacts").resolve()
    file = (store / path).resolve()
    state = file.parent / "state.safetensors"
    if (not file.is_relative_to(store) or not file.is_file()
            or file.name != "manifest.json" or shared._sha256_file(file).lower() != sha
            or manifest != json.loads(file.read_text(encoding="utf-8"))
            or manifest.get("schema") != "t8.modular-sampling.chunked-v5-frozen-window.v1"
            or manifest.get("binding", {}).get("index") != stage
            or manifest.get("automatic_cache_reuse") is not False
            or manifest.get("execution_identity_certified") is not False
            or not state.is_file() or state.stat().st_size != manifest.get("state_bytes")
            or shared._sha256_file(state).lower() != manifest.get("state_sha256")):
        raise ValueError(f"S22 window {stage} manifest/state receipt mismatch")
    return {"path": path, "sha256": sha, "manifest": manifest,
            "manifest_bytes": file.stat().st_size, "state_bytes": state.stat().st_size}


def _copy_receipts(full_root: Path, cold_root: Path, receipts: dict) -> None:
    native_source = full_root / "output/MiniMaxH3/latent_checkpoints" / receipts["native"]["path"]
    native_target = cold_root / "output/MiniMaxH3/latent_checkpoints" / receipts["native"]["path"]
    native_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(native_source, native_target)
    if shared._sha256_file(native_target).lower() != receipts["native"]["sha256"]:
        raise ValueError("S22 copied native partial4 changed")
    for name in ("manifest.json", "state.safetensors"):
        source = (full_root / "output/MiniMaxH3/chunked_v5_window_artifacts" /
                  Path(receipts["window2"]["path"]).parent / name)
        target = (cold_root / "output/MiniMaxH3/chunked_v5_window_artifacts" /
                  Path(receipts["window2"]["path"]).parent / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        expected = (receipts["window2"]["sha256"] if name == "manifest.json"
                    else receipts["window2"]["manifest"]["state_sha256"])
        if shared._sha256_file(target).lower() != expected:
            raise ValueError("S22 copied completed window changed")


def _media(root: Path, kind: str, width: int, height: int) -> dict:
    files = list((root / "output" / OUTPUT).glob(f"{kind}_selected*.mp4"))
    if len(files) != 1:
        raise ValueError(f"Expected one S22 {kind} private MP4")
    path = files[0]
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                           capture_output=True, text=True, timeout=30, check=True)
    streams = json.loads(probe.stdout)["streams"]
    video = [value for value in streams if value.get("codec_type") == "video"]
    audio = [value for value in streams if value.get("codec_type") == "audio"]
    decode = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
                             "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                            capture_output=True, timeout=180, check=False)
    checks = {
        "h264_192_frames_scaled_canvas": len(video) == 1 and video[0].get("codec_name") == "h264"
                                          and int(video[0].get("nb_frames", 0)) == 192
                                          and (video[0].get("width"), video[0].get("height")) ==
                                          (width * 2, height * 2),
        "one_aac_audio": len(audio) == 1 and audio[0].get("codec_name") == "aac",
        "full_av_decode": decode.returncode == 0,
        "private_file_nonempty": path.is_relative_to(root) and path.stat().st_size > 0,
    }
    return {"path": str(path), "sha256": shared._sha256_file(path).lower(),
            "checks": checks,
            "decoded_sha256": {key: native_gpu.decoded_stream_hash(path, key)
                               for key in ("video", "audio")}}


def _effects(phase: dict, keys: tuple[str, ...], expected_windows: tuple[int, ...]) -> dict:
    items = [json.loads(_phase_text(phase, key)) for key in keys]
    return {"reports": items, "valid": len(items) == len(expected_windows) and all(
        item.get("chunked_v5_window_index") == index
        and item.get("status") == "observed_report_only"
        and item.get("relay_required") is True
        and item.get("relay_attention_calls", 0) > 0
        and item.get("completed_forwards") == item.get("planned_forwards") == 4
        for item, index in zip(items, expected_windows))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8873)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=95000)
    parser.add_argument("--timeout-seconds", type=float, default=3600.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.comfy_root, args.python = args.comfy_root.resolve(), args.python.resolve()
    full = build_graph("full", width=args.width, height=args.height)
    build_graph("cold", width=args.width, height=args.height)
    readiness = preflight(args)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_root = RUNS / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-four-window-pair")
    full_root, cold_root = run_root / "full", run_root / "cold"
    full_root.mkdir(parents=True, exist_ok=False)
    cold_root.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "preflight": readiness, "test_dimensions": [args.width, args.height, 192],
              "candidate_sha256": {path.name: value for path, value in HASHES.items()}}
    try:
        image_source = args.comfy_root / "input" / IMAGE
        report["source_image_sha256"] = shared._sha256_file(image_source).lower()
        for root in (full_root, cold_root):
            (root / "input").mkdir()
            shutil.copy2(image_source, root / "input" / IMAGE)
            if shared._sha256_file(root / "input" / IMAGE).lower() != report["source_image_sha256"]:
                raise ValueError("S22 private input image changed during copy")
            (root / "paths.json").write_text(json.dumps(
                probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
        (full_root / "prompt.json").write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
        args.host, args.input_directory = "127.0.0.1", full_root / "input"
        args.extra_model_paths_config = full_root / "paths.json"
        with shared.IsolatedServer(args, full_root, "s22-v5-full") as server:
            report["full_core_pid"] = server.process.pid
            full_phase = asyncio.run(core_capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=full,
                timeout_seconds=args.timeout_seconds))
        (full_root / "phase.json").write_text(json.dumps(full_phase, ensure_ascii=False, indent=2), encoding="utf-8")
        if (full_phase.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("S22 full four-window graph did not succeed")
        native = _native_receipt(full_root, full_phase)
        window2 = _window_receipt(full_root, full_phase, stage=2,
                                  path_node="204", sha_node="205", manifest_node="206")
        final_full = _window_receipt(full_root, full_phase, stage=3,
                                     path_node="211", sha_node="212", manifest_node="213")
        full_effects = _effects(full_phase, ("207", "208", "209", "210"), (0, 1, 2, 3))
        full_media = _media(full_root, "full", args.width, args.height)
        report["full"] = {"native": native, "window2": window2,
                          "final_window": final_full, "effects": full_effects,
                          "media": full_media}
        receipts = {"native": native, "window2": window2}
        _copy_receipts(full_root, cold_root, receipts)
        cold = build_graph("cold", width=args.width, height=args.height, receipts=receipts)
        (cold_root / "prompt.json").write_text(json.dumps(cold, ensure_ascii=False, indent=2), encoding="utf-8")
        args.port += 1
        args.input_directory, args.extra_model_paths_config = cold_root / "input", cold_root / "paths.json"
        with shared.IsolatedServer(args, cold_root, "s22-v5-cold-window3") as server:
            report["cold_core_pid"] = server.process.pid
            cold_phase = asyncio.run(core_capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=cold,
                timeout_seconds=args.timeout_seconds))
        (cold_root / "phase.json").write_text(json.dumps(cold_phase, ensure_ascii=False, indent=2), encoding="utf-8")
        if (cold_phase.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("S22 cold final-window graph did not succeed")
        final_cold = _window_receipt(cold_root, cold_phase, stage=3,
                                     path_node="211", sha_node="212", manifest_node="213")
        cold_effects = _effects(cold_phase, ("210",), (3,))
        cold_media = _media(cold_root, "cold", args.width, args.height)
        report["cold"] = {"final_window": final_cold, "effects": cold_effects,
                          "media": cold_media}
        cold_executing = [str(item.get("node")) for item in cold_phase.get("events") or []
                          if item.get("type") == "executing"]
        checks = {
            "full_and_cold_terminal_success": True,
            "all_four_full_window_effects_executed": full_effects["valid"],
            "only_final_cold_window_effect_executed": cold_effects["valid"],
            "cold_graph_has_no_first_or_earlier_window_sampler":
                all(key not in cold for key in ("11", "12", "30", "31", "32")),
            "cold_events_have_no_first_or_earlier_window_sampler":
                not any(key in cold_executing for key in ("11", "12", "30", "31", "32")),
            "cold_load_and_final_window_executed": all(
                key in cold_executing for key in ("52", "53", "33", "51", "55", "25")),
            "final_joint_av_output_identity_equal":
                final_full["manifest"]["binding"]["output_identity"] ==
                final_cold["manifest"]["binding"]["output_identity"],
            "decoded_video_audio_equal":
                full_media["decoded_sha256"] == cold_media["decoded_sha256"],
            "full_media_valid": all(full_media["checks"].values()),
            "cold_media_valid": all(cold_media["checks"].values()),
            "input_and_candidates_unchanged":
                shared._sha256_file(image_source).lower() == report["source_image_sha256"]
                and all(shared._sha256_file(path).lower() == sha for path, sha in HASHES.items()),
            "both_owned_cores_stopped":
                not shared.port_is_listening("127.0.0.1", args.port - 1)
                and not shared.port_is_listening("127.0.0.1", args.port),
        }
        report["checks"] = checks
        report["status"] = ("real_s22_four_window_small_media_exact_cold_parity_not_quality_acceptance"
                            if all(checks.values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("S22 four-window pair failed a mechanical check")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
