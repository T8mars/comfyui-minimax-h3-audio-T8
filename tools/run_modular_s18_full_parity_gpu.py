"""Compare real S18 uninterrupted three-segment output with cold later segments.

Only private copies of the two immutable saved API candidates are submitted.
Default mode is read-only preflight; --confirm-run starts two owned Cores.
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

import run_modular_s18_chunked_gpu as s18  # noqa: E402
import run_modular_s02_native_gpu as native_gpu  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s18-three-segment-full-cold-parity.v1"
RUNS = PROJECT / "artifacts/development/modular-sampling-s18-full-parity-20260925"
EAV_CANDIDATE = (PROJECT / "artifacts/development/modular-sampling-s18-v1-relay-20260924"
                 / "candidate-relay-eav-v2"
                 / "Chunked_v1_56F_Three_Segments_External_Relay_EAV_EXP.api.json")
EAV_CANDIDATE_SHA256 = "1be8c763338a3949519f2f028468fad5f8a0050153c24a8f0dbb35d5a2f68a17"


def _final_save() -> dict:
    return {"class_type": "MiniMaxH3ChunkedV1SegmentSaveEXPT8", "inputs": {
        "segment_result": ["31", 1], "source_segment": ["29", 0],
        "segment_spec": ["29", 1], "pass2_context": ["21", 0],
        "plan": ["14", 0], "confirm_save": True}}


def _attach_eav(graph: dict, kind: str) -> None:
    if s18._sha(EAV_CANDIDATE) != EAV_CANDIDATE_SHA256:
        raise ValueError("S18 Relay+EAV source candidate changed")
    source = json.loads(EAV_CANDIDATE.read_text(encoding="utf-8"))
    if (source["37"]["class_type"] != "MiniMaxH3StageEAVConfigEXPT8"
            or source["37"]["inputs"]["mode"] != "report_only"
            or source["37"]["inputs"]["g_hard_limit"] != 3.0):
        raise ValueError("S18 pinned EAV recipe changed")
    graph["60"] = deepcopy(source["37"])
    # Relay-only audits cannot certify the MODEL after EAV wraps it; replace
    # them with the combined actual-call audit used by the pinned candidate.
    for node_id in ("35", "37", "39"):
        graph.pop(node_id, None)
    stages = ((0, "25", "61", "64", "205"),
              (1, "28", "62", "65", "206" if kind == "full" else "202"),
              (2, "31", "63", "66", "207" if kind == "full" else "203"))
    for index, sampler, apply_id, audit_id, preview_id in stages:
        if sampler not in graph:
            continue
        inputs = graph[sampler]["inputs"]
        graph[apply_id] = {"class_type": "MiniMaxH3ChunkedPass2EAVApplyEXPT8", "inputs": {
            "model": inputs["model"], "sigmas": inputs["sigmas"],
            "source_segment": inputs["source_segment"],
            "lifted_segment": inputs["lifted_segment"],
            "segment_spec": inputs["segment_spec"],
            "pass2_context": inputs["pass2_context"], "plan": inputs["plan"],
            "eav_config": ["60", 0]}}
        inputs["model"] = [apply_id, 0]
        graph[audit_id] = {"class_type": "MiniMaxH3ChunkedPass2EAVAuditEXPT8", "inputs": {
            "segment_result": [sampler, 1],
            "segment_spec": inputs["segment_spec"], "runtime": [apply_id, 1]}}
        graph[preview_id] = {"class_type": "PreviewAny", "inputs": {
            "source": [audit_id, 1]}}


def build_graph(kind: str, *, width: int = 128, height: int = 128,
                receipts: dict | None = None, with_eav: bool = False) -> dict:
    if kind not in {"full", "cold"}:
        raise ValueError("Expected full or cold S18 graph")
    freeze = s18.build_graph("freeze", width=width, height=height)
    cold = s18.build_graph("resume", width=width, height=height)
    if kind == "full":
        graph = deepcopy(freeze)
        for key, node in cold.items():
            if key not in graph and int(key) < 200:
                graph[key] = deepcopy(node)
        if any(graph.get(key, {}).get("class_type") != name for key, name in {
            "12": "SamplerCustomAdvanced", "25": "MiniMaxH3ChunkedPass2SegmentEXPT8",
            "28": "MiniMaxH3ChunkedPass2SegmentEXPT8",
            "31": "MiniMaxH3ChunkedPass2SegmentEXPT8",
            "40": "MiniMaxH3NativeLatentCheckpointSaveT8Advanced",
            "41": "MiniMaxH3ChunkedV1SegmentSaveEXPT8",
        }.items()):
            raise ValueError("S18 full graph lost an original stage or save")
        if graph["28"]["inputs"]["previous_result"] != ["41", 1]:
            raise ValueError("S18 full graph lost the live saved segment-0 result")
        graph["206"] = {"class_type": "PreviewAny", "inputs": {"source": ["37", 1]}}
        graph["207"] = {"class_type": "PreviewAny", "inputs": {"source": ["39", 1]}}
    else:
        graph = cold
        if "12" in graph or "25" in graph:
            raise ValueError("S18 cold graph unexpectedly contains earlier samplers")
        if receipts is not None:
            s18._fill_resume(graph, receipts)
    graph["19"]["inputs"]["filename_prefix"] = f"MiniMaxH3/S18_Full_Parity/{kind}"
    graph["55"] = _final_save()
    for key, source in {"210": ["55", 2], "211": ["55", 3], "212": ["55", 4]}.items():
        graph[key] = {"class_type": "PreviewAny", "inputs": {"source": source}}
    if with_eav:
        _attach_eav(graph, kind)
    return graph


def preflight(args: argparse.Namespace) -> dict:
    base = s18.preflight(args)
    checks = dict(base["checks"])
    checks.update({
        "cold_port_free": not shared.port_is_listening("127.0.0.1", args.port + 1),
        "system_free_ram_gate": (native_gpu._free_physical_mib() or 0) >= args.min_free_ram_mib,
    })
    if getattr(args, "with_eav", False):
        checks["relay_eav_candidate_sha_pinned"] = (
            s18._sha(EAV_CANDIDATE) == EAV_CANDIDATE_SHA256)
    candidates = dict(base["candidate_sha256"])
    if getattr(args, "with_eav", False):
        candidates["relay_eav"] = EAV_CANDIDATE_SHA256
    return {**base, "schema": SCHEMA + ".preflight",
            "candidate_sha256": candidates, "checks": checks,
            "ready": all(checks.values())}


def _copy_receipts(full_root: Path, cold_root: Path, receipts: dict) -> None:
    for key, subdir in (("native", "latent_checkpoints"),
                        ("segment", "chunked_v1_segment_artifacts")):
        source_root = (full_root / "output/MiniMaxH3" / subdir).resolve()
        target_root = (cold_root / "output/MiniMaxH3" / subdir).resolve()
        selected = (source_root / receipts[key]["path"]).resolve()
        if not selected.is_relative_to(source_root):
            raise ValueError("S18 frozen source escaped private output")
        paths = [selected] if key == "native" else [selected, selected.parent / "state.safetensors"]
        manifest = json.loads(receipts[key]["manifest_json"]) if key == "segment" else None
        for path in paths:
            target = target_root / path.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            digest = (receipts[key]["sha256"] if path == selected else manifest["state_sha256"])
            if s18._sha(target).lower() != digest.lower():
                raise ValueError("S18 frozen artifact changed during private copy")


def _final_receipt(root: Path, phase: dict) -> dict:
    path = capture._phase_text(phase, "210")
    digest = capture._phase_text(phase, "211").lower()
    manifest = json.loads(capture._phase_text(phase, "212"))
    store = (root / "output/MiniMaxH3/chunked_v1_segment_artifacts").resolve()
    file = (store / path).resolve()
    state = file.parent / "state.safetensors"
    if (not file.is_relative_to(store) or file.name != "manifest.json"
            or not file.is_file() or s18._sha(file).lower() != digest
            or json.loads(file.read_text(encoding="utf-8")) != manifest
            or manifest.get("binding", {}).get("index") != 2
            or not state.is_file() or s18._sha(state).lower() != manifest.get("state_sha256")):
        raise ValueError("S18 final segment frozen receipt differs from disk")
    return {"path": path, "sha256": digest, "manifest": manifest}


def _media(root: Path, kind: str, target_width: int, target_height: int) -> dict:
    files = list((root / "output/MiniMaxH3/S18_Full_Parity").glob(f"{kind}_*-audio.mp4"))
    if len(files) != 1:
        raise ValueError("Expected one private S18 decoded media output")
    file = files[0]
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(file)],
                           capture_output=True, text=True, timeout=30, check=True)
    streams = json.loads(probe.stdout)["streams"]
    video = [item for item in streams if item.get("codec_type") == "video"]
    audio = [item for item in streams if item.get("codec_type") == "audio"]
    decode = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(file),
                             "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                            capture_output=True, timeout=120, check=False)
    checks = {
        "h264_56_frames": len(video) == 1 and video[0].get("codec_name") == "h264"
                          and int(video[0].get("nb_frames", 0)) == 56
                          and (video[0].get("width"), video[0].get("height")) ==
                          (target_width, target_height),
        "one_aac_audio": len(audio) == 1 and audio[0].get("codec_name") == "aac",
        "full_av_decode": decode.returncode == 0,
        "private_nonempty": file.is_relative_to(root) and file.stat().st_size > 0,
    }
    return {"path": str(file), "sha256": s18._sha(file), "checks": checks,
            "decoded_md5": {stream: s18._decoded_md5(file, stream)
                            for stream in ("video", "audio")}}


def _progress(phase: dict) -> list[str]:
    return [str(event.get("node")) for event in phase.get("events") or []
            if event.get("type") == "progress" and str(event.get("node")) in
            {"12", "25", "28", "31"}]


def _relay(phase: dict, nodes: tuple[str, ...], indexes: tuple[int, ...]) -> bool:
    reports = [json.loads(capture._phase_text(phase, key)) for key in nodes]
    return len(reports) == len(indexes) and all(
        report.get("segment_index") == index and
        report.get("actual_calls") == {"completed_forwards": 4,
                                       "routed_attention_calls": 200}
        for report, index in zip(reports, indexes))


def _eav(phase: dict, nodes: tuple[str, ...], indexes: tuple[int, ...]) -> bool:
    reports = [json.loads(capture._phase_text(phase, key)) for key in nodes]
    return len(reports) == len(indexes) and all(
        report.get("chunked_segment_index") == index
        and report.get("status") == "observed_report_only"
        and report.get("config", {}).get("mode") == "report_only"
        and report.get("completed_forwards") == report.get("planned_forwards") == 4
        and report.get("selector_calls") == report.get("relay_attention_calls") == 200
        and report.get("relay_required") is True
        and report.get("clock_match") is True
        and report.get("quality_accepted") is False
        for report, index in zip(reports, indexes))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=95000)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--with-eav", action="store_true",
                        help="Use pinned external Relay+EAV report-only candidate on every segment")
    args = parser.parse_args(argv)
    args.comfy_root, args.python = args.comfy_root.resolve(), args.python.resolve()
    full = build_graph("full", width=args.width, height=args.height, with_eav=args.with_eav)
    build_graph("cold", width=args.width, height=args.height, with_eav=args.with_eav)
    ready = preflight(args)
    print(json.dumps(ready, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not ready["ready"]:
        return 0 if ready["ready"] else 2
    run_root = RUNS / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                       + ("-relay-eav-pair" if args.with_eav else "-pair"))
    full_root, cold_root = run_root / "full", run_root / "cold"
    full_root.mkdir(parents=True, exist_ok=False)
    cold_root.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "with_eav": args.with_eav,
              "preflight": ready, "test_dimensions": [args.width, args.height, 56]}
    try:
        image = args.comfy_root / "input" / s18.IMAGE
        image_sha = s18._sha(image)
        report["input_image_sha256"] = image_sha
        for root in (full_root, cold_root):
            (root / "input").mkdir()
            shutil.copy2(image, root / "input" / s18.IMAGE)
            if s18._sha(root / "input" / s18.IMAGE) != image_sha:
                raise ValueError("S18 private input image changed")
            (root / "paths.json").write_text(json.dumps(
                probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
        (full_root / "prompt.json").write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
        args.host, args.input_directory = "127.0.0.1", full_root / "input"
        args.extra_model_paths_config = full_root / "paths.json"
        with shared.IsolatedServer(args, full_root, "s18-full-three-segments") as server:
            report["full_core_pid"] = server.process.pid
            full_phase = asyncio.run(capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=full,
                timeout_seconds=args.timeout_seconds))
        (full_root / "phase.json").write_text(json.dumps(full_phase, ensure_ascii=False, indent=2), encoding="utf-8")
        if (full_phase.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("S18 uninterrupted full graph did not complete")
        receipts = s18._freeze_receipts(full_root, full_phase)
        final_full = _final_receipt(full_root, full_phase)
        media_full = _media(full_root, "full", args.width * 2, args.height * 2)
        report["full"] = {"receipts": receipts, "final": final_full, "media": media_full}
        _copy_receipts(full_root, cold_root, receipts)
        cold = build_graph("cold", width=args.width, height=args.height,
                           receipts=receipts, with_eav=args.with_eav)
        (cold_root / "prompt.json").write_text(json.dumps(cold, ensure_ascii=False, indent=2), encoding="utf-8")
        args.port += 1
        args.input_directory, args.extra_model_paths_config = cold_root / "input", cold_root / "paths.json"
        with shared.IsolatedServer(args, cold_root, "s18-cold-later-segments") as server:
            report["cold_core_pid"] = server.process.pid
            cold_phase = asyncio.run(capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=cold,
                timeout_seconds=args.timeout_seconds))
        (cold_root / "phase.json").write_text(json.dumps(cold_phase, ensure_ascii=False, indent=2), encoding="utf-8")
        if (cold_phase.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("S18 cold later-segments graph did not complete")
        final_cold = _final_receipt(cold_root, cold_phase)
        media_cold = _media(cold_root, "cold", args.width * 2, args.height * 2)
        report["cold"] = {"final": final_cold, "media": media_cold}
        full_effect_key = "full_three_eav_relay_audits" if args.with_eav else "full_three_relay_audits"
        cold_effect_key = "cold_two_eav_relay_audits" if args.with_eav else "cold_two_relay_audits"
        checks = {
            "full_and_cold_terminal_success": True,
            "full_four_sampler_stages": _progress(full_phase) ==
                ["12"] * 4 + ["25"] * 4 + ["28"] * 4 + ["31"] * 4,
            "cold_only_last_two_segments": _progress(cold_phase) == ["28"] * 4 + ["31"] * 4,
            "cold_graph_has_no_firstpass_or_segment0": "12" not in cold and "25" not in cold,
            full_effect_key: (_eav if args.with_eav else _relay)(
                full_phase, ("205", "206", "207"), (0, 1, 2)),
            cold_effect_key: (_eav if args.with_eav else _relay)(
                cold_phase, ("202", "203"), (1, 2)),
            "final_av_tensor_identity_equal": final_full["manifest"]["binding"]["output_identity"] ==
                final_cold["manifest"]["binding"]["output_identity"],
            "decoded_video_audio_equal": media_full["decoded_md5"] == media_cold["decoded_md5"],
            "both_media_valid": all(media_full["checks"].values()) and all(media_cold["checks"].values()),
            "source_input_and_candidates_unchanged": s18._sha(image) == image_sha and
                all(s18._sha(path) == digest for path, digest in s18.HASHES.items()) and
                (not args.with_eav or s18._sha(EAV_CANDIDATE) == EAV_CANDIDATE_SHA256),
            "both_owned_cores_stopped": not shared.port_is_listening("127.0.0.1", args.port - 1)
                                        and not shared.port_is_listening("127.0.0.1", args.port),
        }
        report["checks"] = checks
        report["status"] = (("real_s18_small_relay_eav_full_cold_exact_parity_not_quality_acceptance"
                             if args.with_eav else
                             "real_s18_small_three_segment_full_cold_exact_parity_not_quality_acceptance")
                            if all(checks.values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("S18 full/cold pair failed a mechanical check")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
