"""Real-weight S26 combined tail from a verified PDD freeze, in an owned Core.

Read-only by default. Explicit --confirm-run creates new private evidence and
three labeled media: original, unaccepted candidate, and default selection.
No PDD first pass, fake loader, altered resource gate, or quality acceptance.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import build_modular_audio_tail_effect_workflows as builder  # noqa: E402
from tools import run_modular_s26_pdd_resume_gpu as previous  # noqa: E402
from tools.build_formal_audio_tail_effect_workflows import DESTINATION, generated  # noqa: E402
from tools.build_modular_audio_refine_workflows import split_api  # noqa: E402
from tools.run_candidate_cpu_regression import source_snapshot  # noqa: E402
from tools.vdn_probe_environment import probe_resource_config  # noqa: E402

SCHEMA = "t8.s26.trained-combined-tail.v1"
SAVED = ROOT / "artifacts/development/modular-audio-tail-effects-20260928/matrix-v1"
REPORTS = {"route_report": "701", "quality_decision": "702", "load_status": "703",
           "eav": "704", "relay": "705", "guider": "706"}
BOUNDARY = ("Cold trained-weight tail on a small 22-frame frozen source; no first-pass "
            "rerun, original-size, full-vs-cold parity, sound quality or human acceptance. "
            "API execution evidence is separate from prior browser edit/save evidence.")


def _read(path):
    return json.loads(path.read_text(encoding="utf8"))


def _write(path, data):
    with path.open("x", encoding="utf8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _one(graph, kind):
    found = [key for key, node in graph.items() if node["class_type"] == kind]
    if len(found) != 1:
        raise ValueError(f"Expected one {kind}, got {found}")
    return found[0]


def saved_graph(route):
    stem = "S26_" + previous.freeze.ROUTES[route][0] + "_resume_audio_Separate_EXP_combined_TailEffects"
    path = DESTINATION / (stem + ".json")
    frontend = _read(path)
    if frontend != generated()[path] or frontend != _read(SAVED / path.name):
        raise ValueError("Saved public/candidate combined frontend changed")
    api_path = SAVED / (stem + ".api.json")
    return frontend, _read(api_path), {str(item): previous.freeze._sha(item)
        for item in (path, SAVED / path.name, api_path)}


def build_graph(route, frozen, *, eav_tau=.25):
    if type(eav_tau) not in (int, float) or not math.isfinite(eav_tau) or not -32 <= eav_tau <= 32:
        raise ValueError("Probe EAV tau must be finite and within the node's [-32,32] range")
    frontend, original, sources = saved_graph(route)
    graph = deepcopy(original)
    pins = previous.ROUTES[route]
    receipt = frozen["checkpoint"]
    frames = receipt["embedded_manifest"]["frame_count"]
    if frames != 22:
        raise ValueError("This initial trained probe requires its verified 22-frame freeze")
    graph[pins["load"]]["inputs"].update(checkpoint_path=receipt["relative_path"],
        expected_manifest_json=json.dumps(receipt["embedded_manifest"], ensure_ascii=False,
                                         sort_keys=True, separators=(",", ":")),
        expected_file_sha256=receipt["file_sha256"])
    graph[pins["guard"]]["inputs"]["expected_video_frame_count"] = frames
    plan, query = _one(graph, builder.RELAY_PLAN), _one(graph, builder.RELAY_ROUTE)
    cond, config = _one(graph, builder.RELAY_COND), _one(graph, builder.CONFIG)
    # Explicit probe-only choices; public defaults and the hard limit stay intact.
    graph[plan]["inputs"].update(local_prompts="Soft room ambience.\nClear footsteps.",
                                 timing_mode="auto_equal", time_ranges="")
    graph[query]["inputs"]["query_route"] = "joint_av_exp"
    graph[cond]["inputs"].update(execution_mode="apply_exp", query_chunk_rows=64)
    graph[config]["inputs"].update(mode="apply_exp", tau=float(eav_tau),
                                   start_video_progress=0., end_video_progress=1.)
    gate = _one(graph, "MiniMaxH3AudioRefineQualityGateT8Advanced")
    if graph[gate]["inputs"]["accept_candidate"] is not False:
        raise ValueError("Probe cannot accept candidate sound on the user's behalf")
    if graph[config]["inputs"]["g_hard_limit"] != 1.5:
        raise ValueError("Original EAV gain protection changed")
    save = graph[pins["save"]]
    create = graph[save["inputs"]["video"][0]]
    save["inputs"]["filename_prefix"] = f"MiniMaxH3/TailEffects/{route}_selected"
    for label, first in (("original", 801), ("candidate", 803)):
        decoder = graph[gate]["inputs"][label + "_audio"][0]
        graph[str(first)] = deepcopy(create)
        graph[str(first)]["inputs"].update(images=[decoder, 0], audio=[decoder, 1])
        graph[str(first + 1)] = deepcopy(save)
        graph[str(first + 1)]["inputs"].update(video=[str(first), 0],
            filename_prefix=f"MiniMaxH3/TailEffects/{route}_{label}")
    outputs = {"route_report": pins["route_report"], "quality_decision": pins["decision"],
        "load_status": [pins["load"], 1], "eav": [_one(graph, builder.AUDIT), 1],
        "relay": [cond, 6], "guider": [_one(graph, builder.GUIDER), 1]}
    for name, source in outputs.items():
        graph[REPORTS[name]] = {"class_type": "PreviewAny", "inputs": {"source": source}}
    if _one(graph, "SamplerCustomAdvanced") != pins["sampler"]:
        raise ValueError("Expected the saved graph's sole tail sampler")
    return frontend, original, graph, sources


def gain_observation(eav):
    low, high = (eav.get("feta", {}).get(key) for key in ("g_min", "g_max"))
    limit = eav.get("config", {}).get("g_hard_limit")
    bounded = (all(type(value) in (int, float) and math.isfinite(value) for value in (low, high, limit))
               and 1 <= low <= high <= limit == 1.5)
    return {"g_min": low, "g_max": high, "g_hard_limit": limit,
            "finite_within_original_hard_limit": bool(bounded),
            "non_identity_gain_observed": bool(bounded and high > 1.),
            "boundary": "Non-unit FETA gain proves multiplication, not improved sound or picture."}


def effect_checks(eav, relay, guider, *, require_non_identity=False):
    plan = eav.get("forward_plan", {})
    forwards = plan.get("forwards", [])
    attention = sum(item.get("attention_blocks", 0) for item in forwards)
    measurements = eav.get("feta", {}).get("forwards", [])
    checks = {
        "eav_observed_apply": eav.get("status") == "observed_apply_exp",
        "four_authenticated_forwards": eav.get("completed_forwards") ==
            eav.get("planned_forwards") == len(forwards) == 4 and plan.get("known") is True,
        "absolute_clock_match": eav.get("clock_match") is True,
        "all_relay_attention_observed": eav.get("relay_required") is True and attention > 0
            and eav.get("relay_attention_calls") == attention,
        "all_feta_attention_measured": len(measurements) == 4 and all(
            item.get("active") is True and item.get("attention_count") == expected.get("attention_blocks")
            for item, expected in zip(measurements, forwards)),
        "relay_applied_and_positive_paired": relay.get("status") == "applied_exp"
            and guider.get("external_positive") is True,
        "no_quality_or_portable_cache_claim": eav.get("quality_accepted") is False
            and eav.get("cache_reuse_authorized") is False,
        "gain_finite_within_original_hard_limit": gain_observation(eav)["finite_within_original_hard_limit"],
    }
    if require_non_identity:
        checks["non_identity_gain_observed"] = gain_observation(eav)["non_identity_gain_observed"]
    return checks


def media_audit(run_root, route):
    files, checks = {}, {}
    output = run_root / "output/MiniMaxH3/TailEffects"
    expected_canvas = 128 if route == "pdd8" else 192
    for label in ("original", "candidate", "selected"):
        candidates = list(output.glob(f"{route}_{label}_*.mp4"))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one owned {label} MP4")
        path = candidates[0].resolve()
        if not path.is_relative_to(run_root.resolve()):
            raise ValueError("Media escaped owned store")
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
                                str(path)], capture_output=True, text=True, check=True, timeout=30)
        metadata = json.loads(probe.stdout)
        video = [s for s in metadata["streams"] if s["codec_type"] == "video"]
        audio = [s for s in metadata["streams"] if s["codec_type"] == "audio"]
        checks[label + "_complete_av"] = (len(video) == len(audio) == 1 and
            video[0].get("codec_name") == "h264" and int(video[0].get("nb_frames", 0)) == 22 and
            (video[0].get("width"), video[0].get("height")) == (expected_canvas, expected_canvas) and
            audio[0].get("codec_name") == "aac" and video[0].get("avg_frame_rate") == "24/1" and
            math.isclose(float(video[0].get("duration", 0)), 22 / 24, abs_tol=.001))
        hashes = {}
        for kind, options in (("video", ["-map", "0:v:0", "-pix_fmt", "rgb24", "-f", "rawvideo"]),
                              ("audio", ["-map", "0:a:0", "-f", "f32le"])):
            decoded = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
                                      *options, "-"], capture_output=True, check=True, timeout=60)
            checks[label + "_" + kind + "_fully_decoded"] = bool(decoded.stdout)
            hashes[kind] = hashlib.sha256(decoded.stdout).hexdigest()
        files[label] = {"path": str(path), "sha256": previous.freeze._sha(path),
                        "decoded_sha256": hashes, "metadata": metadata}
    checks["default_selected_media_equals_original"] = (
        files["selected"]["decoded_sha256"] == files["original"]["decoded_sha256"])
    checks["candidate_decoded_audio_changed"] = (files["candidate"]["decoded_sha256"]["audio"] !=
                                                 files["original"]["decoded_sha256"]["audio"])
    return {"checks": checks, "files": files, "quality_accepted": False}


def preflight(args, graph, frozen_file, sources):
    report = previous.preflight(args, graph, frozen_file, next(iter(sources.values())))
    assets = {}
    for key, node in graph.items():
        for field, folder in (("unet_name", "models/diffusion_models"), ("clip_name", "models/text_encoders"),
                              ("vae_name", "models/vae"), ("lora_name", "models/loras"), ("image", "input")):
            value = node["inputs"].get(field)
            if isinstance(value, str):
                base = (args.comfy_root / folder).resolve()
                path = (base / value).resolve()
                if not path.is_relative_to(base):
                    raise ValueError("Asset escaped configured probe directory")
                assets[key + ":" + field] = str(path)
    report["assets"] = assets
    report["checks"].update(assets_exist=bool(assets) and all(Path(p).is_file() for p in assets.values()),
        media_tools_available=all(shutil.which(name) for name in ("ffprobe", "ffmpeg")))
    report["ready"] = all(report["checks"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=tuple(previous.ROUTES), required=True)
    parser.add_argument("--freeze-run-root", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8954)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--eav-tau", type=float, default=.25,
                        help="Explicit copied-graph test parameter; public defaults are never changed")
    parser.add_argument("--require-non-identity-gain", action="store_true",
                        help="Fail qualification unless measured FETA gain actually exceeds one")
    args = parser.parse_args()
    args.comfy_root = args.comfy_root.resolve()
    args.freeze_run_root = args.freeze_run_root.resolve()
    frozen, frozen_file = previous._freeze_evidence(args.freeze_run_root, args.route)
    frontend, original, graph, sources = build_graph(args.route, frozen, eav_tau=args.eav_tau)
    readiness = preflight(args, graph, frozen_file, sources)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_root = ROOT / "artifacts/development/modular-audio-tail-effects-20260928/trained" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.route)
    run_root.mkdir(parents=True, exist_ok=False)
    print("RUN_ROOT=" + str(run_root), flush=True)
    before = source_snapshot(ROOT)
    _write(run_root / "sources-before.json", before)
    _write(run_root / "prompt.json", graph)
    _write(run_root / "paths.json", probe_resource_config(args.comfy_root, ROOT))
    args.host = "127.0.0.1"
    args.extra_model_paths_config = run_root / "paths.json"
    receipt = frozen["checkpoint"]
    copied = previous._copy_freeze_to_owned_store(frozen_file, run_root,
        receipt["relative_path"], receipt["file_sha256"])
    report = {"schema": SCHEMA, "route": args.route, "status": "started", "sources": sources,
              "freeze_run_root": str(args.freeze_run_root), "preflight": readiness,
              "test_settings": {"tau": args.eav_tau, "require_non_identity": args.require_non_identity_gain,
                                "effect_progress": [0., 1.], "relay_route": "joint_av_exp"},
              "boundary": BOUNDARY, "checks": {}, "run_root": str(run_root)}
    server = previous.shared.IsolatedServer(args, run_root, "tail-effects")
    try:
        with server:
            report["owned_core_pid"] = server.process.pid
            url = f"http://127.0.0.1:{args.port}"
            with urllib.request.urlopen(url + "/object_info", timeout=30) as response:
                info = json.load(response)
            if split_api(frontend, info) != original:
                raise ValueError("Live Core serialization differs from saved public graph/API")
            report["checks"]["live_schema_exact_saved_api"] = True
            phase = asyncio.run(previous.pdd._submit_prompt_capture(server=url, prompt=graph,
                timeout_seconds=args.timeout_seconds))
        _write(run_root / "phase.json", phase)
        report["checks"].update(previous._tail_checks(args.route, phase))
        if report["checks"]["terminal_success"]:
            for name, key in REPORTS.items():
                report[name] = previous.pdd._phase_text(phase, key)
            report["checks"].update(previous._delivery_checks(report, phase))
            report["checks"].update(effect_checks(*[json.loads(report[k]) for k in ("eav", "relay", "guider")],
                require_non_identity=args.require_non_identity_gain))
            report["gain_observation"] = gain_observation(json.loads(report["eav"]))
            media = media_audit(run_root, args.route)
            _write(run_root / "media.json", media)
            report["checks"].update(media["checks"])
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["checks"].update(
            owned_process_stopped=server.process is None or server.process.poll() is not None,
            owned_port_stopped=not previous.shared.port_is_listening(args.host, args.port),
            sources_stable=source_snapshot(ROOT) == before,
            saved_graphs_stable=all(previous.freeze._sha(Path(p)) == digest for p, digest in sources.items()),
            original_freeze_unchanged=previous.freeze._sha(frozen_file) == receipt["file_sha256"],
            copied_freeze_unchanged=previous.freeze._sha(copied) == receipt["file_sha256"])
        report["status"] = ("trained_tail_mechanical_pass_not_quality_acceptance"
                            if "error" not in report and report["checks"].get("terminal_success")
                            and all(report["checks"].values()) else "fail")
        _write(run_root / "report.json", report)
        print(json.dumps({"status": report["status"], "run_root": str(run_root),
            "failed_checks": [k for k, v in report["checks"].items() if not v]}), flush=True)
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
