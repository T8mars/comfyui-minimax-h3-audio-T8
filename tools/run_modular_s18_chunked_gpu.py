"""Run the saved S18 freeze/resume pair on an owned Core with installed assets.

The default is read-only preflight. --confirm-run executes reduced-canvas copies
of the two SHA-pinned API candidates in separate isolated Core processes. The
source candidates, formal workflows, Director and user Core remain untouched.
This is a mechanical qualification, not original-size or perceptual acceptance.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as pdd  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s18-real-freeze-resume.v1"
CANDIDATES = PROJECT / "artifacts/development/modular-sampling-s18-v1-storage-20260924/candidate-v1"
FREEZE = CANDIDATES / "01_freeze_after_segment_0.api.json"
RESUME = CANDIDATES / "02_resume_segments_1_and_2_DRAFT.api.json"
HASHES = {
    FREEZE: "f941c3c5cbfb6616000ff4d92d9e280302bda79d8f38ecad89822344a9cfd91e",
    RESUME: "7a959b24176e65dd12bd3f8e969d4abbc04bc38c65410a03c2ba44238c765191",
}
IMAGE = "codex_prompt_relay_fl2va_first.png"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> dict:
    if _sha(path) != HASHES[path]:
        raise ValueError(f"S18 candidate changed: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_graph(kind: str, *, width: int, height: int) -> dict:
    if kind not in ("freeze", "resume") or min(width, height) < 64 or width % 32 or height % 32:
        raise ValueError("Use freeze/resume and a positive 32-aligned test canvas >=64")
    graph = deepcopy(_candidate(FREEZE if kind == "freeze" else RESUME))
    required = {
        "14": "MiniMaxH3ChunkedTwoPassPlanT8Advanced",
        "20": "MiniMaxH3ChunkedSourceSegmentEXPT8",
        "21": "MiniMaxH3ChunkedPass2PrepareEXPT8",
        "32": "MiniMaxH3PromptRelayPlanT8Advanced",
        "33": "MiniMaxH3PromptRelayConditioningT8Advanced",
    }
    required.update({
        "freeze": {"12": "SamplerCustomAdvanced",
                   "25": "MiniMaxH3ChunkedPass2SegmentEXPT8",
                   "40": "MiniMaxH3NativeLatentCheckpointSaveT8Advanced",
                   "41": "MiniMaxH3ChunkedV1SegmentSaveEXPT8"},
        "resume": {"28": "MiniMaxH3ChunkedPass2SegmentEXPT8",
                   "31": "MiniMaxH3ChunkedPass2SegmentEXPT8",
                   "40": "MiniMaxH3NativeLatentCheckpointLoadT8Advanced",
                   "41": "MiniMaxH3ChunkedV1SegmentLoadEXPT8",
                   "19": "VHS_VideoCombine"},
    }[kind])
    if any(graph.get(node, {}).get("class_type") != name for node, name in required.items()):
        raise ValueError("S18 disk candidate lost an expected stage")
    if graph["14"]["inputs"]["spatial_strategy"] != "full_frame_safe" or (
            graph["32"]["inputs"]["length"] != 56):
        raise ValueError("S18 original full-frame/56-frame recipe changed")
    if kind == "resume" and ("12" in graph or "25" in graph):
        raise ValueError("Saved resume graph unexpectedly includes first pass or segment 0")
    target_width, target_height = width * 2, height * 2
    if "6" in graph:
        graph["6"]["inputs"].update(width=width, height=height)
    if "7" in graph:
        graph["7"]["inputs"].update(width=width, height=height)
    graph["14"]["inputs"].update(target_width=target_width, target_height=target_height,
                                  tile_width=target_width, tile_height=target_height,
                                  minimum_tile_size=64)
    graph["33"]["inputs"].update(width=target_width, height=target_height)
    if kind == "freeze":
        graph["40"]["inputs"].update(confirm_save=True, filename_prefix="s18_real_firstpass")
        graph["41"]["inputs"]["confirm_save"] = True
        previews = {"200": ["40", 4], "201": ["40", 5], "202": ["41", 2],
                    "203": ["41", 3], "204": ["41", 4], "205": ["35", 1]}
    else:
        graph["19"]["inputs"]["filename_prefix"] = "MiniMaxH3/S18_Chunked_Real/resume"
        previews = {"200": ["40", 7], "201": ["41", 2], "202": ["37", 1],
                    "203": ["39", 1]}
    for node_id, source in previews.items():
        graph[node_id] = {"class_type": "PreviewAny", "inputs": {"source": source}}
    return graph


def preflight(args: argparse.Namespace) -> dict:
    models = args.comfy_root / "models"
    source = _candidate(FREEZE)
    _candidate(RESUME)
    assets = {
        "base": models / "diffusion_models" / source["4"]["inputs"]["unet_name"],
        "clip": models / "text_encoders" / source["3"]["inputs"]["clip_name"],
        "video_vae": models / "vae" / source["1"]["inputs"]["vae_name"],
        "audio_vae": models / "vae" / source["2"]["inputs"]["vae_name"],
        "pdd_adapter": models / "loras" / source["8"]["inputs"]["pdd_lora_name"],
        "trained_3d_upscaler": models / "latent_upscale_models" /
                              source["14"]["inputs"]["model_name"],
        "first_last_frame": args.comfy_root / "input" / IMAGE,
    }
    gpu = shared.gpu_memory_mib()
    checks = {
        "both_candidates_sha_pinned": True,
        "all_assets_installed": all(path.is_file() for path in assets.values()),
        "private_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "gpu_available": bool(gpu.get("available")),
        "free_vram_gate": bool(gpu.get("available") and
                               gpu["free_mib"] >= args.min_free_vram_mib),
    }
    return {"schema": SCHEMA + ".preflight", "candidate_sha256":
            {"freeze": HASHES[FREEZE], "resume": HASHES[RESUME]},
            "assets": {key: str(value) for key, value in assets.items()},
            "gpu": gpu, "checks": checks, "ready": all(checks.values())}


def _text(phase: dict, node_id: str) -> str:
    return pdd._phase_text(phase, node_id)


def _freeze_receipts(run_root: Path, phase: dict) -> dict:
    native_manifest = _text(phase, "200")
    native_report = json.loads(_text(phase, "201"))
    segment_path = _text(phase, "202")
    segment_sha = _text(phase, "203")
    segment_manifest = _text(phase, "204")
    if native_report.get("status") != "SAVED_VERIFIED":
        raise ValueError("Native first-pass checkpoint did not verify")
    native_root = (run_root / "output/MiniMaxH3/latent_checkpoints").resolve()
    native_path = (native_root / native_report["checkpoint_path"]).resolve()
    segment_root = (run_root / "output/MiniMaxH3/chunked_v1_segment_artifacts").resolve()
    segment_file = (segment_root / segment_path).resolve()
    if (not native_path.is_relative_to(native_root) or not native_path.is_file()
            or not segment_file.is_relative_to(segment_root) or not segment_file.is_file()):
        raise ValueError("S18 freeze receipt path missing or escaped private output")
    if _sha(native_path).upper() != native_report["file_sha256"].upper():
        raise ValueError("Native Save file SHA differs from receipt")
    if _sha(segment_file).lower() != segment_sha.lower():
        raise ValueError("Segment Save file SHA differs from receipt")
    if json.loads(segment_manifest) != json.loads(segment_file.read_text(encoding="utf-8")):
        raise ValueError("Segment Save manifest differs from disk")
    if not isinstance(json.loads(native_manifest), dict):
        raise ValueError("Native Save manifest is not JSON object")
    return {"native": {"path": native_report["checkpoint_path"],
                       "sha256": native_report["file_sha256"],
                       "manifest_json": native_manifest},
            "segment": {"path": segment_path, "sha256": segment_sha,
                        "manifest_json": segment_manifest},
            "relay_audit": json.loads(_text(phase, "205"))}


def _fill_resume(graph: dict, receipts: dict) -> dict:
    graph["40"]["inputs"].update(
        checkpoint_path=receipts["native"]["path"],
        expected_manifest_json=receipts["native"]["manifest_json"],
        expected_file_sha256=receipts["native"]["sha256"],
    )
    graph["41"]["inputs"].update(
        artifact_path=receipts["segment"]["path"],
        artifact_sha256=receipts["segment"]["sha256"],
    )
    return graph


def _stage_checks(freeze: dict, resume: dict) -> dict[str, bool]:
    def progress(phase):
        return [str(event.get("node")) for event in phase.get("events", [])
                if event.get("type") == "progress" and
                str(event.get("node")) in {"12", "25", "28", "31"}]

    freeze_progress, resume_progress = progress(freeze), progress(resume)
    return {
        "freeze_terminal_success": (freeze.get("terminal") or {}).get("type") == "execution_success",
        "resume_terminal_success": (resume.get("terminal") or {}).get("type") == "execution_success",
        "freeze_firstpass_then_segment0": freeze_progress == ["12"] * 4 + ["25"] * 4,
        "resume_only_segments1_and2": resume_progress == ["28"] * 4 + ["31"] * 4,
    }


def _media_checks(run_root: Path, *, width: int, height: int,
                  prefix: str = "resume") -> dict:
    output = run_root / "output/MiniMaxH3/S18_Chunked_Real"
    if prefix not in {"resume", "cancel_retry"} and not re.fullmatch(
            r"cache_v[2-9][0-9]*_(base|repeat|late|early)", prefix):
        raise ValueError("Only known owned S18 media prefixes are auditable")
    files = list(output.glob(f"{prefix}_*-audio.mp4"))
    if len(files) != 1:
        raise ValueError("Expected exactly one owned S18 H.264/AAC output")
    media = files[0]
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(media)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    video = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(media),
         "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
        capture_output=True, timeout=90, check=False,
    )
    return {
        "path": str(media),
        "sha256": _sha(media),
        "checks": {
            "h264_56_frames_scaled_canvas": len(video) == 1 and
                video[0].get("codec_name") == "h264" and
                int(video[0].get("nb_frames", 0)) == 56 and
                (video[0].get("width"), video[0].get("height")) == (width, height),
            "one_aac_audio": len(audio) == 1 and audio[0].get("codec_name") == "aac",
            "full_decode": decode.returncode == 0,
            "nonempty_private_file": media.stat().st_size > 0 and
                media.resolve().is_relative_to(run_root.resolve()),
        },
    }


def audit_existing(run_root: Path, *, port: int) -> dict:
    run_root = run_root.resolve()
    parent = PROJECT / "artifacts/development/modular-sampling-s18-real-gpu-20260924"
    if run_root.parent != parent.resolve():
        raise ValueError("Audit target must be an owned S18 run directory")
    freeze = json.loads((run_root / "freeze-phase.json").read_text(encoding="utf-8"))
    resume = json.loads((run_root / "resume-phase.json").read_text(encoding="utf-8"))
    freeze_prompt = json.loads((run_root / "freeze-prompt.json").read_text(encoding="utf-8"))
    frozen = _freeze_receipts(run_root, freeze)
    prompt = json.loads((run_root / "resume-prompt.json").read_text(encoding="utf-8"))
    native_load = json.loads(_text(resume, "200"))
    segment_load = json.loads(_text(resume, "201"))
    relay = [frozen["relay_audit"], json.loads(_text(resume, "202")),
             json.loads(_text(resume, "203"))]
    target = freeze_prompt["14"]["inputs"]
    media = _media_checks(run_root, width=target["target_width"],
                          height=target["target_height"])
    checks = _stage_checks(freeze, resume)
    checks.update({
        "resume_graph_omits_firstpass_and_segment0": "12" not in prompt and "25" not in prompt,
        "resume_uses_exact_native_receipt": prompt["40"]["inputs"]["checkpoint_path"] ==
            frozen["native"]["path"] and
            prompt["40"]["inputs"]["expected_file_sha256"] == frozen["native"]["sha256"] and
            prompt["40"]["inputs"]["expected_manifest_json"] == frozen["native"]["manifest_json"],
        "resume_uses_exact_segment_receipt": prompt["41"]["inputs"]["artifact_path"] ==
            frozen["segment"]["path"] and
            prompt["41"]["inputs"]["artifact_sha256"] == frozen["segment"]["sha256"],
        "native_load_verified_external": native_load.get("status") == "MATCH_EXTERNAL" and
            native_load.get("external_manifest_verified") is True,
        "segment0_load_verified": segment_load.get("status") ==
            "explicit_frozen_chunked_v1_segment_loaded",
        "three_segment_relay_calls": [item.get("segment_index") for item in relay] == [0, 1, 2] and
            all(item.get("actual_calls") ==
                {"completed_forwards": 4, "routed_attention_calls": 200} for item in relay),
        "candidate_sources_unchanged": all(_sha(path) == value for path, value in HASHES.items()),
        "owned_server_stopped": not shared.port_is_listening("127.0.0.1", port),
        **media["checks"],
    })
    result = {"schema": SCHEMA + ".stage-audit", "run_root": str(run_root),
              "checks": checks, "media": {"path": media["path"], "sha256": media["sha256"]},
              "status": "small_canvas_mechanical_pass_not_quality_acceptance"
                        if all(checks.values()) else "fail",
              "boundary": "Two original SHA-pinned API candidates ran only as reduced-canvas copies; "
                          "pretrained assets and original VHS ran, but no original-size or human quality gate."}
    output = run_root / "stage-audit-v1.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite S18 stage audit")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _negative_variants(resume: dict) -> dict[str, dict]:
    variants = {}
    for name in ("bad_native_file_sha", "bad_native_manifest", "bad_segment_sha"):
        graph = deepcopy(resume)
        graph["19"]["inputs"]["filename_prefix"] = f"MiniMaxH3/S18_Chunked_Real/{name}"
        if name == "bad_native_file_sha":
            graph["40"]["inputs"]["expected_file_sha256"] = "0" * 64
        elif name == "bad_native_manifest":
            graph["40"]["inputs"]["expected_manifest_json"] = "{}"
        else:
            graph["41"]["inputs"]["artifact_sha256"] = "0" * 64
        variants[name] = graph
    return variants


NEGATIVE_ERRORS = {
    "bad_native_file_sha": ("40", "native H3 checkpoint file SHA-256 mismatch"),
    "bad_native_manifest": ("40", "expected_manifest_json must use schema"),
    "bad_segment_sha": ("41", "Chunked v1 frozen manifest SHA mismatch"),
}


def _negative_phase_checks(phase: dict, name: str) -> dict[str, bool]:
    expected_node, expected_message = NEGATIVE_ERRORS[name]
    events = phase.get("events") or []
    sampler_events = [event for event in events
                      if event.get("type") in {"executing", "progress"} and
                      str(event.get("node")) in {"28", "31"}]
    error = (phase.get("terminal") or {}).get("data") or {}
    return {
        "rejected_by_core": (phase.get("terminal") or {}).get("type") == "execution_error",
        "expected_guard_node_and_reason": str(error.get("node_id")) == expected_node and
            error.get("exception_type") == "ValueError" and
            str(error.get("exception_message") or "").startswith(expected_message),
        "no_later_segment_sampler_started": not sampler_events,
    }


def audit_negative_existing(run_root: Path, *, port: int) -> dict:
    run_root = run_root.resolve()
    parent = (PROJECT / "artifacts/development/modular-sampling-s18-real-gpu-20260924").resolve()
    if run_root.parent != parent:
        raise ValueError("Negative audit target must be an owned S18 run directory")
    original = json.loads((run_root / "negative-v1.json").read_text(encoding="utf-8"))
    if original.get("status") != "three_real_core_bad_receipts_rejected_before_sampling":
        raise ValueError("S18 negative run did not reach its recorded terminal")
    resume = json.loads((run_root / "resume-prompt.json").read_text(encoding="utf-8"))
    artifacts = {
        "media": Path(original["run_root"]) / "output/MiniMaxH3/S18_Chunked_Real/resume_00001-audio.mp4",
        "native": (run_root / "output/MiniMaxH3/latent_checkpoints" /
                   resume["40"]["inputs"]["checkpoint_path"]),
        "segment": (run_root / "output/MiniMaxH3/chunked_v1_segment_artifacts" /
                    resume["41"]["inputs"]["artifact_path"]),
    }
    checks = {}
    reasons = {}
    for name in NEGATIVE_ERRORS:
        phase = json.loads((run_root / f"negative-{name}-phase.json").read_text(encoding="utf-8"))
        observed = _negative_phase_checks(phase, name)
        checks[name] = all(observed.values())
        terminal = phase.get("terminal") or {}
        reasons[name] = {
            "node_id": (terminal.get("data") or {}).get("node_id"),
            "exception_type": (terminal.get("data") or {}).get("exception_type"),
            "exception_message": (terminal.get("data") or {}).get("exception_message"),
            "checks": observed,
        }
    checks["positive_files_unchanged"] = all(
        path.resolve().is_relative_to(run_root) and path.is_file() and
        _sha(path) == original["source_file_sha256"][name]
        for name, path in artifacts.items())
    checks["candidate_sources_unchanged"] = all(_sha(path) == value
                                                for path, value in HASHES.items())
    checks["owned_core_stopped"] = not shared.port_is_listening("127.0.0.1", port)
    result = {"schema": SCHEMA + ".negative-reasons-audit", "run_root": str(run_root),
              "checks": checks, "reasons": reasons,
              "status": "pass_exact_guard_reasons_no_sampling" if all(checks.values()) else "fail",
              "boundary": "Real Core receipt failures only; not a cancellation or changed-model "
                          "cache-invalidation qualification."}
    output = run_root / "negative-audit-v2.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite S18 negative reasons audit")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_negative(args: argparse.Namespace) -> dict:
    run_root = args.negative_run_root.resolve()
    parent = (PROJECT / "artifacts/development/modular-sampling-s18-real-gpu-20260924").resolve()
    if run_root.parent != parent:
        raise ValueError("Negative probe requires an owned S18 run directory")
    evidence = json.loads((run_root / "stage-audit-v1.json").read_text(encoding="utf-8"))
    if evidence.get("status") != "small_canvas_mechanical_pass_not_quality_acceptance":
        raise ValueError("Negative probe requires a passing S18 positive-stage audit")
    if not (run_root / "input" / IMAGE).is_file() or not (run_root / "paths.json").is_file():
        raise ValueError("Owned S18 input/model-path profile is missing")
    if (run_root / "negative-v1.json").exists():
        raise FileExistsError("Refusing to overwrite S18 negative receipt audit")
    if (any(run_root.glob("negative-*-phase.json")) or
            any((run_root / "logs").glob("s18-negative-receipts.*.log"))):
        raise FileExistsError("Partial S18 negative run exists; preserve and inspect it first")
    resume = json.loads((run_root / "resume-prompt.json").read_text(encoding="utf-8"))
    variants = _negative_variants(resume)
    artifacts = {
        "media": Path(evidence["media"]["path"]),
        "native": (run_root / "output/MiniMaxH3/latent_checkpoints" /
                   resume["40"]["inputs"]["checkpoint_path"]),
        "segment": (run_root / "output/MiniMaxH3/chunked_v1_segment_artifacts" /
                    resume["41"]["inputs"]["artifact_path"]),
    }
    if not all(path.resolve().is_relative_to(run_root) and path.is_file()
               for path in artifacts.values()):
        raise ValueError("Owned positive media/checkpoint files are missing or escaped")
    frozen_shas = {name: _sha(path) for name, path in artifacts.items()}
    if (frozen_shas["media"] != evidence["media"]["sha256"] or
            not all(_sha(path) == value for path, value in HASHES.items())):
        raise ValueError("S18 positive evidence or source candidates changed")
    gpu = shared.gpu_memory_mib()
    readiness = {
        "private_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "gpu_free_vram_gate": bool(gpu.get("available") and
                                   gpu["free_mib"] >= args.min_free_vram_mib),
    }
    if not args.confirm_run:
        return {"status": "preflight_only", "checks": readiness, "gpu": gpu,
                "run_root": str(run_root)}
    if not all(readiness.values()):
        raise RuntimeError("S18 negative probe resource gate did not pass")
    args.host = "127.0.0.1"
    args.input_directory = run_root / "input"
    args.extra_model_paths_config = run_root / "paths.json"
    report = {"schema": SCHEMA + ".negative-receipts", "run_root": str(run_root),
              "preflight": readiness, "source_file_sha256": frozen_shas,
              "variants": {}, "status": "started"}
    try:
        with shared.IsolatedServer(args, run_root, "s18-negative-receipts") as server:
            report["owned_core_pid"] = server.process.pid
            for name, graph in variants.items():
                phase = asyncio.run(pdd._submit_prompt_capture(
                    server=f"http://127.0.0.1:{args.port}", prompt=graph,
                    timeout_seconds=args.timeout_seconds))
                with (run_root / f"negative-{name}-phase.json").open(
                        "x", encoding="utf-8") as stream:
                    json.dump(phase, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
                checks = _negative_phase_checks(phase, name)
                report["variants"][name] = {
                    "terminal": (phase.get("terminal") or {}).get("type"),
                    "checks": checks, "prompt_id": phase.get("prompt_id"),
                }
                if not all(checks.values()):
                    raise RuntimeError(f"S18 negative guard failed: {name}")
        report["checks"] = {
            "three_bad_receipts_rejected": len(report["variants"]) == 3 and
                all(all(item["checks"].values()) for item in report["variants"].values()),
            "positive_files_unchanged": all(_sha(path) == frozen_shas[name]
                                            for name, path in artifacts.items()),
            "candidate_sources_unchanged": all(_sha(path) == value
                                               for path, value in HASHES.items()),
            "owned_server_stopped": not shared.port_is_listening("127.0.0.1", args.port),
        }
        if not all(report["checks"].values()):
            raise ValueError("S18 negative receipt matrix failed final identity checks")
        report["status"] = "three_real_core_bad_receipts_rejected_before_sampling"
        return report
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "negative-v1.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _cancel_phase_checks(interrupted: dict, retry: dict) -> dict[str, bool]:
    events = interrupted.get("events") or []
    sampler_events = [str(event.get("node")) for event in events
                      if event.get("type") in {"executing", "progress"}]
    retry_progress = [str(event.get("node")) for event in retry.get("events") or []
                      if event.get("type") == "progress" and
                      str(event.get("node")) in {"12", "25", "28", "31"}]
    progress = interrupted.get("progress_at_interrupt") or {}
    return {
        "interrupt_acknowledged_during_segment1":
            interrupted.get("interrupt_response") is not None and
            int(progress.get("value") or 0) >= 1 and
            (interrupted.get("terminal") or {}).get("type") == "execution_interrupted" and
            "28" in sampler_events,
        "interrupted_before_segment2": "31" not in sampler_events,
        "no_firstpass_or_segment0_rerun": not ({"12", "25"} & set(sampler_events)) and
            "12" not in retry_progress and "25" not in retry_progress,
        "fresh_core_retry_only_later_segments":
            (retry.get("terminal") or {}).get("type") == "execution_success" and
            retry_progress == ["28"] * 4 + ["31"] * 4,
    }


def _decoded_md5(media: Path, stream: str) -> str:
    if stream not in {"video", "audio"}:
        raise ValueError("S18 decoded digest supports only video or audio")
    command = ["ffmpeg", "-v", "error", "-i", str(media),
               "-map", "0:v:0" if stream == "video" else "0:a:0"]
    command += (["-pix_fmt", "rgb24"] if stream == "video" else
                ["-acodec", "pcm_s16le"])
    command += ["-f", "md5", "-"]
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=120, check=True)
    digest = result.stdout.strip()
    if not digest.startswith("MD5=") or len(digest) != 36 or any(
            char not in "0123456789abcdef" for char in digest[4:].lower()):
        raise ValueError(f"Invalid ffmpeg {stream} decoded digest")
    return digest[4:].lower()


def audit_cancel_existing(run_root: Path, *, port: int) -> dict:
    run_root = run_root.resolve()
    parent = (PROJECT / "artifacts/development/modular-sampling-s18-real-gpu-20260924").resolve()
    if run_root.parent != parent:
        raise ValueError("Cancellation audit requires an owned S18 run directory")
    report = json.loads((run_root / "cancel-v1.json").read_text(encoding="utf-8"))
    if report.get("status") != "interrupted_segment1_then_fresh_core_retry_pass":
        raise ValueError("S18 cancellation run did not reach its recorded terminal")
    interrupted = json.loads((run_root / "cancel-interrupt-phase.json").read_text(
        encoding="utf-8"))
    retry = json.loads((run_root / "cancel-retry-phase.json").read_text(encoding="utf-8"))
    prompt = json.loads((run_root / "cancel-retry-prompt.json").read_text(encoding="utf-8"))
    original_prompt = json.loads((run_root / "resume-prompt.json").read_text(
        encoding="utf-8"))
    positive = json.loads((run_root / "stage-audit-v1.json").read_text(encoding="utf-8"))
    media = {"original": Path(positive["media"]["path"]),
             "retry": Path(report["media"]["path"])}
    checks = _cancel_phase_checks(interrupted, retry)
    checks.update({
        "retry_graph_omits_earlier_samplers": "12" not in prompt and "25" not in prompt,
        "same_explicit_load_inputs": all(prompt[node]["inputs"] ==
                                          original_prompt[node]["inputs"]
                                          for node in ("40", "41")),
        "media_sha_unchanged_since_run": all(
            path.is_file() and path.resolve().is_relative_to(run_root) and
            _sha(path) == (positive["media"]["sha256"] if name == "original" else
                           report["media"]["sha256"])
            for name, path in media.items()),
        "candidate_sources_unchanged": all(_sha(path) == value for path, value in HASHES.items()),
        "owned_core_stopped": not shared.port_is_listening("127.0.0.1", port),
    })
    decoded = {name: {stream: _decoded_md5(path, stream) for stream in ("video", "audio")}
               for name, path in media.items()} if checks["media_sha_unchanged_since_run"] else {}
    checks["decoded_video_identical"] = bool(decoded) and (
        decoded["original"]["video"] == decoded["retry"]["video"])
    checks["decoded_audio_identical"] = bool(decoded) and (
        decoded["original"]["audio"] == decoded["retry"]["audio"])
    result = {"schema": SCHEMA + ".cancellation-decoded-audit", "run_root": str(run_root),
              "checks": checks, "decoded_md5": decoded,
              "status": "decoded_av_identity_after_real_interrupt_fresh_core_retry"
                        if all(checks.values()) else "fail",
              "boundary": "Reduced canvas, one interruption in segment 1; not other segment, "
                          "configuration-change, multi-tile, full-size, or human qualification."}
    output = run_root / "cancel-audit-v2.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite S18 cancellation decoded audit")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_cancel(args: argparse.Namespace) -> dict:
    run_root = args.cancel_run_root.resolve()
    parent = (PROJECT / "artifacts/development/modular-sampling-s18-real-gpu-20260924").resolve()
    if run_root.parent != parent:
        raise ValueError("Cancellation probe requires an owned S18 run directory")
    evidence = json.loads((run_root / "stage-audit-v1.json").read_text(encoding="utf-8"))
    if evidence.get("status") != "small_canvas_mechanical_pass_not_quality_acceptance":
        raise ValueError("Cancellation probe requires a passing S18 positive-stage audit")
    if not (run_root / "input" / IMAGE).is_file() or not (run_root / "paths.json").is_file():
        raise ValueError("Owned S18 input/model-path profile is missing")
    if ((run_root / "cancel-v1.json").exists() or
            any(run_root.glob("cancel-*-phase.json")) or
            any((run_root / "logs").glob("s18-cancel.*.log")) or
            any((run_root / "logs").glob("s18-cancel-retry.*.log"))):
        raise FileExistsError("Partial S18 cancellation run exists; preserve and inspect it")
    resume = json.loads((run_root / "resume-prompt.json").read_text(encoding="utf-8"))
    if "12" in resume or "25" in resume:
        raise ValueError("S18 resume graph includes first pass or segment 0")
    artifacts = {
        "media": Path(evidence["media"]["path"]),
        "native": (run_root / "output/MiniMaxH3/latent_checkpoints" /
                   resume["40"]["inputs"]["checkpoint_path"]),
        "segment": (run_root / "output/MiniMaxH3/chunked_v1_segment_artifacts" /
                    resume["41"]["inputs"]["artifact_path"]),
    }
    if not all(path.resolve().is_relative_to(run_root) and path.is_file()
               for path in artifacts.values()):
        raise ValueError("Owned positive media/checkpoint files are missing or escaped")
    frozen_shas = {name: _sha(path) for name, path in artifacts.items()}
    if (frozen_shas["media"] != evidence["media"]["sha256"] or
            not all(_sha(path) == value for path, value in HASHES.items())):
        raise ValueError("S18 positive evidence or source candidates changed")
    gpu = shared.gpu_memory_mib()
    readiness = {
        "private_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "gpu_free_vram_gate": bool(gpu.get("available") and
                                   gpu["free_mib"] >= args.min_free_vram_mib),
    }
    if not args.confirm_run:
        return {"status": "preflight_only", "checks": readiness, "gpu": gpu,
                "run_root": str(run_root)}
    if not all(readiness.values()):
        raise RuntimeError("S18 cancellation probe resource gate did not pass")
    args.host = "127.0.0.1"
    args.input_directory = run_root / "input"
    args.extra_model_paths_config = run_root / "paths.json"
    retry_graph = deepcopy(resume)
    retry_graph["19"]["inputs"]["filename_prefix"] = (
        "MiniMaxH3/S18_Chunked_Real/cancel_retry")
    report = {"schema": SCHEMA + ".cancellation-retry", "run_root": str(run_root),
              "preflight": readiness, "source_file_sha256": frozen_shas,
              "status": "started"}
    try:
        with shared.IsolatedServer(args, run_root, "s18-cancel") as server:
            report["interrupt_core_pid"] = server.process.pid
            interrupted = asyncio.run(shared.submit_prompt(
                server=f"http://127.0.0.1:{args.port}", prompt=resume,
                timeout_seconds=args.timeout_seconds,
                interrupt_node="28", interrupt_after_step=1))
        with (run_root / "cancel-interrupt-phase.json").open(
                "x", encoding="utf-8") as stream:
            json.dump(interrupted, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        if (interrupted.get("terminal") or {}).get("type") != "execution_interrupted":
            raise RuntimeError("S18 owned Core did not interrupt segment 1")
        with (run_root / "cancel-retry-prompt.json").open(
                "x", encoding="utf-8") as stream:
            json.dump(retry_graph, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        with shared.IsolatedServer(args, run_root, "s18-cancel-retry") as server:
            report["retry_core_pid"] = server.process.pid
            retry = asyncio.run(pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=retry_graph,
                timeout_seconds=args.timeout_seconds))
        with (run_root / "cancel-retry-phase.json").open(
                "x", encoding="utf-8") as stream:
            json.dump(retry, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        target = json.loads((run_root / "freeze-prompt.json").read_text(
            encoding="utf-8"))["14"]["inputs"]
        media = _media_checks(run_root, width=target["target_width"],
                              height=target["target_height"], prefix="cancel_retry")
        checks = _cancel_phase_checks(interrupted, retry)
        checks.update({
            "same_verified_native_receipt": json.loads(_text(retry, "200")).get("status") ==
                "MATCH_EXTERNAL",
            "same_verified_segment_receipt": json.loads(_text(retry, "201")).get("status") ==
                "explicit_frozen_chunked_v1_segment_loaded",
            "positive_files_unchanged": all(_sha(path) == frozen_shas[name]
                                            for name, path in artifacts.items()),
            "candidate_sources_unchanged": all(_sha(path) == value
                                                for path, value in HASHES.items()),
            "owned_core_stopped": not shared.port_is_listening("127.0.0.1", args.port),
            **media["checks"],
        })
        report["checks"] = checks
        report["media"] = {"path": media["path"], "sha256": media["sha256"]}
        if not all(checks.values()):
            raise ValueError("S18 cancellation/retry mechanical audit failed")
        report["status"] = "interrupted_segment1_then_fresh_core_retry_pass"
        return report
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "cancel-v1.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_variants(resume: dict, *, attempt: int = 2) -> dict[str, dict]:
    if not 2 <= attempt <= 99:
        raise ValueError("S18 cache attempt must be 2..99")
    if "12" in resume or "25" in resume:
        raise ValueError("S18 cache probe cannot contain first pass or segment 0")
    original_noise = resume["11"]
    if original_noise["class_type"] != "RandomNoise":
        raise ValueError("S18 cache probe needs the original explicit noise node")
    base = deepcopy(resume)
    for node_id in ("70", "71"):
        if node_id in base:
            raise ValueError("S18 cache probe node IDs are already occupied")
        base[node_id] = deepcopy(original_noise)
    base["28"]["inputs"]["noise"] = ["71", 0]
    base["31"]["inputs"]["noise"] = ["70", 0]
    variants = {}
    for name in ("base", "repeat", "late", "early"):
        graph = deepcopy(base)
        graph["19"]["inputs"]["filename_prefix"] = (
            f"MiniMaxH3/S18_Chunked_Real/cache_v{attempt}_{name}")
        if name in {"late", "early"}:
            graph["70"]["inputs"]["noise_seed"] += 1
        if name == "early":
            graph["71"]["inputs"]["noise_seed"] += 1
        variants[name] = graph
    return variants


def _cache_phase_checks(phases: dict[str, dict]) -> dict[str, bool]:
    expected = {"base": ["28"] * 4 + ["31"] * 4, "repeat": [],
                "late": ["31"] * 4, "early": ["28"] * 4 + ["31"] * 4}
    checks = {}
    for name, wanted in expected.items():
        phase = phases.get(name) or {}
        progress = [str(event.get("node")) for event in phase.get("events") or []
                    if event.get("type") == "progress" and
                    str(event.get("node")) in {"12", "25", "28", "31"}]
        checks[name + "_only_expected_sampling"] = (
            (phase.get("terminal") or {}).get("type") == "execution_success" and
            progress == wanted)
    return checks


def audit_cache_existing(run_root: Path, *, attempt: int, port: int) -> dict:
    run_root = run_root.resolve()
    parent = (PROJECT / "artifacts/development/modular-sampling-s18-real-gpu-20260924").resolve()
    if run_root.parent != parent:
        raise ValueError("Cache audit requires an owned S18 run directory")
    report = json.loads((run_root / f"cache-v{attempt}.json").read_text(encoding="utf-8"))
    if report.get("status") != "same_core_late_only_then_early_dependency_pass":
        raise ValueError("S18 cache run did not reach its recorded terminal")
    positive = json.loads((run_root / "stage-audit-v1.json").read_text(encoding="utf-8"))
    resume = json.loads((run_root / "resume-prompt.json").read_text(encoding="utf-8"))
    expected = _cache_variants(resume, attempt=attempt)
    phases = {name: json.loads((run_root / f"cache-v{attempt}-{name}-phase.json").read_text(
        encoding="utf-8")) for name in expected}
    actual_prompts = {name: json.loads((run_root / f"cache-v{attempt}-{name}-prompt.json")
                                      .read_text(encoding="utf-8")) for name in expected}
    media = {"original": Path(positive["media"]["path"]),
             **{name: Path(report["media"][name]["path"]) for name in expected}}
    native = (run_root / "output/MiniMaxH3/latent_checkpoints" /
              resume["40"]["inputs"]["checkpoint_path"])
    segment = (run_root / "output/MiniMaxH3/chunked_v1_segment_artifacts" /
               resume["41"]["inputs"]["artifact_path"])
    checks = _cache_phase_checks(phases)
    checks.update({
        "exact_private_graph_edits": actual_prompts == expected,
        "media_sha_unchanged_since_run": all(
            path.is_file() and path.resolve().is_relative_to(run_root) and
            _sha(path) == (positive["media"]["sha256"] if name == "original" else
                           report["media"][name]["sha256"])
            for name, path in media.items()),
        "frozen_receipts_unchanged": all(
            path.is_file() and path.resolve().is_relative_to(run_root) and
            _sha(path) == report["source_file_sha256"][name]
            for name, path in (("native", native), ("segment", segment))),
        "candidate_sources_unchanged": all(_sha(path) == value for path, value in HASHES.items()),
        "owned_core_stopped": not shared.port_is_listening("127.0.0.1", port),
    })
    decoded = ({name: {stream: _decoded_md5(path, stream) for stream in ("video", "audio")}
                for name, path in media.items()} if checks["media_sha_unchanged_since_run"] else {})
    checks["unchanged_and_repeat_decoded_av_identity"] = bool(decoded) and all(
        decoded[name] == decoded["original"] for name in ("base", "repeat"))
    checks["late_and_early_video_changes"] = bool(decoded) and (
        decoded["late"]["video"] != decoded["base"]["video"] and
        decoded["early"]["video"] != decoded["late"]["video"])
    checks["first_pass_audio_preserved"] = bool(decoded) and len({
        item["audio"] for item in decoded.values()}) == 1
    result = {"schema": SCHEMA + ".real-core-cache-decoded-audit",
              "run_root": str(run_root), "attempt": attempt, "checks": checks,
              "decoded_md5": decoded,
              "status": "stage_specific_cache_and_decoded_av_pass"
                        if all(checks.values()) else "fail",
              "boundary": "Owned Core LRU64, reduced canvas and independent stage noise seeds; "
                          "not a changed-MODEL/LoRA, automatic frozen-recipe, multi-tile, "
                          "full-size, or human qualification."}
    output = run_root / f"cache-v{attempt}-audit.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite S18 cache decoded audit")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_cache(args: argparse.Namespace) -> dict:
    run_root = args.cache_run_root.resolve()
    parent = (PROJECT / "artifacts/development/modular-sampling-s18-real-gpu-20260924").resolve()
    if run_root.parent != parent:
        raise ValueError("Cache probe requires an owned S18 run directory")
    positive = json.loads((run_root / "stage-audit-v1.json").read_text(encoding="utf-8"))
    if positive.get("status") != "small_canvas_mechanical_pass_not_quality_acceptance":
        raise ValueError("Cache probe requires a passing S18 positive-stage audit")
    if not (run_root / "input" / IMAGE).is_file() or not (run_root / "paths.json").is_file():
        raise ValueError("Owned S18 input/model-path profile is missing")
    attempt = args.cache_attempt
    label = f"cache-v{attempt}"
    if ((run_root / f"{label}.json").exists() or
            any(run_root.glob(f"{label}-*-phase.json")) or
            any((run_root / "logs").glob(f"s18-{label}.*.log"))):
        raise FileExistsError("Partial S18 cache attempt exists; preserve and inspect it")
    resume = json.loads((run_root / "resume-prompt.json").read_text(encoding="utf-8"))
    variants = _cache_variants(resume, attempt=attempt)
    artifacts = {
        "media": Path(positive["media"]["path"]),
        "native": (run_root / "output/MiniMaxH3/latent_checkpoints" /
                   resume["40"]["inputs"]["checkpoint_path"]),
        "segment": (run_root / "output/MiniMaxH3/chunked_v1_segment_artifacts" /
                    resume["41"]["inputs"]["artifact_path"]),
    }
    if not all(path.resolve().is_relative_to(run_root) and path.is_file()
               for path in artifacts.values()):
        raise ValueError("Owned positive media/checkpoint files are missing or escaped")
    frozen_shas = {name: _sha(path) for name, path in artifacts.items()}
    if (frozen_shas["media"] != positive["media"]["sha256"] or
            not all(_sha(path) == value for path, value in HASHES.items())):
        raise ValueError("S18 positive evidence or source candidates changed")
    gpu = shared.gpu_memory_mib()
    readiness = {
        "private_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "gpu_free_vram_gate": bool(gpu.get("available") and
                                   gpu["free_mib"] >= args.min_free_vram_mib),
    }
    if not args.confirm_run:
        return {"status": "preflight_only", "checks": readiness, "gpu": gpu,
                "run_root": str(run_root)}
    if not all(readiness.values()):
        raise RuntimeError("S18 cache probe resource gate did not pass")
    args.host = "127.0.0.1"
    args.input_directory = run_root / "input"
    args.extra_model_paths_config = run_root / "paths.json"
    report = {"schema": SCHEMA + ".real-core-cache-edit", "run_root": str(run_root),
              "attempt": attempt, "cache_mode": "lru_64",
              "preflight": readiness, "source_file_sha256": frozen_shas,
              "status": "started"}
    try:
        phases = {}
        original_command = shared._server_command

        def cached_command(server_args, server_root):
            command = original_command(server_args, server_root)
            if command.count("--cache-none") != 1:
                raise ValueError("Owned S18 Core no-cache startup flag changed")
            command[command.index("--cache-none"):command.index("--cache-none") + 1] = [
                "--cache-lru", "64"]
            return command

        # Only this owned Core enables caching; the shared probe helper stays unchanged.
        with patch.object(shared, "_server_command", cached_command):
            with shared.IsolatedServer(args, run_root, f"s18-{label}") as server:
                report["owned_core_pid"] = server.process.pid
                for name, graph in variants.items():
                    with (run_root / f"{label}-{name}-prompt.json").open(
                            "x", encoding="utf-8") as stream:
                        json.dump(graph, stream, ensure_ascii=False, indent=2)
                        stream.write("\n")
                    phase = asyncio.run(pdd._submit_prompt_capture(
                        server=f"http://127.0.0.1:{args.port}", prompt=graph,
                        timeout_seconds=args.timeout_seconds))
                    phases[name] = phase
                    with (run_root / f"{label}-{name}-phase.json").open(
                            "x", encoding="utf-8") as stream:
                        json.dump(phase, stream, ensure_ascii=False, indent=2)
                        stream.write("\n")
                    if not _cache_phase_checks({name: phase}).get(
                            name + "_only_expected_sampling", False):
                        raise RuntimeError(f"S18 real Core cache dependency failed at {name}")
        target = json.loads((run_root / "freeze-prompt.json").read_text(
            encoding="utf-8"))["14"]["inputs"]
        media = {name: _media_checks(run_root, width=target["target_width"],
                                     height=target["target_height"],
                                     prefix=f"cache_v{attempt}_{name}")
                 for name in variants}
        checks = _cache_phase_checks(phases)
        checks.update({
            "all_four_media_decode": all(all(item["checks"].values())
                                         for item in media.values()),
            "positive_files_unchanged": all(_sha(path) == frozen_shas[name]
                                            for name, path in artifacts.items()),
            "candidate_sources_unchanged": all(_sha(path) == value
                                                for path, value in HASHES.items()),
            "owned_core_stopped": not shared.port_is_listening("127.0.0.1", args.port),
        })
        report["checks"] = checks
        report["media"] = {name: {"path": item["path"], "sha256": item["sha256"]}
                           for name, item in media.items()}
        if not all(checks.values()):
            raise ValueError("S18 real Core cache edit checks failed")
        report["status"] = "same_core_late_only_then_early_dependency_pass"
        return report
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / f"{label}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8233)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--server-start-timeout", type=float, default=240)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--audit-run-root", type=Path,
                        help="Audit an existing owned run without restarting Core/GPU")
    parser.add_argument("--negative-run-root", type=Path,
                        help="Test three bad resume receipts on a fresh owned Core")
    parser.add_argument("--audit-negative-run-root", type=Path,
                        help="Verify exact rejection reasons from an existing negative run")
    parser.add_argument("--cancel-run-root", type=Path,
                        help="Interrupt segment 1 then retry only later segments on a fresh Core")
    parser.add_argument("--audit-cancel-run-root", type=Path,
                        help="Verify decoded AV parity from an existing cancellation run")
    parser.add_argument("--cache-run-root", type=Path,
                        help="Probe real Core late-only and upstream edit cache dependencies")
    parser.add_argument("--cache-attempt", type=int, default=2,
                        help="Append-only cache probe attempt number (2..99)")
    parser.add_argument("--audit-cache-run-root", type=Path,
                        help="Verify stage-specific cache and decoded AV in an existing run")
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    if args.audit_run_root is not None:
        result = audit_existing(args.audit_run_root, port=args.port)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] != "fail" else 1
    if args.negative_run_root is not None:
        result = run_negative(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in {
            "preflight_only", "three_real_core_bad_receipts_rejected_before_sampling"} else 1
    if args.audit_negative_run_root is not None:
        result = audit_negative_existing(args.audit_negative_run_root, port=args.port)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] != "fail" else 1
    if args.cancel_run_root is not None:
        result = run_cancel(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in {
            "preflight_only", "interrupted_segment1_then_fresh_core_retry_pass"} else 1
    if args.audit_cancel_run_root is not None:
        result = audit_cancel_existing(args.audit_cancel_run_root, port=args.port)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] != "fail" else 1
    if args.cache_run_root is not None:
        result = run_cache(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in {
            "preflight_only", "same_core_late_only_then_early_dependency_pass"} else 1
    if args.audit_cache_run_root is not None:
        result = audit_cache_existing(args.audit_cache_run_root,
                                      attempt=args.cache_attempt, port=args.port)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] != "fail" else 1
    freeze = build_graph("freeze", width=args.width, height=args.height)
    resume = build_graph("resume", width=args.width, height=args.height)
    readiness = preflight(args)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "artifacts/development/modular-sampling-s18-real-gpu-20260924" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "input").mkdir()
    shutil.copy2(args.comfy_root / "input" / IMAGE, run_root / "input" / IMAGE)
    (run_root / "paths.json").write_text(
        json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    (run_root / "freeze-prompt.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.input_directory = run_root / "input"
    args.extra_model_paths_config = run_root / "paths.json"
    report = {"schema": SCHEMA, "run_root": str(run_root), "preflight": readiness,
              "test_canvas": [args.width, args.height, 56],
              "status": "started", "candidate_sha256": readiness["candidate_sha256"]}
    try:
        with shared.IsolatedServer(args, run_root, "s18-freeze") as server:
            report["freeze_core_pid"] = server.process.pid
            freeze_phase = asyncio.run(pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=freeze,
                timeout_seconds=args.timeout_seconds))
        (run_root / "freeze-phase.json").write_text(
            json.dumps(freeze_phase, ensure_ascii=False, indent=2), encoding="utf-8")
        if (freeze_phase.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("S18 freeze graph did not complete; inspect freeze-phase.json")
        receipts = _freeze_receipts(run_root, freeze_phase)
        report["receipts"] = receipts
        _fill_resume(resume, receipts)
        (run_root / "resume-prompt.json").write_text(
            json.dumps(resume, ensure_ascii=False, indent=2), encoding="utf-8")
        with shared.IsolatedServer(args, run_root, "s18-resume") as server:
            report["resume_core_pid"] = server.process.pid
            resume_phase = asyncio.run(pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=resume,
                timeout_seconds=args.timeout_seconds))
        (run_root / "resume-phase.json").write_text(
            json.dumps(resume_phase, ensure_ascii=False, indent=2), encoding="utf-8")
        stage_audit = audit_existing(run_root, port=args.port)
        report["checks"] = stage_audit["checks"]
        report["media"] = stage_audit["media"]
        if stage_audit["status"] == "fail":
            raise ValueError("S18 real freeze/resume stage checks failed")
        report["status"] = "small_canvas_real_assets_mechanical_pass_not_quality_acceptance"
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
