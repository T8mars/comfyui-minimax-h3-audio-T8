"""Audit an owned H16 cold-resume GPU run without queuing or rerunning it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.append(str(PROJECT))
from tools.audit_modular_sampling_compat import write_new  # noqa: E402

PRIVATE = (PROJECT / "artifacts/development").resolve()


def _json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media(root: Path, history: dict, *, output: Path | None = None) -> dict:
    gifs = history["outputs"]["21"]["gifs"]
    if len(gifs) != 1:
        raise ValueError("Expected exactly one VHS delivery record")
    item = gifs[0]
    output = (output if output is not None else root / "output").resolve()
    path = Path(item["fullpath"]).resolve(strict=True)
    if (not path.is_relative_to(output) or path.name != item["filename"]
            or not path.name.endswith("-audio.mp4")
            or item["type"] != "output" or item["format"] != "video/h265-mp4"):
        raise ValueError("VHS delivery path or format differs from owned H16 output")
    all_mp4 = sorted(output.rglob("*.mp4"))
    expected_silent = path.with_name(path.name.replace("-audio.mp4", ".mp4"))
    if all_mp4 != sorted((path, expected_silent)):
        raise ValueError("H16 VHS output inventory differs from one muxed and one intermediate MP4")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
         "-of", "json", str(path)], capture_output=True, text=True, check=True)
    streams = json.loads(probe.stdout)["streams"]
    video = [stream for stream in streams if stream["codec_type"] == "video"]
    audio = [stream for stream in streams if stream["codec_type"] == "audio"]
    if (len(video), len(audio)) != (1, 1):
        raise ValueError("H16 delivery needs exactly one video and one audio stream")
    if (int(video[0]["width"]), int(video[0]["height"]),
            int(video[0]["nb_read_frames"]), video[0]["avg_frame_rate"]) != (
            448, 448, 124, "24/1"):
        raise ValueError("H16 reduced delivery geometry/timeline mismatch")
    for selection in (("-map", "0:v:0"), ("-map", "0:a:0"),
                      ("-map", "0:v:0", "-map", "0:a:0")):
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-xerror", "-err_detect", "explode",
             "-i", str(path), *selection, "-f", "null", "-"],
            capture_output=True, check=False)
        if decoded.returncode or decoded.stderr.strip():
            raise RuntimeError("H16 strict AV decode failed: "
                               + decoded.stderr.decode("utf-8", errors="replace"))
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-map", "0:a:0",
         "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        capture_output=True, check=True).stdout
    import numpy as np
    samples = np.frombuffer(pcm, dtype="<f4")
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("H16 decoded audio is empty or non-finite")
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if rms <= 1e-6:
        raise ValueError("H16 decoded audio is silent")
    return {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size,
            "intermediate_video_only_path": str(expected_silent),
            "width": 448, "height": 448, "frames": 124, "fps": "24/1",
            "video_seconds": video[0].get("duration"),
            "audio_seconds": audio[0].get("duration"),
            "decoded_pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "decoded_audio_rms": rms, "strict_decode": True}


def audit(root: Path) -> dict:
    root = root.resolve(strict=True)
    if not root.is_relative_to(PRIVATE):
        raise ValueError("Only a private owned H16 run may be audited")
    outer = _json(root / "terminal.json")
    if (outer.get("status") != "failed"
            or outer.get("error") != "RuntimeError: Expected exactly one output MP4, found 2"):
        raise ValueError("This audit handles only the preserved VHS inventory false failure")
    receipt = _json(root / "receipt.json")
    graphs = {phase: _json(root / phase / "generation/prompt.json")
              for phase in ("freeze", "resume")}
    phases = {phase: _json(root / phase / "terminal.json")
              for phase in ("freeze", "resume")}
    histories = {phase: _json(root / phase / "generation/history.json")
                 for phase in ("freeze", "resume")}
    for phase, terminal in phases.items():
        if (terminal["status"] != "mechanical_execution_pass"
                or not terminal["history_status"]["completed"]
                or terminal["timing"]["terminal"] != "execution_success"
                or not terminal["timing"]["complete_uncached_graph"]
                or terminal["server_stop"]["owned_children_remaining"]
                or terminal["resources"]["status"] != "observations_within_policy"):
            raise ValueError(f"H16 {phase} did not complete an owned resource-gated graph")
    freeze, resume = graphs["freeze"], graphs["resume"]
    def kinds(graph):
        return [value["class_type"] for value in graph.values()]
    if (kinds(freeze).count("SamplerCustomAdvanced") != 1
            or kinds(freeze).count("MiniMaxH3H16Pass2WindowEXPT8") != 3
            or kinds(resume).count("SamplerCustomAdvanced") != 0
            or kinds(resume).count("MiniMaxH3H16Pass2WindowEXPT8") != 4
            or resume["51"]["inputs"]["checkpoint_path"] != receipt["native_path"]
            or resume["51"]["inputs"]["expected_file_sha256"] != receipt["native_file_sha256"]
            or resume["51"]["inputs"]["expected_manifest_json"] != receipt["native_manifest_json"]
            or resume["52"]["inputs"]["artifact_path"] != receipt["window_path"]
            or resume["52"]["inputs"]["artifact_sha256"] != receipt["window_manifest_sha256"]):
        raise ValueError("H16 cold-resume stage or receipt binding differs")
    output = root / "output/MiniMaxH3"
    native = output / "latent_checkpoints" / receipt["native_path"]
    window = output / "h16_window_artifacts" / receipt["window_path"]
    if (_sha(native), _sha(window)) != (receipt["native_file_sha256"],
                                        receipt["window_manifest_sha256"]):
        raise ValueError("H16 frozen artifact changed after GPU execution")
    from safetensors import safe_open

    with safe_open(str(native), framework="pt", device="cpu") as handle:
        payload = json.loads(handle.metadata()["t8_native_latent_checkpoint_json"])
    if (json.loads(receipt["native_manifest_json"]) != payload["manifest"]
            or _json(window)["binding"]["index"] != 2):
        raise ValueError("H16 embedded native manifest or frozen index differs")
    executed = {phase: {row["node"] for row in terminal["timing"]["node_intervals"]}
                for phase, terminal in phases.items()}
    if not {"32", "35", "38", "51", "52"} <= executed["freeze"]:
        raise ValueError("H16 freeze stage execution missing")
    if not {"41", "44", "47", "50", "20", "21"} <= executed["resume"]:
        raise ValueError("H16 resume stage/media execution missing")
    media = _media(root, histories["resume"])
    return {"schema": "t8.modular-h16-cold-media-gpu-audit.v1",
            "status": "mechanical_av_pass_human_pending",
            "original_controller_status": outer["status"],
            "original_controller_error": outer["error"],
            "phase_elapsed_seconds": {phase: phases[phase]["elapsed_seconds"]
                                      for phase in phases},
            "frozen_artifact_sha256": {
                "native": receipt["native_file_sha256"],
                "window_manifest": receipt["window_manifest_sha256"]},
            "media": media,
            "qualification": "Real pretrained assets on a derived 124f 224x224->448x448 H16 plain "
                             "freeze and new-Core HIGH-only continuation, exact frozen receipt "
                             "and strict H.265/AAC decoding. Not the original 736->1472 graph, "
                             "Relay/EAV combination, full backend/geometry matrix or human review."}


def _effect_report(history: dict, node_id: str, index: int, mode: str) -> dict:
    values = history["outputs"][node_id]["text"]
    if len(values) != 1:
        raise ValueError(f"H16 window {index} has no unique EAV audit")
    report = json.loads(values[0])
    forwards = report["forward_plan"]["forwards"]
    expected_calls = sum(item["attention_blocks"] for item in forwards)
    if (report["schema"] != "t8.modular-sampling.eav-audit.v1"
            or report["h16_window_index"] != index
            or report["audio_output"] != "refined_exp"
            or report["config"]["mode"] != mode
            or report["status"] != ("observed_apply_exp" if mode == "apply_exp"
                                    else "observed_report_only")
            or report["completed_forwards"] != report["planned_forwards"]
            or report["completed_forwards"] != 4
            or not report["clock_match"]
            or not report["relay_required"]
            or report["relay_attention_calls"] < expected_calls
            or report["selector_calls"] < expected_calls
            or report["sparse_producer_calls"] != 0
            or report["feta"]["aborted"] is not None
            or report["feta"]["model_forward_count"] != 4
            or report["feta"]["active_forward_count"] <= 0
            or (mode == "apply_exp" and report["feta"]["g_max"] <= 1.0)):
        raise ValueError(f"H16 window {index} Relay/EAV actual call audit failed")
    return {"window_index": index, "status": report["status"],
            "completed_forwards": report["completed_forwards"],
            "selector_calls": report["selector_calls"],
            "relay_attention_calls": report["relay_attention_calls"],
            "g_max": report["feta"]["g_max"],
            "plan_sha256": report["h16_plan_sha256"]}


def audit_effect(root: Path) -> dict:
    root = root.resolve(strict=True)
    if not root.is_relative_to(PRIVATE):
        raise ValueError("Only a private owned H16 effect run may be audited")
    outer = _json(root / "terminal.json")
    if outer.get("status") != "mechanical_av_pass_human_pending" or outer.get("route") != "effects":
        raise ValueError("H16 effect controller did not finish its owned media run")
    receipt = _json(root / "receipt.json")
    graphs = {phase: _json(root / phase / "generation/prompt.json")
              for phase in ("freeze", "resume")}
    phases = {phase: _json(root / phase / "terminal.json")
              for phase in ("freeze", "resume")}
    histories = {phase: _json(root / phase / "generation/history.json")
                 for phase in ("freeze", "resume")}
    for phase, terminal in phases.items():
        if (terminal["status"] != "mechanical_execution_pass"
                or not terminal["history_status"]["completed"]
                or terminal["timing"]["terminal"] != "execution_success"
                or not terminal["timing"]["complete_uncached_graph"]
                or terminal["server_stop"]["owned_children_remaining"]
                or terminal["resources"]["status"] != "observations_within_policy"):
            raise ValueError(f"H16 effect {phase} did not complete an owned graph")
    freeze, resume = graphs["freeze"], graphs["resume"]
    def kinds(graph):
        return [node["class_type"] for node in graph.values()]
    if (kinds(freeze).count("SamplerCustomAdvanced") != 1
            or kinds(freeze).count("MiniMaxH3H16Pass2WindowEXPT8") != 3
            or kinds(resume).count("SamplerCustomAdvanced") != 0
            or kinds(resume).count("MiniMaxH3H16Pass2WindowEXPT8") != 4
            or resume["20"]["inputs"]["av_latent"] != ["80", 0]
            or resume["52"]["inputs"]["width"] != 448
            or resume["52"]["inputs"]["height"] != 448
            or any(freeze[key]["inputs"]["mode"] != "report_only"
                   for key in ("60", "63", "66"))
            or any(resume[key]["inputs"]["mode"] != "report_only"
                   for key in ("69", "72", "75"))
            or resume["78"]["inputs"]["mode"] != "apply_exp"
            or resume["88"]["inputs"]["checkpoint_path"] != receipt["native_path"]
            or resume["88"]["inputs"]["expected_file_sha256"] != receipt["native_file_sha256"]
            or resume["88"]["inputs"]["expected_manifest_json"] != receipt["native_manifest_json"]
            or resume["89"]["inputs"]["artifact_path"] != receipt["window_path"]
            or resume["89"]["inputs"]["artifact_sha256"] != receipt["window_manifest_sha256"]):
        raise ValueError("H16 effect stage, mode or frozen receipt binding differs")
    output = root / "output/MiniMaxH3"
    native = output / "latent_checkpoints" / receipt["native_path"]
    window = output / "h16_window_artifacts" / receipt["window_path"]
    if (_sha(native), _sha(window)) != (receipt["native_file_sha256"],
                                        receipt["window_manifest_sha256"]):
        raise ValueError("H16 effect frozen artifact changed")
    from safetensors import safe_open

    with safe_open(str(native), framework="pt", device="cpu") as handle:
        payload = json.loads(handle.metadata()["t8_native_latent_checkpoint_json"])
    if (payload["manifest"] != json.loads(receipt["native_manifest_json"])
            or _json(window)["binding"]["index"] != 2):
        raise ValueError("H16 effect saved manifest/index differs")
    executed = {phase: {row["node"] for row in terminal["timing"]["node_intervals"]}
                for phase, terminal in phases.items()}
    if not {"32", "35", "38", "62", "65", "68", "88", "89"} <= executed["freeze"]:
        raise ValueError("H16 effect freeze execution missing")
    if not {"41", "44", "47", "50", "71", "74", "77", "80", "20", "21"} <= executed["resume"]:
        raise ValueError("H16 effect resume or media execution missing")
    reports = [_effect_report(histories["freeze"], node, index, "report_only")
               for index, node in enumerate(("62", "65", "68"))]
    reports.extend(_effect_report(histories["resume"], node, index,
                                  "apply_exp" if index == 6 else "report_only")
                   for index, node in enumerate(("71", "74", "77", "80"), start=3))
    if len({row["plan_sha256"] for row in reports}) != 1:
        raise ValueError("H16 effect windows do not share one saved plan")
    media = _media(root, histories["resume"])
    if media["sha256"] != outer["media"]["sha256"]:
        raise ValueError("H16 effect controller and independent media SHA differ")
    return {"schema": "t8.modular-h16-effect-cold-media-gpu-audit.v1",
            "status": "mechanical_av_effect_calls_pass_human_pending",
            "phase_elapsed_seconds": {phase: phases[phase]["elapsed_seconds"]
                                      for phase in phases},
            "frozen_artifact_sha256": {
                "native": receipt["native_file_sha256"],
                "window_manifest": receipt["window_manifest_sha256"]},
            "window_effect_reports": reports, "media": media,
            "qualification": "Real pretrained Relay+EAV calls in seven H16 HIGH windows, "
                             "last HIGH window apply_exp, explicit frozen receipt and new-Core "
                             "HIGH-only continuation, strict 124f H.265/AAC decode at reduced "
                             "224->448 geometry. Not an uninterrupted parity, original-size, "
                             "multi-backend or human-quality acceptance."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("plain", "effects"), default="plain")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(PRIVATE):
        parser.error("Use a new private audit output; never overwrite earlier evidence")
    result = audit(args.root) if args.route == "plain" else audit_effect(args.root)
    write_new(output, result)
    print(json.dumps({"status": result["status"], "media": result["media"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
