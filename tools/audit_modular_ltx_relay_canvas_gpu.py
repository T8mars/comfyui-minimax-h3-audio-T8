"""Read-only postflight of one already completed native S27 Relay canvas run.

The original failed controller receipt is immutable. Only an exact empty
LoadVideo preview field may differ between pre-click and POST serialization.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from tools import run_modular_ltx_relay_canvas_gpu as canvas
from tools import run_modular_ltx_rgb_source_gpu as source
from tools import run_modular_ltx_relay_gpu as relay

PRIVATE = canvas.ROOT / "artifacts/development/modular-ltx-relay-20260928/canvas"
API_BASELINES = {
    ("ordinary", False): "20260927T235450Z-ordinary-cold",
    ("identity", False): "20260927T235706Z-identity-cold",
    ("ordinary", True): "20260927T235921Z-ordinary-eav-relay-cold",
    ("identity", True): "20260928T000138Z-identity-eav-relay-cold",
}
ORIGINAL_AUDIO_SHA = "9e6844a213817e8a712e6d44a4b28bcc590b115f01a8457605c0d2cbecc68adc"

normalize_empty_preview = canvas.normalize_empty_preview


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, output = args.run_root.resolve(), args.output.resolve()
    if (not root.is_relative_to(PRIVATE.resolve()) or not output.is_relative_to(root)
            or output.exists() or not (root / "report.json").is_file()):
        parser.error("Read one existing private canvas run and write a new private receipt")
    original = json.loads((root / "report.json").read_text(encoding="utf8"))
    if (original["status"] not in ("fail", "pass_native_canvas_single_route_mechanical_not_quality")
            or (original["status"] == "fail" and
                "failed its execution checks" not in original.get("error", ""))
            or original["canvas"]["response_status"] != 200
            or original["canvas"]["request_url"] != "http://127.0.0.1:8958/api/prompt"):
        raise ValueError("This postflight only covers the exact completed canvas capture")
    prequeue = json.loads((root / "canvas-before-queue.api.json").read_text(encoding="utf8"))
    prepared = json.loads((root / "prepared.api.json").read_text(encoding="utf8"))
    post = json.loads((root / "canvas-post.json").read_text(encoding="utf8"))
    history = json.loads((root / "history.json").read_text(encoding="utf8"))
    if prequeue != prepared or post.get("prompt") is None:
        raise ValueError("Prepared, visible canvas or submitted API is incomplete")
    normalize_empty_preview(prequeue, post["prompt"])
    prompt_id = original["canvas"]["prompt_id"]
    if (not isinstance(history.get("prompt"), list) or len(history["prompt"]) < 3
            or history["prompt"][1] != prompt_id or history["prompt"][2] != post["prompt"]):
        raise ValueError("History is not the exact browser-submitted prompt")
    output_node = history.get("outputs", {}).get("20", {})
    images = output_node.get("images") or []
    if len(images) != 1 or images[0].get("filename") != "video.mp4" or images[0].get("type") != "output":
        raise ValueError("Canvas history does not contain the expected isolated video output")
    media = source.media(root, canvas.MEDIA_PREFIX, isolated=True)
    baseline_path = (canvas.ROOT / "artifacts/development/modular-ltx-relay-20260928/trained"
                     / API_BASELINES[(original.get("route", "ordinary"),
                                      original.get("combined_eav", False))] / "report.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf8"))
    if baseline["status"] != "full_cold_relay_media_mechanical_not_canvas_or_quality":
        raise ValueError("Independent API media baseline is not a completed control")
    baseline_full = baseline["media"]["full"]
    path = Path(media["path"]).resolve()
    relative = path.relative_to((root / "output").resolve()).as_posix()
    if relative != images[0]["subfolder"].replace("\\", "/") + "/video.mp4":
        raise ValueError("History media path does not name the strictly decoded candidate")
    video = next(item for item in media["streams"] if item["codec_type"] == "video")
    fps_num, fps_den = map(int, video["avg_frame_rate"].split("/"))
    source_file = root / "input/source.mp4"
    logs = (root / "logs/canvas.stderr.log").read_text(encoding="utf8", errors="replace")
    checks = {
        "native_button_post_accepted": original["canvas"]["queue_button_testid"] == "queue-button",
        "only_empty_frontend_preview_added": True,
        "history_success": history["status"].get("status_str") == "success"
                           and history["status"].get("completed") is True,
        "history_exact_browser_prompt": True,
        "strict_complete_av": media["fully_decoded"],
        "geometry_113_frames_24fps": (video["width"], video["height"], int(video.get("nb_frames", 0)))
                                      == (1024, 576, 113) and fps_den != 0 and fps_num / fps_den == 24,
        "duration_113_over_24": math.isclose(float(video.get("duration", 0)), 113 / 24, abs_tol=1e-5),
        "original_audio_bypass": media["decoded_sha"]["audio"] == ORIGINAL_AUDIO_SHA,
        "independent_api_decoded_picture_and_audio_parity": (
            source.sha(Path(baseline_full["path"])) == baseline_full["sha"]
            and media["decoded_sha"] == baseline_full["decoded_sha"]),
        "saved_input_stable": source.sha(source_file) == original["source_sha256"],
        "three_real_sampler_steps_logged": bool(re.search(r"100%.*3/3", logs)),
        "original_controller_source_stable": original["checks"].get("source_stable") is True,
        "original_controller_media_stable": (
            original["status"] == "fail" or
            (original.get("media", {}).get("sha") == media["sha"]
             and original["media"]["decoded_sha"] == media["decoded_sha"])),
        "original_controller_8940_untouched": original["checks"].get("user_8940_untouched") is True,
        "owned_server_and_port_closed": (original["checks"].get("owned_service_stopped") is True
                                         and original["checks"].get("owned_port_closed") is True
                                         and not canvas.shared.port_is_listening("127.0.0.1", 8958)),
    }
    receipt = {"schema": "t8.s27.relay-native-canvas-postflight.v1",
               "status": "pass_single_native_canvas_media_mechanical_not_quality" if all(checks.values()) else "fail",
               "original_failed_controller_receipt": str(root / "report.json"),
               "original_failure": original.get("error"), "prompt_id": prompt_id,
               "checks": checks, "media": media, "api_control_report": str(baseline_path),
               "boundary": "One ordinary full-save native Queue click and complete media only; no cold canvas, "
                           "other routes, multi-source, human quality or full dual-pass completion."}
    relay._write(output, receipt)
    print(json.dumps({"status": receipt["status"], "failed": [k for k, v in checks.items() if not v],
                      "output": str(output)}), flush=True)
    return 0 if receipt["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
