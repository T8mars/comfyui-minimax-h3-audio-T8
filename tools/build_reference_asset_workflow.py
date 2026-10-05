"""Build one explicit native reference-asset canvas, never queue or overwrite.

This creates assets from an explicitly selected complete video, not a sampler
workflow or a claim of identity/voice quality. Saved packages are separate from
drive/final audio. The two VAE filenames must be actual installed selections.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request

try:
    from .api_to_frontend_workflow import convert
    from .check_windows_paths import validate_paths
except ImportError:
    from api_to_frontend_workflow import convert
    from check_windows_paths import validate_paths


CREATE = "MiniMaxH3ReferenceCreateEXPT8"
SAVE = "MiniMaxH3ReferenceSaveEXPT8"


def choices(spec):
    return spec[0] if isinstance(spec[0], list) else spec[1].get("options", [])


def make_workflow(info, *, source_video, video_vae, audio_vae, voice_filename,
                  image_filename, confirm=False, width=256, height=256):
    if not source_video or source_video not in choices(info["LoadVideo"]["input"]["required"]["file"]):
        raise ValueError("Choose one exact installed input video, not an inferred source")
    vaes = choices(info["VAELoader"]["input"]["required"]["vae_name"])
    if video_vae not in vaes or audio_vae not in vaes:
        raise ValueError("Choose the exact installed native video/audio VAE files")
    if voice_filename == image_filename:
        raise ValueError("Voice and image packages need separate new filenames")
    for name in (voice_filename, image_filename):
        import re
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}\.safetensors", name):
            raise ValueError("Reference asset filenames must be explicit short ASCII names")
    if type(confirm) is not bool or type(width) is not int or type(height) is not int:
        raise ValueError("Explicit confirmation and integer geometry are required")
    if min(width, height) < 32 or max(width, height) > 2048 or width % 32 or height % 32:
        raise ValueError("Reference geometry needs the native positive32-grid")
    graph = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": source_video}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "ImageFromBatch", "inputs": {"image": ["2", 0], "batch_index": 0, "length": 1}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae}},
        "6": {"class_type": CREATE, "inputs": {"role_id": "A", "kind": "audio", "width": width,
            "height": height, "frame_limit": 124, "audio": ["2", 1], "audio_vae": ["4", 0]}},
        "7": {"class_type": CREATE, "inputs": {"role_id": "B", "kind": "image", "width": width,
            "height": height, "frame_limit": 124, "frames": ["3", 0], "video_vae": ["5", 0]}},
        "8": {"class_type": SAVE, "inputs": {"reference_package": ["6", 0],
            "filename": voice_filename, "confirm_save": confirm}},
        "9": {"class_type": SAVE, "inputs": {"reference_package": ["7", 0],
            "filename": image_filename, "confirm_save": confirm}},
    }
    workflow = convert(graph, info, "Reference asset encode and explicit save EXP")
    positions = {1: [0, 0], 2: [430, 0], 3: [430, 420], 4: [850, 0], 5: [850, 420],
                 6: [1260, 0], 7: [1260, 420], 8: [1700, 0], 9: [1700, 420]}
    for node in workflow["nodes"]:
        node["pos"] = positions[node["id"]]
        node["size"] = [380, 320]
    note = ("Explicit complete source video → native voice A and first-frame image B.\n"
            "These are independent role assets, NOT proof of two different people or voice cloning.\n"
            "No sampler, Qwen, NFE, drive/final audio replacement, or automatic review.\n"
            "Actual producer/contents are bound before and after encode.\n"
            "Save confirmation is explicit; new files only in models/refmods, no overwrites.\n"
            "Refresh the node definitions before selecting saved packages in Load.\n"
            "Use Route then fresh Apply; a voice anchor is not master drive/final_audio.")
    workflow["nodes"].append({"id": 10, "type": "Note", "title": "Reference assets · contract",
        "pos": [0, 800], "size": [1100, 280], "flags": {}, "order": 9, "mode": 0,
        "inputs": [], "outputs": [], "properties": {"text": note}, "widgets_values": [note]})
    workflow["last_node_id"] = 10
    workflow["extra"]["radar_r6_reference_assets"] = {"schema": "t8.r6.reference-assets.canvas/v1",
        "source_video": source_video, "new_sampling_NFE": 0, "automatic_accept": False,
        "pretrained_runtime_verified": False}
    return workflow, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--video-vae", required=True)
    parser.add_argument("--audio-vae", required=True)
    parser.add_argument("--voice-filename", required=True)
    parser.add_argument("--image-filename", required=True)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.output.resolve()
    owned = root / "artifacts/development/radar-r6-20261005"
    if not path.is_relative_to(owned) or path.exists():
        raise ValueError("Use one new owned RADAR canvas, never overwrite an existing workflow")
    errors = validate_paths([path.relative_to(root).as_posix()])
    if errors:
        raise ValueError(errors)
    with urllib.request.urlopen(args.server.rstrip("/") + "/object_info", timeout=30) as response:
        info = json.load(response)
    workflow, _ = make_workflow(info, source_video=args.source_video, video_vae=args.video_vae,
        audio_vae=args.audio_vae, voice_filename=args.voice_filename, image_filename=args.image_filename,
        confirm=args.confirm)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(workflow, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(path)


if __name__ == "__main__":
    main()
