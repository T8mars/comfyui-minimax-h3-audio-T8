"""Read actual UI-queued history and cold-verify its saved postprocess movie.

Writes only new private evidence. Never queues, mutates the graph, accepts
quality, or synthesizes human feedback. Use an explicit existing prompt ID.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import urllib.request


def sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--native-workflow", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep previous evidence; use a fresh output directory")
    project = Path(__file__).resolve().parents[1]
    # Same process-local CPU boundary as the existing regression runner. --cpu
    # alone does not prevent imported helpers from probing available CUDA.
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    sys.path[:0] = [str(project), str(args.core.resolve())]
    sys.argv = ["r6-canvas-cold-check", "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    assert comfy.cli_args.args.cpu is True
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
    package_name = "r6_canvas_t8_pkg"
    spec = importlib.util.spec_from_file_location(package_name, project / "__init__.py",
                                                submodule_search_locations=[str(project)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    from r6_canvas_t8_pkg.postprocess_delivery import read_postprocess_state
    from r6_canvas_t8_pkg.video_outpaint_delivery import read_delivery_report
    from r6_canvas_t8_pkg.video_outpaint_media import _validate_final_file
    from comfy_api.latest import InputImpl

    def get(suffix):
        with urllib.request.urlopen(args.server.rstrip("/") + suffix, timeout=30) as response:
            return json.load(response)

    history = get("/history/" + args.prompt_id)[args.prompt_id]
    if history["status"].get("status_str") != "success" or history["status"].get("completed") is not True:
        raise ValueError("Actual canvas task is not successfully terminal")
    queue = get("/queue")
    if queue["queue_running"] or queue["queue_pending"]:
        raise ValueError("Owned canvas still has queued work")
    prompt = history["prompt"][2]
    if ([node["class_type"] for node in prompt.values()] != ["LoadVideo", "GetVideoComponents", "ImageCropV2", "MiniMaxH3PostprocessSaveEXPT8"]
            or history["prompt"][3].get("comfy_usage_source") != "comfyui-frontend"):
        raise ValueError("Not the specified native cold-crop canvas task")
    native = json.loads(args.native_workflow.read_text(encoding="utf8"))
    nodes = {str(node["id"]): node for node in native["nodes"]}
    region = prompt["3"]["inputs"]["crop_region"]
    crop = nodes["3"]
    if (nodes["1"]["widgets_values"][0] != prompt["1"]["inputs"]["file"]
            or crop["widgets_values"][0] != region
            or crop["widgets_values_named"]["crop_region"] != region
            or nodes["4"]["widgets_values"][:4] != [prompt["4"]["inputs"][key] for key in (
                "filename_prefix", "confirm_postprocess", "crf", "expected_master_sha256")]
            or native["links"] != [[1, 1, 0, 2, 0, "VIDEO"], [2, 2, 0, 3, 0, "IMAGE"],
                                   [3, 1, 0, 4, 0, "VIDEO"], [4, 3, 0, 4, 1, "IMAGE"]]):
        raise ValueError("Actual saved native parameters/edges differ from queued crop task")
    state = read_postprocess_state(args.state)
    if state["state"] != "postprocess_complete" or not state["master_available"] or not state["output_published"]:
        raise ValueError("Cold state reader did not verify a delivered movie")
    report = read_delivery_report(state["output_path"], report_kind="postprocess")
    master = state["master"]
    check = _validate_final_file(state["output_path"], frame_count=master["frames"],
        width=region["width"], height=region["height"], rate=24,
        source_audio_packets=master["audio_packets"], source_audio_pcm=master["audio_pcm"],
        interrupt_check=lambda: None)
    for key in ("audio_packet_payload_exact", "audio_packet_timeline_exact",
                "audio_decoded_pcm_exact", "audio_decoded_timeline_exact"):
        if check[key] is not True or report["media"][key] is not True:
            raise ValueError("Actual cold audio verification failed: " + key)
    if (master["sha256"] != prompt["4"]["inputs"]["expected_master_sha256"]
            or sha(master["path"]) != master["sha256"]
            or sha(state["output_path"]) != state["output_sha256"]
            or InputImpl.VideoFromFile(state["output_path"]).get_dimensions() != (region["width"], region["height"])):
        raise ValueError("Media bytes/geometry no longer match the selected master and crop")
    # Verify the exact pre-encoder RGB came from the native master decode/crop,
    # not another source or generated replacement. No lossless H264 claim.
    frames = InputImpl.VideoFromFile(master["path"]).get_components().images
    x, y, width, height = (region[key] for key in ("x", "y", "width", "height"))
    pixels = frames[:, y:y + height, x:x + width, :3]
    raw_hash = hashlib.sha256()
    for frame in pixels:
        raw_hash.update((frame.clamp(0, 1) * 255).round().to(torch.uint8).contiguous().numpy().tobytes())
    if raw_hash.hexdigest() != report["source_rgb8_sha256"]:
        raise ValueError("Canvas encoder pixels differ from actual Core crop")
    assert not torch.cuda.is_initialized()
    args.output.mkdir(parents=True)
    sources = ["h3_t8/postprocess_delivery.py", "h3_t8/nodes_postprocess.py", "h3_t8/h3_av_delivery.py",
               "h3_t8/video_outpaint_delivery.py", "h3_t8/video_outpaint_media.py",
               "h3_t8/video_outpaint_packet_mux.py", "web/readable_audio.js",
               "tools/build_radar_delivery_workflows.py", "tools/collect_radar_delivery_canvas.py"]
    evidence = {"status": "pass", "prompt_id": args.prompt_id, "native_queued": True,
        "native_saved_parameter_edges_exact": True, "native_workflow_sha256": sha(args.native_workflow),
        "source_rgb8_sha256": raw_hash.hexdigest(), "master_sha256": master["sha256"],
        "output_sha256": state["output_sha256"], "cold_new_process_media_verify": check,
        "frames": master["frames"], "output_dimensions": [width, height], "new_sampling_NFE": 0,
        "human_quality_accepted": False, "automatic_accept": False, "cuda_initialized": False,
        "state_path": str(args.state.resolve()), "output_path": state["output_path"],
        "protected_index_sha256": sha(project / ".git/index"),
        "protected_sampling_sha256": sha(project / "h3_t8/sampling.py"),
        "current_source_epoch": {name: sha(project / name) for name in sources}}
    for name, value in (("history.json", history), ("evidence.json", evidence), ("queue.json", queue)):
        with (args.output / name).open("x", encoding="utf8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: evidence[key] for key in ("status", "frames", "output_dimensions", "master_sha256",
        "output_sha256", "new_sampling_NFE", "cuda_initialized")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
