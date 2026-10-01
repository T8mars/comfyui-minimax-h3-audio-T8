"""Build private Chunked v1 freeze/resume drafts without changing old graphs.

The first graph explicitly saves the completed first-pass AV and v1 segment 0.
The second graph loads those two exact artifacts and samples only segments 1/2.
Both independent path/SHA pairs are deliberately blank until the user pastes
the values returned by the freeze graph.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from tools.build_modular_chunked_v1_relay_workflow import build_pair
from tools.build_modular_h16_storage_workflow import Draft
from tools.build_modular_chunked_workflow import ROOT, PLAN, SPEC, CONTEXT, RESULT


NATIVE_SAVE = "MiniMaxH3NativeLatentCheckpointSaveT8Advanced"
NATIVE_LOAD = "MiniMaxH3NativeLatentCheckpointLoadT8Advanced"
SEGMENT_SAVE = "MiniMaxH3ChunkedV1SegmentSaveEXPT8"
SEGMENT_LOAD = "MiniMaxH3ChunkedV1SegmentLoadEXPT8"


def _expect_base(frontend, api):
    nodes = {node["id"]: node for node in frontend["nodes"]}
    frontend_expected = {12: "SamplerCustomAdvanced",
                         14: "MiniMaxH3ChunkedTwoPassPlanT8Advanced",
                         20: "MiniMaxH3ChunkedSourceSegmentEXPT8",
                         21: "MiniMaxH3ChunkedPass2PrepareEXPT8",
                         23: "MiniMaxH3ChunkedPass2SegmentEXPT8",
                         28: "MiniMaxH3ChunkedPass2SegmentEXPT8",
                         31: "MiniMaxH3ChunkedPass2SegmentEXPT8",
                         36: "MiniMaxH3ChunkedV1RelayProjectEXPT8"}
    api_expected = frontend_expected.copy()
    api_expected[25] = api_expected.pop(23)
    if (any(nodes.get(key, {}).get("type") != kind
            for key, kind in frontend_expected.items()) or
            any(api.get(str(key), {}).get("class_type") != kind
                for key, kind in api_expected.items())):
        raise ValueError("Unexpected fixed three-segment Chunked v1 Relay source graph")


def _prune_api(graph, roots):
    keep = set(roots)
    pending = list(roots)
    while pending:
        node = graph[pending.pop()]
        for value in node["inputs"].values():
            if (isinstance(value, list) and len(value) == 2 and
                    isinstance(value[0], str) and value[0] in graph and
                    isinstance(value[1], int) and value[0] not in keep):
                keep.add(value[0])
                pending.append(value[0])
    return {key: value for key, value in graph.items() if key in keep}


def freeze_frontend(base):
    draft = Draft(base)
    native = draft.make(
        NATIVE_SAVE, "STEP 1 · freeze completed first-pass AV (enable confirm_save)",
        [("av_latent", "LATENT")],
        [("av_latent", "LATENT"), ("status", "STRING"),
         ("checkpoint_path", "STRING"), ("file_sha256", "STRING"),
         ("manifest_json", "STRING"), ("report_json", "STRING")],
        ["chunked_v1_56f_first_pass", "chunked_v1_56f_first_pass", False, True, 8],
        (3200, -500),
    )
    draft.connect((12, 1), native, "av_latent", "LATENT")
    frozen = draft.make(
        SEGMENT_SAVE, "STEP 2 · freeze completed segment 0 (enable confirm_save)",
        [("segment_result", RESULT), ("source_segment", "LATENT"),
         ("segment_spec", SPEC), ("pass2_context", CONTEXT), ("plan", PLAN)],
        [("cumulative_av_latent", "LATENT"), ("segment_result", RESULT),
         ("artifact_path", "STRING"), ("artifact_sha256", "STRING"),
         ("report_json", "STRING")], [False], (4200, 500),
    )
    for name, source, dtype in (
        ("segment_result", (23, 1), RESULT),
        ("source_segment", (20, 0), "LATENT"),
        ("segment_spec", (20, 1), SPEC),
        ("pass2_context", (21, 0), CONTEXT),
        ("plan", (14, 0), PLAN),
    ):
        draft.connect(source, frozen, name, dtype)
    # Audit is independently visible. The Save node does not claim its report
    # is a prerequisite or an automatic MODEL/conditioning cache validation.
    return draft.prune((native["id"], frozen["id"], 35))


def resume_frontend(base):
    draft = Draft(base)
    native = draft.make(
        NATIVE_LOAD, "PASTE exact first-pass checkpoint path + manifest + file SHA",
        [], [("av_latent", "LATENT"), ("status", "STRING"),
             ("resume_verified", "BOOLEAN"), ("checkpoint_id", "STRING"),
             ("content_sha256", "STRING"), ("file_sha256", "STRING"),
             ("manifest_json", "STRING"), ("report_json", "STRING")],
        ["", "", "", 8], (300, -700),
    )
    for key in (20, 21, 26, 29):
        draft.disconnect(draft.nodes[key], "first_pass_latent")
        draft.connect((native["id"], 0), draft.nodes[key], "first_pass_latent", "LATENT")
    frozen = draft.make(
        SEGMENT_LOAD, "PASTE exact segment-0 manifest path + SHA; resume at segment 1",
        [("source_segment", "LATENT"), ("segment_spec", SPEC),
         ("pass2_context", CONTEXT), ("plan", PLAN)],
        [("cumulative_av_latent", "LATENT"), ("segment_result", RESULT),
         ("report_json", "STRING")], ["", ""], (3600, 700),
    )
    for name, source, dtype in (
        ("source_segment", (20, 0), "LATENT"),
        ("segment_spec", (20, 1), SPEC),
        ("pass2_context", (21, 0), CONTEXT),
        ("plan", (14, 0), PLAN),
    ):
        draft.connect(source, frozen, name, dtype)
    for key in (28, 36):
        draft.disconnect(draft.nodes[key], "previous_result")
        draft.connect((frozen["id"], 1), draft.nodes[key], "previous_result", RESULT)
    return draft.prune((19, 37, 39))


def freeze_api(base):
    graph = deepcopy(base)
    next_id = max(map(int, graph)) + 1
    native = str(next_id)
    frozen = str(next_id + 1)
    graph[native] = {"class_type": NATIVE_SAVE, "inputs": {
        "av_latent": ["12", 1], "filename_prefix": "chunked_v1_56f_first_pass",
        "checkpoint_id": "chunked_v1_56f_first_pass", "confirm_save": False,
        "verify_after_write": True, "hash_chunk_megabytes": 8}}
    graph[frozen] = {"class_type": SEGMENT_SAVE, "inputs": {
        "segment_result": ["25", 1], "source_segment": ["20", 0],
        "segment_spec": ["20", 1], "pass2_context": ["21", 0],
        "plan": ["14", 0], "confirm_save": False}}
    return _prune_api(graph, (native, frozen, "35"))


def resume_api(base):
    graph = deepcopy(base)
    next_id = max(map(int, graph)) + 1
    native = str(next_id)
    frozen = str(next_id + 1)
    graph[native] = {"class_type": NATIVE_LOAD, "inputs": {
        "checkpoint_path": "", "expected_manifest_json": "",
        "expected_file_sha256": "", "hash_chunk_megabytes": 8}}
    for key in ("20", "21", "26", "29"):
        if graph[key]["inputs"]["first_pass_latent"] != ["12", 1]:
            raise ValueError("Expected first-pass AV source on every v1 segment")
        graph[key]["inputs"]["first_pass_latent"] = [native, 0]
    graph[frozen] = {"class_type": SEGMENT_LOAD, "inputs": {
        "source_segment": ["20", 0], "segment_spec": ["20", 1],
        "pass2_context": ["21", 0], "plan": ["14", 0],
        "artifact_path": "", "artifact_sha256": ""}}
    for key in ("28", "36"):
        if graph[key]["inputs"]["previous_result"] != ["25", 1]:
            raise ValueError("Expected segment-0 result on both segment-1 consumers")
        graph[key]["inputs"]["previous_result"] = [frozen, 1]
    return _prune_api(graph, ("19", "37", "39"))


def _current_ui_frontend(frontend, api):
    """Align only the new private candidate with its saved API and installed VHS UI."""
    result = deepcopy(frontend)
    nodes = {node["id"]: node for node in result["nodes"]}
    image = api.get("5", {}).get("inputs", {}).get("image")
    if (not isinstance(image, str) or not image or
            nodes.get(5, {}).get("type") != "LoadImage" or
            nodes[5].get("widgets_values") != ["10A.jpg"]):
        raise ValueError("Unexpected source image in Chunked v1 candidate")
    nodes[5]["widgets_values"] = [image]

    video = nodes.get(19)
    if video is not None:
        settings = api.get("19", {}).get("inputs", {})
        expected = [24, 0, "MiniMaxH3_Chunked_Global_Noise_v2/output",
                    "video/h264-mp4", "yuv420p", 19, True, False, False, True]
        defaults = {"pix_fmt": "yuv420p", "crf": 19,
                    "save_metadata": True, "trim_to_audio": False}
        if (video.get("type") != "VHS_VideoCombine" or
                video.get("widgets_values") != expected or
                any(settings.get(name) != value for name, value in defaults.items()) or
                settings.get("format") != "video/h264-mp4"):
            raise ValueError("Unexpected VHS contract in Chunked v1 candidate")
        video["widgets_values"] = [settings[name] for name in (
            "frame_rate", "loop_count", "filename_prefix", "format",
            "pingpong", "save_output")]
    return result


def build_storage_pair(*, current_ui=False):
    frontend, api = build_pair()
    _expect_base(frontend, api)
    freeze = (freeze_frontend(frontend), freeze_api(api))
    resume = (resume_frontend(frontend), resume_api(api))
    if current_ui:
        freeze = (_current_ui_frontend(*freeze), freeze[1])
        resume = (_current_ui_frontend(*resume), resume[1])
    return freeze, resume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-ui", action="store_true",
                        help="Use API-matched image and current VHS six-widget layout")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifact directory; never overwrite existing graphs")
    freeze, resume = build_storage_pair(current_ui=args.current_ui)
    output.mkdir(parents=True)
    for name, (frontend, api) in (
        ("01_freeze_after_segment_0", freeze),
        ("02_resume_segments_1_and_2_DRAFT", resume),
    ):
        (output / f"{name}.json").write_text(
            json.dumps(frontend, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / f"{name}.api.json").write_text(
            json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"freeze_frontend_nodes": len(freeze[0]["nodes"]),
                      "resume_frontend_nodes": len(resume[0]["nodes"]),
                      "freeze_api_nodes": len(freeze[1]),
                      "resume_api_nodes": len(resume[1])}))


if __name__ == "__main__":
    main()
