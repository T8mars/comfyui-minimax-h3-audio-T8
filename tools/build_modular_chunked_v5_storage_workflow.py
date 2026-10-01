"""Build private v5 freeze/final-window-only drafts without changing the old example.

The resume graph loads a native partial4 AV checkpoint and a separately frozen
completed joint-AV window result. Both exact paths/SHAs must be pasted before
queueing. It regenerates the global learned lift/noise, but never runs LOW or
any completed PASS2 window.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from build_modular_chunked_v5_workflow import (
    ROOT, TEMPLATE, add_eav_api, add_eav_frontend, add_relay_api,
    add_relay_frontend, split_api, split_frontend,
)
sys.path.insert(0, str(ROOT))
from build_modular_h16_storage_workflow import Draft
from run_chunked_parity_probe import build_graph


RESULT = "T8_CHUNKED_V5_WINDOW_RESULT"
PREPARED = "T8_CHUNKED_V5_PREPARED"
PLAN = "T8_H3_CHUNKED_TWO_PASS_PLAN"


def _window_geometry(window_count):
    profiles = {2: (136, 34), 3: (102, 34), 4: (85, 34)}
    if window_count not in profiles:
        raise ValueError("Only the verified 192-frame 2/3/4-window plans are available")
    return profiles[window_count]


def _base_frontend(window_count=2):
    chunk, overlap = _window_geometry(window_count)
    return add_eav_frontend(add_relay_frontend(
        split_frontend(json.loads(TEMPLATE.read_text(encoding="utf-8")),
                       window_count=window_count, temporal_chunk_frames=chunk,
                       temporal_overlap_frames=overlap),
        with_audit=False,
    ))


def _base_api(window_count=2):
    chunk, overlap = _window_geometry(window_count)
    original = build_graph()
    del original["26"]  # The frozen plain route has no optional KJ selector.
    original["8"]["inputs"]["model"] = ["23", 0]
    return add_eav_api(add_relay_api(split_api(
        original, window_count=window_count, temporal_chunk_frames=chunk,
        temporal_overlap_frames=overlap), with_audit=False))


def _window_frontend(draft, index):
    found = [node for node in draft.graph["nodes"]
             if node["type"] == "MiniMaxH3ChunkedV5PASS2WindowEXPT8"
             and node["widgets_values"][0] == index]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one v5 frontend window {index}")
    return found[0]


def _window_api(graph, index):
    found = [(key, item) for key, item in graph.items()
             if item["class_type"] == "MiniMaxH3ChunkedV5PASS2WindowEXPT8"
             and item["inputs"]["window_index"] == index]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one v5 API window {index}")
    return found[0][0]


def _audit_frontend(draft, window_id):
    found = []
    for node in draft.graph["nodes"]:
        if node["type"] != "MiniMaxH3ChunkedV5EAVAuditEXPT8":
            continue
        input_slot = next(item for item in node["inputs"]
                          if item["name"] == "window_result")
        if draft.links[input_slot["link"]][1:3] == [window_id, 1]:
            found.append(node["id"])
    if len(found) != 1:
        raise ValueError("Expected one EAV audit of the selected window")
    return found[0]


def _audit_api(graph, window_id):
    found = [key for key, item in graph.items()
             if item["class_type"] == "MiniMaxH3ChunkedV5EAVAuditEXPT8"
             and item["inputs"]["window_result"] == [window_id, 1]]
    if len(found) != 1:
        raise ValueError("Expected one API EAV audit of the selected window")
    return found[0]


def freeze_frontend(base, *, saved_index=0):
    draft = Draft(base)
    saved_window = _window_frontend(draft, saved_index)
    native = draft.make(
        "MiniMaxH3NativeLatentCheckpointSaveT8Advanced",
        "STEP 1 · Freeze partial4 AV; enable confirm_save explicitly",
        [("av_latent", "LATENT")],
        [("av_latent", "LATENT"), ("status", "STRING"),
         ("checkpoint_path", "STRING"), ("file_sha256", "STRING"),
         ("manifest_json", "STRING"), ("report_json", "STRING")],
        ["v5_partial4_source", "v5_partial4_source", False, True, 8],
        (2550, -800),
    )
    draft.connect((12, 1), native, "av_latent", "LATENT")
    frozen = draft.make(
        "MiniMaxH3ChunkedV5WindowSaveEXPT8",
        f"STEP 2 · Freeze completed joint AV window {saved_index}; enable confirm_save",
        [("window_result", RESULT), ("partial4_denoised_output", "LATENT"),
         ("lifted_full_av", "LATENT"), ("prepared", PREPARED), ("plan", PLAN)],
        [("cumulative_av_latent", "LATENT"), ("window_result", RESULT),
         ("artifact_path", "STRING"), ("artifact_sha256", "STRING"),
         ("report_json", "STRING")], [False], (3550, -600),
    )
    for name, source, dtype in (
        ("window_result", (saved_window["id"], 1), RESULT),
        ("partial4_denoised_output", (12, 1), "LATENT"),
        ("lifted_full_av", (28, 0), "LATENT"),
        ("prepared", (29, 0), PREPARED), ("plan", (14, 0), PLAN),
    ):
        draft.connect(source, frozen, name, dtype)
    audits = tuple(_audit_frontend(draft, _window_frontend(draft, index)["id"])
                   for index in range(saved_index + 1))
    return draft.prune((native["id"], frozen["id"], *audits))


def resume_frontend(base, *, saved_index=0):
    draft = Draft(base)
    next_window = _window_frontend(draft, saved_index + 1)
    native = draft.make(
        "MiniMaxH3NativeLatentCheckpointLoadT8Advanced",
        "PASTE partial4 checkpoint path, manifest JSON and file SHA",
        [], [("av_latent", "LATENT"), ("status", "STRING"),
             ("resume_verified", "BOOLEAN"), ("checkpoint_id", "STRING"),
             ("content_sha256", "STRING"), ("file_sha256", "STRING"),
             ("manifest_json", "STRING"), ("report_json", "STRING")],
        ["", "", "", 8], (100, -750),
    )
    frozen = draft.make(
        "MiniMaxH3ChunkedV5WindowLoadEXPT8",
        f"PASTE exact window-{saved_index} manifest path + SHA; only window "
        f"{saved_index + 1} remains",
        [("partial4_denoised_output", "LATENT"),
         ("lifted_full_av", "LATENT"), ("prepared", PREPARED), ("plan", PLAN)],
        [("cumulative_av_latent", "LATENT"), ("window_result", RESULT),
         ("report_json", "STRING")], [saved_index, "", ""], (3500, -500),
    )
    bridge = draft.make(
        "MiniMaxH3ChunkedV5VerifiedNativeSourceEXPT8",
        "Verify native receipt and restore original partial4 source identity",
        [("av_latent", "LATENT"), ("resume_verified", "BOOLEAN"),
         ("checkpoint_id", "STRING"), ("content_sha256", "STRING"),
         ("file_sha256", "STRING"), ("manifest_json", "STRING"),
         ("report_json", "STRING")],
        [("partial4_denoised_output", "LATENT"),
         ("verification_report_json", "STRING")], [], (550, -750),
    )
    for name, slot, dtype in (
        ("av_latent", 0, "LATENT"), ("resume_verified", 2, "BOOLEAN"),
        ("checkpoint_id", 3, "STRING"), ("content_sha256", 4, "STRING"),
        ("file_sha256", 5, "STRING"), ("manifest_json", 6, "STRING"),
        ("report_json", 7, "STRING"),
    ):
        draft.connect((native["id"], slot), bridge, name, dtype)
    # The fresh process must reconstruct a plan/lift/noise from the exact
    # native partial4 artifact. No LOW sampler or first PASS2 window survives.
    for link in list(draft.graph["links"]):
        if link[1:3] != [12, 1]:
            continue
        target = draft.nodes[link[3]]
        name = target["inputs"][link[4]]["name"]
        draft.disconnect(target, name)
        draft.connect((bridge["id"], 0), target, name, link[5])
    for name, source, dtype in (
        ("partial4_denoised_output", (bridge["id"], 0), "LATENT"),
        ("lifted_full_av", (28, 0), "LATENT"),
        ("prepared", (29, 0), PREPARED), ("plan", (14, 0), PLAN),
    ):
        draft.connect(source, frozen, name, dtype)
    for target in draft.graph["nodes"]:
        if target["type"] not in {
                "MiniMaxH3ChunkedV5PASS2WindowEXPT8",
                "MiniMaxH3ChunkedV5RelayProjectEXPT8",
                "MiniMaxH3ChunkedV5EAVApplyEXPT8"} or target["widgets_values"][0] != saved_index + 1:
            continue
        draft.disconnect(target, "previous_result")
        draft.connect((frozen["id"], 1), target, "previous_result", RESULT)
    return draft.prune((25, _audit_frontend(draft, next_window["id"])))


def _prune_api(graph, roots):
    keep, pending = set(roots), list(roots)
    while pending:
        for value in graph[pending.pop()]["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                source = str(value[0])
                if source not in keep:
                    keep.add(source)
                    pending.append(source)
    return {key: value for key, value in graph.items() if key in keep}


def candidate_api(frontend, *, freeze, window_count=2):
    graph = deepcopy(_base_api(window_count))
    expected = {str(node["id"]): node["type"] for node in frontend["nodes"]}
    saved_index = window_count - 2
    saved_window = _window_api(graph, saved_index)
    next_window = _window_api(graph, saved_index + 1)
    native = str(max(map(int, graph)) + 1)
    window = str(int(native) + 1)
    bridge = str(int(window) + 1)
    if freeze:
        graph[native] = {"class_type": "MiniMaxH3NativeLatentCheckpointSaveT8Advanced",
                         "inputs": {"av_latent": ["12", 1],
                                    "filename_prefix": "v5_partial4_source",
                                    "checkpoint_id": "v5_partial4_source",
                                    "confirm_save": False, "verify_after_write": True,
                                    "hash_chunk_megabytes": 8}}
        graph[window] = {"class_type": "MiniMaxH3ChunkedV5WindowSaveEXPT8",
                         "inputs": {"window_result": [saved_window, 1],
                                    "partial4_denoised_output": ["12", 1],
                                    "lifted_full_av": ["28", 0],
                                    "prepared": ["29", 0], "plan": ["14", 0],
                                    "confirm_save": False}}
        audits = tuple(_audit_api(graph, _window_api(graph, index))
                       for index in range(saved_index + 1))
        graph = _prune_api(graph, (native, window, *audits))
    else:
        graph[native] = {"class_type": "MiniMaxH3NativeLatentCheckpointLoadT8Advanced",
                         "inputs": {"checkpoint_path": "", "expected_manifest_json": "",
                                    "expected_file_sha256": "", "hash_chunk_megabytes": 8}}
        graph[window] = {"class_type": "MiniMaxH3ChunkedV5WindowLoadEXPT8",
                         "inputs": {"partial4_denoised_output": [bridge, 0],
                                    "lifted_full_av": ["28", 0], "prepared": ["29", 0],
                                    "plan": ["14", 0],
                                    "expected_window_index": saved_index,
                                    "artifact_path": "", "artifact_sha256": ""}}
        for item in graph.values():
            for name, value in item["inputs"].items():
                if value == ["12", 1]:
                    item["inputs"][name] = [bridge, 0]
        graph[bridge] = {"class_type": "MiniMaxH3ChunkedV5VerifiedNativeSourceEXPT8",
                         "inputs": {"av_latent": [native, 0],
                                    "resume_verified": [native, 2],
                                    "checkpoint_id": [native, 3],
                                    "content_sha256": [native, 4],
                                    "file_sha256": [native, 5],
                                    "manifest_json": [native, 6],
                                    "report_json": [native, 7]}}
        for item in graph.values():
            if item["class_type"] in {
                    "MiniMaxH3ChunkedV5PASS2WindowEXPT8",
                    "MiniMaxH3ChunkedV5RelayProjectEXPT8",
                    "MiniMaxH3ChunkedV5EAVApplyEXPT8"} and item["inputs"]["window_index"] == saved_index + 1:
                item["inputs"]["previous_result"] = [window, 1]
        graph = _prune_api(graph, ("25", _audit_api(graph, next_window)))
    if any(expected.get(key) != item["class_type"] for key, item in graph.items()):
        raise ValueError("v5 storage frontend/API node identities diverged")
    return graph


def build_pair(window_count=2):
    base = _base_frontend(window_count)
    saved_index = window_count - 2
    freeze = freeze_frontend(base, saved_index=saved_index)
    resume = resume_frontend(base, saved_index=saved_index)
    return {
        f"01_freeze_window_{saved_index}": (
            freeze, candidate_api(freeze, freeze=True, window_count=window_count)),
        f"02_resume_window_{saved_index + 1}_DRAFT": (
            resume, candidate_api(resume, freeze=False, window_count=window_count)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-count", type=int, choices=(2, 3, 4), default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifact directory; never overwrite old candidates")
    pair = build_pair(args.window_count)
    output.mkdir(parents=True)
    for name, (frontend, api) in pair.items():
        (output / f"{name}.json").write_text(
            json.dumps(frontend, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / f"{name}.api.json").write_text(
            json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"frontend": len(frontend["nodes"]), "api": len(api)}
                      for name, (frontend, api) in pair.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
