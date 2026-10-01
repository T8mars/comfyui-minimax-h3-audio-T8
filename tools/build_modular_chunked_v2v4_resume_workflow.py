"""Build private v2-v4 first-pass freeze / HIGH-only Relay+EAV workflow pairs.

These are explicit checkpoint workflows, not automatic recipe-cache hits. The
saved first-pass file path, external manifest and file SHA must be copied into
the matching resume graph. Frozen source graphs and formal examples are read
only; outputs may be written only to a new private artifacts directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from tools import run_community_update_real_validation as legacy_api
from tools.build_modular_chunked_v1_storage_workflow import (
    NATIVE_LOAD, NATIVE_SAVE, _prune_api,
)
from tools.build_modular_chunked_workflow import (
    ROOT, add_external_eav_api, add_external_relay_api, split_api_graph,
)
from tools.build_modular_h16_storage_workflow import Draft


SOURCE = ROOT / "artifacts/development/modular-sampling-m4-chunked-20260923/candidate-relay-eav-v1"
SOURCES = {
    "v2": (SOURCE / "Chunked_v2_External_Relay_EAV_EXP.json",
           "349565ca69c2d088794f0805935f5fe52778566118c6ac3a898d006e5bc63201"),
    "v3": (SOURCE / "Chunked_v3_External_Relay_EAV_EXP.json",
           "f3bf7db46a40eee409f72cbb051bf1be09e1a83d1e0b17510c2e7e4931e961c1"),
    "v4": (SOURCE / "Chunked_v4_External_Relay_EAV_EXP.json",
           "3271d65a17a36a3a67d6ec25a41841dcff6efb182a159b2503ecb0936d5af55e"),
}
BUILDERS = {
    "v2": ("chunked_two_pass_global_noise", legacy_api._chunked_prompt),
    "v3": ("chunked_two_pass_low_sigma", legacy_api._chunked_low_sigma_prompt),
    "v4": ("chunked_two_pass_masked_low_sigma", legacy_api._chunked_masked_low_sigma_prompt),
}
SOURCE_TYPE = "MiniMaxH3ChunkedSourceSegmentEXPT8"
PREPARE_TYPE = "MiniMaxH3ChunkedPass2PrepareEXPT8"
PASS_TYPE = "MiniMaxH3ChunkedPass2SegmentEXPT8"
AUDIT_TYPE = "MiniMaxH3ChunkedPass2EAVAuditEXPT8"


def _frontend_source(variant: str) -> dict:
    path, digest = SOURCES[variant]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Frozen v2-v4 source candidate changed: {path}")
    graph = json.loads(raw)
    kinds = [node["type"] for node in graph["nodes"]]
    if (kinds.count("SamplerCustomAdvanced") != 1 or
            kinds.count(SOURCE_TYPE) != 1 or kinds.count(PREPARE_TYPE) != 1 or
            kinds.count(PASS_TYPE) != 1 or kinds.count(AUDIT_TYPE) != 1 or
            kinds.count("MiniMaxH3ChunkedPass2RelayBindEXPT8") != 1 or
            kinds.count("MiniMaxH3ChunkedPass2EAVApplyEXPT8") != 1):
        raise ValueError("Frozen v2-v4 source is not the expected Relay+EAV split")
    return graph


def _align_api_ids(frontend: dict, api: dict) -> dict:
    """Match API-only numbering to saved frontend IDs, then verify every wire."""
    frontend_nodes = {str(node["id"]): node for node in frontend["nodes"]}
    mapping = {
        key: key for key, node in api.items()
        if key in frontend_nodes and frontend_nodes[key]["type"] == node["class_type"]
    }
    available = set(frontend_nodes) - set(mapping.values())
    for key, node in api.items():
        if key in mapping:
            continue
        matches = [item for item in available
                   if frontend_nodes[item]["type"] == node["class_type"]]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous frontend/API node alignment: {key}")
        mapping[key] = matches[0]
        available.remove(matches[0])
    aligned = {}
    for key, node in api.items():
        copy = deepcopy(node)
        for name, value in copy["inputs"].items():
            if (isinstance(value, list) and len(value) == 2 and
                    isinstance(value[0], str) and value[0] in api and
                    isinstance(value[1], int)):
                copy["inputs"][name] = [mapping[value[0]], value[1]]
        aligned[mapping[key]] = copy
    _align_api_widgets(frontend, aligned)
    _assert_wires_match(frontend, aligned, require_exact=False)
    return aligned


def _align_api_widgets(frontend: dict, api: dict) -> None:
    nodes = {str(node["id"]): node for node in frontend["nodes"]}
    trailing_ui_only = {
        "MiniMaxH3AudioConditioningT8": [True],
        "RandomNoise": ["fixed"],
    }
    for key, item in api.items():
        node = nodes[key]
        values = list(node.get("widgets_values") or [])
        scalar_names = [name for name, value in item["inputs"].items()
                        if not (isinstance(value, list) and len(value) == 2 and
                                isinstance(value[0], str) and value[0] in api and
                                isinstance(value[1], int))]
        if not scalar_names and not values:
            continue
        tail = trailing_ui_only.get(node["type"], [])
        if len(values) != len(scalar_names) + len(tail) or (
                tail and values[-len(tail):] != tail):
            raise ValueError(f"Unexpected saved widgets for {key}:{node['type']}")
        for name, value in zip(scalar_names, values, strict=False):
            item["inputs"][name] = value


def _assert_wires_match(frontend: dict, api: dict, *, require_exact: bool = True) -> None:
    links = {link[0]: link for link in frontend["links"]}
    frontend_nodes = {str(node["id"]): node for node in frontend["nodes"]}
    if (set(api) != set(frontend_nodes) if require_exact
            else not set(api).issubset(frontend_nodes)):
        raise ValueError("Frontend/API execution nodes do not match")
    for key, item in api.items():
        node = frontend_nodes[key]
        if item["class_type"] != node["type"]:
            raise ValueError("Frontend/API class type mismatch")
        for pin in node.get("inputs", []):
            if pin.get("link") is not None:
                link = links[pin["link"]]
                expected = [str(link[1]), link[2]]
                if item["inputs"].get(pin["name"]) != expected:
                    raise ValueError(f"Frontend/API wire mismatch: {node['id']}.{pin['name']}")


def _api_source(variant: str, frontend: dict) -> dict:
    mode, builder = BUILDERS[variant]
    args = legacy_api._parser().parse_args(["--mode", mode])
    original, _ = builder(args, "STATIC")
    graph = add_external_eav_api(add_external_relay_api(
        split_api_graph(original), with_audit=False))
    if (graph.get("12", {}).get("class_type") != "SamplerCustomAdvanced" or
            sum(node["class_type"] == PASS_TYPE for node in graph.values()) != 1 or
            sum(node["class_type"] == AUDIT_TYPE for node in graph.values()) != 1):
        raise ValueError("Current API builder lost v2-v4 stage topology")
    # The legacy API builder omits frontend-only notes (and v3/v4 first-pass
    # previews), so its appended IDs differ from the frozen frontend graph.
    # Align IDs and saved frontend controls; no published source file is rewritten.
    return _align_api_ids(frontend, graph)


def _only(frontend: dict, kind: str) -> dict:
    matches = [node for node in frontend["nodes"] if node["type"] == kind]
    if len(matches) != 1:
        raise ValueError(f"Expected one frontend {kind}")
    return matches[0]


def _only_api(api: dict, kind: str) -> str:
    matches = [key for key, node in api.items() if node["class_type"] == kind]
    if len(matches) != 1:
        raise ValueError(f"Expected one API {kind}")
    return matches[0]


def _ancestors_api(graph: dict, root: str) -> set[str]:
    seen = set()
    pending = [root]
    while pending:
        for value in graph[pending.pop()]["inputs"].values():
            if (isinstance(value, list) and len(value) == 2 and
                    isinstance(value[0], str) and value[0] in graph and
                    isinstance(value[1], int) and value[0] not in seen):
                seen.add(value[0])
                pending.append(value[0])
    return seen


def _final_vhs_frontend(graph: dict) -> int:
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    pass_id = _only(graph, PASS_TYPE)["id"]

    def ancestors(root):
        seen, pending = set(), [root]
        while pending:
            for item in nodes[pending.pop()].get("inputs", []):
                if item.get("link") is not None:
                    parent = links[item["link"]][1]
                    if parent not in seen:
                        seen.add(parent)
                        pending.append(parent)
        return seen

    matches = [node["id"] for node in graph["nodes"]
               if node["type"] == "VHS_VideoCombine" and pass_id in ancestors(node["id"])]
    if len(matches) != 1:
        raise ValueError("Expected one final v2-v4 VHS terminal downstream of PASS2")
    return matches[0]


def _final_vhs_api(graph: dict) -> str:
    pass_id = _only_api(graph, PASS_TYPE)
    matches = [key for key, node in graph.items()
               if node["class_type"] == "VHS_VideoCombine" and
               pass_id in _ancestors_api(graph, key)]
    if len(matches) != 1:
        raise ValueError("Expected one final v2-v4 API VHS downstream of PASS2")
    return matches[0]


def freeze_frontend(base: dict, variant: str) -> dict:
    draft = Draft(base)
    save = draft.make(
        NATIVE_SAVE, f"{variant} STEP 1 · save completed first-pass AV (enable confirm_save)",
        [("av_latent", "LATENT")],
        [("av_latent", "LATENT"), ("status", "STRING"),
         ("checkpoint_path", "STRING"), ("file_sha256", "STRING"),
         ("manifest_json", "STRING"), ("report_json", "STRING")],
        [f"chunked_{variant}_first_pass", f"chunked_{variant}_first_pass", False, True, 8],
        (3200, -600),
    )
    draft.connect((12, 1), save, "av_latent", "LATENT")
    return draft.prune((save["id"],))


def resume_frontend(base: dict, variant: str) -> dict:
    draft = Draft(base)
    final_vhs = _final_vhs_frontend(base)
    audit_id = _only(base, AUDIT_TYPE)["id"]
    load = draft.make(
        NATIVE_LOAD, f"{variant} STEP 2 · paste exact path + manifest + file SHA, run HIGH only",
        [], [("av_latent", "LATENT"), ("status", "STRING"),
             ("resume_verified", "BOOLEAN"), ("checkpoint_id", "STRING"),
             ("content_sha256", "STRING"), ("file_sha256", "STRING"),
             ("manifest_json", "STRING"), ("report_json", "STRING")],
        ["", "", "", 8], (300, -700),
    )
    for kind in (SOURCE_TYPE, PREPARE_TYPE):
        node = _only(base, kind)
        item = next(item for item in node["inputs"] if item["name"] == "first_pass_latent")
        link = next(link for link in base["links"] if link[0] == item["link"])
        if link[1:3] != [12, 1]:
            raise ValueError("v2-v4 first-pass source changed")
        draft.disconnect(draft.nodes[node["id"]], "first_pass_latent")
        draft.connect((load["id"], 0), draft.nodes[node["id"]],
                      "first_pass_latent", "LATENT")
    return draft.prune((final_vhs, audit_id))


def freeze_api(base: dict, variant: str) -> dict:
    graph = deepcopy(base)
    save_id = str(max(map(int, graph)) + 1)
    graph[save_id] = {"class_type": NATIVE_SAVE, "inputs": {
        "av_latent": ["12", 1], "filename_prefix": f"chunked_{variant}_first_pass",
        "checkpoint_id": f"chunked_{variant}_first_pass", "confirm_save": False,
        "verify_after_write": True, "hash_chunk_megabytes": 8}}
    return _prune_api(graph, (save_id,))


def resume_api(base: dict) -> dict:
    graph = deepcopy(base)
    final_vhs = _final_vhs_api(graph)
    audit_id = _only_api(graph, AUDIT_TYPE)
    load_id = str(max(map(int, graph)) + 1)
    graph[load_id] = {"class_type": NATIVE_LOAD, "inputs": {
        "checkpoint_path": "", "expected_manifest_json": "",
        "expected_file_sha256": "", "hash_chunk_megabytes": 8}}
    for kind in (SOURCE_TYPE, PREPARE_TYPE):
        node = graph[_only_api(graph, kind)]
        if node["inputs"]["first_pass_latent"] != ["12", 1]:
            raise ValueError("v2-v4 API first-pass source changed")
        node["inputs"]["first_pass_latent"] = [load_id, 0]
    return _prune_api(graph, (final_vhs, audit_id))


def _current_ui(frontend: dict, api: dict) -> dict:
    result = deepcopy(frontend)
    for node in result["nodes"]:
        item = api.get(str(node["id"]), {}).get("inputs", {})
        if node["type"] == "LoadImage" and isinstance(item.get("image"), str):
            node["widgets_values"] = [item["image"]]
        if node["type"] == "VHS_VideoCombine":
            names = ("frame_rate", "loop_count", "filename_prefix", "format",
                     "pingpong", "save_output")
            if not all(name in item for name in names):
                raise ValueError("VHS API source lost a current control")
            node["widgets_values"] = [item[name] for name in names]
    return result


def build_pair(variant: str) -> tuple[tuple[dict, dict], tuple[dict, dict]]:
    if variant not in SOURCES:
        raise ValueError("Only Chunked v2-v4 full-clip contracts are supported")
    frontend = _frontend_source(variant)
    api = _api_source(variant, frontend)
    freeze = (freeze_frontend(frontend, variant), freeze_api(api, variant))
    resume = (resume_frontend(frontend, variant), resume_api(api))
    pairs = ((_current_ui(*freeze), freeze[1]),
             (_current_ui(*resume), resume[1]))
    for graph, prompt in pairs:
        _assert_wires_match(graph, prompt)
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifacts directory; never overwrite existing graphs")
    pairs = {variant: build_pair(variant) for variant in SOURCES}
    output.mkdir(parents=True)
    summary = {}
    for variant, (freeze, resume) in pairs.items():
        branch = output / variant
        branch.mkdir()
        for label, (frontend, api) in (("01_freeze_first_pass", freeze),
                                       ("02_resume_high_only_DRAFT", resume)):
            (branch / f"{label}.json").write_text(
                json.dumps(frontend, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (branch / f"{label}.api.json").write_text(
                json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary[variant] = {"freeze_nodes": len(freeze[0]["nodes"]),
                            "resume_nodes": len(resume[0]["nodes"]),
                            "freeze_api": len(freeze[1]), "resume_api": len(resume[1])}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
