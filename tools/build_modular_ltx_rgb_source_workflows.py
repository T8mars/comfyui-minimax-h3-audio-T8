"""Additive full/freeze/cold RGB-LTX input graphs; never alter old examples."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_formal_ltx_rgb_split_workflows import generated as original_graphs  # noqa: E402
from tools.build_modular_ltx_rgb_workflows import Draft, _unique, _source, BIND, AUDIT  # noqa: E402
from tools.build_modular_audio_refine_workflows import split_api  # noqa: E402

SAVE = "MiniMaxH3LTXRGBSourceSaveEXPT8"
LOAD = "MiniMaxH3LTXRGBSourceLoadEXPT8"
SERIAL_READER = "MiniMaxH3VideoComponentsSerialEXPT8"
ISOLATED_WRITER = "MiniMaxH3SaveVideoIsolatedEXPT8"
DATA = [("source_frames", "IMAGE"), ("source_audio", "AUDIO"), ("prepared_frames", "IMAGE"),
        ("prep_report_json", "STRING"), ("ltx_latent", "LATENT")]
OUTPUTS = [*DATA, ("fps", "FLOAT"), ("duration_seconds", "FLOAT"), ("bit_depth", "COMBO")]


def build(source, variant):
    if variant not in {"full_save", "freeze_source", "resume_ltx"}:
        raise ValueError("Unknown RGB source graph variant")
    draft = Draft(source)
    bind, audit = _unique(draft, BIND), _unique(draft, AUDIT)
    create, trim = _unique(draft, "CreateVideo"), _unique(draft, "MiniMaxH3OutputTrimT8")
    if variant == "resume_ltx":
        io_node = draft.make(LOAD, "PASTE exact frozen input manifest and SHA; only LTX reruns", [],
            [*OUTPUTS, ("report_json", "STRING")], ["", ""], (1400, 300))
    else:
        io_node = draft.make(SAVE, "Freeze RGB/audio + encoded/upscaled LTX input; explicit enable", [*DATA, ("bit_depth", "COMBO")],
            [*OUTPUTS, ("artifact_path", "STRING"), ("artifact_sha256", "STRING"), ("report_json", "STRING")],
            ["", 8, "rgb_source", False], (1400, 300))
        for name, dtype in DATA:
            slot, actual_type = _source(draft, bind, name)
            if actual_type != dtype:
                raise ValueError("Original RGB/LTX source wire type changed")
            draft.connect(slot, io_node, name, dtype)
        depth, dtype = _source(draft, create, "bit_depth")
        if dtype != "COMBO":
            raise ValueError("Original source bit-depth wire changed")
        draft.connect(depth, io_node, "bit_depth", dtype)
        if variant == "freeze_source":
            return draft.prune((io_node["id"],))
    for slot, (name, dtype) in enumerate(DATA):
        for target in (bind, audit):
            if target is audit and name == "ltx_latent":
                continue  # Audit still consumes the bound input, not a bypass.
            draft.disconnect(target, name)
            draft.connect((io_node["id"], slot), target, name, dtype)
    for target, name, slot, dtype in ((trim, "fps", 5, "FLOAT"), (trim, "duration_seconds", 6, "FLOAT"),
                                    (create, "fps", 5, "FLOAT"), (create, "bit_depth", 7, "COMBO")):
        draft.disconnect(target, name)
        draft.connect((io_node["id"], slot), target, name, dtype)
    return draft.prune((_unique(draft, "SaveVideo")["id"], audit["id"]))


def isolated_media_graph(graph):
    """Opt-in candidate only: retain every non-I/O node, value and wire."""
    from copy import deepcopy
    graph = deepcopy(graph)
    for node in graph["nodes"]:
        if node["type"] == "GetVideoComponents":
            node["type"] = SERIAL_READER
            node["title"] = "Serial source read; original RGB/audio/trim retained (EXP)"
            node["outputs"].append({"name": "report_json", "type": "STRING", "links": []})
        elif node["type"] == "SaveVideo":
            if node["widgets_values"][1:] != ["auto", "auto"]:
                raise ValueError("Explicit nondefault format/codec needs a separate export choice")
            node["type"] = ISOLATED_WRITER
            node["title"] = "Isolated H.264/AAC; strict decode before save (Windows EXP)"
            node["widgets_values"] = [node["widgets_values"][0], 600, 16., 2.]
            node["outputs"].extend({"name": name, "type": "STRING", "links": []}
                                   for name in ("video_path", "report_json"))
            node["size"] = [470, 260]
        else:
            continue
        node["properties"] = {"cnr_id": "minimax-h3-audio-T8", "Node name for S&R": node["type"]}
    return graph


def generated(*, isolated_media=False):
    from tools.workflow_paths import legacy_workflow_path
    graphs = {legacy_workflow_path(path).stem + "_" + variant: build(graph, variant)
            for path, graph in original_graphs().items()
            for variant in ("full_save", "freeze_source", "resume_ltx")}
    return {name: isolated_media_graph(graph) for name, graph in graphs.items()} if isolated_media else graphs


async def validate(graphs):
    import nodes
    import execution
    import importlib.util
    from tools.validate_modular_ltx_rgb_core import MODULES
    for name in MODULES:
        if not await nodes.load_custom_node(str(ROOT.parents[1] / "comfy_extras" / name), module_parent="comfy_extras"):
            raise RuntimeError("Failed to load Core node module: " + name)
    spec = importlib.util.spec_from_file_location("_t8_ltx_source_validate", ROOT / "__init__.py",
                                               submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    for cls in await package.comfy_entrypoint().get_node_list():
        nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
    kinds = {node["type"] for graph in graphs.values() for node in graph["nodes"]} - {"MarkdownNote"}
    info = {kind: (nodes.NODE_CLASS_MAPPINGS[kind].GET_NODE_INFO_V1()
                   if hasattr(nodes.NODE_CLASS_MAPPINGS[kind], "GET_NODE_INFO_V1")
                   else nodes.NODE_CLASS_MAPPINGS[kind].INPUT_TYPES()) for kind in kinds}
    results, apis = {}, {}
    for name, graph in graphs.items():
        api = split_api(graph, info)
        # Only the static validation copy uses an existing local video name.
        # Saved frontend retains the original explicit placeholder.
        for node in api.values():
            if node["class_type"] == "LoadVideo":
                node["inputs"]["file"] = "0.6.mp4"
        checked = await execution.validate_prompt(name, api, None)
        expected = {key for key, node in api.items() if node["class_type"] in {
            SAVE, AUDIT, "SaveVideo", ISOLATED_WRITER, "MiniMaxH3LTXEAVAuditEXPT8",
            "MiniMaxH3LTXPromptRelayAuditEXPT8"}}
        results[name] = {"valid": bool(checked[0]) and not checked[3] and set(checked[2]) == expected,
                         "result": checked, "api_nodes": len(api), "expected_outputs": sorted(expected)}
        apis[name] = api
    return results, apis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isolated-media", action="store_true")
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to((ROOT / "artifacts").resolve()) or destination.exists():
        raise ValueError("A new private artifacts directory is required")
    graphs = generated(isolated_media=args.isolated_media)
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2")
    sys.path[:0] = [str(ROOT.parents[1]), str(ROOT)]
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    torch.set_num_threads(2)
    results, apis = asyncio.run(validate(graphs))
    if torch.cuda.is_initialized():
        raise RuntimeError("Static validation initialized CUDA")
    destination.mkdir(parents=True)
    for name, graph in graphs.items():
        (destination / (name + ".json")).write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf8")
        (destination / (name + ".api.json")).write_text(json.dumps(apis[name], ensure_ascii=False, indent=2), encoding="utf8")
    report = {"cases": results, "status": "pass" if all(row["valid"] for row in results.values()) else "fail",
              "queued": False, "cuda_initialized": False, "isolated_media": args.isolated_media,
              "api_source_video_schema_substitution": "0.6.mp4"}
    (destination / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps({"status": report["status"], "cases": len(results), "directory": str(destination)}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
