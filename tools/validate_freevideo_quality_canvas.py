"""Validate NativeSaved graphs with actual UI API or explicit cold CPU projection. No Queue."""
import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def project_cold_api(native):
    """Pure projection, never label this as a browser export or GPU execution."""
    links = {row[0]: row for row in native["links"]}
    api = {}
    for node in native["nodes"]:
        if node["type"] in ("Note", "MarkdownNote"):
            continue
        values = {name: value for name, value in node.get("widgets_values_named", {}).items()
                  if name not in ("control_after_generate", "videopreview")}
        if node["type"] == "MiniMaxH3AudioConditioningT8" and values.get("task_type") == "T2VA — 文生音视频":
            values["task_type"] = "T2VA"  # check_serialized verifies the existing actual UI rule.
        for entry in node.get("inputs", []):
            if entry.get("link") is not None:
                edge = links[entry["link"]]
                values[entry["name"]] = [str(edge[1]), edge[2]]
        api[str(node["id"])] = dict(class_type=node["type"], inputs=values)
    return api


def check_serialized(native, api):
    links = {row[0]: row for row in native["links"]}
    if len(links) != len(native["links"]):
        raise ValueError("Duplicate link identity")
    nodes = {str(node["id"]): node for node in native["nodes"] if node["type"] not in ("Note", "MarkdownNote")}
    if set(nodes) != set(api):
        raise ValueError("Native/API node identities differ")
    aliases = []
    for key, node in nodes.items():
        exported = api[key]
        if exported["class_type"] != node["type"] or node.get("mode", 0) != 0:
            raise ValueError("Native/API node type/mode differs: " + key)
        values = dict(node.get("widgets_values_named", {}))
        positional = node.get("widgets_values")
        if isinstance(positional, list) and positional != list(values.values()):
            raise ValueError("Native positional/named widget values differ: " + key)
        # The existing task_type_labels.js deliberately displays Chinese labels
        # but serializes canonical API values. Verify the exact existing rule,
        # not arbitrary string truncation or ignoring the task input.
        if node["type"] == "MiniMaxH3AudioConditioningT8" and values.get("task_type") == "T2VA — 文生音视频":
            policy = (ROOT / "web/task_type_labels.js").read_bytes()
            if (b'widget.serializeValue = () => toBackendValue(widget.value);' not in policy
                or '["T2VA", "T2VA — 文生音视频"]' not in policy.decode("utf8")):
                raise ValueError("Actual task display/API serialization policy changed")
            aliases.append(dict(node=key, input="task_type", display=values["task_type"], backend="T2VA", policy_sha256=digest(policy)))
            values["task_type"] = "T2VA"
        # UI-only controls are not execution inputs; do not drop real parameters.
        expected = {name: value for name, value in values.items()
                    if name not in ("control_after_generate", "videopreview")}
        for index, entry in enumerate(node.get("inputs", [])):
            if entry.get("link") is None:
                continue
            edge = links[entry["link"]]
            if edge[3:5] != [node["id"], index]:
                raise ValueError("Native destination backlink differs")
            source = nodes[str(edge[1])]
            output = source["outputs"][edge[2]]
            if edge[0] not in (output.get("links") or []) or output["type"] != entry["type"] or edge[5] != entry["type"]:
                raise ValueError("Native source/type backlink differs")
            expected[entry["name"]] = [str(edge[1]), edge[2]]
        if expected != exported["inputs"]:
            differences = {name: dict(native=expected.get(name), exported=exported["inputs"].get(name))
                           for name in set(expected) | set(exported["inputs"])
                           if expected.get(name) != exported["inputs"].get(name)}
            raise ValueError("Native/API complete execution parameters differ: " + key + " " + repr(differences))
    return dict(nodes=len(nodes), links=len(links), every_execution_parameter_exact=True,
                native_positional_named_exact=True, typed_backlinks_exact=True, actual_UI_display_aliases=aliases)


async def validate(core, graphs):
    import nodes
    import server
    import torch
    from app.assets.manager import default_asset_manager
    # Register in this private CPU process. No socket listener or model execution.
    instance = server.PromptServer(asyncio.get_running_loop(), default_asset_manager())
    if not await nodes.load_custom_node(str(ROOT)):
        raise RuntimeError("Actual T8 entrypoint registration failed")
    if not await nodes.load_custom_node(str(core / "custom_nodes/ComfyUI-VideoHelperSuite")):
        raise RuntimeError("Actual VHS registration failed")
    import execution
    results = {}
    for name, graph in graphs.items():
        result = await execution.validate_prompt("fvq-cpu-no-queue-" + name, copy.deepcopy(graph), None)
        results[name] = result
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise RuntimeError("CPU-only validator unexpectedly has CUDA")
    # Keep the instance alive until registrations/validation finish, no start/setup.
    assert server.PromptServer.instance is instance
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--canvas-dir", type=Path, required=True)
    parser.add_argument("--api-dir", type=Path)
    parser.add_argument("--project-cold", action="store_true",
        help="Explicit CPU projection of actual saved Cold files, not a UI API export or execution")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--names", nargs="+", choices=["FVQ_Light_8plus3", "FVQ_Single_12",
        "FVQ_Light_Cold_HIGH3", "FVQ_Medium_Cold_Decode"],
        default=["FVQ_Light_8plus3", "FVQ_Single_12"])
    args = parser.parse_args()
    if len(args.names) != len(set(args.names)):
        raise ValueError("Duplicate native graph identity")
    cold_names = {"FVQ_Light_Cold_HIGH3", "FVQ_Medium_Cold_Decode"}
    if args.project_cold and (args.api_dir is not None or not set(args.names) <= cold_names):
        raise ValueError("CPU projection is only permitted for explicit saved Cold files")
    if not args.project_cold and args.api_dir is None:
        raise ValueError("Actual native UI API directory required")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    sys.path.insert(0, str(args.core))
    sys.argv = ["fvq-native-validation", "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    torch.set_num_threads(2)
    graphs, evidence = {}, {}
    for name in args.names:
        canvas = args.canvas_dir / (name + ".json")
        raw = canvas.read_bytes()
        native = json.loads(raw)
        if args.project_cold:
            api = project_cold_api(native)
            api_raw = json.dumps(api, ensure_ascii=False, allow_nan=False).encode("utf8")
            exported, provenance = None, "CPU_projection_not_UI_export_not_execution"
        else:
            exported = args.api_dir / (name + ".json")
            api_raw = exported.read_bytes()
            api, provenance = json.loads(api_raw), "actual_native_UI_API_export"
        print(json.dumps(dict(canvas=str(canvas), sha256=digest(raw), api_sha256=digest(api_raw))), flush=True)
        graphs[name] = api
        evidence[name] = dict(canvas_path=str(canvas), canvas_sha256=digest(raw),
            exported_api_path=str(exported) if exported else None, exported_api_sha256=digest(api_raw),
            API_provenance=provenance,
            serialization=check_serialized(native, api))
        with (args.output / (name + ".api.json")).open("xb") as stream:
            stream.write(api_raw)
    results = asyncio.run(validate(args.core, graphs))
    for name, result in results.items():
        evidence[name]["actual_Core_validation"] = result
    passed = all(result[0] and not result[3] for result in results.values())
    record = dict(status="pass" if passed else "failed", cases=evidence,
        CPU_only=True, CUDA_initialized=torch.cuda.is_initialized(), queued=False,
        GPU_quality_accepted=False, models_loaded=False,
        scope="Actual NativeSaved + recorded API provenance; exact parameters/wires + real Core validator, not sampling")
    write(args.output / "audit.json", record)
    print(json.dumps(record, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
