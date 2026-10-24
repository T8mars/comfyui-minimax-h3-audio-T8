"""Explicit raw-API preparation, sharing the owned H21 execution controller.

No dummy Director project or unused sampling override is invented. Original
graph bytes remain frozen alongside the private-input/output remapped graph.
Only graph validation and full-file hashing execute here; no weight is loaded.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
import re
import sys
import uuid


FILE_READERS = {"LoadVideo": "file", "LoadImage": "image",
                "LoadImageMask": "image", "LoadAudio": "audio"}


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def validate_config(config):
    allowed = {"schema", "mode", "raw_api_cases", "assets", "models",
               "model_directories", "reserve_vram_gb", "disable_pinned_memory", "allocator_telemetry"}
    if set(config) - allowed or config.get("mode") != "raw_api":
        raise ValueError("Raw API configuration has unsupported or unapplied fields")
    if "disable_pinned_memory" in config and type(config["disable_pinned_memory"]) is not bool:
        raise ValueError("disable_pinned_memory must be an explicit bool")
    if "allocator_telemetry" in config and type(config["allocator_telemetry"]) is not bool:
        raise ValueError("allocator_telemetry must be an explicit bool")
    cases = config.get("raw_api_cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Raw API preparation requires explicit graph cases")
    for case in cases:
        if (not isinstance(case, dict) or set(case) != {"name", "api", "api_sha256"}
                or not isinstance(case["api"], str) or not _sha(case["api_sha256"])):
            raise ValueError("Each raw case requires name, API file and its full SHA256")
    assets = config.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("Raw assets require explicit UUID/bytes/node-input bindings")
    for asset_id, item in assets.items():
        try:
            if str(uuid.UUID(asset_id)) != asset_id:
                raise ValueError("Noncanonical asset UUID")
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError("Raw asset identity must be an explicit UUID") from error
        if (not isinstance(item, dict)
                or set(item) != {"source", "sha256", "private_relative", "nodes"}
                or not _sha(item["sha256"]) or not isinstance(item["nodes"], list)
                or not item["nodes"]):
            raise ValueError("Raw asset needs source SHA, private path and input bindings")
        for binding in item["nodes"]:
            if (not isinstance(binding, dict) or set(binding) != {"node", "input"}
                    or any(not isinstance(value, str) or not value for value in binding.values())):
                raise ValueError("Raw asset binding requires exact node/input strings")
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Raw execution requires explicit selected model bindings")
    for item in models:
        if (not isinstance(item, dict)
                or set(item) - {"node", "input", "folder", "name", "sha256"}
                or not {"node", "input", "folder", "name"} <= set(item)
                or any(not isinstance(item[key], str) or not item[key]
                       for key in ("node", "input", "folder", "name"))
                or "sha256" in item and not _sha(item["sha256"])):
            raise ValueError("Raw model bindings need exact node/input/folder/name and optional SHA")


def build(plan, folders, core_nodes, execution, resolver, batch, *, api):
    """Prepare actual raw graphs under already configured, CPU-only Core."""
    read, write, identity = (api[key] for key in ("read_json", "write_json", "file_identity"))
    inside, copy = api["inside"], api["copy_exact"]
    from comfy_extras.nodes_images import ImageFromBatch
    from comfy_extras.nodes_audio import LoadAudio

    core_nodes.NODE_CLASS_MAPPINGS["ImageFromBatch"] = ImageFromBatch
    core_nodes.NODE_CLASS_MAPPINGS["LoadAudio"] = LoadAudio
    root = Path(plan["run_root"])
    config = read(root / "config.json")
    validate_config(config)
    assets, bindings, occupied = {}, {}, set()
    for asset_id, item in config["assets"].items():
        target = inside(root / "input", item["private_relative"])
        before = copy(item["source"], target)
        if before["sha256"] != item["sha256"]:
            raise ValueError("Raw asset source SHA differs: " + asset_id)
        assets[asset_id] = {"sha256": before["sha256"],
                            "server_path": target.relative_to(root / "input").as_posix()}
        for row in item["nodes"]:
            key = row["node"], row["input"]
            if key in occupied:
                raise ValueError("Raw asset node/input is bound more than once")
            occupied.add(key)
            bindings[key] = assets[asset_id]["server_path"]
    models, declared_slots = {}, {}
    for row in config["models"]:
        slot = row["node"], row["input"]
        if slot in declared_slots:
            raise ValueError("Raw model node/input is bound more than once")
        declared_slots[slot] = row
        key = row["folder"] + ":" + row["name"]
        if key not in models:
            path = Path(resolver(row["folder"], row["name"])).resolve(strict=True)
            frozen = identity(path)
            models[key] = {"folder": row["folder"], "name": row["name"],
                           "path": frozen["path"], "size": frozen["bytes"],
                           "sha256": frozen["sha256"]}
        if row.get("sha256", models[key]["sha256"]) != models[key]["sha256"]:
            raise ValueError("Raw selected model SHA differs: " + key)
    cases, kinds = [], set()
    for case in config["raw_api_cases"]:
        relative = "cases/" + case["name"]
        before = copy(case["api"], root / relative / "original-api.json")
        if before["sha256"] != case["api_sha256"]:
            raise ValueError("Raw graph source SHA differs: " + case["name"])
        original = read(root / relative / "original-api.json")
        if not isinstance(original, dict) or not original:
            raise ValueError("Raw graph must be an actual nonempty API node mapping")
        graph, remaps = deepcopy(original), []
        for (node, field), target in bindings.items():
            inputs = graph.get(node, {}).get("inputs", {})
            if field not in inputs or not isinstance(inputs[field], str):
                raise ValueError("Raw input binding is absent or not a literal path")
            remaps.append({"node": node, "input": field, "before": inputs[field], "after": target})
            inputs[field] = target
        for (node, field), row in declared_slots.items():
            if graph.get(node, {}).get("inputs", {}).get(field) != row["name"]:
                raise ValueError("Raw model binding differs from selected graph input")
        for node, value in graph.items():
            kind, inputs = value["class_type"], value["inputs"]
            spec = batch._MODEL_INPUTS.get(kind)
            if spec is not None and (node, spec[0]) not in declared_slots:
                raise ValueError("Raw graph model consumer lacks an explicit resource binding")
            if spec is not None:
                actual = Path(resolver(spec[1], inputs[spec[0]])).resolve(strict=True)
                row = declared_slots[node, spec[0]]
                frozen = models[row["folder"] + ":" + row["name"]]
                if str(actual) != frozen["path"]:
                    raise ValueError("Raw declaration resolves differently from the actual node resource")
            if kind in FILE_READERS and (node, FILE_READERS[kind]) not in bindings:
                raise ValueError("Raw graph file reader lacks an explicit private asset binding: "
                                 + node + "/" + FILE_READERS[kind])
            if kind == "MiniMaxH3SafeAVSaveT8Advanced":
                target = "H21_Control/" + plan["run_id"] + "/" + case["name"]
                remaps.append({"node": node, "input": "filename_prefix",
                               "before": inputs["filename_prefix"], "after": target})
                inputs["filename_prefix"] = target
        base = deepcopy(graph)
        validation = asyncio.run(execution.validate_prompt(str(uuid.uuid4()), base, None))
        write(root / relative / "base-core-validation.json", _validation(validation))
        if validation[0] is not True or validation[3]:
            raise RuntimeError("Real CPU Core rejected raw base graph " + case["name"])
        graph, observation = api["append_report_observers"](base, validation[2])
        validation = asyncio.run(execution.validate_prompt(str(uuid.uuid4()), graph, None))
        write(root / relative / "core-validation.json", _validation(validation))
        if validation[0] is not True or validation[3]:
            raise RuntimeError("Real CPU Core rejected observed raw graph " + case["name"])
        write(root / relative / "base-prompt.json", base)
        write(root / relative / "prompt.json", graph)
        write(root / relative / "remaps.json", remaps)
        cases.append({"name": case["name"], "prompt": relative + "/prompt.json",
                      "base_prompt": relative + "/base-prompt.json",
                      "original_api": relative + "/original-api.json",
                      "validation": relative + "/core-validation.json",
                      "output_nodes": list(validation[2]), "observation": observation,
                      "graph_origin": "explicit_raw_api", "source_api_sha256": before["sha256"]})
        kinds.update(value["class_type"] for value in graph.values())
    imports = {}
    for key, module in list(sys.modules.items()):
        if not key.startswith(plan["package_name"] + ".") or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(Path(plan["package_root"])):
            raise RuntimeError("Raw preparation imported live instead of frozen extension")
        imports[key] = identity(path)
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("Raw CPU preparation unexpectedly initialized CUDA")
    for relative, frozen in plan["source_files"].items():
        if identity(root / relative)["sha256"] != frozen["sha256"]:
            raise RuntimeError("Frozen source changed during raw CPU preparation")
    write(root / "build-result.json", {
        "cases": cases, "resources": {"models": models, "assets": assets},
        "graph_class_types": sorted(kinds), "builder_imports": imports,
        "effective_model_directories": {
            key: {"paths": list(paths), "extensions": sorted(extensions)}
            for key, (paths, extensions) in folders.folder_names_and_paths.items()},
        "model_execution": False, "core_validate_prompt": True, "cuda_initialized": False,
        "graph_origin": "explicit_raw_api",
        "dependency_scope": "current host packages; frozen raw graphs/extension, not historical environment pin",
    })


def _validation(result):
    return {"valid": result[0], "error": result[1],
            "outputs": list(result[2]), "node_errors": result[3]}
