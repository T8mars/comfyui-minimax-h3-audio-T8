"""Frozen, isolated Director control qualification controller.

Default prepare copies explicitly supplied media and runtime source, then builds
real Director graphs in a CPU-only subprocess. It starts no Core or GPU queue.
Only --execute MANIFEST starts the separately owned localhost Core process.
Model search roots are explicit, weights stay in place, and no browser is used.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import socket
import subprocess
import sys
import time
import uuid


SCHEMA = "t8.h21.control_probe.v1"
MODEL_FOLDERS = (
    "diffusion_models",
    "text_encoders",
    "vae",
    "loras",
    "latent_upscale_models",
    "controlnet",
    "model_patches",
    "hyperflow",
)
TOOLS = (
    "h21_control_probe.py",
    "h21_raw_control_probe.py",
    "observe_h21_owned_events.py",
    "run_progressive_pilot.py",
    "progressive_probe_control.py",
    "vdn_probe_environment.py",
    "progressive_qualification.py",
)
STATIC_SUFFIXES = {".js", ".mjs", ".html", ".css", ".json", ".svg"}
CORE_TREES = ("comfy", "comfy_api", "comfy_extras", "app", "utils")
REPORT_OUTPUTS = {
    "MiniMaxH3SafeAVSaveT8Advanced": 2,
    "MiniMaxH3DirectorControlSourceEXPT8": 3,
    "MiniMaxH3OutputTrimT8": 2,
    "MiniMaxH3AudioWindowT8": 5,
    "MiniMaxH3HyperFlowLoaderT8Advanced": 1,
    "MiniMaxH3HyperFlowPlanT8Advanced": 1,
    "MiniMaxH3HyperFlowSamplerT8Advanced": 3,
    "MiniMaxH3HyperFlowSplitT8Advanced": 1,
    "MiniMaxH3HyperFlowTailPlanT8Advanced": 1,
    "MiniMaxH3HyperFlowHeadPlanT8Advanced": 1,
    "MiniMaxH3HyperFlowCoarseSamplerT8Advanced": 3,
    "MiniMaxH3HyperFlowRefineSamplerT8Advanced": 3,
    "MiniMaxH3HyperFlowPartialRefineSamplerT8Advanced": 3,
    "MiniMaxH3HyperFlowHeadStageEXPT8": 1,
    "MiniMaxH3HyperFlowTailStageEXPT8": 1,
    "MiniMaxH3StageSamplerEXPT8": 3,
}
AUDIT_CLASSES = {
    "MiniMaxH3ControlRegionAuditEXPT8",
    "MiniMaxH3DirectorControlHeadAuditEXPT8",
}


def _raw_api_helper():
    """Load this controller's adjacent frozen helper, never an ambient module."""
    spec = importlib.util.spec_from_file_location(
        "t8_h21_raw_api_helper", Path(__file__).with_name("h21_raw_control_probe.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def output_dependency_closure(graph, outputs):
    """Follow the real validated output roots, not disconnected legacy nodes."""
    pending, selected = list(outputs), set()
    while pending:
        node_id = str(pending.pop())
        if node_id not in graph:
            raise ValueError("Validated output dependency is absent: " + node_id)
        if node_id in selected:
            continue
        selected.add(node_id)
        for value in graph[node_id].get("inputs", {}).values():
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and type(value[1]) is int
                and value[0] in graph
            ):
                pending.append(value[0])
    return selected


def append_report_observers(graph, outputs):
    """Append Core string previews on a NEW graph; never alter a producer input."""
    result = deepcopy(graph)
    active = output_dependency_closure(graph, outputs)
    reports = []
    for node_id, node in graph.items():
        if node_id not in active or node["class_type"] not in REPORT_OUTPUTS:
            continue
        preview = "h21_report_" + node_id
        while preview in result:
            preview += "_"
        output = REPORT_OUTPUTS[node["class_type"]]
        result[preview] = {
            "class_type": "PreviewAny",
            "inputs": {"source": [node_id, output]},
        }
        reports.append(
            {
                "producer": node_id,
                "class_type": node["class_type"],
                "output": output,
                "preview": preview,
            }
        )
    return result, {
        "schema": "t8.h21.control_report_observation.v1",
        "reports": reports,
        "base_output_nodes": list(outputs),
        "base_dependency_closure": sorted(active),
        "sampling_inputs_changed": False,
        "sampling_math_changed": False,
        "NFE_sigmas_noise_audio_routes_changed": False,
        "scope": "Only existing report strings consumed by Core PreviewAny; not a MODEL forward observer",
    }


def collect_case_reports(graph, observation, history):
    reports, invalid = {}, []
    active = set(observation.get("base_dependency_closure", graph))
    selected = [
        {"producer": node_id, "preview": node_id, "class_type": node["class_type"]}
        for node_id, node in graph.items()
        if node_id in active and node["class_type"] in AUDIT_CLASSES
    ] + observation.get("reports", [])
    outputs = (history or {}).get("outputs", {})
    for row in selected:
        producer, preview = row["producer"], row["preview"]
        texts = outputs.get(preview, {}).get("text")
        value = {
            "producer": producer,
            "class_type": graph[producer]["class_type"],
            "history_output": preview,
            "raw_text": texts,
        }
        try:
            if (
                not isinstance(texts, (list, tuple))
                or len(texts) != 1
                or not isinstance(texts[0], str)
            ):
                raise ValueError("Expected one actual JSON report string in history")
            value["report"] = json.loads(texts[0])
        except (TypeError, ValueError) as error:
            value["observation_error"] = str(error)
            invalid.append(producer)
        reports[producer] = value
    return {
        "reports": reports,
        "missing_or_invalid": invalid,
        "media_quality_qualified": False,
        "scope": "Raw runtime reports; no declaration promoted to measured NFE or media acceptance",
    }


def file_identity(path):
    requested = Path(path).absolute()
    target = requested.resolve(strict=True)
    if not target.is_file():
        raise ValueError(f"Expected a regular file: {requested}")
    before = target.stat()
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    after = target.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise RuntimeError(f"File changed during read-only hashing: {requested}")
    return {
        "declared_path": str(requested),
        "path": str(target),
        "bytes": after.st_size,
        "sha256": digest.hexdigest(),
    }


def inside(root, relative):
    parts = PurePosixPath(str(relative).replace("\\", "/"))
    if (
        not parts.parts
        or parts.is_absolute()
        or any(part in {".", ".."} or ":" in part for part in parts.parts)
    ):
        raise ValueError("Expected an ordinary contained relative path")
    target = (Path(root) / parts).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError("Path escaped its private root")
    return target


def copy_exact(source, target):
    before = file_identity(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(before["path"]).open("rb") as incoming, target.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(8 * 1024**2), b""):
            outgoing.write(block)
    after = file_identity(source)
    copied = file_identity(target)
    if before != after or copied["sha256"] != before["sha256"]:
        raise RuntimeError(
            "Source changed while freezing; partial private run retained"
        )
    return before


def runtime_sources(source_root):
    root = Path(source_root).absolute()
    if root.is_symlink():
        raise ValueError("Runtime source root must not be a symlink")
    root = root.resolve(strict=True)
    paths = [root / "__init__.py"]
    for directory in ("h3_t8", "web"):
        tree = root / directory
        if tree.is_symlink():
            raise ValueError("Runtime source directories must not be symlinks")
        for path in tree.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Runtime source symlink is not frozen: {path}")
            if path.is_file() and (
                directory == "h3_t8"
                and path.suffix == ".py"
                or directory == "web"
                and path.suffix in STATIC_SUFFIXES
            ):
                paths.append(path)
    # __path__ is h3_t8 first, then the flat package root. A few real runtime
    # modules (for example Sol) still live only in the latter. Resolve the
    # relative-import closure explicitly instead of copying all legacy files
    # or falling back to the mutable live package at import time.
    selected = set(paths)
    pending = [path for path in selected if path.suffix == ".py"]
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        relative = path.relative_to(root)
        parts = list(relative.parts)
        if parts[0] == "h3_t8":
            parts.pop(0)
        parent = parts[:-1]
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for item in ast.walk(tree):
            if not isinstance(item, ast.ImportFrom) or item.level < 1:
                continue
            anchor = (
                parent[: len(parent) - item.level + 1]
                if item.level <= len(parent) + 1
                else None
            )
            if anchor is None:
                continue
            modules = (
                [item.module.split(".")]
                if item.module
                else [[alias.name] for alias in item.names]
            )
            for module in modules:
                for base in (root / "h3_t8", root):
                    target = base.joinpath(*anchor, *module)
                    candidate = next(
                        (
                            p
                            for p in (target.with_suffix(".py"), target / "__init__.py")
                            if p.is_file()
                        ),
                        None,
                    )
                    if candidate is not None:
                        if candidate not in selected:
                            selected.add(candidate)
                            pending.append(candidate)
                        break
    paths = sorted(selected)
    for path in paths:
        if (
            not path.resolve().is_relative_to(root)
            or not path.is_file()
            or path.is_symlink()
            or any(
                parent.is_symlink()
                for parent in path.parents
                if parent != root and parent.is_relative_to(root)
            )
        ):
            raise ValueError("Runtime source missing or escaped")
    return sorted(set(paths))


def core_identity(core):
    core = Path(core).resolve(strict=True)
    for name in ("main.py", "folder_paths.py", "comfy/ldm/minimax/model.py"):
        if not (core / name).is_file():
            raise ValueError("Select an actual H3 ComfyUI Core source tree")
    paths = [
        core / name
        for name in ("main.py", "execution.py", "server.py", "folder_paths.py")
    ]
    paths += [
        path
        for tree in CORE_TREES
        for path in (core / tree).rglob("*.py")
        if path.is_file()
    ]
    return {
        str(path.relative_to(core)).replace("\\", "/"): file_identity(path)
        for path in sorted(set(paths))
    }


def model_directories(raw):
    if not isinstance(raw, dict):
        raise ValueError(
            "model_directories must explicitly map Core categories to absolute directory lists"
        )
    value = deepcopy(raw)
    if "clip" in value:
        if "text_encoders" in value and value["text_encoders"] != value["clip"]:
            raise ValueError(
                "clip and text_encoders aliases must declare the same ordered roots"
            )
        value["text_encoders"] = value.pop("clip")
    if not set(MODEL_FOLDERS) <= set(value) or set(value) - set(MODEL_FOLDERS) - {
        "semantic_bridge"
    }:
        raise ValueError(
            "Explicit diffusion/text/vae/lora/upscaler/controlnet/model_patches/hyperflow roots are required; unused categories may be []"
        )
    value.setdefault("semantic_bridge", [])
    for roots in value.values():
        if not isinstance(roots, list) or any(
            not isinstance(p, str) or not Path(p).is_absolute() or not Path(p).is_dir()
            for p in roots
        ):
            raise ValueError(
                "Model roots must be existing absolute directories, or an explicit empty list"
            )
    return value


def _pinned_memory_arguments(value):
    """Only an explicit boolean opts a NEW owned Core out of pinning."""
    selected = value.get("disable_pinned_memory", False)
    if type(selected) is not bool:
        raise ValueError("disable_pinned_memory must be an explicit bool")
    return ["--disable-pinned-memory"] if selected else []


def _pinned_memory_flag_count(arguments):
    """Count this long option's argparse prefixes, before the option terminator."""
    count = 0
    for argument in arguments:
        if argument == "--":
            break
        if argument.startswith("--") and "--disable-pinned-memory".startswith(argument):
            count += 1
    return count


def _verify_pinned_memory_binding(plan, root):
    """Legacy absent fields stay absent; explicit choices bind sealed config."""
    key = "disable_pinned_memory"
    config_path = root / "config.json"
    if not config_path.is_file():
        if key in plan:
            raise RuntimeError("disable_pinned_memory plan has no sealed config")
        return
    declared = read_json(config_path)
    if key not in plan and key not in declared:
        return
    if (
        "config.json" not in plan["sealed_files"]
        or (key in plan) != (key in declared)
        or type(plan.get(key)) is not bool
        or type(declared.get(key)) is not bool
        or plan[key] is not declared[key]
    ):
        raise RuntimeError("disable_pinned_memory plan differs from its sealed config")


def server_command(plan, *, cpu=False):
    root = Path(plan["run_root"])
    command = [
        plan["python"],
        "-B",
        "-X",
        "utf8",
        str(root / "runtime_tools/h21_control_probe.py"),
        "--_serve",
        str(root / "manifest.json"),
        "--",
        "--listen",
        "127.0.0.1",
        "--port",
        str(plan["port"]),
        "--disable-auto-launch",
        "--preview-method",
        "none",
        "--cache-none",
        "--disable-comfy-compiler",
        "--base-directory",
        str(root),
        "--models-directory",
        str(root / "models"),
        "--disable-all-custom-nodes",
        "--whitelist-custom-nodes",
        plan["package_name"],
        plan["identity_extension"],
        "--extra-model-paths-config",
        str(root / "paths.json"),
        "--input-directory",
        str(root / "input"),
        "--output-directory",
        str(root / "output"),
        "--temp-directory",
        str(root / "temp"),
        "--user-directory",
        str(root / "user"),
        "--database-url",
        "sqlite:///:memory:",
    ]
    if cpu:
        command.append("--cpu")
    if plan.get("reserve_vram_gb") is not None:
        command.extend(["--reserve-vram", str(plan["reserve_vram_gb"])])
    command.extend(_pinned_memory_arguments(plan))
    return command


def _allocator_telemetry_enabled(value):
    selected = value.get("allocator_telemetry", False)
    if type(selected) is not bool:
        raise ValueError("allocator_telemetry must be an explicit bool")
    return selected


def _verify_allocator_binding(plan, root):
    """Optional controller field, sealed by presence as well as exact bool."""
    _allocator_telemetry_enabled(plan)
    key = "allocator_telemetry"
    if "config.json" not in plan["sealed_files"]:
        if key in plan:
            raise RuntimeError("allocator_telemetry plan has no sealed config")
        return
    config = read_json(root / "config.json")
    _allocator_telemetry_enabled(config)
    if (key in plan) != (key in config) or plan.get(key) != config.get(key):
        raise RuntimeError("allocator_telemetry plan differs from its sealed config")


def _allocator_extension_source():
    """Endpoints exist only in a fresh explicitly enabled owned extension."""
    return """
# Explicit owned allocator telemetry; absent in the default extension bytes.
import os, re, time
_allocator_interval = None
_allocator_used_intervals = set()

def _allocator_bound(payload, finishing=False):
    required={'run_id','manifest_sha256','expected_pid','case_name','prompt_sha256','interval_id'}
    if finishing: required.add('prompt_id')
    if type(payload) is not dict or set(payload)!=required:
        raise ValueError('Exact allocator request fields required')
    manifest_path=ROOT/'manifest.json'
    raw=manifest_path.read_bytes()
    manifest=json.loads(raw)
    if Path(manifest['run_root']).resolve()!=ROOT.resolve() or manifest.get('allocator_telemetry') is not True:
        raise ValueError('Allocator is not enabled for this exact owned root')
    if payload['run_id']!=manifest['run_id'] or payload['manifest_sha256']!=hashlib.sha256(raw).hexdigest():
        raise ValueError('Allocator run or manifest identity differs')
    if type(payload['expected_pid']) is not int or payload['expected_pid']!=os.getpid():
        raise ValueError('Allocator expected process owner differs')
    if Path(sys.executable).resolve()!=Path(manifest['python']).resolve():
        raise ValueError('Allocator Python owner differs')
    actual=Path(__file__).resolve(strict=True)
    expected=(ROOT/'runtime_nodes'/manifest['identity_extension']/'__init__.py').resolve(strict=True)
    if actual!=expected or not actual.is_relative_to(ROOT.resolve()):
        raise ValueError('Allocator extension path owner differs')
    relative=actual.relative_to(ROOT.resolve()).as_posix()
    sealed=manifest['sealed_files'].get(relative)
    if sealed is None or actual.stat().st_size!=sealed['bytes'] or hashlib.sha256(actual.read_bytes()).hexdigest()!=sealed['sha256']:
        raise ValueError('Allocator extension source differs from sealed owner')
    cases=[row for row in manifest['cases'] if row['name']==payload['case_name']]
    if len(cases)!=1:
        raise ValueError('Allocator case is not uniquely prepared')
    relative=cases[0]['prompt']
    prompt=(ROOT/relative).resolve(strict=True)
    if not prompt.is_relative_to(ROOT.resolve()) or prompt.is_symlink():
        raise ValueError('Allocator prompt escapes owned root')
    sealed=manifest['sealed_files'].get(relative)
    raw_prompt=prompt.read_bytes()
    if sealed is None or len(raw_prompt)!=sealed['bytes'] or hashlib.sha256(raw_prompt).hexdigest()!=sealed['sha256'] or payload['prompt_sha256']!=sealed['sha256']:
        raise ValueError('Allocator prepared prompt identity differs')
    if type(payload['interval_id']) is not str or re.fullmatch('[0-9a-f]{32}',payload['interval_id']) is None:
        raise ValueError('Allocator interval requires explicit UUID hex')
    if finishing and (type(payload['prompt_id']) is not str or not payload['prompt_id']):
        raise ValueError('Allocator finish needs the actual submitted prompt ID')
    return manifest,json.loads(raw_prompt)

def _allocator_snapshot(reset):
    value={'available':False,'peaks':None,'device':None,'reset_attempted':False,'reset_completed':False}
    try:
        from comfy.cli_args import args as actual_args
        if actual_args.cpu:
            return dict(value,status='unavailable',reason='actual_CPU_mode')
        import torch
        if not torch.cuda.is_initialized():
            return dict(value,status='unavailable',reason='no_existing_CUDA_context')
        from comfy import model_management as actual_mm
        device=actual_mm.get_torch_device()
        if device.type!='cuda' or type(device.index) is not int or device.index<0:
            return dict(value,status='unavailable',reason='actual_device_is_not_explicit_CUDA')
        index=device.index
        value['device']={'type':'cuda','index':index,'allocator_backend':torch.cuda.get_allocator_backend(),
                         'torch_version':torch.__version__}
        if reset:
            value['reset_attempted']=True
            torch.cuda.reset_peak_memory_stats(index)
            value['reset_completed']=True
        peaks={'allocated_bytes':torch.cuda.memory_allocated(index),'reserved_bytes':torch.cuda.memory_reserved(index),
               'max_allocated_bytes':torch.cuda.max_memory_allocated(index),'max_reserved_bytes':torch.cuda.max_memory_reserved(index)}
        if any(type(item) is not int or item<0 for item in peaks.values()):
            raise ValueError('Allocator statistics are missing, negative or not integer counters')
        if peaks['allocated_bytes']>peaks['max_allocated_bytes'] or peaks['reserved_bytes']>peaks['max_reserved_bytes']:
            raise ValueError('Allocator current counter exceeds recorded peak')
        return dict(value,available=True,peaks=peaks,status='active')
    except Exception as error:
        return dict(value,status='failed',error=type(error).__name__+':'+str(error))

def _allocator_response(payload,state,status):
    value={key:payload[key] for key in ('run_id','manifest_sha256','case_name','prompt_sha256','interval_id')}
    value.update(pid=os.getpid(),status=status,peaks=state.get('peaks'),device=state.get('device'),
                 reset_count=int(state.get('reset_completed',False)),reset_attempted=state.get('reset_attempted',False),
                 reset_effect_uncertain=state.get('reset_attempted',False) and not state.get('reset_completed',False),
                 scope='whole_prompt_not_DiT_or_stage_peak',forced_CUDA_initialization=False,
                 forced_CUDA_synchronization=False,empty_cache_called=False,performance_qualified=False,
                 allocator_scope='actual_backend_counters_not_whole_device_or_tensor_exclusive_guarantee')
    for key in ('reason','error'):
        if key in state:value[key]=state[key]
    return value

def _allocator_reject(error,kind='identity'):
    return web.json_response({'status':'rejected','error_kind':kind,'error':type(error).__name__+':'+str(error),'peaks':None},status=409)

@PromptServer.instance.routes.post('/h21_probe/allocator/begin')
async def allocator_begin(request):
    global _allocator_interval
    try:
        payload=await request.json()
        _allocator_bound(payload)
    except Exception as error:
        return _allocator_reject(error)
    if _allocator_interval is not None or payload['interval_id'] in _allocator_used_intervals:
        return _allocator_reject(ValueError('Allocator interval already active or previously used'),'interval')
    _allocator_used_intervals.add(payload['interval_id'])
    _allocator_interval={'payload':dict(payload),'started_monotonic':time.perf_counter(),'started_unix':time.time()}
    state=_allocator_snapshot(True)
    _allocator_interval['begin']=state
    return web.json_response(_allocator_response(payload,state,state['status']))

@PromptServer.instance.routes.post('/h21_probe/allocator/finish')
async def allocator_finish(request):
    global _allocator_interval
    try:
        payload=await request.json()
        _manifest,graph=_allocator_bound(payload,True)
        if _allocator_interval is None or {key:value for key,value in payload.items() if key!='prompt_id'}!=_allocator_interval['payload']:
            raise ValueError('Allocator finish does not match the active owned interval')
        histories=PromptServer.instance.prompt_queue.get_history(prompt_id=payload['prompt_id'])
        history=histories.get(payload['prompt_id'])
        if type(history) is not dict or history.get('status',{}).get('status_str') not in ('success','error'):
            raise ValueError('Allocator finish requires authoritative terminal history')
        submitted=history.get('prompt')
        if type(submitted) not in (list,tuple) or len(submitted)<3 or submitted[1]!=payload['prompt_id'] or submitted[2]!=graph:
            raise ValueError('Allocator terminal history belongs to a different submitted graph')
    except Exception as error:
        return _allocator_reject(error)
    begun=_allocator_interval['begin']
    state=_allocator_snapshot(False) if begun['available'] else dict(begun)
    state['reset_attempted'],state['reset_completed']=begun['reset_attempted'],begun['reset_completed']
    status='complete' if state['available'] else state['status']
    result=_allocator_response(payload,state,status)
    result.update(prompt_id=payload['prompt_id'],generation_status=history['status']['status_str'],
                  interval_wall_seconds=time.perf_counter()-_allocator_interval['started_monotonic'],
                  interval_started_unix=_allocator_interval['started_unix'],interval_finished_unix=time.time(),
                  begin_peaks=begun['peaks'])
    _allocator_interval=None
    return web.json_response(result)

"""


def _identity_extension(plan):
    """Own endpoint proves loaded graph classes and model resolution before queue."""
    _allocator_telemetry_enabled(plan)
    base = """from aiohttp import web
from pathlib import Path
import hashlib, importlib, json, sys
import nodes
from server import PromptServer
ROOT=Path(__file__).resolve().parents[2]
@PromptServer.instance.routes.get('/h21_probe/identity')
async def identity(request):
    run_id=None; errors=[]; classes={}; loaded_namespace=None
    try:
        manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
        run_id=manifest['run_id']
        package=Path(manifest['package_root']).resolve(strict=True)
        initializer=(package/'__init__.py').resolve(strict=True)
        def digest(path):
            result=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8388608),b''):result.update(chunk)
            return result.hexdigest()
        for relative,frozen in manifest['sealed_files'].items():
            path=ROOT/relative
            if not path.is_file() or digest(path)!=frozen['sha256']:
                errors.append('changed:'+relative)
        for kind in manifest['graph_class_types']:
            cls=nodes.NODE_CLASS_MAPPINGS.get(kind)
            if cls is None:
                errors.append('missing_class:'+kind)
                continue
            module=sys.modules.get(cls.__module__)
            path=Path(module.__file__).resolve() if getattr(module,'__file__',None) else None
            classes[kind]={'module':cls.__module__,'path':str(path)}
            if kind.startswith(('MiniMaxH3','T8')) and (path is None or not path.is_relative_to(package)):
                errors.append('foreign_class:'+kind)
        # Core names a directory-loaded package by its full normalized path.
        # Match the actual frozen initializer, not a guessed import name.
        # Aliases of one module object are legitimate, not multiple owners.
        owners={}
        for module in list(sys.modules.values()):
            source=getattr(module,'__file__',None)
            if source is not None and Path(source).resolve()==initializer:
                owners[id(module)]=module
        if len(owners)!=1:
            errors.append('missing_frozen_runtime_owner' if not owners else 'ambiguous_frozen_runtime_owner')
        else:
            owner=next(iter(owners.values()))
            candidates=[getattr(getattr(owner,'__spec__',None),'name',None),getattr(owner,'__name__',None)]
            loaded_namespace=next((name for name in candidates if isinstance(name,str) and sys.modules.get(name) is owner),None)
            if loaded_namespace is None:
                errors.append('unregistered_frozen_runtime_owner')
            else:
                routes=importlib.import_module(loaded_namespace+'.director_routes')
                route_path=Path(routes.__file__).resolve(strict=True)
                if not route_path.is_relative_to(package):
                    errors.append('foreign_resource_resolver')
                else:
                    resolver=routes._resolve_director_resource
                    for key,frozen in manifest['resources']['models'].items():
                        path=Path(resolver(frozen['folder'],frozen['name'])).resolve(strict=True)
                        if str(path)!=frozen['path'] or path.stat().st_size!=frozen['size'] or digest(path)!=frozen['sha256']:
                            errors.append('changed_model:'+key)
    except Exception as error:
        # Integrity failure is a negative JSON proof, never an HTML HTTP500
        # that hides the actual missing owner/module/file from the controller.
        errors.append('identity_error:'+type(error).__name__+':'+str(error))
    return web.json_response({'run_id':run_id,'ready':not errors,'errors':errors,'classes':classes,'loaded_namespace':loaded_namespace})
NODE_CLASS_MAPPINGS={}
"""
    return base + _allocator_extension_source() if _allocator_telemetry_enabled(plan) else base


def prepare(config, run_root, core_root, *, source_root=None, python=None, port=0):
    """Create a new private snapshot and real CPU graph; never start Core/GPU."""
    config_path = Path(config).resolve(strict=True)
    declared = read_json(config_path)
    _pinned_memory_arguments(declared)  # Validate before socket/source preparation.
    _allocator_telemetry_enabled(declared)
    source = Path(source_root or Path(__file__).resolve().parents[1]).absolute()
    core = Path(core_root).resolve(strict=True)
    root = Path(run_root).absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError(
            "Private run root must be new; existing evidence is never overwritten"
        )
    root = root.resolve()
    for protected in [
        core / name
        for name in ("input", "output", "temp", "user", "models", "custom_nodes")
    ]:
        protected = protected.resolve()
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError(
                "Private probe root must not be inside or replace a user Core data directory"
            )
    if type(port) is not int or not 0 <= port <= 65535 or port == 8189:
        raise ValueError("Select a separate localhost port; 8189 is never used")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
        port = probe.getsockname()[1]
    directories = model_directories(declared.get("model_directories"))
    reserve = declared.get("reserve_vram_gb")
    if reserve is not None and (
        type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve < 0
    ):
        raise ValueError(
            "reserve_vram_gb must be an explicit finite nonnegative number"
        )
    raw_api = declared.get("mode", "director") == "raw_api"
    if declared.get("mode", "director") not in {"director", "raw_api"}:
        raise ValueError("Select explicit director or raw_api graph origin")
    if raw_api:
        _raw_api_helper().validate_config(declared)
        project_file, project = None, None
        cases = declared["raw_api_cases"]
    else:
        project_file = Path(declared["project"]).resolve(strict=True)
        project = read_json(project_file)
        cases = declared.get("cases", [{"name": "current"}])
    names = [case.get("name") for case in cases] if isinstance(cases, list) else []
    if (
        not names
        or len(set(names)) != len(names)
        or any(
            not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name)
            for name in names
        )
    ):
        raise ValueError(
            "Cases require unique contained names; sampling overrides must be explicit"
        )
    if not raw_api and any(set(case) - {"name", "sampling", "controlRegions"} for case in cases):
        raise ValueError(
            "Unknown case override; only name, sampling and controlRegions are applied"
        )
    generation = {} if raw_api else project.get("doc", {}).get("generation", {})
    if not raw_api and any(
        not isinstance(generation.get(key), str)
        or generation[key] in {"", "auto", "none"}
        for key in ("unet", "clip", "video_vae", "audio_vae")
    ):
        raise ValueError(
            "Qualification requires explicit existing model selections; no auto selection is invented"
        )
    media = declared.get("assets", {})
    ids = set(media) if raw_api else {a["id"] for a in project.get("assets", [])}
    if not isinstance(media, dict) or set(media) != ids:
        raise ValueError(
            "Explicit asset UUID to source-file mapping must exactly cover the supplied owned project assets"
        )
    sources = runtime_sources(source)
    for name in TOOLS:
        path = source / "tools" / name
        if (
            path.is_symlink()
            or path.parent.is_symlink()
            or not path.resolve().is_relative_to(source.resolve())
        ):
            raise ValueError(
                "Frozen controller helpers must be ordinary contained source files"
            )
    provenance = core_identity(core)
    root.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    package_name, extension = "t8_h21_" + run_id, "t8_h21_identity_" + run_id
    package = root / "runtime_nodes" / package_name
    for name in (
        "input",
        "output",
        "temp",
        "user",
        "models",
        "runtime_tools",
        "runtime_nodes",
    ):
        (root / name).mkdir()
    snapshots = {}
    for path in sources:
        relative = path.relative_to(source.resolve()).as_posix()
        snapshots[(Path("runtime_nodes") / package_name / relative).as_posix()] = (
            copy_exact(path, package / relative)
        )
    for name in TOOLS:
        snapshots["runtime_tools/" + name] = copy_exact(
            source / "tools" / name, root / "runtime_tools" / name
        )
    for identity in snapshots.values():
        if file_identity(identity["declared_path"]) != identity:
            raise RuntimeError(
                "Runtime source changed across the freeze; partial private run retained"
            )
    write_json(root / "config.json", declared)
    if not raw_api:
        write_json(root / "source-project.json", project)
    plan = {
        "schema": SCHEMA,
        "run_id": run_id,
        "run_root": str(root),
        "core_root": str(core),
        "package_name": package_name,
        "package_root": str(package),
        "identity_extension": extension,
        "python": str(Path(python or sys.executable).resolve(strict=True)),
        "python_identity": file_identity(python or sys.executable),
        "port": port,
        "reserve_vram_gb": reserve,
        "config_identity": file_identity(config_path),
        "project_identity": None if raw_api else file_identity(project_file),
        "graph_origin": "explicit_raw_api" if raw_api else "director_compiler",
        "model_directories": directories,
        "model_roots_real": {
            key: [str(Path(p).resolve()) for p in roots]
            for key, roots in directories.items()
        },
        "core_files": provenance,
        "source_files": snapshots,
        "runtime_allowlist": [
            path.relative_to(source.resolve()).as_posix() for path in sources
        ],
        "stage": "prepare_cpu_only",
        "sampling_formula_changed": False,
        "gpu_started": False,
    }
    if "disable_pinned_memory" in declared:
        plan["disable_pinned_memory"] = declared["disable_pinned_memory"]
    if "allocator_telemetry" in declared:
        plan["allocator_telemetry"] = declared["allocator_telemetry"]
    extra = {
        "h21_models": {
            "is_default": True,
            **{key: "\n".join(roots) for key, roots in directories.items()},
        },
        "h21_nodes": {"custom_nodes": str(root / "runtime_nodes")},
    }
    write_json(root / "paths.json", extra)
    extension_file = root / "runtime_nodes" / extension / "__init__.py"
    extension_file.parent.mkdir()
    with extension_file.open("x", encoding="utf-8") as stream:
        stream.write(_identity_extension(plan))
    write_json(root / "prepare-plan.json", plan)
    env = _environment(plan, cpu=True)
    command = [
        plan["python"],
        "-B",
        "-X",
        "utf8",
        str(root / "runtime_tools/h21_control_probe.py"),
        "--_build",
        str(root / "prepare-plan.json"),
    ]
    with (
        (root / "prepare.stdout.log").open("x", encoding="utf-8") as out,
        (root / "prepare.stderr.log").open("x", encoding="utf-8") as error,
    ):
        result = subprocess.run(
            command, cwd=root, env=env, stdout=out, stderr=error, check=False
        )
    if result.returncode:
        raise RuntimeError(
            f"CPU graph preparation failed ({result.returncode}); private evidence retained at {root}"
        )
    built = read_json(root / "build-result.json")
    plan.update(built)
    plan["stage"] = "prepared_cpu_graphs_only"
    plan["server_argv"] = server_command(plan)
    plan["sealed_files"] = {
        path.relative_to(root).as_posix(): file_identity(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.endswith(".log")
    }
    write_json(root / "manifest.json", plan)
    verify_prepared(root / "manifest.json")
    return plan


def _environment(plan, *, cpu):
    env = os.environ.copy()
    env.update(
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2",
        PYTHONPATH=plan["core_root"],
    )
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def _configure(plan, *, cpu):
    root, core = Path(plan["run_root"]), Path(plan["core_root"])
    sys.path.insert(0, str(core))
    sys.argv = [
        str(core / "main.py"),
        "--base-directory",
        str(root),
        "--models-directory",
        str(root / "models"),
        "--disable-comfy-compiler",
        *(["--cpu"] if cpu else []),
        *_pinned_memory_arguments(plan),
    ]
    import comfy.options

    comfy.options.enable_args_parsing()
    import comfy.cli_args

    comfy.cli_args.args.cpu = cpu
    comfy.cli_args.args.disable_comfy_compiler = True
    import folder_paths

    empty_categories = {
        key: ([], extensions)
        for key, (_paths, extensions) in folder_paths.folder_names_and_paths.items()
    }
    folder_paths.folder_names_and_paths.clear()
    folder_paths.folder_names_and_paths.update(empty_categories)
    for key, roots in plan["model_directories"].items():
        folder_paths.folder_names_and_paths[key] = (
            list(roots),
            folder_paths.supported_pt_extensions,
        )
    folder_paths.folder_names_and_paths["custom_nodes"] = (
        [str(root / "runtime_nodes")],
        set(),
    )
    folder_paths.set_input_directory(str(root / "input"))
    folder_paths.set_output_directory(str(root / "output"))
    folder_paths.set_temp_directory(str(root / "temp"))
    folder_paths.set_user_directory(str(root / "user"))
    import nodes  # Resolve real Core nodes before any extension namespace.

    return folder_paths, nodes


def _build(plan_path):
    plan = read_json(plan_path)
    root = Path(plan["run_root"])
    folders, core_nodes = _configure(plan, cpu=True)
    name = plan["package_name"]
    spec = importlib.util.spec_from_file_location(
        name,
        Path(plan["package_root"]) / "__init__.py",
        submodule_search_locations=[plan["package_root"]],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    for cls in asyncio.run(package.comfy_entrypoint().get_node_list()):
        core_nodes.NODE_CLASS_MAPPINGS[cls.__name__] = cls
    from comfy_extras import nodes_custom_sampler
    from comfy_extras.nodes_video import LoadVideo, GetVideoComponents
    from comfy_extras.nodes_preview_any import PreviewAny
    from comfy_extras.nodes_audio import LoadAudio

    for cls in (
        nodes_custom_sampler.BasicGuider,
        nodes_custom_sampler.RandomNoise,
        nodes_custom_sampler.SamplerCustomAdvanced,
        LoadVideo,
        GetVideoComponents,
        PreviewAny,
        LoadAudio,
    ):
        core_nodes.NODE_CLASS_MAPPINGS[cls.__name__] = cls
    import execution

    generation = importlib.import_module(name + ".director_generation")
    project_api = importlib.import_module(name + ".director_project")
    batch = importlib.import_module(name + ".director_batch")
    resolver = importlib.import_module(
        name + ".director_routes"
    )._resolve_director_resource
    if plan.get("graph_origin") == "explicit_raw_api":
        return _raw_api_helper().build(
            plan, folders, core_nodes, execution, resolver, batch, api=globals()
        )
    config, project = (
        read_json(root / "config.json"),
        read_json(root / "source-project.json"),
    )
    store = project_api.ProjectStore(root / "user", root / "input")
    assets = []
    for declared in project["assets"]:
        asset_id = project_api.identity(declared["id"])
        source = Path(config["assets"][declared["id"]]).resolve(strict=True)
        target = inside(
            root / "input",
            "t8_director/" + asset_id + "/source" + source.suffix.lower(),
        )
        source_id = copy_exact(source, target)
        actual = store.register_asset(
            target,
            declared["id"],
            declared["name"],
            source_frame=declared.get("source_frame"),
        )
        if (
            source_id["sha256"] != declared["sha256"]
            or actual["size"] != declared["size"]
        ):
            raise ValueError(
                "Owned input bytes differ from the supplied project identity"
            )
        assets.append(actual)
    project["assets"] = assets
    for relative, source in config.get("output_files", {}).items():
        if (
            not str(relative)
            .replace("\\", "/")
            .startswith(
                ("MiniMaxH3/hyperflow_stage_artifacts/", "MiniMaxH3/stage_artifacts/")
            )
        ):
            raise ValueError(
                "Only explicitly selected owned stage archive files may seed private output"
            )
        copy_exact(source, inside(root / "output", relative))
    shot_id = config.get("shot_id", project["current"])
    seed = config.get(
        "seed",
        next(s for s in project["doc"]["shots"] if s["id"] == shot_id).get(
            "seed", 26091901
        ),
    )
    plans, built_plans, class_types = [], [], set()
    for case in config.get("cases", [{"name": "current"}]):
        candidate = deepcopy(project)
        if "sampling" in case:
            current = next(s for s in candidate["doc"]["shots"] if s["id"] == shot_id)
            current["samplingInherit"] = False
            current["sampling"] = deepcopy(case["sampling"])
        if "controlRegions" in case:
            current = next(s for s in candidate["doc"]["shots"] if s["id"] == shot_id)
            current["controlRegions"] = deepcopy(case["controlRegions"])
        built = generation.build_director_generation_prompt(
            candidate, shot_id, store, seed=seed
        )
        for node in built["prompt"].values():
            if node["class_type"] == "MiniMaxH3SafeAVSaveT8Advanced":
                node["inputs"]["filename_prefix"] = (
                    "H21_Control/" + plan["run_id"] + "/" + case["name"]
                )
        relative = "cases/" + case["name"]
        base_graph = deepcopy(built["prompt"])
        base_validation = asyncio.run(
            execution.validate_prompt(str(uuid.uuid4()), base_graph, None)
        )
        if base_validation[0] is not True or base_validation[3]:
            write_json(
                root / relative / "base-core-validation.json",
                {
                    "valid": base_validation[0],
                    "error": base_validation[1],
                    "outputs": list(base_validation[2]),
                    "node_errors": base_validation[3],
                },
            )
            raise RuntimeError(
                "Real CPU Core rejected the uninstrumented base case " + case["name"]
            )
        built["prompt"], observation = append_report_observers(
            base_graph, base_validation[2]
        )
        built["observation"] = observation
        write_json(root / relative / "project.json", candidate)
        write_json(root / relative / "base-prompt.json", base_graph)
        write_json(
            root / relative / "base-core-validation.json",
            {
                "valid": base_validation[0],
                "error": base_validation[1],
                "outputs": list(base_validation[2]),
                "node_errors": base_validation[3],
            },
        )
        write_json(root / relative / "built.json", built)
        write_json(root / relative / "prompt.json", built["prompt"])
        validation = asyncio.run(
            execution.validate_prompt(str(uuid.uuid4()), built["prompt"], None)
        )
        write_json(
            root / relative / "core-validation.json",
            {
                "valid": validation[0],
                "error": validation[1],
                "outputs": list(validation[2]),
                "node_errors": validation[3],
            },
        )
        if validation[0] is not True or validation[3]:
            raise RuntimeError(
                "Real CPU Core rejected case "
                + case["name"]
                + "; node errors retained without fallback"
            )
        class_types.update(node["class_type"] for node in built["prompt"].values())
        built_plans.append(built)
        plans.append(
            {
                "name": case["name"],
                "prompt": relative + "/prompt.json",
                "base_prompt": relative + "/base-prompt.json",
                "project": relative + "/project.json",
                "built": relative + "/built.json",
                "recipe": built["recipe"],
                "sampling": built["sampling"],
                "validation": relative + "/core-validation.json",
                "output_nodes": list(validation[2]),
                "observation": observation,
            }
        )
    # The real batch API hashes each selected file once across all actual graphs.
    # A 22-case matrix must not read the same multi-GB checkpoint 22 times.
    # Final prepared/execute verification still checks the captured bytes.
    resources = batch.capture_resources(store, project, built_plans, resolver)
    imports = {}
    for key, module in list(sys.modules.items()):
        if not key.startswith(name + ".") or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(Path(plan["package_root"])):
            raise RuntimeError(
                "CPU build imported live extension source instead of frozen runtime"
            )
        imports[key] = file_identity(path)
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation unexpectedly initialized CUDA")
    for relative, identity in plan["source_files"].items():
        if file_identity(root / relative)["sha256"] != identity["sha256"]:
            raise RuntimeError("Frozen runtime changed during CPU preparation")
    write_json(
        root / "build-result.json",
        {
            "cases": plans,
            "resources": resources,
            "graph_class_types": sorted(class_types),
            "builder_imports": imports,
            "effective_model_directories": {
                key: {
                    "paths": list(paths),
                    "extensions": sorted(extensions),
                }
                for key, (paths, extensions) in folders.folder_names_and_paths.items()
            },
            "model_execution": False,
            "core_validate_prompt": True,
            "cuda_initialized": False,
            "dependency_scope": "current host packages; frozen extension source, not a historical environment pin",
        },
    )


def verify_prepared(manifest):
    manifest_path = Path(manifest).resolve(strict=True)
    plan = read_json(manifest_path)
    root = Path(plan["run_root"]).resolve()
    if plan.get("schema") != SCHEMA or manifest_path != root / "manifest.json":
        raise ValueError("Not a contained prepared H21 control manifest")
    for relative, identity in plan["sealed_files"].items():
        path = inside(root, relative)
        if file_identity(path)["sha256"] != identity["sha256"]:
            raise RuntimeError("Prepared file changed: " + relative)
    _verify_pinned_memory_binding(plan, root)
    _verify_allocator_binding(plan, root)
    for relative, identity in plan["core_files"].items():
        if file_identity(Path(plan["core_root"]) / relative) != identity:
            raise RuntimeError("Selected Core source changed: " + relative)
    for identity in plan["resources"]["models"].values():
        current = file_identity(identity["path"])
        if (
            current["bytes"] != identity["size"]
            or current["sha256"] != identity["sha256"]
        ):
            raise RuntimeError("Selected model bytes changed: " + identity["name"])
    if file_identity(plan["python"]) != plan["python_identity"]:
        raise RuntimeError("Selected Python executable changed")
    actual = {
        path.relative_to(root).as_posix()
        for directory in ("runtime_nodes", "runtime_tools")
        for path in (root / directory).rglob("*")
        if path.is_file()
    }
    expected = {
        path
        for path in plan["sealed_files"]
        if path.startswith(("runtime_nodes/", "runtime_tools/"))
    }
    if actual != expected:
        raise RuntimeError("Frozen runtime gained or lost a file")
    if server_command(plan) != plan["server_argv"] or plan["port"] == 8189:
        raise RuntimeError("Owned server command no longer matches preparation")
    return plan


def _serve(manifest, arguments):
    plan = verify_prepared(manifest)
    if _pinned_memory_flag_count(arguments) != len(_pinned_memory_arguments(plan)):
        raise RuntimeError("Actual owned Core disable_pinned_memory flag count differs from preparation")
    core = Path(plan["core_root"])
    sys.path.insert(0, str(core))
    sys.argv = [str(core / "main.py"), *arguments]
    import comfy.options

    comfy.options.enable_args_parsing()
    import utils.extra_config

    original = utils.extra_config.load_extra_path_config

    def owned_paths(path):
        if Path(path).resolve() != Path(plan["run_root"]) / "paths.json":
            print(
                "H21 isolated bootstrap ignores ambient Core extra-model configuration:",
                path,
                flush=True,
            )
            return
        import folder_paths

        empty_categories = {
            key: ([], extensions)
            for key, (_paths, extensions) in folder_paths.folder_names_and_paths.items()
        }
        folder_paths.folder_names_and_paths.clear()
        folder_paths.folder_names_and_paths.update(empty_categories)
        return original(path)

    utils.extra_config.load_extra_path_config = owned_paths
    runpy.run_path(str(core / "main.py"), run_name="__main__")


def select_cases(plan, names=None, *, all_cases=False):
    if all_cases:
        if names:
            raise ValueError("Choose selected cases or all cases, never both")
        return list(plan["cases"])
    if (
        not names
        or len(set(names)) != len(names)
        or any(name not in {case["name"] for case in plan["cases"]} for name in names)
    ):
        raise ValueError(
            "Explicit --select-case NAME is required; --all-cases is a separate deliberate opt-in"
        )
    return [case for case in plan["cases"] if case["name"] in names]


def _allocator_request(server, action, payload):
    """One observation attempt, never a queue retry or a foreign-owner bypass."""
    server.assert_port_owner()  # Ownership failures remain fatal, not telemetry warnings.
    try:
        value = server.request("POST", "/h21_probe/allocator/" + action, json=payload)
    except Exception as error:
        response = getattr(error, "response", None)
        try:
            rejected = response.json() if response is not None else None
        except Exception:
            rejected = None
        if isinstance(rejected, dict) and rejected.get("error_kind") == "identity":
            raise RuntimeError("Allocator actual owner/request rejected: " + str(rejected.get("error"))) from error
        return {"status": "failed", "peaks": None, "error": type(error).__name__ + ": " + str(error),
                "observation_only": True, "retry_performed": False}
    if not isinstance(value, dict):
        return {"status": "failed", "peaks": None, "raw_response": value,
                "error": "Allocator endpoint returned non-object data", "observation_only": True}
    if value.get("status") == "rejected":
        if value.get("error_kind") == "identity":
            raise RuntimeError("Allocator actual owner/request rejected: " + str(value.get("error")))
        return {"status": "failed", "peaks": None, "raw_response": value, "observation_only": True}
    for key, expected in {**{key: payload[key] for key in (
            "run_id", "manifest_sha256", "case_name", "prompt_sha256", "interval_id")},
            "pid": payload["expected_pid"]}.items():
        if key in value and (type(value[key]) is not type(expected) or value[key] != expected):
            raise RuntimeError("Allocator response belongs to a different actual owner/request: " + key)
        if key not in value:
            return {"status": "failed", "peaks": None, "raw_response": value,
                    "error": "Allocator endpoint omitted identity: " + key, "observation_only": True}
    if value.get("status") in ("active", "complete"):
        peaks = value.get("peaks")
        keys = {"allocated_bytes", "reserved_bytes", "max_allocated_bytes", "max_reserved_bytes"}
        if (type(peaks) is not dict or set(peaks) != keys
                or any(type(item) is not int or item < 0 for item in peaks.values())
                or type(value.get("reset_count")) is not int or value["reset_count"] != 1):
            return {"status": "failed", "peaks": None, "raw_response": value,
                    "error": "Allocator counters/reset are unavailable or invalid", "observation_only": True}
    return value


def _allocator_event_intervals(events, prompt_id, sampler_nodes):
    """Observer receive clocks, not CUDA events or DiT work/NFE estimates."""
    starts, terminals, progress = [], [], {}
    for row in events:
        data = row.get("data")
        stamp = row.get("received_at")
        if (not isinstance(data, dict) or data.get("prompt_id") != prompt_id
                or type(stamp) not in (int, float) or not math.isfinite(stamp)):
            continue
        if row.get("type") == "execution_start":
            starts.append(stamp)
        elif row.get("type") in ("execution_success", "execution_error", "execution_interrupted"):
            terminals.append(stamp)
        elif row.get("type") == "progress" and str(data.get("node")) in sampler_nodes:
            progress.setdefault(str(data["node"]), []).append({
                "received_at": stamp, "value": data.get("value"), "max": data.get("max")})
    return {"prompt_event_wall_seconds": terminals[0] - starts[0]
            if len(starts) == len(terminals) == 1 and terminals[0] >= starts[0] else None,
            "sampler_callback_intervals": {key: {"first_received_at": rows[0]["received_at"],
                "last_received_at": rows[-1]["received_at"],
                "first_to_last_seconds": rows[-1]["received_at"] - rows[0]["received_at"],
                "callbacks": rows} for key, rows in progress.items()},
            "clock_scope": "WS receive wall clocks; first-to-last callback omits initial work and is not full sampler/GPU duration",
            "NFE_or_full_DiT_calls_inferred": False}


def execute_observed_case(server, plan, manifest, case, folder, *, timeout=1800):
    """Exactly one POST after an owned ACK; loss of telemetry never retries work."""
    # This helper is frozen beside the controller for future prepares. Existing
    # immutable r1/r2 snapshots retain their original tool and graph bytes.
    from observe_h21_owned_events import OwnedEventCapture
    from requests.exceptions import ReadTimeout, ConnectTimeout, ConnectionError

    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Case observation timeout must be positive and finite")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    graph = read_json(Path(plan["run_root"]) / case["prompt"])
    validation = read_json(Path(plan["run_root"]) / case["validation"])
    if validation.get("valid") is not True or validation.get("node_errors"):
        raise RuntimeError("Case lacks a successful actual Core CPU graph validation")
    required = output_dependency_closure(graph, validation["outputs"])
    sampler_nodes = {
        node_id: node["class_type"]
        for node_id, node in graph.items()
        if node_id in required
        and (
            node["class_type"]
            in {"SamplerCustomAdvanced", "MiniMaxH3StageSamplerEXPT8"}
            or node["class_type"].startswith("MiniMaxH3HyperFlow")
            and node["class_type"]
            in {
                "MiniMaxH3HyperFlowSplitT8Advanced",
                "MiniMaxH3HyperFlowHeadStageEXPT8",
                "MiniMaxH3HyperFlowTailStageEXPT8",
            }
        )
    }
    write_json(
        folder / "case.json",
        {
            "run_id": plan["run_id"],
            "case": case["name"],
            "manifest_sha256": file_identity(manifest)["sha256"],
            "prompt_identity": file_identity(Path(plan["run_root"]) / case["prompt"]),
            "validated_outputs": validation["outputs"],
            "dependency_closure": sorted(required),
            "sampler_nodes": sampler_nodes,
            "sampling_inputs_changed": False,
            "sampling_math_changed": False,
        },
    )
    client = "h21-" + plan["run_id"] + "-" + uuid.uuid4().hex
    capture = OwnedEventCapture(
        manifest,
        plan,
        folder / "observation",
        client_id=client,
        expected_pid=server.process.pid,
    )
    history, prompt_id, error = None, None, None
    allocator = None
    history_transport_errors = 0
    try:
        server.assert_port_owner()
        capture.connect()
        if _allocator_telemetry_enabled(plan):
            root = Path(plan["run_root"])
            prompt_path = inside(root, case["prompt"])
            frozen = plan["sealed_files"].get(case["prompt"])
            actual_prompt = file_identity(prompt_path)
            if frozen is None or actual_prompt["sha256"] != frozen["sha256"]:
                raise RuntimeError("Allocator request lacks the actual sealed case prompt")
            payload = {"run_id": plan["run_id"], "manifest_sha256": file_identity(manifest)["sha256"],
                       "expected_pid": server.process.pid, "case_name": case["name"],
                       "prompt_sha256": frozen["sha256"], "interval_id": uuid.uuid4().hex}
            allocator = {"schema": "t8.h21.owned_allocator_observation.v1", "request": payload,
                         "begin": _allocator_request(server, "begin", payload), "finish": None}
            write_json(folder / "allocator-begin.json", allocator["begin"])
        capture.mark_post()
        submitted = server.request(
            "POST", "/prompt", json={"prompt": graph, "client_id": client}
        )
        write_json(folder / "submitted.json", submitted)
        if not submitted.get("prompt_id") or submitted.get("node_errors"):
            raise RuntimeError(
                "Real Core rejected the frozen graph; no automatic recipe fallback"
            )
        prompt_id = submitted["prompt_id"]
        deadline, next_poll, history_seen_at = time.monotonic() + timeout, 0.0, None
        while time.monotonic() < deadline:
            if server.process.poll() is not None:
                raise RuntimeError("Owned Core exited during execution")
            capture.receive()
            now = time.monotonic()
            if now >= next_poll:
                next_poll = now + 1.0
                try:
                    result = server.request("GET", "/history/" + prompt_id)
                except (ReadTimeout, ConnectTimeout, ConnectionError) as transport_error:
                    # Only the read-only history GET may be repeated. The
                    # original POST, WS connection and case deadline stay fixed.
                    history_transport_errors += 1
                    failure = {"prompt_id": prompt_id, "method": "GET",
                        "endpoint": "/history/" + prompt_id,
                        "error_type": type(transport_error).__name__,
                        "error": str(transport_error), "observed_at": time.time(),
                        "remaining_case_seconds": max(0.0, deadline - time.monotonic()),
                        "owned_process_and_port_revalidated": False,
                        "sampling_resubmitted": False, "WS_reconnected": False}
                    try:
                        if server.process.poll() is not None:
                            raise RuntimeError("Owned Core exited after history transport failure") from transport_error
                        server.assert_port_owner()
                        failure["owned_process_and_port_revalidated"] = True
                    finally:
                        write_json(folder / "history-poll-transport" /
                            (str(history_transport_errors).zfill(6) + ".json"), failure)
                    # Bound immediate ConnectionError retries too; never spin
                    # or wait past the original per-case deadline on purpose.
                    time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
                else:
                    if prompt_id in result:
                        history = result[prompt_id]
                        if history_seen_at is None:
                            history_seen_at = now
                            write_json(folder / "history.json", history)
            if history is not None and (
                capture.terminal_for(prompt_id)
                or capture.sock is None
                or time.monotonic() - history_seen_at >= 5.0
            ):
                break
            if capture.sock is None:
                # A closed observer is not a reason to reconnect/requeue or to
                # busy-spin history. Preserve a successful authoritative run.
                time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        if history is None:
            raise TimeoutError(
                "Owned graph has no authoritative history before timeout; no resubmission"
            )
        if (allocator is not None and "interval_id" in allocator["begin"]
                and allocator["begin"].get("status") in ("active", "unavailable", "failed")):
            allocator["finish"] = _allocator_request(
                server, "finish", {**allocator["request"], "prompt_id": prompt_id})
            write_json(folder / "allocator-finish.json", allocator["finish"])
        reports = collect_case_reports(graph, case.get("observation", {}), history)
        write_json(folder / "reports.json", reports)
        if history.get("status", {}).get("status_str") != "success":
            raise RuntimeError(
                "Real Core execution failed; original history and raw events retained"
            )
        return history
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
        raise
    finally:
        capture.finish(
            prompt_id, required, history, error=error, sampler_nodes=sampler_nodes
        )
        if allocator is not None:
            finished = allocator.get("finish") or {}
            allocator.update(status="complete" if finished.get("status") == "complete"
                else "unavailable" if allocator["begin"].get("status") == "unavailable" else "failed_or_incomplete",
                prompt_id=prompt_id, actual_generation_status=(history or {}).get("status"),
                source_prompt_unchanged=True, telemetry_error_does_not_retry_prompt=True,
                scope="whole_prompt_not_DiT_or_stage_peak", performance_qualified=False,
                external_device_memory_telemetry="not_collected_by_this_endpoint",
                **_allocator_event_intervals(getattr(capture, "events", []), prompt_id, sampler_nodes))
            write_json(folder / "allocator-telemetry.json", allocator)


def _execute(manifest, timeout=1800, *, names=None, all_cases=False):
    plan = verify_prepared(manifest)
    root = Path(plan["run_root"])
    cases = select_cases(plan, names, all_cases=all_cases)
    if type(timeout) is not int or timeout <= 0:
        raise ValueError(
            "Execution timeout must be a positive integer number of seconds"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise RuntimeError(
            "Explicit GPU execution inherited a CPU-only CUDA mask; no server was started"
        )
    attempt = root / "attempts" / uuid.uuid4().hex
    attempt.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "runtime_tools"))
    import run_progressive_pilot as pilot
    from progressive_probe_control import SerialProbeLease

    class H21OwnedServer(pilot.OwnedServer):
        def request(self, method, endpoint, **kwargs):
            if endpoint != "/h21_probe/identity":
                return super().request(method, endpoint, **kwargs)
            # The endpoint hashes actual multi-GB selected weights. The old
            # pilot's 2-second polling timeout is not an identity-scan timeout.
            import requests

            self.assert_port_owner()
            with requests.Session() as session:
                session.trust_env = False
                response = session.request(
                    method, self.url + endpoint, timeout=(3, timeout), **kwargs
                )
                response.raise_for_status()
                return response.json()

        def start(self):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", self.port))
            self.logs = [
                (attempt / f"server.{name}.log").open("x", encoding="utf-8")
                for name in ("stdout", "stderr")
            ]
            flags = (
                subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0
            )
            self.process = subprocess.Popen(
                plan["server_argv"],
                cwd=root,
                env=_environment(plan, cpu=False),
                stdout=self.logs[0],
                stderr=self.logs[1],
                creationflags=flags,
            )

    with SerialProbeLease(root.parent / "h21-control-gpu.serial.lock"):
        write_json(
            attempt / "execute-started.json",
            {
                "explicit_execute": True,
                "run_id": plan["run_id"],
                "started_at": time.time(),
                "cases": [case["name"] for case in cases],
                "case_permission": {
                    "scope": "all" if all_cases else "selected",
                    "selected_names": [case["name"] for case in cases],
                },
                "timeout_seconds": timeout,
            },
        )
        server = H21OwnedServer(root, plan["port"], False)
        try:
            server.start()
            pilot.wait_ready(server, lambda: None, seconds=240)
            live = server.request("GET", "/h21_probe/identity")
            write_json(attempt / "live-runtime.json", live)
            if live.get("run_id") != plan["run_id"] or live.get("ready") is not True:
                raise RuntimeError(
                    "Owned Core did not prove the frozen runtime/model identity"
                )
            for case in cases:
                verify_prepared(manifest)
                folder = attempt / "executions" / case["name"]
                execute_observed_case(
                    server, plan, manifest, case, folder, timeout=timeout
                )
            verified = verify_prepared(manifest)
            write_json(
                attempt / "verified-after-execution.json",
                {
                    "run_id": verified["run_id"],
                    "cases": [case["name"] for case in cases],
                    "frozen_runtime_core_models_unchanged": True,
                    "media_quality_qualified": False,
                },
            )
        finally:
            server.stop()
            write_json(attempt / "owned-stop.json", server.stop_receipt)
            print(
                json.dumps(
                    {
                        "attempt": str(attempt),
                        "cases": [case["name"] for case in cases],
                        "owned_stop": server.stop_receipt,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--core", type=Path)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicit alias of default CPU prepare; starts no Core/GPU",
    )
    parser.add_argument(
        "--execute",
        type=Path,
        metavar="MANIFEST",
        help="Explicitly start the prepared, isolated GPU Core",
    )
    parser.add_argument("--_build", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_serve", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_execute", type=Path, help=argparse.SUPPRESS)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--select-case", action="append", metavar="NAME")
    selection.add_argument("--all-cases", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    options, rest = parser.parse_known_args(argv)
    if options._build:
        _build(options._build)
        return 0
    if options._serve:
        _serve(options._serve, rest[1:] if rest[:1] == ["--"] else rest)
        return 0
    if options._execute:
        _execute(
            options._execute,
            timeout=options.timeout_seconds,
            names=options.select_case,
            all_cases=options.all_cases,
        )
        return 0
    if rest:
        parser.error("Unrecognized arguments: " + " ".join(rest))
    if options.execute:
        if options.dry_run or options.config or options.run_root:
            parser.error("--execute cannot be combined with prepare options")
        plan = verify_prepared(options.execute)
        select_cases(plan, options.select_case, all_cases=options.all_cases)
        if options.timeout_seconds <= 0:
            parser.error("--timeout-seconds must be positive")
        return subprocess.call(
            [
                plan["python"],
                "-B",
                "-X",
                "utf8",
                str(Path(plan["run_root"]) / "runtime_tools/h21_control_probe.py"),
                "--_execute",
                str(options.execute.resolve()),
                "--timeout-seconds",
                str(options.timeout_seconds),
                *(
                    ["--all-cases"]
                    if options.all_cases
                    else [
                        arg
                        for name in options.select_case
                        for arg in ("--select-case", name)
                    ]
                ),
            ],
            cwd=plan["run_root"],
            env=_environment(plan, cpu=False),
        )
    if not all((options.config, options.run_root, options.core)):
        parser.error(
            "Default prepare requires --config, --run-root and --core; no server was started"
        )
    if options.select_case or options.all_cases:
        parser.error("Case execution selection is only valid with --execute")
    plan = prepare(
        options.config,
        options.run_root,
        options.core,
        source_root=options.source_root,
        python=options.python,
        port=options.port,
    )
    print(
        json.dumps(
            {
                "manifest": str(Path(plan["run_root"]) / "manifest.json"),
                "stage": plan["stage"],
                "gpu_started": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
