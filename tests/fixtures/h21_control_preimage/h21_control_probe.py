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
    "run_progressive_pilot.py",
    "progressive_probe_control.py",
    "vdn_probe_environment.py",
    "progressive_qualification.py",
)
STATIC_SUFFIXES = {".js", ".mjs", ".html", ".css", ".json", ".svg"}
CORE_TREES = ("comfy", "comfy_api", "comfy_extras", "app", "utils")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


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
    return command


def _identity_extension(plan):
    """Own endpoint proves loaded graph classes and model resolution before queue."""
    return """from aiohttp import web
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


def prepare(config, run_root, core_root, *, source_root=None, python=None, port=0):
    """Create a new private snapshot and real CPU graph; never start Core/GPU."""
    config_path = Path(config).resolve(strict=True)
    declared = read_json(config_path)
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
    if any(set(case) - {"name", "sampling", "controlRegions"} for case in cases):
        raise ValueError(
            "Unknown case override; only name, sampling and controlRegions are applied"
        )
    generation = project.get("doc", {}).get("generation", {})
    if any(
        not isinstance(generation.get(key), str)
        or generation[key] in {"", "auto", "none"}
        for key in ("unet", "clip", "video_vae", "audio_vae")
    ):
        raise ValueError(
            "Qualification requires explicit existing model selections; no auto selection is invented"
        )
    media = declared.get("assets", {})
    ids = {a["id"] for a in project.get("assets", [])}
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
        "project_identity": file_identity(project_file),
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

    for cls in (
        nodes_custom_sampler.BasicGuider,
        nodes_custom_sampler.RandomNoise,
        nodes_custom_sampler.SamplerCustomAdvanced,
        LoadVideo,
        GetVideoComponents,
    ):
        core_nodes.NODE_CLASS_MAPPINGS[cls.__name__] = cls
    import execution

    generation = importlib.import_module(name + ".director_generation")
    project_api = importlib.import_module(name + ".director_project")
    batch = importlib.import_module(name + ".director_batch")
    resolver = importlib.import_module(
        name + ".director_routes"
    )._resolve_director_resource
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
        write_json(root / relative / "project.json", candidate)
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
                "project": relative + "/project.json",
                "built": relative + "/built.json",
                "recipe": built["recipe"],
                "sampling": built["sampling"],
                "validation": relative + "/core-validation.json",
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
                graph = read_json(root / case["prompt"])
                submitted = server.request(
                    "POST",
                    "/prompt",
                    json={
                        "prompt": graph,
                        "client_id": "h21-control-" + plan["run_id"],
                    },
                )
                folder = attempt / "executions" / case["name"]
                write_json(folder / "submitted.json", submitted)
                if not submitted.get("prompt_id") or submitted.get("node_errors"):
                    raise RuntimeError(
                        "Real Core rejected the frozen graph; no automatic recipe fallback"
                    )
                prompt_id = submitted["prompt_id"]
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    history = server.request("GET", "/history/" + prompt_id)
                    if prompt_id in history:
                        result = history[prompt_id]
                        write_json(folder / "history.json", result)
                        if result.get("status", {}).get("status_str") != "success":
                            raise RuntimeError(
                                "Real Core execution failed; original history retained"
                            )
                        break
                    if server.process.poll() is not None:
                        raise RuntimeError("Owned Core exited during execution")
                    time.sleep(1)
                else:
                    raise TimeoutError("Owned graph did not complete before timeout")
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
