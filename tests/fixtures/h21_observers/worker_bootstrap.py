"""Opt-in private source-copy bootstrap, inactive during ordinary CPU prepare.

This file will be copied only into an explicitly derived, separate source root.
It is not imported by the live project. No Core function or sampler is changed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def semantic_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write_new(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def inactive_or_contract(package_file, output_directory):
    root = Path(output_directory).resolve().parent
    marker = root / "D-worker-observer.json"
    if not marker.is_file():
        return None
    record = read(marker)
    if record["schema"] != "t8.h21.D.private-worker-activation.v1" or record["enabled"] is not True:
        raise ValueError("Explicit private D observer activation required")
    manifest_path = root / "manifest.json"
    manifest = read(manifest_path)
    if (digest(manifest_path) != record["manifest_sha256"]
            or Path(manifest["run_root"]).resolve() != root
            or Path(package_file).resolve() != Path(manifest["package_root"]).resolve() / "__init__.py"
            or Path(sys.executable).resolve() != Path(manifest["python"]).resolve()):
        raise ValueError("D observer actual frozen package/root/Python/manifest differs")
    binding_path = root / record["activation_binding"]
    if (not binding_path.resolve().is_relative_to(root) or digest(binding_path) != record["activation_binding_sha256"]):
        raise ValueError("Actual extra activation binding changed or escaped owned root")
    binding = read(binding_path)
    # Bind the contract, not the marker's bytes: the marker already seals this
    # binding's SHA, so a reciprocal marker SHA would be an impossible hash
    # cycle. The execute guard separately seals the final marker pre/post.
    expected = {key: record[key] for key in ("enabled", "manifest_sha256", "output_relative", "cases")}
    if (binding["schema"] != "t8.h21.D.private-execution-binding.v1"
            or binding["activation_contract"] != expected):
        raise ValueError("Actual private activation contract differs from its guard binding")
    output = (root / record["output_relative"]).resolve()
    if (not output.is_relative_to(root / "observations")
            or output == root / "observations" or output.exists()):
        raise ValueError("New owned observer output directory required")
    output.mkdir(parents=True, exist_ok=False)
    return root, manifest, record, output


def install(package_file):
    import folder_paths
    contract = inactive_or_contract(package_file, folder_paths.get_output_directory())
    if contract is None:
        return None
    root, manifest, record, output = contract
    import execution
    import nodes
    import torch
    from comfy.ldm.minimax import model as native
    from comfy.sd import VAE
    from . import hyperflow_runtime_advanced as hyperflow
    from . import multikeyframe_advanced as advanced
    from .h21_d_profiler import ConsumptionProfiler, tensor_record
    from .h21_d_router import OwnedWorkerRouter
    from .h21_d_adapter import RequestObservation

    main = sys.modules.get("__main__")
    worker = None if main is None else getattr(main, "prompt_worker", None)
    core_main = Path(manifest["core_root"]) / "main.py"
    if (worker is None or Path(worker.__code__.co_filename).resolve() != core_main.resolve()
            or digest(core_main) != manifest["core_files"]["main.py"]["sha256"]):
        write_new(output / "bootstrap.json", dict(status="actual_owned_worker_not_found_no_observation", generation_blocked=False))
        return None
    selected = {}
    for item in record["cases"]:
        case, = [case for case in manifest["cases"] if case["name"] == item["name"]]
        path = (root / case["prompt"]).resolve(strict=True)
        frozen = manifest["sealed_files"][case["prompt"]]["sha256"]
        if digest(path) != frozen or frozen != item["prompt_sha256"]:
            raise ValueError("D activation actual selected prompt changed")
        graph = read(path)
        key = semantic_sha(graph)
        if key in selected:
            raise ValueError("Actual D selected prompt identity duplicated")
        selected[key] = dict(item, actual_prompt=graph)
    assets = {}
    for asset_id, item in manifest["resources"]["assets"].items():
        path = (root / "input" / item["server_path"]).resolve(strict=True)
        if not path.is_relative_to(root / "input"):
            raise ValueError("Actual selected source escaped owned input")
        row = assets.setdefault(str(path), dict(sha256=item["sha256"], asset_ids=[]))
        if row["sha256"] != item["sha256"]:
            raise ValueError("Actual source path has contradictory asset identities")
        row["asset_ids"].append(asset_id)
    deps = SimpleNamespace(native=native, hyperflow=hyperflow, advanced=advanced,
        profiler_base=ConsumptionProfiler, tensor_record=tensor_record,
        module_call_code=torch.nn.Module._call_impl.__code__, load_image_code=nodes.LoadImage.load_image.__code__,
        keyframe_encode_code=advanced._encode_keyframe.__code__, vae_encode_code=VAE.encode.__code__,
        actual_frame_index_key=advanced.ACTUAL_FRAME_INDEX)

    def begin(frame):
        prompt = frame.f_locals["prompt"]
        item = selected.get(semantic_sha(prompt))
        if item is None or item["actual_prompt"] != prompt:
            return None
        prompt_id = str(uuid.UUID(frame.f_locals["prompt_id"]))
        return RequestObservation(item, prompt_id, deps=deps, assets=assets)

    def observe(state, frame, event, returned):
        state.event(frame, event, returned)

    def finish(state, frame, returned, aborted):
        report = state.finish(frame, aborted=aborted)
        report["actual_owned_worker_router"] = router.report()
        report["actual_process_PID"] = os.getpid()
        report["actual_request_return_value_is_None"] = returned is None
        report["activation_sha256"] = digest(root / "D-worker-observer.json")
        report["manifest_sha256"] = record["manifest_sha256"]
        write_new(output / ("prompt-" + state.prompt_id + ".json"), report)

    router = OwnedWorkerRouter(worker, execution.PromptExecutor.execute, begin=begin, observe=observe, end=finish)
    active = router.arm()
    write_new(output / "bootstrap.json", dict(status=router.status, activated=active,
        actual_worker_code_file=worker.__code__.co_filename, actual_worker_code_qualname=worker.__code__.co_qualname,
        manifest_sha256=record["manifest_sha256"], activation_sha256=digest(root / "D-worker-observer.json"),
        module_hooks_installed=False, generation_blocked=False, actual_consumption_qualified=False))
    # Keep only the router, which closes request records itself and stores no
    # model. The worker's inherited profile owns its normal process lifetime.
    return router
