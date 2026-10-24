"""v4 owned-worker transport; old runtime classes, schemas and defaults are untouched."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

from .source import Source, REVISION, canonical, digest
from .plans import RUNTIME_SCHEMA, plan
from .stages import make, tensor_record, validate


@dataclass(frozen=True)
class Model:
    config_path: str
    config_sha256: str
    loras: tuple = ()
    effects_json: str = "{}"


def config_for(model, *, full=False):
    from ..freevideo_exp.runtime import verify_files
    if not isinstance(model, Model) or digest(model.config_path) != model.config_sha256:
        raise ValueError("H21 runtime config content changed")
    config = json.loads(Path(model.config_path).read_text(encoding="utf8"))
    if config.get("schema") != RUNTIME_SCHEMA or config.get("freevideo_revision") != REVISION:
        raise ValueError("Use an independently prepared H21 runtime config, not another producer's cache identity")
    for key in ("python", "source_root", "vdn_root", "cache", "base", "checkpoint", "home", "sampling_root"):
        if not isinstance(config.get(key), str) or not Path(config[key]).is_absolute():
            raise ValueError("H21 runtime needs explicit absolute " + key)
    if (not config.get("files") or
        hashlib.sha256(canonical(config["files"]).encode()).hexdigest() != config.get("inventory_sha256")):
        raise ValueError("H21 runtime inventory is incomplete or changed")
    if full:
        verify_files(config["files"])
    return config


def load_model(path):
    path = Path(path).resolve(strict=True)
    model = Model(str(path), digest(path))
    config_for(model)
    return model


def with_lora(model, path, strength):
    if type(strength) not in (int, float) or not math.isfinite(strength):
        raise ValueError("LoRA strength must be finite")
    if strength == 0:
        row = dict(path=str(path), strength=strength, disabled=True)
    else:
        path = Path(path).resolve(strict=True)
        row = dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path), strength=strength)
    return replace(model, loras=(*model.loras, canonical(row)))


def pack_condition(value, tensors=None):
    """Only actual typed tensor/JSON data, never pickle foreign executable owners."""
    import torch
    tensors = {} if tensors is None else tensors
    def pack(item):
        if isinstance(item, torch.Tensor):
            if item.is_floating_point():
                tensor_record(item)
            key = "condition_" + str(len(tensors))
            tensors[key] = item.detach().cpu().contiguous()
            return {"$h21_tensor": key}
        if type(item) in (str, bool, int, type(None)):
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("H21 conditioning JSON contains non-finite scalar")
            return item
        if type(item) in (tuple, list):
            return [pack(v) for v in item]
        if type(item) is dict and all(type(k) is str for k in item):
            if "$h21_tensor" in item:
                raise ValueError("Reserved H21 tensor transport key in user metadata")
            return {key: pack(v) for key, v in item.items()}
        raise TypeError("H21 owned-worker data transport needs tensors/JSON, not an executable object: " + type(item).__name__)
    return pack(value), tensors


def unpack_condition(value, tensors):
    if type(value) is dict:
        if set(value) == {"$h21_tensor"}:
            return tensors[value["$h21_tensor"]]
        return {key: unpack_condition(v, tensors) for key, v in value.items()}
    if type(value) is list:
        return [unpack_condition(v, tensors) for v in value]
    return value


def worker_environment(config):
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "FREEVIDEO_STEP_CACHE",
                "FREEVIDEO_RUNTIME_LOCK_FD", "FREEVIDEO_RUNTIME_LOCK_HANDLE"):
        env.pop(key, None)
    env.update(FREEVIDEO_HOME=config["home"], FREEVIDEO_VDN_ROOT=config["vdn_root"],
        TRITON_CACHE_DIR=str(Path(config["home"]) / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(Path(config["home"]) / "inductor"),
        TORCHINDUCTOR_COMPILE_THREADS="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    return env


def sample(model, conditioning, width, height, frames, seed, *, tier="light", role="FULL", low=None,
           lifted=None, interrupt=None, progress=None):
    from safetensors.torch import load_file, save_file
    from ..freevideo_exp.supervision import owned_process
    from ..freevideo_exp.runtime import verify_files, write_json_new
    config = config_for(model, full=True)
    source = Source(config["source_root"])
    canvas = dict(width=width, height=height, frames=frames)
    from .assets import model_structure
    text_dim = model_structure(config["base"])["text_dim"]
    from .effects import transport as transport_effects
    effects = transport_effects(model, conditioning, canvas, text_dim=text_dim)
    if type(conditioning) is dict:
        value = dict(conditioning)
        task = value.get("task", "t2va")
    else:
        from ..freevideo_exp.runtime import conditioning_transport
        tensors, meta = (conditioning_transport(conditioning, canvas) if text_dim == 5120 else
                         conditioning_transport(conditioning, canvas, text_dim=text_dim))
        value = dict(prompt_embeds=tensors["embeds"], text_token_tags=tensors["tags"], task=meta["task"], width=width, height=height)
        if meta["keyframes"]:
            value.update(keyframe_anchors=[r["anchor"] for r in meta["keyframes"]],
                         condition_latents=[tensors[r["tensor"]] for r in meta["keyframes"]])
        if meta["refs"]:
            value["references"] = [dict(kind=r["kind"], **({"latent": tensors[r["tensor"]]} if "tensor" in r else {}),
                 **({"audio_latent": tensors[r["audio_tensor"]]} if "audio_tensor" in r else {})) for r in meta["refs"]]
        task = meta["task"]
    selected = plan(source, tier, width, height, frames, task=task, role=role)
    if type(seed) is not int or not 0 <= seed < 1 << 64:
        raise ValueError("Actual author sampler seed must fit uint64")
    transport, tensors = pack_condition(value)
    low_receipt = None
    if role == "HIGH":
        low_receipt = validate(low)
        if (low_receipt["plan"]["role"] != "LOW" or low_receipt["actual"]["completed_callbacks"] != 8
            or low_receipt["plan"]["actual_canvas"]["frames"] != frames):
            raise ValueError("H21 HIGH needs an actual completed LOW8 with the same temporal canvas")
        if lifted is None:
            raise ValueError("H21 HIGH needs the actual lifted/cropped video tensor")
        tensor_record(lifted)
        if (tuple(lifted.shape[:3]) != tuple(low.video.shape[:3])
            or tuple(lifted.shape[-2:]) != (height // 16, width // 16)):
            raise ValueError("Actual HIGH learned lift shape disagrees with the completed LOW time axes and target canvas")
        tensors.update(initial_video=lifted.detach().cpu().contiguous(), low_video=low.video,
                       initial_audio=low.audio[0].permute(1, 0, 2).contiguous())
    root = Path(config["home"]) / "runs-v4" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(root / "input.safetensors"))
    request = dict(schema="t8.freevideo.h21.request.v4", config_path=model.config_path, config_sha256=model.config_sha256,
        plan=selected, seed=seed, conditioning=transport, input_sha256=digest(root / "input.safetensors"),
        loras=list(model.loras), effects=effects,
        low_receipt_sha256=low.receipt_sha256 if low else None, low_audio=tensor_record(low.audio) if low else None)
    if low_receipt is not None:
        request["low_stack"] = dict(config_sha256=low_receipt["config_sha256"], loras=low_receipt["loras"],
            HIGH_stack_equal=low_receipt["config_sha256"] == model.config_sha256 and low_receipt["loras"] == list(model.loras),
            warning="Stage selections are preserved; differing stacks have no trained composite-quality qualification.")
    write_json_new(root / "request.json", request)
    command = [config["python"], "-I", "-B", "-X", "utf8", str(Path(__file__).with_name("worker.py")), str(root)]
    with (root / "worker.log").open("x", encoding="utf8") as log, owned_process(command, worker_environment(config), root, log) as process:
        while process.poll() is None:
            if interrupt:
                interrupt()
            if progress and (root / "progress.json").is_file():
                try:
                    status = json.loads((root / "progress.json").read_text(encoding="utf8"))
                    progress(status["completed"], status["total"])
                except (OSError, ValueError, KeyError):
                    pass
            time.sleep(.2)
        if process.returncode:
            raise RuntimeError("H21 owned worker failed; original evidence retained: " + str(root) + "\n" +
                               (root / "worker.log").read_text(encoding="utf8")[-6000:])
    config_for(model, full=True)
    result = json.loads((root / "result.json").read_text(encoding="utf8"))
    if result["request_sha256"] != digest(root / "request.json") or result["output_sha256"] != digest(root / "output.safetensors"):
        raise ValueError("H21 worker output/request content changed")
    verify_files([json.loads(row) for row in model.loras if not json.loads(row).get("disabled")])
    values = load_file(str(root / "output.safetensors"), device="cpu")
    receipt = dict(result, plan=selected, config_sha256=model.config_sha256, inventory_sha256=config["inventory_sha256"],
        loras=list(model.loras), low_receipt_sha256=request["low_receipt_sha256"], low_audio=request["low_audio"],
        run_directory=str(root), source_revision=REVISION)
    return make(values["video"], values["audio"], receipt)
