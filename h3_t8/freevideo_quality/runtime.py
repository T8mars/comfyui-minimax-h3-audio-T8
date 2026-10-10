"""Independent quality descriptor, completed stages and owned process transport."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

from ..freevideo_exp.runtime import (MODEL_REVISION, VDN_REVISION, FreeVideoModel, canonical,
    conditioning_transport, digest, geometry, tensor_record, verify_files, write_json_new)
from .profiles import (LEGACY_RUNTIME, binding, receipt_binding, plan, profile, clock, validate_clock)


@dataclass(frozen=True)
class QualityModel:
    # Intentionally NOT a FreeVideoModel subclass: legacy runtime rejects wrong dispatch.
    config_path: str
    config_sha256: str
    loras: tuple = ()
    eav: str | None = None
    relay: str | None = None


def facade(model):
    if not isinstance(model, QualityModel):
        raise ValueError("Use the new Quality Loader, not the legacy FreeVideo Loader")
    return FreeVideoModel(model.config_path, model.config_sha256, model.loras, model.eav, model.relay)


def from_facade(value):
    return QualityModel(value.config_path, value.config_sha256, value.loras, value.eav, value.relay)


def load_model(path, *, schema=LEGACY_RUNTIME):
    path = Path(path).resolve(strict=True)
    model = QualityModel(str(path), digest(path))
    config = config_for(model)
    if schema is not None and config["schema"] != schema:
        raise ValueError("Choose the loader for the explicit FreeVideo runtime producer family")
    return model


def config_for(model, full=False):
    facade(model)
    if digest(model.config_path) != model.config_sha256:
        raise ValueError("Quality runtime/config content changed")
    value = json.loads(Path(model.config_path).read_text(encoding="utf8"))
    producer = binding(value.get("schema"))
    if (
        tuple(value.get(key) for key in ("freevideo_revision", "vdn_revision", "model_revision")) !=
            (producer["revision"], VDN_REVISION, MODEL_REVISION)):
        raise ValueError("Quality runtime/source differs from its explicit pinned producer family")
    for name in ("python", "source_root", "vdn_root", "cache", "base", "checkpoint", "home", "sampling_root"):
        if not value.get(name) or not Path(value[name]).is_absolute():
            raise ValueError("Quality runtime requires an explicit absolute " + name)
    if not value.get("files") or value.get("inventory_sha256") != hashlib.sha256(canonical(value["files"]).encode()).hexdigest():
        raise ValueError("Quality runtime has no intact source/model inventory")
    if full:
        verify_files(value["files"])
    return value


@dataclass(frozen=True)
class QualityStage:
    video: object
    audio: object
    receipt_json: str
    receipt_sha256: str


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_completion(video, audio, receipt):
    producer = receipt_binding(receipt.get("schema"))
    selected = plan(receipt.get("profile"), receipt.get("role"))
    canvas = geometry(**receipt["geometry"])
    vt = (canvas["frames"] - 5) // 17 * 5 + 2
    if tuple(video.shape) != (1, 24, vt, canvas["height"] // 16, canvas["width"] // 16):
        raise ValueError("Quality video shape mismatch")
    if tuple(audio.shape) != (1, 32, 2, round(canvas["frames"] / 24 * 40)):
        raise ValueError("Quality audio shape mismatch")
    if (receipt.get("plan") != selected
        or receipt.get("freevideo_revision") != producer["revision"] or receipt.get("vdn_revision") != VDN_REVISION
        or receipt.get("model_revision") != MODEL_REVISION):
        raise ValueError("Quality stage producer/profile mismatch")
    validate_clock(receipt.get("clock"), selected["profile"], selected["role"], receipt.get("task"),
                   reference_audio_t=producer["reference_audio_t"])
    count = receipt.get("completed_nfe")
    seconds = receipt.get("sample", {}).get("step_seconds", [])
    if (type(count) is not int or count != selected["nfe"] or len(seconds) != count
        or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in seconds)):
        raise ValueError("Quality stage does not contain all actual completed NFE")
    for name in ("request_sha256", "output_sha256", "source_inventory_sha256", "weight_identity", "tables_sha256"):
        if not is_sha(receipt.get(name)):
            raise ValueError("Missing quality producer identity: " + name)
    from .assets import evidence_sha
    from .profiles import table_identity
    tables = receipt.get("tables")
    roles = ["LOW", "HIGH"] if selected["role"] == "HIGH" else [selected["role"]]
    identities = [table_identity(receipt["weight_identity"], selected["profile"], role, receipt["task"],
                                reference_audio_t=producer["reference_audio_t"]) for role in roles]
    if (not isinstance(tables, list) or len(tables) != 50 * len(identities)
        or evidence_sha(tables) != receipt["tables_sha256"]
        or any(row.get("identity") not in identities or not is_sha(row.get("sha256"))
               or type(row.get("bytes")) is not int or row["bytes"] < 1 for row in tables)
        or any(len([row for row in tables if row["identity"] == identity]) != 50
               or {row["index"] for row in tables if row["identity"] == identity} != set(range(50)) for identity in identities)):
        raise ValueError("Quality stage needs exact task/clock table identity and all 50 actual layer receipts")
    if selected["role"] == "HIGH":
        if not is_sha(receipt.get("low_receipt_sha256")) or receipt.get("low_audio") != tensor_record(audio):
            raise ValueError("Community HIGH3 must preserve completed LOW audio and its actual lineage")
    return selected


def make_stage(video, audio, receipt):
    validate_completion(video, audio, receipt)
    video, audio = video.detach().cpu().clone(), audio.detach().cpu().clone()
    receipt = dict(receipt, video=tensor_record(video), audio=tensor_record(audio))
    text = canonical(receipt)
    return QualityStage(video, audio, text, hashlib.sha256(text.encode()).hexdigest())


def validate_stage(stage, role=None):
    if not isinstance(stage, QualityStage) or hashlib.sha256(stage.receipt_json.encode()).hexdigest() != stage.receipt_sha256:
        raise ValueError("Expected a quality-v2 completed stage, not legacy Stage, MID or arbitrary LATENT")
    receipt = json.loads(stage.receipt_json)
    validate_completion(stage.video, stage.audio, receipt)
    if role is not None and receipt["role"] != role:
        raise ValueError("Community HIGH3 needs Light completed LOW8; this stage is " + receipt["role"])
    if tensor_record(stage.video) != receipt["video"] or tensor_record(stage.audio) != receipt["audio"]:
        raise ValueError("Quality stage tensor content changed")
    return receipt


def av_output(stage):
    from comfy.nested_tensor import NestedTensor
    validate_stage(stage)
    return {"samples": NestedTensor((stage.video.clone(), stage.audio.clone()))}


def save_stage(stage, directory):
    from safetensors.torch import save_file
    receipt = validate_stage(stage)
    target = Path(directory).resolve() / (receipt["role"].lower() + "-" + uuid.uuid4().hex)
    target.mkdir(parents=True, exist_ok=False)
    temporary = target / "latents.partial"
    save_file({"video": stage.video.contiguous(), "audio": stage.audio.contiguous()}, str(temporary))
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    tensor_path = target / "latents.safetensors"
    temporary.rename(tensor_path)
    marker = target / "stage.json"
    write_json_new(marker, dict(schema=receipt_binding(receipt["schema"])["saved"], tensors="latents.safetensors",
                               tensor_sha256=digest(tensor_path), receipt=receipt))
    return str(marker), digest(marker)


def load_stage(path, sha256):
    from safetensors.torch import load_file
    if not isinstance(sha256, str) or not is_sha(sha256.lower()):
        raise ValueError("Supply the actual saved Quality Stage manifest SHA256")
    path = Path(path).resolve(strict=True)
    if digest(path) != sha256.lower():
        raise ValueError("Saved quality manifest SHA mismatch")
    value = json.loads(path.read_text(encoding="utf8"))
    producer = receipt_binding(value.get("receipt", {}).get("schema"))
    if value.get("schema") != producer["saved"] or value.get("tensors") != "latents.safetensors":
        raise ValueError("Wrong Quality Stage layout; legacy Stage and MID remain separate")
    payload = path.parent / "latents.safetensors"
    if payload.is_symlink() or payload.resolve().parent != path.parent or digest(payload) != value["tensor_sha256"]:
        raise ValueError("Quality stage tensor path/content mismatch")
    values = load_file(str(payload), device="cpu")
    if set(values) != {"video", "audio"}:
        raise ValueError("Quality stage tensor keys mismatch")
    stage = make_stage(values["video"], values["audio"], value["receipt"])
    if json.loads(stage.receipt_json) != value["receipt"]:
        raise ValueError("Saved quality receipt does not describe actual tensors")
    return stage


def sample(model, conditioning, width, height, frames, seed, *, quality="light", low=None, lifted=None,
           interrupt=None, progress=None):
    from safetensors.torch import load_file, save_file
    from ..freevideo_exp.effects import transport_effects
    from ..freevideo_exp.supervision import owned_process
    config = config_for(model, full=True)
    producer = binding(config["schema"])
    quality = profile(quality)
    selected = plan(quality, "HIGH" if low is not None else None)
    canvas = geometry(width, height, frames)
    tensors, conditions = conditioning_transport(conditioning, canvas)
    effects = transport_effects(facade(model), conditioning, canvas)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Quality seed must fit signed int64")
    low_receipt = None
    if low is not None:
        from ..core import nested_av_parts
        low_receipt = validate_stage(low, "LOW")
        if low_receipt["freevideo_revision"] != producer["revision"]:
            raise ValueError("HIGH3 cannot mix old LOW and the new reference-audio producer family")
        video, audio = nested_av_parts(lifted)
        if (tuple(video.shape[:3]) != tuple(low.video.shape[:3]) or
            tuple(video.shape[-2:]) != (height // 16, width // 16) or tensor_record(audio) != tensor_record(low.audio)):
            raise ValueError("HIGH3 needs external lift with the exact completed LOW audio and frame count")
        if low_receipt["geometry"]["frames"] != frames or width < low.video.shape[-1] * 16 or height < low.video.shape[-2] * 16:
            raise ValueError("HIGH3 cannot change frame count or shrink its LOW canvas")
        tensors.update(initial_video=video.detach().cpu().contiguous(), initial_audio=audio[0].permute(1, 0, 2).cpu().contiguous())
    run = Path(config["home"]) / "quality-runs" / uuid.uuid4().hex
    run.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(run / "input.safetensors"))
    request = dict(schema=producer["request"], plan=selected, geometry=canvas, seed=seed,
                   clock=clock(quality, selected["role"], conditions["task"],
                               reference_audio_t=producer["reference_audio_t"]), conditions=conditions, effects=effects,
                   loras=[json.loads(row) for row in model.loras], config_path=model.config_path,
                   config_sha256=model.config_sha256, input_sha256=digest(run / "input.safetensors"),
                   low_receipt_sha256=low.receipt_sha256 if low else None,
                   low_audio=tensor_record(low.audio) if low else None)
    write_json_new(run / "request.json", request)
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "FREEVIDEO_RUNTIME_LOCK_FD", "FREEVIDEO_RUNTIME_LOCK_HANDLE"):
        env.pop(key, None)
    env.update(FREEVIDEO_HOME=config["home"], FREEVIDEO_VDN_ROOT=config["vdn_root"],
               TRITON_CACHE_DIR=str(Path(config["home"]) / "triton"), TORCHINDUCTOR_CACHE_DIR=str(Path(config["home"]) / "inductor"),
               TORCHINDUCTOR_COMPILE_THREADS="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    command = [config["python"], "-I", "-B", "-X", "utf8", str(Path(__file__).with_name("worker.py")), str(run)]
    with (run / "worker.log").open("x", encoding="utf8") as log, owned_process(command, env, run, log) as process:
        while process.poll() is None:
            if interrupt:
                interrupt()
            status = run / "progress.json"
            if progress and status.is_file():
                try:
                    progress(json.loads(status.read_text(encoding="utf8"))["completed"], selected["nfe"])
                except (OSError, ValueError, KeyError):
                    pass
            time.sleep(.2)
        if process.returncode:
            raise RuntimeError(f"Quality worker failed; evidence at {run}\n" + (run / "worker.log").read_text(encoding="utf8")[-5000:])
    config_for(model, full=True)
    verify_files([row for row in request["loras"] if row["strength"] != 0])
    result = json.loads((run / "result.json").read_text(encoding="utf8"))
    if result.get("request_sha256") != digest(run / "request.json") or result.get("output_sha256") != digest(run / "output.safetensors"):
        raise ValueError("Quality worker result identity mismatch")
    values = load_file(str(run / "output.safetensors"), device="cpu")
    receipt = dict(result, schema=producer["stage"], plan=selected, role=selected["role"], profile=quality,
                   geometry=canvas, task=conditions["task"], clock=request["clock"], freevideo_revision=producer["revision"],
                   vdn_revision=VDN_REVISION, model_revision=MODEL_REVISION, seed=seed,
                   completed_nfe=len(result["sample"]["step_seconds"]), low_receipt_sha256=request["low_receipt_sha256"],
                   low_audio=request["low_audio"], run_directory=str(run), condition_transport=conditions, lora_slots=request["loras"])
    return make_stage(values["video"], values["audio"], receipt)
