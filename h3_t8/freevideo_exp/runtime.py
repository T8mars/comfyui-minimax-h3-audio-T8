"""Content-bound FreeVideo process adapter. No engine imports in ComfyUI."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

FREEVIDEO_REVISION = "98b3550ae551a70ae3f0a9a463f9dfc9a23b21a7"
VDN_REVISION = "30b6b380c2482f3519469350810c2955d8847fd9"
MODEL_REVISION = "ae041e5aec51f8516a3416bd86d579cb29c4c796"
_verified = {}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
        return result.hexdigest()


def fingerprint(path):
    stat = Path(path).stat()
    change = stat.st_ctime_ns
    if os.name == "nt":
        from .supervision import file_change_time_ns
        change = file_change_time_ns(path)
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, change)


def verify_files(rows):
    for row in rows:
        path = Path(row["path"]).resolve(strict=True)
        before = fingerprint(path)
        if before[2] != row["bytes"]:
            raise ValueError(f"FreeVideo file size changed: {path}")
        key = (str(path), row["sha256"])
        if before[-1] is None or _verified.get(key) != before:
            if digest(path) != row["sha256"] or fingerprint(path) != before:
                raise ValueError(f"FreeVideo file content changed: {path}")
            _verified[key] = before


def write_json_new(path, value):
    """Create-only metadata. Never overwrite a user's runtime or stage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class FreeVideoModel:
    config_path: str
    config_sha256: str
    loras: tuple = ()
    eav: str | None = None
    relay: str | None = None


def load_model(config_path):
    path = Path(config_path).resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = (FREEVIDEO_REVISION, VDN_REVISION, MODEL_REVISION)
    if value.get("schema") != "t8-freevideo-runtime-v1" or tuple(value.get(k) for k in
            ("freevideo_revision", "vdn_revision", "model_revision")) != expected:
        raise ValueError("Unsupported FreeVideo runtime pins; run the preparation tool for this version")
    for name in ("python", "source_root", "vdn_root", "cache", "base", "checkpoint", "home"):
        if not value.get(name) or not Path(value[name]).is_absolute():
            raise ValueError("FreeVideo runtime needs explicit absolute " + name)
    if not value.get("files"):
        raise ValueError("FreeVideo runtime has no verified source/model inventory")
    return FreeVideoModel(str(path), digest(path))


def config_for(model, full=False):
    if not isinstance(model, FreeVideoModel) or digest(model.config_path) != model.config_sha256:
        raise ValueError("FreeVideo loader/config identity changed")
    value = json.loads(Path(model.config_path).read_text(encoding="utf-8"))
    if full:
        verify_files(value["files"])
    return value


def with_lora(model, path, strength):
    if not isinstance(model, FreeVideoModel) or not math.isfinite(strength):
        raise ValueError("FreeVideo LoRA needs its own model and a finite strength")
    # Zero branches are intentional bypasses; do not touch a missing disabled file.
    if strength == 0:
        return replace(model, loras=(*model.loras, canonical(dict(path=str(path), strength=0.0, disabled=True))))
    path = Path(path).resolve(strict=True)
    if path.suffix.lower() not in (".safetensors", ".sft"):
        raise ValueError("FreeVideo LoRA expects safetensors")
    row = canonical(dict(path=str(path), strength=float(strength), sha256=digest(path), bytes=path.stat().st_size))
    return replace(model, loras=(*model.loras, row))


def geometry(width, height, frames):
    if any(type(n) is not int or n < 256 or n % 32 for n in (width, height)):
        raise ValueError("FreeVideo dimensions must be multiples of 32, at least 256")
    if type(frames) is not int or frames < 39 or (frames - 5) % 17:
        raise ValueError("Use an aligned H3 frame count 17*n+5, at least 39 (124 for five-second output)")
    return dict(width=width, height=height, frames=frames)


def tensor_record(tensor):
    import torch
    value = tensor.detach().to("cpu").contiguous()
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("FreeVideo requires finite floating point tensors")
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def conditioning_transport(conditioning, canvas):
    """Export raw native H3 inputs, not a Core-transformed minimax_payload."""
    import torch
    if not isinstance(conditioning, (list, tuple)) or len(conditioning) != 1:
        raise ValueError("FreeVideo requires one native H3 conditioning sequence")
    embeds, extra = conditioning[0]
    allowed = {"minimax_token_tags", "minimax_keyframes", "minimax_refs", "pooled_output", "minimax_frame_count", "t8_semantic_bridge", "t8_freevideo_relay"}
    if set(extra) - allowed:
        raise ValueError("FreeVideo does not silently ignore conditioning modifiers: " + str(sorted(set(extra) - allowed)))
    if extra.get("minimax_frame_count", canvas["frames"]) != canvas["frames"]:
        raise ValueError("Conditioning frame count differs from FreeVideo sampling")
    if tuple(embeds.shape[:1]) != (1,) or embeds.ndim != 3 or embeds.shape[-1] != 5120 or not embeds.shape[1]:
        raise ValueError("Expected native H3 layer-50 [1,L,5120] embeddings")
    tensor_record(embeds)
    tags = extra.get("minimax_token_tags")
    if not isinstance(tags, torch.Tensor) or tags.ndim != 1 or tags.numel() != embeds.shape[1] or not bool(((tags == 0) | (tags == 1)).all()):
        raise ValueError("Invalid native H3 token tags")
    tensors = {"embeds": embeds[0].detach().cpu().contiguous(), "tags": tags.detach().to("cpu", torch.int64).contiguous()}
    meta = {"keyframes": [], "refs": []}
    if extra.get("t8_semantic_bridge"):
        bridge = extra["t8_semantic_bridge"]
        native = embeds.detach().cpu().contiguous()
        bound = hashlib.sha256(str((tuple(native.shape), native.dtype)).encode())
        bound.update(native.view(torch.uint8).numpy().tobytes())
        if not isinstance(bridge, dict) or bridge.get("output_sha256") != bound.hexdigest():
            raise ValueError("Applied semantic bridge receipt does not describe current embeddings")
        meta["semantic_bridge"] = bridge  # Already applied; never apply a second time.
    for i, row in enumerate(extra.get("minimax_keyframes") or []):
        frame = row.get("resolved_frame_index")
        if frame not in (0, canvas["frames"] - 1):
            raise ValueError("FreeVideo supports first/last keyframe anchors only")
        key = f"keyframe_{i}"
        latent = row["latent"]
        tensor_record(latent)
        if tuple(latent.shape) != (1, 24, 1, canvas["height"] // 16, canvas["width"] // 16):
            raise ValueError("Keyframe latent must match this stage's canvas")
        tensors[key] = latent.detach().cpu().contiguous()
        meta["keyframes"].append({"anchor": "first" if frame == 0 else "last", "tensor": key})
    for i, row in enumerate(extra.get("minimax_refs") or []):
        kind = row.get("kind")
        if kind not in ("image", "video", "audio", "video_audio"):
            raise ValueError("Unsupported FreeVideo reference modality")
        ref = {"kind": "video" if kind == "video_audio" else kind}
        if kind != "audio":
            latent = row["latent"]
            tensor_record(latent)
            if (latent.ndim != 5 or tuple(latent.shape[:2]) != (1, 24) or latent.shape[2] < 1
                    or (kind == "image" and latent.shape[2] != 1)
                    or any(n < 2 or n % 2 for n in latent.shape[-2:])):
                raise ValueError("Invalid FreeVideo reference video shape")
            key = f"ref_{i}"
            tensors[key] = latent.detach().cpu().contiguous()
            ref["tensor"] = key
        if kind in ("audio", "video_audio") or row.get("audio_latent") is not None:
            audio = row.get("audio_latent")
            if not isinstance(audio, torch.Tensor) or audio.ndim != 4 or tuple(audio.shape[:3]) != (1, 32, 2) or audio.shape[-1] < 1:
                raise ValueError("Invalid FreeVideo native reference audio shape")
            tensor_record(audio)
            key = f"ref_audio_{i}"
            tensors[key] = audio[0].permute(1, 2, 0).reshape(-1, 32).detach().cpu().contiguous()
            ref["audio_tensor"] = key
        meta["refs"].append(ref)
    if meta["refs"] and meta["keyframes"]:
        raise ValueError("Combined keyframe/reference layout is not supported by the official engine")
    anchors = [row["anchor"] for row in meta["keyframes"]]
    has_video = any("tensor" in r for r in meta["refs"])
    has_audio = any("audio_tensor" in r for r in meta["refs"])
    task = (("ref2va_av" if has_video and has_audio else "ref2va_audio" if has_audio else "ref2va") if meta["refs"] else "fl2va" if anchors == ["first", "last"] else
            "i2va" if anchors == ["first"] else "l2va" if anchors == ["last"] else "t2va")
    if anchors not in ([], ["first"], ["last"], ["first", "last"]):
        raise ValueError("Invalid FreeVideo keyframe order")
    if (task in ("t2va", "ref2va_audio") and not bool((tags == 1).all())) or (task not in ("t2va", "ref2va_audio") and not bool((tags == 0).any())):
        raise ValueError("Conditioning token modalities differ from requested task")
    return tensors, dict(meta, task=task, embedding_conversion="official worker casts to BF16")


@dataclass(frozen=True)
class FreeVideoStage:
    video: object
    audio: object
    receipt_json: str
    receipt_sha256: str


def validate_completion(video, audio, receipt):
    """Apply the same producer/clock/completion checks to live and cold stages."""
    canvas = geometry(**receipt["geometry"])
    vt = (canvas["frames"] - 5) // 17 * 5 + 2
    if tuple(video.shape) != (1, 24, vt, canvas["height"] // 16, canvas["width"] // 16):
        raise ValueError("FreeVideo output video shape mismatch")
    if tuple(audio.shape) != (1, 32, 2, round(canvas["frames"] / 24 * 40)):
        raise ValueError("FreeVideo output audio shape mismatch")
    role, tail = receipt.get("role"), receipt.get("tail_steps")
    if role == "HIGH" and (type(tail) is not int or not 1 <= tail < 8):
        raise ValueError("FreeVideo HIGH requires a complete integer tail of 1..7 steps")
    nfe = 8 if role == "LOW" else tail
    steps = receipt.get("sample", {}).get("step_seconds", [])
    if (receipt.get("schema") != "t8-freevideo-stage-v1" or type(receipt.get("completed_nfe")) is not int
            or receipt["completed_nfe"] != nfe or len(steps) != nfe
            or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in steps)):
        raise ValueError("FreeVideo stage is not a complete supported sampling result")
    if (role not in ("LOW", "HIGH") or receipt.get("model_revision") != MODEL_REVISION
            or receipt.get("freevideo_revision") != FREEVIDEO_REVISION or receipt.get("vdn_revision") != VDN_REVISION):
        raise ValueError("Invalid FreeVideo stage producer")
    for name in ("request_sha256", "output_sha256", "source_inventory_sha256"):
        value = receipt.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("FreeVideo stage lacks a complete producer identity: " + name)


def make_stage(video, audio, receipt):
    video, audio = video.detach().cpu().clone(), audio.detach().cpu().clone()
    validate_completion(video, audio, receipt)
    receipt = dict(receipt, video=tensor_record(video), audio=tensor_record(audio))
    text = canonical(receipt)
    return FreeVideoStage(video, audio, text, hashlib.sha256(text.encode()).hexdigest())


def validate_stage(stage, role=None):
    if not isinstance(stage, FreeVideoStage) or hashlib.sha256(stage.receipt_json.encode()).hexdigest() != stage.receipt_sha256:
        raise ValueError("Expected an unchanged FreeVideo Stage, not an arbitrary completion dictionary")
    receipt = json.loads(stage.receipt_json)
    validate_completion(stage.video, stage.audio, receipt)
    if role and receipt["role"] != role:
        raise ValueError("This input requires a completed " + role + " stage")
    if tensor_record(stage.video) != receipt["video"] or tensor_record(stage.audio) != receipt["audio"]:
        raise ValueError("FreeVideo Stage tensor content changed")
    return receipt


def av_output(stage):
    from comfy.nested_tensor import NestedTensor
    validate_stage(stage)
    # Clone so a downstream in-place operator cannot mutate the saved producer.
    return {"samples": NestedTensor((stage.video.clone(), stage.audio.clone()))}


def save_stage(stage, directory):
    from safetensors.torch import save_file
    receipt = validate_stage(stage)
    root = Path(directory).resolve()
    target = root / (receipt["role"].lower() + "-" + uuid.uuid4().hex)
    target.mkdir(parents=True, exist_ok=False)
    temporary = target / "latents.partial"
    save_file({"video": stage.video.contiguous(), "audio": stage.audio.contiguous()}, str(temporary))
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    tensor_path = target / "latents.safetensors"
    temporary.rename(tensor_path)
    marker = target / "stage.json"
    write_json_new(marker, dict(schema="t8-freevideo-saved-v1", tensors="latents.safetensors",
                               tensor_sha256=digest(tensor_path), receipt=receipt))
    return str(marker), digest(marker)


def load_stage(path, sha256, role="LOW"):
    from safetensors.torch import load_file
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256.lower()):
        raise ValueError("Supply the actual 64-character saved FreeVideo manifest SHA256")
    path = Path(path).resolve(strict=True)
    if digest(path) != sha256.lower():
        raise ValueError("Saved FreeVideo manifest SHA mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "t8-freevideo-saved-v1" or value.get("tensors") != "latents.safetensors":
        raise ValueError("Invalid FreeVideo saved-stage path/layout")
    tensor_path = path.parent / value["tensors"]
    if tensor_path.is_symlink() or tensor_path.resolve().parent != path.parent or digest(tensor_path) != value["tensor_sha256"]:
        raise ValueError("Saved FreeVideo tensor path/content mismatch")
    values = load_file(str(tensor_path), device="cpu")
    if set(values) != {"video", "audio"}:
        raise ValueError("Invalid saved FreeVideo tensor keys")
    stage = make_stage(values["video"], values["audio"], value["receipt"])
    if json.loads(stage.receipt_json) != value["receipt"]:
        raise ValueError("Saved FreeVideo receipt does not describe actual tensors")
    validate_stage(stage, role)
    return stage


def sample(model, conditioning, width, height, frames, seed, *, low=None, lifted=None, tail_steps=2, interrupt=None, progress=None):
    """One owned external process per stage; no global Comfy model unloading."""
    from safetensors.torch import load_file, save_file
    config = config_for(model, full=True)
    canvas = geometry(width, height, frames)
    tensors, conditions = conditioning_transport(conditioning, canvas)
    from .effects import transport_effects
    effects = transport_effects(model, conditioning, canvas)
    role = "HIGH" if low is not None else "LOW"
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("FreeVideo seed must fit signed int64")
    low_receipt = None
    if low is not None:
        from ..core import nested_av_parts
        low_receipt = validate_stage(low, "LOW")
        if type(tail_steps) is not int or not 1 <= tail_steps < 8:
            raise ValueError("FreeVideo HIGH uses 1..7 tail steps of the original eight-step schedule")
        video, audio = nested_av_parts(lifted)
        if tuple(video.shape[:3]) != tuple(low.video.shape[:3]) or tuple(video.shape[-2:]) != (height // 16, width // 16) or tensor_record(audio) != tensor_record(low.audio):
            raise ValueError("HIGH requires an external video lift with the exact completed LOW audio and frame count")
        if low_receipt["geometry"]["frames"] != frames:
            raise ValueError("HIGH frame count must equal completed LOW")
        if width < low_receipt["geometry"]["width"] or height < low_receipt["geometry"]["height"]:
            raise ValueError("HIGH cannot shrink the completed LOW canvas")
        tensors.update(initial_video=video.detach().cpu().contiguous(), initial_audio=audio[0].permute(1, 0, 2).cpu().contiguous())
    run_dir = Path(config["home"]) / "t8-runs" / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(run_dir / "input.safetensors"))
    request = dict(schema="t8-freevideo-request-v1", geometry=canvas, role=role, seed=seed,
                   tail_steps=tail_steps, conditions=conditions, effects=effects, loras=[json.loads(row) for row in model.loras],
                   config_path=model.config_path, config_sha256=model.config_sha256,
                   input_sha256=digest(run_dir / "input.safetensors"))
    write_json_new(run_dir / "request.json", request)
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "FREEVIDEO_RUNTIME_LOCK_FD", "FREEVIDEO_RUNTIME_LOCK_HANDLE"):
        environment.pop(key, None)
    environment.update(FREEVIDEO_HOME=config["home"], FREEVIDEO_VDN_ROOT=config["vdn_root"],
                       TRITON_CACHE_DIR=str(Path(config["home"]) / "triton"),
                       TORCHINDUCTOR_CACHE_DIR=str(Path(config["home"]) / "inductor"),
                       TORCHINDUCTOR_COMPILE_THREADS="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    command = [config["python"], "-I", "-B", "-X", "utf8", str(Path(__file__).with_name("worker.py")), str(run_dir)]
    from .supervision import owned_process
    with (run_dir / "worker.log").open("w", encoding="utf-8") as log, owned_process(command, environment, run_dir, log) as process:
        while process.poll() is None:
            if interrupt:
                interrupt()
            status = run_dir / "progress.json"
            if progress and status.is_file():
                try:
                    progress(json.loads(status.read_text(encoding="utf-8"))["completed"], tail_steps if low else 8)
                except (OSError, ValueError, KeyError):
                    pass  # An atomic progress update must not affect sampling math.
            time.sleep(0.2)
        if process.returncode:
            tail = (run_dir / "worker.log").read_text(encoding="utf-8")[-5000:]
            raise RuntimeError(f"FreeVideo {role} worker failed; evidence kept at {run_dir}\n{tail}")
    config_for(model, full=True)
    verify_files([row for row in request["loras"] if row["strength"] != 0])
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if result.get("request_sha256") != digest(run_dir / "request.json") or result.get("output_sha256") != digest(run_dir / "output.safetensors"):
        raise ValueError("FreeVideo worker result identity mismatch")
    values = load_file(str(run_dir / "output.safetensors"), device="cpu")
    receipt = dict(result, schema="t8-freevideo-stage-v1", role=role, geometry=canvas,
                   model_revision=MODEL_REVISION, freevideo_revision=FREEVIDEO_REVISION, seed=seed,
                   vdn_revision=VDN_REVISION,
                   completed_nfe=len(result["sample"]["step_seconds"]), tail_steps=tail_steps if low else None,
                   low_receipt_sha256=low.receipt_sha256 if low else None, run_directory=str(run_dir),
                   condition_transport=conditions, lora_slots=request["loras"])
    stage = make_stage(values["video"], values["audio"], receipt)
    if low is not None and tensor_record(stage.audio) != tensor_record(low.audio):
        raise ValueError("FreeVideo HIGH changed preserved LOW audio")
    return stage
