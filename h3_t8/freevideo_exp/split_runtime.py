"""Typed, content-bound MID state and isolated split worker dispatch.

MID is deliberately not a FreeVideoStage: incomplete audio must never satisfy
the old completed-LOW contract or be mistaken for accepted final media.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from .runtime import (FREEVIDEO_REVISION, MODEL_REVISION, VDN_REVISION, canonical,
                      config_for, digest, geometry, conditioning_transport,
                      make_stage, tensor_record, verify_files, write_json_new)
from .split_sampler import PROFILE


@dataclass(frozen=True)
class FreeVideoMid:
    video: object  # partial clean prediction for the external learned lift
    video_state: object  # real x4, not x0
    audio: object  # real a4, not completed audio
    receipt_json: str
    receipt_sha256: str


def check_clock(value):
    import torch
    if (type(value.get("range_start")) is not int or type(value.get("range_end")) is not int
            or value["range_start"] != 0 or value["range_end"] != 4):
        raise ValueError("MID must describe the original eight-step range [0,4)")
    for name, shift in (("video_sigmas", 12.), ("audio_sigmas", 3.)):
        base = torch.linspace(1., 0., 9, dtype=torch.float32)
        expected = (shift * base / (1 + (shift - 1) * base)).tolist()
        if value.get(name) != expected:
            raise ValueError("MID has a foreign video/audio clock: " + name)
    if value.get("audio_policy") != "continue_noisy_audio_no_freeze":
        raise ValueError("MID cannot freeze unfinished audio")


def validate_mid(stage):
    if not isinstance(stage, FreeVideoMid) or hashlib.sha256(stage.receipt_json.encode()).hexdigest() != stage.receipt_sha256:
        raise ValueError("Expected an unchanged FreeVideo MID state, not completed LOW")
    value = json.loads(stage.receipt_json)
    canvas = geometry(**value["geometry"])
    vt = (canvas["frames"] - 5) // 17 * 5 + 2
    shape = (1, 24, vt, canvas["height"] // 16, canvas["width"] // 16)
    if tuple(stage.video.shape) != shape or tuple(stage.video_state.shape) != shape:
        raise ValueError("MID video shapes differ from the low canvas")
    if tuple(stage.audio.shape) != (1, 32, 2, round(canvas["frames"] / 24 * 40)):
        raise ValueError("MID audio shape differs from the actual audio clock")
    if (value.get("schema") != "t8-freevideo-mid-v1" or value.get("split_profile") != PROFILE
            or value.get("role") != "SPLIT_LOW" or type(value.get("completed_nfe")) is not int or value["completed_nfe"] != 4
            or value.get("complete_AV") is not False or value.get("model_revision") != MODEL_REVISION
            or value.get("freevideo_revision") != FREEVIDEO_REVISION or value.get("vdn_revision") != VDN_REVISION):
        raise ValueError("Invalid FreeVideo MID producer/completion identity")
    steps = value.get("sample", {}).get("step_seconds", [])
    if len(steps) != 4 or any(type(s) not in (int, float) or not 0 <= s < float("inf") for s in steps):
        raise ValueError("MID must contain four actual completed NFE")
    check_clock(value.get("split", {}))
    for name in ("request_sha256", "output_sha256", "source_inventory_sha256"):
        text = value.get(name)
        if not isinstance(text, str) or len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
            raise ValueError("MID lacks its actual producer SHA: " + name)
    for name in ("video", "video_state", "audio"):
        if tensor_record(getattr(stage, name)) != value.get(name):
            raise ValueError("MID tensor content changed: " + name)
    return value


def make_mid(video, video_state, audio, receipt):
    tensors = {name: tensor.detach().cpu().clone() for name, tensor in
               (("video", video), ("video_state", video_state), ("audio", audio))}
    value = dict(receipt, **{name: tensor_record(tensor) for name, tensor in tensors.items()})
    text = canonical(value)
    state = FreeVideoMid(**tensors, receipt_json=text, receipt_sha256=hashlib.sha256(text.encode()).hexdigest())
    validate_mid(state)
    return state


def mid_av(stage):
    from comfy.nested_tensor import NestedTensor
    validate_mid(stage)
    return {"samples": NestedTensor((stage.video.clone(), stage.audio.clone()))}


def save_mid(stage, root):
    from safetensors.torch import save_file
    value = validate_mid(stage)
    directory = Path(root).resolve() / ("mid-" + uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "latents.safetensors"
    temporary = directory / "latents.partial"
    save_file({name: getattr(stage, name).contiguous() for name in ("video", "video_state", "audio")}, str(temporary))
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    temporary.rename(path)
    marker = directory / "mid.json"
    write_json_new(marker, dict(schema="t8-freevideo-saved-mid-v1", tensors=path.name,
                                tensor_sha256=digest(path), receipt=value))
    return str(marker), digest(marker)


def load_mid(path, sha256):
    from safetensors.torch import load_file
    path = Path(path).resolve(strict=True)
    if not isinstance(sha256, str) or len(sha256) != 64 or digest(path) != sha256.lower():
        raise ValueError("Supply the actual MID manifest SHA256")
    value = json.loads(path.read_text(encoding="utf8"))
    if value.get("schema") != "t8-freevideo-saved-mid-v1" or value.get("tensors") != "latents.safetensors":
        raise ValueError("Not a FreeVideo MID manifest; completed LOW is a different contract")
    tensors = path.parent / value["tensors"]
    if tensors.is_symlink() or tensors.resolve().parent != path.parent or digest(tensors) != value["tensor_sha256"]:
        raise ValueError("MID tensor path/content changed")
    values = load_file(str(tensors), device="cpu")
    if set(values) != {"video", "video_state", "audio"}:
        raise ValueError("MID tensor keys are incomplete")
    state = make_mid(values["video"], values["video_state"], values["audio"], value["receipt"])
    if json.loads(state.receipt_json) != value["receipt"]:
        raise ValueError("MID receipt differs from saved tensors")
    return state


def sample_split(model, conditioning, width, height, frames, seed, *, mid=None, lifted=None,
                 interrupt=None, progress=None):
    from safetensors.torch import load_file, save_file
    config = config_for(model, full=True)
    canvas = geometry(width, height, frames)
    tensors, conditions = conditioning_transport(conditioning, canvas)
    from .effects import transport_effects
    effects = transport_effects(model, conditioning, canvas)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("FreeVideo split seed must fit signed int64")
    previous = None
    if mid is not None:
        from ..core import nested_av_parts
        previous = validate_mid(mid)
        video, audio = nested_av_parts(lifted)
        tensor_record(video)
        if (tuple(video.shape[:3]) != tuple(mid.video.shape[:3])
                or tuple(video.shape[-2:]) != (height // 16, width // 16)
                or tensor_record(audio) != tensor_record(mid.audio)
                or previous["geometry"]["frames"] != frames
                or width < previous["geometry"]["width"] or height < previous["geometry"]["height"]):
            raise ValueError("Split HIGH needs lifted partial x0 and unchanged noisy MID audio/frame count")
        tensors.update(initial_video=video.detach().cpu().contiguous(),
                       initial_audio=mid.audio[0].permute(1, 0, 2).contiguous())
    role = "SPLIT_HIGH" if mid is not None else "SPLIT_LOW"
    directory = Path(config["home"]) / "t8-runs" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(directory / "input.safetensors"))
    request = dict(schema="t8-freevideo-request-v1", split_profile=PROFILE, geometry=canvas, role=role,
        seed=seed, conditions=conditions, effects=effects, loras=[json.loads(row) for row in model.loras],
        config_path=model.config_path, config_sha256=model.config_sha256, input_sha256=digest(directory / "input.safetensors"),
        mid_receipt_sha256=mid.receipt_sha256 if mid else None)
    write_json_new(directory / "request.json", request)
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "FREEVIDEO_RUNTIME_LOCK_FD", "FREEVIDEO_RUNTIME_LOCK_HANDLE"):
        env.pop(key, None)
    env.update(FREEVIDEO_HOME=config["home"], FREEVIDEO_VDN_ROOT=config["vdn_root"],
        TRITON_CACHE_DIR=str(Path(config["home"]) / "triton"), TORCHINDUCTOR_CACHE_DIR=str(Path(config["home"]) / "inductor"),
        TORCHINDUCTOR_COMPILE_THREADS="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    from .supervision import owned_process
    command = [config["python"], "-I", "-B", "-X", "utf8", str(Path(__file__).with_name("worker.py")), str(directory)]
    with (directory / "worker.log").open("x", encoding="utf8") as log, owned_process(command, env, directory, log) as process:
        while process.poll() is None:
            if interrupt:
                interrupt()
            if progress and (directory / "progress.json").is_file():
                progress(json.loads((directory / "progress.json").read_text(encoding="utf8"))["completed"], 4)
            time.sleep(.2)
        if process.returncode:
            tail = (directory / "worker.log").read_text(encoding="utf8")[-5000:]
            raise RuntimeError(f"FreeVideo {role} failed; evidence at {directory}\n{tail}")
    config_for(model, full=True)
    verify_files([row for row in request["loras"] if row["strength"] != 0])
    result = json.loads((directory / "result.json").read_text(encoding="utf8"))
    if result.get("request_sha256") != digest(directory / "request.json") or result.get("output_sha256") != digest(directory / "output.safetensors"):
        raise ValueError("Split worker output identity mismatch")
    values = load_file(str(directory / "output.safetensors"), device="cpu")
    receipt = dict(result, geometry=canvas, model_revision=MODEL_REVISION, freevideo_revision=FREEVIDEO_REVISION,
        vdn_revision=VDN_REVISION, seed=seed, completed_nfe=len(result["sample"]["step_seconds"]),
        split_profile=PROFILE, run_directory=str(directory), condition_transport=conditions, lora_slots=request["loras"])
    if mid is None:
        return make_mid(values["video"], values["video_state"], values["audio"],
                        dict(receipt, schema="t8-freevideo-mid-v1", role=role, complete_AV=False))
    validate_mid(mid)
    split = result.get("split", {})
    if (split.get("range_start"), split.get("range_end"), receipt["completed_nfe"]) != (4, 8, 4):
        raise ValueError("Split HIGH did not complete original steps 4..7")
    if split.get("audio_policy") != "continue_noisy_audio_no_freeze" or any(split.get(k) != previous["split"][k] for k in ("video_sigmas", "audio_sigmas")):
        raise ValueError("Split HIGH changed the MID audio/video clock")
    if split.get("audio_resume_input") != tensor_record(mid.audio):
        raise ValueError("Split HIGH did not consume the actual noisy MID audio")
    return make_stage(values["video"], values["audio"], dict(receipt, schema="t8-freevideo-stage-v1", role="HIGH",
        tail_steps=4, mid_receipt_sha256=mid.receipt_sha256, split_total_nfe=8, complete_AV=True,
        audio_boundary_input=tensor_record(mid.audio)))
