from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import uuid

from safetensors import safe_open
from safetensors.torch import save_file
import torch

import comfy.samplers
import comfy.utils
from comfy.k_diffusion.sampling import to_d

from .core import nested_av_parts
from .native_latent_checkpoint_advanced import (
    MAX_METADATA_JSON_BYTES,
    _cpu_tensor,
    _reject_symlink_components,
    _relative_parts,
    _resolved_checkpoint_root,
    _sha256_file,
)
from .native_latent_timeline_advanced import _tensor_content_digest
from .sampling import (
    DEFAULT_SAMPLER_NAME,
    DEFAULT_SCHEDULER_NAME,
    _audio_step_scale,
    _rebase_partial_audio_start,
    model_uses_raw_audio_velocity,
    setup_dual_clock_sampling,
    time_shift_sigma,
    time_shift_slope,
)


NFE_RESUME_SCHEMA = "t8.minimax_h3.nfe_resume.v1"
NFE_RESUME_METADATA_KEY = "t8_minimax_h3_nfe_resume_json"
NFE_RESUME_EXTENSION = ".h3nfe.safetensors"
NFE_RESUME_MODES = ("disabled", "checkpoint_each_step", "resume")


def _json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def _validate_model_contract_id(value: str) -> str:
    result = str(value or "").strip()
    if not result or len(result.encode("utf-8")) > 4096:
        raise ValueError(
            "model_contract_id must contain 1 to 4096 UTF-8 bytes and identify the exact "
            "base model, LoRAs/strengths and sampling patches"
        )
    return result


def _canonical_run_contract(value: str) -> tuple[str, str]:
    text = str(value or "").strip()
    if not text:
        raise ValueError(
            "run_contract_json is required for checkpoint or resume mode; connect the output "
            "of MiniMax H3 NFE Run Contract or paste another strict immutable JSON object"
        )
    if len(text.encode("utf-8")) > MAX_METADATA_JSON_BYTES:
        raise ValueError("run_contract_json exceeds the checkpoint metadata safety limit")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("run_contract_json must be valid JSON") from exc
    if not isinstance(parsed, Mapping) or not parsed:
        raise ValueError("run_contract_json must contain one non-empty JSON object")
    canonical = _json(parsed)
    return canonical, _sha256_text(canonical)


def _type_path(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _mapping_type_signature(
    value: Mapping[Any, Any],
    *,
    excluded_keys: frozenset[str] = frozenset(),
) -> dict[str, str]:
    return {
        str(key): _type_path(item)
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        if str(key) not in excluded_keys
    }


def _resolve_nfe_checkpoint_path(
    storage_root: str | Path,
    checkpoint_path: str,
    *,
    create_parent: bool,
    require_file: bool,
) -> tuple[Path, str]:
    root = _resolved_checkpoint_root(storage_root, create=create_parent)
    parts = _relative_parts(checkpoint_path, "checkpoint_path")
    if not parts[-1].endswith(NFE_RESUME_EXTENSION):
        raise ValueError(f"checkpoint_path must end with {NFE_RESUME_EXTENSION}")
    _reject_symlink_components(root, parts)
    parent = root.joinpath(*parts[:-1])
    if create_parent:
        parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise FileNotFoundError(f"NFE checkpoint parent directory does not exist: {parent}")
    resolved_parent = parent.resolve()
    if root != resolved_parent and root not in resolved_parent.parents:
        raise ValueError("checkpoint_path escaped the NFE checkpoint storage root")
    target = resolved_parent / parts[-1]
    if target.exists() and target.is_symlink():
        raise ValueError("NFE checkpoint paths cannot be symbolic links")
    resolved = target.resolve()
    if root not in resolved.parents:
        raise ValueError("checkpoint_path escaped the NFE checkpoint storage root")
    if require_file and not resolved.is_file():
        raise FileNotFoundError(f"NFE checkpoint does not exist: {resolved}")
    return resolved, resolved.relative_to(root).as_posix()


@contextmanager
def _exclusive_checkpoint_lock(target: Path):
    lock_path = target.with_name(f"{target.name}.lock")
    if lock_path.exists() and lock_path.is_symlink():
        raise ValueError("NFE checkpoint lock cannot be a symbolic link")
    handle = lock_path.open("a+b")
    locked = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                f"another execution is writing the NFE checkpoint: {target.name}"
            ) from exc
        locked = True
        yield
    finally:
        if locked:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()
        try:
            lock_path.unlink()
        except OSError:
            pass


def _runtime_signature(model: Any, extra_args: Mapping[str, Any]) -> dict[str, Any]:
    guider = getattr(model, "inner_model", None)
    patcher = getattr(guider, "model_patcher", None)
    base_model = getattr(patcher, "model", None)
    model_options = extra_args.get("model_options", {})
    if not isinstance(model_options, Mapping):
        model_options = {}
    transformer_options = model_options.get("transformer_options", {})
    if not isinstance(transformer_options, Mapping):
        transformer_options = {}
    object_patches = getattr(patcher, "object_patches", {})
    if not isinstance(object_patches, Mapping):
        object_patches = {}
    patches = getattr(patcher, "patches", {})
    if not isinstance(patches, Mapping):
        patches = {}
    weight_patch_keys = sorted(str(key) for key in patches)
    return {
        "sampler_wrapper_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "guider_class": (
            f"{type(guider).__module__}.{type(guider).__qualname__}"
            if guider is not None
            else "none"
        ),
        "base_model_class": (
            f"{type(base_model).__module__}.{type(base_model).__qualname__}"
            if base_model is not None
            else "none"
        ),
        "model_patcher_class": _type_path(patcher) if patcher is not None else "none",
        "cfg": float(getattr(guider, "cfg", 1.0)),
        "model_option_keys": sorted(str(key) for key in model_options),
        "model_option_types": _mapping_type_signature(model_options),
        "transformer_option_keys": sorted(
            str(key) for key in transformer_options if key != "sample_sigmas"
        ),
        "transformer_option_types": _mapping_type_signature(
            transformer_options,
            excluded_keys=frozenset({"sample_sigmas"}),
        ),
        "object_patch_keys": sorted(str(key) for key in object_patches),
        "object_patch_types": _mapping_type_signature(object_patches),
        "weight_patch_key_count": len(weight_patch_keys),
        "weight_patch_keys_sha256": _sha256_text(_json(weight_patch_keys)),
    }


def _tensor_manifest(
    tensors: Mapping[str, torch.Tensor], *, max_chunk_bytes: int
) -> list[dict[str, Any]]:
    result = []
    for key in sorted(tensors):
        tensor = tensors[key]
        digest, tensor_bytes = _tensor_content_digest(
            tensor,
            max_chunk_bytes=max_chunk_bytes,
        )
        result.append(
            {
                "key": key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": digest,
                "tensor_bytes": tensor_bytes,
            }
        )
    return result


def _validate_finite_tensor(name: str, value: torch.Tensor) -> None:
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"NFE checkpoint refuses non-finite values in {name}")


def _state_tensors(
    *,
    state_x: torch.Tensor,
    original_noise: torch.Tensor,
    original_latent_image: torch.Tensor,
    full_sigmas: torch.Tensor,
    denoise_mask: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    tensors = {
        "state_x": _cpu_tensor(state_x),
        "original_noise": _cpu_tensor(original_noise),
        "original_latent_image": _cpu_tensor(original_latent_image),
        "full_sigmas": _cpu_tensor(full_sigmas.to(dtype=torch.float32)),
    }
    if denoise_mask is not None:
        tensors["denoise_mask"] = _cpu_tensor(denoise_mask)
    for key, tensor in tensors.items():
        _validate_finite_tensor(key, tensor)
    return tensors


def _build_payload(
    tensors: Mapping[str, torch.Tensor],
    *,
    completed_steps: int,
    video_values: int,
    packed_values: int,
    shift_video: float,
    shift_audio: float,
    audio_velocity_is_raw: bool,
    seed: int,
    session_id: str,
    model_contract_id: str,
    run_contract_sha256: str,
    runtime_signature: Mapping[str, Any],
    max_chunk_bytes: int,
) -> dict[str, Any]:
    full_sigmas = tensors["full_sigmas"]
    total_steps = int(full_sigmas.numel() - 1)
    manifest = _tensor_manifest(tensors, max_chunk_bytes=max_chunk_bytes)
    state_content_sha256 = _sha256_text(_json(manifest))
    return {
        "schema": NFE_RESUME_SCHEMA,
        "experimental": True,
        "sampler": DEFAULT_SAMPLER_NAME,
        "scheduler": DEFAULT_SCHEDULER_NAME,
        "completed_steps": int(completed_steps),
        "total_steps": total_steps,
        "remaining_steps": total_steps - int(completed_steps),
        "resume_video_sigma": float(full_sigmas[int(completed_steps)]),
        "video_values": int(video_values),
        "packed_values": int(packed_values),
        "shift_video": float(shift_video),
        "shift_audio": float(shift_audio),
        "audio_velocity_is_raw": bool(audio_velocity_is_raw),
        "seed": int(seed),
        "session_id": session_id,
        "model_contract_id": model_contract_id,
        "model_contract_sha256": _sha256_text(model_contract_id),
        "run_contract_sha256": run_contract_sha256,
        "runtime_signature": dict(runtime_signature),
        "tensor_keys": sorted(tensors),
        "tensor_manifest": manifest,
        "state_content_sha256": state_content_sha256,
        "has_denoise_mask": "denoise_mask" in tensors,
        "pickle_used": False,
        "checkpoint_scope": (
            "packed post-step AV state, original processed noise/latent, optional packed "
            "denoise mask, and the complete native-flow sigma schedule"
        ),
        "scientific_boundary": (
            "Exact continuation is supported only for T8 dual_clock_euler/native_flow and "
            "requires the identical model, patches, conditioning, device kernels and seed. "
            "The explicit model/run contract detects operator-declared mismatches but cannot "
            "cryptographically hash all loaded model weights inside ComfyUI."
        ),
    }


def _validate_payload_and_tensors(
    payload: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    *,
    max_chunk_bytes: int,
) -> None:
    if payload.get("schema") != NFE_RESUME_SCHEMA:
        raise ValueError(f"NFE checkpoint must use schema {NFE_RESUME_SCHEMA}")
    if payload.get("pickle_used") is not False:
        raise ValueError("NFE checkpoint must declare pickle_used=false")
    declared = payload.get("tensor_keys")
    if not isinstance(declared, list) or declared != sorted(tensors):
        raise ValueError("NFE checkpoint tensor keys do not match its descriptor")
    required = {"state_x", "original_noise", "original_latent_image", "full_sigmas"}
    if not required <= tensors.keys():
        raise ValueError("NFE checkpoint is missing required sampler-state tensors")
    if bool(payload.get("has_denoise_mask")) != ("denoise_mask" in tensors):
        raise ValueError("NFE checkpoint denoise-mask declaration is inconsistent")
    state_shape = tensors["state_x"].shape
    if tensors["original_noise"].shape != state_shape:
        raise ValueError("NFE checkpoint original noise shape does not match state_x")
    if tensors["original_latent_image"].shape != state_shape:
        raise ValueError("NFE checkpoint original latent shape does not match state_x")
    if "denoise_mask" in tensors and tensors["denoise_mask"].shape != state_shape:
        raise ValueError("NFE checkpoint denoise mask shape does not match state_x")
    sigmas = tensors["full_sigmas"]
    if sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError("NFE checkpoint full_sigmas must be a one-dimensional schedule")
    if not bool((sigmas[:-1] > sigmas[1:]).all()) or not math.isclose(
        float(sigmas[-1]), 0.0, abs_tol=1.0e-8
    ):
        raise ValueError("NFE checkpoint sigma schedule must strictly decrease to zero")
    total_steps = int(sigmas.numel() - 1)
    completed_steps = payload.get("completed_steps")
    if not isinstance(completed_steps, int) or not 1 <= completed_steps <= total_steps:
        raise ValueError("NFE checkpoint completed_steps is outside its schedule")
    if payload.get("total_steps") != total_steps:
        raise ValueError("NFE checkpoint total_steps does not match full_sigmas")
    if payload.get("remaining_steps") != total_steps - completed_steps:
        raise ValueError("NFE checkpoint remaining_steps is inconsistent")
    if not math.isclose(
        float(payload.get("resume_video_sigma", float("nan"))),
        float(sigmas[completed_steps]),
        rel_tol=0.0,
        abs_tol=1.0e-7,
    ):
        raise ValueError("NFE checkpoint resume sigma is inconsistent")
    if payload.get("sampler") != DEFAULT_SAMPLER_NAME:
        raise ValueError("NFE checkpoint sampler is not dual_clock_euler")
    if payload.get("scheduler") != DEFAULT_SCHEDULER_NAME:
        raise ValueError("NFE checkpoint scheduler is not native_flow")
    if not re.fullmatch(r"[0-9a-f]{32}", str(payload.get("session_id", ""))):
        raise ValueError("NFE checkpoint session_id is invalid")
    model_contract_id = payload.get("model_contract_id")
    if not isinstance(model_contract_id, str) or not model_contract_id:
        raise ValueError("NFE checkpoint model_contract_id is missing")
    if payload.get("model_contract_sha256") != _sha256_text(model_contract_id):
        raise ValueError("NFE checkpoint model contract digest mismatch")
    if not re.fullmatch(r"[0-9A-F]{64}", str(payload.get("run_contract_sha256", ""))):
        raise ValueError("NFE checkpoint run contract digest is invalid")
    if not isinstance(payload.get("runtime_signature"), Mapping):
        raise ValueError("NFE checkpoint runtime signature is invalid")
    if not isinstance(payload.get("seed"), int):
        raise ValueError("NFE checkpoint seed is invalid")
    video_values = payload.get("video_values")
    packed_values = payload.get("packed_values")
    if (
        not isinstance(video_values, int)
        or not isinstance(packed_values, int)
        or not 0 < video_values < packed_values
        or state_shape[-1] != packed_values
    ):
        raise ValueError("NFE checkpoint packed AV layout is invalid")
    for key in ("shift_video", "shift_audio"):
        value = payload.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"NFE checkpoint {key} is invalid")
    if not isinstance(payload.get("audio_velocity_is_raw"), bool):
        raise ValueError("NFE checkpoint audio velocity protocol is invalid")
    manifest = _tensor_manifest(tensors, max_chunk_bytes=max_chunk_bytes)
    if payload.get("tensor_manifest") != manifest:
        raise ValueError("NFE checkpoint tensor content/shape/dtype digest mismatch")
    if payload.get("state_content_sha256") != _sha256_text(_json(manifest)):
        raise ValueError("NFE checkpoint state_content_sha256 mismatch")
    for key, tensor in tensors.items():
        _validate_finite_tensor(key, tensor)


def read_nfe_resume_checkpoint(
    storage_root: str | Path,
    checkpoint_path: str,
    *,
    hash_chunk_megabytes: int = 8,
) -> dict[str, Any]:
    if not 1 <= int(hash_chunk_megabytes) <= 64:
        raise ValueError("hash_chunk_megabytes must be between 1 and 64")
    target, relative = _resolve_nfe_checkpoint_path(
        storage_root,
        checkpoint_path,
        create_parent=False,
        require_file=True,
    )
    with safe_open(str(target), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        raw = metadata.get(NFE_RESUME_METADATA_KEY)
        if not raw or len(raw.encode("utf-8")) > MAX_METADATA_JSON_BYTES:
            raise ValueError("NFE checkpoint metadata is missing or oversized")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("NFE checkpoint metadata is invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("NFE checkpoint metadata must be one JSON object")
        tensors = {key: _cpu_tensor(handle.get_tensor(key)) for key in handle.keys()}
    _validate_payload_and_tensors(
        payload,
        tensors,
        max_chunk_bytes=int(hash_chunk_megabytes) * 1024 * 1024,
    )
    return {
        "path": target,
        "relative_path": relative,
        "file_sha256": _sha256_file(target),
        "payload": dict(payload),
        "tensors": tensors,
    }


def fingerprint_nfe_resume_checkpoint(
    storage_root: str | Path,
    checkpoint_path: str,
) -> str:
    target, _relative = _resolve_nfe_checkpoint_path(
        storage_root,
        checkpoint_path,
        create_parent=False,
        require_file=True,
    )
    return _sha256_file(target)


def _existing_session_id(target: Path) -> str:
    if not target.exists():
        return ""
    with safe_open(str(target), framework="pt", device="cpu") as handle:
        raw = (handle.metadata() or {}).get(NFE_RESUME_METADATA_KEY)
    if not raw:
        raise ValueError("existing target is not a T8 NFE checkpoint")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("existing NFE checkpoint metadata is invalid") from exc
    if payload.get("schema") != NFE_RESUME_SCHEMA:
        raise ValueError("existing target uses a different checkpoint schema")
    return str(payload.get("session_id", ""))


def _write_nfe_resume_checkpoint(
    target: Path,
    tensors: Mapping[str, torch.Tensor],
    payload: Mapping[str, Any],
    *,
    allow_replace_existing: bool,
    max_chunk_bytes: int,
) -> str:
    encoded = _json(payload)
    if len(encoded.encode("utf-8")) > MAX_METADATA_JSON_BYTES:
        raise ValueError("NFE checkpoint metadata exceeds the safety limit")
    session_id = str(payload["session_id"])
    with _exclusive_checkpoint_lock(target):
        existing_session = _existing_session_id(target)
        if (
            existing_session
            and existing_session != session_id
            and not allow_replace_existing
        ):
            raise FileExistsError(
                "refusing to replace an NFE checkpoint from a different session; choose a "
                "new checkpoint_path or explicitly allow replacement"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            save_file(dict(tensors), str(temporary), metadata={NFE_RESUME_METADATA_KEY: encoded})
            with temporary.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            with safe_open(str(temporary), framework="pt", device="cpu") as handle:
                written_raw = (handle.metadata() or {}).get(NFE_RESUME_METADATA_KEY)
                written_tensors = {
                    key: _cpu_tensor(handle.get_tensor(key)) for key in handle.keys()
                }
            if written_raw != encoded:
                raise RuntimeError("NFE checkpoint metadata changed during temporary write")
            _validate_payload_and_tensors(
                payload,
                written_tensors,
                max_chunk_bytes=max_chunk_bytes,
            )
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
    return _sha256_file(target)


def sample_minimax_h3_dual_clock_euler_resumable(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    *,
    video_values: int,
    packed_values: int,
    shift_video: float,
    shift_audio: float,
    audio_velocity_is_raw: bool,
    checkpoint_config: Mapping[str, Any],
):
    extra_args = {} if extra_args is None else extra_args
    if x.shape[-1] != packed_values:
        raise ValueError(
            "MiniMax H3 packed latent changed after NFE resume setup: "
            f"expected {packed_values} values, got {x.shape[-1]}"
        )
    full_sigmas = checkpoint_config["full_sigmas"].to(
        device=sigmas.device,
        dtype=sigmas.dtype,
    )
    resume_state = checkpoint_config.get("resume_state")
    global_step_offset = 0
    if resume_state is not None:
        payload = resume_state["payload"]
        tensors = resume_state["tensors"]
        global_step_offset = int(payload["completed_steps"])
        expected_sigmas = full_sigmas[global_step_offset:]
        if not torch.equal(sigmas, expected_sigmas):
            raise ValueError("resume sigmas do not match the checkpoint's remaining schedule")
        runtime_signature = _runtime_signature(model, extra_args)
        if payload.get("runtime_signature") != runtime_signature:
            raise ValueError("runtime model/guider/patch signature changed since the checkpoint")
        seed = int(extra_args.get("seed", -1))
        if seed != int(payload["seed"]):
            raise ValueError(
                f"resume seed {seed} does not match checkpoint seed {payload['seed']}"
            )
        x = tensors["state_x"].to(device=x.device, dtype=x.dtype)
        model.noise = tensors["original_noise"].to(device=x.device, dtype=x.dtype)
        model.latent_image = tensors["original_latent_image"].to(
            device=x.device,
            dtype=x.dtype,
        )
        denoise_mask = tensors.get("denoise_mask")
        if denoise_mask is not None:
            denoise_mask = denoise_mask.to(device=x.device, dtype=x.dtype)
        extra_args["denoise_mask"] = denoise_mask
        model.sigmas = full_sigmas
        model_options = extra_args.get("model_options")
        if isinstance(model_options, dict):
            transformer_options = model_options.setdefault("transformer_options", {})
            if isinstance(transformer_options, dict):
                transformer_options["sample_sigmas"] = full_sigmas
    else:
        if not torch.equal(sigmas, full_sigmas):
            raise ValueError("new NFE checkpoint run requires the complete native-flow schedule")
        x = _rebase_partial_audio_start(
            model,
            x,
            sigmas[0],
            video_values=video_values,
            shift_video=shift_video,
            shift_audio=shift_audio,
        )

    denoise_mask = extra_args.get("denoise_mask")
    audio_mask = None
    if denoise_mask is not None:
        if denoise_mask.shape[-1] != packed_values:
            raise ValueError("MiniMax H3 denoise mask does not match the packed AV latent")
        audio_mask = denoise_mask[..., video_values:]

    write_enabled = bool(checkpoint_config.get("write_enabled"))
    if write_enabled:
        original_noise = getattr(model, "noise", None)
        original_latent_image = getattr(model, "latent_image", None)
        if not isinstance(original_noise, torch.Tensor) or not isinstance(
            original_latent_image, torch.Tensor
        ):
            raise RuntimeError(
                "NFE checkpointing requires ComfyUI's KSamplerX0Inpaint noise/latent state"
            )
        if original_noise.shape != x.shape or original_latent_image.shape != x.shape:
            raise ValueError("sampler noise/latent shapes changed before NFE checkpointing")
    else:
        original_noise = None
        original_latent_image = None

    runtime_signature = _runtime_signature(model, extra_args)
    s_in = x.new_ones([x.shape[0]])
    for local_step in comfy.utils.model_trange(len(sigmas) - 1, disable=disable):
        sigma_video = sigmas[local_step]
        sigma_video_next = sigmas[local_step + 1]
        denoised = model(x, sigma_video * s_in, **extra_args)
        derivative = to_d(x, sigma_video, denoised)

        sigma_audio = time_shift_sigma(sigma_video, shift_video, shift_audio)
        sigma_audio_next = time_shift_sigma(
            sigma_video_next,
            shift_video,
            shift_audio,
        )
        slope_audio = time_shift_slope(sigma_video, shift_video, shift_audio)
        video_delta = sigma_video_next - sigma_video
        audio_delta = sigma_audio_next - sigma_audio
        if not audio_velocity_is_raw:
            audio_delta = audio_delta / slope_audio
        if audio_mask is not None:
            audio_delta = video_delta + audio_mask * (audio_delta - video_delta)

        if callback is not None:
            endpoint_scale = _audio_step_scale(
                sigma_video,
                sigma_audio,
                slope_audio,
                audio_mask,
                audio_velocity_is_raw,
            )
            denoised[..., video_values:] = (
                x[..., video_values:]
                + derivative[..., video_values:] * endpoint_scale
            )
            callback(
                {
                    "x": x,
                    "i": local_step,
                    "sigma": sigma_video,
                    "sigma_hat": sigma_video,
                    "denoised": denoised,
                }
            )

        x = torch.cat(
            (
                x[..., :video_values]
                + derivative[..., :video_values] * video_delta,
                x[..., video_values:]
                + derivative[..., video_values:] * audio_delta,
            ),
            dim=-1,
        )

        if write_enabled:
            completed_steps = global_step_offset + local_step + 1
            tensors = _state_tensors(
                state_x=x,
                original_noise=original_noise,
                original_latent_image=original_latent_image,
                full_sigmas=full_sigmas,
                denoise_mask=denoise_mask,
            )
            payload = _build_payload(
                tensors,
                completed_steps=completed_steps,
                video_values=video_values,
                packed_values=packed_values,
                shift_video=shift_video,
                shift_audio=shift_audio,
                audio_velocity_is_raw=audio_velocity_is_raw,
                seed=int(extra_args.get("seed", -1)),
                session_id=checkpoint_config["session_id"],
                model_contract_id=checkpoint_config["model_contract_id"],
                run_contract_sha256=checkpoint_config["run_contract_sha256"],
                runtime_signature=runtime_signature,
                max_chunk_bytes=checkpoint_config["max_chunk_bytes"],
            )
            _write_nfe_resume_checkpoint(
                checkpoint_config["target"],
                tensors,
                payload,
                allow_replace_existing=checkpoint_config[
                    "allow_replace_existing"
                ],
                max_chunk_bytes=checkpoint_config["max_chunk_bytes"],
            )
    return x


def setup_nfe_resume_sampling(
    model,
    av_latent: Mapping[str, Any],
    *,
    steps: int,
    shift_video: float,
    shift_audio: float,
    mode: str,
    checkpoint_path: str,
    model_contract_id: str,
    run_contract_json: str,
    confirm_checkpoint_write: bool,
    allow_replace_existing: bool,
    hash_chunk_megabytes: int,
    storage_root: str | Path,
) -> tuple[Any, Any, torch.Tensor, str, str, str]:
    if mode not in NFE_RESUME_MODES:
        raise ValueError(f"unsupported NFE resume mode: {mode!r}")
    if not 1 <= int(steps) <= 100:
        raise ValueError("steps must be between 1 and 100")
    if not math.isfinite(shift_video) or shift_video <= 0.0:
        raise ValueError("shift_video must be finite and greater than zero")
    if not math.isfinite(shift_audio) or shift_audio <= 0.0:
        raise ValueError("shift_audio must be finite and greater than zero")
    if not 1 <= int(hash_chunk_megabytes) <= 64:
        raise ValueError("hash_chunk_megabytes must be between 1 and 64")

    video, audio = nested_av_parts(dict(av_latent))
    if video.shape[1] != 24 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError("NFE resume requires one native MiniMax H3 nested AV latent")
    video_values = math.prod(video.shape[1:])
    packed_values = video_values + math.prod(audio.shape[1:])
    patched_model, _stable_sampler, full_sigmas = setup_dual_clock_sampling(
        model,
        dict(av_latent),
        int(steps),
        float(shift_video),
        float(shift_audio),
        DEFAULT_SAMPLER_NAME,
        DEFAULT_SCHEDULER_NAME,
    )
    full_sigmas = full_sigmas.to(device="cpu", dtype=torch.float32).contiguous()
    audio_velocity_is_raw = model_uses_raw_audio_velocity(model)
    max_chunk_bytes = int(hash_chunk_megabytes) * 1024 * 1024
    resume_state = None
    relative_path = ""
    target = None
    session_id = uuid.uuid4().hex
    canonical_contract = ""
    run_contract_sha256 = ""
    model_contract = ""
    status = "DISABLED"
    output_sigmas = full_sigmas

    if mode != "disabled":
        model_contract = _validate_model_contract_id(model_contract_id)
        canonical_contract, run_contract_sha256 = _canonical_run_contract(
            run_contract_json
        )
        target, relative_path = _resolve_nfe_checkpoint_path(
            storage_root,
            checkpoint_path,
            create_parent=mode == "checkpoint_each_step",
            require_file=mode == "resume",
        )

    if mode == "checkpoint_each_step":
        if not confirm_checkpoint_write:
            raise ValueError(
                "confirm_checkpoint_write must be enabled for checkpoint_each_step mode"
            )
        if target.exists() and not allow_replace_existing:
            raise FileExistsError(
                "checkpoint_path already exists; choose a new path or explicitly allow replacement"
            )
        status = "CHECKPOINT_ARMED"
    elif mode == "resume":
        resume_state = read_nfe_resume_checkpoint(
            storage_root,
            relative_path,
            hash_chunk_megabytes=int(hash_chunk_megabytes),
        )
        payload = resume_state["payload"]
        tensors = resume_state["tensors"]
        if payload["completed_steps"] >= payload["total_steps"]:
            raise ValueError("NFE checkpoint is already complete and has no remaining steps")
        if payload["total_steps"] != int(steps):
            raise ValueError("requested steps do not match the NFE checkpoint")
        if not torch.equal(tensors["full_sigmas"], full_sigmas):
            raise ValueError("native-flow sigma schedule changed since the NFE checkpoint")
        checks = {
            "video_values": video_values,
            "packed_values": packed_values,
            "shift_video": float(shift_video),
            "shift_audio": float(shift_audio),
            "audio_velocity_is_raw": audio_velocity_is_raw,
            "model_contract_id": model_contract,
            "run_contract_sha256": run_contract_sha256,
        }
        for key, expected in checks.items():
            actual = payload.get(key)
            if isinstance(expected, float):
                matches = math.isclose(
                    float(actual), expected, rel_tol=0.0, abs_tol=1.0e-8
                )
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(f"NFE checkpoint {key} does not match the current run")
        session_id = payload["session_id"]
        output_sigmas = full_sigmas[int(payload["completed_steps"]) :]
        status = "RESUME_READY_AND_WRITING" if confirm_checkpoint_write else "RESUME_READY_READ_ONLY"

    checkpoint_config = {
        "full_sigmas": full_sigmas,
        "resume_state": resume_state,
        "write_enabled": mode == "checkpoint_each_step"
        or (mode == "resume" and bool(confirm_checkpoint_write)),
        "target": target,
        "session_id": session_id,
        "model_contract_id": model_contract,
        "run_contract_sha256": run_contract_sha256,
        "max_chunk_bytes": max_chunk_bytes,
        "allow_replace_existing": bool(allow_replace_existing)
        or mode == "resume",
    }

    def sampler_function(
        model_wrap,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
    ):
        return sample_minimax_h3_dual_clock_euler_resumable(
            model_wrap,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            video_values=video_values,
            packed_values=packed_values,
            shift_video=float(shift_video),
            shift_audio=float(shift_audio),
            audio_velocity_is_raw=audio_velocity_is_raw,
            checkpoint_config=checkpoint_config,
        )

    sampler_function.__name__ = "sample_minimax_h3_dual_clock_euler_resumable"
    sampler = comfy.samplers.KSAMPLER(sampler_function)
    report = {
        "schema": NFE_RESUME_SCHEMA,
        "status": status,
        "mode": mode,
        "checkpoint_path": relative_path,
        "checkpoint_file_exists_at_setup": bool(target and target.exists()),
        "steps": int(steps),
        "remaining_steps": int(output_sigmas.numel() - 1),
        "shift_video": float(shift_video),
        "shift_audio": float(shift_audio),
        "sampler": DEFAULT_SAMPLER_NAME,
        "scheduler": DEFAULT_SCHEDULER_NAME,
        "video_values": video_values,
        "packed_values": packed_values,
        "audio_velocity_is_raw": audio_velocity_is_raw,
        "confirm_checkpoint_write": bool(confirm_checkpoint_write),
        "allow_replace_existing": bool(allow_replace_existing),
        "hash_chunk_megabytes": int(hash_chunk_megabytes),
        "model_contract_id": model_contract,
        "model_contract_sha256": _sha256_text(model_contract) if model_contract else "",
        "run_contract_sha256": run_contract_sha256,
        "canonical_run_contract": canonical_contract,
        "checkpoint_after_every_completed_step": bool(
            checkpoint_config["write_enabled"]
        ),
        "atomic_replace": bool(checkpoint_config["write_enabled"]),
        "same_path_concurrent_execution": "fail_closed",
        "files_written_during_setup": False,
        "old_sampler_or_workflow_modified": False,
        "supported_scope": "dual_clock_euler + native_flow only",
        "unsupported": [
            "ancestral/SDE RNG-state resume",
            "DPM++/multistep history resume",
            "Comfy native AV or third-party sampler resume",
            "mid-model-call resume",
            "automatic proof that external model weights match model_contract_id",
        ],
        "scientific_boundary": (
            "The last atomically completed Euler boundary can survive cancellation or process "
            "failure. Resume still requires the exact model, LoRAs, wrappers, conditioning, "
            "seed and deterministic kernels; this is not a universal sampler checkpoint."
        ),
    }
    return patched_model, sampler, output_sigmas, status, relative_path, _json(report, indent=2)
