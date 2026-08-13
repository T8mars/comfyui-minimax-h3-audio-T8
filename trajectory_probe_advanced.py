from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import comfy.model_sampling
from comfy.nested_tensor import NestedTensor
import folder_paths

from .core import nested_av_parts
from .sampling import MiniMaxH3FlowSampling


TRAJECTORY_SCHEMA = "t8.minimax_h3.trajectory_probe.v2"
CHECKPOINT_METADATA_KEY = "h3_t8_trajectory_contract"
SUPPORTED_SAMPLER_FUNCTION = "sample_minimax_h3_dual_clock_euler"


class MiniMaxH3TrajectorySampling(MiniMaxH3FlowSampling):
    """Keep nonzero-sigma checkpoints in the sampler's internal x_sigma space."""

    h3_t8_trajectory_transport = True

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        # This Advanced-only transport always starts a complete run at sigma=1,
        # where CONST would return the unscaled noise, and resumes by supplying
        # the saved internal x_sigma as ``noise``. Returning it directly avoids
        # a numerically lossy sigma*x + (1-sigma)*x reconstruction at resume.
        return noise

    def inverse_noise_scaling(self, sigma, latent):
        return latent


def prepare_trajectory_model(model: Any):
    original = model.get_model_object("model_sampling")
    if isinstance(original, MiniMaxH3TrajectorySampling):
        return model
    if not isinstance(original, MiniMaxH3FlowSampling):
        raise ValueError(
            "Trajectory Probe requires the T8 stable dual_clock_euler MODEL output"
        )
    if float(getattr(original, "noise_scale", 1.0)) != 1.0:
        raise ValueError("Trajectory Probe requires model_sampling noise_scale=1")
    sampling = MiniMaxH3TrajectorySampling(model.model.model_config)
    sampling.set_parameters(shift=float(original.shift))
    if hasattr(original, "noise_scale"):
        sampling.set_noise_scale(original.noise_scale)
    clone = model.clone()
    clone.add_object_patch("model_sampling", sampling)
    return clone


class TrajectoryResumeNoise:
    """Return the internal checkpoint x_sigma as both flow initialization terms."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def generate_noise(self, input_latent: Mapping[str, Any]):
        samples = input_latent["samples"]
        if getattr(samples, "is_nested", False):
            return NestedTensor(tuple(item.detach().clone() for item in samples.unbind()))
        return samples.detach().clone()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str)


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _sampler_identity(sampler: Any) -> dict[str, Any]:
    function = getattr(sampler, "sampler_function", None)
    name = getattr(function, "__name__", "")
    module = getattr(function, "__module__", "")
    if name != SUPPORTED_SAMPLER_FUNCTION:
        raise ValueError(
            "Trajectory Probe currently supports only T8 stable dual_clock_euler; "
            "multistep/native/unknown samplers may carry hidden history across a split"
        )
    return {
        "class": f"{type(sampler).__module__}.{type(sampler).__qualname__}",
        "function": f"{module}.{name}",
        "extra_options": getattr(sampler, "extra_options", {}),
    }


def _model_identity(model: Any) -> dict[str, Any]:
    sampling = model.get_model_object("model_sampling")
    if not isinstance(sampling, comfy.model_sampling.CONST):
        raise ValueError("Trajectory Probe requires a CONST flow sampling model")
    if not isinstance(sampling, MiniMaxH3TrajectorySampling):
        raise ValueError("use the trajectory_model output emitted by Trajectory Probe")
    options = getattr(model, "model_options", {})
    transformer = options.get("transformer_options", {}) if isinstance(options, Mapping) else {}
    wrappers = transformer.get("wrappers", {}) if isinstance(transformer, Mapping) else {}
    if wrappers:
        raise ValueError(
            "Trajectory Probe refuses model wrappers because a split may reset hidden step state"
        )
    patch_replace = transformer.get("patches_replace", {}) if isinstance(transformer, Mapping) else {}
    owners = {}
    has_patch_replace = False
    if isinstance(patch_replace, Mapping):
        for group, entries in patch_replace.items():
            if isinstance(entries, Mapping):
                values = list(entries.values())
            elif entries:
                values = [entries]
            else:
                values = []
            if values:
                has_patch_replace = True
            owners[str(group)] = sorted(
                {
                    f"{getattr(value, '__module__', type(value).__module__)}."
                    f"{getattr(value, '__qualname__', type(value).__qualname__)}"
                    for value in values
                }
            )
    elif patch_replace:
        has_patch_replace = True
        owners["unknown"] = [
            f"{type(patch_replace).__module__}.{type(patch_replace).__qualname__}"
        ]
    if has_patch_replace:
        raise ValueError(
            "Trajectory Probe refuses patches_replace because replacement blocks or attention "
            "may carry hidden state across a split"
        )
    base = getattr(model, "model", None)
    return {
        "session_exact_patches_uuid": str(getattr(model, "patches_uuid", "")),
        "base_model_class": f"{type(base).__module__}.{type(base).__qualname__}",
        "sampling_class": f"{type(sampling).__module__}.{type(sampling).__qualname__}",
        "sampling_shift": float(getattr(sampling, "shift", math.nan)),
        "sampling_audio_shift": getattr(sampling, "audio_shift", None),
        "patch_replace_owners": owners,
    }


def build_trajectory_probe(
    model: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    split_step: int,
    maximum_checkpoint_mib: float,
    av_latent: Mapping[str, Any],
    noise_seed: int = 0,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or sigmas.numel() < 3:
        raise ValueError("sigmas must be a one-dimensional schedule with at least two steps")
    if not torch.isfinite(sigmas).all().item():
        raise ValueError("sigmas contain non-finite values")
    if float(sigmas[0].detach().cpu()) != 1.0 or float(sigmas[-1].detach().cpu()) != 0.0:
        raise ValueError("Trajectory Probe requires a complete sigma schedule from 1 to 0")
    step = int(split_step)
    total_steps = int(sigmas.numel() - 1)
    if step < 1 or step >= total_steps:
        raise ValueError(f"split_step must be between 1 and {total_steps - 1}")
    video, audio = nested_av_parts(dict(av_latent))
    if av_latent.get("noise_mask") is not None:
        raise ValueError(
            "Trajectory Probe currently refuses noise_mask because exact resume would also "
            "need the original pre-split inpaint state"
        )
    checkpoint_mib = (
        video.numel() * video.element_size() + audio.numel() * audio.element_size()
    ) / (1024**2)
    limit = float(maximum_checkpoint_mib)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("maximum_checkpoint_mib must be finite and positive")
    if checkpoint_mib > limit:
        raise ValueError(
            f"trajectory checkpoint estimate {checkpoint_mib:.2f}MiB exceeds gate {limit:.2f}MiB"
        )
    sigma_values = sigmas.detach().to(device="cpu", dtype=torch.float32).contiguous()
    identity = {
        "schema": TRAJECTORY_SCHEMA,
        "model": _model_identity(model),
        "sampler": _sampler_identity(sampler),
        "sigma_sha256": _tensor_hash(sigma_values),
        "sigma_values": sigma_values.tolist(),
        "split_step": step,
        "total_steps": total_steps,
        "checkpoint_sigma": float(sigma_values[step]),
        "latent_shapes": [list(video.shape), list(audio.shape)],
        "latent_dtypes": [str(video.dtype), str(audio.dtype)],
        "estimated_checkpoint_mib": checkpoint_mib,
        "maximum_checkpoint_mib": limit,
        "noise_seed": int(noise_seed),
        "checkpoint_space": "internal_x_sigma_direct_transport",
        "resume_noise_contract": (
            "connect Trajectory Checkpoint Load resume_noise and checkpoint_latent to the "
            "second SamplerCustomAdvanced; the trajectory sampling model transports "
            "internal x_sigma without reconstruction"
        ),
        "scope": "same ComfyUI process and exact ModelPatcher patches_uuid",
        "full_run_equivalence_claim": (
            "requires real H3 split-versus-full validation for every published frame/step profile"
        ),
    }
    identity["contract_hash"] = _hash_json(identity)
    return identity, sigmas[: step + 1], sigmas[step:]


def validate_trajectory_contract(
    contract: Mapping[str, Any],
    model: Any,
    sampler: Any,
    sigmas: torch.Tensor,
) -> dict[str, Any]:
    if not isinstance(contract, Mapping) or contract.get("schema") != TRAJECTORY_SCHEMA:
        raise ValueError("trajectory contract is invalid")
    value = dict(contract)
    expected_hash = value.pop("contract_hash", None)
    if expected_hash != _hash_json(value):
        raise ValueError("trajectory contract hash is invalid")
    value["contract_hash"] = expected_hash
    if value["model"] != _model_identity(model):
        raise ValueError("trajectory checkpoint MODEL identity does not match this session")
    if value["sampler"] != _sampler_identity(sampler):
        raise ValueError("trajectory checkpoint sampler identity does not match")
    sigma_cpu = sigmas.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if _tensor_hash(sigma_cpu) != value["sigma_sha256"]:
        raise ValueError("trajectory checkpoint sigma schedule does not match")
    return value


def _checkpoint_root() -> Path:
    output = Path(folder_paths.get_output_directory()).resolve()
    root = (output / "minimax_h3_t8_trajectory_checkpoints").resolve()
    if output not in root.parents:
        raise ValueError("trajectory checkpoint root escaped the output directory")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_checkpoint(value: str) -> Path:
    path = Path(str(value or "")).resolve()
    root = _checkpoint_root()
    if root not in path.parents or path.is_symlink():
        raise ValueError("trajectory checkpoint path must stay inside its output directory")
    return path


def save_trajectory_checkpoint(
    trajectory_contract: Mapping[str, Any],
    latent: Mapping[str, Any],
    checkpoint_name: str,
    confirm_save: bool,
) -> tuple[str, dict[str, Any]]:
    if not confirm_save:
        return "", {
            "schema": TRAJECTORY_SCHEMA,
            "status": "not_saved",
            "reason": "confirm_save is false",
        }
    import re

    name = str(checkpoint_name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
        raise ValueError("checkpoint_name contains unsupported characters")
    contract = dict(trajectory_contract)
    expected_hash = contract.pop("contract_hash", None)
    if expected_hash != _hash_json(contract):
        raise ValueError("trajectory contract hash is invalid")
    contract["contract_hash"] = expected_hash
    video, audio = nested_av_parts(dict(latent))
    estimated_mib = (
        video.numel() * video.element_size() + audio.numel() * audio.element_size()
    ) / (1024**2)
    if estimated_mib > float(contract["maximum_checkpoint_mib"]):
        raise ValueError("actual checkpoint latent exceeds the planned size gate")
    if [list(video.shape), list(audio.shape)] != contract["latent_shapes"]:
        raise ValueError("checkpoint latent shape differs from the planned AV latent")
    root = _checkpoint_root()
    path = root / f"{name}.{contract['contract_hash'][:12]}.safetensors"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=root,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    tensors = {
        "video": video.detach().to(device="cpu").contiguous(),
        "audio": audio.detach().to(device="cpu").contiguous(),
        "sigmas": torch.tensor(contract["sigma_values"], dtype=torch.float32),
    }
    metadata = {
        CHECKPOINT_METADATA_KEY: json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    }
    try:
        save_file(tensors, str(temporary), metadata=metadata)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    report = {
        "schema": TRAJECTORY_SCHEMA,
        "status": "saved",
        "checkpoint_path": str(path),
        "checkpoint_sha256": _sha256_file(path),
        "contract_hash": contract["contract_hash"],
        "checkpoint_mib": path.stat().st_size / (1024**2),
        "source_latent_mutated": False,
    }
    return str(path), report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trajectory_checkpoint(
    checkpoint_path: str,
    model: Any,
    sampler: Any,
    sigmas: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
    path = _resolve_checkpoint(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"trajectory checkpoint does not exist: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        raw_contract = metadata.get(CHECKPOINT_METADATA_KEY)
        if not raw_contract:
            raise ValueError("trajectory checkpoint metadata is missing")
        try:
            contract = json.loads(raw_contract)
        except json.JSONDecodeError as error:
            raise ValueError("trajectory checkpoint metadata is invalid") from error
        validated = validate_trajectory_contract(contract, model, sampler, sigmas)
        video = handle.get_tensor("video")
        audio = handle.get_tensor("audio")
        stored_sigmas = handle.get_tensor("sigmas")
    if [list(video.shape), list(audio.shape)] != validated["latent_shapes"]:
        raise ValueError("trajectory checkpoint latent shape does not match metadata")
    expected_sigmas = sigmas.detach().to(device="cpu", dtype=torch.float32)
    if not torch.equal(stored_sigmas, expected_sigmas):
        raise ValueError("trajectory checkpoint stored sigmas do not match the workflow")
    latent = {"samples": NestedTensor((video, audio))}
    report = {
        "schema": TRAJECTORY_SCHEMA,
        "status": "loaded",
        "checkpoint_path": str(path),
        "checkpoint_sha256": _sha256_file(path),
        "contract_hash": validated["contract_hash"],
        "resume_step": validated["split_step"],
        "remaining_steps": validated["total_steps"] - validated["split_step"],
        "use_disable_noise": False,
        "resume_noise_output_required": True,
        "noise_seed": int(validated.get("noise_seed", 0)),
        "checkpoint_space": validated.get("checkpoint_space"),
        "legacy_disable_noise_contract_detected": False,
        "same_process_model_identity_verified": True,
    }
    return latent, report, sigmas[int(validated["split_step"]) :]
