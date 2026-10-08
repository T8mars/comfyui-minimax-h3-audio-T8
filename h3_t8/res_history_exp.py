"""Deterministic Core RES history boundaries; independent of legacy Euler resume.

Only eta=0 / cfg_pp=False on one unchanged full sigma table. A boundary contains
post-update x AND the preceding denoised prediction / sigma_down. It is not x0,
a LOW/HIGH handoff, or a claim that caller declarations fingerprint all weights.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from types import CodeType

from safetensors import safe_open
from safetensors.torch import save_file
import torch

import comfy.k_diffusion.sampling as core

from .native_latent_checkpoint_advanced import (
    MAX_METADATA_JSON_BYTES, _cpu_tensor, _relative_parts,
    _reject_symlink_components, _resolved_checkpoint_root, _sha256_file,
)
from .nfe_resume_advanced import _json, _tensor_manifest as _legacy_tensor_manifest
from .source_code_identity import executable_code_equal

SCHEMA = "t8.minimax_h3.res_history.v1"
METADATA_KEY = "t8_minimax_h3_res_history_json"
EXTENSION = ".h3res.safetensors"
MAX_FILE_BYTES = 2 * 1024**3
REQUIRED_TENSORS = frozenset({
    "state_x", "old_denoised", "old_sigma_down", "full_sigmas",
    "original_noise", "original_latent_image",
})
CORE_SOURCE_PINS = {
    "to_d": "79b063e3cb9561d840b552ddd71406e9204036cede1eafd025ea0d7b62426a01",
    "get_ancestral_step": "a1a0ae86591b89213a0e35f5d4a6c4a47e1836219628f150726c909de470d2d1",
    "res_multistep": "7e727d70ff211b356350a9101023537a9253c3147e1fc7824b5b2f4ba7d9789e",
    "sample_res_multistep": "a190687ce6c4c9d231194626e84507558c117fc3402abbdf4d8628436992f2c4",
}


@dataclass(frozen=True)
class RESHistory:
    completed_steps: int
    state_x: torch.Tensor
    old_denoised: torch.Tensor
    old_sigma_down: torch.Tensor


def source_contract() -> dict[str, str]:
    """Bind actual loaded Core implementation, not an asserted version label."""
    result = {}
    source_path = Path(core.__file__).resolve()
    compiled = compile(source_path.read_text(encoding="utf-8"),
                       inspect.unwrap(core.to_d).__code__.co_filename, "exec", dont_inherit=True)
    implementations = {code.co_name: code for code in compiled.co_consts if type(code) is CodeType}
    for name in ("res_multistep", "sample_res_multistep", "to_d", "get_ancestral_step"):
        actual = getattr(core, name)
        if not executable_code_equal(inspect.unwrap(actual).__code__, implementations[name]):
            raise ValueError("actual loaded Core RES executable differs from its pinned source")
        result[name] = hashlib.sha256(inspect.getsource(actual).encode()).hexdigest()
    if result != CORE_SOURCE_PINS:
        raise ValueError("unqualified Core RES implementation; ordinary Core samplers remain available")
    result["adapter_file"] = _sha256_file(Path(__file__))
    return result


def _manifest(tensors, *, max_chunk_bytes):
    # The legacy digest uses dtype-view directly, which is undefined for 0-D
    # tensors. Bind the original scalar shape/type and reshape only its raw-byte
    # reader here; do not change old checkpoint identity or normalize precision.
    result = []
    for name in sorted(tensors):
        value = tensors[name]
        if value.ndim:
            result.extend(_legacy_tensor_manifest({name: value}, max_chunk_bytes=max_chunk_bytes))
        else:
            digest = hashlib.sha256(_json({"kind": "tensor", "dtype": str(value.dtype),
                                          "shape": []}).encode())
            digest.update(value.detach().cpu().reshape(1).view(torch.uint8).numpy().tobytes())
            result.append({"key": name, "dtype": str(value.dtype), "shape": [],
                           "sha256": digest.hexdigest().upper(), "tensor_bytes": value.element_size()})
    return result


def validate_sigmas(sigmas: torch.Tensor) -> None:
    if (not isinstance(sigmas, torch.Tensor) or sigmas.layout != torch.strided
            or sigmas.ndim != 1 or not sigmas.is_floating_point()
            or not 2 <= sigmas.numel() <= 101):
        raise ValueError("RES requires a floating full sigma table of 1 to 100 steps")
    if (not bool(torch.isfinite(sigmas).all()) or float(sigmas[-1]) != 0.
            or not bool((sigmas[:-1] > sigmas[1:]).all())):
        raise ValueError("RES full sigmas must be finite and strictly decrease to zero")


def _finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
            or not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all())):
        raise ValueError(f"RES {name} must be a finite strided floating tensor")


def validate_history(state: RESHistory, full_sigmas: torch.Tensor) -> None:
    validate_sigmas(full_sigmas)
    step = state.completed_steps
    if isinstance(step, bool) or not isinstance(step, int) or not 1 <= step < len(full_sigmas):
        raise ValueError("RES completed step is outside the full sigma table")
    _finite_tensor(state.state_x, "post-step x")
    if state.state_x.ndim < 2 or not state.state_x.numel():
        raise ValueError("RES post-step x must have non-empty batch and sample dimensions")
    _finite_tensor(state.old_denoised, "old_denoised")
    _finite_tensor(state.old_sigma_down, "old_sigma_down")
    if (state.state_x.shape != state.old_denoised.shape
            or state.state_x.dtype != state.old_denoised.dtype
            or state.state_x.device != state.old_denoised.device):
        raise ValueError("RES old_denoised must match the post-step x shape/dtype/device")
    if (state.old_sigma_down.ndim != 0 or state.old_sigma_down.dtype != full_sigmas.dtype
            or not torch.equal(state.old_sigma_down.cpu(), full_sigmas[step].cpu())):
        raise ValueError("RES old_sigma_down is not the completed global sigma boundary")


@torch.no_grad()
def sample_res_history(model, x, sigmas, extra_args=None, callback=None, disable=None,
                       *, full_sigmas=None, resume=None, post_step=None):
    """Preserve actual Core operation order; callback remains pre-update/local i.

    post_step is independent and runs AFTER both x and RES history are advanced.
    An exception in a later model call leaves the preceding committed file intact.
    No global Core monkeypatch, stochastic RNG, or CFG++ conversion is used.
    """
    extra_args = {} if extra_args is None else extra_args
    source_contract()
    full = sigmas if full_sigmas is None else full_sigmas
    validate_sigmas(full)
    _finite_tensor(x, "initial x")
    offset = 0 if resume is None else resume.completed_steps
    if not torch.equal(sigmas, full[offset:]):
        raise ValueError("RES remaining schedule differs from the full global sigma table")
    old_denoised = old_sigma_down = None
    if resume is not None:
        validate_history(resume, full)
        if offset >= len(full) - 1:
            raise ValueError("RES checkpoint is complete; there are no remaining steps")
        if x.shape != resume.state_x.shape or x.dtype != resume.state_x.dtype:
            raise ValueError("RES runtime x shape/dtype differs from the saved boundary")
        # No cast of history to a different numerical precision is allowed.
        x = resume.state_x.to(device=x.device).clone()
        old_denoised = resume.old_denoised.to(device=x.device).clone()
        old_sigma_down = resume.old_sigma_down.to(device=full.device).clone()
    s_in = x.new_ones([x.shape[0]])
    def sigma_fn(t):
        return t.neg().exp()

    def t_fn(sigma):
        return sigma.log().neg()

    def phi1_fn(t):
        return torch.expm1(t) / t

    def phi2_fn(t):
        return (phi1_fn(t) - 1.0) / t
    for i in core.trange(len(sigmas) - 1, disable=disable):
        global_i = offset + i
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, _sigma_up = core.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=0.)
        if callback is not None:
            callback({"x": x, "i": i, "global_i": global_i, "sigma": sigmas[i],
                      "sigma_hat": sigmas[i], "denoised": denoised})
        if sigma_down == 0 or old_denoised is None:
            d = core.to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            t, t_old = t_fn(sigmas[i]), t_fn(old_sigma_down)
            t_next, t_prev = t_fn(sigma_down), t_fn(full[global_i - 1])
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)
            x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)
        old_denoised, old_sigma_down = denoised, sigma_down
        if post_step is not None:
            state = RESHistory(global_i + 1, x, old_denoised, old_sigma_down)
            validate_history(state, full)
            post_step(state)
    return x


def resolve_path(root, relative, *, create=False, require_file=True):
    root = _resolved_checkpoint_root(root, create=create)
    parts = _relative_parts(relative, "RES checkpoint_path")
    if not parts[-1].endswith(EXTENSION):
        raise ValueError(f"RES checkpoint_path must end with {EXTENSION}")
    _reject_symlink_components(root, parts)
    parent = root.joinpath(*parts[:-1])
    if create:
        parent.mkdir(parents=True, exist_ok=True)
    parent = parent.resolve()
    if parent != root and root not in parent.parents:
        raise ValueError("RES checkpoint_path escaped its storage root")
    target = parent / parts[-1]
    if target.is_symlink():
        raise ValueError("RES checkpoint cannot be a symbolic link")
    if not create and require_file and not target.is_file():
        raise FileNotFoundError("RES checkpoint does not exist")
    return target, target.relative_to(root).as_posix()


def _contract(value):
    if not isinstance(value, Mapping) or not value:
        raise ValueError("RES checkpoint needs a non-empty explicit run contract")
    encoded = _json(dict(value))
    if len(encoded.encode()) > MAX_METADATA_JSON_BYTES // 2:
        raise ValueError("RES run contract exceeds its metadata budget")
    return json.loads(encoded)


def _validate_record(payload, tensors, *, hash_chunk_bytes):
    if (not isinstance(payload, dict) or payload.get("schema") != SCHEMA
            or payload.get("eta") != 0. or payload.get("cfg_pp") is not False
            or payload.get("pickle_used") is not False
            or not isinstance(payload.get("has_mask"), bool)):
        raise ValueError("not a deterministic RES history checkpoint")
    expected = REQUIRED_TENSORS | ({"denoise_mask"} if payload.get("has_mask") else set())
    if set(tensors) != expected:
        raise ValueError("RES checkpoint tensor keys are missing or unexpected")
    full = tensors["full_sigmas"]
    state = RESHistory(payload.get("completed_steps"), tensors["state_x"],
                       tensors["old_denoised"], tensors["old_sigma_down"])
    validate_history(state, full)
    if payload.get("total_steps") != len(full) - 1:
        raise ValueError("RES total step count differs from full_sigmas")
    for name, tensor in tensors.items():
        _finite_tensor(tensor, name)
        if name not in {"old_sigma_down", "full_sigmas"}:
            if tensor.shape != state.state_x.shape or tensor.dtype != state.state_x.dtype:
                raise ValueError(f"RES {name} shape/dtype differs from state_x")
    mask = tensors.get("denoise_mask")
    if mask is not None and not bool(((mask >= 0.) & (mask <= 1.)).all()):
        raise ValueError("RES denoise mask must be within zero and one")
    actual_manifest = _manifest(tensors, max_chunk_bytes=hash_chunk_bytes)
    if payload.get("tensor_manifest") != actual_manifest:
        raise ValueError("RES checkpoint tensor content digest mismatch")
    _contract(payload.get("run_contract"))
    if "compiled_assets" in payload:
        from .res_compiled_assets import validate_compiled_assets_data
        validate_compiled_assets_data(payload["compiled_assets"])
    if payload.get("source_contract") != source_contract():
        raise ValueError("RES actual Core/adapter source changed since checkpoint creation")
    return state


def read_checkpoint(root, relative, *, expected_contract=None, hash_chunk_bytes=8 * 1024**2):
    if not 1 <= hash_chunk_bytes <= 64 * 1024**2:
        raise ValueError("RES hash chunk budget must be within 1 byte and 64 MiB")
    target, relative = resolve_path(root, relative)
    if target.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("RES checkpoint exceeds the 2 GiB file budget")
    before = _sha256_file(target)
    with safe_open(str(target), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        raw = metadata.get(METADATA_KEY, "")
        if not raw or len(raw.encode()) > MAX_METADATA_JSON_BYTES:
            raise ValueError("RES checkpoint metadata is missing or oversized")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            raise ValueError("not a deterministic RES history checkpoint")
        if "compiled_assets" in payload:
            from .res_compiled_assets import validate_compiled_assets_data
            validate_compiled_assets_data(payload["compiled_assets"])
        expected = REQUIRED_TENSORS | ({"denoise_mask"} if payload.get("has_mask") else set())
        if set(handle.keys()) != expected:
            raise ValueError("RES checkpoint tensor keys are missing or unexpected")
        # Budget and exact member set are checked before loading any tensor.
        tensors = {name: handle.get_tensor(name) for name in sorted(expected)}
    state = _validate_record(payload, tensors, hash_chunk_bytes=hash_chunk_bytes)
    if expected_contract is not None and payload["run_contract"] != _contract(expected_contract):
        raise ValueError("RES model/run/conditioning contract changed")
    if _sha256_file(target) != before:
        raise ValueError("RES checkpoint changed while it was being read")
    return {"path": target, "relative_path": relative, "file_sha256": before,
            "payload": payload, "tensors": tensors, "history": state}


def save_checkpoint(root, relative, state, full_sigmas, *, original_noise,
                    original_latent_image, denoise_mask=None, run_contract,
                    hash_chunk_bytes=8 * 1024**2, compiled_assets=None):
    """Create-only atomic publish. Never replace an existing boundary or source."""
    if not 1 <= hash_chunk_bytes <= 64 * 1024**2:
        raise ValueError("RES hash chunk budget must be within 1 byte and 64 MiB")
    target, relative = resolve_path(root, relative, create=True)
    if target.exists():
        raise FileExistsError("RES checkpoint already exists; choose a new filename")
    tensors = {"state_x": _cpu_tensor(state.state_x),
               "old_denoised": _cpu_tensor(state.old_denoised),
               "old_sigma_down": _cpu_tensor(state.old_sigma_down),
               "full_sigmas": _cpu_tensor(full_sigmas),
               "original_noise": _cpu_tensor(original_noise),
               "original_latent_image": _cpu_tensor(original_latent_image)}
    if denoise_mask is not None:
        tensors["denoise_mask"] = _cpu_tensor(denoise_mask)
    if sum(t.numel() * t.element_size() for t in tensors.values()) > MAX_FILE_BYTES - MAX_METADATA_JSON_BYTES:
        raise ValueError("RES checkpoint exceeds the 2 GiB file budget")
    payload = {"schema": SCHEMA, "eta": 0., "cfg_pp": False, "pickle_used": False,
               "completed_steps": state.completed_steps, "total_steps": len(full_sigmas) - 1,
        "has_mask": denoise_mask is not None, "run_contract": _contract(run_contract),
               "source_contract": source_contract(),
               "tensor_manifest": _manifest(tensors, max_chunk_bytes=hash_chunk_bytes)}
    if compiled_assets is not None:
        from .res_compiled_assets import validate_compiled_assets_data
        payload["compiled_assets"] = validate_compiled_assets_data(compiled_assets)
    _validate_record(payload, tensors, hash_chunk_bytes=hash_chunk_bytes)
    encoded = _json(payload)
    if len(encoded.encode()) > MAX_METADATA_JSON_BYTES:
        raise ValueError("RES checkpoint metadata exceeds its budget")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".res-", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(tensors, str(temporary), metadata={METADATA_KEY: encoded})
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        with safe_open(str(temporary), framework="pt", device="cpu") as handle:
            written = {name: handle.get_tensor(name) for name in handle.keys()}
            if (handle.metadata() or {}).get(METADATA_KEY) != encoded:
                raise ValueError("RES temporary metadata changed")
        _validate_record(payload, written, hash_chunk_bytes=hash_chunk_bytes)
        # Same-directory hard link publishes atomically WITHOUT overwriting any
        # other writer. Unsupported filesystems fail; there is no unsafe fallback.
        _reject_symlink_components(Path(root).resolve(), _relative_parts(relative, "RES checkpoint_path"))
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return read_checkpoint(root, relative, expected_contract=run_contract,
                           hash_chunk_bytes=hash_chunk_bytes)
