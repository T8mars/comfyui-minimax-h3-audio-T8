"""Native ComfyUI runtime for Alibaba PAI MiniMax-H3 PDD 8-step adapters.

The released adapters are not ordinary LoRAs.  In addition to the backbone
adapters they contain 32 absolute video heads and 32 absolute audio heads.
Each of the eight model evaluations consumes the weighted mean of four
consecutive heads.  The math follows the Apache-2.0 reference implementation
published with ``alibaba-pai/MiniMax-H3-Acc-LoRAs`` at revision
``78db175437ee05df7ec492ee366f01b68b8d20e6``.

Unlike a regular ComfyUI LoRA loader, this module keeps the LoRA residual
dynamic.  That matches the upstream ``base(x) + LoRA(x)`` implementation and
does not merge the residual into an INT8 ConvRot base only to quantize it
again.
"""

from __future__ import annotations

import json
import inspect
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

import comfy.lora
import comfy.lora_convert
import comfy.patcher_extension
import comfy.utils
import comfy.weight_adapter

from .sampling import native_flow_sigmas, setup_dual_clock_sampling


PDD_NUM_STEPS = 32
PDD_BLOCK_SIZE = 4
PDD_NFE = 8
PDD_SHIFT_VIDEO = 12.0
PDD_SHIFT_AUDIO = 3.0
PDD_SAMPLER = "euler"
PDD_SCHEDULER = "simple"
PDD_VARIANTS = ("FL2VA", "Ref2VA")
PDD_WRAPPER_KEY = "t8_minimax_h3_pdd"
PDD_INJECTION_KEY = "t8_minimax_h3_pdd_lora"
PDD_ATTACHMENT_KEY = "t8_minimax_h3_pdd_contract"

PDD_HEAD_SPECS: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "pdd.final_layer.video_out.weight": ((32, 96, 5376), torch.bfloat16),
    "pdd.final_layer.video_out.bias": ((32, 96), torch.bfloat16),
    "pdd.final_layer.audio_out.weight": ((32, 32, 5376), torch.bfloat16),
    "pdd.final_layer.audio_out.bias": ((32, 32), torch.bfloat16),
}
PDD_REQUIRED_METADATA = {
    "adapter_type": "MiniMax-H3-PDD",
    "base_model": "MiniMax-H3",
    "pdd_num_steps": "32",
    "pdd_block_size": "4",
    "pdd_requires_dynamic_head": "true",
    "sampler_steps": "8",
    "sigma_shift_video": "12.0",
    "sigma_shift_audio": "3.0",
    "sampler": "euler",
    "scheduler": "simple",
    "comfyui_loader": "MiniMax H3 PDD Loader",
}


def probe_native_pdd_core(diffusion=None) -> dict[str, Any]:
    """Probe the semantics added by ComfyUI PR #15908 without version gates."""
    probe_weight = torch.zeros((3, 2))
    probe_bias = torch.zeros((3,))
    details: dict[str, Any] = {
        "set_weight_loaded": False,
        "set_bias_loaded": False,
        "shape_changing_set": False,
        "final_layer_schedule_args": False,
    }

    try:
        loaded = comfy.lora.load_lora(
            {
                "probe.set_weight": probe_weight,
                "probe.set_bias": probe_bias,
            },
            {"probe": "diffusion_model.final_layer.video_out.weight"},
            log_missing=False,
        )
        details["set_weight_loaded"] = (
            "diffusion_model.final_layer.video_out.weight" in loaded
        )
        details["set_bias_loaded"] = (
            "diffusion_model.final_layer.video_out.bias" in loaded
        )

        patches = [(1.0, ("set", (probe_weight,)), 1.0, None, None)]
        shape = comfy.lora.calculate_shape(
            patches,
            torch.zeros((1, 2)),
            "t8_pdd_semantic_probe",
        )
        details["shape_changing_set"] = tuple(shape) == tuple(probe_weight.shape)
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
        details["patch_probe_error"] = f"{type(error).__name__}: {error}"

    final_layer = getattr(diffusion, "final_layer", None)
    forward = getattr(final_layer, "forward", None)
    if callable(forward):
        try:
            parameters = set(inspect.signature(forward).parameters)
            details["final_layer_schedule_args"] = {
                "sigma",
                "sample_sigmas",
                "shifts",
            }.issubset(parameters)
            details["final_layer_parameters"] = sorted(parameters)
        except (TypeError, ValueError) as error:
            details["final_layer_probe_error"] = f"{type(error).__name__}: {error}"

    details["available"] = all(
        bool(details[key])
        for key in (
            "set_weight_loaded",
            "set_bias_loaded",
            "shape_changing_set",
            "final_layer_schedule_args",
        )
    )
    details["policy"] = "semantic_capability_probe_no_version_or_hash_gate"
    return details


def _native_pdd_lora_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    lora_state = {
        key: value for key, value in state.items() if key.startswith("diffusion_model.")
    }
    converted = dict(comfy.lora_convert.convert_lora(lora_state))
    for source, target in (
        (
            "pdd.final_layer.video_out.weight",
            "diffusion_model.final_layer.video_out.set_weight",
        ),
        (
            "pdd.final_layer.video_out.bias",
            "diffusion_model.final_layer.video_out.set_bias",
        ),
        (
            "pdd.final_layer.audio_out.weight",
            "diffusion_model.final_layer.audio_out.set_weight",
        ),
        (
            "pdd.final_layer.audio_out.bias",
            "diffusion_model.final_layer.audio_out.set_bias",
        ),
    ):
        tensor = state[source]
        if target.endswith(".set_weight"):
            tensor = tensor.reshape(-1, tensor.shape[-1])
        else:
            tensor = tensor.reshape(-1)
        converted[target] = tensor.contiguous()
    return converted


def _apply_native_pdd_lora(model, state: dict[str, torch.Tensor], strength: float):
    converted = _native_pdd_lora_state(state)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(converted, key_map, log_missing=False)
    adapter_count = sum(
        key.endswith(".lora_A.weight") for key in state if key.startswith("diffusion_model.")
    )
    expected_targets = adapter_count + 4
    if len(loaded) != expected_targets:
        raise ValueError(
            "MiniMax H3 PDD native loading did not map every backbone adapter and "
            f"head tensor: expected {expected_targets} targets, mapped {len(loaded)}."
        )

    patched = model.clone()
    applied = set(patched.add_patches(loaded, float(strength)))
    if applied != set(loaded):
        missing = sorted(str(key) for key in set(loaded) - applied)
        raise ValueError(
            "MiniMax H3 PDD native loading could not apply all mapped targets: "
            + ", ".join(missing[:8])
        )
    return patched, len(loaded), adapter_count


def shifted_sigma(shift: float, sigma: torch.Tensor) -> torch.Tensor:
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def base_sigma(shift: float, sigma: float) -> float:
    denominator = shift + sigma * (1.0 - shift)
    if denominator <= 0.0:
        raise ValueError(f"invalid shifted sigma {sigma} for shift {shift}")
    return sigma / denominator


def pdd_time_grid(shift: float, num_steps: int = PDD_NUM_STEPS) -> torch.Tensor:
    base = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - shifted_sigma(float(shift), base)


def pdd_plan(
    shift: float,
    block_index: int,
    num_steps: int = PDD_NUM_STEPS,
    block_size: int = PDD_BLOCK_SIZE,
) -> torch.Tensor:
    blocks = int(num_steps) // int(block_size)
    if block_index < 0 or block_index >= blocks:
        raise ValueError(f"PDD block index out of range: {block_index}")
    step_sizes = pdd_time_grid(float(shift), int(num_steps)).diff()
    start = int(block_index) * int(block_size)
    selected = step_sizes[start : start + int(block_size)]
    plan = torch.zeros(int(num_steps), dtype=torch.float64)
    plan[start : start + int(block_size)] = selected / selected.sum()
    return plan


def pdd_runtime_sigmas() -> torch.Tensor:
    """The eight shifted sigma inputs on which the released heads are defined."""
    base = torch.linspace(1.0, 0.0, PDD_NFE + 1, dtype=torch.float64)
    return shifted_sigma(PDD_SHIFT_VIDEO, base)


def pdd_block_index(shifted_video_sigma: float) -> int:
    sigma = min(1.0, max(0.0, float(shifted_video_sigma)))
    unshifted = base_sigma(PDD_SHIFT_VIDEO, sigma)
    progress = (1.0 - unshifted) * PDD_NUM_STEPS / PDD_BLOCK_SIZE
    return min(PDD_NFE - 1, max(0, int(round(progress))))


def validate_pdd_sigmas(sigmas: torch.Tensor) -> dict[str, Any]:
    values = torch.as_tensor(sigmas, dtype=torch.float64).flatten().cpu()
    expected = pdd_runtime_sigmas()
    if values.numel() != expected.numel():
        raise ValueError(
            "MiniMax H3 PDD requires exactly 8 model evaluations and a terminal "
            f"sigma (9 values); received {values.numel()} values."
        )
    max_error = float(torch.max(torch.abs(values - expected)))
    if max_error > 5e-6:
        raise ValueError(
            "MiniMax H3 PDD requires the official Euler/simple 8-step sigma "
            f"schedule with video shift 12; maximum error was {max_error:.8g}."
        )
    return {
        "sigma_count": int(values.numel()),
        "nfe": PDD_NFE,
        "max_abs_error": max_error,
        "block_indices": [pdd_block_index(float(value)) for value in values[:-1]],
        "sigmas": [float(value) for value in values],
    }


def _dtype_name(dtype: object) -> str:
    value = str(dtype).upper()
    value = value.removeprefix("TORCH.")
    # safetensors reports compact names (BF16/F32) while torch reports
    # bfloat16/float32.  Normalize both representations before comparing the
    # converted-file contract.
    return {
        "BFLOAT16": "BF16",
        "FLOAT16": "F16",
        "FLOAT32": "F32",
        "FLOAT64": "F64",
    }.get(value, value)


def inspect_pdd_adapter(path: str | Path, expected_variant: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_variant not in PDD_VARIANTS:
        raise ValueError(
            f"Unknown PDD base variant {expected_variant!r}; choose FL2VA or Ref2VA."
        )
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        keys = set(handle.keys())
        metadata_matches = {
            key: metadata.get(key) == expected
            for key, expected in PDD_REQUIRED_METADATA.items()
        }
        actual_variant = metadata.get("base_variant")
        head_contract = {}
        for key, (shape, dtype) in PDD_HEAD_SPECS.items():
            if key not in keys:
                head_contract[key] = {"present": False, "reference_match": False}
                continue
            tensor = handle.get_slice(key)
            actual_shape = tuple(int(value) for value in tensor.get_shape())
            actual_dtype = _dtype_name(tensor.get_dtype())
            head_contract[key] = {
                "present": True,
                "shape": list(actual_shape),
                "dtype": actual_dtype,
                "reference_match": (
                    actual_shape == shape and actual_dtype == _dtype_name(dtype)
                ),
            }
    lora_keys = keys - set(PDD_HEAD_SPECS)
    a_keys = {key for key in lora_keys if key.endswith(".lora_A.weight")}
    expected_lora_keys = {
        key.removesuffix(".lora_A.weight") + suffix
        for key in a_keys
        for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha")
    }
    unexpected = sorted(lora_keys - expected_lora_keys)
    missing = sorted(expected_lora_keys - lora_keys)
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "base_variant": actual_variant,
        "selected_base_variant": expected_variant,
        "base_variant_reference_match": actual_variant == expected_variant,
        "tensor_count": len(keys),
        "adapter_count": len(a_keys),
        "head_tensor_count": len(PDD_HEAD_SPECS),
        "metadata": metadata,
        "metadata_reference_match": metadata_matches,
        "head_contract": head_contract,
        "unexpected_lora_keys": unexpected,
        "missing_lora_keys": missing,
        "model_identity_policy": "diagnostic_only_not_a_load_gate",
    }


def _fuse_head_bank(
    weight_bank: torch.Tensor,
    bias_bank: torch.Tensor,
    shift: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    plans = torch.stack(
        [pdd_plan(float(shift), index) for index in range(PDD_NFE)]
    ).to(dtype=weight_bank.dtype, device=weight_bank.device)
    weight = torch.einsum("bn,noi->boi", plans, weight_bank)
    bias = torch.einsum("bn,no->bo", plans, bias_bank)
    return weight.contiguous(), bias.contiguous()


class PDDHeadFinalLayer(nn.Module):
    """Wrap H3's final layer with the eight released PDD block heads."""

    def __init__(
        self,
        base_layer: nn.Module,
        video_weight_bank: torch.Tensor,
        video_bias_bank: torch.Tensor,
        audio_weight_bank: torch.Tensor,
        audio_bias_bank: torch.Tensor,
        *,
        strength: float = 1.0,
        variant: str,
    ):
        super().__init__()
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("PDD strength must be in [0, 1].")
        if variant not in PDD_VARIANTS:
            raise ValueError(f"Unknown PDD variant {variant!r}.")
        for attribute in ("norm", "adaln_proj", "video_out", "audio_out"):
            if not hasattr(base_layer, attribute):
                raise TypeError(f"MiniMax-H3 final layer lacks {attribute!r}.")
        for name, tensor, (shape, dtype) in (
            ("video_weight", video_weight_bank, PDD_HEAD_SPECS["pdd.final_layer.video_out.weight"]),
            ("video_bias", video_bias_bank, PDD_HEAD_SPECS["pdd.final_layer.video_out.bias"]),
            ("audio_weight", audio_weight_bank, PDD_HEAD_SPECS["pdd.final_layer.audio_out.weight"]),
            ("audio_bias", audio_bias_bank, PDD_HEAD_SPECS["pdd.final_layer.audio_out.bias"]),
        ):
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(
                    f"{name}: expected {_dtype_name(dtype)} {shape}, found "
                    f"{_dtype_name(tensor.dtype)} {tuple(tensor.shape)}."
                )
        video_weight, video_bias = _fuse_head_bank(
            video_weight_bank, video_bias_bank, PDD_SHIFT_VIDEO
        )
        audio_weight, audio_bias = _fuse_head_bank(
            audio_weight_bank, audio_bias_bank, PDD_SHIFT_AUDIO
        )
        self.base = base_layer
        self.strength = float(strength)
        self.variant = variant
        self.register_buffer("video_weight", video_weight, persistent=False)
        self.register_buffer("video_bias", video_bias, persistent=False)
        self.register_buffer("audio_weight", audio_weight, persistent=False)
        self.register_buffer("audio_bias", audio_bias, persistent=False)
        self._block_index = 0
        self._selection_count = 0

    def move_head_buffers_to(self, device) -> None:
        """Move only the PDD heads without moving the native final layer."""

        target = torch.device(device)
        for name in ("video_weight", "video_bias", "audio_weight", "audio_bias"):
            self._buffers[name] = self._buffers[name].to(device=target)

    @property
    def block_index(self) -> int:
        return self._block_index

    @property
    def selection_count(self) -> int:
        return self._selection_count

    def select_for_sigma(
        self,
        sigma_video: float,
        shift_video: float = PDD_SHIFT_VIDEO,
        shift_audio: float = PDD_SHIFT_AUDIO,
    ) -> int:
        if not math.isclose(float(shift_video), PDD_SHIFT_VIDEO, abs_tol=1e-6):
            raise ValueError(
                f"MiniMax-H3 PDD requires video shift {PDD_SHIFT_VIDEO}, "
                f"got {shift_video}."
            )
        if not math.isclose(float(shift_audio), PDD_SHIFT_AUDIO, abs_tol=1e-6):
            raise ValueError(
                f"MiniMax-H3 PDD requires audio shift {PDD_SHIFT_AUDIO}, "
                f"got {shift_audio}."
            )
        expected = pdd_runtime_sigmas()[:-1]
        value = torch.tensor(float(sigma_video), dtype=torch.float64)
        distance, index = torch.min(torch.abs(expected - value), dim=0)
        if float(distance) > 5e-5:
            raise ValueError(
                "MiniMax-H3 PDD received a sigma outside its official 8-step "
                f"Euler/simple grid: sigma={float(value):.9g}, "
                f"nearest_error={float(distance):.8g}."
            )
        self._block_index = int(index)
        self._selection_count += 1
        return self._block_index

    @staticmethod
    def _linear(
        hidden: torch.Tensor,
        weight_bank: torch.Tensor,
        bias_bank: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        weight = weight_bank[index].to(device=hidden.device)
        bias = bias_bank[index].to(device=hidden.device)
        return F.linear(hidden.to(weight.dtype), weight, bias)

    def forward(self, x, t_emb, video_seg, audio_seg):
        shift, scale = self.base.adaln_proj(t_emb)

        def mod(segment):
            start, stop, row = segment
            return (
                self.base.norm(x[start:stop])
                * (1.0 + scale[row].to(scale.dtype))
                + shift[row].to(shift.dtype)
            )

        video_hidden = mod(video_seg)
        audio_hidden = mod(audio_seg)
        pdd_video = self._linear(
            video_hidden, self.video_weight, self.video_bias, self._block_index
        )
        pdd_audio = self._linear(
            audio_hidden, self.audio_weight, self.audio_bias, self._block_index
        )
        if self.strength == 1.0:
            return pdd_video, pdd_audio
        base_video = self.base.video_out(video_hidden.to(torch.float32))
        base_audio = self.base.audio_out(audio_hidden.to(torch.float32))
        if self.strength == 0.0:
            return base_video, base_audio
        return (
            torch.lerp(base_video, pdd_video.to(base_video.dtype), self.strength),
            torch.lerp(base_audio, pdd_audio.to(base_audio.dtype), self.strength),
        )


def _transformer_options(args, kwargs) -> dict:
    options = kwargs.get("transformer_options")
    if isinstance(options, dict) and "minimax_h3_pdd_final" in options:
        return options
    for value in reversed(args):
        if isinstance(value, dict) and "minimax_h3_pdd_final" in value:
            return value
    raise RuntimeError("MiniMax-H3 PDD could not locate transformer_options.")


def pdd_forward_wrapper(executor, *args, **kwargs):
    options = _transformer_options(args, kwargs)
    final_layer: PDDHeadFinalLayer = options["minimax_h3_pdd_final"]
    timestep = kwargs.get("timestep")
    if timestep is None:
        if len(args) < 2:
            raise RuntimeError("MiniMax-H3 PDD could not locate the model timestep.")
        timestep = args[1]
    sigma_video = float(timestep.flatten()[0].detach().cpu()) / 1000.0
    final_layer.select_for_sigma(
        sigma_video,
        float(options.get("minimax_h3_sigma_shift_video", PDD_SHIFT_VIDEO)),
        float(options.get("minimax_h3_sigma_shift_audio", PDD_SHIFT_AUDIO)),
    )
    return executor(*args, **kwargs)


def _assert_clean_lora_stack(model) -> None:
    if model.get_attachment(PDD_ATTACHMENT_KEY) is not None:
        raise ValueError("A MiniMax-H3 PDD adapter is already attached.")
    if model.get_wrappers(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, PDD_WRAPPER_KEY
    ):
        raise ValueError("A MiniMax-H3 PDD diffusion wrapper is already attached.")
    existing_patches = getattr(model, "patches", {})
    if existing_patches:
        raise ValueError(
            "MiniMax-H3 PDD must be the only weight adapter on the base MODEL. "
            f"Found {len(existing_patches)} existing weight patch targets."
        )
    existing_injections = getattr(model, "injections", {})
    conflicting = sorted(
        key for key in existing_injections if "lora" in str(key).lower()
    )
    if conflicting:
        raise ValueError(
            "MiniMax-H3 PDD must be the only LoRA injection on the base MODEL; "
            f"found {conflicting}."
        )


def _apply_dynamic_lora(model, state: dict[str, torch.Tensor], strength: float):
    lora_state = {
        key: value for key, value in state.items() if key.startswith("diffusion_model.")
    }
    converted = comfy.lora_convert.convert_lora(lora_state)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(converted, key_map, log_missing=False)
    rank_counts: dict[int, int] = {}
    for key, adapter in loaded.items():
        up, down, alpha, mid, dora_scale, reshape = adapter.weights
        del mid, dora_scale, reshape
        rank = int(up.shape[1])
        del alpha, down
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    patched = model.clone()
    manager = comfy.weight_adapter.BypassInjectionManager()
    for key, adapter in loaded.items():
        manager.add_adapter(key, adapter, strength=float(strength))
    injections = _create_offloading_bypass_injections(
        manager,
        patched.model,
        tuple(loaded.values()),
    )
    hook_count = int(manager.get_hook_count())
    patched.set_injections(PDD_INJECTION_KEY, injections)
    return patched, hook_count, rank_counts


def _move_adapter_weights_to_device(adapters, device) -> None:
    """Move bypass-only adapter tensors away from CUDA after model ejection.

    Current ComfyUI's ``BypassInjectionManager`` moves inference-adapter tensors
    to the compute device during injection, but its ejection callback only
    restores the original module forwards. PDD carries roughly 1.6 GB of
    adapter tensors, so leaving those tensors on CUDA while the cached MODEL is
    ejected can retain a material amount of VRAM. Keep this lifecycle local to
    the new PDD node instead of changing ComfyUI core behavior for every bypass
    user.
    """

    target = torch.device(device)
    for adapter in adapters:
        if isinstance(adapter, nn.Module):
            adapter.to(device=target)
            continue
        weights = getattr(adapter, "weights", None)
        if weights is None:
            continue
        if isinstance(weights, torch.Tensor):
            adapter.weights = weights.to(device=target)
            continue
        if isinstance(weights, (list, tuple)):
            moved = [
                value.to(device=target)
                if isinstance(value, torch.Tensor)
                else value
                for value in weights
            ]
            adapter.weights = tuple(moved) if isinstance(weights, tuple) else moved


def _create_offloading_bypass_injections(manager, model, adapters):
    """Wrap ComfyUI bypass injection with failure-safe adapter offloading."""

    native = manager.create_injections(model)
    if len(native) != 1:
        raise RuntimeError(
            "MiniMax-H3 PDD expected one ComfyUI bypass injection manager, "
            f"found {len(native)}."
        )
    native_injection = native[0]
    adapters = tuple(adapters)

    def offload(model_patcher):
        device = getattr(model_patcher, "offload_device", torch.device("cpu"))
        _move_adapter_weights_to_device(adapters, device)

    def inject(model_patcher):
        try:
            native_injection.inject(model_patcher)
        except BaseException:
            # Injection may fail after only part of the 258 hooks moved to CUDA.
            # Restore every hook and return all adapter tensors before reraising.
            try:
                native_injection.eject(model_patcher)
            finally:
                offload(model_patcher)
            raise

    def eject(model_patcher):
        try:
            native_injection.eject(model_patcher)
        finally:
            offload(model_patcher)

    return [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)]


def _create_pdd_final_layer_injection(base_final, pdd_final):
    """Patch only ``forward`` so native parameter paths remain stable.

    Replacing ``diffusion_model.final_layer`` as an object changes native paths
    such as ``final_layer.video_out.weight`` into ``final_layer.base.video_out``.
    ComfyUI's dynamic loader records those temporary paths and restores the
    native object before restoring its weight backups, making cleanup fail with
    ``FinalLayer has no attribute base``. A lifecycle injection preserves the
    native module tree and is safe for dynamic unload after sampling errors.
    """

    state = {"original_forward": None}

    def offload(model_patcher):
        device = getattr(model_patcher, "offload_device", torch.device("cpu"))
        pdd_final.move_head_buffers_to(device)

    def inject(model_patcher):
        if state["original_forward"] is not None:
            return
        device = getattr(model_patcher, "load_device", torch.device("cpu"))
        try:
            pdd_final.move_head_buffers_to(device)
            original_forward = base_final.forward
            base_final.forward = pdd_final.forward
            state["original_forward"] = original_forward
        except BaseException:
            offload(model_patcher)
            raise

    def eject(model_patcher):
        try:
            original_forward = state["original_forward"]
            if original_forward is not None:
                base_final.forward = original_forward
                state["original_forward"] = None
        finally:
            offload(model_patcher)

    return comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)


def _create_pdd_runtime_injection(backbone_injection, base_final, pdd_final):
    """Combine final-head and LoRA lifecycle into one transactional injection."""

    final_injection = _create_pdd_final_layer_injection(base_final, pdd_final)

    def inject(model_patcher):
        final_injection.inject(model_patcher)
        try:
            backbone_injection.inject(model_patcher)
        except BaseException:
            final_injection.eject(model_patcher)
            raise

    def eject(model_patcher):
        try:
            backbone_injection.eject(model_patcher)
        finally:
            final_injection.eject(model_patcher)

    return comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)


def build_pdd_8step_setup(
    model,
    av_latent: dict,
    pdd_lora_path: str | Path,
    *,
    base_variant: str,
    strength: float = 1.0,
):
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("PDD strength must be in [0, 1].")
    contract = inspect_pdd_adapter(pdd_lora_path, base_variant)
    diffusion = model.get_model_object("diffusion_model")
    if diffusion.__class__.__name__ != "MiniMaxH3Model":
        raise TypeError(
            "MiniMax H3 PDD requires ComfyUI's native MiniMaxH3Model MODEL."
        )
    if hasattr(diffusion, "adaln_t_table"):
        raise ValueError(
            "MiniMax H3 PDD does not support pruned/AdaLN-curve bases. Select "
            "the matching full non-pruned FL2VA or Ref2VA model."
        )
    adaln = diffusion.blocks[0].adaln_proj.linear.weight
    if int(adaln.shape[1]) != 2688:
        raise ValueError(
            "MiniMax H3 PDD requires a full non-pruned base with AdaLN input "
            f"2688; the selected MODEL uses {int(adaln.shape[1])}."
        )
    _assert_clean_lora_stack(model)

    state, metadata = comfy.utils.load_torch_file(
        str(pdd_lora_path), safe_load=True, return_metadata=True
    )
    native_capability = probe_native_pdd_core(diffusion)
    use_native_core = bool(native_capability["available"] and strength == 1.0)
    if use_native_core:
        patched, native_patch_targets, mapped_adapters = _apply_native_pdd_lora(
            model, state, strength
        )
        hook_count = 0
        rank_counts: dict[int, int] = {}
        application_mode = "comfyui_native_pdd_set_weight_set_bias"
        base_weight_mutation = True
        final_paths_preserved = True
    else:
        patched, hook_count, rank_counts = _apply_dynamic_lora(model, state, strength)
        mapped_adapters = sum(rank_counts.values())
        native_patch_targets = 0
        base_final = patched.get_model_object("diffusion_model.final_layer")
        if isinstance(base_final, PDDHeadFinalLayer):
            raise ValueError("A MiniMax-H3 PDD final layer is already attached.")
        pdd_final = PDDHeadFinalLayer(
            base_final,
            state["pdd.final_layer.video_out.weight"],
            state["pdd.final_layer.video_out.bias"],
            state["pdd.final_layer.audio_out.weight"],
            state["pdd.final_layer.audio_out.bias"],
            strength=strength,
            variant=base_variant,
        )
        backbone_injections = list(patched.get_injections(PDD_INJECTION_KEY) or [])
        if len(backbone_injections) != 1:
            raise RuntimeError(
                "MiniMax-H3 PDD expected one prepared backbone injection, "
                f"found {len(backbone_injections)}."
            )
        patched.set_injections(
            PDD_INJECTION_KEY,
            [
                _create_pdd_runtime_injection(
                    backbone_injections[0], base_final, pdd_final
                )
            ],
        )
        transformer_options = dict(
            patched.model_options.get("transformer_options", {})
        )
        transformer_options.update(
            {
                "minimax_h3_pdd_final": pdd_final,
                "minimax_h3_sigma_shift_video": PDD_SHIFT_VIDEO,
                "minimax_h3_sigma_shift_audio": PDD_SHIFT_AUDIO,
            }
        )
        patched.model_options["transformer_options"] = transformer_options
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            PDD_WRAPPER_KEY,
            pdd_forward_wrapper,
        )
        application_mode = (
            "comfyui_dynamic_model_only_bypass_plus_final_forward_injection"
        )
        base_weight_mutation = False
        final_paths_preserved = True
    if metadata and hasattr(patched, "set_attachments"):
        patched.set_attachments("t8_minimax_h3_pdd_lora_metadata", dict(metadata))

    sampled_model, sampler, sigmas = setup_dual_clock_sampling(
        patched,
        av_latent,
        PDD_NFE,
        PDD_SHIFT_VIDEO,
        PDD_SHIFT_AUDIO,
        PDD_SAMPLER,
        PDD_SCHEDULER,
    )
    schedule = validate_pdd_sigmas(sigmas)
    expected_native = native_flow_sigmas(PDD_NFE, PDD_SHIFT_VIDEO).to(torch.float64)
    schedule["native_flow_max_abs_error"] = float(
        torch.max(torch.abs(torch.as_tensor(sigmas, dtype=torch.float64) - expected_native))
    )
    report = {
        "schema": "t8_minimax_h3_pdd_8step_setup_v2",
        "status": "ready_for_real_render_validation",
        "adapter": {key: value for key, value in contract.items() if key != "metadata"},
        "base": {
            "variant_declared_by_user": base_variant,
            "native_model_class": diffusion.__class__.__name__,
            "adaln_input_width": int(adaln.shape[1]),
            "pruned_curve": False,
            "variant_identity_limit": (
                "Native MODEL does not retain the diffusion filename; the node proves "
                "the adapter variant but the workflow/user must select the matching base."
            ),
        },
        "lora": {
            "application_mode": application_mode,
            "mapped_adapters": mapped_adapters,
            "bypass_hooks": hook_count,
            "rank_counts": {str(key): value for key, value in rank_counts.items()},
            "strength": strength,
            "base_weight_mutation": base_weight_mutation,
            "native_patch_targets": native_patch_targets,
            "eject_policy": (
                "comfyui_model_patcher"
                if use_native_core
                else "move_adapter_weights_to_model_offload_device"
            ),
            "partial_injection_failure_cleanup": not use_native_core,
            "native_final_layer_parameter_paths_preserved": final_paths_preserved,
        },
        "pdd_heads": {
            "source_intervals": PDD_NUM_STEPS,
            "block_size": PDD_BLOCK_SIZE,
            "runtime_heads_per_modality": PDD_NFE,
            "fixed_video_shift": PDD_SHIFT_VIDEO,
            "fixed_audio_shift": PDD_SHIFT_AUDIO,
        },
        "sampling": {
            "sampler": PDD_SAMPLER,
            "scheduler": PDD_SCHEDULER,
            **schedule,
        },
        "compatibility": {
            "native_core_probe": native_capability,
            "native_core_used": use_native_core,
            "native_core_bypass_reason": (
                None
                if use_native_core
                else (
                    "strength_below_one_requires_t8_head_blending"
                    if native_capability["available"] and strength != 1.0
                    else "native_pdd_semantics_not_available"
                )
            ),
            "full_non_pruned_only": True,
            "pruned_supported": False,
            "ordinary_lora_loader_supported": bool(native_capability["available"]),
            "additional_lora_stack_supported": False,
            "real_fl2va_render_validated": False,
            "real_ref2va_render_validated": False,
        },
        "source": {
            "repository": "https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs",
            "revision": "78db175437ee05df7ec492ee366f01b68b8d20e6",
            "reference_license": "Apache-2.0",
        },
    }
    sampled_model.set_attachments(PDD_ATTACHMENT_KEY, report)
    return sampled_model, sampler, sigmas, json.dumps(
        report, ensure_ascii=False, sort_keys=True
    )
