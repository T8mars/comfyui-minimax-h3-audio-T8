from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .sol_engine_taehv import TAEHVLTX23Wide


OFFICIAL_STAGE2_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)
OFFICIAL_STAGE2_TAUS = (1.0, 1.25, 1.5)
OFFICIAL_REFINER_LORA_STRENGTH = 0.8
OFFICIAL_REFINER_LAYER_COUNT = 48
OFFICIAL_SOURCE_WIDTH = 864
OFFICIAL_SOURCE_HEIGHT = 480
OFFICIAL_SOURCE_FRAMES = 243
OFFICIAL_OUTPUT_WIDTH = 1920
OFFICIAL_OUTPUT_HEIGHT = 1088
OFFICIAL_OUTPUT_FRAMES = 241
TAEHV_PARALLEL_ELEMENT_LIMIT = 100_000_000


@dataclass
class T8TAEHVHandle:
    model: Any
    source_path: str


def load_taehv_wide(path: str | Path) -> T8TAEHVHandle:
    """Load the official wide architecture without filename/hash gating."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"TAEHV weight was not found: {source}")
    model = TAEHVLTX23Wide(str(source)).eval()
    return T8TAEHVHandle(model=model, source_path=str(source))


def _taehv_device_and_dtype(precision: str):
    if precision not in {"bf16_official", "fp32_reference"}:
        raise ValueError(f"unsupported TAEHV precision: {precision!r}")
    try:
        import comfy.model_management as model_management

        device = model_management.get_torch_device()
        intermediate = model_management.intermediate_device()
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        intermediate = torch.device("cpu")
    dtype = torch.bfloat16 if precision == "bf16_official" and device.type == "cuda" else torch.float32
    return device, intermediate, dtype


def _taehv_parallel(mode: str, elements: int) -> bool:
    if mode == "auto_official":
        return int(elements) < TAEHV_PARALLEL_ELEMENT_LIMIT
    if mode == "sequential_low_vram":
        return False
    if mode == "parallel_high_vram_exp":
        return True
    raise ValueError(f"unsupported TAEHV execution mode: {mode!r}")


def encode_h3_frames_with_taehv(
    frames: torch.Tensor,
    taehv: T8TAEHVHandle,
    execution_mode: str = "auto_official",
    precision: str = "bf16_official",
) -> tuple[dict[str, torch.Tensor], str]:
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise ValueError("TAEHV encode expects ComfyUI IMAGE [frames,height,width,channels]")
    if int(frames.shape[1]) % 32 or int(frames.shape[2]) % 32:
        raise ValueError("TAEHV LTX input height and width must be multiples of 32")
    device, intermediate, dtype = _taehv_device_and_dtype(precision)
    pixels = frames[..., :3].unsqueeze(0).permute(0, 1, 4, 2, 3)
    parallel = _taehv_parallel(execution_mode, pixels.numel())
    model = taehv.model.to(device=device, dtype=dtype)
    try:
        with torch.inference_mode():
            encoded = model.encode_video(
                pixels.to(device=device, dtype=dtype),
                parallel=parallel,
            )
            latent = encoded.permute(0, 2, 1, 3, 4).contiguous().to(intermediate)
    finally:
        model.to(device=torch.device("cpu"))
    report = {
        "status": "encoded",
        "codec": "taeltx2_3_wide",
        "source_path": taehv.source_path,
        "input_shape_bthwc": [1, *map(int, frames.shape)],
        "latent_shape_bcthw": list(map(int, latent.shape)),
        "execution": "parallel" if parallel else "sequential",
        "precision": str(dtype).replace("torch.", ""),
        "model_identity_policy": "selected_weight_native_load_only_no_filename_hash_or_byte_size_gate",
    }
    return {"samples": latent}, _json(report)


def decode_ltx_latent_with_taehv(
    latent: dict[str, torch.Tensor],
    taehv: T8TAEHVHandle,
    execution_mode: str = "auto_official",
    precision: str = "bf16_official",
) -> tuple[torch.Tensor, str]:
    samples = latent.get("samples")
    if not torch.is_tensor(samples) or samples.ndim != 5 or int(samples.shape[1]) != 128:
        raise ValueError("TAEHV decode expects an LTX LATENT shaped [batch,128,time,height,width]")
    device, intermediate, dtype = _taehv_device_and_dtype(precision)
    expected_frames = int(samples.shape[2]) * 8 - 7
    expected_height = int(samples.shape[3]) * 32
    expected_width = int(samples.shape[4]) * 32
    output_elements = int(samples.shape[0]) * expected_frames * 3 * expected_height * expected_width
    parallel = _taehv_parallel(execution_mode, output_elements)
    model = taehv.model.to(device=device, dtype=dtype)
    try:
        with torch.inference_mode():
            decoded = model.decode_video(
                samples.permute(0, 2, 1, 3, 4).to(device=device, dtype=dtype),
                parallel=parallel,
            )
            batch, frames, channels, height, width = decoded.shape
            images = decoded.permute(0, 1, 3, 4, 2).reshape(
                batch * frames,
                height,
                width,
                channels,
            ).contiguous().to(device=intermediate, dtype=torch.float32)
    finally:
        model.to(device=torch.device("cpu"))
    report = {
        "status": "decoded",
        "codec": "taeltx2_3_wide",
        "source_path": taehv.source_path,
        "latent_shape_bcthw": list(map(int, samples.shape)),
        "output_shape_fhwc": list(map(int, images.shape)),
        "execution": "parallel" if parallel else "sequential",
        "precision": str(dtype).replace("torch.", ""),
        "scientific_boundary": "TAEHV is NVIDIA's published fast approximate codec path, not the full LTX VAE.",
    }
    return images, _json(report)


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def ltx_8n_plus_1_frame_count(frame_count: int) -> int:
    """Return the largest valid LTX video frame count that does not add frames."""

    count = int(frame_count)
    if count < 1:
        raise ValueError("H3 draft must contain at least one frame")
    return 1 + ((count - 1) // 8) * 8


def _center_crop_resize(
    frames: torch.Tensor,
    output_height: int,
    output_width: int,
) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "frames must be a ComfyUI IMAGE batch shaped [frames, height, width, channels]"
        )
    if output_height < 1 or output_width < 1:
        raise ValueError("output dimensions must be positive")

    tensor = frames[..., :3].permute(0, 3, 1, 2)
    source_height, source_width = int(tensor.shape[-2]), int(tensor.shape[-1])
    scale = max(output_height / source_height, output_width / source_width)
    resized_height = max(output_height, int(math.ceil(source_height * scale)))
    resized_width = max(output_width, int(math.ceil(source_width * scale)))
    resized = F.interpolate(
        tensor,
        size=(resized_height, resized_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    top = (resized_height - output_height) // 2
    left = (resized_width - output_width) // 2
    cropped = resized[
        :,
        :,
        top : top + output_height,
        left : left + output_width,
    ]
    return cropped.permute(0, 2, 3, 1).contiguous().to(frames.dtype)


def prepare_h3_draft_for_ltx_refiner(
    frames: torch.Tensor,
    target_width: int,
    target_height: int,
    frame_policy: str = "trim_to_8n_plus_1",
    fps: float = 24.0,
) -> tuple[torch.Tensor, int, int, int, int, float, str]:
    """Prepare decoded H3 RGB frames for the official LTX Stage-2 handoff.

    The NVIDIA reference encodes an RGB draft at half the requested output
    geometry and then applies the learned LTX x2 latent upsampler.  This helper
    performs only the deterministic RGB/frame preparation.  The workflow keeps
    the original H3 AUDIO object on a separate wire.
    """

    width = int(target_width)
    height = int(target_height)
    if width <= 0 or height <= 0:
        raise ValueError("target width and height must be positive")
    if width % 32 or height % 32:
        raise ValueError("LTX Stage-2 target width and height must be multiples of 32")
    frame_rate = float(fps)
    if not math.isfinite(frame_rate) or frame_rate <= 0.0:
        raise ValueError("fps must be a positive finite number")
    source_frames = int(frames.shape[0])
    if frame_policy == "trim_to_8n_plus_1":
        kept_frames = ltx_8n_plus_1_frame_count(source_frames)
    elif frame_policy == "preserve_all_exp":
        kept_frames = source_frames
    else:
        raise ValueError(f"unsupported frame policy: {frame_policy!r}")

    encoder_width = width // 2
    encoder_height = height // 2
    prepared = _center_crop_resize(
        frames[:kept_frames],
        encoder_height,
        encoder_width,
    )
    dropped = source_frames - kept_frames
    exact_reference_geometry = (
        int(frames.shape[2]) == OFFICIAL_SOURCE_WIDTH
        and int(frames.shape[1]) == OFFICIAL_SOURCE_HEIGHT
        and source_frames == OFFICIAL_SOURCE_FRAMES
        and width == OFFICIAL_OUTPUT_WIDTH
        and height == OFFICIAL_OUTPUT_HEIGHT
        and kept_frames == OFFICIAL_OUTPUT_FRAMES
    )
    report = {
        "status": "prepared",
        "pipeline": "nvidia_h3_super_acceleration_stage2_handoff",
        "source": {
            "width": int(frames.shape[2]),
            "height": int(frames.shape[1]),
            "frames": source_frames,
        },
        "ltx_encoder_input": {
            "width": encoder_width,
            "height": encoder_height,
            "frames": kept_frames,
            "resize": "aspect_preserving_center_crop",
        },
        "target": {"width": width, "height": height, "frames": kept_frames},
        "fps": frame_rate,
        "output_duration_seconds": kept_frames / frame_rate,
        "dropped_tail_frames": dropped,
        "frame_policy": frame_policy,
        "audio_policy": "bypass_stage2_and_preserve_original_h3_audio_object",
        "pixel_limit_policy": "no_project_pixel_area_limit",
        "exact_nvidia_reference_geometry": exact_reference_geometry,
        "scientific_boundary": (
            "The NVIDIA benchmark fixes 864x480x243 input and 1920x1088x241 output. "
            "Other user-selected geometries use the same handoff formula but are not that benchmark."
        ),
    }
    return (
        prepared,
        encoder_width,
        encoder_height,
        kept_frames,
        dropped,
        kept_frames / frame_rate,
        _json(report),
    )


def official_stage2_sigmas(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor(OFFICIAL_STAGE2_SIGMAS, dtype=torch.float32, device=device)


def stage2_tau_for_sigma(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        sigma = float(value.flatten()[0].detach().cpu())
    elif isinstance(value, (list, tuple)):
        if not value:
            return None
        sigma = float(value[0])
    else:
        sigma = float(value)
    if not math.isfinite(sigma) or sigma <= 0.0:
        return None
    stage_sigmas = OFFICIAL_STAGE2_SIGMAS[:-1]
    nearest = min(range(len(stage_sigmas)), key=lambda index: abs(stage_sigmas[index] - sigma))
    return OFFICIAL_STAGE2_TAUS[nearest]


def find_loaded_sol_attn_backend() -> Any | None:
    """Find Kijai's already-loaded Sol-Attn module without importing by path.

    ComfyUI chooses a private package name for custom nodes, so the stable
    runtime contract is the module API rather than that generated name.  This
    deliberately performs no installation, download, or second package load.
    """

    for module in tuple(sys.modules.values()):
        if module is None or not callable(getattr(module, "make_override", None)):
            continue
        path = str(getattr(module, "__file__", "")).replace("\\", "/").lower()
        if "solattn_triton" in path or "sol-attn_triton" in path:
            return module
    return None


def _call_dense(
    previous: Callable[..., Any] | None,
    func: Callable[..., Any],
    q: Any,
    k: Any,
    v: Any,
    heads: int,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if previous is not None:
        return previous(func, q, k, v, heads, *args, **kwargs)
    return func(q, k, v, heads, *args, **kwargs)


def _make_official_stage2_override(
    backend: Any,
    previous: Callable[..., Any] | None,
    *,
    min_tokens: int,
    int8_qk: bool,
    int8_pv: bool,
    verbose: bool,
) -> Callable[..., Any]:
    delegates = {
        tau: backend.make_override(
            tau=tau,
            min_tokens=int(min_tokens),
            verbose=bool(verbose),
            int8_qk=bool(int8_qk),
            int8_pv=bool(int8_pv),
            sink_conditioning="off",
            dense_blocks=frozenset({0}),
            previous=previous,
        )
        for tau in OFFICIAL_STAGE2_TAUS
    }

    def override(func, q, k, v, heads, *args, **kwargs):
        transformer_options = kwargs.get("transformer_options")
        options = transformer_options if isinstance(transformer_options, dict) else {}
        block = options.get("sol_block")
        tau = stage2_tau_for_sigma(options.get("sigmas"))
        if block is None or int(block) == 0 or tau is None:
            return _call_dense(previous, func, q, k, v, heads, *args, **kwargs)
        return delegates[tau](func, q, k, v, heads, *args, **kwargs)

    override._t8_sol_engine_h3_super = True
    override._t8_sol_engine_previous = previous
    return override


def _transformer_blocks(diffusion_model: Any) -> Any | None:
    blocks = getattr(diffusion_model, "transformer_blocks", None)
    if blocks is not None:
        return blocks
    transformer = getattr(diffusion_model, "transformer", None)
    return getattr(transformer, "transformer_blocks", None)


def _existing_dit_replacements(model: Any) -> dict[Any, Any]:
    options = getattr(model, "model_options", {}).get("transformer_options", {})
    replacements = options.get("patches_replace", {}).get("dit", {})
    return dict(replacements) if isinstance(replacements, dict) else {}


def _block_tagger(index: int, previous_patch: Callable[..., Any] | None):
    def wrapper(args: dict[str, Any], extra_args: dict[str, Any]):
        tagged_args = dict(args)
        transformer_options = dict(tagged_args.get("transformer_options") or {})
        transformer_options["sol_block"] = index
        transformer_options["t8_sol_engine_stage2"] = True
        tagged_args["transformer_options"] = transformer_options
        if previous_patch is not None:
            return previous_patch(tagged_args, extra_args)
        return extra_args["original_block"](tagged_args)

    wrapper._t8_sol_engine_block = index
    wrapper._t8_previous_patch = previous_patch
    return wrapper


def setup_ltx_stage2_refiner(
    model: Any,
    enabled: bool = True,
    attention_backend: str = "auto_sol_attn",
    min_tokens: int = 4096,
    kernel_precision: str = "bf16_official",
    verbose: bool = False,
    *,
    sol_backend: Any | None = None,
) -> tuple[Any, torch.Tensor, float, str]:
    """Apply the NVIDIA Stage-2 schedule to an LTX model clone.

    Unsupported/missing optional Sol kernels never block execution.  The model
    is passed through with the exact 3-step schedule and a report explaining
    that attention stayed dense.
    """

    if attention_backend not in {"auto_sol_attn", "dense_reference"}:
        raise ValueError(f"unsupported attention backend: {attention_backend!r}")
    if kernel_precision not in {"bf16_official", "int8_experimental"}:
        raise ValueError(f"unsupported kernel precision: {kernel_precision!r}")

    sigmas = official_stage2_sigmas()
    if not enabled:
        return model, sigmas, OFFICIAL_REFINER_LORA_STRENGTH, _json(
            {
                "status": "disabled_passthrough",
                "stage2_sigmas": list(OFFICIAL_STAGE2_SIGMAS),
                "audio_policy": "unchanged_external_h3_audio",
            }
        )

    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except Exception as exc:
        return model, sigmas, OFFICIAL_REFINER_LORA_STRENGTH, _json(
            {
                "status": "unsupported_model_passthrough",
                "reason": f"{type(exc).__name__}: {exc}",
                "model_identity_policy": "no_filename_hash_or_size_gate",
            }
        )

    blocks = _transformer_blocks(diffusion_model)
    if blocks is None:
        return model, sigmas, OFFICIAL_REFINER_LORA_STRENGTH, _json(
            {
                "status": "unsupported_model_passthrough",
                "reason": "connected model exposes no LTX transformer_blocks",
                "model_identity_policy": "no_filename_hash_or_size_gate",
            }
        )

    cloned = model.clone()
    block_count = len(blocks)
    existing = _existing_dit_replacements(cloned)
    for index in range(block_count):
        previous_patch = existing.get(("double_block", index))
        cloned.set_model_patch_replace(
            _block_tagger(index, previous_patch),
            "dit",
            "double_block",
            index,
        )

    options = dict(cloned.model_options.get("transformer_options", {}))
    previous_override = options.get("optimized_attention_override")
    backend = sol_backend if sol_backend is not None else find_loaded_sol_attn_backend()
    use_sol = attention_backend == "auto_sol_attn" and backend is not None
    if use_sol:
        int8 = kernel_precision == "int8_experimental"
        options["optimized_attention_override"] = _make_official_stage2_override(
            backend,
            previous_override,
            min_tokens=int(min_tokens),
            int8_qk=int8,
            int8_pv=int8,
            verbose=bool(verbose),
        )
        attention_status = "sol_attn_active"
    else:
        attention_status = (
            "dense_reference_requested"
            if attention_backend == "dense_reference"
            else "dense_fallback_sol_attn_not_loaded"
        )
    options["t8_sol_engine_h3_super"] = {
        "stage2_sigmas": list(OFFICIAL_STAGE2_SIGMAS),
        "stage2_taus": list(OFFICIAL_STAGE2_TAUS),
        "dense_layers": [0],
        "sol_layers": list(range(1, block_count)),
        "audio_policy": "external_h3_audio_passthrough",
    }
    cloned.model_options["transformer_options"] = options

    report = {
        "status": "configured",
        "pipeline": "nvidia_h3_super_acceleration_ltx25_stage2",
        "attention": attention_status,
        "attention_backend_requested": attention_backend,
        "kernel_precision": kernel_precision,
        "min_tokens": int(min_tokens),
        "transformer_layers_observed": block_count,
        "official_transformer_layer_count": OFFICIAL_REFINER_LAYER_COUNT,
        "official_layer_contract_matched": block_count == OFFICIAL_REFINER_LAYER_COUNT,
        "dense_self_attention_layers": [0],
        "sol_self_attention_layers": list(range(1, block_count)) if use_sol else [],
        "cross_attention": "dense_by_shape_fallback",
        "stage2_sigmas": list(OFFICIAL_STAGE2_SIGMAS),
        "stage2_taus": list(OFFICIAL_STAGE2_TAUS),
        "sampler": "euler",
        "refiner_lora_strength": OFFICIAL_REFINER_LORA_STRENGTH,
        "audio_policy": "do_not_encode_or_denoise_h3_audio_in_ltx_stage2",
        "model_identity_policy": "no_filename_hash_byte_size_or_pixel_area_execution_gate",
        "composition": {
            "existing_dit_replacements_preserved": sum(
                1 for index in range(block_count) if ("double_block", index) in existing
            ),
            "existing_attention_override_chained": previous_override is not None and use_sol,
        },
        "scientific_boundary": (
            "NVIDIA's published 22.2x number is a fixed 4xGB200 resident benchmark. "
            "This ComfyUI adaptation preserves the algorithmic Stage-2 schedule but makes "
            "no speed, memory, or bit-exact claim on other hardware."
        ),
    }
    return cloned, sigmas, OFFICIAL_REFINER_LORA_STRENGTH, _json(report)
