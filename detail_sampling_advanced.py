from __future__ import annotations

import math
from typing import Any

import comfy.model_patcher
import comfy.samplers
from comfy.ldm.minimax.model import MiniMaxH3Model
import torch
import torch.nn.functional as torch_functional

from .core import nested_av_parts, split_noise_masks
from .motion_quality_advanced import (
    _inverse_shift_sigma,
    _schedule_sha,
    _validate_h3_sigmas,
    canonical_json,
)
from .sampling import (
    DEFAULT_SAMPLER_NAME,
    DEFAULT_SCHEDULER_NAME,
    model_uses_raw_audio_velocity,
    sample_minimax_h3_dual_clock_euler,
    setup_dual_clock_sampling,
    shift_sigma,
    time_shift_sigma,
)


H3_DOUBLE_BLOCK_COUNT = 50
_TAIL_SPACING = {"video_sigma_linear", "video_sigma_cosine", "base_flow_linear"}
_TIME_BIAS_DOMAINS = {"video_sigma", "base_flow"}


def build_tail_detail_schedule(
    sigmas: torch.Tensor,
    *,
    extra_tail_steps: int,
    spacing: str,
    shift_video: float,
    shift_audio: float,
    profile: str,
) -> tuple[torch.Tensor, int, str]:
    """Subdivide only the final H3 interval; connection of the node is the opt-in."""
    values = _validate_h3_sigmas(sigmas, profile)
    if extra_tail_steps < 0 or extra_tail_steps > 8:
        raise ValueError("extra_tail_steps must be between 0 and 8")
    if spacing not in _TAIL_SPACING:
        raise ValueError(f"unsupported tail spacing: {spacing!r}")
    if not math.isfinite(shift_video) or shift_video <= 0.0:
        raise ValueError("shift_video must be finite and greater than zero")
    if not math.isfinite(shift_audio) or shift_audio <= 0.0:
        raise ValueError("shift_audio must be finite and greater than zero")

    if extra_tail_steps == 0:
        output = sigmas
        inserted_cpu = values.new_empty((0,))
    else:
        fractions = torch.arange(
            1,
            extra_tail_steps + 1,
            dtype=torch.float64,
        ) / (extra_tail_steps + 1)
        start_video = values[-2]
        if spacing == "video_sigma_cosine":
            fractions = 0.5 * (1.0 - torch.cos(math.pi * fractions))
            inserted_cpu = start_video * (1.0 - fractions)
        elif spacing == "video_sigma_linear":
            inserted_cpu = start_video * (1.0 - fractions)
        else:
            start_base = _inverse_shift_sigma(values[-2:-1], shift_video)[0]
            inserted_base = start_base * (1.0 - fractions)
            inserted_cpu = shift_sigma(inserted_base, shift_video)
        inserted = inserted_cpu.to(device=sigmas.device, dtype=sigmas.dtype)
        output = torch.cat((sigmas[:-1], inserted, sigmas[-1:]))

    output_values = _validate_h3_sigmas(output, "custom_strict")
    audio_values = time_shift_sigma(output_values, shift_video, shift_audio)
    original_nfe = int(values.numel() - 1)
    actual_nfe = int(output_values.numel() - 1)
    report = {
        "schema": "minimax_h3_tail_detail_schedule_t8_v1",
        "status": "applied_exp" if extra_tail_steps else "noop",
        "applied": bool(extra_tail_steps),
        "spacing": spacing,
        "requested_extra_tail_steps": int(extra_tail_steps),
        "original_nfe": original_nfe,
        "actual_nfe": actual_nfe,
        "extra_joint_av_forwards": int(extra_tail_steps),
        "estimated_forward_time_increase_percent": (
            100.0 * extra_tail_steps / original_nfe
        ),
        "shift_video": float(shift_video),
        "shift_audio": float(shift_audio),
        "profile": profile,
        "input_schedule_sha256": _schedule_sha(values),
        "output_schedule_sha256": _schedule_sha(output_values),
        "video_sigmas": [float(value) for value in output_values],
        "audio_sigmas": [float(value) for value in audio_values],
        "inserted_video_sigmas": [float(value) for value in inserted_cpu],
        "inserted_audio_sigmas": [
            float(value)
            for value in time_shift_sigma(inserted_cpu, shift_video, shift_audio)
        ],
        "all_original_knots_preserved": True,
        "audio_sigmas_are_parameter_projection": True,
        "runtime_upstream_shift_contract_verified": False,
        "final_zero_is_endpoint_not_model_call": True,
        "joint_av_notice": (
            "Every inserted point is one full H3 joint audio-video forward. This is "
            "schedule subdivision, not random re-noising and not a sharpening filter."
        ),
        "turbo_intermediate_times_are_experimental": profile.startswith("turbo_"),
        "quality_validated": False,
        "memory_safe_claim": False,
    }
    return output, actual_nfe, canonical_json(report, indent=2)


def model_time_bias_sigma(
    sigma: torch.Tensor,
    *,
    bias: float,
    start_progress: float,
    end_progress: float,
    shift_video: float,
    domain: str,
) -> torch.Tensor:
    """Smoothly bias only the sigma seen by H3; the integrator keeps actual sigma."""
    if not isinstance(sigma, torch.Tensor):
        raise TypeError("sigma must be a torch.Tensor")
    if not math.isfinite(bias) or not -0.5 <= bias <= 0.0:
        raise ValueError("bias must be finite and between -0.5 and 0.0")
    if not (
        math.isfinite(start_progress)
        and math.isfinite(end_progress)
        and 0.0 <= start_progress < end_progress <= 1.0
    ):
        raise ValueError("progress range must satisfy 0 <= start < end <= 1")
    if domain not in _TIME_BIAS_DOMAINS:
        raise ValueError(f"unsupported model-time bias domain: {domain!r}")
    if not math.isfinite(shift_video) or shift_video <= 0.0:
        raise ValueError("shift_video must be finite and greater than zero")

    denominator = shift_video + sigma * (1.0 - shift_video)
    base_sigma = sigma / denominator
    progress = 1.0 - base_sigma
    u = ((progress - start_progress) / (end_progress - start_progress)).clamp(0.0, 1.0)
    active = ((progress >= start_progress) & (progress <= end_progress)).to(sigma.dtype)
    envelope = torch.sin(math.pi * u).square() * active
    factor = 1.0 + bias * envelope

    if domain == "video_sigma":
        return (sigma * factor).clamp(0.0, 1.0)
    biased_base = (base_sigma * factor).clamp(0.0, 1.0)
    return shift_sigma(biased_base, shift_video)


def setup_model_time_bias_sampling(
    model,
    av_latent: dict,
    *,
    steps: int,
    shift_video: float,
    shift_audio: float,
    bias: float,
    start_progress: float,
    end_progress: float,
    bias_domain: str,
) -> tuple[Any, Any, torch.Tensor, str]:
    patched_model, sampler, sigmas = setup_dual_clock_sampling(
        model,
        av_latent,
        steps,
        shift_video,
        shift_audio,
        DEFAULT_SAMPLER_NAME,
        DEFAULT_SCHEDULER_NAME,
    )
    expected = model_time_bias_sigma(
        sigmas.to(dtype=torch.float64),
        bias=bias,
        start_progress=start_progress,
        end_progress=end_progress,
        shift_video=shift_video,
        domain=bias_domain,
    )
    applied = not math.isclose(bias, 0.0, rel_tol=0.0, abs_tol=1e-12)
    if applied:
        if patched_model.model_options.get("model_function_wrapper") is not None:
            raise ValueError(
                "model-time bias refuses to overwrite an existing model_function_wrapper"
            )

        def model_function_wrapper(apply_model, options):
            biased_sigma = model_time_bias_sigma(
                options["timestep"],
                bias=bias,
                start_progress=start_progress,
                end_progress=end_progress,
                shift_video=shift_video,
                domain=bias_domain,
            )
            return apply_model(options["input"], biased_sigma, **options["c"])

        patched_model.set_model_unet_function_wrapper(model_function_wrapper)

    report = {
        "schema": "minimax_h3_model_time_bias_sampler_t8_v1",
        "status": "applied_exp" if applied else "noop",
        "applied": applied,
        "steps": int(steps),
        "actual_nfe": int(sigmas.numel() - 1),
        "bias": float(bias),
        "bias_domain": bias_domain,
        "start_progress": float(start_progress),
        "end_progress": float(end_progress),
        "shift_video": float(shift_video),
        "shift_audio": float(shift_audio),
        "actual_video_sigmas": [float(value) for value in sigmas],
        "model_visible_video_call_sigmas": [float(value) for value in expected[:-1]],
        "model_visible_audio_sigmas": [
            float(value)
            for value in time_shift_sigma(expected[:-1], shift_video, shift_audio)
        ],
        "final_zero_is_endpoint_not_model_call": True,
        "integrator_schedule_unchanged": True,
        "random_noise_injected": False,
        "joint_av_notice": (
            "H3 derives both audio and video model times inside one Transformer. The bias "
            "therefore changes the joint AV prediction while integration remains on the "
            "original dual-clock schedule."
        ),
        "quality_validated": False,
        "memory_safe_claim": False,
    }
    return patched_model, sampler, sigmas, canonical_json(report, indent=2)


class _RestartKSAMPLER(comfy.samplers.KSAMPLER):
    """KSAMPLER with honest progress accounting for hidden RF restart calls."""

    def __init__(self, sampler_function, *, total_steps: int):
        super().__init__(sampler_function)
        self._reported_total_steps = int(total_steps)

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        extra_args = dict(extra_args)
        extra_args["denoise_mask"] = denoise_mask
        model_k = comfy.samplers.KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        model_k.noise = noise
        noise = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0],
            noise,
            latent_image,
            self.max_denoise(model_wrap, sigmas),
        )

        def k_callback(args):
            if callback is not None:
                callback(
                    args["i"],
                    args["denoised"],
                    args["x"],
                    self._reported_total_steps,
                )

        samples = self.sampler_function(
            model_k,
            noise,
            sigmas,
            extra_args=extra_args,
            callback=k_callback,
            disable=disable_pbar,
            **self.extra_options,
        )
        return model_wrap.inner_model.model_sampling.inverse_noise_scaling(
            sigmas[-1], samples
        )


def setup_rectified_flow_restart_sampling(
    model,
    av_latent: dict,
    *,
    steps: int,
    shift_video: float,
    shift_audio: float,
    restart_video_sigma: float,
    restart_steps: int,
    restart_seed: int,
) -> tuple[Any, Any, torch.Tensor, str]:
    if not math.isfinite(restart_video_sigma) or not 0.0 <= restart_video_sigma <= 0.5:
        raise ValueError("restart_video_sigma must be finite and between 0 and 0.5")
    if restart_steps < 0 or restart_steps > 8:
        raise ValueError("restart_steps must be between 0 and 8")
    if restart_seed < 0 or restart_seed > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("restart_seed must be a uint64 value")

    patched_model, _stable_sampler, sigmas = setup_dual_clock_sampling(
        model,
        av_latent,
        steps,
        shift_video,
        shift_audio,
        DEFAULT_SAMPLER_NAME,
        DEFAULT_SCHEDULER_NAME,
    )
    video, audio = nested_av_parts(av_latent)
    video_mask, audio_mask = split_noise_masks(av_latent, video, audio)
    video_values = math.prod(video.shape[1:])
    packed_values = video_values + math.prod(audio.shape[1:])
    audio_velocity_is_raw = model_uses_raw_audio_velocity(model)
    applied = restart_steps > 0 and restart_video_sigma > 0.0

    def validate_mask(mask, name: str, *, require_all_active: bool) -> dict[str, Any]:
        if mask is None:
            return {
                "present": False,
                "active_fraction": 1.0,
                "all_active": True,
            }
        if not torch.isfinite(mask).all():
            raise ValueError(f"RF Restart {name} mask contains NaN or Inf")
        binary = mask.round()
        if not torch.allclose(mask, binary, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                f"RF Restart requires a binary {name} mask; fractional denoise masks "
                "would mismatch restart state and model time"
            )
        active_fraction = float(binary.to(dtype=torch.float32).mean().cpu())
        all_active = bool(torch.all(binary == 1).cpu())
        if require_all_active and not all_active:
            raise ValueError(
                "RF Restart requires the complete audio latent to participate; locked or "
                "partially masked audio would silently turn joint AV restart into video-only restart"
            )
        if active_fraction <= 0.0:
            raise ValueError(f"RF Restart requires at least one active {name} latent value")
        return {
            "present": True,
            "active_fraction": active_fraction,
            "all_active": all_active,
        }

    if applied:
        video_mask_contract = validate_mask(
            video_mask,
            "video",
            require_all_active=False,
        )
        audio_mask_contract = validate_mask(
            audio_mask,
            "audio",
            require_all_active=True,
        )
    else:
        video_mask_contract = {"present": video_mask is not None, "not_validated_noop": True}
        audio_mask_contract = {"present": audio_mask is not None, "not_validated_noop": True}
    restart_audio_sigma = float(
        time_shift_sigma(
            torch.tensor(restart_video_sigma, dtype=torch.float64),
            shift_video,
            shift_audio,
        )
    )
    total_nfe = int(sigmas.numel() - 1) + (restart_steps if applied else 0)

    def sampler_function(model_wrap, x, input_sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        base_nfe = int(input_sigmas.numel() - 1)
        x = sample_minimax_h3_dual_clock_euler(
            model_wrap,
            x,
            input_sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            video_values=video_values,
            packed_values=packed_values,
            shift_video=shift_video,
            shift_audio=shift_audio,
            audio_velocity_is_raw=audio_velocity_is_raw,
        )
        if not applied:
            return x

        generator = torch.Generator(device=x.device)
        generator.manual_seed(int(restart_seed))
        restarted = torch.empty_like(x)
        restarted.normal_(generator=generator)
        restarted[..., :video_values].mul_(restart_video_sigma).add_(
            x[..., :video_values], alpha=1.0 - restart_video_sigma
        )
        restarted[..., video_values:].mul_(restart_audio_sigma).add_(
            x[..., video_values:], alpha=1.0 - restart_audio_sigma
        )
        denoise_mask = extra_args.get("denoise_mask")
        if denoise_mask is not None:
            if denoise_mask.shape[-1] != packed_values:
                raise ValueError("MiniMax H3 restart mask does not match the packed AV latent")
            validate_mask(
                denoise_mask[..., :video_values],
                "packed video",
                require_all_active=False,
            )
            validate_mask(
                denoise_mask[..., video_values:],
                "packed audio",
                require_all_active=True,
            )
            restarted.sub_(x).mul_(denoise_mask).add_(x)

        restart_sigmas = torch.linspace(
            restart_video_sigma,
            0.0,
            restart_steps + 1,
            dtype=input_sigmas.dtype,
            device=input_sigmas.device,
        )

        def restart_callback(args):
            if callback is None:
                return
            forwarded = dict(args)
            forwarded["i"] = base_nfe + int(args["i"])
            callback(forwarded)

        previous_sigmas = getattr(model_wrap, "sigmas", None)
        model_wrap.sigmas = restart_sigmas
        try:
            return sample_minimax_h3_dual_clock_euler(
                model_wrap,
                restarted,
                restart_sigmas,
                extra_args=extra_args,
                callback=restart_callback,
                disable=disable,
                video_values=video_values,
                packed_values=packed_values,
                shift_video=shift_video,
                shift_audio=shift_audio,
                audio_velocity_is_raw=audio_velocity_is_raw,
            )
        finally:
            model_wrap.sigmas = previous_sigmas

    sampler_function.__name__ = "sample_minimax_h3_joint_av_rectified_flow_restart"
    sampler = _RestartKSAMPLER(sampler_function, total_steps=total_nfe)
    restart_video_sigmas = (
        torch.linspace(restart_video_sigma, 0.0, restart_steps + 1).tolist()
        if applied
        else []
    )
    report = {
        "schema": "minimax_h3_rectified_flow_restart_sampler_t8_v1",
        "status": "applied_exp" if applied else "noop",
        "applied": applied,
        "base_nfe": int(sigmas.numel() - 1),
        "restart_nfe": int(restart_steps if applied else 0),
        "actual_total_nfe": total_nfe,
        "restart_seed": int(restart_seed),
        "restart_video_sigma": float(restart_video_sigma),
        "restart_audio_sigma": restart_audio_sigma,
        "restart_video_sigmas": restart_video_sigmas,
        "restart_audio_sigmas": [
            float(value)
            for value in time_shift_sigma(
                torch.tensor(restart_video_sigmas, dtype=torch.float64),
                shift_video,
                shift_audio,
            )
        ],
        "restart_equation": "x_sigma=(1-sigma)*x_clean+sigma*epsilon",
        "restart_modalities": "joint_audio_video",
        "video_mask_contract": video_mask_contract,
        "audio_mask_contract": audio_mask_contract,
        "conditioned_binary_video_rows_preserved": True,
        "single_transformer_joint_distribution": True,
        "video_only_restart_supported": False,
        "quality_validated": False,
        "memory_safe_claim": False,
    }
    return patched_model, sampler, sigmas, canonical_json(report, indent=2)


def _parse_h3_blocks(value: str) -> list[int]:
    try:
        blocks = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise ValueError("double_blocks must be comma-separated integers") from exc
    if not blocks:
        raise ValueError("double_blocks must select at least one H3 block")
    invalid = [block for block in blocks if block < 0 or block >= H3_DOUBLE_BLOCK_COUNT]
    if invalid:
        raise ValueError(f"H3 double block indices must be between 0 and 49: {invalid}")
    return blocks


def apply_h3_spatiotemporal_guidance(
    model,
    *,
    scale: float,
    double_blocks: str,
    start_progress: float,
    end_progress: float,
    shift_video: float,
    rescale: float,
) -> tuple[Any, str]:
    if not math.isfinite(scale) or not 0.0 <= scale <= 5.0:
        raise ValueError("scale must be finite and between 0 and 5")
    if not math.isfinite(rescale) or rescale != 0.0:
        raise ValueError(
            "H3 STG currently requires rescale=0; a shared AV global-std rescale is not validated"
        )
    if not (
        math.isfinite(start_progress)
        and math.isfinite(end_progress)
        and 0.0 <= start_progress < end_progress <= 1.0
    ):
        raise ValueError("progress range must satisfy 0 <= start < end <= 1")
    if not math.isfinite(shift_video) or shift_video <= 0.0:
        raise ValueError("shift_video must be finite and greater than zero")

    blocks = _parse_h3_blocks(double_blocks)
    applied = scale > 0.0
    if not applied:
        report = {
            "schema": "minimax_h3_spatiotemporal_guidance_t8_v1",
            "status": "noop",
            "applied": False,
            "scale": 0.0,
            "double_blocks": blocks,
            "extra_joint_av_forward_per_active_step": 0,
        }
        return model, canonical_json(report, indent=2)

    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    if not isinstance(diffusion_model, MiniMaxH3Model):
        raise ValueError("H3 STG requires a native ComfyUI MiniMax H3 diffusion MODEL")

    transformer_options = model.model_options.get("transformer_options", {})
    replacements = transformer_options.get("patches_replace", {}).get("dit", {})
    conflicts = [block for block in blocks if ("double_block", block) in replacements]
    if conflicts:
        raise ValueError(
            "H3 STG refuses to overwrite existing double-block replacements: "
            + ", ".join(map(str, conflicts))
        )

    def skip_block(args, _extra_args):
        return args

    def post_cfg_function(args):
        sigma = args["sigma"]
        denominator = shift_video + sigma * (1.0 - shift_video)
        progress = 1.0 - sigma / denominator
        progress_scalar = float(progress.flatten()[0].detach().cpu())
        if not start_progress <= progress_scalar <= end_progress:
            return args["denoised"]

        cond = args.get("cond")
        if cond is None:
            raise RuntimeError("H3 STG requires a positive conditioning branch")
        runtime_replacements = (
            args["model_options"]
            .get("transformer_options", {})
            .get("patches_replace", {})
            .get("dit", {})
        )
        runtime_conflicts = [
            block
            for block in blocks
            if ("double_block", block) in runtime_replacements
        ]
        if runtime_conflicts:
            raise RuntimeError(
                "H3 STG refuses runtime double-block replacement conflicts: "
                + ", ".join(map(str, runtime_conflicts))
            )
        stg_options = comfy.model_patcher.create_model_options_clone(
            args["model_options"]
        )
        for block in blocks:
            stg_options = comfy.model_patcher.set_model_options_patch_replace(
                stg_options,
                skip_block,
                "dit",
                "double_block",
                block,
            )
        (weak_prediction,) = comfy.samplers.calc_cond_batch(
            args["model"],
            [cond],
            args["input"],
            sigma,
            stg_options,
        )
        result = args["denoised"] + (
            args["cond_denoised"] - weak_prediction
        ) * scale
        return result

    patched_model = model.clone()
    patched_model.set_model_sampler_post_cfg_function(post_cfg_function)
    report = {
        "schema": "minimax_h3_spatiotemporal_guidance_t8_v1",
        "status": "applied_exp",
        "applied": True,
        "scale": float(scale),
        "double_blocks": blocks,
        "start_progress": float(start_progress),
        "end_progress": float(end_progress),
        "shift_video": float(shift_video),
        "rescale": float(rescale),
        "extra_joint_av_forward_per_active_step": 1,
        "joint_av_notice": (
            "H3 STG perturbs the shared packed audio-video Transformer. It is not a "
            "face repair node and can change both picture and sound."
        ),
        "quality_validated": False,
        "memory_safe_claim": False,
    }
    return patched_model, canonical_json(report, indent=2)


def _gaussian_kernel(radius: int, sigma: float, device, dtype) -> torch.Tensor:
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _blur_luma(luma: torch.Tensor, radius: int, sigma: float) -> torch.Tensor:
    if radius == 0:
        return luma
    kernel = _gaussian_kernel(radius, sigma, luma.device, luma.dtype)
    horizontal = kernel.view(1, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1)
    mode = "reflect" if luma.shape[-1] > radius and luma.shape[-2] > radius else "replicate"
    out = torch_functional.pad(luma, (radius, radius, 0, 0), mode=mode)
    out = torch_functional.conv2d(out, horizontal)
    out = torch_functional.pad(out, (0, 0, radius, radius), mode=mode)
    return torch_functional.conv2d(out, vertical)


def _best_aspect_aligned_image_size(
    source_width: int,
    source_height: int,
    scale: float,
) -> tuple[int, int, float]:
    ideal_width = source_width * scale
    ideal_height = source_height * scale

    def floor_32(value: float) -> int:
        return max(32, math.floor(value / 32.0) * 32)

    def ceil_32(value: float) -> int:
        return max(32, math.ceil(value / 32.0) * 32)

    minimum_width = ceil_32(source_width)
    minimum_height = ceil_32(source_height)
    width_options = {
        max(minimum_width, floor_32(ideal_width)),
        max(minimum_width, ceil_32(ideal_width)),
    }
    height_options = {
        max(minimum_height, floor_32(ideal_height)),
        max(minimum_height, ceil_32(ideal_height)),
    }
    source_aspect = source_width / source_height

    def score(candidate: tuple[int, int]) -> tuple[float, float, float]:
        width, height = candidate
        aspect_error = abs(math.log((width / height) / source_aspect))
        size_error = math.hypot(
            (width - ideal_width) / ideal_width,
            (height - ideal_height) / ideal_height,
        )
        area_error = abs(math.log((width * height) / (ideal_width * ideal_height)))
        return aspect_error, size_error, area_error

    target_width, target_height = min(
        (
            (width, height)
            for width in width_options
            for height in height_options
        ),
        key=score,
    )
    aspect_error_percent = abs(
        ((target_width / target_height) / source_aspect) - 1.0
    ) * 100.0
    return target_width, target_height, aspect_error_percent


def temporal_detail_enhance(
    frames: torch.Tensor,
    *,
    upscale_factor: float,
    strength: float,
    blur_radius: int,
    blur_sigma: float,
    motion_threshold: float,
    temporal_guard: float,
    frame_chunk_size: int = 8,
    maximum_output_megapixels: float = 2.1,
) -> tuple[torch.Tensor, str]:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise TypeError("frames must be a ComfyUI IMAGE tensor shaped [frames,H,W,C]")
    if frames.shape[-1] < 3:
        raise ValueError("frames must contain at least RGB channels")
    if frames.shape[0] < 1:
        raise ValueError("frames must contain at least one image")
    if not frames.dtype.is_floating_point:
        raise TypeError("frames must use a floating-point dtype")
    if not math.isfinite(upscale_factor) or not 1.0 <= upscale_factor <= 4.0:
        raise ValueError("upscale_factor must be finite and between 1 and 4")
    if not math.isfinite(strength) or not 0.0 <= strength <= 2.0:
        raise ValueError("strength must be finite and between 0 and 2")
    if blur_radius < 0 or blur_radius > 8:
        raise ValueError("blur_radius must be between 0 and 8")
    if not math.isfinite(blur_sigma) or blur_sigma <= 0.0:
        raise ValueError("blur_sigma must be finite and greater than zero")
    if not math.isfinite(motion_threshold) or motion_threshold <= 0.0:
        raise ValueError("motion_threshold must be finite and greater than zero")
    if not math.isfinite(temporal_guard) or not 0.0 <= temporal_guard <= 1.0:
        raise ValueError("temporal_guard must be finite and between 0 and 1")
    if frame_chunk_size < 1 or frame_chunk_size > 64:
        raise ValueError("frame_chunk_size must be between 1 and 64")
    if (
        not math.isfinite(maximum_output_megapixels)
        or not 0.1 <= maximum_output_megapixels <= 64.0
    ):
        raise ValueError("maximum_output_megapixels must be finite and between 0.1 and 64")

    source_h, source_w = int(frames.shape[1]), int(frames.shape[2])
    if math.isclose(upscale_factor, 1.0, rel_tol=0.0, abs_tol=1e-12):
        target_h, target_w = source_h, source_w
        aspect_error_percent = 0.0
    else:
        target_w, target_h, aspect_error_percent = _best_aspect_aligned_image_size(
            source_w,
            source_h,
            upscale_factor,
        )
    output_megapixels = target_w * target_h / 1_000_000.0
    if output_megapixels > maximum_output_megapixels + 1.0e-12:
        raise ValueError(
            f"temporal detail output {target_w}x{target_h} is {output_megapixels:.3f}MP, "
            f"above the configured {maximum_output_megapixels:.3f}MP safety budget"
        )

    def resize_batch(batch: torch.Tensor) -> torch.Tensor:
        if target_h == source_h and target_w == source_w:
            return batch
        return torch_functional.interpolate(
            batch.movedim(-1, 1),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).movedim(1, -1)

    applied = strength > 0.0
    mean_gate = 1.0
    if not applied and target_h == source_h and target_w == source_w:
        output = frames
    else:
        output_chunks = []
        gate_sum = 0.0
        gate_count = 0
        frame_count = int(frames.shape[0])
        for start in range(0, frame_count, frame_chunk_size):
            end = min(frame_count, start + frame_chunk_size)
            halo_start = max(0, start - 1)
            halo_end = min(frame_count, end + 1)
            halo = resize_batch(frames[halo_start:halo_end])
            local_start = start - halo_start
            local_end = local_start + (end - start)
            central = halo[local_start:local_end]
            if not applied:
                output_chunks.append(central)
                continue

            halo_rgb = halo[..., :3]
            rgb = halo_rgb[local_start:local_end]
            luma = (
                rgb[..., 0] * 0.2126
                + rgb[..., 1] * 0.7152
                + rgb[..., 2] * 0.0722
            ).unsqueeze(1)
            detail = luma - _blur_luma(luma, blur_radius, blur_sigma)

            halo_motion = torch.zeros(
                (halo.shape[0], 1, target_h, target_w),
                device=halo.device,
                dtype=halo.dtype,
            )
            if halo_rgb.shape[0] > 1:
                delta = (halo_rgb[1:] - halo_rgb[:-1]).abs().mean(dim=-1).unsqueeze(1)
                halo_motion[1:] = torch.maximum(halo_motion[1:], delta)
                halo_motion[:-1] = torch.maximum(halo_motion[:-1], delta)
            motion = halo_motion[local_start:local_end]
            gate = torch.exp(-motion / motion_threshold)
            effective = strength * ((1.0 - temporal_guard) + temporal_guard * gate)
            enhanced_rgb = (
                rgb + (detail * effective).movedim(1, -1)
            ).clamp(0.0, 1.0)
            if central.shape[-1] == 3:
                output_chunks.append(enhanced_rgb)
            else:
                output_chunks.append(torch.cat((enhanced_rgb, central[..., 3:]), dim=-1))
            gate_sum += float(gate.detach().sum().cpu())
            gate_count += gate.numel()
        output = torch.cat(output_chunks, dim=0)
        if gate_count:
            mean_gate = gate_sum / gate_count

    report = {
        "schema": "minimax_h3_temporal_detail_enhance_t8_v1",
        "status": (
            "applied"
            if applied or target_h != source_h or target_w != source_w
            else "noop"
        ),
        "source_width": source_w,
        "source_height": source_h,
        "output_width": target_w,
        "output_height": target_h,
        "output_multiple_of_32": target_w % 32 == 0 and target_h % 32 == 0,
        "multiple_of_32_applies_when_upscaling": True,
        "aspect_ratio_error_percent": aspect_error_percent,
        "output_megapixels": output_megapixels,
        "maximum_output_megapixels": float(maximum_output_megapixels),
        "frame_chunk_size": int(frame_chunk_size),
        "upscale_factor": float(upscale_factor),
        "strength": float(strength),
        "blur_radius": int(blur_radius),
        "blur_sigma": float(blur_sigma),
        "motion_threshold": float(motion_threshold),
        "temporal_guard": float(temporal_guard),
        "mean_temporal_gate": mean_gate,
        "luma_only_detail": True,
        "audio_touched": False,
        "semantic_detail_notice": (
            "This is a motion-gated luma detail filter. It cannot reconstruct missing "
            "identity or geometry and must not be described as generative restoration."
        ),
    }
    return output, canonical_json(report, indent=2)
