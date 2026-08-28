from __future__ import annotations

import json
from collections.abc import Mapping

import comfy.patcher_extension
import torch


CADS_WRAPPER_KEY = "minimax_h3_cads_visual_reference_t8_v1"
NOISE_MODES = ("paper_independent", "stable_fixed_path")


def cads_gamma(t: torch.Tensor, tau1: float, tau2: float) -> torch.Tensor:
    if not 0.0 <= float(tau1) < float(tau2) <= 1.0:
        raise ValueError("CADS thresholds must satisfy 0 <= tau1 < tau2 <= 1")
    value = torch.as_tensor(t, dtype=torch.float32)
    middle = (float(tau2) - value) / (float(tau2) - float(tau1))
    return torch.where(
        value <= float(tau1),
        torch.ones_like(value),
        torch.where(value >= float(tau2), torch.zeros_like(value), middle),
    ).clamp(0.0, 1.0)


def anneal_visual_condition(
    clean: torch.Tensor,
    noise: torch.Tensor,
    gamma: torch.Tensor | float,
    noise_scale: float,
    rescale_mix: float,
) -> torch.Tensor:
    if not isinstance(clean, torch.Tensor) or not isinstance(noise, torch.Tensor):
        raise TypeError("CADS visual conditions and noise must be tensors")
    if clean.shape != noise.shape:
        raise ValueError("CADS visual condition and noise shapes must match")
    if not 0.0 <= float(noise_scale) <= 2.0:
        raise ValueError("noise_scale must be between 0.0 and 2.0")
    if not 0.0 <= float(rescale_mix) <= 1.0:
        raise ValueError("rescale_mix must be between 0.0 and 1.0")

    clean32 = clean.to(torch.float32)
    noise32 = noise.to(device=clean.device, dtype=torch.float32)
    gamma32 = torch.as_tensor(gamma, device=clean.device, dtype=torch.float32).clamp(
        0.0, 1.0
    )
    corrupted = (
        torch.sqrt(gamma32) * clean32
        + float(noise_scale) * torch.sqrt(1.0 - gamma32) * noise32
    )
    if float(rescale_mix) <= 0.0:
        return corrupted

    reduce_dims = tuple(range(corrupted.ndim))
    clean_mean = clean32.mean(dim=reduce_dims, keepdim=True)
    clean_std = clean32.std(dim=reduce_dims, keepdim=True, unbiased=False)
    corrupted_mean = corrupted.mean(dim=reduce_dims, keepdim=True)
    corrupted_std = corrupted.std(dim=reduce_dims, keepdim=True, unbiased=False)
    rescaled = (
        (corrupted - corrupted_mean)
        / corrupted_std.clamp_min(torch.finfo(torch.float32).eps)
        * clean_std
        + clean_mean
    )
    return float(rescale_mix) * rescaled + (1.0 - float(rescale_mix)) * corrupted


def _noise_for(
    clean: torch.Tensor,
    *,
    seed: int,
    condition_index: int,
    timestep_key: int,
    noise_mode: str,
) -> torch.Tensor:
    effective_step = timestep_key if noise_mode == "paper_independent" else 0
    effective_seed = (
        int(seed)
        + 1_000_003 * int(condition_index)
        + 97_409 * int(effective_step)
    ) & 0x7FFF_FFFF_FFFF_FFFF
    generator = torch.Generator(device=clean.device).manual_seed(effective_seed)
    return torch.randn(
        clean.shape,
        generator=generator,
        device=clean.device,
        dtype=torch.float32,
    )


def build_cads_visual_reference_model(
    model,
    noise_scale: float,
    tau1: float,
    tau2: float,
    rescale_mix: float,
    noise_mode: str,
    seed: int,
):
    if noise_mode not in NOISE_MODES:
        raise ValueError(f"unsupported noise_mode: {noise_mode!r}")
    if not 0.0 <= float(noise_scale) <= 2.0:
        raise ValueError("noise_scale must be between 0.0 and 2.0")
    if not 0.0 <= float(rescale_mix) <= 1.0:
        raise ValueError("rescale_mix must be between 0.0 and 1.0")
    # Validate the thresholds without allocating a condition tensor.
    cads_gamma(torch.tensor(0.5), float(tau1), float(tau2))
    if not hasattr(model, "clone") or not hasattr(model, "add_wrapper_with_key"):
        raise RuntimeError(
            "This ComfyUI build does not expose composable diffusion-model wrappers"
        )

    patched = model.clone()

    def _cads_wrapper(
        executor,
        x,
        timestep,
        context,
        transformer_options=None,
        **kwargs,
    ):
        payload = kwargs.get("minimax_payload")
        conditions = payload.get("cond_video_latents") if isinstance(payload, Mapping) else None
        if not conditions:
            return executor(
                x,
                timestep,
                context,
                transformer_options,
                **kwargs,
            )

        sigma_video = (timestep.flatten()[0].to(torch.float32) / 1000.0).clamp(
            0.0, 1.0
        )
        gamma = cads_gamma(sigma_video, float(tau1), float(tau2)).to(
            device=sigma_video.device
        )
        timestep_key = int(round(float(sigma_video.detach().cpu()) * 1_000_000.0))
        annealed = []
        for index, clean in enumerate(conditions):
            if not isinstance(clean, torch.Tensor):
                raise TypeError(
                    f"MiniMax H3 visual condition {index} is not a tensor"
                )
            noise = _noise_for(
                clean,
                seed=int(seed),
                condition_index=index,
                timestep_key=timestep_key,
                noise_mode=noise_mode,
            )
            annealed.append(
                anneal_visual_condition(
                    clean,
                    noise,
                    gamma.to(device=clean.device),
                    float(noise_scale),
                    float(rescale_mix),
                )
            )

        runtime_payload = dict(payload)
        runtime_payload["cond_video_latents"] = annealed
        # CADS already applied the paper coefficients. Disable H3's additional
        # static a*x + (1-a)*noise pass to avoid corrupting the condition twice.
        runtime_payload["visual_cond_noise_aug"] = 1.0
        kwargs["minimax_payload"] = runtime_payload
        return executor(
            x,
            timestep,
            context,
            transformer_options,
            **kwargs,
        )

    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        CADS_WRAPPER_KEY,
        _cads_wrapper,
    )
    report = {
        "status": "experimental",
        "method": "CADS visual-condition annealing",
        "paper_equation": "sqrt(gamma(t))*condition + s*sqrt(1-gamma(t))*noise",
        "time_coordinate": "MiniMax H3 shifted video sigma (1 early -> 0 late)",
        "noise_scale": float(noise_scale),
        "tau1": float(tau1),
        "tau2": float(tau2),
        "rescale_mix": float(rescale_mix),
        "noise_mode": noise_mode,
        "seed": int(seed),
        "visual_references_changed": True,
        "audio_conditioning_changed": False,
        "target_audio_latent_changed": False,
        "h3_quality_validated": False,
        "warnings": [
            "CADS was not trained or calibrated for MiniMax H3.",
            "Strong early corruption can reduce identity, keyframe, action and composition adherence.",
            "paper_independent follows the paper's fresh-noise interpretation; stable_fixed_path is an H3-oriented temporal-path experiment.",
        ],
    }
    if hasattr(patched, "set_attachments"):
        patched.set_attachments(CADS_WRAPPER_KEY, dict(report))
    return patched, json.dumps(report, ensure_ascii=False, indent=2)
