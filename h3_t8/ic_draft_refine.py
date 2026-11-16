"""IC recipe helpers adapted from slmonker's Apache-2.0 companion.

Source a85bd9664a752637c5ec88bf4b1930b4688fc251; see IC-APACHE2.txt and
IC-UPSTREAM-NOTICE.txt. T8 changes: explicit silence fitting, video mask and
template metadata preservation, content reports. No copied draft crop/length
restrictions; use the existing source-conform/reference-video consumers.
The silence asset and model weights retain their separate upstream license.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import torch
import comfy.nested_tensor
import comfy.utils

from .core import nested_av_parts, split_noise_masks


TASK = (
    "Enhance <Video 1> into a high-quality version of the same video. "
    "Preserve the scene, objects, camera motion, timing, and composition of every frame. "
    "Restore fine details and remove degradation. Do not continue or change the scene."
)
GRIDS = {
    "up3": (0.834942, 0.696852, 0.468753, 0.0),
    "up4": (0.923077, 0.834942, 0.696852, 0.468753, 0.0),
    "hf8": (
        1.0,
        0.99391,
        0.984288,
        0.966064,
        0.923077,
        0.834942,
        0.696852,
        0.468753,
        0.0,
    ),
}


def ic_recipe(caption, preset="up3", custom_sigmas=""):
    if not isinstance(caption, str):
        raise TypeError("IC scene caption must be text")
    if custom_sigmas.strip():
        sigmas = torch.tensor(
            [float(x) for x in re.split(r"[,;\s]+", custom_sigmas.strip())],
            dtype=torch.float32,
        )
    else:
        sigmas = torch.tensor(GRIDS[preset], dtype=torch.float32)
    if sigmas.ndim != 1 or len(sigmas) < 2 or not torch.isfinite(sigmas).all():
        raise ValueError("IC sigmas must be a finite one-dimensional sequence")
    if (sigmas < 0).any() or (sigmas[1:] >= sigmas[:-1]).any() or sigmas[-1] != 0:
        raise ValueError(
            "IC sigmas must be strictly decreasing, nonnegative and explicitly end at zero"
        )
    prompt = TASK + "\n\nScene description: " + caption
    report = {
        "schema": "t8.ic.recipe.v1",
        "preset": "custom" if custom_sigmas.strip() else preset,
        "sigmas": sigmas.tolist(),
        "additional_shift_applied": False,
        "reference": "Draft video must actually be connected as <Video 1>; prompt text alone is not a reference",
        "sampling": "Euler, BasicGuider CFG1 is the source baseline; upstream user MODEL/LoRAs remain caller-owned",
        "audio": "Fixed silence conditions the IC stage; mux the original draft PCM at delivery",
        "quality_accepted": False,
    }
    return prompt, sigmas, json.dumps(report, ensure_ascii=False)


def setup_ic_sampler(model, av_latent, preset="up3"):
    """Source IC tail restart with the actual dedicated two-time HF consumer."""
    from .hyperflow_sampling_advanced import (
        build_hyperflow_plan,
        setup_hyperflow_sampler,
    )

    start = {"up3": 5, "up4": 4, "hf8": 0}[preset]
    plan = build_hyperflow_plan(model, start, 8)
    selected, sampler, sigmas = setup_hyperflow_sampler(
        model,
        av_latent,
        plan,
        internal_continuation=start > 0,
        new_noise_restart=start > 0,
    )
    report = {
        "schema": "t8.ic.sampling.v1",
        "preset": preset,
        "absolute_interval": [start, 8],
        "actual_HyperFlow_plan": plan.as_report(),
        "init": "clean upscaled latent re-noised at first sigma"
        if start
        else "pure noise",
        "extra_shift_applied": False,
        "endpoint_time": "dedicated HyperFlow runtime observes each actual current/next interval",
        "quality_accepted": False,
    }
    return selected, sampler, sigmas, json.dumps(report, ensure_ascii=False)


def compose_ic_target(
    video_latent, reference_latent, silence_path, audio_fit="strict", pin_audio=True
):
    video = video_latent.get("samples")
    if getattr(video, "is_nested", False):
        video, old_audio = nested_av_parts(video_latent)
        video_mask, _ = split_noise_masks(video_latent, video, old_audio)
    else:
        video_mask = video_latent.get("noise_mask")
    template_video, template_audio = nested_av_parts(reference_latent)
    if not isinstance(video, torch.Tensor) or video.shape != template_video.shape:
        raise ValueError(
            "Upscaled video and actual reference target must have the same latent [B,C,T,H,W] shape"
        )
    if not torch.isfinite(video).all():
        raise ValueError("Upscaled video contains nonfinite values")
    path = Path(silence_path)
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    state = comfy.utils.load_torch_file(str(path), safe_load=True)
    raw = state.get("audio_latents")
    if not isinstance(raw, torch.Tensor) or raw.ndim != 3:
        raise ValueError("Silence asset needs audio_latents [channels,features,time]")
    silence = raw.permute(1, 0, 2).unsqueeze(0)
    if (
        silence.shape[:3] != template_audio.shape[:3]
        or not torch.isfinite(silence).all()
    ):
        raise ValueError(
            "Silence audio channels/batch must match the actual target and contain finite values"
        )
    delta = int(template_audio.shape[-1] - silence.shape[-1])
    if audio_fit not in ("strict", "trim_or_zero_pad_exp"):
        raise ValueError("Choose an explicit silence fitting policy")
    if delta and audio_fit == "strict":
        raise ValueError(
            f"Silence time rows {silence.shape[-1]} differ from target {template_audio.shape[-1]}; "
            "use a matching asset or explicitly select experimental trim/zero-pad"
        )
    if delta > 0:
        silence = torch.cat(
            (silence, silence.new_zeros((*silence.shape[:-1], delta))), dim=-1
        )
    elif delta < 0:
        silence = silence[..., : template_audio.shape[-1]]
    video = video.to(device=template_video.device, dtype=template_video.dtype)
    silence = silence.to(device=template_audio.device, dtype=template_audio.dtype)
    if video_mask is None:
        video_mask, _ = split_noise_masks(
            reference_latent, template_video, template_audio
        )
    if video_mask is None:
        video_mask = torch.ones_like(video)
    else:
        if (
            not isinstance(video_mask, torch.Tensor)
            or not torch.isfinite(video_mask).all()
        ):
            raise ValueError("IC video noise mask must be a finite tensor")
        try:
            if tuple(torch.broadcast_shapes(video_mask.shape, video.shape)) != tuple(
                video.shape
            ):
                raise ValueError("IC video mask would expand the target video shape")
        except RuntimeError as error:
            raise ValueError(
                "IC video mask does not broadcast to the target video"
            ) from error
    report = {
        "schema": "t8.ic.compose.v1",
        "silence_file": path.name,
        "silence_sha256": digest.hexdigest(),
        "silence_time_rows": int(raw.shape[-1]),
        "target_audio_time_rows": int(template_audio.shape[-1]),
        "audio_fit": audio_fit,
        "audio_rows_added": max(0, delta),
        "audio_rows_removed": max(0, -delta),
        "normalization": "Official raw audio_latents permute(1,0,2).unsqueeze(0), native LATENT; no extra scale",
        "audio_pinned": bool(pin_audio),
        "video_mask_preserved": True,
        "target_metadata_preserved": True,
        "draft_PCM_for_delivery": "must be connected separately",
        "quality_accepted": False,
    }
    output = dict(reference_latent)
    output["samples"] = comfy.nested_tensor.NestedTensor((video, silence))
    output["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (
            video_mask,
            torch.zeros_like(silence) if pin_audio else torch.ones_like(silence),
        )
    )
    output["t8_ic"] = report
    return output, json.dumps(report, ensure_ascii=False)
