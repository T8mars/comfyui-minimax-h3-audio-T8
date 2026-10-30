"""Pika's six-knot renoise recipe, executed by native Core LCM/FLOW_AV.

The adapter is selected upstream with an existing LoRA loader. This module does
not infer or replace weights, conditions, masks, references or user patches.
"""
import json

import torch

from .core import nested_av_parts
from .sampling import setup_dual_clock_sampling


KNOTS = (1000, 750, 500, 250, 100, 40, 0)
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def pika_sigmas():
    # Convert the author's unshifted knots exactly once, instead of applying a
    # generic six-step scheduler or shifting Comfy's already shifted list again.
    raw = torch.tensor(KNOTS, dtype=torch.float64) / 1000
    return (VIDEO_SHIFT * raw / (1 + (VIDEO_SHIFT - 1) * raw)).float()


def prepare_pika_flex(model, av_latent):
    selected, sampler, _ = setup_dual_clock_sampling(
        model, av_latent, 6, VIDEO_SHIFT, AUDIO_SHIFT, sampler_name="lcm",
    )
    sigmas = pika_sigmas()
    video, audio = nested_av_parts(av_latent)
    sampling = selected.get_model_object("model_sampling")
    report = {
        "schema": "t8.pika_h3_flex.prepare.v1",
        "status": "prepared_sampling_unobserved",
        "knots": list(KNOTS), "steps": 6, "sampler": "lcm",
        "video_sigmas": sigmas.tolist(),
        "video_shift": VIDEO_SHIFT, "audio_shift": AUDIO_SHIFT,
        "intermediate_noise": "native Core LCM draws fresh noise at each of five nonterminal transitions",
        "guidance": "Connect BasicGuider (CFG 1) to this MODEL and your selected positive conditioning",
        "adapter": "Load the selected Pika FL2VA or Ref2VA adapter upstream; this node does not load, remove or substitute LoRAs",
        "adapter_strength_baseline": 1.0,
        "patch_policy": "Existing LoRAs, strengths, hooks and wrappers are retained; unknown combinations may change results and are not quality-qualified",
        "noise_scale": float(getattr(sampling, "noise_scale", 1.0)),
        "audio_scale": float(sampling.audio_scale),
        "video_shape": list(video.shape), "audio_shape": list(audio.shape),
        "conditioning": "Caller-owned prompts, endpoint frames, references and masks remain unchanged",
        "trained_GPU_quality_verified": False,
        "official_source": "https://huggingface.co/Pika-Labs/Pika-H3-Flex/blob/main/fl2va/manifest.json",
    }
    return selected, sampler, sigmas, json.dumps(report, ensure_ascii=False, sort_keys=True)
