"""Explicit generated-audio prefix carry/release on actual Core mask consumers.

This is independent of voice-reference tags. Neither feature touches final
retained PCM, video samples/masks, steps, noise seeds or sigma tables.
"""

from __future__ import annotations

import json
import math
import uuid

import torch
import comfy.nested_tensor
import comfy.patcher_extension

from .core import nested_av_parts, split_noise_masks
from .prompt_edit_guidance import sampling_step


KEY = "t8_generated_audio_tail_release_v1"


def _rows(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return value


def carry_audio_prefix(target, previous, context_rows=0, previous_end_row=0):
    rows = _rows(context_rows, "context_rows")
    end = _rows(previous_end_row, "previous_end_row")
    video, audio = nested_av_parts(target)
    if rows == 0:
        return target, json.dumps({"status": "disabled", "copied_rows": 0})
    if previous is None:
        raise ValueError("Nonzero context_rows needs an actual previous AV latent")
    _, prior = nested_av_parts(previous)
    if audio.shape[:3] != prior.shape[:3]:
        raise ValueError("Previous and target audio batch/channels/stereo must match")
    end = end or int(prior.shape[-1])
    if end > prior.shape[-1] or rows > end or rows > audio.shape[-1]:
        raise ValueError(
            "Selected carried audio interval exceeds actual source or target rows"
        )
    if not torch.isfinite(prior[..., end - rows : end]).all():
        raise ValueError("Carried audio contains nonfinite samples")
    carried = audio.clone()
    carried[..., :rows] = prior[..., end - rows : end].to(carried)
    video_mask, audio_mask = split_noise_masks(target, video, audio)
    video_mask = torch.ones_like(video) if video_mask is None else video_mask
    if audio_mask is None:
        audio_mask = torch.ones_like(audio)
    else:
        try:
            audio_mask = torch.broadcast_to(audio_mask, audio.shape).clone()
        except RuntimeError as error:
            raise ValueError("Audio mask does not broadcast to target audio") from error
    audio_mask[..., :rows] = 0.0
    output = dict(target)
    output["samples"] = comfy.nested_tensor.NestedTensor((video, carried))
    output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return output, json.dumps(
        {
            "schema": "t8.audio_prefix_carry.v1",
            "copied_rows": rows,
            "previous_interval": [end - rows, end],
            "target_interval": [0, rows],
            "actual_target_audio_mask_pinned": True,
            "voice_reference_tags_are_not_a_pin": True,
            "final_PCM_untouched": True,
            "quality_accepted": False,
        }
    )


class TailRelease:
    def __init__(self, video, audio, rows, release_step, prior_mask, prior_wrapper):
        self.video_values = math.prod(video.shape[1:])
        self.audio_shape = tuple(audio.shape[1:])
        self.audio_values = math.prod(audio.shape[1:])
        self.rows = rows
        self.release_step = release_step
        self.prior_mask = prior_mask
        self.prior_wrapper = prior_wrapper
        self.calls = []
        self.mask_calls = []

    def scope(self, executor, *args, **kwargs):
        self.calls = []
        self.mask_calls = []
        return executor(*args, **kwargs)

    def mask(self, sigma, denoise_mask, extra_options=None):
        options = {} if extra_options is None else extra_options
        original = denoise_mask
        if self.prior_mask is not None:
            original = self.prior_mask(sigma, original, extra_options=options)
        step = sampling_step(
            sigma, {"transformer_options": {"sample_sigmas": options.get("sigmas")}}
        )
        released = step >= self.release_step
        if (
            not isinstance(original, torch.Tensor)
            or original.shape[-1] != self.video_values + self.audio_values
        ):
            raise ValueError(
                "Audio release needs the actual full packed AV denoise mask"
            )
        output = original
        if released:
            output = original.clone()
            tail = output[..., self.video_values :].reshape(
                output.shape[0], *self.audio_shape
            )
            tail[..., : self.rows] = 1.0
            # The T8 dual-clock Euler loop keeps a view of this same Core-owned
            # packed mask for its audio update. Releasing only X0-inpaint or
            # the DiT condition would leave that actual update on the pin clock.
            packed_audio = denoise_mask[..., self.video_values :].reshape(
                denoise_mask.shape[0], *self.audio_shape
            )
            packed_audio[..., : self.rows] = 1.0
        self.mask_calls.append({"step": step, "released": released})
        return output

    def forward(self, apply, options):
        conditioning = options.get("c", {})
        schedule = conditioning.get("transformer_options", {}).get("sample_sigmas")
        step = sampling_step(
            options["timestep"], {"transformer_options": {"sample_sigmas": schedule}}
        )
        released = step >= self.release_step
        actual = conditioning.get("audio_denoise_mask")
        observed = False
        selected = options
        if actual is not None:
            if (
                not isinstance(actual, torch.Tensor)
                or actual.shape[-1] != self.audio_shape[-1]
            ):
                raise ValueError(
                    "Audio release conditioning has a different time axis than the selected target"
                )
            observed = True
            if released:
                changed = actual.clone()
                changed[..., : self.rows] = 1.0
                selected = {
                    **options,
                    "c": {**conditioning, "audio_denoise_mask": changed},
                }
        self.calls.append(
            {
                "step": step,
                "released": released,
                "actual_audio_condition_mask_present": observed,
            }
        )
        if self.prior_wrapper is not None:
            return self.prior_wrapper(apply, selected)
        return apply(selected["input"], selected["timestep"], **selected["c"])

    def report(self):
        return {
            "schema": "t8.audio_tail_release.v1",
            "prefix_rows": self.rows,
            "release_step": self.release_step,
            "step_clock": "zero_based_actual_sigma_interval",
            "inpaint_mask_calls": self.mask_calls,
            "model_input_calls": self.calls,
            "status": "observed" if self.calls and self.mask_calls else "unobserved",
            "consumer_scope": "Core inpaint mask, model-input audio mask and T8 dual-clock packed mask update; later foreign wrappers may override inputs",
            "final_PCM_untouched": True,
            "human_quality_accepted": False,
        }


def setup_audio_tail_release(
    model, av_latent, mode="disabled", prefix_rows=0, release_step=0
):
    if mode == "disabled":
        return model, json.dumps({"status": "disabled", "model_unchanged": True})
    if mode != "release_after_step":
        raise ValueError("Choose disabled or release_after_step")
    rows = _rows(prefix_rows, "prefix_rows")
    step = _rows(release_step, "release_step")
    video, audio = nested_av_parts(av_latent)
    if rows == 0 or rows > audio.shape[-1]:
        raise ValueError(
            "Release requires a nonempty prefix within actual target audio rows"
        )
    _, audio_mask = split_noise_masks(av_latent, video, audio)
    if audio_mask is None:
        raise ValueError(
            "No target audio pin exists; voice-reference tags are not a pinned audio prefix"
        )
    try:
        prefix_mask = torch.broadcast_to(audio_mask, audio.shape)[..., :rows]
    except RuntimeError as error:
        raise ValueError("Audio mask does not broadcast to target audio") from error
    if not (prefix_mask == 0).all():
        raise ValueError(
            "Selected audio prefix must actually be hard-pinned in the target denoise mask"
        )
    prior_mask = model.model_options.get("denoise_mask_function")
    prior_wrapper = model.model_options.get("model_function_wrapper")
    if any(
        value is not None and not callable(value)
        for value in (prior_mask, prior_wrapper)
    ):
        raise TypeError("Existing mask/model delegates must be callable")
    runtime = TailRelease(video, audio, rows, step, prior_mask, prior_wrapper)
    selected = model.clone()
    selected.set_model_denoise_mask_function(runtime.mask)
    selected.set_model_unet_function_wrapper(runtime.forward)
    selected.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        "t8_audio_tail_" + uuid.uuid4().hex,
        runtime.scope,
    )
    selected.set_attachments(KEY, (*tuple(selected.get_attachment(KEY) or ()), runtime))
    return selected, json.dumps(runtime.report())


def audit_audio_tail(model, after_sampling):
    return after_sampling, json.dumps(
        {
            "owners": [r.report() for r in model.get_attachment(KEY) or ()],
            "human_quality_accepted": False,
        }
    )
