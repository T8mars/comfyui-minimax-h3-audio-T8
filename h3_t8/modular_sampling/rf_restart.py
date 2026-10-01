"""RF endpoint handoff and exactly ONE restart descent (not a base+restart loop).

The old restart's original template remains the inpaint/audio-rebase anchor.
Do not replace it with the completed endpoint or ordinary fresh-noise setup.
Public StageContext/effect/result adapters are added separately after parity.
"""
from dataclasses import dataclass
import json
import math

import torch
import comfy.samplers
import comfy.utils

from .. import detail_sampling_advanced as legacy
from .. import sampling
from ..progressive_continuation_runtime import _input_identity

RAW_ENDPOINT = "t8_modular_rf_model_endpoint_v1"


def retained_endpoint(completed_av):
    """Read an explicitly captured endpoint, never infer it from rounded x0."""
    packet = completed_av.get(RAW_ENDPOINT)
    if packet is None:
        return None
    if type(packet) is not dict or set(packet) != {"packed", "packed_identity", "visible_identity"}:
        raise ValueError("Invalid retained RF model endpoint")
    packed = packet["packed"]
    visible, _ = comfy.utils.pack_latents(completed_av["samples"].unbind())
    if (not isinstance(packed, torch.Tensor) or packed.shape != visible.shape
            or not packed.is_floating_point() or not bool(torch.isfinite(packed).all())
            or _identity(packed) != packet["packed_identity"]
            or _identity(completed_av["samples"]) != packet["visible_identity"]):
        raise ValueError("RF retained endpoint or its visible output changed")
    return packed


def attach_endpoint(output, packed):
    return {**output, RAW_ENDPOINT: {"packed": packed,
        "packed_identity": _identity(packed), "visible_identity": _identity(output["samples"])}}


class EndpointCapture:
    """Observe one real sampler return before Core's float32 public boundary."""
    def __init__(self, sampler):
        self.sampler, self.endpoint = sampler, None

    def sample(self, *args, **kwargs):
        result = self.sampler.sample(*args, **kwargs)
        self.endpoint = result.detach().to(device="cpu", copy=True)
        return result


def _identity(value):
    return json.dumps(_input_identity(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class RFHandoff:
    completed_av: dict
    original_template: dict
    completed_identity: str
    template_identity: str

    def verify(self):
        if _identity(self.completed_av) != self.completed_identity or _identity(self.original_template) != self.template_identity:
            raise ValueError("RF handoff endpoint or original template changed")


def prepare_handoff(completed_av, original_template):
    """Bind explicit endpoint and original anchor; no model call/no random draw."""
    completed = sampling.nested_av_parts(completed_av)
    original = sampling.nested_av_parts(original_template)
    if [tuple(x.shape) for x in completed] != [tuple(x.shape) for x in original]:
        raise ValueError("RF endpoint and original AV template must have matching layouts")
    if any(not bool(torch.isfinite(x).all()) for x in (*completed, *original)):
        raise ValueError("RF handoff requires finite AV tensors")
    retained_endpoint(completed_av)
    return RFHandoff(completed_av, original_template, _identity(completed_av), _identity(original_template))


def _model_space(model, av_latent, like):
    packed, _ = comfy.utils.pack_latents(av_latent["samples"].unbind())
    packed = packed.to(device=like.device, dtype=like.dtype)
    # Same empty-latent convention as Core CFGGuider.inner_sample.
    return model.process_latent_in(packed) if torch.count_nonzero(packed) > 0 else packed


def _validate_packed_mask(mask, video_values, packed_values):
    if mask is None:
        return
    if mask.shape[-1] != packed_values:
        raise ValueError("RF restart mask does not match packed AV layout")
    if not bool(torch.isfinite(mask).all()) or not torch.allclose(mask, mask.round(), rtol=0, atol=1e-6):
        raise ValueError("RF restart requires finite binary masks")
    if not bool((mask[..., video_values:] == 1).all()):
        raise ValueError("RF restart requires the complete audio latent to participate")
    if not bool((mask[..., :video_values] == 1).any()):
        raise ValueError("RF restart requires active video values")


def _renoise(endpoint, *, video_values, video_sigma, audio_sigma, restart_seed, mask):
    generator = torch.Generator(device=endpoint.device)
    generator.manual_seed(restart_seed)
    restarted = torch.empty_like(endpoint)
    restarted.normal_(generator=generator)
    restarted[..., :video_values].mul_(video_sigma).add_(endpoint[..., :video_values], alpha=1.0 - video_sigma)
    restarted[..., video_values:].mul_(audio_sigma).add_(endpoint[..., video_values:], alpha=1.0 - audio_sigma)
    if mask is not None:
        _validate_packed_mask(mask, video_values, endpoint.shape[-1])
        restarted.sub_(endpoint).mul_(mask).add_(endpoint)
    return restarted


class RFRestartSampler(comfy.samplers.KSAMPLER):
    """Core sampler shell without a base descent or generic noise_scaling call.

    NOISE still supplies the original inpaint noise/seed. RF random noise uses
    its separate restart_seed on the actual sample device, exactly as legacy.
    """

    def __init__(self, sampler_function, handoff, report):
        super().__init__(sampler_function)
        self.handoff = handoff
        self.report_json = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        self.handoff.verify()
        report = json.loads(self.report_json)
        if not report["applied"]:
            return latent_image
        expected = torch.linspace(report["restart_video_sigma"], 0., report["restart_nfe"] + 1,
                                  dtype=sigmas.dtype, device=sigmas.device)
        if sigmas.shape != expected.shape or not torch.equal(sigmas, expected):
            raise ValueError("RF restart SIGMAS differ from the selected linear restart trajectory")
        if not torch.equal(latent_image, _model_space(model_wrap.inner_model, self.handoff.completed_av, latent_image)):
            raise ValueError("RF restart latent_image is not its bound completed endpoint")
        endpoint = retained_endpoint(self.handoff.completed_av)
        if endpoint is None:
            endpoint = latent_image
        else:
            endpoint = endpoint.to(device=latent_image.device)
            # Preserve the precise model state, but reject a stale visible LATENT
            # or a different latent format. Do not cast the restart state itself.
            visible, _ = comfy.utils.pack_latents(self.handoff.completed_av["samples"].unbind())
            projected = model_wrap.inner_model.process_latent_out(endpoint.to(torch.float32))
            if not torch.equal(projected, visible.to(device=projected.device)):
                raise ValueError("RF retained endpoint does not match the selected model latent format")
        video, audio = sampling.nested_av_parts(self.handoff.completed_av)
        video_values = math.prod(video.shape[1:])
        packed_values = video_values + math.prod(audio.shape[1:])
        _validate_packed_mask(denoise_mask, video_values, packed_values)
        args = dict(extra_args)
        args["denoise_mask"] = denoise_mask
        model_k = comfy.samplers.KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = _model_space(model_wrap.inner_model, self.handoff.original_template, latent_image)
        model_k.noise = noise
        restarted = _renoise(endpoint, video_values=video_values,
            video_sigma=report["restart_video_sigma"], audio_sigma=report["restart_audio_sigma"],
            restart_seed=report["restart_seed"], mask=denoise_mask)

        def k_callback(item):
            if callback is not None:
                callback(item["i"], item["denoised"], item["x"], report["restart_nfe"])

        output = self.sampler_function(model_k, restarted, sigmas, extra_args=args,
                                       callback=k_callback, disable=disable_pbar)
        self.handoff.verify()
        return model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], output)


def build_restart_stage(model, handoff, *, shift_video=12., shift_audio=3.,
                        restart_video_sigma=.15, restart_steps=3, restart_seed=1234, sigma_dtype=torch.float32):
    if type(handoff) is not RFHandoff:
        raise ValueError("Connect an explicit RF endpoint/original-template handoff")
    handoff.verify()
    # Original setup owns numerical parameter/mask validation. Its combined
    # sampler is discarded and NEVER invoked, even with restart disabled.
    prepared, _, _, raw = legacy.setup_rectified_flow_restart_sampling(model, handoff.original_template,
        steps=max(1, restart_steps), shift_video=shift_video, shift_audio=shift_audio,
        restart_video_sigma=restart_video_sigma, restart_steps=restart_steps, restart_seed=restart_seed)
    report = json.loads(raw)
    video, audio = sampling.nested_av_parts(handoff.completed_av)
    base = sampling._build_dual_clock_sampler(video_values=math.prod(video.shape[1:]),
        packed_values=math.prod(video.shape[1:]) + math.prod(audio.shape[1:]),
        shift_video=shift_video, shift_audio=shift_audio,
        audio_velocity_is_raw=sampling.model_uses_raw_audio_velocity(model))
    sampler = RFRestartSampler(base.sampler_function, handoff, report)
    sigmas = (torch.linspace(restart_video_sigma, 0., restart_steps + 1, dtype=sigma_dtype)
              if report["applied"] else torch.empty(0, dtype=sigma_dtype))
    details = {"schema": "t8.modular-sampling.rf-restart-stage.v1", "legacy_restart": report,
        "diffusion_calls_in_setup": 0, "planned_stage_calls": report["restart_nfe"],
        "endpoint_identity": json.loads(handoff.completed_identity),
        "anchor_identity": json.loads(handoff.template_identity),
        "boundary": "Restart descent only. Connect bound completed_av as latent_image and original base NOISE. "
                    "Preserves legacy original-template audio rebase, including its second rebase; not an algorithm fix. "
                    "Generic public StageContext/effect/portable-result integration is not yet qualified."}
    return prepared, sampler, sigmas, json.dumps(details, ensure_ascii=False, indent=2)
