"""Request-scoped Solar silence wrapper and unmodified Core LCM selection.

Independent adapter for Enderfga/SolarRenderer 00b89093234a194bb06e12b7df0d2ed354769a8d.
No causal KV protocol, node registration, audio driver or trained-quality claim.
"""
from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
import hashlib
import math

import torch
import comfy.model_management
import comfy.samplers
from comfy.ldm.minimax.model import time_shift_sigma
from comfy.patcher_extension import WrappersMP

from .patch_stack_policy import warn_patch_stack
from .solar_structural import finite_tensor

KEY = "t8_solar_structural_v1"


def _digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _shift(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Solar sigma shifts must be positive finite numbers") from error
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError("Solar sigma shifts must be positive finite numbers")
    return number


def _core_shifts(executor, transformer_options):
    # These are the exact inputs used by MiniMaxH3Model._forward. The selected
    # sampler's audio_shift may be None and is not the DiT's fallback clock.
    network = getattr(executor, "class_obj", None)
    values = []
    for stream in ("video", "audio"):
        name = "sigma_shift_" + stream
        option = "minimax_h3_" + name
        if option in transformer_options:
            value = transformer_options[option]
        elif hasattr(network, name):
            value = getattr(network, name)
        else:
            raise ValueError(f"Solar silence needs actual diffusion {name} or transformer_options[{option!r}]")
        values.append(_shift(value))
    return tuple(values)


def _replacement_rows(audio, mask):
    """Only wholly editable native stereo/tick rows receive encoded silence."""
    if mask is None:
        return torch.ones((audio.shape[0], 1, *audio.shape[2:]), dtype=torch.bool, device=audio.device)
    finite_tensor(mask, "Solar audio denoise mask")
    if bool(((mask < 0) | (mask > 1)).any()):
        raise ValueError("Solar audio denoise mask coverage must be in [0,1]")
    try:
        expanded = torch.broadcast_to(mask.to(audio.device), audio.shape)
    except RuntimeError as error:
        raise ValueError("Solar audio denoise mask cannot broadcast to the actual audio stream") from error
    # A token contains all 32 features. Retain the whole incoming token if any
    # feature is protected or partial; do not invent a partial-noise schedule.
    return (expanded == 1).all(dim=1, keepdim=True)


def sampling_contract(steps=4, video_shift=12.):
    if type(steps) is not int or steps < 1:
        raise ValueError("Solar sampling steps must be a positive integer")
    shift = _shift(video_shift)
    base = torch.linspace(1., 0., steps + 1, dtype=torch.float32)
    sigmas = shift * base / (1 + (shift - 1) * base)
    return comfy.samplers.ksampler("lcm"), sigmas, {
        "sampler": "Core lcm", "steps": steps, "video_shift": shift,
        "published_schedule": steps == 4 and shift == 12., "trained_consumer_qualified": False}


class SilenceOwner:
    def __init__(self, silence, seed, identity=None):
        finite_tensor(silence, "Solar silence")
        if silence.ndim != 4 or tuple(silence.shape[:3]) != (1, 32, 2):
            raise ValueError("Solar silence requires [1, 32, stereo=2, audio ticks]")
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("Solar silence seed must be an unsigned 64-bit integer")
        self.silence = silence.detach().cpu().float().clone().contiguous()
        self.seal = _digest(self.silence)
        self.seed = seed
        self.binding = seed
        self.identity = deepcopy(identity)
        self.request = ContextVar(f"solar_request_{id(self)}", default=None)
        self.last = {"status": "selected_not_executed", "diffusion_calls": 0}

    def validate(self):
        if self.seed != self.binding or _digest(self.silence) != self.seal:
            raise ValueError("Solar private silence/seed binding changed after selection")

    def sampler(self, executor, guider, sigmas, extra_args, callback, noise, *args, **kwargs):
        self.validate()
        state = {"noise": None, "audio_ticks": None, "diffusion_calls": 0,
                 "attempted_calls": 0, "clock": [], "status": "running",
                 "silence_audio_row_evaluations": 0, "preserved_audio_row_evaluations": 0}
        token = self.request.set(state)
        try:
            result = executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)
            finite_tensor(result, "Solar sampler output")
            if result.shape != noise.shape:
                raise ValueError("Solar sampler output shape differs from its packed input")
            self.validate()
            if not state["diffusion_calls"]:
                state["status"] = "delegated_without_silence_consumer"
            elif not state["silence_audio_row_evaluations"]:
                state["status"] = "delegated_preserving_masked_audio"
            elif state["preserved_audio_row_evaluations"]:
                state["status"] = "silence_input_partially_delegated_unqualified"
            else:
                state["status"] = "silence_input_delegated_unqualified"
            return result
        except BaseException as error:
            state.update(status="failed", error=type(error).__name__ + ": " + str(error))
            raise
        finally:
            self.last = {key: value for key, value in state.items() if key != "noise"}
            self.request.reset(token)

    def diffusion(self, executor, x, timestep, context, transformer_options, **kwargs):
        self.validate()
        state = self.request.get()
        if state is None:
            raise ValueError("Solar silence mixing needs its active sampler request scope")
        comfy.model_management.throw_exception_if_processing_interrupted()
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("Solar diffusion requires two AV tensors")
        video, audio = (finite_tensor(value, "Solar AV input") for value in x)
        if (video.ndim != 5 or audio.ndim != 4 or tuple(audio.shape[1:3]) != (32, 2)
                or video.shape[0] != audio.shape[0]):
            raise ValueError("Solar diffusion AV batch/channel dimensions disagree")
        ticks = audio.shape[-1]
        if ticks > self.silence.shape[-1]:
            raise ValueError(f"Selected silence covers {self.silence.shape[-1]} audio ticks, actual clip needs {ticks}")
        if state["audio_ticks"] not in (None, ticks):
            raise ValueError("Solar audio extent changed inside one sampler request")
        finite_tensor(timestep, "Solar timestep")
        if timestep.numel() not in (1, audio.shape[0]) or bool((timestep < 0).any()) or bool((timestep > 1000).any()):
            raise ValueError("Solar timestep must describe each actual batch at physical time 0..1000")
        sv, sa = _core_shifts(executor, transformer_options)
        sigma_v = (timestep.flatten() / 1000.).float().clamp(min=1e-6)
        sigma_a = time_shift_sigma(sigma_v, sv, sa).reshape(-1, 1, 1, 1)
        clean = self.silence[..., :ticks].to(device=audio.device, dtype=torch.float32)
        if state["noise"] is None:
            generator = torch.Generator("cpu").manual_seed(self.seed)
            state["noise"] = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
            state["audio_ticks"] = ticks
        held = ((1 - sigma_a) * clean + sigma_a * state["noise"].to(clean.device))
        held = held.expand(audio.shape[0], -1, -1, -1).to(audio.dtype)
        editable = _replacement_rows(audio, kwargs.get("audio_denoise_mask"))
        changed_rows = int(editable.sum().item())
        kept_rows = editable.numel() - changed_rows
        if not changed_rows:
            held = audio
        elif kept_rows:
            held = torch.where(editable, held, audio)
        state["attempted_calls"] += 1
        # Only this request's new input list is replaced; the caller's streams,
        # model objects, existing wrappers and selected LoRA stack stay owned.
        result = executor([video, held], timestep, context, transformer_options, **kwargs)
        if not isinstance(result, (list, tuple)) or len(result) != 2:
            raise ValueError("Solar diffusion must return two AV predictions")
        for value, expected in zip(result, x):
            finite_tensor(value, "Solar prediction")
            if value.shape != expected.shape:
                raise ValueError("Solar prediction shape differs from its actual AV input")
        self.validate()
        state["diffusion_calls"] += 1
        state["silence_audio_row_evaluations"] += changed_rows
        state["preserved_audio_row_evaluations"] += kept_rows
        state["clock"].append({"video_sigma": sigma_v.detach().cpu().tolist(),
                               "audio_sigma": sigma_a.flatten().detach().cpu().tolist(),
                               "shifts": [sv, sa], "silence_audio_rows": changed_rows,
                               "preserved_audio_rows": kept_rows})
        return result


def configure_silence(model, silence, *, seed=0, identity=None):
    for name in ("clone", "get_attachment", "set_attachments", "add_wrapper_with_key"):
        if not callable(getattr(model, name, None)):
            raise TypeError(f"Solar MODEL requires callable {name}")
    prior = model.get_attachment(KEY)
    if prior is not None and not isinstance(prior, SilenceOwner):
        warn_patch_stack("Solar attachment key has a foreign owner; selected MODEL retained without silence installation")
        return model, {"status": "not_installed_foreign_owner", "trained_consumer_qualified": False}
    owner = SilenceOwner(silence, seed, identity)
    selected = model.clone()
    for kind, method in ((WrappersMP.SAMPLER_SAMPLE, "sampler"), (WrappersMP.DIFFUSION_MODEL, "diffusion")):
        entries = selected.wrappers.get(kind, {}).get(KEY, [])
        replaced = False
        updated = []
        for entry in entries:
            old = getattr(prior, method, None)
            if (prior is not None and getattr(entry, "__self__", None) is prior
                    and getattr(entry, "__func__", None) is getattr(old, "__func__", None)):
                updated.append(getattr(owner, method))
                replaced = True
            else:
                updated.append(entry)
        if entries:
            selected.wrappers[kind][KEY] = updated
        if not replaced:
            selected.add_wrapper_with_key(kind, KEY, getattr(owner, method))
    selected.set_attachments(KEY, owner)
    return selected, audit_silence(selected)


def audit_silence(model):
    owner = model.get_attachment(KEY)
    if not isinstance(owner, SilenceOwner):
        return {"status": "not_selected", "trained_consumer_qualified": False}
    owner.validate()
    return deepcopy({"schema": KEY, "last_request": owner.last, "request_active": owner.request.get() is not None,
                     "silence": owner.identity, "silence_ticks": owner.silence.shape[-1], "seed": owner.seed,
                     "audio_mask_policy": "silence_on_fully_editable_rows; zero_and_partial_rows_retain_incoming_audio",
                     "audio_row_counts": "batch_stereo_tick_rows_summed_over_successful_diffusion_calls",
                     "clock_source": "actual_diffusion_owner_with_transformer_options_overrides",
                     "audio_driving": False, "generated_audio_delivery_qualified": False,
                     "soundtrack": "encoded silence during evaluation; original audio requires separate mux",
                     "portable_cache_identity": False, "trained_consumer_qualified": False})
