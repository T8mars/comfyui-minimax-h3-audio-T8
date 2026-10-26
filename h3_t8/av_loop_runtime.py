"""Unregistered, opt-in H24 periodic AV prototype with actual Core consumers.

See av_loop_clock.SOURCE for the fixed upstream algorithm and GPLv3 provenance.
No upstream class-name admission rules, defaults, media or framework are copied.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from copy import copy, deepcopy
from dataclasses import asdict
import json
import math
from types import MethodType

import torch
import comfy.model_management
import comfy.model_sampling
import comfy.utils
from comfy.patcher_extension import WrappersMP
from comfy_extras.nodes_audio import vae_decode_audio
from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlPatch

from .av_loop_clock import LoopPlan, SOURCE, pcm_plan, periodic
from .h3_projection_chunks import shared_network_lease
from .patch_stack_policy import warn_patch_stack


KEY = "t8_h24_periodic_av_prototype_v1"
_REQUEST = ContextVar(KEY, default=None)
_CONTROL_METHODS = KEY + "_control_methods"
_CONTROL_GROUP = KEY + "_control_group"
_NATIVE_CONTROL_ENTRIES = ContextVar(KEY + "_native_control_entries", default=())


def _guard_control_entry(name):
    def entry(owner, *args, **kwargs):
        with _control_leases(getattr(owner, _CONTROL_GROUP)):
            method = getattr(owner, _CONTROL_METHODS)[name]
            if (name != "diffusion_model_wrapper"
                    or getattr(method, "__func__", None)
                    is not MiniMaxH3FunControlPatch.diffusion_model_wrapper):
                return method(*args, **kwargs)
            # The actual native wrapper decides active before its continuation.
            # Pending owners and unknown overrides do not carry this evidence.
            token = _NATIVE_CONTROL_ENTRIES.set((*_NATIVE_CONTROL_ENTRIES.get(), owner))
            try:
                return method(*args, **kwargs)
            finally:
                _NATIVE_CONTROL_ENTRIES.reset(token)
    return entry


# Guard the native owner's whole wrapper and its lifecycle, including cleanup
# after SAMPLER_SAMPLE. All entries acquire the same sorted group, so an outer
# control wrapper cannot mutate a view before the periodic wrapper takes a lease.
_CONTROL_ENTRIES = {name: _guard_control_entry(name) for name in (
    "diffusion_model_wrapper", "prepare_control_latent", "before_block", "after_block", "to", "cleanup")}


def _finite(value, label):
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be a finite tensor")


def _parts(values, plan):
    parts = tuple(values.unbind()) if getattr(values, "is_nested", False) else tuple(values)
    if len(parts) != 2:
        raise ValueError("periodic AV requires exactly video and stereo audio streams")
    video, audio = parts
    _finite(video, "video")
    _finite(audio, "audio")
    if (video.ndim != 5 or video.shape[1] != 24 or min(video.shape) < 1
            or audio.ndim != 4 or audio.shape[1:3] != (32, 2)
            or audio.shape[0] != video.shape[0]
            or video.shape[2] != plan.video_ticks + 2
            or audio.shape[-1] != plan.input_audio_ticks):
        raise ValueError("periodic AV requires matching [B,24,5k+2,H,W] and "
                         "[B,32,2,round((17k+5)*40/24)] shapes")
    return video, audio


def _view_payload(payload, plan, shift):
    if payload is None:
        return None
    out = dict(payload)
    # Same-shaped views have different keyframe coordinates. Never reuse a
    # stale layout solely because its dimensional signature still matches.
    out.pop("layout", None)
    if "frame_count" in out:
        out["frame_count"] = plan.length + 34 * plan.pad_clips
    if payload.get("keyframes"):
        rows = []
        for keyframe in payload["keyframes"]:
            index = keyframe.get("resolved_frame_index")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < plan.length:
                raise ValueError("keyframe resolved_frame_index is outside the actual input")
            rows.append({**keyframe, "resolved_frame_index": plan.pad_clips * 17
                         + (index + shift * 17) % plan.frames})
        out["keyframes"] = rows
    return out


@contextmanager
def _control_leases(owners):
    # Acquire the complete identity-ordered set before any snapshot/mutation.
    # These leases serialize cooperators, not ordinary native owner methods.
    ordered = sorted({id(owner): owner for owner in owners}.values(), key=id)
    with ExitStack() as stack:
        for owner in ordered:
            if not stack.enter_context(shared_network_lease(owner)):
                raise TypeError("periodic control scope needs a weak-referenceable identity lease owner")
        yield


@contextmanager
def periodic_control_scope(owner, plan, shift, target_shape):
    """Explicit capability binding for an existing Core-compatible control owner.

    This explicit scope requires exclusive/cooperating access to this owner.
    configure_loop supplies branch-local owners so ordinary source MODEL
    requests never see these temporary fields. Its input media,
    encoded cache and transient state are restored even when encode/forward fails.
    Unknown control wrappers are still delegated; only explicitly supplied owners
    are mapped, and their actual APIs/shapes are checked instead of class names.
    """
    with _control_leases(getattr(owner, _CONTROL_GROUP, (owner,))):
        with _periodic_control_state(owner, plan, shift, target_shape):
            yield


@contextmanager
def _periodic_control_state(owner, plan, shift, target_shape):
    names = ("control_video", "mask", "source_video", "control_latent",
             "control_latent_shape", "control_stream", "pristine_stream", "active")
    if not callable(getattr(owner, "prepare_control_latent", None)) or any(
            not hasattr(owner, name) for name in names):
        raise TypeError("bound temporal control needs its actual prepare/media/cache APIs")
    verify = getattr(owner, "verify_derived", None)
    if callable(verify) != hasattr(owner, "derived_content"):
        raise TypeError("derived control cache needs both verify_derived and derived_content")
    if callable(verify):
        verify()
        names += ("derived_content",)
    original = {name: getattr(owner, name) for name in names}
    try:
        for name in ("control_video", "mask", "source_video"):
            value = original[name]
            if value is None:
                continue
            _finite(value, f"control {name}")
            if value.ndim not in (3, 4) or value.shape[0] not in (1, plan.frames, plan.length):
                raise ValueError("bound control media must be static or cover one complete loop")
            core = value[:plan.frames]
            if core.shape[0] == 1:
                core = core.expand(plan.frames, *core.shape[1:])
            setattr(owner, name, periodic(core, plan.pad_clips * 17,
                                         plan.pad_clips * 17 + 5, 0, shift=shift * 17))
        owner.control_latent = owner.control_latent_shape = None
        if callable(verify):
            owner.derived_content = None
        owner.control_stream = owner.pristine_stream = None
        owner.prepare_control_latent(target_shape)
        if callable(verify):
            verify()
        _finite(owner.control_latent, "control encoded latent")
        if (owner.control_latent.ndim != 5 or owner.control_latent.shape[0] != target_shape[0]
                or tuple(owner.control_latent.shape[2:]) != tuple(target_shape[2:])):
            raise ValueError("encoded control batch/time/spatial shape differs from the actual view")
        yield
    finally:
        for name, value in original.items():
            setattr(owner, name, value)
        if callable(verify):
            verify()


def _isolate_controls(selected, controls):
    """Copy explicit mutable owner state and rebind actual native registrations.

    Weight/VAE objects and unrelated user hooks remain selected. This protects
    the source owner's media/cache/streams; it is not parallel weight loading
    or arbitrary user-closure isolation. Unknown callable patches are retained.
    """
    controls = tuple({id(value): value for value in controls}.values())
    if not controls:
        return (), ()
    replacements, registered, patched = {}, set(), set()
    for entries in selected.wrappers.get(WrappersMP.DIFFUSION_MODEL, {}).values():
        for wrapper in entries:
            for original in controls:
                method = getattr(original, "diffusion_model_wrapper", None)
                if (getattr(wrapper, "__self__", None) is original and hasattr(wrapper, "__func__")
                        and wrapper.__func__ is getattr(method, "__func__", None)):
                    registered.add(id(original))
    with _control_leases(controls):
        for original in controls:
            local = copy(original)
            fields = getattr(local, "__dict__", None)
            if local is original or (fields is not None and fields is getattr(original, "__dict__", None)):
                raise TypeError("periodic control owner copy must have independent mutable fields")
            replacements[id(original)] = (original, local)
            # Preserve an explicitly attached bound user method on the copy.
            for name, value in (fields or {}).items():
                if getattr(value, "__self__", None) is original and hasattr(value, "__func__"):
                    setattr(local, name, MethodType(value.__func__, local))
    locals_ = tuple(replacements[id(original)][1] for original in controls)
    for original, local in zip(controls, locals_):
        methods = {}
        for name, guard in _CONTROL_ENTRIES.items():
            method = getattr(original, name, None)
            if not callable(method):
                continue
            if getattr(method, "__func__", None) is guard:
                method = getattr(original, _CONTROL_METHODS)[name]
            if getattr(method, "__self__", None) is original and hasattr(method, "__func__"):
                method = MethodType(method.__func__, local)
            methods[name] = method
            setattr(local, name, MethodType(guard, local))
        setattr(local, _CONTROL_METHODS, methods)
        setattr(local, _CONTROL_GROUP, locals_)

    memo = {}

    def replace(value):
        pair = replacements.get(id(getattr(value, "__self__", None)))
        if pair is not None and getattr(value, "__self__", None) is pair[0] and hasattr(value, "__func__"):
            for name in _CONTROL_ENTRIES:
                if getattr(getattr(pair[0], name, None), "__func__", None) is value.__func__:
                    return getattr(pair[1], name)
            return MethodType(value.__func__, pair[1])
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, tuple):
            return tuple(replace(item) for item in value)
        # The native block-patch contract carries an owner and a previous patch.
        # Copy that chain, retaining arbitrary previous callables unchanged.
        if hasattr(value, "control_patch") and hasattr(value, "previous"):
            if id(value) in memo:
                return memo[id(value)]
            previous = replace(value.previous)
            pair = replacements.get(id(value.control_patch))
            if pair is None and previous is value.previous:
                return value
            local_patch = copy(value)
            if local_patch is value:
                raise TypeError("periodic control block patch copy must be independent")
            memo[id(value)] = local_patch
            if pair is not None and value.control_patch is pair[0]:
                local_patch.control_patch = pair[1]
                patched.add(id(pair[0]))
            local_patch.previous = previous
            return local_patch
        return value

    selected.wrappers = replace(selected.wrappers)
    selected.callbacks = replace(selected.callbacks)
    selected.model_options = replace(selected.model_options)
    if registered != patched:
        raise ValueError("explicit periodic control registration needs both its actual diffusion wrapper and block-patch chain")
    pending = tuple(local for original, local in zip(controls, locals_)
                    if id(original) not in registered)
    return locals_, pending


def _noise_contract(sampling):
    """Recognize executable Core math, never a selected model's class/name."""
    scaling = getattr(sampling, "noise_scaling", None)
    input_fn = getattr(sampling, "calculate_input", None)
    known = (getattr(scaling, "__func__", None) is comfy.model_sampling.CONST.noise_scaling
             and getattr(input_fn, "__func__", None) is comfy.model_sampling.CONST.calculate_input)
    if not known:
        return None
    scale = float(getattr(sampling, "noise_scale", 1.0))
    if not math.isfinite(scale):
        raise ValueError("actual selected sampling noise_scale must be finite")
    return scale


def _audio_weight(count, rotated_joint, like):
    positions = torch.arange(count, dtype=torch.float32, device=like.device) + .5
    own = torch.minimum((positions - rotated_joint) % count, (rotated_joint - positions) % count)
    first = torch.minimum(positions, count - positions)
    return (own / (own + first)).to(like.dtype).reshape(1, 1, 1, -1)


class _Owner:
    def __init__(self, plan, controls):
        self.plan, self.controls = plan, tuple(controls)
        self.binding = json.dumps(asdict(plan), sort_keys=True)
        self.last = {"status": "selected_not_executed", "forward_calls": 0, "view_calls": 0}

    def validate(self):
        # Check our own mutable Python object, never a user's model class name.
        if json.dumps(asdict(self.plan), sort_keys=True) != self.binding:
            raise ValueError("periodic plan binding changed after selection")
        self.plan.__post_init__()

    def sampler(self, executor, guider, sigmas, extra_args, callback, noise, *args, **kwargs):
        self.validate()
        shapes = getattr(guider.inner_model, "latent_shapes", None)
        if shapes is None:
            raise ValueError("periodic sampler needs the actual packed AV latent_shapes")
        unpacked = comfy.utils.unpack_latents(noise, shapes)
        video, _ = _parts(unpacked, self.plan)
        sampling = guider.inner_model.model_sampling
        scale = _noise_contract(sampling)
        if scale is None:
            warn_patch_stack("periodic geometry delegates the selected sampling transform; "
                             "fresh padding noise is not applied because its input transform is unbound")
        generator = torch.Generator(device=video.device).manual_seed(
            (int(extra_args.get("seed") or 0) + 1) % (2**64))
        disabled = not bool(torch.count_nonzero(video))
        swap = torch.zeros_like(video) if disabled or scale is None else (
            torch.randn(video.shape, generator=generator, device=video.device, dtype=video.dtype) - video) * scale
        state = {"owner": self, "swap": swap, "sampling": sampling, "sampling_host": guider.inner_model,
                 "noise_scale": scale, "fresh_padding_noise_bound": scale is not None,
                 "fresh_padding_noise_applied": False,
                 "forward_calls": 0, "view_calls": 0,
                 "noise_disabled": not bool(torch.count_nonzero(video)), "status": "running"}
        token = _REQUEST.set(state)
        try:
            result = executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)
            _finite(result, "sampler output")
            if result.shape != noise.shape:
                raise ValueError("sampler output shape differs from its actual packed AV input")
            self.validate()
            state["status"] = ("periodic_consumer_executed_unqualified" if state["forward_calls"]
                               else "delegated_without_periodic_consumer")
            return result
        except BaseException as error:
            state["status"] = "failed"
            state["error"] = type(error).__name__ + ": " + str(error)
            raise
        finally:
            self.last = {key: value for key, value in state.items()
                         if key not in ("owner", "swap", "sampling", "sampling_host")}
            _REQUEST.reset(token)

    def diffusion(self, executor, x, timestep, context, transformer_options, **kwargs):
        self.validate()
        state = _REQUEST.get()
        if state is None or state["owner"] is not self:
            raise ValueError("periodic diffusion requires its request-scoped sampler noise owner")
        if (state["sampling_host"].model_sampling is not state["sampling"]
                or _noise_contract(state["sampling"]) != state["noise_scale"]):
            raise ValueError("selected sampling noise transform changed during the periodic request")
        plan = self.plan
        video, audio = _parts(x, plan)
        swap = state["swap"]
        if video.shape[0] % swap.shape[0]:
            raise ValueError("diffusion batch is not an integral expansion of sampler noise")
        swap = swap.repeat(video.shape[0] // swap.shape[0], 1, 1, 1, 1)
        if swap.shape != video.shape:
            raise ValueError("request noise shape no longer matches the diffusion video")
        sigma = transformer_options.get("sigmas")
        _finite(sigma, "current sigma")
        if sigma.numel() not in (1, video.shape[0]):
            raise ValueError("current sigma count must match the diffusion batch")
        sigma = sigma.reshape(-1, 1, 1, 1, 1).to(video)
        tail = plan.input_audio_ticks - plan.audio_ticks
        predictions = []
        for shift in plan.shifts:
            comfy.model_management.throw_exception_if_processing_interrupted()
            info = plan.view(shift)
            local_args = dict(kwargs)
            local_args["minimax_payload"] = _view_payload(kwargs.get("minimax_payload"), plan, shift)
            opts = dict(transformer_options)
            opts.pop("minimax_h3_layout", None)
            opts[KEY + "_view"] = info
            padded = periodic(video[:, :, :plan.video_ticks], plan.video_pad, plan.video_pad + 2,
                              2, shift=info["shift_video_ticks"])
            padded_audio = periodic(audio[..., :plan.audio_ticks], plan.audio_pad, plan.audio_pad + tail,
                                    -1, shift=info["shift_audio_ticks"])
            noise_delta = periodic(swap[:, :, :plan.video_ticks], plan.video_pad, plan.video_pad + 2,
                                   2, shift=info["shift_video_ticks"])
            noise_delta[:, :, plan.video_pad:plan.video_pad + plan.video_ticks] = 0
            for name, stream, ticks, pad, extra, dim, rotation in (
                ("denoise_mask", video, plan.video_ticks, plan.video_pad, 2, 2, info["shift_video_ticks"]),
                ("audio_denoise_mask", audio, plan.audio_ticks, plan.audio_pad, tail, -1, info["shift_audio_ticks"]),
            ):
                mask = kwargs.get(name)
                if mask is None:
                    continue
                _finite(mask, name)
                try:
                    if torch.broadcast_shapes(mask.shape, stream.shape) != stream.shape:
                        raise ValueError("broadcast result differs from stream")
                    expanded = mask.expand_as(stream)
                except (RuntimeError, ValueError) as error:
                    raise ValueError(f"{name} cannot broadcast to its actual AV stream") from error
                cut = expanded.narrow(dim, 0, ticks)
                local_args[name] = periodic(cut, pad, pad + extra, dim, shift=rotation)
                if name == "denoise_mask":
                    noise_delta = noise_delta * local_args[name]
            state["fresh_padding_noise_applied"] |= bool(torch.count_nonzero(sigma * noise_delta))
            padded = padded + sigma * noise_delta.to(padded)
            with ExitStack() as stack:
                stack.enter_context(_control_leases(self.controls))
                for control in self.controls:
                    if (any(entered is control for entered in _NATIVE_CONTROL_ENTRIES.get())
                            and control.active is False):
                        continue
                    stack.enter_context(_periodic_control_state(control, plan, shift, tuple(padded.shape)))
                output = executor([padded, padded_audio], timestep, context, opts, **local_args)
            if not isinstance(output, (list, tuple)) or len(output) != 2:
                raise ValueError("periodic consumer must return two actual AV tensors")
            for out, expected in zip(output, (padded, padded_audio)):
                _finite(out, "periodic prediction")
                if out.shape != expected.shape:
                    raise ValueError("periodic consumer prediction shape differs from padded input")
            v = output[0][:, :, plan.video_pad:plan.video_pad + plan.video_ticks].roll(-info["shift_video_ticks"], 2)
            a = output[1][..., plan.audio_pad:plan.audio_pad + plan.audio_ticks].roll(-info["shift_audio_ticks"], -1)
            predictions.append((v, a))
            state["view_calls"] += 1
        v, a = predictions[0]
        if len(predictions) == 2:
            second_shift = plan.view(plan.shifts[1])["shift_audio_ticks"]
            a = torch.lerp(a, predictions[1][1], _audio_weight(plan.audio_ticks, plan.audio_ticks - second_shift, a))
        self.validate()
        state["forward_calls"] += 1
        return [periodic(v, 0, 2, 2), periodic(a, 0, tail, -1)]


def configure_loop(model, plan, *, control_owners=(), enabled=True):
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    if not enabled:
        return model, {"status": "disabled", "quality_accepted": False}
    if not isinstance(plan, LoopPlan):
        raise TypeError("an explicit LoopPlan is required")
    plan.__post_init__()
    for name in ("clone", "get_attachment", "set_attachments", "add_wrapper_with_key"):
        if not callable(getattr(model, name, None)):
            raise TypeError(f"periodic MODEL needs actual {name} capability")
    prior = model.get_attachment(KEY)
    if prior is not None and not isinstance(prior, _Owner):
        warn_patch_stack("periodic key already has a foreign owner; leaving that MODEL intact")
        return model, {"status": "not_installed_foreign_owner", "quality_accepted": False}
    selected = model.clone()
    controls = tuple(control_owners) or (() if prior is None else prior.controls)
    local_controls, pending = _isolate_controls(selected, controls)
    owner = _Owner(plan, local_controls)
    if selected.wrappers:
        warn_patch_stack("periodic wrapper preserves the selected external stack; unbound temporal controls remain unqualified")
    for kind, method in ((WrappersMP.SAMPLER_SAMPLE, "sampler"), (WrappersMP.DIFFUSION_MODEL, "diffusion")):
        entries = selected.wrappers.get(kind, {}).get(KEY, [])
        if prior is not None:
            prior.validate()
            old = getattr(prior, method)
            def own(entry):
                return (getattr(entry, "__self__", None) is prior
                        and getattr(entry, "__func__", None) is old.__func__)
            selected.wrappers.setdefault(kind, {})[KEY] = [getattr(owner, method) if own(entry) else entry for entry in entries]
            if not any(own(entry) for entry in entries):
                selected.add_wrapper_with_key(kind, KEY, getattr(owner, method))
        else:
            selected.add_wrapper_with_key(kind, KEY, getattr(owner, method))
    # Bind newly supplied owners on this branch only, after the periodic wrapper.
    # Already registered owners were rebound in place without reordering hooks.
    for control in pending:
        register = getattr(control, "register", None)
        if not callable(register):
            raise TypeError("unregistered explicit control needs its actual register(MODEL) callable")
        register(selected)
    selected.set_attachments(KEY, owner)
    return selected, audit_loop(selected)


def audit_loop(model):
    owner = model.get_attachment(KEY)
    if not isinstance(owner, _Owner):
        return {"status": "not_selected", "quality_accepted": False}
    owner.validate()
    state = _REQUEST.get()
    return deepcopy({"schema": KEY, "source": SOURCE, "plan": owner.plan.report(),
                     "last_request": owner.last, "request_active": state is not None and state["owner"] is owner,
                     "explicit_control_owners": len(owner.controls), "portable_cache_identity": False,
                     "control_state_isolation": "branch_local_explicit_owner_and_bound_patch_chain",
                     "control_leases": "identity_ordered_cooperating_scopes_only",
                     "regional_derived_cache_lifecycle_bound": True,
                     "regional_gate_clock_remapped": False,
                     "user_closure_state_isolated": False,
                     "concurrent_control_owner_access_qualified": False,
                     "trained_GPU_qualified": False, "quality_accepted": False})


def decode_loop(samples, video_vae, audio_vae, plan):
    """Delegate real decoder owners, then crop by actual video/PCM clocks."""
    video, audio = _parts(samples["samples"], plan)
    if video.shape[0] != 1:
        raise ValueError("IMAGE loop output needs one video batch")
    if not callable(getattr(video_vae, "decode", None)) or not callable(getattr(audio_vae, "decode", None)):
        raise TypeError("both selected VAEs need an actual decode callable")
    comfy.model_management.throw_exception_if_processing_interrupted()
    padded_v = periodic(video[:, :, :plan.video_ticks], plan.video_pad, plan.video_pad + 2, 2)
    pixels = video_vae.decode(padded_v)
    _finite(pixels, "video decoder output")
    if pixels.ndim == 5 and pixels.shape[0] == 1:
        pixels = pixels.flatten(0, 1)
    expected_frames = plan.length + 34 * plan.pad_clips
    if pixels.ndim != 4 or pixels.shape[0] != expected_frames or pixels.shape[-1] not in (3, 4):
        raise ValueError(f"video decoder must return {expected_frames} actual channel-last frames")
    images = pixels[plan.pad_clips * 17:plan.pad_clips * 17 + plan.frames]
    comfy.model_management.throw_exception_if_processing_interrupted()
    padded_a = periodic(audio[..., :plan.audio_ticks], plan.audio_pad, plan.audio_pad, -1)
    sound = vae_decode_audio(audio_vae, {"samples": padded_a})
    waveform = sound["waveform"]
    _finite(waveform, "audio decoder output")
    if waveform.ndim != 3 or waveform.shape[:2] != (1, 2):
        raise ValueError("audio decoder must return actual [1,2,PCM] stereo")
    crop = pcm_plan(plan, sound["sample_rate"], padded_a.shape[-1], waveform.shape[-1])
    sound = {**sound, "waveform": waveform[..., crop["start_sample"]:crop["end_sample"]]}
    return images, sound, {"schema": KEY + "_decode", "plan": plan.report(), "pcm": crop,
                           "video_frames": images.shape[0], "source": deepcopy(SOURCE),
                           "trained_GPU_qualified": False, "quality_accepted": False}
