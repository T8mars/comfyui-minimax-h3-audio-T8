"""Independent pruned-curve owner; original full HyperFlow remains untouched.

Local EXP module, NOT yet registered. Uses native H3 forward delegation,
the original T8 interval Euler, exact per-projection fit, and call-local row
contexts. Does not pretend a selected source file proves all live MODEL weights.
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
import hashlib
import math
import uuid

import comfy.lora
import comfy.patcher_extension
import comfy.samplers
import torch

from . import hyperflow_curve_fit_exp as fitting
from . import hyperflow_runtime_advanced as full_runtime
from .hyperflow_sampling_advanced import HyperFlowPlan, sample_hyperflow_continuous, shifted_grid
from .hyperflow_weights_advanced import HyperFlowWeights
from .core import nested_av_parts
from .sampling import _make_sampling, model_uses_raw_audio_velocity

KEY = "t8_h3_hyperflow_curve_exp_v1"
DETAILS_KEY = "t8_h3_hyperflow_curve_details_exp_v1"
ACTIVE = contextvars.ContextVar("t8_h3_hyperflow_curve_forward", default=None)


@dataclass(frozen=True)
class CurveBinding:
    owner: str
    identity: str
    adapter_sha256: str
    fit_sha256: str
    base_sha256: str
    teacher_sha256: str
    raw_sigmas: tuple
    gate: float
    model_identity: int
    adapter_content_sha256: str
    basis_content_sha256: str


@dataclass(frozen=True)
class CurveDetails:
    weights: HyperFlowWeights
    fit: fitting.CurveFit
    audit: dict


@dataclass
class Forward:
    native: object
    coordinates: torch.Tensor | None = None
    consumed: list[int] = field(default_factory=list)
    verified: list[int] = field(default_factory=list)


def _unwrap(function):
    from .vdn_attention_compat import _factory_closure
    seen = set()
    while getattr(function, "_t8_curve_owner", None) is not None:
        # Marker attributes alone do not authorize stripping a user's callable.
        block = _factory_closure(function, _install, "block_forward")
        final = _factory_closure(function, _install, "final_forward")
        if block is None and final is None:
            break
        if id(function) in seen:
            raise ValueError("Cyclic curve forward owner")
        seen.add(id(function))
        function = function._t8_curve_inner
    return function


def _curve_context(owner, diffusion, args, kwargs):
    """Match Core's mask clocks, independently of conditioning augmentation.

    Core uses payload augmentation for conditioning rows, but its two mask
    pins are fixed constants. Never infer a masked stream row from an unrelated
    conditioning pin. The original full HyperFlow implementation is unchanged.
    """
    from comfy.ldm.minimax import model as native
    plain = dict(kwargs)
    plain.pop("denoise_mask", None)
    plain.pop("audio_denoise_mask", None)
    ctx = full_runtime._forward_context(owner, diffusion, args, plain)
    timestep = args[1] if len(args) > 1 else kwargs["timestep"]
    sigma_v = (timestep.flatten()[0] / 1000.).float().clamp(min=1e-6)
    sigma_next = sigma_v.new_tensor(ctx.step.sigma_video_next)
    sigma_a = native.time_shift_sigma(sigma_v, 12., 3.)
    sigma_a_next = native.time_shift_sigma(sigma_next, 12., 3.)
    video_mask, audio_mask = kwargs.get("denoise_mask"), kwargs.get("audio_denoise_mask")
    if video_mask is not None:
        levels = video_mask[0, 0].to(torch.float32)
        full_runtime._mask_time_endpoints(levels, sigma_v, sigma_next, max(ctx.t_video, native.VISUAL_COND_TIMESTEP))
        video_x = args[0][0] if args else kwargs["x"][0]
        length, height, width = video_x.shape[2:]
        levels = native.mask_row_values(levels, length, (height + 1) // 2 * 2, (width + 1) // 2 * 2)
        ctx.video_masks = full_runtime._mask_time_endpoints(
            levels, sigma_v, sigma_next, max(ctx.t_video, native.VISUAL_COND_TIMESTEP))
    if audio_mask is not None:
        levels = audio_mask[0, 0].to(torch.float32).reshape(-1)
        maps = full_runtime._mask_time_endpoints(
            levels, sigma_a, sigma_a_next, max(ctx.t_audio, native.AUDIO_COND_TIMESTEP))
        ctx.audio_masks = {} if bool((levels >= 1. - 1e-3).all()) else maps
    return ctx


def _mark(function, original, owner):
    function._t8_curve_owner = owner
    function._t8_curve_inner = original
    return function


def _native_projection(delegate, module):
    """Only authenticated native delegates can certify projection coverage.

    Opaque users still run, including a callable which delegates correctly;
    its invocation alone cannot prove what it consumed internally.
    """
    from comfy.ldm.minimax import model as native
    from .modular_sampling.hyperflow_identity import _native
    from .patch_stack_policy import UnverifiedModelStack
    if type(module) not in (native.DiTBlock, native.FinalLayer) or type(module.adaln_proj) is not native.AdalnProj:
        return False
    try:
        _native(delegate, module)
        for child in (module, module.adaln_proj, module.adaln_proj.linear):
            if child._forward_pre_hooks or child._forward_hooks:
                return False
        _native(module.adaln_proj.forward, module.adaln_proj)
        _native(module.adaln_proj.linear.forward, module.adaln_proj.linear)
        return type(module.adaln_proj.linear).__module__ in {"comfy.ops", "torch.nn.modules.linear"}
    except UnverifiedModelStack:
        return False


def verify_basis(model, fit, base_path):
    fit.verify()
    actual_file = fitting._file(base_path)
    if (actual_file["sha256"], actual_file["size"]) != (
            fit.metadata["base"]["sha256"], fit.metadata["base"]["size"]):
        raise ValueError("Actual selected pruned base file does not match fit")
    from safetensors import safe_open
    from .long_video_dual_identity import _original_state
    state = _original_state(model, model.model.state_dict())
    with safe_open(str(base_path), framework="pt", device="cpu") as handle:
        prefix = fitting._prefix(handle)
        if fitting.basis_identity(handle, prefix) != fit.metadata["basis_sha256"]:
            raise ValueError("Fit stored basis differs from its actual base file")
        for name in fit.metadata["basis_sha256"]:
            original = handle.get_tensor(prefix + name)
            actual = state.get("diffusion_model." + name)
            if actual is None:
                actual = state.get(name)
            if (not isinstance(actual, torch.Tensor) or actual.is_meta
                    or actual.shape != original.shape):
                raise ValueError("Missing materialized native curve basis: " + name)
            # Native pruned AdaLN is FP32, loaded from FP16. Compare exact
            # numerical values in FP32, not a lossy round-trip to FP16.
            if not torch.equal(actual.detach().float().cpu(), original.float()):
                raise ValueError("Actual loaded curve basis differs from fit: " + name)


def _curve_basis_identity(model, fit):
    from .long_video_dual_identity import _original_state, content_identity
    state = _original_state(model, model.model_state_dict())
    actual = {}
    for name in fit.metadata["basis_sha256"]:
        tensor = state.get("diffusion_model." + name)
        if tensor is None:
            tensor = state.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.is_meta:
            raise ValueError("Curve basis lost its actual materialized tensor: " + name)
        actual[name] = content_identity(tensor.detach().float().cpu().contiguous())
    return hashlib.sha256(fitting.canonical(actual).encode()).hexdigest()


def _recipe(weights, fit):
    from safetensors import safe_open
    fit.verify()
    with safe_open(str(fit.path), framework="pt", device="cpu") as handle:
        import json
        original_metadata = json.loads((handle.metadata() or {})["t8_curve_fit"])
    if fitting.canonical(original_metadata) != fitting.canonical(fit.metadata):
        raise ValueError("Curve fit in-memory metadata differs from actual immutable file")
    data = fit.metadata
    meta = weights.metadata
    if (data["gate"], data["strength"], data["video_shift"], data["audio_shift"], tuple(data["raw_sigmas"])) != (
            meta.gate, 1.0, meta.video_shift, meta.audio_shift, meta.raw_sigmas):
        raise ValueError("Curve fit recipe and original adapter disagree")
    if (data["adapter"]["sha256"] != weights.source_sha256 or data["quality_verified"] is not False):
        raise ValueError("Curve fit source adapter/qualification mismatch")
    t, r = fitting.trained_times(meta)
    if not torch.equal(t, fit.t) or not torch.equal(r, fit.r):
        raise ValueError("Curve fit trained AV pairs differ from actual adapter")


def _coordinates(fit, pairs, device):
    result = []
    for t, r in pairs:
        t, r = full_runtime._f32(t), full_runtime._f32(r)
        if t == r:
            if not 0 <= t <= 1:
                raise ValueError("Pinned curve time outside [0,1]")
            value = fitting.interpolate(fit.pinned.to(device), torch.tensor([t], device=device)).squeeze(1)
        else:
            distance = torch.maximum((fit.t - t).abs(), (fit.r - r).abs())
            matches = (distance <= 2e-6).nonzero().reshape(-1)
            if not matches.numel():
                raise ValueError("Actual off-grid/soft-mask two-time pair has no trained curve fit; use full HyperFlow")
            value = fit.generated[:, int(matches[0])].to(device)
        result.append(value)
    return torch.stack(result, dim=1)


def _install(patched, diffusion, fit, owner, audit, basis_content):
    immutable_metadata = fitting.canonical(fit.metadata)
    def time_embedder(times):
        ctx = ACTIVE.get()
        if ctx is None or ctx.native.step.owner != owner:
            raise RuntimeError("Curve embedder requires its own interval forward context")
        native = ctx.native
        values = [full_runtime._f32(float(t)) for t in times.tolist()]
        pairs = [(t, t) for t in values]
        positions = {pair: i for i, pair in enumerate(pairs)}

        def add(t, r):
            pair = (full_runtime._f32(t), full_runtime._f32(r))
            if pair not in positions:
                positions[pair] = len(pairs)
                pairs.append(pair)
            return positions[pair]

        maps = {"pin": list(range(len(values))), "video": [], "audio": []}
        for t in values:
            rv = native.video_masks.get(t, native.r_video if t == native.t_video else None)
            ra = native.audio_masks.get(t, native.r_audio if t == native.t_audio else None)
            # Unused conditioning/other-stream rows need no generated mapping.
            # If Core actually selects one, remap must refuse, never pin it.
            maps["video"].append(-1 if rv is None else add(t, rv))
            maps["audio"].append(-1 if ra is None else add(t, ra))
        if not {native.t_video, native.t_audio}.issubset(values):
            raise RuntimeError("Native Core omitted generated AV time rows")
        native.role_maps = maps
        ctx.coordinates = _coordinates(fit, pairs, times.device)
        audit["pairs"].append({"interval": native.step.absolute_index, "pairs": pairs})
        return ctx.coordinates[0]

    patched.add_object_patch("diffusion_model.use_adaln_curves", False)
    # Pruned Core has no time_embedder attribute. Object patch inserts a callable
    # and restores its prior absence; no module registration/constructor change.
    patched.add_object_patch("diffusion_model.time_embedder", time_embedder)

    def remap(native, row, role):
        indices = native.role_maps[role]
        if isinstance(row, int):
            selected = indices[row]
            if selected < 0:
                raise ValueError("Actual Core stream row has no curve two-time mapping; use full HyperFlow")
            return selected
        key = (role, row.device)
        if key not in native.tensor_maps:
            native.tensor_maps[key] = row.new_tensor(indices)
        selected = native.tensor_maps[key][row]
        if bool((selected < 0).any()):
            raise ValueError("Actual Core stream row has no curve two-time mapping; use full HyperFlow")
        return selected

    for index, block in enumerate(diffusion.blocks):
        original = _unwrap(patched.get_model_object(f"diffusion_model.blocks.{index}.forward"))

        def block_forward(x, t_emb, mod_segments, rope, *args, _inner=original, _index=index, **kwargs):
            ctx = ACTIVE.get()
            if ctx is None or ctx.coordinates is None or ctx.native.step.owner != owner:
                raise RuntimeError("Curve block invoked outside its matching interval")
            native = ctx.native
            if native.original_segments is not mod_segments:
                layout = native.transformer_options.get("minimax_h3_layout")
                if layout is None:
                    raise RuntimeError("Curve forward requires actual native packed AV layout")
                source = iter(layout.segments)
                _, stop, kind = next(source)
                segments = []
                for a, b, row in mod_segments:
                    while a >= stop:
                        _, stop, kind = next(source)
                    role = "video" if kind in ("text", "video") else "audio" if kind == "audio" else "pin"
                    segments.append((a, b, remap(native, row // 3, role) * 3 + row % 3))
                native.original_segments, native.remapped_segments = mod_segments, segments
            # Pruned Core normally keeps curve coordinates FP32, even when
            # backbone x/t_emb-placeholder is BF16. Do not round this fit.
            verified = _native_projection(_inner, diffusion.blocks[_index])
            result = _inner(x, ctx.coordinates[_index].to(device=t_emb.device), native.remapped_segments, rope, *args, **kwargs)
            ctx.consumed.append(_index)
            if verified:
                ctx.verified.append(_index)
            return result

        patched.add_object_patch(f"diffusion_model.blocks.{index}.forward", _mark(block_forward, original, owner))
    original_final = _unwrap(patched.get_model_object("diffusion_model.final_layer.forward"))

    def final_forward(x, t_emb, video_seg, audio_seg, *args, **kwargs):
        ctx = ACTIVE.get()
        if ctx is None or ctx.coordinates is None or ctx.native.step.owner != owner:
            raise RuntimeError("Curve final projection invoked outside matching interval")
        video_seg = (*video_seg[:2], remap(ctx.native, video_seg[2], "video"))
        audio_seg = (*audio_seg[:2], remap(ctx.native, audio_seg[2], "audio"))
        verified = _native_projection(original_final, diffusion.final_layer)
        result = original_final(x, ctx.coordinates[50].to(device=t_emb.device), video_seg, audio_seg, *args, **kwargs)
        ctx.consumed.append(50)
        if verified:
            ctx.verified.append(50)
        return result

    patched.add_object_patch("diffusion_model.final_layer.forward", _mark(final_forward, original_final, owner))

    def scope(executor, *args, **kwargs):
        fit.verify()
        if fitting.canonical(fit.metadata) != immutable_metadata:
            raise ValueError("Curve fit metadata mutated after installation")
        if _curve_basis_identity(patched, fit) != basis_content:
            raise ValueError("Actual curve basis changed after its fitted installation")
        native = _curve_context(owner, diffusion, args, kwargs)
        ctx = Forward(native)
        token = ACTIVE.set(ctx)
        try:
            result = executor(*args, **kwargs)
            # Unknown external blocks keep executing even if they skip our
            # original block. Coverage report stays unverified, not a ban.
            audit["forwards"].append({"interval": native.step.absolute_index,
                "invoked_projection_wrappers": list(ctx.consumed),
                "verified_native_projections": list(ctx.verified),
                "complete_51": ctx.verified == list(range(51))})
            return result
        finally:
            ACTIVE.reset(token)

    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                 "t8_curve_" + owner, scope)


def install_curve(model, weights: HyperFlowWeights, fit: fitting.CurveFit, base_path):
    if model.get_attachment(KEY) is not None or model.get_attachment(full_runtime.ATTACHMENT_KEY) is not None:
        raise ValueError("HyperFlow already applied; branch from the selected base")
    _recipe(weights, fit)
    verify_basis(model, fit, base_path)
    diffusion = model.get_model_object("diffusion_model")
    if (getattr(diffusion, "use_adaln_curves", None) is not True
            or len(diffusion.blocks) != 50 or len(diffusion.token_refiner.blocks) != 2):
        raise ValueError("This independent route requires native fifty-block pruned H3")
    modules = dict(diffusion.named_modules())
    state, mapping = {}, {}
    for name, (a, b, alpha) in weights.patches.items():
        if name.startswith("time_embedder."):
            continue  # Full teacher time adapters are consumed by the fit.
        module = modules.get(name)
        weight = getattr(module, "weight", None)
        if weight is None or tuple(weight.shape) != (b.shape[0], a.shape[1]):
            raise ValueError("Original HyperFlow backbone shape mismatch: " + name)
        state[name + ".lora_A.weight"] = a
        state[name + ".lora_B.weight"] = b
        state[name + ".alpha"] = torch.tensor(alpha)
        mapping[name] = "diffusion_model." + name + ".weight"
    if len(mapping) != 208:
        raise ValueError("Curve route must consume every 208 original backbone targets")
    parsed = comfy.lora.load_lora(state, mapping, log_missing=False)
    if set(parsed) != set(mapping.values()):
        raise ValueError("Incomplete original HyperFlow pruned backbone mapping")
    patched = model.clone()
    if set(patched.add_patches(parsed)) != set(mapping.values()):
        raise ValueError("Actual pruned MODEL did not accept every original backbone target")
    digest = hashlib.sha256(fitting.canonical({"fit": fit.sha256,
        "base": fit.metadata["base"]["sha256"], "teacher": fit.metadata["teacher"]["sha256"],
        "adapter": weights.source_sha256}).encode()).hexdigest()
    from .long_video_dual_identity import content_identity
    adapter_content = hashlib.sha256(fitting.canonical(content_identity({
        "patches": weights.patches, "endpoint": weights.endpoint,
        "metadata": vars(weights.metadata)})).encode()).hexdigest()
    basis_content = _curve_basis_identity(model, fit)
    binding = CurveBinding(uuid.uuid4().hex, digest, weights.source_sha256, fit.sha256,
                           fit.metadata["base"]["sha256"], fit.metadata["teacher"]["sha256"],
                           weights.metadata.raw_sigmas, weights.metadata.gate, id(model.model), adapter_content, basis_content)
    audit = {"forwards": [], "pairs": [], "portable_cache_reuse": False,
             "full_backbone_file_identity_certified": False, "human_quality_accepted": False}
    _install(patched, diffusion, fit, binding.owner, audit, basis_content)
    patched.set_attachments(KEY, binding)
    patched.set_attachments(DETAILS_KEY, CurveDetails(weights, fit, audit))
    return patched, binding, audit


def build_plan(model, start=0, stop=8):
    binding = model.get_attachment(KEY)
    if type(binding) is not CurveBinding or id(model.model) != binding.model_identity:
        raise ValueError("Curve Plan requires its own exact MODEL instance")
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= 8:
        raise ValueError("Curve absolute interval must satisfy 0 <= start < stop <= 8")
    return HyperFlowPlan(binding.owner, binding.identity, binding.raw_sigmas,
                         tuple(float(x) for x in shifted_grid(binding.raw_sigmas, 12)),
                         tuple(float(x) for x in shifted_grid(binding.raw_sigmas, 3)),
                         start, stop, recipe="hyperflow8_pruned_curve_approx_exp_v1")


def setup_sampler(model, av_latent, plan, *, continuation=False, x_sigma=None, capture=None):
    binding = model.get_attachment(KEY)
    if (type(binding) is not CurveBinding or id(model.model) != binding.model_identity
            or (binding.owner, binding.identity) != (plan.owner, plan.source_sha256)
            or plan.recipe != "hyperflow8_pruned_curve_approx_exp_v1"):
        raise ValueError("Curve MODEL/PLAN identity mismatch")
    if plan != build_plan(model, plan.start_interval, plan.stop_interval):
        raise ValueError("Curve actual AV plan changed")
    if not continuation and (plan.start_interval, plan.stop_interval) != (0, 8):
        raise ValueError("Partial curve plan requires captured x_sigma, not generic restart")
    if x_sigma is not None and (not continuation or plan.start_interval == 0):
        raise ValueError("Curve state override needs an actual nonzero continuation")
    if plan.start_interval > 0 and not callable(x_sigma):
        raise ValueError("Nonzero curve continuation requires its captured x_sigma provider")
    video, audio = nested_av_parts(av_latent)
    if video.shape[1] != 24 or audio.shape[1:3] != (32, 2):
        raise ValueError("Curve sampler needs native joint AV latent")
    branch = model.clone()
    branch.add_object_patch("model_sampling", _make_sampling(model, model.get_model_object("model_sampling"), 12., 3., False))
    options = dict(branch.model_options.get("transformer_options", {}))
    options.update(minimax_h3_sigma_shift_video=12., minimax_h3_sigma_shift_audio=3.)
    branch.model_options["transformer_options"] = options
    video_values = math.prod(video.shape[1:])
    packed_values = video_values + math.prod(audio.shape[1:])
    raw_audio = model_uses_raw_audio_velocity(model)

    def sampler(network, x, sigmas, extra_args=None, callback=None, disable=None):
        if not torch.equal(sigmas.cpu(), plan.video_segment):
            raise ValueError("Curve sampler actual sigma grid differs from typed plan")
        if x_sigma is not None:
            saved = x_sigma()
            if saved.shape != x.shape or not bool(torch.isfinite(saved).all()):
                raise ValueError("Curve captured state shape or values changed")
            x = saved.to(x, copy=True)
        result = sample_hyperflow_continuous(network, x, plan, video_values=video_values,
            packed_values=packed_values, audio_velocity_is_raw=raw_audio,
            extra_args=extra_args, callback=callback, disable=disable,
            skip_start_rebase=continuation and plan.start_interval > 0)
        if capture is not None:
            capture(result.detach().to(device="cpu", dtype=torch.float32, copy=True))
        return result

    return branch, comfy.samplers.KSAMPLER(sampler), plan.video_segment
