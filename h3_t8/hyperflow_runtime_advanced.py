"""Owned, reversible two-time HyperFlow patch for native ComfyUI MiniMax-H3.

The interval embedding follows Video-Rebirth/hyperflow (Apache-2.0), revision
1dd2f342aba5ab51da02b62885939655e8e268da. This is a separate execution
route: normal H3 samplers cannot silently drive the interval-conditioned model.
"""

from __future__ import annotations

import contextvars
import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import comfy.lora
import comfy.model_management
import comfy.patcher_extension
import torch
import torch.nn.functional as F

from .hyperflow_weights_advanced import HyperFlowWeights
from .logical_object import logical_object


LOG = logging.getLogger(__name__)
ATTACHMENT_KEY = "t8_h3_hyperflow_v1"
_ACTIVE_STEP: contextvars.ContextVar["HyperFlowStep | None"] = contextvars.ContextVar(
    "t8_h3_hyperflow_step", default=None
)
_ACTIVE_FORWARD: contextvars.ContextVar["_ForwardContext | None"] = (
    contextvars.ContextVar("t8_h3_hyperflow_forward", default=None)
)


@dataclass(frozen=True)
class HyperFlowBinding:
    owner: str
    sha256: str
    version: str
    gate: float
    video_shift: float
    audio_shift: float
    raw_sigmas: tuple[float, ...]
    model_identity: int


@dataclass(frozen=True)
class HyperFlowStep:
    owner: str
    absolute_index: int
    sigma_video: float
    sigma_video_next: float


@dataclass
class _ForwardContext:
    step: HyperFlowStep
    t_video: float
    r_video: float
    t_audio: float
    r_audio: float
    pin_video: float
    pin_audio: float
    video_masks: dict[float, float] = field(default_factory=dict)
    audio_masks: dict[float, float] = field(default_factory=dict)
    role_maps: dict[str, list[int]] = field(default_factory=dict)
    tensor_maps: dict[tuple[str, torch.device], torch.Tensor] = field(
        default_factory=dict
    )
    remapped_segments: list | None = None
    original_segments: list | None = None
    transformer_options: dict | None = None
    time_values: torch.Tensor | None = None


def active_step() -> HyperFlowStep | None:
    return _ACTIVE_STEP.get()


def active_time_values() -> torch.Tensor | None:
    """Actual expanded current-time rows, only within this two-time forward."""
    context = _ACTIVE_FORWARD.get()
    return None if context is None else context.time_values


def push_step(step: HyperFlowStep):
    return _ACTIVE_STEP.set(step)


def pop_step(token) -> None:
    _ACTIVE_STEP.reset(token)


def _f32(value: float) -> float:
    return float(torch.tensor(value, dtype=torch.float32))


def _scale_shift(
    sigma: torch.Tensor, from_shift: float, to_shift: float
) -> torch.Tensor:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def _unpatch_own(forward):
    seen: set[int] = set()
    while getattr(forward, "_t8_hyperflow_owner", None) is not None:
        if id(forward) in seen:
            raise RuntimeError("Cyclic HyperFlow forward wrapper on shared Comfy model")
        seen.add(id(forward))
        forward = forward._t8_hyperflow_inner
    return forward


def _mark_patch(forward, inner, owner: str):
    forward._t8_hyperflow_owner = owner
    forward._t8_hyperflow_inner = inner
    return forward


def _weights_snapshot(model) -> tuple[torch.Tensor, ...]:
    values = []
    for projection in ("proj_in", "proj_out"):
        for attribute in ("weight", "bias"):
            tensor = logical_object(
                model, f"diffusion_model.time_embedder.{projection}.{attribute}"
            )
            if not isinstance(tensor, torch.Tensor) or tensor.is_meta:
                raise ValueError(
                    "HyperFlow requires materialized full-structure FP32 time projections"
                )
            values.append(tensor.detach().to(device="cpu", dtype=torch.float32).clone())
    return tuple(values)


def _preflight_structure(
    model, weights: HyperFlowWeights
) -> tuple[Any, dict[str, object]]:
    diffusion = logical_object(model, "diffusion_model")
    if getattr(diffusion, "use_adaln_curves", False) or not hasattr(
        diffusion, "time_embedder"
    ):
        raise ValueError(
            "Full HyperFlow requires the base time_embedder; pruned/curve H3 is not two-time compatible"
        )
    blocks = list(logical_object(model, "diffusion_model.blocks"))
    refiners = list(logical_object(model, "diffusion_model.token_refiner.blocks"))
    if len(blocks) != 50 or len(refiners) != 2:
        LOG.warning(
            "HyperFlow v1 was trained with 50 DiT and 2 text-refiner blocks; selected %d/%d. Continuing with the actual adapter target/shape checks; quality is unverified.",
            len(blocks),
            len(refiners),
        )
    time_module = logical_object(model, "diffusion_model.time_embedder")
    if not callable(
        logical_object(model, "diffusion_model.final_layer.forward")
    ) or not hasattr(time_module, "freq_dim"):
        raise ValueError(
            "HyperFlow needs a callable final_layer.forward and time_embedder.freq_dim"
        )
    modules = dict(diffusion.named_modules())
    for target, (a, b, _) in weights.patches.items():
        try:
            module = logical_object(model, "diffusion_model." + target)
        except (AttributeError, KeyError, IndexError) as error:
            raise ValueError(
                f"HyperFlow target missing from selected base: {target}"
            ) from error
        modules[target] = module
        weight = getattr(module, "weight", None)
        if module is None or weight is None:
            raise ValueError(f"HyperFlow target missing from selected base: {target}")
        if len(weight.shape) != 2 or tuple(weight.shape) != (b.shape[0], a.shape[1]):
            raise ValueError(
                f"HyperFlow target shape mismatch at {target}: model={tuple(weight.shape)}, "
                f"adapter={(b.shape[0], a.shape[1])}"
            )
    for part, (a, b, _) in weights.endpoint.items():
        weight = logical_object(model, f"diffusion_model.time_embedder.{part}.weight")
        if tuple(weight.shape) != (b.shape[0], a.shape[1]):
            raise ValueError(f"HyperFlow endpoint shape mismatch at {part}")
    return diffusion, modules


def _endpoint_forward(time_module, snapshot, weights: HyperFlowWeights):
    wi, bi, wo, bo = snapshot
    ai, ui, alpha_i = weights.endpoint["proj_in"]
    ao, uo, alpha_o = weights.endpoint["proj_out"]
    freq_dim = int(time_module.freq_dim)
    cache: dict[torch.device, tuple[torch.Tensor, ...]] = {}

    def forward(r: torch.Tensor) -> torch.Tensor:
        if r.device not in cache:
            cache[r.device] = tuple(
                value.to(device=r.device, dtype=torch.float32)
                for value in (wi, bi, wo, bo, ai, ui, ao, uo)
            )
        w_in, b_in, w_out, b_out, a_in, b_up_in, a_out, b_up_out = cache[r.device]
        half = freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=r.device, dtype=torch.float32)
            / half
        )
        args = r.to(torch.float32)[:, None] * freqs[None]
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        z = F.linear(embedding, w_in, b_in)
        z = z + F.linear(F.linear(embedding, a_in), b_up_in) * (alpha_i / a_in.shape[0])
        z = F.silu(z)
        result = F.linear(z, w_out, b_out)
        return result + F.linear(F.linear(z, a_out), b_up_out) * (
            alpha_o / a_out.shape[0]
        )

    return forward


def _mask_time_endpoints(
    mask: torch.Tensor | None, sigma: torch.Tensor, sigma_next: torch.Tensor, pin: float
) -> dict[float, float]:
    if mask is None:
        return {}
    if not bool(torch.isfinite(mask).all()) or not bool(
        ((mask >= 0) & (mask <= 1)).all()
    ):
        raise ValueError("HyperFlow denoise mask must be finite in [0, 1]")
    levels = torch.unique(mask.to(torch.float32)).to(device=sigma.device)
    current = (1.0 - levels * sigma).clamp(max=pin)
    endpoint = 1.0 - levels * sigma_next
    result = {}
    for m, t, r in zip(levels.tolist(), current.tolist(), endpoint.tolist()):
        t = _f32(t)
        r = t if t >= _f32(pin) else _f32(r)
        previous = result.get(t)
        if previous is not None and previous != r:
            raise ValueError(
                "HyperFlow soft mask has two endpoint times for one native Core timestep; "
                "this geometry needs a distinct row-time representation"
            )
        result[t] = r
    return result


def _forward_context(
    owner: str, diffusion, args: tuple, kwargs: dict
) -> _ForwardContext:
    step = _ACTIVE_STEP.get()
    if step is None or step.owner != owner:
        raise RuntimeError(
            "HyperFlow MODEL requires its own typed sampler/plan; no interval step is active"
        )
    timestep = args[1] if len(args) > 1 else kwargs["timestep"]
    options = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
    if options is None:
        options = {}
    payload = kwargs.get("minimax_payload") or {}
    shift_v = float(
        options.get("minimax_h3_sigma_shift_video", diffusion.sigma_shift_video)
    )
    shift_a = float(
        options.get("minimax_h3_sigma_shift_audio", diffusion.sigma_shift_audio)
    )
    if (shift_v, shift_a) != (12.0, 3.0):
        raise ValueError("HyperFlow v1 sampler requires video/audio shifts 12/3")
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    if abs(float(sigma_v) - step.sigma_video) > 2e-6:
        raise ValueError(
            "HyperFlow model timestep disagrees with the typed interval plan"
        )
    sigma_next = torch.tensor(
        step.sigma_video_next, device=sigma_v.device, dtype=torch.float32
    )
    sigma_a = _scale_shift(sigma_v, shift_v, shift_a)
    sigma_a_next = _scale_shift(sigma_next, shift_v, shift_a)
    t_v = _f32(float(1.0 - sigma_v))
    t_a = _f32(float(1.0 - sigma_a))
    r_v = _f32(float(1.0 - sigma_next))
    r_a = _f32(float(1.0 - sigma_a_next))
    vis_aug = float(payload.get("visual_cond_noise_aug", 0.999))
    aud_aug = float(payload.get("audio_cond_noise_aug", 1.0))
    pin_v = _f32(max(t_v, vis_aug))
    pin_a = _f32(max(t_a, aud_aug))
    video_mask = kwargs.get("denoise_mask")
    audio_mask = kwargs.get("audio_denoise_mask")
    video_levels = None
    if video_mask is not None:
        from comfy.ldm.minimax.model import mask_row_values

        video_x = args[0][0] if args else kwargs["x"][0]
        latent_t, h, w = video_x.shape[2:]
        video_levels = mask_row_values(
            video_mask[0, 0].to(torch.float32),
            latent_t,
            (h + 1) // 2 * 2,
            (w + 1) // 2 * 2,
        )
    audio_levels = (
        audio_mask[0, 0].to(torch.float32).reshape(-1)
        if audio_mask is not None
        else None
    )
    return _ForwardContext(
        step=step,
        t_video=t_v,
        r_video=r_v,
        t_audio=t_a,
        r_audio=r_a,
        pin_video=pin_v,
        pin_audio=pin_a,
        video_masks=_mask_time_endpoints(video_levels, sigma_v, sigma_next, pin_v),
        audio_masks=_mask_time_endpoints(audio_levels, sigma_a, sigma_a_next, pin_a),
        transformer_options=options,
    )


def _install_two_time(
    patched, diffusion, weights: HyperFlowWeights, snapshot, owner: str
) -> None:
    te = logical_object(patched, "diffusion_model.time_embedder")
    source_forward = _unpatch_own(
        logical_object(patched, "diffusion_model.time_embedder.forward")
    )
    endpoint_forward = _endpoint_forward(te, snapshot, weights)

    def time_forward(t_values: torch.Tensor):
        ctx = _ACTIVE_FORWARD.get()
        if ctx is None or ctx.step.owner != owner:
            raise RuntimeError(
                "HyperFlow time embedder invoked outside its interval-conditioned forward"
            )
        times = [_f32(float(t)) for t in t_values.tolist()]
        pairs: list[tuple[float, float]] = [(t, t) for t in times]
        positions = {pair: index for index, pair in enumerate(pairs)}

        def add_pair(t, r):
            pair = (_f32(t), _f32(r))
            if pair not in positions:
                positions[pair] = len(pairs)
                pairs.append(pair)
            return positions[pair]

        maps = {"pin": list(range(len(times))), "video": [], "audio": []}
        for t in times:
            rv = ctx.video_masks.get(t)
            if rv is None and t == ctx.t_video:
                rv = ctx.r_video
            ra = ctx.audio_masks.get(t)
            if ra is None and t == ctx.t_audio:
                ra = ctx.r_audio
            maps["video"].append(add_pair(t, t if rv is None else rv))
            maps["audio"].append(add_pair(t, t if ra is None else ra))
        required = {ctx.t_video, ctx.t_audio}
        if not required.issubset(set(times)):
            raise RuntimeError("HyperFlow Core row plan omits a generated AV timestep")
        ctx.role_maps = maps
        t = t_values.new_tensor([pair[0] for pair in pairs])
        r = t_values.new_tensor([pair[1] for pair in pairs])
        ctx.time_values = t
        base = source_forward(t)
        endpoint = endpoint_forward(r).to(dtype=base.dtype)
        return base + weights.metadata.gate * (endpoint - base)

    patched.add_object_patch(
        "diffusion_model.time_embedder.forward",
        _mark_patch(time_forward, source_forward, owner),
    )

    def remap(ctx: _ForwardContext, row, role: str):
        indices = ctx.role_maps[role]
        if isinstance(row, int):
            return indices[row]
        key = (role, row.device)
        if key not in ctx.tensor_maps:
            ctx.tensor_maps[key] = row.new_tensor(indices)
        return ctx.tensor_maps[key][row]

    for index, block in enumerate(logical_object(patched, "diffusion_model.blocks")):
        original = _unpatch_own(
            logical_object(patched, f"diffusion_model.blocks.{index}.forward")
        )

        def block_forward(
            x, t_emb, mod_segments, rope_freqs, *args, _inner=original, **kwargs
        ):
            ctx = _ACTIVE_FORWARD.get()
            if ctx is None or ctx.step.owner != owner:
                raise RuntimeError(
                    "HyperFlow DiT block invoked without interval context"
                )
            if ctx.original_segments is not mod_segments:
                layout = ctx.transformer_options.get("minimax_h3_layout")
                if layout is None:
                    raise RuntimeError("HyperFlow needs the native H3 packed AV layout")
                source_segments = iter(layout.segments)
                _, stop, kind = next(source_segments)
                result = []
                for a, b, row in mod_segments:
                    while a >= stop:
                        _, stop, kind = next(source_segments)
                    role = (
                        "video"
                        if kind in ("text", "video")
                        else "audio"
                        if kind == "audio"
                        else "pin"
                    )
                    result.append((a, b, remap(ctx, row // 3, role) * 3 + row % 3))
                ctx.original_segments = mod_segments
                ctx.remapped_segments = result
            return _inner(x, t_emb, ctx.remapped_segments, rope_freqs, *args, **kwargs)

        patched.add_object_patch(
            f"diffusion_model.blocks.{index}.forward",
            _mark_patch(block_forward, original, owner),
        )

    final_original = _unpatch_own(
        logical_object(patched, "diffusion_model.final_layer.forward")
    )

    def final_forward(x, t_emb, video_seg, audio_seg, *args, **kwargs):
        ctx = _ACTIVE_FORWARD.get()
        if ctx is None or ctx.step.owner != owner:
            raise RuntimeError("HyperFlow output head invoked without interval context")
        video_seg = (*video_seg[:2], remap(ctx, video_seg[2], "video"))
        audio_seg = (*audio_seg[:2], remap(ctx, audio_seg[2], "audio"))
        return final_original(x, t_emb, video_seg, audio_seg, *args, **kwargs)

    patched.add_object_patch(
        "diffusion_model.final_layer.forward",
        _mark_patch(final_forward, final_original, owner),
    )

    def scope(executor, *args, **kwargs):
        ctx = _forward_context(owner, diffusion, args, kwargs)
        token = _ACTIVE_FORWARD.set(ctx)
        try:
            return executor(*args, **kwargs)
        finally:
            _ACTIVE_FORWARD.reset(token)

    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "t8_hyperflow_" + owner,
        scope,
    )


def install_hyperflow(
    model, weights: HyperFlowWeights
) -> tuple[Any, HyperFlowBinding, dict]:
    if model.get_attachment(ATTACHMENT_KEY) is not None:
        raise ValueError(
            "HyperFlow is already applied to this MODEL; branch from the undecorated base"
        )
    prior_lora_metadata = model.get_attachment("t8_h3_lora_metadata")
    if (
        isinstance(prior_lora_metadata, dict)
        and str(prior_lora_metadata.get("hyperflow", "")).lower() == "true"
    ):
        raise ValueError(
            "Original HyperFlow was loaded as a content LoRA; remove it and use only the dedicated HyperFlow Loader"
        )
    content_time_targets = sorted(
        key for key in model.patches if key.startswith("diffusion_model.time_embedder.")
    )
    diffusion, _ = _preflight_structure(model, weights)
    snapshot = _weights_snapshot(model)
    state = {}
    key_map = {}
    for target, (a, b, alpha) in weights.patches.items():
        state[f"{target}.lora_A.weight"] = a
        state[f"{target}.lora_B.weight"] = b
        state[f"{target}.alpha"] = torch.tensor(alpha, dtype=torch.float32)
        key_map[target] = f"diffusion_model.{target}.weight"
    parsed = comfy.lora.load_lora(state, key_map, log_missing=False)
    if set(parsed) != set(key_map.values()):
        raise ValueError(
            "HyperFlow conversion did not map every backbone/time LoRA target"
        )
    patched = model.clone()
    applied = set(patched.add_patches(parsed))
    if applied != set(key_map.values()):
        raise ValueError(
            "HyperFlow base MODEL did not accept every required LoRA target"
        )
    owner = uuid.uuid4().hex
    binding = HyperFlowBinding(
        owner=owner,
        sha256=weights.source_sha256,
        version=weights.metadata.version,
        gate=weights.metadata.gate,
        video_shift=weights.metadata.video_shift,
        audio_shift=weights.metadata.audio_shift,
        raw_sigmas=weights.metadata.raw_sigmas,
        model_identity=id(model.model),
    )
    _install_two_time(patched, diffusion, weights, snapshot, owner)
    patched.set_attachments(ATTACHMENT_KEY, binding)
    report = {
        "schema": "t8.minimax_h3.hyperflow.loader.v1",
        "status": "installed_for_typed_sampler",
        "source_sha256": weights.source_sha256,
        "source_tensor_count": weights.source_tensor_count,
        "source_path": str(weights.path),
        "converted_targets": len(weights.patches) + len(weights.endpoint),
        "applied_backbone_and_base_time_targets": len(applied),
        "endpoint_targets": len(weights.endpoint),
        "fused_qkv_groups": weights.fused_qkv_groups,
        "gate": weights.metadata.gate,
        "rank": weights.metadata.rank,
        "license": weights.metadata.license,
        "runtime_note": "A typed HyperFlow sampler/plan is mandatory; no pruned or single-time fallback.",
        "preexisting_content_time_patch_count": len(content_time_targets),
        "content_time_scope_warning": (
            "Existing content LoRA patches affect the base time branch only; the HyperFlow endpoint branch remains original. This combination is not quality-qualified."
            if content_time_targets
            else None
        ),
    }
    if content_time_targets:
        LOG.warning(
            "HyperFlow MODEL has %d preexisting base-time LoRA target(s); endpoint branch is unchanged",
            len(content_time_targets),
        )
    return patched, binding, report
