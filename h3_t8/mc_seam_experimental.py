"""Independent opt-in MC operators; never sample or alter an existing runner.

Adapted from NikoDemon80/ComfyUI-H3-Motion-Context at 518c083d11d80fa5920222ad89889924ee4401c9.
Copyright (C) 2026 NikoDemon80. GNU GPL version 3, WITHOUT ANY WARRANTY.
T8 modifications 2026-10-21: metadata/AV preservation, explicit keep-pin variant,
actual-input validation, cancellation, reports, and unbounded cycle arithmetic.
See licenses/MC_MOTION_CONTEXT_NOTICE.md and MC_MOTION_CONTEXT_GPL3.txt.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import json
import math
import operator

import torch
import torch.nn.functional as F

SOURCE_COMMIT = "518c083d11d80fa5920222ad89889924ee4401c9"
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_PREFIX = (0, 1, 5, 9, 13)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _report(operator_name, status, **values):
    return {"schema": "t8.h21.mc_seam.v1", "operator": operator_name,
            "status": status, "source_commit": SOURCE_COMMIT,
            "own_diffusion_nfe": 0, "sampler_called": False,
            "trained_sigma_table_modified": False,
            "default_long_video_runner_modified": False,
            "quality_accepted": False, "GPU_validated": False,
            "portable_cache_identity": False, **values}


def _interrupt():
    from comfy.model_management import throw_exception_if_processing_interrupted
    throw_exception_if_processing_interrupted()


def _integer(value, name, *, minimum=None):
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer, not a truncated frame value") from error
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def pixel_frames(steps):
    steps = _integer(steps, "latent steps", minimum=0)
    cycles, remainder = divmod(steps, 5)
    return 17 * cycles + FRAME_PREFIX[remainder]


def steps_for_frames(frames):
    frames = _integer(frames, "pixel frames", minimum=0)
    cycles, remainder = divmod(frames, 17)
    if remainder not in FRAME_PREFIX:
        raise ValueError(f"{frames} pixel frames is not a native 1/4/4/4/4 latent boundary")
    return 5 * cycles + FRAME_PREFIX.index(remainder)


def refine_weights(total_steps, hold_frames, blend_frames):
    total = _integer(total_steps, "total steps", minimum=0)
    hold = _integer(hold_frames, "hold_frames", minimum=0)
    blend = _integer(blend_frames, "blend_frames", minimum=0)
    steps_for_frames(hold)
    return [0.0 if pixel_frames(k) < hold else 1.0 if not blend else
            min(1.0, max(0.0, (pixel_frames(k) - hold) / blend)) for k in range(total)]


def _tensor(value, name):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.device.type == "meta" or not value.is_floating_point()
            or value.is_quantized or not value.numel()):
        raise ValueError(f"{name} must be a nonempty materialized strided floating tensor")
    # Check leading source rows without cloning the full floating input; cancellation
    # is a real Core check, not a successful-completion receipt.
    for part in value.unbind(0) if value.ndim else (value,):
        _interrupt()
        if not bool(torch.isfinite(part).all()):
            raise ValueError(f"{name} contains nonfinite values")


def _streams(latent, name, *, check_audio=True):
    if not isinstance(latent, Mapping) or "samples" not in latent:
        raise ValueError(f"{name} must be a LATENT mapping containing samples")
    samples = latent["samples"]
    if isinstance(samples, torch.Tensor):
        raise ValueError(f"{name} requires an explicit video/audio AV pair, not an ordinary Tensor")
    if isinstance(samples, (tuple, list)):
        parts = tuple(samples)
    elif getattr(samples, "is_nested", False) and callable(getattr(samples, "unbind", None)):
        parts = tuple(samples.unbind())
    else:
        raise ValueError(f"{name} samples must expose an AV pair via a sequence or nested unbind()")
    if len(parts) != 2:
        raise ValueError(f"{name} must contain exactly two AV streams, not {len(parts)}")
    video, audio = parts
    _tensor(video, f"{name} video")
    if video.ndim not in (4, 5):
        raise ValueError(f"{name} video must be [B,C,T,H,W] or [C,T,H,W]")
    if any(not dimension for dimension in video.shape):
        raise ValueError(f"{name} video dimensions must be nonzero")
    if not isinstance(audio, torch.Tensor) or audio.ndim != 4 or audio.shape[-2] != 2:
        raise ValueError(f"{name} audio must have actual H3 shape [B,C,2,T]")
    if check_audio:
        _tensor(audio, f"{name} audio")
    batch = video.shape[0] if video.ndim == 5 else 1
    if audio.shape[0] != batch:
        raise ValueError(f"{name} video/audio batch sizes differ")
    return video.unsqueeze(0) if video.ndim == 4 else video, audio, video.ndim


def _repack(original, video, audio, raw_ndim):
    if raw_ndim == 4:
        video = video.squeeze(0)
    source = original["samples"]
    if type(source) is list:
        replacement = [video, audio]
    elif type(source) is tuple:
        replacement = (video, audio)
    elif isinstance(source, list):
        replacement = copy.copy(source)
        if replacement is source:
            raise ValueError("AV list copy returned the original object; a fresh container is required before replacing streams")
        replacement[:] = [video, audio]
        _selected_streams(replacement, video, audio)
    elif isinstance(source, tuple):
        replacement = type(source)((video, audio))
        if replacement is source:
            raise ValueError("AV tuple reconstruction returned the original object; a fresh container is required")
        # A fresh immutable pair cannot be edited after a shallow copy. Preserve
        # the user's own state separately, by reference rather than deepcopy.
        state = getattr(source, "__dict__", None)
        if state is not None:
            vars(replacement).update(state)
        for owner in type(source).__mro__:
            slots = owner.__dict__.get("__slots__", ())
            for name in (slots,) if isinstance(slots, str) else slots:
                if name in ("__dict__", "__weakref__"):
                    continue
                if name.startswith("__") and not name.endswith("__"):
                    name = "_" + owner.__name__.lstrip("_") + name
                if hasattr(source, name):
                    setattr(replacement, name, getattr(source, name))
        _selected_streams(replacement, video, audio)
    elif getattr(source, "is_nested", False) and hasattr(source, "tensors"):
        # Core NestedTensor and compatible subclasses keep their own extra
        # attributes. No live mutation of the original container or streams.
        replacement = copy.copy(source)
        if replacement is source:
            raise ValueError("Nested AV copy returned the original object; a fresh container is required before replacing streams")
        replacement.tensors = [video, audio]
        actual = tuple(replacement.unbind())
        if len(actual) != 2 or actual[0] is not video or actual[1] is not audio:
            raise ValueError("Nested AV container cannot reconstruct the selected actual streams")
    else:
        try:
            replacement = type(source)((video, audio))
        except (TypeError, ValueError) as error:
            raise ValueError("AV container has no usable two-stream reconstruction API") from error
        actual = tuple(replacement.unbind()) if getattr(replacement, "is_nested", False) else tuple(replacement)
        if len(actual) != 2 or actual[0] is not video or actual[1] is not audio:
            raise ValueError("AV reconstruction API did not preserve the selected streams")
    result = dict(original)
    result["samples"] = replacement
    return result


def _selected_streams(container, video, audio):
    actual = tuple(container)
    if len(actual) != 2 or actual[0] is not video or actual[1] is not audio:
        raise ValueError("AV reconstruction API did not preserve the selected streams")


def refine_blend(original, refined=None, *, enabled=False, hold_frames=22,
                 blend_frames=22, audio="original", clip_index=1):
    if not enabled:
        return original, _report("refine_blend", "disabled_identity", source_identity=True, audio_changed=False)
    if refined is None:
        raise ValueError("enabled Refine Blend needs the explicit upstream refined LATENT")
    if audio not in ("original", "refined"):
        raise ValueError("audio must explicitly select original or refined")
    video, sound, raw_ndim = _streams(original, "original")
    changed, refined_sound, _ = _streams(refined, "refined")
    if video.shape != changed.shape:
        raise ValueError("Refine Blend video shape must match the original clip")
    # Preserve the source formula's actual PyTorch promotion/copy behavior.
    # Audio is selected, never algebraically paired with the old audio. A new
    # sound's dtype/channel/time layout is not a reason to ban the user's choice.
    first = _integer(clip_index, "clip_index") <= 0
    hold = 0 if first else _integer(hold_frames, "hold_frames", minimum=0)
    hold_steps = steps_for_frames(hold)
    if hold_steps > video.shape[2]:
        raise ValueError("hold_frames needs more native latent steps than the actual clip contains")
    weights = [1.0] * video.shape[2] if first else refine_weights(video.shape[2], hold, blend_frames)
    result = video.clone()
    for k, weight in enumerate(weights):
        _interrupt()
        if weight >= 1:
            result[:, :, k] = changed[:, :, k]
        elif weight > 0:
            result[:, :, k] = (1.0 - weight) * video[:, :, k] + weight * changed[:, :, k]
    _tensor(result, "Refine Blend output")
    selected_audio = refined_sound if audio == "refined" else sound
    output = _repack(original, result, selected_audio, raw_ndim)
    return output, _report("refine_blend", "computed_unreviewed", source_identity=False,
                          audio_changed=selected_audio is not sound, first_clip=first,
                          hold_frames_used=hold, held_native_steps=hold_steps,
                          zero_weight_steps=sum(weight == 0 for weight in weights),
                          weights=weights, metadata_preserved=True,
                          original_video_dtype=str(video.dtype), refined_video_dtype=str(changed.dtype),
                          output_video_dtype=str(result.dtype), selected_audio_shape=list(selected_audio.shape),
                          selected_audio_dtype=str(selected_audio.dtype),
                          warning="Explicit refined audio compatibility is not certified" if audio == "refined" else "",
                          explicit_upstream_refine_cost_not_included_in_own_nfe=True)


def _blur(video):
    padded = F.pad(video, (1, 1, 1, 1, 0, 0), mode="replicate")
    height, width = video.shape[-2:]
    result = 0
    for i in range(3):
        for j in range(3):
            result = result + padded[..., i:i + height, j:j + width]
    return result / 9.0


def _join_finite(value, computation):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Join Match {computation} computation is nonfinite or overflows the actual dtype")
    return value


def _stat(value, kind, dim=None, *, keepdim=False):
    def compute(source):
        if kind in ("RMS", "spatial RMS"):
            square = source.square() if kind == "spatial RMS" else source * source
            return square.mean(dim, keepdim=keepdim).sqrt()
        if kind == "mean":
            return source.mean(dim, keepdim=keepdim)
        return source.std(dim, keepdim=keepdim)
    result = compute(value)
    if bool(torch.isfinite(result).all()):
        return result  # Preserve the fixed-source ordinary-input operation exactly.
    _join_finite(value, f"{kind} input")
    # Squaring/summing large finite values can overflow although the reduced
    # statistic is representable. Only those failed reductions use a scaled
    # equivalent; real nonfinite input/output is not hidden in the JSON report.
    scale = value.abs().amax(dim, keepdim=True)
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    reduced_scale = scale if keepdim else scale.squeeze() if dim is None else scale.squeeze(dim)
    fallback = compute(value / safe_scale) * reduced_scale
    _join_finite(fallback, kind)
    return torch.where(torch.isfinite(result), result, fallback)


def _rms(value):
    return float(_stat(value, "RMS"))


def _score(numerator, denominator, name):
    result = numerator / max(denominator, 1e-9)
    if not math.isfinite(result):
        raise ValueError(f"Join Match {name} computation is nonfinite or overflows the actual dtype")
    return result


def _motion(video, start):
    values = []
    for t in range(start, video.shape[2] - 5):
        _interrupt()
        x, y = video[0, :, t], video[0, :, t + 5]
        centered = x - _stat(x, "mean", (-2, -1), keepdim=True)
        values.append(_score(_rms(y - x), _rms(centered), "motion ratio"))
    if not values:
        return None
    result = sum(values) / len(values)
    if not math.isfinite(result):
        result = sum(value / len(values) for value in values)
    if not math.isfinite(result):
        raise ValueError("Join Match motion mean computation is nonfinite or overflows")
    return result


def _join_one(previous, current, ta, tb, hold_steps, strength, tuning,
              min_excess, match_spread, min_excess_spread, match_detail, min_excess_detail):
    before = current
    motion = _motion(current, hold_steps) if tuning == "auto" else None
    if tuning == "auto":
        still = motion is not None and motion <= .48
        min_excess, min_excess_spread, min_excess_detail = 1.5, 1.0 if still else 1.5, 1.0 if still else 1.5
    detail_done, detail_excess = False, None
    if match_detail:
        smooth_a, smooth_b = _blur(previous), _blur(current)
        fine_a, fine_b = previous - smooth_a, current - smooth_b
        def measure(value, t):
            return _stat(value[0, :, t], "spatial RMS", (-2, -1))
        ratio = measure(fine_a, ta) / measure(fine_b, tb).clamp(min=1e-6)
        normal = measure(fine_b, tb + 5) / measure(fine_b, tb).clamp(min=1e-6)
        detail_excess = _score(_rms(ratio - 1), _rms(normal - 1), "detail ratio")
        if detail_excess >= min_excess_detail:
            gain = (1.0 + strength * (ratio - 1.0)).clamp(min=.9, max=1.1)
            current = smooth_b + fine_b * gain.view(1, -1, 1, 1, 1)
            detail_done = True
    def means(value, t):
        return _stat(value[0, :, t], "mean", (-2, -1))
    step = _join_finite(means(current, tb) - means(previous, ta), "mean difference")
    normal = means(current, tb + 5) - means(current, tb)
    mean = means(current, tb)
    mean_excess = _score(_rms(step), _rms(normal), "mean ratio")
    fix_mean = mean_excess >= min_excess
    gain, spread_excess = None, None
    if match_spread:
        if current.shape[-1] * current.shape[-2] < 2:
            raise ValueError("Join Match unbiased spatial spread needs at least two actual spatial elements")
        def spreads(value, t):
            return _stat(value[0, :, t], "spread", (-2, -1))
        ratio = spreads(previous, ta) / spreads(current, tb).clamp(min=1e-6)
        normal_ratio = spreads(current, tb + 5) / spreads(current, tb).clamp(min=1e-6)
        spread_excess = _score(_rms(ratio - 1), _rms(normal_ratio - 1), "spread ratio")
        if spread_excess >= min_excess_spread:
            gain = (1.0 + strength * (ratio - 1.0)).clamp(min=.9, max=1.1)
    note = {"motion_source_threshold_not_T8_quality_calibration": motion,
            "mean_excess": mean_excess, "mean_applied": fix_mean,
            "spread_excess": spread_excess, "spread_applied": gain is not None,
            "detail_excess": detail_excess, "detail_applied": detail_done}
    if not fix_mean and gain is None and not detail_done:
        return before, note
    target = mean - strength * step if fix_mean else mean
    if gain is not None:
        current = (current - mean.view(1, -1, 1, 1, 1)) * gain.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
    return current - (mean - target).view(1, -1, 1, 1, 1), note


def join_match(samples, context_latent=None, *, enabled=False, strength=1.0,
               context_length=22, variant="keep_pin_T8_exp", tuning="auto",
               min_excess=1.5, match_spread=True, min_excess_spread=1.5,
               match_detail=True, min_excess_detail=1.5):
    if not enabled:
        return samples, _report("join_match", "disabled_identity", source_identity=True, audio_changed=False)
    amount = _finite(strength, "strength")
    hold = _integer(context_length, "context_length", minimum=0)
    if context_latent is None or amount == 0 or hold == 0:
        return samples, _report("join_match", "no_context_or_zero_identity", source_identity=True, audio_changed=False)
    if variant not in ("source_whole_clip_exp", "keep_pin_T8_exp"):
        raise ValueError("Join Match variant must explicitly select source_whole_clip_exp or keep_pin_T8_exp")
    if tuning not in ("auto", "manual"):
        raise ValueError("Join Match tuning must be auto or manual")
    thresholds = [_finite(v, n) for v, n in ((min_excess, "min_excess"),
                  (min_excess_spread, "min_excess_spread"), (min_excess_detail, "min_excess_detail"))]
    hold_steps = steps_for_frames(hold)
    video, sound, raw_ndim = _streams(samples, "current")
    previous, _, _ = _streams(context_latent, "context", check_audio=False)
    if (previous.shape[0] not in (1, video.shape[0]) or previous.shape[1] != video.shape[1]
            or previous.shape[-2:] != video.shape[-2:]):
        raise ValueError("Join Match requires matching channel/spatial sizes and either one shared context batch or matching batches")
    if hold_steps > video.shape[2]:
        raise ValueError("context_length is longer than the actual current video latent")
    broadcast = previous.shape[0] == 1 and video.shape[0] > 1
    previous = previous.to(video)
    if broadcast:
        previous = previous.expand(video.shape[0], -1, -1, -1, -1)
    ta = previous.shape[2] - 1
    tb = hold_steps + (ta - hold_steps) % 5
    if tb + 5 >= video.shape[2]:
        return samples, _report("join_match", "too_short_identity", source_identity=True,
                                audio_changed=False, previous_step=ta, measured_step=tb,
                                next_same_phase_step=tb + 5)
    result, notes = [], []
    for batch in range(video.shape[0]):
        _interrupt()
        value, note = _join_one(previous[batch:batch + 1], video[batch:batch + 1], ta, tb,
                                hold_steps, amount, tuning, thresholds[0], bool(match_spread),
                                thresholds[1], bool(match_detail), thresholds[2])
        result.append(value)
        notes.append(note)
    changed = torch.cat(result, dim=0)
    if variant == "keep_pin_T8_exp":
        changed[:, :, :hold_steps] = video[:, :, :hold_steps]
    _tensor(changed, "Join Match output")
    identity = torch.equal(changed, video)
    output = samples if identity else _repack(samples, changed, sound, raw_ndim)
    return output, _report("join_match", "unchanged_guard" if identity else "computed_unreviewed",
                          source_identity=identity, audio_changed=False, variant=variant,
                          source_original_changes_pinned_rows=True,
                          T8_keep_pin_variant=variant == "keep_pin_T8_exp",
                          held_native_steps=hold_steps, hold_frames=hold,
                          pin_exact=torch.equal(changed[:, :, :hold_steps], video[:, :, :hold_steps]),
                          previous_step=ta, measured_step=tb, next_same_phase_step=tb + 5,
                          batch_independent_extension=video.shape[0] != 1,
                          context_batch_broadcast=broadcast,
                          per_batch=notes, metadata_preserved=True,
                          RGB_flash_cut_texture_or_PCM_quality_not_inferred_from_latent_stats=True)


def sigma_tail(sigmas, *, enabled=False, tail_steps=3, strength=.5):
    if not enabled:
        return sigmas, _report("sigma_tail", "disabled_identity", source_identity=True)
    amount = _finite(strength, "strength")
    count = _integer(tail_steps, "tail_steps", minimum=0)
    if amount == 0 or count == 0:
        return sigmas, _report("sigma_tail", "zero_identity", source_identity=True)
    if amount == -1:
        raise ValueError("Sigma Tail strength=-1 gives a zero denominator in 1/(1+strength)")
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1:
        raise ValueError("Sigma Tail requires the actual one-dimensional SIGMAS tensor")
    _tensor(sigmas, "sigmas")
    values = sigmas.detach().cpu().tolist()
    if any(value < 0 for value in values):
        raise ValueError("Sigma Tail requires actual nonnegative noise levels for a real-valued fractional power")
    nonincreasing = all(a >= b for a, b in zip(values, values[1:]))
    end = len(values) - 1 if values[-1] == 0 else len(values)
    start = max(1, end - count)
    if start >= end or values[start - 1] <= 0:
        return sigmas, _report("sigma_tail", "too_short_identity", source_identity=True)
    anchor, power, output = values[start - 1], 1.0 / (1.0 + amount), list(values)
    adjusted = 0
    for index in range(start, end):
        _interrupt()
        try:
            output[index] = anchor * (values[index] / anchor) ** power
        except (ZeroDivisionError, OverflowError) as error:
            raise ValueError(f"Sigma Tail actual power at index {index} is undefined or overflows: {error}") from error
        if not math.isfinite(output[index]):
            raise ValueError(f"Sigma Tail actual power at index {index} is nonfinite")
        if output[index] >= output[index - 1]:
            output[index] = output[index - 1] * .999
            adjusted += 1
    result = sigmas.new_tensor(output)
    _tensor(result, "Sigma Tail output")
    return result, _report("sigma_tail", "computed_unreviewed", source_identity=False,
                          changed_indices=[i for i in range(len(values)) if result[i] != sigmas[i]],
                          source_levels=values, selected_levels=result.detach().cpu().tolist(),
                          first_level_exact=bool(result[0] == sigmas[0]),
                          terminal_level_exact=bool(result[-1] == sigmas[-1]),
                          input_nonincreasing=nonincreasing, positive_power=power > 0,
                          source_order_adjustment_count=adjusted,
                          strict_order_in_actual_dtype=bool(torch.all(result[:-1] > result[1:])),
                          explicit_modified_schedule_not_trained_table=True,
                          downstream_joint_sampler_may_change_audio=True,
                          warning="Explicit EXP schedule; original sampler/table inputs are not rewritten or certified")
