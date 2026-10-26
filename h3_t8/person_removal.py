"""Independent T8 person-removal window contracts, built on native H3 nodes.

No third-party Relay installation, global layout patch, sampler or weight loader.
Green RGB is a visual guide, NEVER an inpaint/denoise mask. Generated PCM is
continuation context; delivery of the original soundtrack remains explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import types

import torch
import torch.nn.functional as F

from . import source_conform as source

FPS = 24
WINDOW = 22
HISTORY = 18
MARKER = "t8_person_history_v1"
GUIDE_KIND = "t8_person_history_stream"
_OWNER = object()


def _rgb(value, name):
    if (not isinstance(value, torch.Tensor) or value.ndim != 4
            or value.shape[-1] != 3 or not len(value)):
        raise ValueError(f"{name} requires non-empty RGB [N,H,W,3]")
    source.identity(value)  # bounded, finite, actual content, interruptible
    if value.min() < 0 or value.max() > 1:
        raise ValueError(f"{name} RGB must be in [0,1]")


def _grow_square(value, radius):
    """Bitwise-qualified CPU square dilation; other devices retain their path."""
    kernel = 2 * radius + 1
    if value.device.type == "cpu":
        horizontal = F.max_pool2d(value, (1, kernel), stride=1, padding=(0, radius))
        return F.max_pool2d(horizontal, (kernel, 1), stride=1, padding=(radius, 0))
    return F.max_pool2d(value, kernel, stride=1, padding=radius)


def prepare_mask(frames, mask, grow=5, feather=0, frame_map=None, lineage=None):
    """Manual/external masks remain usable without a fabricated origin claim."""
    _rgb(frames, "Conformed frames")
    if (not isinstance(grow, int) or isinstance(grow, bool) or not 0 <= grow <= 64
            or not isinstance(feather, int) or isinstance(feather, bool)
            or not 0 <= feather <= 32):
        raise ValueError("grow/feather require bounded integer pixel radii")
    mapping = None
    if frame_map is not None:
        source.validate_map(frame_map)
        if source.identity(frames) != frame_map.output_sha:
            raise ValueError("Connected RGB differs from Source Conform output")
        mask, mapping = source.map_control(frame_map, mask, "mask", lineage=lineage)
    elif lineage is not None:
        raise ValueError("Lineage requires its actual Source Conform map")
    if (not isinstance(mask, torch.Tensor) or tuple(mask.shape) != tuple(frames.shape[:3])
            or not mask.is_floating_point() or not torch.isfinite(mask).all()
            or mask.min() < 0 or mask.max() > 1):
        raise ValueError("MASK needs exactly one finite [0,1] plane per RGB frame; no broadcast")
    original_sha = source.identity(mask)
    # Process each frame independently. Empty/occluded frames are not fabricated
    # by temporal smoothing, propagation, or interpolation.
    processed = []
    for item in mask:
        value = item[None, None].float()
        if grow:
            value = _grow_square(value, grow)
        if feather:
            value = F.avg_pool2d(value, 2 * feather + 1, stride=1,
                                 padding=feather, count_include_pad=False)
        processed.append(value[0, 0].to(mask))
    result = torch.stack(processed)
    green = frames.clone()
    green[result.to(device=frames.device) > .5] = frames.new_tensor([0., 1., 0.])
    area = result.double().mean(dim=(1, 2)).cpu().tolist()
    report = dict(schema="t8.person-mask.v1", input_mask_sha=original_sha,
                  processed_mask_sha=source.identity(result), green_sha=source.identity(green),
                  frames_sha=source.identity(frames), frame_count=len(frames),
                  grow_radius_pixels=grow, feather_radius_pixels=feather,
                  area_fraction_by_frame=area,
                  empty_frame_indices=[i for i, item in enumerate(area) if item == 0.],
                  green_threshold=.5, temporal_smoothing=False,
                  frame_map_report=mapping, external_origin_unverified=mapping is None,
                  annotation_semantics_verified=False, sampler_mask_attached=False,
                  guide_type="RGB_green_reference_not_denoise_mask",
                  sampler_executed=False, quality_verified=False)
    return result, green, report


def window_starts(count):
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4096:
        raise ValueError("Person removal requires 1..4096 delivered source frames")
    starts = [0]
    while starts[-1] + WINDOW < count:
        starts.append(max(WINDOW - 1, min(starts[-1] + WINDOW - 1, count - WINDOW)))
    return tuple(starts)


@dataclass(frozen=True)
class RemovalPlan:
    frames: torch.Tensor
    green: torch.Tensor
    clean: torch.Tensor
    hashes: tuple[str, str, str]
    starts: tuple[int, ...]
    _owner: object


def make_plan(frames, green, clean, source_fps=24.):
    if not math.isfinite(float(source_fps)) or float(source_fps) != FPS:
        raise ValueError("Conform source and MASK together to 24fps before person removal")
    for value, name in ((frames, "Source"), (green, "Green guide"), (clean, "Clean first frame")):
        _rgb(value, name)
    if tuple(frames.shape) != tuple(green.shape) or tuple(clean.shape) != (1, *frames.shape[1:]):
        raise ValueError("Source/green need the same grid and a single same-geometry clean first frame")
    if frames.shape[1] % 32 or frames.shape[2] % 32:
        raise ValueError("Person removal canvas must be 32-aligned; do not resize MASK independently")
    plan = RemovalPlan(frames, green, clean,
                       tuple(source.identity(v) for v in (frames, green, clean)),
                       window_starts(len(frames)), _OWNER)
    report = dict(schema="t8.person-plan.v1", source_frames=len(frames), fps=FPS,
                  width=int(frames.shape[2]), height=int(frames.shape[1]),
                  starts=plan.starts, window_frames=WINDOW, history_frames=HISTORY,
                  per_window_steps=12, windows=len(plan.starts),
                  planned_total_sampler_steps=12 * len(plan.starts),
                  generated_frames=WINDOW * len(plan.starts),
                  delivered_frames=len(frames), pixel_locked=False,
                  original_voice_removed=False, sampler_executed=False,
                  quality_verified=False)
    return plan, report


def validate_plan(plan, index):
    if not isinstance(plan, RemovalPlan) or plan._owner is not _OWNER:
        raise ValueError("Connect an actual Person Removal Plan, not JSON")
    if (isinstance(index, bool) or not isinstance(index, int)
            or not 0 <= index < len(plan.starts)):
        raise ValueError("Window index is outside the actual plan")
    if plan.starts != window_starts(len(plan.frames)):
        raise ValueError("Person window schedule changed")
    if tuple(source.identity(v) for v in (plan.frames, plan.green, plan.clean)) != plan.hashes:
        raise ValueError("Person source, green guide or clean frame changed after planning")
    return plan


def _sample(frame, rate):
    # Integer sample boundaries on the full timeline, not rounded window lengths.
    from fractions import Fraction
    return round(Fraction(frame * rate, FPS))


def _pcm(audio, frames=None, sample_tolerance=0):
    if not isinstance(audio, dict) or not isinstance(audio.get("sample_rate"), int):
        raise ValueError("Continuation requires generated AUDIO with integer sample_rate")
    wave, rate = audio.get("waveform"), audio["sample_rate"]
    if (rate <= 0 or not isinstance(wave, torch.Tensor) or wave.ndim != 3
            or tuple(wave.shape[:2]) != (1, 2) or not wave.is_floating_point()
            or not torch.isfinite(wave).all()):
        raise ValueError("Continuation PCM requires finite [1,2,samples] stereo")
    if frames is not None and abs(wave.shape[-1] - _sample(frames, rate)) > sample_tolerance:
        raise ValueError("Predecessor PCM does not cover its exact delivered timeline")
    return wave, rate


def select_window(plan, index, previous_frames=None, previous_audio=None):
    validate_plan(plan, index)
    start = plan.starts[index]
    green = plan.green[start:start + WINDOW]
    if len(green) < WINDOW:
        green = torch.cat((green, green[-1:].expand(WINDOW - len(green), -1, -1, -1)))
    history, audio = None, None
    anchor = plan.clean
    if index:
        expected = min(len(plan.frames), plan.starts[index - 1] + WINDOW)
        _rgb(previous_frames, "Delivered predecessor")
        if tuple(previous_frames.shape) != (expected, *plan.frames.shape[1:]):
            raise ValueError("Predecessor frames are not this plan's preceding delivered range")
        if start < HISTORY - 1:
            raise ValueError("Window has insufficient native 18-frame history")
        wave, rate = _pcm(previous_audio, expected)
        history = previous_frames[start - HISTORY + 1:start + 1]
        anchor = history[-1:]
        lo, hi = _sample(start - HISTORY + 1, rate), _sample(start + 1, rate)
        audio = dict(waveform=wave[..., lo:hi], sample_rate=rate)
    elif previous_frames is not None or previous_audio is not None:
        raise ValueError("First window must not silently use predecessor context")
    report = dict(schema="t8.person-window.v1", window_index=index, global_start_frame=start,
                  global_end_frame=min(start + WINDOW, len(plan.frames)), fps=FPS,
                  generated_frames=WINDOW, history_frames=0 if history is None else len(history),
                  history_pcm_global_frames=None if history is None else [start - 17, start + 1],
                  sampler_executed=False)
    return anchor, green, history, audio, report


def append_window(plan, index, frames, audio, previous_frames=None, previous_audio=None):
    validate_plan(plan, index)
    _rgb(frames, "Decoded window")
    if tuple(frames.shape) != (WINDOW, *plan.frames.shape[1:]):
        raise ValueError("Decode the complete native 22-frame window before append")
    wave, rate = _pcm(audio)
    start, end = plan.starts[index], min(plan.starts[index] + WINDOW, len(plan.frames))
    delivered = 0 if not index else min(len(plan.frames), plan.starts[index - 1] + WINDOW)
    if index:
        _rgb(previous_frames, "Append predecessor")
        if previous_frames is None or tuple(previous_frames.shape) != (delivered, *frames.shape[1:]):
            raise ValueError("Append needs the exact preceding delivered frame range")
        previous_wave, previous_rate = _pcm(previous_audio, delivered)
        if previous_rate != rate:
            raise ValueError("Generated audio sample rate changed between windows")
    elif previous_frames is not None or previous_audio is not None:
        raise ValueError("First append cannot contain predecessor frames/audio")
    offset = _sample(start, rate)
    lo, hi = _sample(delivered, rate) - offset, _sample(end, rate) - offset
    if not 0 <= lo < hi <= wave.shape[-1]:
        raise ValueError("Decoded window PCM is shorter than its global delivery range")
    fresh = frames[delivered - start:end - start]
    output_frames = fresh if not index else torch.cat((previous_frames, fresh))
    output_wave = wave[..., lo:hi] if not index else torch.cat((previous_wave, wave[..., lo:hi]), dim=-1)
    if len(output_frames) != end or output_wave.shape[-1] != _sample(end, rate):
        raise RuntimeError("Person append produced a gap/duplicate in the full AV timeline")
    report = dict(schema="t8.person-append.v1", index=index, global_start_frame=start,
                  discarded_overlap_frames=delivered - start, delivered_frames=end,
                  delivered_pcm_samples=output_wave.shape[-1], sample_rate=rate,
                  complete=end == len(plan.frames), native_sampler_attested=False,
                  generated_pcm_purpose="continuation_context_not_automatic_final_audio",
                  quality_verified=False)
    return output_frames, dict(waveform=output_wave, sample_rate=rate), report


def relocate_history_payload(out, kwargs):
    """Shift only this cloned MODEL's explicitly marked conditioning layout.

    Core remains the constructor/packer. T8 keeps all spatial, reference,
    modality and update rows; only this new task's temporal origins move.
    """
    if kwargs.get(MARKER) != 1:
        return out
    payload_cond = out.get("minimax_payload")
    payload = getattr(payload_cond, "cond", None)
    if not isinstance(payload, dict) or payload.get("layout") is None:
        raise RuntimeError("Person history requires an actual native H3 payload/layout")
    keyframes = kwargs.get("minimax_keyframes", [])
    from comfy.ldm.minimax.model import _video_t_spans
    span = sum(sum(_video_t_spans(row["latent"].shape[2])) for row in keyframes
               if row.get(GUIDE_KIND) == "video")
    if span <= 0:
        raise ValueError("Person history has no video-history span")
    layout = copy.copy(payload["layout"])
    layout.position_ids = layout.position_ids.clone()
    segments = layout.segments
    if [row[2] for row in segments[-2:]] != ["audio", "video"]:
        raise RuntimeError("Native H3 target AV segment topology changed")
    target = float(layout.position_ids[segments[-1][0], 0])
    if not math.isclose(float(layout.position_ids[segments[-2][0], 0]), target, abs_tol=1e-9):
        raise RuntimeError("Native H3 audio/video timeline origins differ")
    guide_segments = [row for row in segments if row[2] in ("cond", "cond_audio")]
    expected = []
    for guide in keyframes:
        if guide.get("latent") is not None:
            expected.append(("cond", guide.get(GUIDE_KIND) == "video"))
        if guide.get("audio_latent") is not None:
            expected.append(("cond_audio", guide.get(GUIDE_KIND) == "audio"))
    if [row[2] for row in guide_segments] != [row[0] for row in expected]:
        raise RuntimeError("Person keyframes do not match native packed guide rows")
    # The native constructor placed all history origins at target zero. Preserve
    # those history rows; move targets, boundary and current green guide together.
    for segment, (_, history) in zip(guide_segments, expected):
        if not history:
            layout.position_ids[segment[0]:segment[1], 0] += span
    for lo, hi, _ in segments[-2:]:
        layout.position_ids[lo:hi, 0] += span
    result = dict(out)
    payload = dict(payload, layout=layout, t8_person_history_span=span)
    import comfy.conds
    result["minimax_payload"] = comfy.conds.CONDConstant(payload)
    return result


def patch_history_model(model):
    patched = model.clone()
    original = patched.get_model_object("extra_conds")
    if not callable(original):
        raise ValueError("Person history needs a callable native MODEL extra_conds")
    if getattr(getattr(original, "__func__", original), MARKER, False):
        return patched

    def extra_conds(_self, **kwargs):
        return relocate_history_payload(original(**kwargs), kwargs)

    setattr(extra_conds, MARKER, True)
    patched.add_object_patch("extra_conds", types.MethodType(extra_conds, patched.model))
    return patched


def condition_window(model, clip, vae, audio_vae, anchor, green, prompt,
                     history_frames=None, history_audio=None):
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, _encode_ref_audio
    from comfy.nested_tensor import NestedTensor
    from .nodes_branch_guide import append_branch_guide

    _rgb(green, "Green window")
    _rgb(anchor, "Window anchor")
    if len(green) != WINDOW or tuple(anchor.shape) != (1, *green.shape[1:]):
        raise ValueError("Window conditioning needs 22 green frames and one aligned anchor")
    continuation = history_frames is not None
    if continuation:
        _rgb(history_frames, "History")
        if tuple(history_frames.shape) != (HISTORY, *green.shape[1:]):
            raise ValueError("Native person continuation needs exactly 18 same-grid frames")
        if not torch.equal(anchor, history_frames[-1:]):
            raise ValueError("Continuation anchor must be its exact generated boundary frame")
        # At rates such as 44100, two globally rounded endpoints can differ by
        # one sample from a separately rounded 18-frame duration. Keep the real
        # crop; do not pad/shift the original absolute delivery clock.
        _pcm(history_audio, HISTORY, sample_tolerance=1)
    elif history_audio is not None:
        raise ValueError("Audio history cannot be supplied without video history")
    positive, latent = MiniMaxH3ReferenceToVideo.execute(
        clip, prompt, green.shape[2], green.shape[1], WINDOW + (17 if continuation else 0),
        ref_images={"ref_image_0": anchor},
        ref_videos={"ref_video_0": green}).result
    output_model = model
    if continuation:
        # A legal 22-frame encode retains all five tokens of the preceding 17
        # frames. The public VAE manager still owns encoding/offload; no private
        # keep-all encoder or alternate normalization is introduced.
        padded = torch.cat((history_frames[:17], history_frames[-1:].expand(5, -1, -1, -1)))
        encoded = vae.encode(padded)
        boundary = vae.encode(history_frames[-1:])
        if tuple(encoded.shape[:3]) != (1, 24, 7) or tuple(boundary.shape[:3]) != (1, 24, 1):
            raise ValueError("History video VAE returned an unexpected H3 token contract")
        sound, sound_steps = _encode_ref_audio(audio_vae, history_audio)
        if tuple(sound.shape[:3]) != (1, 32, 2) or sound_steps < 3:
            raise ValueError("History audio VAE returned an unexpected H3 token contract")
        guides = [dict(resolved_frame_index=0, latent=encoded[:, :, :5].clone(),
                       **{GUIDE_KIND: "video"}),
                  dict(resolved_frame_index=0, latent=boundary),
                  dict(resolved_frame_index=0, audio_latent=sound[..., :-2].clone(),
                       **{GUIDE_KIND: "audio"}),
                  dict(resolved_frame_index=0, audio_latent=sound[..., -2:].clone())]
        positive = [[embedding, dict(metadata,
                     minimax_keyframes=[*metadata.get("minimax_keyframes", []), *guides],
                     **{MARKER: 1})] for embedding, metadata in positive]
        video, audio = latent["samples"].tensors
        if video.shape[2] != 12 or audio.shape[-1] != 65:
            raise ValueError("Native 39-frame latent contract changed")
        latent = dict(latent, samples=NestedTensor((video[:, :, :7].clone(), audio[..., :37].clone())))
        output_model = patch_history_model(model)
    positive, guide_report = append_branch_guide(positive, latent, 0, vae=vae, image=green)
    report = dict(schema="t8.person-conditioning.v1", continuation=continuation,
                  source_window_frames=WINDOW, video_target_tokens=7, audio_target_tokens=37,
                  history_frames=HISTORY if continuation else 0,
                  model_clone_history_patch=continuation, global_core_patched=False,
                  references="clean_or_boundary_Picture1_then_green_Video1",
                  reference_encoding="Qwen_only_as_author_recipe_no_extra_DiT_reference_latents",
                  green_guide_report=__import__("json").loads(guide_report),
                  sampler_mask_attached=False, weights_loaded_inside_node=False,
                  sampler_executed=False, quality_verified=False)
    return output_model, positive, latent, report
