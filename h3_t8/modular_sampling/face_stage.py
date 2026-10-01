"""Source-bound native stages for the existing Face Refine variants.

The crop VAE, low-denoise sampler setup and final stitch remain separate
public nodes. This adapter neither resamples nor silently replaces them.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from .. import face_refine_advanced as face
from .. import face_refine_parity_advanced as parity
from .. import multiface_refine_advanced as multiface
from .. import face_refine_window_advanced as window
from ..core import nested_av_parts, split_noise_masks, validate_audio, video_latent_t
from . import native_explicit
from .results import StageResult, _input_identity, canonical

SCHEMA = "t8.modular-sampling.face-standard-stage.v1"
PARITY_SCHEMA = "t8.modular-sampling.face-parity-stage.v1"
MULTIFACE_SCHEMA = "t8.modular-sampling.face-multiface-stage.v1"
WINDOW_SCHEMA = "t8.modular-sampling.face-window-stage.v1"


def _face_implementation_sha256():
    return hashlib.sha256(Path(face.__file__).read_bytes()).hexdigest()


def _window_contract(plan, mapping, parent_frames, source_frames, source_audio, window_audio):
    plan = window._validate_signed(plan, window.WINDOW_PLAN_SCHEMA, "plan_sha256", "window_plan")
    mapping = window._validate_signed(mapping, window.WINDOW_MAPPING_SCHEMA, "mapping_sha256", "window_mapping")
    window._validate_source_binding(parent_frames, plan["source"], "parent_frames")
    index = int(mapping["window"]["window_index"])
    if (index < 0 or index >= len(plan["windows"]) or
            mapping["window_plan_sha256"] != plan["plan_sha256"] or
            mapping["source"] != plan["source"] or mapping["window"] != plan["windows"][index]):
        raise ValueError("Face window mapping differs from the bound plan")
    item = mapping["window"]
    start, end = int(item["render_source_start_frame"]), int(item["render_source_end_frame"])
    pre, post = int(item["pre_pad_frames"]), int(item["post_pad_frames"])
    source_count = end - start + 1
    if (not 0 <= start <= end < int(parent_frames.shape[0]) or min(pre, post) < 0 or
            source_count + pre + post != int(item["render_frame_count"]) or
            int(source_frames.shape[0]) != int(item["render_frame_count"])):
        raise ValueError("Face window source range or padding differs from plan")
    source_section = parent_frames[start:end + 1]
    if (not torch.equal(source_frames[pre:pre + source_count], source_section) or
            (pre and not torch.equal(source_frames[:pre], source_section[:1].expand(pre, -1, -1, -1))) or
            (post and not torch.equal(source_frames[-post:], source_section[-1:].expand(post, -1, -1, -1)))):
        raise ValueError("Face window frames differ from the mapped parent source")
    expected_map = []
    for relative in range(int(item["render_frame_count"])):
        if relative < pre:
            expected_map.append({"render_frame": relative, "source_frame": None,
                                 "edge_source_frame": start, "kind": "padding"})
        elif relative < pre + source_count:
            absolute = start + relative - pre
            expected_map.append({"render_frame": relative, "source_frame": absolute,
                                 "edge_source_frame": absolute, "kind": "source"})
        else:
            expected_map.append({"render_frame": relative, "source_frame": None,
                                 "edge_source_frame": end, "kind": "padding"})
    if mapping["frame_map"] != expected_map:
        raise ValueError("Face window frame map differs from current source")
    waveform, rate = validate_audio(source_audio, "source_audio")
    rendered, rendered_rate = validate_audio(window_audio, "window_audio")
    start_sample = window._frame_boundary_sample(start, rate)
    end_sample = window._frame_boundary_sample(end + 1, rate)
    render_start = window._frame_boundary_sample(pre, rate)
    render_end = window._frame_boundary_sample(pre + source_count, rate)
    target_count = window._frame_boundary_sample(int(item["render_frame_count"]), rate)
    audio_map = {"connected": True, "sample_rate": rate,
                 "source_start_sample": start_sample, "source_end_sample_exclusive": end_sample,
                 "render_start_sample": render_start, "render_end_sample_exclusive": render_end,
                 "target_sample_count": target_count, "padding_samples_are_zero": True}
    if (mapping["audio"] != audio_map or rendered_rate != rate or
            tuple(rendered.shape[:2]) != tuple(waveform.shape[:2]) or int(rendered.shape[-1]) != target_count):
        raise ValueError("Face window audio mapping differs from source audio")
    expected_audio = torch.zeros_like(rendered)
    source_slice = waveform[..., start_sample:end_sample]
    writable = min(int(source_slice.shape[-1]), max(0, render_end - render_start),
                   max(0, target_count - render_start))
    if writable:
        expected_audio[..., render_start:render_start + writable] = source_slice[..., :writable]
    if not torch.equal(rendered, expected_audio):
        raise ValueError("Face window audio differs from the mapped source audio")
    return {"window_plan_sha256": plan["plan_sha256"], "window_mapping_sha256": mapping["mapping_sha256"],
            "window_index": index, "absolute_source_window": [start, end],
            "parent_frames": _input_identity(parent_frames),
            "source_audio": _input_identity(source_audio), "window_audio": _input_identity(window_audio),
            "window_implementation_sha256": hashlib.sha256(Path(window.__file__).read_bytes()).hexdigest()}


def _source_contract(face_plan, source_frames, av_latent, audio_policy, *, variant="standard",
                     parent_frames=None, window_plan=None, window_mapping=None,
                     source_audio=None, window_audio=None):
    if variant == "standard":
        plan = face._validate_plan(face_plan)
    elif variant in ("parity", "multiface", "window"):
        plan = parity._validate_parity_plan(face_plan)
    else:
        raise ValueError("Unknown Face refinement stage variant")
    count, height, width = face._validate_frames(source_frames, name="source_frames")
    source = plan["source"]
    if (count, width, height) != (int(source["frame_count"]), int(source["width"]), int(source["height"])):
        raise ValueError("Face source dimensions differ from the bound plan")
    if face.source_proxy_sha256(source_frames) != source["proxy_sha256"]:
        raise ValueError("Face source frames differ from the bound plan proxy")
    video, audio = nested_av_parts(av_latent)
    canvas = plan["canvas"]
    aligned_count = count
    if variant in ("parity", "multiface", "window"):
        aligned_count = int(source.get("h3_aligned_frame_count", count))
        if aligned_count - count not in (0, 1) or int(source.get("h3_alignment_tail_frames", -1)) != aligned_count - count:
            raise ValueError("Face parity plan has an unsupported H3 alignment tail")
    expected = (video_latent_t(aligned_count), int(canvas["height"]) // 16,
                int(canvas["width"]) // 16)
    if tuple(video.shape[2:]) != expected:
        raise ValueError("Face AV latent time/canvas differs from the bound plan")
    if audio_policy not in ("require_locked", "preserve_existing"):
        raise ValueError("Unknown face audio policy")
    _, audio_mask = split_noise_masks(av_latent, video, audio)
    if audio_policy == "require_locked":
        if audio_mask is None or tuple(audio_mask.shape) != tuple(audio.shape) or bool(torch.count_nonzero(audio_mask)):
            raise ValueError("Face require_locked needs an exactly zero nested audio mask")
    schema = {"standard": SCHEMA, "parity": PARITY_SCHEMA, "multiface": MULTIFACE_SCHEMA,
              "window": WINDOW_SCHEMA}[variant]
    contract = {"schema": schema,
            "variant": variant, "plan_sha256": plan["plan_sha256"],
            "source_frames": _input_identity(source_frames), "av_latent": _input_identity(av_latent),
            "audio_policy": audio_policy, "face_implementation_sha256": _face_implementation_sha256()}
    if variant in ("parity", "multiface", "window"):
        contract["parity_implementation_sha256"] = hashlib.sha256(Path(parity.__file__).read_bytes()).hexdigest()
        contract["h3_aligned_frame_count"] = aligned_count
    if variant == "multiface":
        group = plan.get("multiface")
        if not isinstance(group, dict) or group.get("sequential_generation_required") is not True:
            raise ValueError("Multi-face stage needs a sequential repair-job plan")
        if parent_frames is None:
            raise ValueError("Multi-face stage needs current parent frames")
        parent_count, parent_height, parent_width = face._validate_frames(parent_frames, name="parent_frames")
        start = int(group["window_start_absolute"])
        end = int(group["window_end_absolute"])
        source_count = int(group["source_window_frame_count"])
        model_count = int(group["model_window_frame_count"])
        pad = int(group["alignment_context_pad_frames"])
        if (parent_height, parent_width) != (height, width) or not (0 <= start <= end < parent_count):
            raise ValueError("Multi-face parent/source window dimensions differ")
        if source_count != end - start + 1 or model_count != count or not (0 <= pad <= 16) or pad != count - source_count:
            raise ValueError("Multi-face window/padding contract differs from plan")
        source_window = parent_frames[start:end + 1]
        if (face.source_proxy_sha256(parent_frames) != group["parent_source_proxy_sha256"] or
                face.source_proxy_sha256(source_window) != group["source_window_proxy_sha256"] or
                not torch.equal(source_frames[:source_count], source_window) or
                (pad and not torch.equal(source_frames[source_count:], source_window[-1:].expand(pad, -1, -1, -1)))):
            raise ValueError("Multi-face source frames do not match the parent repair window")
        contract["parent_frames"] = _input_identity(parent_frames)
        contract["multiface_implementation_sha256"] = hashlib.sha256(Path(multiface.__file__).read_bytes()).hexdigest()
        contract["character_id"] = group["character_id"]
        contract["track_key"] = group["track_key"]
        contract["absolute_window"] = [start, end]
    if variant == "window":
        if any(value is None for value in (parent_frames, window_plan, window_mapping,
                                           source_audio, window_audio)):
            raise ValueError("Face window stage needs parent frames, mapping and both audio sources")
        contract.update(_window_contract(window_plan, window_mapping, parent_frames, source_frames,
                                         source_audio, window_audio))
    return contract


def bind_standard_face_stage(face_plan, source_frames, model, sampler, sigmas, av_latent,
                             audio_policy="require_locked"):
    return _bind_face_stage(face_plan, source_frames, model, sampler, sigmas, av_latent,
                            audio_policy, variant="standard")


def bind_parity_face_stage(face_plan, source_frames, model, sampler, sigmas, av_latent,
                           audio_policy="require_locked"):
    return _bind_face_stage(face_plan, source_frames, model, sampler, sigmas, av_latent,
                            audio_policy, variant="parity")


def bind_multiface_face_stage(face_plan, source_frames, parent_frames, model, sampler, sigmas,
                              av_latent, audio_policy="require_locked"):
    return _bind_face_stage(face_plan, source_frames, model, sampler, sigmas, av_latent,
                            audio_policy, variant="multiface", parent_frames=parent_frames)


def bind_window_face_stage(face_plan, source_frames, parent_frames, window_plan, window_mapping,
                           source_audio, window_audio, model, sampler, sigmas, av_latent,
                           audio_policy="require_locked"):
    return _bind_face_stage(face_plan, source_frames, model, sampler, sigmas, av_latent,
                            audio_policy, variant="window", parent_frames=parent_frames,
                            window_plan=window_plan, window_mapping=window_mapping,
                            source_audio=source_audio, window_audio=window_audio)


def _bind_face_stage(face_plan, source_frames, model, sampler, sigmas, av_latent, audio_policy, *, variant,
                     parent_frames=None, window_plan=None, window_mapping=None,
                     source_audio=None, window_audio=None):
    source = _source_contract(face_plan, source_frames, av_latent, audio_policy, variant=variant,
                              parent_frames=parent_frames, window_plan=window_plan,
                              window_mapping=window_mapping, source_audio=source_audio,
                              window_audio=window_audio)
    prepared, selected, table, context, _ = native_explicit.bind_stage(
        model, sampler, sigmas, av_latent, stage="native_high")
    profile = json.loads(context.profile)
    profile["face_refine"] = source
    context = replace(context, profile=canonical(profile),
                      input_semantics="face_crop_video_with_original_audio_policy",
                      output_semantics="face_refinement_candidate_av",
                      denoised_semantics="face_refinement_candidate_prediction_x0")
    owner = native_explicit.capture_owner(prepared)
    prepared.set_attachments(native_explicit.KEY, replace(owner, context=context))
    report = {"schema": source["schema"], "stage_context": context.to_dict(),
              "plan_sha256": source["plan_sha256"], "variant": variant,
              "sampled": False, "cache_reuse_authorized": False,
              "portable_completion_sampler_adapted": json.loads(context.profile)["sampler_kind"] != "unadapted",
              "eav_forward_coverage_adapter": profile.get("eav_forward_coverage_adapter", "none"),
              "boundary": "Keep crop conditioning, the connected sampler/sigma table, source-bound delivery audit "
                          "and stitch external. Unadapted Core samplers may run but cannot claim completed portable "
                          "StageResult or Save/Load eligibility; use the variant-specific source adapter."}
    return prepared, selected, table, context, canonical(report)


def audit_standard_face_stage(stage_result, face_plan, source_frames, av_latent):
    return _audit_face_stage(stage_result, face_plan, source_frames, av_latent, variant="standard")


def audit_parity_face_stage(stage_result, face_plan, source_frames, av_latent):
    return _audit_face_stage(stage_result, face_plan, source_frames, av_latent, variant="parity")


def audit_multiface_face_stage(stage_result, face_plan, source_frames, parent_frames, av_latent):
    return _audit_face_stage(stage_result, face_plan, source_frames, av_latent, variant="multiface",
                             parent_frames=parent_frames)


def audit_window_face_stage(stage_result, face_plan, source_frames, parent_frames, window_plan,
                            window_mapping, source_audio, window_audio, av_latent):
    return _audit_face_stage(stage_result, face_plan, source_frames, av_latent, variant="window",
                             parent_frames=parent_frames, window_plan=window_plan,
                             window_mapping=window_mapping, source_audio=source_audio,
                             window_audio=window_audio)


def _audit_face_stage(stage_result, face_plan, source_frames, av_latent, *, variant, parent_frames=None,
                      window_plan=None, window_mapping=None, source_audio=None, window_audio=None):
    if type(stage_result) is not StageResult:
        raise ValueError("Face stage audit needs a sampler-produced StageResult")
    receipt = stage_result.verify()
    from .contracts import StageContext
    context = StageContext.from_dict(receipt["request"]["stage_context"])
    if context.recipe != native_explicit.RECIPE or context.stage != "native_high":
        raise ValueError("Face audit received another stage recipe")
    profile = json.loads(context.profile)
    bound = profile.get("face_refine")
    if not isinstance(bound, dict) or bound.get("schema") != {
            "standard": SCHEMA, "parity": PARITY_SCHEMA, "multiface": MULTIFACE_SCHEMA,
            "window": WINDOW_SCHEMA}[variant]:
        raise ValueError("Face audit received a stage without source binding")
    current = _source_contract(face_plan, source_frames, av_latent, bound["audio_policy"],
                               variant=variant, parent_frames=parent_frames,
                               window_plan=window_plan, window_mapping=window_mapping,
                               source_audio=source_audio, window_audio=window_audio)
    if current != bound or receipt["request"]["source"] != bound["av_latent"]:
        raise ValueError("Face plan, source frames or input AV changed after stage binding")
    _, original_audio = nested_av_parts(av_latent)
    _, output_audio = nested_av_parts(stage_result.output)
    audio_unchanged = torch.equal(original_audio, output_audio)
    report = {"schema": bound["schema"], "status": "source_bound_candidate_audited",
              "variant": variant,
              "plan_sha256": bound["plan_sha256"], "receipt_sha256": receipt["receipt_sha256"],
              "verified_recipe_completion": receipt["verified_recipe_completion"],
              "audio_policy": bound["audio_policy"], "audio_latent_unchanged": audio_unchanged,
              "delivery_audio": "original_source_audio_only",
              "automatic_accept": False, "cache_reuse_authorized": False,
              "boundary": "The input mask is locked, but Core's sampled audio latent may drift. "
                          "Decode candidate VIDEO only and deliver the original source audio separately; "
                          "this audit is not face quality or identity approval."}
    return stage_result.output, canonical(report)
