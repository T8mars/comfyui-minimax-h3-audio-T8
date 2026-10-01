"""Source-bound stage receipt for an externally sampled Motion Recovery pass."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from .. import motion_recovery_advanced as motion
from ..core import AUDIO_LATENT_FPS, FPS, nested_av_parts, split_noise_masks, validate_audio, video_latent_t
from . import native_explicit
from .contracts import StageContext
from .results import StageResult, _input_identity, canonical

SCHEMA = "t8.modular-sampling.motion-recovery-stage.v1"


def _source_contract(plan, baseline_frames, baseline_audio, smeared_frames, smeared_audio,
                     prepare_report_json, av_latent, *, parent_plan=None, parent_frames=None,
                     parent_audio=None):
    plan = motion.validate_motion_plan(plan)
    if plan["status"] != "ready" or plan["expanded_length"] <= plan["world_length"]:
        raise ValueError("Motion stage requires a ready second-pass plan")
    if (not isinstance(baseline_frames, torch.Tensor) or baseline_frames.ndim != 4 or
            list(baseline_frames.shape) != plan["source"]["frame_shape"]):
        raise ValueError("Motion baseline frames differ from the signed plan")
    if (not isinstance(smeared_frames, torch.Tensor) or smeared_frames.ndim != 4 or
            tuple(smeared_frames.shape) != (plan["expanded_length"], *baseline_frames.shape[1:])):
        raise ValueError("Motion smeared frames differ from the expanded plan")
    indices = torch.tensor([index for index, hold in enumerate(plan["holds"]) for _ in range(hold)],
                           dtype=torch.long, device=baseline_frames.device)
    if not torch.equal(smeared_frames, baseline_frames.index_select(0, indices).to(smeared_frames.device)):
        raise ValueError("Motion smeared frames do not match baseline hold mapping")
    source_wave, source_rate = validate_audio(baseline_audio, "baseline_audio")
    seed_wave, seed_rate = validate_audio(smeared_audio, "smeared_audio")
    try:
        prepare_report = json.loads(prepare_report_json)
    except (TypeError, ValueError) as error:
        raise ValueError("Motion prepare report is not valid JSON") from error
    if not isinstance(prepare_report, dict):
        raise ValueError("Motion prepare report must be an object")
    seed_mode = prepare_report.get("audio_seed_mode")
    if (seed_mode not in motion.ALLOWED_AUDIO_SEED_MODES or
            prepare_report.get("plan_sha256") != plan["plan_sha256"] or
            prepare_report.get("world_length") != plan["world_length"] or
            prepare_report.get("expanded_length") != plan["expanded_length"]):
        raise ValueError("Motion prepare report differs from the current plan")
    expected_rate = 32000 if seed_mode == "none_invent_exp" else source_rate
    if (seed_rate != expected_rate or
            (seed_mode != "none_invent_exp" and
             tuple(seed_wave.shape[:2]) != tuple(source_wave.shape[:2])) or
            seed_wave.shape[-1] != round(plan["expanded_length"] / FPS * expected_rate)):
        raise ValueError("Motion smeared audio clock differs from preparation")
    if seed_mode == "none_invent_exp" and (tuple(seed_wave.shape[:2]) != (1, 2) or
                                           bool(torch.count_nonzero(seed_wave))):
        raise ValueError("Motion invented audio seed must be stereo silence")
    if (av_latent.get("h3_t8_motion_plan_sha256") != plan["plan_sha256"] or
            av_latent.get("h3_t8_motion_world_length") != plan["world_length"] or
            av_latent.get("h3_t8_motion_expanded_length") != plan["expanded_length"]):
        raise ValueError("Motion AV latent is not from this signed plan")
    video, audio = nested_av_parts(av_latent)
    if (tuple(video.shape) != (1, 24, video_latent_t(plan["expanded_length"]),
                              int(baseline_frames.shape[1]) // 16, int(baseline_frames.shape[2]) // 16) or
            tuple(audio.shape) != (1, 32, 2, round(plan["expanded_length"] / FPS * AUDIO_LATENT_FPS))):
        raise ValueError("Motion AV latent time/canvas differs from retimed source")
    video_mask, audio_mask = split_noise_masks(av_latent, video, audio)
    strength = motion.ALLOWED_AUDIO_SEED_MODES[seed_mode]
    if (video_mask is None or audio_mask is None or
            tuple(video_mask.shape) != tuple(video.shape) or tuple(audio_mask.shape) != tuple(audio.shape) or
            not bool(torch.allclose(video_mask, torch.ones_like(video_mask))) or
            not bool(torch.allclose(audio_mask, torch.full_like(audio_mask, strength), atol=1e-6, rtol=0))):
        raise ValueError("Motion AV noise masks differ from audio seed policy")
    contract = {"schema": SCHEMA, "plan_sha256": plan["plan_sha256"],
            "variant": "windowed" if plan["schema"] == motion.MOTION_WINDOW_SCHEMA else "full_clip",
            "baseline_frames": _input_identity(baseline_frames),
            "baseline_audio": _input_identity(baseline_audio),
            "smeared_frames": _input_identity(smeared_frames),
            "smeared_audio": _input_identity(smeared_audio),
            "prepare_report": _input_identity(prepare_report_json),
            "av_latent": _input_identity(av_latent), "audio_seed_mode": seed_mode,
            "motion_implementation_sha256": hashlib.sha256(Path(motion.__file__).read_bytes()).hexdigest()}
    if plan["schema"] == motion.MOTION_WINDOW_SCHEMA:
        if any(value is None for value in (parent_plan, parent_frames, parent_audio)):
            raise ValueError("Motion window stage requires its current parent plan, frames and audio")
        parent = motion.validate_motion_plan(parent_plan, allow_window=False)
        if (parent["plan_sha256"] != plan["parent_plan_sha256"] or
                not isinstance(parent_frames, torch.Tensor) or parent_frames.ndim != 4 or
                list(parent_frames.shape) != parent["source"]["frame_shape"]):
            raise ValueError("Motion window parent plan or frames differ")
        window = plan["window"]
        expected = motion._window_plan_from_parent(
            parent, start=window["source_start"], end=window["source_end"],
            core_start=window["core_start"], core_end=window["core_end"],
            window_index=window["index"], window_count=window["count"],
            hot_in=window["hot_in"], hot_out=window["hot_out"])
        if expected != plan:
            raise ValueError("Motion window plan differs from the current parent plan")
        start, end = window["source_start"], window["source_end"]
        if not torch.equal(baseline_frames, parent_frames[start:end + 1].detach().cpu()):
            raise ValueError("Motion window baseline frames differ from parent slice")
        parent_wave, parent_rate = validate_audio(parent_audio, "parent_audio")
        if parent_rate != source_rate:
            raise ValueError("Motion window source audio rate differs from parent")
        start_sample = round(start / FPS * parent_rate)
        end_sample = round((end + 1) / FPS * parent_rate)
        expected_audio = motion._fit_audio_samples(
            {"waveform": parent_wave[..., start_sample:end_sample], "sample_rate": parent_rate},
            round(plan["world_length"] / FPS * parent_rate))
        if not torch.equal(source_wave, expected_audio["waveform"]):
            raise ValueError("Motion window baseline audio differs from parent slice")
        contract.update({"parent_plan_sha256": parent["plan_sha256"],
                         "parent_frames": _input_identity(parent_frames),
                         "parent_audio": _input_identity(parent_audio),
                         "absolute_window": [start, end]})
    elif any(value is not None for value in (parent_plan, parent_frames, parent_audio)):
        raise ValueError("Full-clip motion stage must not claim a window parent")
    return contract


def bind_motion_stage(plan, baseline_frames, baseline_audio, smeared_frames, smeared_audio,
                      prepare_report_json, model, sampler, sigmas, av_latent, *,
                      parent_plan=None, parent_frames=None, parent_audio=None):
    source = _source_contract(plan, baseline_frames, baseline_audio, smeared_frames,
                              smeared_audio, prepare_report_json, av_latent,
                              parent_plan=parent_plan, parent_frames=parent_frames,
                              parent_audio=parent_audio)
    prepared, selected, table, context, _ = native_explicit.bind_stage(
        model, sampler, sigmas, av_latent, stage="native_high")
    profile = json.loads(context.profile)
    profile["motion_recovery"] = source
    context = replace(context, profile=canonical(profile),
                      input_semantics="motion_retimed_av_with_original_audio_delivery",
                      output_semantics="motion_refinement_candidate_av",
                      denoised_semantics="motion_refinement_prediction_x0")
    owner = native_explicit.capture_owner(prepared)
    prepared.set_attachments(native_explicit.KEY, replace(owner, context=context))
    report = {"schema": SCHEMA, "stage_context": context.to_dict(),
              "plan_sha256": source["plan_sha256"], "variant": source["variant"],
              "sampled": False, "cache_reuse_authorized": False,
              "portable_completion_sampler_adapted": profile["sampler_kind"] != "unadapted",
              "boundary": "Binds exact connected frames/audio/AV but does not re-encode with the VAE or "
                          "re-run the phase vocoder. Keep source audio and final Recover/Collect/Gate external."}
    return prepared, selected, table, context, canonical(report)


def audit_motion_stage(stage_result, plan, baseline_frames, baseline_audio, smeared_frames,
                       smeared_audio, prepare_report_json, av_latent, *,
                       parent_plan=None, parent_frames=None, parent_audio=None):
    if type(stage_result) is not StageResult:
        raise ValueError("Motion stage audit needs a sampler-produced StageResult")
    receipt = stage_result.verify()
    context = StageContext.from_dict(receipt["request"]["stage_context"])
    if context.recipe != native_explicit.RECIPE or context.stage != "native_high":
        raise ValueError("Motion audit received another stage recipe")
    bound = json.loads(context.profile).get("motion_recovery")
    if not isinstance(bound, dict) or bound.get("schema") != SCHEMA:
        raise ValueError("Motion audit received an unbound stage")
    current = _source_contract(plan, baseline_frames, baseline_audio, smeared_frames,
                               smeared_audio, prepare_report_json, av_latent,
                               parent_plan=parent_plan, parent_frames=parent_frames,
                               parent_audio=parent_audio)
    if current != bound or receipt["request"]["source"] != bound["av_latent"]:
        raise ValueError("Motion source, plan or input AV changed after binding")
    report = {"schema": SCHEMA, "status": "source_bound_candidate_audited",
              "plan_sha256": bound["plan_sha256"], "variant": bound["variant"],
              "receipt_sha256": receipt["receipt_sha256"],
              "verified_recipe_completion": receipt["verified_recipe_completion"],
              "delivery_audio": "pass1_original_default_only",
              "automatic_accept": False, "cache_reuse_authorized": False,
              "boundary": "Recover AV, optional experimental audio choice, window collection and human "
                          "motion/audio quality review remain external."}
    return stage_result.output, canonical(report)
