"""Non-sampling source guard for existing, already-separated Audio Refine tails.

This adapter deliberately preserves the old empty-SIGMAS abstain path. It is
not a portable StageResult or proof that an arbitrary external sampler ran.
"""
import hashlib
import json
import math

import comfy.samplers
import torch

from .. import audio_refine_advanced as refine
from ..core import nested_av_parts, split_noise_masks
from ..learned_latent_upscale_advanced import audit_two_pass_h3_audio
from .results import _input_identity, canonical

SCHEMA = "t8.modular-sampling.audio-refine-boundary.v1"
PLAN_SCHEMAS = {
    refine.AUDIO_REFINE_PLAN_SCHEMA: "dual_clock",
    refine.AUDIO_REFINE_PHASE2_PLAN_SCHEMA: "dual_model",
    refine.AUDIO_REFINE_COMPAT_PLAN_SCHEMA: "compatibility",
}


def _sha(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _validate(plan, original_av_latent, model, noise, guider, sampler, sigmas,
              stage_latent, setup_report_json, *, expected_variant=None):
    if not isinstance(plan, dict) or plan.get("schema") not in PLAN_SCHEMAS:
        raise ValueError("Unsupported Audio Refine signed Plan")
    variant = PLAN_SCHEMAS[plan["schema"]]
    if expected_variant is not None and variant != expected_variant:
        raise ValueError("Audio Refine Plan belongs to another stage variant")
    plan = refine._validate_signed_descriptor(plan, schema=plan["schema"], label="audio refine Plan")
    if plan["decision"] == "REJECT":
        raise ValueError("Rejected Audio Refine Plan cannot bind a sampler")
    try:
        setup = json.loads(setup_report_json)
    except (TypeError, ValueError) as error:
        raise ValueError("Audio Refine setup report is not valid JSON") from error
    if not isinstance(setup, dict) or setup.get("plan_payload_sha256") != plan["payload_sha256"]:
        raise ValueError("Audio Refine setup report differs from signed Plan")
    if type(guider) is not refine.AudioRefineBasicGuider or guider.model_patcher is not model or guider.cfg != 1.0:
        raise ValueError("Audio Refine connected guider/model differ from Setup")
    if type(sampler) is not comfy.samplers.KSAMPLER:
        raise ValueError("Audio Refine connected sampler is not a Core KSAMPLER")
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1:
        raise ValueError("Audio Refine connected SIGMAS are invalid")
    source_video, source_audio = nested_av_parts(original_av_latent)
    stage_video, stage_audio = nested_av_parts(stage_latent)
    if (tuple(source_video.shape) != tuple(stage_video.shape) or
            tuple(source_audio.shape) != tuple(stage_audio.shape) or
            not torch.equal(source_video, stage_video) or not torch.equal(source_audio, stage_audio)):
        raise ValueError("Audio Refine Setup changed original AV samples")
    bypassed = bool(setup.get("bypassed"))
    if bypassed:
        if (setup.get("decision") != "ABSTAIN" or sigmas.numel() != 0 or
                type(noise) is not refine.AudioRefineBypassNoise or
                sampler.sampler_function is not refine._never_sample):
            raise ValueError("Audio Refine abstain route no longer has exact no-sample controls")
    else:
        if (setup.get("decision") != "ALLOW" or plan["decision"] != "ALLOW" or
                type(noise) is not refine.AudioRefineRandomNoise or
                noise.seed != plan["refine_seed"] or sigmas.numel() != plan["actual_refine_nfe"] + 1 or
                not bool(torch.allclose(sigmas.cpu(), torch.tensor(plan["video_sigmas"], dtype=sigmas.dtype),
                                        rtol=1e-6, atol=1e-7))):
            raise ValueError("Audio Refine sample controls differ from signed Plan")
        video_mask, audio_mask = split_noise_masks(stage_latent, stage_video, stage_audio)
        if (video_mask is None or audio_mask is None or
                tuple(video_mask.shape) != tuple(stage_video.shape) or
                tuple(audio_mask.shape) != tuple(stage_audio.shape) or
                bool(torch.count_nonzero(video_mask)) or
                not bool(torch.all(audio_mask == 1))):
            raise ValueError("Audio Refine stage needs the exact 0-video/1-audio masks")
    if (setup.get("bypassed") is not bypassed or
            setup.get("sigmas") != [float(value) for value in sigmas.tolist()]):
        raise ValueError("Audio Refine setup report and connected controls differ")
    return {"variant": variant, "plan": plan, "setup": setup, "bypassed": bypassed,
            "source_identity": _input_identity(original_av_latent),
            "stage_identity": _input_identity(stage_latent)}


def bind_audio_refine_stage(plan, original_av_latent, model, noise, guider, sampler,
                            sigmas, stage_latent, setup_report_json, *, expected_variant=None):
    validated = _validate(plan, original_av_latent, model, noise, guider, sampler,
                          sigmas, stage_latent, setup_report_json,
                          expected_variant=expected_variant)
    context = {"schema": SCHEMA, "variant": validated["variant"],
               "plan_payload_sha256": validated["plan"]["payload_sha256"],
               "setup_report_sha256": hashlib.sha256(setup_report_json.encode("utf-8")).hexdigest(),
               "source": validated["source_identity"], "stage_latent": validated["stage_identity"],
               "bypassed": validated["bypassed"], "model_object_id": id(model),
               "noise_object_id": id(noise), "guider_object_id": id(guider),
               "sampler_object_id": id(sampler), "sigmas": [float(value) for value in sigmas.tolist()],
               "sampling_completion_verified": False, "portable_cache_reuse_authorized": False}
    context["context_sha256"] = _sha(context)
    report = {"schema": SCHEMA, "status": "abstain_passthrough" if context["bypassed"] else "ready_to_sample",
              "variant": context["variant"], "plan_payload_sha256": context["plan_payload_sha256"],
              "context_sha256": context["context_sha256"], "sampled": False,
              "portable_cache_reuse_authorized": False,
              "boundary": "Existing Setup controls and Core SamplerCustomAdvanced are passed through unchanged. "
                          "Empty-SIGMAS abstain stays a no-sample path; external effects/resume are not certified."}
    return model, noise, guider, sampler, sigmas, stage_latent, context, canonical(report)


def audit_audio_refine_stage(context, plan, original_av_latent, stage_latent,
                             setup_report_json, candidate_av_latent):
    if not isinstance(context, dict) or context.get("schema") != SCHEMA:
        raise ValueError("Audio Refine candidate lacks a stage boundary")
    unsigned = dict(context)
    digest = unsigned.pop("context_sha256", None)
    if digest != _sha(unsigned):
        raise ValueError("Audio Refine stage boundary SHA-256 differs")
    checked = refine._validate_signed_descriptor(plan, schema=plan.get("schema"), label="audio refine Plan")
    if (context["variant"] != PLAN_SCHEMAS.get(checked["schema"]) or
            context["plan_payload_sha256"] != checked["payload_sha256"] or
            context["setup_report_sha256"] != hashlib.sha256(setup_report_json.encode("utf-8")).hexdigest() or
            context["source"] != _input_identity(original_av_latent) or
            context["stage_latent"] != _input_identity(stage_latent)):
        raise ValueError("Audio Refine candidate source or Plan changed after binding")
    source_video, source_audio = nested_av_parts(original_av_latent)
    candidate_video, candidate_audio = nested_av_parts(candidate_av_latent)
    if (tuple(candidate_video.shape) != tuple(source_video.shape) or
            tuple(candidate_audio.shape) != tuple(source_audio.shape) or
            not bool(torch.isfinite(candidate_video).all()) or
            not bool(torch.isfinite(candidate_audio).all())):
        raise ValueError("Audio Refine candidate AV shape or values are invalid")
    if context["bypassed"] and (not torch.equal(candidate_video, source_video) or
                                not torch.equal(candidate_audio, source_audio)):
        raise ValueError("Audio Refine abstain path sampled or changed original AV")
    report = {"schema": SCHEMA, "status": "abstain_passthrough" if context["bypassed"] else "candidate_received",
              "variant": context["variant"], "plan_payload_sha256": context["plan_payload_sha256"],
              "context_sha256": digest, "sampler_execution_proven": False,
              "video_latent_unchanged": torch.equal(candidate_video, source_video),
              "quality_acceptance": False, "portable_cache_reuse_authorized": False,
              "boundary": "The existing Quality Gate must decide audio adoption and relock original video. "
                          "A candidate connection alone is not an executed StageResult or human listening."}
    return candidate_av_latent, canonical(report)


def audit_audio_refine_tail_delivery(stage_boundary, stage_report_json,
                                     second_pass_input, second_pass_output,
                                     expected_audio_strength, fail_on_locked_mismatch,
                                     locked_atol):
    """Keep the old sampled-audio audit, but explicitly pass through abstains."""
    if not isinstance(stage_boundary, dict) or stage_boundary.get("schema") != SCHEMA:
        raise ValueError("Audio Refine delivery lacks a stage boundary")
    unsigned = dict(stage_boundary)
    digest = unsigned.pop("context_sha256", None)
    if digest != _sha(unsigned):
        raise ValueError("Audio Refine delivery boundary SHA-256 differs")
    try:
        stage_report = json.loads(stage_report_json)
    except (TypeError, ValueError) as error:
        raise ValueError("Audio Refine delivery stage report is invalid") from error
    if (not isinstance(stage_report, dict) or stage_report.get("schema") != SCHEMA or
            stage_report.get("context_sha256") != digest or
            stage_report.get("variant") != stage_boundary.get("variant") or
            stage_boundary.get("stage_latent") != _input_identity(second_pass_input)):
        raise ValueError("Audio Refine delivery stage report or input changed")
    strength = float(expected_audio_strength)
    tolerance = float(locked_atol)
    if (not math.isfinite(strength) or not 0.0 <= strength <= 1.0 or
            not math.isfinite(tolerance) or tolerance < 0.0):
        raise ValueError("Audio Refine delivery audit parameters are invalid")
    if not stage_boundary["bypassed"]:
        if stage_report.get("status") != "candidate_received":
            raise ValueError("Audio Refine sampled tail lacks candidate audit")
        return audit_two_pass_h3_audio(
            second_pass_input, second_pass_output, strength,
            fail_on_locked_mismatch, tolerance)
    if stage_report.get("status") != "abstain_passthrough":
        raise ValueError("Audio Refine abstain tail lacks passthrough audit")
    input_video, input_audio = nested_av_parts(second_pass_input)
    output_video, output_audio = nested_av_parts(second_pass_output)
    if (tuple(input_video.shape) != tuple(output_video.shape) or
            tuple(input_audio.shape) != tuple(output_audio.shape) or
            not torch.equal(input_video, output_video) or
            not torch.equal(input_audio, output_audio) or
            not bool(torch.isfinite(input_video).all()) or
            not bool(torch.isfinite(input_audio).all())):
        raise ValueError("Audio Refine abstain tail changed original AV")
    report = {"schema": "t8.modular-sampling.audio-refine-tail-delivery.v1",
              "status": "abstain_no_sample", "variant": stage_boundary["variant"],
              "context_sha256": digest, "sampled": False,
              "output_original_av_exact": True,
              "portable_cache_reuse_authorized": False,
              "quality_claim": "none; original AV is retained"}
    return second_pass_input, canonical(report)
