"""Source-bound opt-in handoff for learned H3 video LATENT -> external LTX sampler.

The learned adapter is not an RGB/VAE/upscaler bridge. This boundary makes its
video-only conversion and original H3 AV/audio handoff explicit, without a
portable completion or quality claim.
"""

import hashlib
import json
import math

import comfy.samplers
import torch

from .. import sol_engine_h3_super_advanced as sol
from ..h3_ltx_adapter_runtime import MODEL_SHA, MODEL_REVISION, SOURCE_REVISION
from ..h3_ltx_latent_contract import extract_video, timeline
from .results import _input_identity, canonical

SCHEMA = "t8.modular-sampling.ltx-learned-stage.v1"
ADAPTER_SCHEMA = "t8.h3_ltx.standard_latent.v1"
OFFICIAL_PIPELINE = "nvidia_h3_super_acceleration_ltx25_stage2"
IDENTITY_PIPELINE = "ltx25_stage2_identity_preserve_low_sigma_exp"


def _report(raw, label):
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _tensor(value, label, ndim):
    if (not isinstance(value, torch.Tensor) or value.is_meta or value.ndim != ndim
            or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{label} must be a finite floating {ndim}D tensor")
    return value


def _adapter_source(h3_latent, original_h3_av, ltx_video_latent, adapter_report_json):
    if original_h3_av is not h3_latent:
        raise ValueError("Learned adapter must return the unchanged source H3 AV object")
    report = _report(adapter_report_json, "Learned adapter report")
    if (report.get("schema") != ADAPTER_SCHEMA or
            report.get("model_sha256") != MODEL_SHA or
            report.get("model_revision") != MODEL_REVISION or
            report.get("source_revision") != SOURCE_REVISION or
            report.get("output_normalization") != "normalized_ltx_video" or
            report.get("additional_upscaler_calls") != 0 or
            report.get("mask_policy") != "H3 noise masks/reference metadata are not transferred to LTX"):
        raise ValueError("Learned adapter report is not the owned video-only conversion")
    source_frames = report.get("source_frames")
    policy = report.get("temporal_policy")
    geometry = timeline(source_frames, report.get("source_fps"), policy)
    if any(report.get(key) != geometry[key] for key in geometry):
        raise ValueError("Learned adapter report differs from the native temporal contract")
    video, audio = extract_video(h3_latent,
        expected_frames=geometry["h3_latent_frames"],
        reference_prefix_latents=report.get("reference_prefix_latents"))
    if (report.get("source_video_shape") != list(video.shape) or
            report.get("source_audio_shape") != (list(audio.shape) if audio is not None else None)):
        raise ValueError("Learned adapter source AV shape differs from report")
    if not isinstance(ltx_video_latent, dict) or set(ltx_video_latent) != {"samples"}:
        raise ValueError("Learned LTX output must contain only video samples")
    samples = _tensor(ltx_video_latent["samples"], "Learned LTX video", 5)
    width, height = int(video.shape[-1]) * 16, int(video.shape[-2]) * 16
    expected = (video.shape[0], 128, geometry["output_latent_frames"], height // 32, width // 32)
    if (tuple(samples.shape) != expected or report.get("output_video_shape") != list(expected)
            or report.get("width") != width or report.get("height") != height):
        raise ValueError("Learned LTX latent differs from adapter geometry")
    # Explicit trim/pad needs a separate audio policy; the candidate video+audio
    # workflow must never silently mux a source track with a changed duration.
    if geometry["audio_duration_adjustment_required"]:
        raise ValueError("Learned LTX duration change requires an explicit audio reconciliation stage")
    return report, samples, audio


def _setup(model, guider, sampler, sigmas, setup_report_json):
    report = _report(setup_report_json, "LTX Setup report")
    profile = report.get("pipeline")
    if profile == OFFICIAL_PIPELINE:
        expected = sol.OFFICIAL_STAGE2_SIGMAS
        variant = "official_stage2"
    elif profile == IDENTITY_PIPELINE:
        expected = report.get("stage2_sigmas")
        variant = "identity_preserve"
    elif report.get("status") in {"disabled_passthrough", "unsupported_model_passthrough"}:
        expected = report.get("stage2_sigmas", sol.OFFICIAL_STAGE2_SIGMAS)
        variant = "official_stage2_passthrough"
    else:
        raise ValueError("LTX Setup report is not a supported refiner profile")
    if report.get("status") not in {"configured", "disabled_passthrough", "unsupported_model_passthrough"}:
        raise ValueError("LTX Setup did not configure or explicitly bypass the refiner")
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) != 4:
        raise ValueError("Learned LTX stage needs three explicit sigma updates")
    values = [float(value) for value in sigmas.detach().cpu().tolist()]
    if (not all(math.isfinite(value) for value in values) or values[-1] != 0
            or any(left <= right for left, right in zip(values, values[1:]))
            or not isinstance(expected, (list, tuple)) or len(expected) != 4
            or any(not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-7)
                   for left, right in zip(values, expected))):
        raise ValueError("LTX SIGMAS differ from the connected Setup profile")
    if not isinstance(guider, comfy.samplers.CFGGuider) or guider.model_patcher is not model:
        raise ValueError("LTX guider is not bound to the connected MODEL")
    if not isinstance(sampler, comfy.samplers.KSAMPLER):
        raise ValueError("LTX stage requires an external Core sampler")
    return variant, values


def _contract(h3_latent, original_h3_av, ltx_video_latent, adapter_report_json,
              model, noise, guider, sampler, sigmas, setup_report_json):
    report, samples, audio = _adapter_source(
        h3_latent, original_h3_av, ltx_video_latent, adapter_report_json)
    variant, values = _setup(model, guider, sampler, sigmas, setup_report_json)
    return {"schema": SCHEMA, "variant": variant,
            "h3_source": _input_identity(h3_latent),
            "ltx_video": _input_identity(ltx_video_latent),
            "adapter_report_sha256": hashlib.sha256(adapter_report_json.encode()).hexdigest(),
            "setup_report_sha256": hashlib.sha256(setup_report_json.encode()).hexdigest(),
            "model_object_id": id(model), "noise_object_id": id(noise),
            "guider_object_id": id(guider), "sampler_object_id": id(sampler),
            "sigmas": values, "video_shape": list(samples.shape),
            "source_audio_present": audio is not None,
            "duration_seconds": report["output_frames"] / report["output_fps"],
            "output_fps": report["output_fps"],
            "sampler_execution_proven": False, "portable_cache_reuse_authorized": False}


def bind_ltx_latent_stage(h3_latent, original_h3_av, ltx_video_latent, adapter_report_json,
                          model, noise, guider, sampler, sigmas, setup_report_json):
    boundary = _contract(h3_latent, original_h3_av, ltx_video_latent, adapter_report_json,
                         model, noise, guider, sampler, sigmas, setup_report_json)
    boundary["boundary_sha256"] = hashlib.sha256(canonical(boundary).encode()).hexdigest()
    summary = {"schema": SCHEMA, "status": "ready_to_sample", "variant": boundary["variant"],
               "source_audio_present": boundary["source_audio_present"],
               "sampled": False, "portable_cache_reuse_authorized": False}
    return noise, guider, sampler, sigmas, ltx_video_latent, boundary, canonical(summary)


def audit_ltx_latent_stage(boundary, h3_latent, original_h3_av, ltx_video_latent,
                           adapter_report_json, model, noise, guider, sampler, sigmas,
                           setup_report_json, candidate_latent):
    if not isinstance(boundary, dict) or boundary.get("schema") != SCHEMA:
        raise ValueError("Learned LTX candidate lacks a source-bound stage")
    unsigned = dict(boundary)
    digest = unsigned.pop("boundary_sha256", None)
    if digest != hashlib.sha256(canonical(unsigned).encode()).hexdigest():
        raise ValueError("Learned LTX boundary SHA256 differs")
    current = _contract(h3_latent, original_h3_av, ltx_video_latent, adapter_report_json,
                        model, noise, guider, sampler, sigmas, setup_report_json)
    if current != unsigned:
        raise ValueError("Learned LTX source, adapter report or sampler controls changed")
    if not isinstance(candidate_latent, dict) or "samples" not in candidate_latent:
        raise ValueError("Learned LTX candidate must be a LATENT")
    candidate = _tensor(candidate_latent["samples"], "Learned LTX candidate", 5)
    if list(candidate.shape) != boundary["video_shape"]:
        raise ValueError("Learned LTX candidate shape differs from adapter output")
    summary = {"schema": SCHEMA, "status": "source_bound_candidate_audited",
               "variant": boundary["variant"], "sampler_execution_proven": False,
               "portable_cache_reuse_authorized": False, "quality_acceptance": False,
               "audio_policy": "original_H3_AV_passed_through_unchanged_for_separate_decode"}
    return candidate_latent, original_h3_av, boundary["duration_seconds"], canonical(summary)


def decode_original_h3_audio(original_h3_av, adapter_report_json, audio_vae):
    # Reuse the native H3 audio-VAE decoder; never reinterpret H3 audio as LTX.
    report = _report(adapter_report_json, "Learned adapter report")
    if report.get("schema") != ADAPTER_SCHEMA:
        raise ValueError("Expected the learned adapter report")
    video, audio = extract_video(original_h3_av,
        expected_frames=timeline(report.get("source_frames"), report.get("source_fps"),
                                 report.get("temporal_policy"))["h3_latent_frames"],
        reference_prefix_latents=report.get("reference_prefix_latents"))
    if (audio is None or report.get("source_video_shape") != list(video.shape)
            or report.get("source_audio_shape") != list(audio.shape)
            or report.get("audio_duration_adjustment_required")):
        raise ValueError("Matching original H3 joint AV audio is required for this delivery")
    from comfy_extras.nodes_audio import vae_decode_audio
    decoded = vae_decode_audio(audio_vae, {"samples": audio})
    return decoded, canonical({"schema": SCHEMA, "status": "original_h3_audio_decoded",
                               "audio_policy": "H3_audio_VAE_only_no_LTX_audio_conversion"})
