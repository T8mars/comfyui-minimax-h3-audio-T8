"""Source-bound guard for the already external H3 RGB -> LTX refiner sampler.

This is deliberately not a second sampler or a portable completion receipt.
The H3 audio object remains outside LTX and the old Sol workflows are untouched.
"""
import hashlib
import json
import math

import comfy.samplers
import torch

from .. import sol_engine_h3_super_advanced as sol
from .results import _input_identity, canonical

SCHEMA = "t8.modular-sampling.ltx-rgb-stage.v1"
PREP_PIPELINE = "nvidia_h3_super_acceleration_stage2_handoff"
OFFICIAL_PIPELINE = "nvidia_h3_super_acceleration_ltx25_stage2"
IDENTITY_PIPELINE = "ltx25_stage2_identity_preserve_low_sigma_exp"


def _sha(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _report(value, label):
    try:
        result = json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(result, dict):
        raise ValueError(f"{label} must be a JSON object")
    return result


def _tensor(value, label, ndim):
    if (not isinstance(value, torch.Tensor) or value.ndim != ndim or
            not value.is_floating_point() or value.is_meta or
            not bool(torch.isfinite(value).all())):
        raise ValueError(f"{label} must be a finite floating tensor with {ndim} dimensions")
    return value


def _contract(source_frames, source_audio, prepared_frames, prep_report_json,
              ltx_latent, model, noise, guider, sampler, sigmas, setup_report_json):
    source = _tensor(source_frames, "H3 source frames", 4)
    prepared = _tensor(prepared_frames, "LTX encoder frames", 4)
    if source.shape[-1] != 3 or prepared.shape[-1] != 3:
        raise ValueError("RGB handoff requires three-channel source and prepared frames")
    prep = _report(prep_report_json, "H3 to LTX preparation report")
    if (prep.get("status") != "prepared" or prep.get("pipeline") != PREP_PIPELINE or
            prep.get("audio_policy") != "bypass_stage2_and_preserve_original_h3_audio_object"):
        raise ValueError("H3 to LTX preparation report is not the RGB/audio-bypass contract")
    source_shape = {"width": int(source.shape[2]), "height": int(source.shape[1]),
                    "frames": int(source.shape[0])}
    if prep.get("source") != source_shape:
        raise ValueError("H3 source frames differ from the preparation report")
    target = prep.get("target")
    encoder = prep.get("ltx_encoder_input")
    if not isinstance(target, dict) or not isinstance(encoder, dict):
        raise ValueError("H3 to LTX preparation geometry is missing")
    frames = int(prepared.shape[0])
    height, width = int(prepared.shape[1]), int(prepared.shape[2])
    if (encoder != {"width": width, "height": height, "frames": frames,
                    "resize": "aspect_preserving_center_crop"} or
            target != {"width": 2 * width, "height": 2 * height, "frames": frames} or
            width % 16 or height % 16 or frames < 1 or frames > source.shape[0] or
            prep.get("dropped_tail_frames") != source.shape[0] - frames):
        raise ValueError("LTX prepared frames differ from the reported RGB geometry")
    policy = prep.get("frame_policy")
    if policy == "trim_to_8n_plus_1":
        if frames != sol.ltx_8n_plus_1_frame_count(source.shape[0]):
            raise ValueError("LTX prepared frames differ from the 8n+1 trim policy")
    elif policy != "preserve_all_exp" or frames != source.shape[0]:
        raise ValueError("Unsupported H3 to LTX frame policy")
    if (frames - 1) % 8:
        raise ValueError("The source-bound RGB stage needs an exact LTX 8n+1 grid")
    fps = prep.get("fps")
    if (isinstance(fps, bool) or not isinstance(fps, (int, float)) or
            not math.isfinite(fps) or fps <= 0 or
            not math.isclose(prep.get("output_duration_seconds", -1), frames / fps,
                             rel_tol=1e-8, abs_tol=1e-8)):
        raise ValueError("LTX preparation duration or frame rate differs")
    if not isinstance(ltx_latent, dict) or "samples" not in ltx_latent:
        raise ValueError("LTX lifted source must be a LATENT with samples")
    samples = _tensor(ltx_latent["samples"], "LTX lifted latent", 5)
    expected = (1, 128, (frames - 1) // 8 + 1, height // 16, width // 16)
    if tuple(samples.shape) != expected:
        raise ValueError("LTX lifted latent differs from target canvas and frame grid")
    if source_audio is not None and (not isinstance(source_audio, dict) or
                                     "waveform" not in source_audio or
                                     "sample_rate" not in source_audio):
        raise ValueError("Connected H3 AUDIO must stay on the bypass wire")
    setup = _report(setup_report_json, "LTX Setup report")
    profile = setup.get("pipeline")
    if profile == OFFICIAL_PIPELINE:
        expected_sigmas = sol.OFFICIAL_STAGE2_SIGMAS
        variant = "official_stage2"
    elif profile == IDENTITY_PIPELINE:
        expected_sigmas = setup.get("stage2_sigmas")
        variant = "identity_preserve"
    elif setup.get("status") in {"disabled_passthrough", "unsupported_model_passthrough"}:
        # The old official disabled/unsupported reports have no pipeline.
        expected_sigmas = setup.get("stage2_sigmas", sol.OFFICIAL_STAGE2_SIGMAS)
        variant = "official_stage2_passthrough"
    else:
        raise ValueError("LTX Setup report is not a supported refiner profile")
    if setup.get("status") not in {"configured", "disabled_passthrough", "unsupported_model_passthrough"}:
        raise ValueError("LTX Setup did not configure or explicitly bypass the refiner")
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) != 4:
        raise ValueError("LTX refiner needs three explicit sigma updates")
    values = [float(value) for value in sigmas.detach().cpu().tolist()]
    if (not all(math.isfinite(value) for value in values) or
            any(left <= right for left, right in zip(values, values[1:])) or
            values[-1] != 0 or not isinstance(expected_sigmas, (list, tuple)) or
            len(expected_sigmas) != 4 or
            any(not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-7)
                for left, right in zip(values, expected_sigmas))):
        raise ValueError("LTX SIGMAS differ from the connected Setup profile")
    if not isinstance(guider, comfy.samplers.CFGGuider) or guider.model_patcher is not model:
        raise ValueError("LTX guider is not bound to the connected refiner MODEL")
    if not isinstance(sampler, comfy.samplers.KSAMPLER):
        raise ValueError("LTX stage requires an externally connected Core sampler")
    return {"schema": SCHEMA, "variant": variant,
            "source_frames": _input_identity(source), "source_audio": _input_identity(source_audio),
            "prepared_frames": _input_identity(prepared),
            "prep_report_sha256": hashlib.sha256(prep_report_json.encode("utf-8")).hexdigest(),
            "lifted_latent": _input_identity(ltx_latent), "target": target, "fps": fps,
            "setup_report_sha256": hashlib.sha256(setup_report_json.encode("utf-8")).hexdigest(),
            "model_object_id": id(model), "noise_object_id": id(noise),
            "guider_object_id": id(guider), "sampler_object_id": id(sampler),
            "sigmas": values, "sampling_completion_verified": False,
            "portable_cache_reuse_authorized": False}


def bind_ltx_rgb_stage(source_frames, source_audio, prepared_frames, prep_report_json,
                       ltx_latent, model, noise, guider, sampler, sigmas, setup_report_json):
    boundary = _contract(source_frames, source_audio, prepared_frames, prep_report_json,
                         ltx_latent, model, noise, guider, sampler, sigmas, setup_report_json)
    boundary["boundary_sha256"] = _sha(boundary)
    report = {"schema": SCHEMA, "status": "ready_to_sample", "variant": boundary["variant"],
              "sampled": False, "portable_cache_reuse_authorized": False,
              "audio_policy": "original_H3_audio_bypasses_LTX_refinement_if_present",
              "boundary_sha256": boundary["boundary_sha256"]}
    return noise, guider, sampler, sigmas, ltx_latent, boundary, canonical(report)


def audit_ltx_rgb_stage(boundary, source_frames, source_audio, prepared_frames,
                        prep_report_json, ltx_latent, model, noise, guider, sampler,
                        sigmas, setup_report_json, candidate_latent):
    if not isinstance(boundary, dict) or boundary.get("schema") != SCHEMA:
        raise ValueError("LTX RGB candidate lacks a source-bound stage")
    unsigned = dict(boundary)
    digest = unsigned.pop("boundary_sha256", None)
    if digest != _sha(unsigned):
        raise ValueError("LTX RGB stage boundary SHA-256 differs")
    current = _contract(source_frames, source_audio, prepared_frames, prep_report_json,
                        ltx_latent, model, noise, guider, sampler, sigmas, setup_report_json)
    if current != unsigned:
        raise ValueError("LTX RGB source, Setup or sampling controls changed after binding")
    if not isinstance(candidate_latent, dict) or "samples" not in candidate_latent:
        raise ValueError("LTX RGB candidate must be a LATENT with samples")
    samples = _tensor(candidate_latent["samples"], "LTX candidate", 5)
    if tuple(samples.shape) != tuple(ltx_latent["samples"].shape):
        raise ValueError("LTX candidate shape differs from the lifted source")
    report = {"schema": SCHEMA, "status": "source_bound_candidate_audited",
              "variant": boundary["variant"], "boundary_sha256": digest,
              "sampler_execution_proven": False, "portable_cache_reuse_authorized": False,
              "quality_acceptance": False,
              "audio_policy": "original_H3_audio_object_passed_through_unchanged_if_present",
              "boundary": "The external sampler, TAEHV decode and OutputTrim remain distinct. "
                          "LTX effects, image quality and audio sync need separate validation."}
    return candidate_latent, source_audio, canonical(report)
