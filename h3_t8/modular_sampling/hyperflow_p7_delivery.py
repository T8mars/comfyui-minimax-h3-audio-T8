"""Explicit P7 candidate delivery on top of the unchanged long-video manifest.

The old candidate writer stores only the completed HIGH context.  A P7
continuation also needs the LOW-video/completed-HIGH-audio context and a
recipe audit.  These are committed as candidate sidecars before this route
will offer an Accept action.  An interrupted write is an unacceptably
incomplete candidate, never an implicitly reusable segment.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from comfy.nested_tensor import NestedTensor
import torch

from .. import long_video_delivery as delivery
from ..audio_ops import decode_av_latent, trim_av_output
from ..long_video_dual_color import correct_dual_segment_color
from ..hyperflow_long_video_exp.runner import RECIPE
from ..long_video_in_node_loop_effects_advanced import _load_effects_audit, _write_effects_audit
from ..sampling import nested_av_parts
from . import hyperflow_p7 as p7
from .continuation import chain_guard
from .results import canonical

AUDIT_DELIVERY_SCHEMA = "t8.modular-sampling.hyperflow-p7-delivery.v1"


def _source(high):
    if type(high) is not p7.P7HighResult:
        raise ValueError("P7 delivery requires an authenticated completed HIGH result")
    job_sha256 = high.verify()["sha256"]
    contexts = high.handoff.phase.contexts
    low = high.handoff.lift.low
    request = (contexts.parent.binding["request"] if type(contexts) is p7.P7Contexts
               else contexts.request)
    if low.phase.contexts is not contexts:
        raise ValueError("P7 LOW and HIGH delivery contexts differ")
    low.verify()
    details = high.handoff.phase.result[-1]
    low_details = low.phase.result[-1]
    length = details["frame_count"]
    if type(length) is not int or length != low_details["frame_count"]:
        raise ValueError("P7 completed LOW/HIGH frame counts differ")
    if length <= request["context_frames"]:
        raise ValueError("P7 delivery contains no new frames")
    return contexts, request, low, job_sha256, length


def _parent_identity(contexts, request):
    if type(contexts) is p7.P7InitialContexts:
        contexts.verify()
        return "", 0, 0, None
    binding = contexts.parent.revalidate()
    root = contexts.parent.root
    manifest_path = delivery._resolve_inside(root, root / delivery.MANIFEST_NAME)
    manifest = delivery._validate_manifest(json.loads(manifest_path.read_bytes()), request["chain_id"])
    if (manifest["revision"] != request["parent_revision"]
            or manifest["segments"][request["segment_index"] - 1]["candidate_id"]
            != request["parent_candidate_id"]
            or delivery._sha256_file(manifest_path) != binding["manifest_sha256"]):
        raise ValueError("P7 delivery accepted predecessor changed")
    entry = manifest["segments"][request["segment_index"] - 1]
    return entry["candidate_id"], manifest["revision"], entry["timeline_end_frame"], entry


def _source_under_lease(contexts, request):
    """Recheck the source while the caller owns loop.lock; never reenter it."""
    if type(contexts) is p7.P7InitialContexts:
        current = p7._initial_binding(contexts.root, request)
        expected = contexts.binding_json
    elif type(contexts) is p7.P7Contexts:
        current, *_ = p7._capture(contexts.parent.root, request)
        expected = contexts.parent.binding_json
    else:
        raise ValueError("P7 delivery source type changed")
    if canonical(current) != expected:
        raise ValueError("P7 delivery source changed while saving candidate")


def verify_candidate(candidate_json_path, job_sha256):
    """Read all three candidate artifacts before a P7-specific Accept."""
    p7._digest(job_sha256, "P7 delivery job")
    candidate, root, movie = delivery._load_candidate(str(candidate_json_path))
    path = delivery._resolve_inside(root, candidate_json_path)
    audit = _load_effects_audit(path, contract_sha256=job_sha256,
        segment_index=candidate["index"], candidate_id=candidate["candidate_id"])
    plan = audit.get("sampling_plan", {})
    dual = plan.get("dual_model", {})
    if (audit.get("p7_delivery_schema") != AUDIT_DELIVERY_SCHEMA
            or plan.get("mode") != RECIPE
            or plan.get("hyperflow", {}).get("intervals") != [[0, 4], [4, 8]]
            or dual.get("audio_policy") != "joint_native_av_not_frozen_low_audio"
            or any(re.fullmatch(r"[0-9a-f]{64}", str(dual.get(key, ""))) is None for key in
                   ("first_pass_receipt_sha256", "second_pass_receipt_sha256"))):
        raise ValueError("P7 candidate audit does not describe the completed 4+4 AV recipe")
    if candidate["is_final_segment"]:
        if dual.get("low_context") is not None:
            raise ValueError("Final P7 candidate must not claim a continuation LOW context")
    else:
        record = dual.get("low_context")
        if (not isinstance(record, dict)
                or record.get("audio_source") != "completed_second_pass_output"
                or record.get("audio_policy_version") != 3):
            raise ValueError("P7 continuation LOW context is missing or has the wrong audio policy")
        low_path = delivery._resolve_inside(root, record.get("path", ""))
        if (low_path.parent != path.parent or low_path.name != "low.context.safetensors"
                or not low_path.is_file()
                or delivery._sha256_file(low_path) != record.get("sha256")):
            raise ValueError("P7 continuation LOW context path or SHA changed")
        low, _ = delivery._load_accepted_context_file(low_path, candidate["chain_id"],
            candidate["index"], candidate["index"] + 1)
        high_path = delivery._resolve_inside(root, candidate["context_path"])
        high, _ = delivery._load_accepted_context_file(high_path, candidate["chain_id"],
            candidate["index"], candidate["index"] + 1)
        if (low["metadata"]["sampling_summary"] != job_sha256
                or high["metadata"]["sampling_summary"] != candidate["sampling_summary"]
                or low["metadata"]["source_total_frames"] != high["metadata"]["source_total_frames"]
                or low["audio_tail"].shape != high["audio_tail"].shape
                or not torch.isfinite(low["video_tail"]).all()
                or not torch.isfinite(low["audio_tail"]).all()
                or not torch.isfinite(high["video_tail"]).all()
                or not torch.isfinite(high["audio_tail"]).all()
                or not torch.equal(low["audio_tail"], high["audio_tail"])):
            raise ValueError("P7 LOW/HIGH candidate contexts have different completed audio or identity")
    return candidate, audit, str(movie)


def save_candidate(high, video_vae, audio_vae, *, candidate_id="", is_final_segment=False,
                   final_frame_count=0, model_id="h3_p7_modular_exp", seed=0,
                   color_match=True, bit_depth=8, crf=18):
    """Decode, trim and persist one exact completed P7 segment, without accepting it."""
    contexts, request, low, job_sha256, length = _source(high)
    for label, latent in (("LOW", low.sampled.denoised_output), ("HIGH", high.sampled.output)):
        video_latent, audio_latent = nested_av_parts(latent)
        if not torch.isfinite(video_latent).all() or not torch.isfinite(audio_latent).all():
            raise ValueError(f"P7 {label} completed AV contains nonfinite values")
    if type(is_final_segment) is not bool or type(color_match) is not bool:
        raise ValueError("P7 final/color options must be booleans")
    if type(final_frame_count) is not int or final_frame_count < 0:
        raise ValueError("P7 final_frame_count must be zero or a positive integer")
    if type(candidate_id) is not str or (candidate_id and
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", candidate_id) is None):
        raise ValueError("P7 candidate_id must be empty or an exact safe token")
    if not is_final_segment and final_frame_count:
        raise ValueError("A continuable P7 segment cannot hide sampled tail frames")
    available = length - request["context_frames"]
    accepted_frames = final_frame_count or available
    if accepted_frames > available:
        raise ValueError("P7 requested final frames exceed sampled continuation")
    parent_id, parent_revision, start_frame, parent_entry = _parent_identity(contexts, request)
    chain = request["chain_id"]
    root = delivery.long_video_chain_root(chain)
    token = delivery._safe_token(candidate_id, fallback_prefix=f"p7seg{request['segment_index']:05d}")
    if candidate_id and token != candidate_id:
        raise ValueError("P7 candidate_id must already be normalized")
    candidate_dir = root / "candidates" / f"segment_{request['segment_index']:05d}" / token
    if candidate_dir.exists():
        raise FileExistsError("P7 candidate directory already exists; use a new candidate_id")
    if parent_entry is None:
        if type(model_id) is not str or not model_id.strip():
            raise ValueError("P7 first segment needs a stable chain model_id")
        chain_model_id = model_id.strip()
        summary = "P7 modular partial4+fresh4 native AV"
    else:
        chain_model_id = parent_entry["model_id"]
        summary = parent_entry["sampling_summary"]
    frames, generated_audio, _, _ = decode_av_latent(high.sampled.output, video_vae, audio_vae)
    if frames.shape[0] != length or not torch.isfinite(frames).all():
        raise ValueError("P7 decoded HIGH frames differ from the prepared render length")
    mux_audio = high.handoff.phase.result[2]
    audio = mux_audio if mux_audio is not None else generated_audio
    trimmed_frames, trimmed_audio, trim_json = trim_av_output(frames,
        request["context_frames"] / 24, accepted_frames / 24, audio, 24.0)
    if trimmed_audio is None:
        raise ValueError("P7 completed segment has no audio delivery")
    if not torch.isfinite(trimmed_audio["waveform"]).all():
        raise ValueError("P7 completed segment audio is nonfinite")
    colored_frames, color_report = correct_dual_segment_color(trimmed_frames, root, chain,
        request["segment_index"], parent_id, color_match, "bounded_spatial_v2")
    if not torch.isfinite(colored_frames).all():
        raise ValueError("P7 color-matched segment frames are nonfinite")
    _parent_identity(contexts, request)
    low_receipt = low.sampled.verify()
    high_receipt = high.sampled.verify()
    root.mkdir(parents=True, exist_ok=True)
    with chain_guard(root):
        _source_under_lease(contexts, request)
        # Reserve the entire namespace, including sidecars unknown to the
        # generic writer. Failure leaves a non-acceptable candidate for
        # diagnosis; another attempt must use a new ID.
        candidate_dir.mkdir(parents=True, exist_ok=False)
        candidate_json, movie, save_report = delivery.save_long_video_candidate(
            colored_frames, trimmed_audio, high.sampled.output, chain, request["segment_index"],
            start_frame / 24, not is_final_segment, parent_id, parent_revision, token,
            chain_model_id, summary, high.handoff.phase.result[3], seed, 24, bit_depth, crf)
        context_record = None
        if not is_final_segment:
            low_video, low_audio = nested_av_parts(low.sampled.denoised_output)
            _, completed_audio = nested_av_parts(high.sampled.output)
            if (low_audio.shape != completed_audio.shape or low_audio.dtype != completed_audio.dtype
                    or not torch.isfinite(low_video).all() or not torch.isfinite(completed_audio).all()):
                raise ValueError("P7 completed HIGH audio cannot populate LOW continuation context")
            continuation = {**low.sampled.denoised_output,
                            "samples": NestedTensor((low_video, completed_audio))}
            low_path = delivery._resolve_inside(root, Path(candidate_json).parent / "low.context.safetensors")
            context_record = delivery._write_context_candidate(continuation, low_path, chain,
                request["segment_index"], chain_model_id, job_sha256)
            context_record.update(path=low_path.relative_to(root).as_posix(),
                audio_source="completed_second_pass_output", audio_policy_version=3)
        _write_effects_audit(candidate_json, {
            "p7_delivery_schema": AUDIT_DELIVERY_SCHEMA,
            "contract_sha256": job_sha256, "segment_index": request["segment_index"],
            "candidate_id": token,
            "sampling_plan": {"mode": RECIPE,
                "hyperflow": {"intervals": [[0, 4], [4, 8]], "nfe": 8,
                              "human_quality": "unverified"},
                "dual_model": {"low_context": context_record,
                               "audio_policy": "joint_native_av_not_frozen_low_audio",
                               "first_pass_receipt_sha256": low_receipt["receipt_sha256"],
                               "second_pass_receipt_sha256": high_receipt["receipt_sha256"],
                               "external_effects": {"low": low_receipt.get("execution", {}),
                                                    "high": high_receipt.get("execution", {})}}},
            "trim": json.loads(trim_json), "color_match": color_report,
            "candidate_save": json.loads(save_report),
        })
        verify_candidate(candidate_json, job_sha256)
        _source_under_lease(contexts, request)
        if (low.sampled.verify()["receipt_sha256"] != low_receipt["receipt_sha256"]
                or high.sampled.verify()["receipt_sha256"] != high_receipt["receipt_sha256"]):
            raise ValueError("P7 completed sampling results changed during candidate save")
    if high.verify()["sha256"] != job_sha256:
        raise ValueError("P7 source changed immediately after candidate save")
    report = {"schema": AUDIT_DELIVERY_SCHEMA,
              "candidate_json_path": candidate_json, "candidate_video_path": movie,
              "job_sha256": job_sha256, "accepted": False, "segment_index": request["segment_index"],
              "timeline_start_frame": start_frame, "frame_count": accepted_frames,
              "low_context_saved": context_record is not None}
    return candidate_json, movie, job_sha256, canonical(report)


def accept_candidate(candidate_json_path, job_sha256, *, accept=False):
    preliminary, root, _ = delivery._load_candidate(str(candidate_json_path))
    with chain_guard(root):
        candidate, audit, movie = verify_candidate(candidate_json_path, job_sha256)
        if candidate != preliminary:
            raise ValueError("P7 candidate changed while acquiring the chain lease")
        if not accept:
            return movie, False, "", "", 0, canonical({"candidate": candidate,
                "audit_sha256": audit["audit_sha256"], "accepted": False})
        accepted_video, accepted, manifest_path, old_report = delivery.accept_long_video_candidate(
            str(candidate_json_path), True, "reject_existing", True)
        if not accepted:
            raise RuntimeError("P7 candidate was not accepted")
        manifest, _ = delivery.load_delivery_manifest(candidate["chain_id"])
        entry = manifest["segments"][candidate["index"]]
        if (entry["candidate_id"] != candidate["candidate_id"]
                or entry["video_sha256"] != candidate["video_sha256"]):
            raise ValueError("P7 accepted manifest changed after commit")
        report = {"schema": "t8.modular-sampling.hyperflow-p7-accept.v1",
                  "accepted": True, "candidate_id": candidate["candidate_id"],
                  "manifest_revision": manifest["revision"], "job_sha256": job_sha256,
                  "legacy_delivery": json.loads(old_report)}
        return (accepted_video, True, manifest_path, candidate["candidate_id"],
                manifest["revision"], canonical(report))
