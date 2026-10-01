"""Accepted-parent ports for an explicit FastH3 V2 continuation graph.

These ports do not plan, sample, accept, or append a segment.  In particular,
the HIGH prefix is installed only after the external learned lift/reconcile.
"""

import json

import torch
from comfy.nested_tensor import NestedTensor

from ..long_video import CONTEXT_FRAME_STEPS
from ..core import FPS, align_frame_count
from ..long_video_dual_model_runner import lock_high_video_prefix
from ..learned_latent_upscale_advanced import reconcile_two_pass_h3_latent
from . import continuation
from . import continuation_effects


def accepted_phase_context(contexts, phase):
    if type(contexts) is not continuation.ContinuationContexts or phase not in ("low", "high"):
        raise ValueError("FastH3 V2 requires verified LOW or HIGH accepted contexts")
    with continuation.chain_guard(contexts.source.root):
        descriptor = contexts.verify()
        request = contexts.source.binding["request"]
        context = contexts.low if phase == "low" else contexts.high
        width, height = ((request["low_width"], request["low_height"]) if phase == "low"
                         else (request["width"], request["height"]))
        report = {"schema": "t8.modular-sampling.fasth3-v2-accepted-port.v1",
                  "phase": phase, "accepted_source_sha256": contexts.source.sha256,
                  "contexts_sha256": descriptor["sha256"], "segment_index": request["segment_index"],
                  "context_frames": request["context_frames"], "width": width, "height": height,
                  "source": "accepted_rgb24_resize_vae" if phase == "low" else "accepted_completed_av_context",
                  "boundary": "Context port only; connect separate native Long Video/Relay conditioning and V2 stage."}
        contexts.verify()
        return (context, request["segment_index"], request["context_frames"],
                width, height, json.dumps(report, ensure_ascii=False, sort_keys=True))


def plan_accepted_window(contexts, total_accepted_frames=192, render_policy="compact_remainder"):
    if type(contexts) is not continuation.ContinuationContexts:
        raise ValueError("FastH3 V2 window requires verified accepted contexts")
    if type(total_accepted_frames) is not int or total_accepted_frames <= 0:
        raise ValueError("Final accepted frame count must be positive")
    if render_policy not in ("compact_remainder", "old_fixed_124"):
        raise ValueError("Unknown accepted V2 render-window policy")
    with continuation.chain_guard(contexts.source.root):
        descriptor = contexts.verify()
        binding = contexts.source.binding
        request = binding["request"]
        start = binding["accepted"]["timeline_end_frame"]
        new_frames = total_accepted_frames - start
        compact_length = request["context_frames"] + new_frames
        if new_frames <= 0 or align_frame_count(compact_length) != compact_length:
            raise ValueError("Accepted remainder must produce a native 17n+5 render window")
        if render_policy == "old_fixed_124":
            if (start != 124 or request["context_frames"] != 22
                    or total_accepted_frames != 192 or new_frames != 68):
                raise ValueError("Old fixed 124-frame window requires the exact accepted 8s V2 contract")
            length = 124
        else:
            length = compact_length
        report = {"schema": "t8.modular-sampling.fasth3-v2-accepted-window.v1",
                  "accepted_source_sha256": contexts.source.sha256,
                  "contexts_sha256": descriptor["sha256"], "accepted_start_frame": start,
                  "context_frames": request["context_frames"], "new_frames": new_frames,
                  "render_frames": length, "render_policy": render_policy,
                  "discarded_suffix_frames": length - compact_length,
                  "total_accepted_frames": total_accepted_frames,
                  "boundary": "Geometry only; this node does not sample, trim, accept or assemble"}
        contexts.verify()
        return length, new_frames, json.dumps(report, ensure_ascii=False, sort_keys=True)


def accepted_delivery_window(contexts, total_accepted_frames=192, render_policy="compact_remainder"):
    """Expose only verified parent and AV trim coordinates for the final remainder.

    This port does not certify today's edited model/LoRA recipe as the parent's
    execution contract. Candidate Save must receive that contract separately.
    """
    length, new_frames, _ = plan_accepted_window(contexts, total_accepted_frames, render_policy)
    with continuation.chain_guard(contexts.source.root):
        descriptor = contexts.verify()
        binding = contexts.source.binding
        request = binding["request"]
        start = binding["accepted"]["timeline_end_frame"]
        overlap = request["context_frames"]
        if length < overlap + new_frames or start + new_frames != total_accepted_frames:
            raise RuntimeError("Accepted delivery window changed after planning")
        report = {"schema": "t8.modular-sampling.fasth3-v2-delivery-window.v1",
                  "accepted_source_sha256": contexts.source.sha256,
                  "contexts_sha256": descriptor["sha256"],
                  "render_frames": length, "render_policy": render_policy,
                  "discarded_suffix_frames": length - overlap - new_frames,
                  "trim_start_frame": overlap,
                  "new_frames": new_frames, "timeline_start_frame": start,
                  "timeline_end_frame": total_accepted_frames,
                  "save_context": False,
                  "boundary": "Coordinates and parent identity only; no trim, candidate save, acceptance or job-contract certification"}
        return (request["chain_id"], request["segment_index"],
                request["parent_candidate_id"], request["parent_revision"],
                start / FPS, overlap / FPS, new_frames / FPS,
                False, json.dumps(report, ensure_ascii=False, sort_keys=True))


def lock_accepted_high_prefix(contexts, reconciled_av, mode="high_native_mask_ramp_exp"):
    if type(contexts) is not continuation.ContinuationContexts:
        raise ValueError("FastH3 V2 HIGH prefix requires verified accepted contexts")
    if mode not in ("high_native_mask_exp", "high_native_mask_ramp_exp"):
        raise ValueError("Unknown FastH3 V2 HIGH prefix mode")
    with continuation.chain_guard(contexts.source.root):
        descriptor = contexts.verify()
        request = contexts.source.binding["request"]
        if request["context_frames"] not in CONTEXT_FRAME_STEPS:
            raise ValueError("Invalid accepted HIGH context length")
        result, legacy_report = lock_high_video_prefix(
            reconciled_av, contexts.high, chain_id=request["chain_id"],
            segment_index=request["segment_index"], context_frames=request["context_frames"], mode=mode)
        if result["samples"].unbind()[1] is not reconciled_av["samples"].unbind()[1]:
            raise RuntimeError("HIGH prefix must preserve the reconciled audio tensor")
        contexts.verify()
        report = {"schema": "t8.modular-sampling.fasth3-v2-accepted-high-prefix.v1",
                  "accepted_source_sha256": contexts.source.sha256,
                  "contexts_sha256": descriptor["sha256"], "segment_index": request["segment_index"],
                  "after": "external_learned_upscale_and_reconcile", **legacy_report}
        return result, json.dumps(report, ensure_ascii=False, sort_keys=True)


def project_accepted_relay(contexts, global_plan, length, accepted_end_frame=0):
    """Return the native Long Video Relay plan, projected from the accepted frame clock."""
    if type(contexts) is not continuation.ContinuationContexts:
        raise ValueError("FastH3 V2 Relay requires verified accepted contexts")
    if type(accepted_end_frame) is not int or accepted_end_frame < 0:
        raise ValueError("Accepted Relay end frame must be a nonnegative integer")
    if accepted_end_frame:
        fixed_length, _, _ = plan_accepted_window(contexts, accepted_end_frame, "old_fixed_124")
        if length != fixed_length:
            raise ValueError("Accepted Relay end frame requires the old fixed render window")
    projected = continuation_effects.project_relay(
        contexts, global_plan, length,
        accepted_end_frame=accepted_end_frame or None)
    descriptor = projected.verify()
    plan = projected.projected
    report = {"schema": "t8.modular-sampling.fasth3-v2-accepted-relay.v1",
              "accepted_source_sha256": contexts.source.sha256,
              "contexts_sha256": descriptor["contexts_sha256"],
              "global_plan_hash": plan["global_plan_hash"], "plan_hash": plan["plan_hash"],
              "length": length, "projection": plan["long_video_projection"]}
    return plan, plan["compiled_prompt"], json.dumps(report, ensure_ascii=False, sort_keys=True)


def reconcile_accepted_high(contexts, learned_latent, highres_template, positive):
    """Exact old 4+4 legacy-policy joint-audio handoff, without a hidden HIGH pass."""
    if type(contexts) is not continuation.ContinuationContexts:
        raise ValueError("FastH3 V2 reconcile requires verified accepted contexts")
    with continuation.chain_guard(contexts.source.root):
        descriptor = contexts.verify()
        prepared, selected_positive, report_json = reconcile_two_pass_h3_latent(
            learned_latent, highres_template, positive, "first_pass",
            second_pass_audio_source="legacy_policy", second_pass_audio_strength=0.0)
        if "noise_mask" in highres_template:
            video, coarse_audio = prepared["samples"].unbind()
            template_audio = highres_template["samples"].unbind()[1].to(coarse_audio)
            mask = highres_template["noise_mask"].unbind()[1].to(coarse_audio.device)
            if (not torch.isfinite(mask).all() or torch.any(mask < 0) or torch.any(mask > 1)
                    or not torch.isfinite(template_audio).all()):
                raise ValueError("Continuation audio template/mask is invalid")
            audio = torch.where(mask == 0, template_audio, coarse_audio)
            prepared["samples"] = NestedTensor((video, audio))
        report = json.loads(report_json)
        if "noise_mask" in highres_template:
            report["dual_audio_handoff"] = "coarse_unlocked_template_locked_v1"
        report.update(schema="t8.modular-sampling.fasth3-v2-accepted-reconcile.v1",
                      accepted_source_sha256=contexts.source.sha256,
                      contexts_sha256=descriptor["sha256"],
                      boundary="Original legacy-policy audio selection only; HIGH prefix is a separate post-reconcile node")
        contexts.verify()
        return prepared, selected_positive, json.dumps(report, ensure_ascii=False, sort_keys=True)
