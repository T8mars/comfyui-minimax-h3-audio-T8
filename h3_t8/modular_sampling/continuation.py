"""Accepted-parent contexts and independently editable native continuation stages.

No whole-segment executor, auto-accept, append or compose. Each operation holds
the existing chain's OS loop lease and verifies the selected immediate parent.
LOW uses accepted RGB motion guides, not a resized HIGH known-prefix template.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .. import core, long_video, long_video_delivery as delivery
from .. import progressive_continuation as legacy_source
from ..long_video_dual_model_runner import lock_high_video_prefix
from ..long_video_in_node_loop_advanced import LOOP_LOCK_NAME
from . import progressive as stages
from . import progressive_high_result as high_results

KEY = "t8_modular_progressive_continuation_v1"
CONTEXT_SCHEMA = "t8.modular-sampling.continuation-contexts.v1"
PHASE_SCHEMA = "t8.modular-sampling.continuation-phase.v1"
OPTIONS = {"task_type", "audio_mode", "audio_denoise_strength", "add_source_as_reference",
    "prompt_primary_audio_ordinal", "strict_prompt_tags", "ref_image_size", "reference_video_policy",
    "drive_audio", "final_audio", "first_frame", "last_frame", "ref_images", "ref_videos",
    "ref_video_audios", "ref_audios", "first_frame_reuse", "persistent_identity_image",
    "persistent_identity_strategy", "persistent_identity_interval", "semantic_bridge"}
PROTECTED = (long_video.LONG_VIDEO_CONDITIONING_KEY, "minimax_keyframes", "minimax_frame_count", "minimax_refs")


def implementation():
    from .. import conditioning, native_masked_context_advanced, long_video_dual_picture_context
    from .. import long_video_dual_model_runner
    modules = (core, long_video, delivery, legacy_source, conditioning, native_masked_context_advanced,
               long_video_dual_picture_context, long_video_dual_model_runner)
    paths = {Path(module.__file__).resolve() for module in modules} | {Path(__file__).resolve()}
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


@contextmanager
def chain_guard(root):
    """Same OS lease as legacy runners; never create a missing chain or unlock a foreign lease."""
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Select an existing accepted chain directory")
    handle = delivery._open_advisory_lock(root / LOOP_LOCK_NAME)
    acquired = False
    try:
        if not delivery._try_advisory_lock(handle):
            raise RuntimeError("Another runner owns this continuation chain")
        acquired = True
        yield
    finally:
        try:
            if acquired:
                delivery._release_advisory_lock(handle)
        finally:
            handle.close()


def capture_source(root, **request):
    with chain_guard(root):
        return legacy_source.capture_continuation_source(root, **request)


@dataclass(frozen=True)
class ContinuationContexts:
    source: legacy_source.ProgressiveContinuationSource
    low: dict
    high: dict
    preparation: dict
    contract_json: str

    def descriptor(self):
        if type(self.source) is not legacy_source.ProgressiveContinuationSource:
            raise ValueError("A verified accepted immediate parent is required")
        binding = self.source.revalidate()
        request = binding["request"]
        for phase, context in (("low", self.low), ("high", self.high)):
            width, height = ((request["low_width"], request["low_height"]) if phase == "low"
                             else (request["width"], request["height"]))
            long_video._validate_context(context, request["segment_index"], request["context_frames"], width, height)
            for field in ("video_tail", "audio_tail"):
                stages.masks._finite(context[field], phase + " accepted " + field)
        if self.low["audio_tail"] is not self.high["audio_tail"]:
            raise ValueError("Continuation preparation must retain the same completed audio object")
        return {"schema": CONTEXT_SCHEMA, "source": binding, "source_sha256": self.source.sha256,
            "low": stages.snapshot(self.low), "high": stages.snapshot(self.high),
            "preparation": self.preparation, "implementation": implementation()}

    def verify(self):
        expected = json.loads(self.contract_json)
        current = self.descriptor()
        current["sha256"] = stages.sha(current)
        if current != expected:
            raise ValueError("Accepted continuation contexts changed")
        return expected


def prepare_contexts(source, video_vae):
    if type(source) is not legacy_source.ProgressiveContinuationSource:
        raise ValueError("Capture the selected accepted parent first")
    with chain_guard(source.root):
        low, high, report = source.prepare_contexts(video_vae)
        provisional = ContinuationContexts(source, low, high, report, "")
        descriptor = provisional.descriptor()
        descriptor["sha256"] = stages.sha(descriptor)
        result = ContinuationContexts(source, low, high, report, stages.canonical(descriptor))
        result.verify()
        return result


def _phase(value):
    if value not in ("low", "high"):
        raise ValueError("Select LOW or HIGH continuation phase")
    return value


@dataclass(frozen=True)
class ContinuationPhase:
    contexts: ContinuationContexts
    phase: str
    result: tuple
    contract_json: str

    def descriptor(self):
        _phase(self.phase)
        contexts = self.contexts.verify()
        if type(self.result) is not tuple or len(self.result) != 7:
            raise ValueError("Expected complete native phase conditioning details")
        stages.masks._av_parts(self.result[1]["samples"], "continuation " + self.phase)
        return {"schema": PHASE_SCHEMA, "phase": self.phase, "contexts_sha256": contexts["sha256"],
                "result": stages.snapshot(self.result)}

    def verify(self):
        expected = json.loads(self.contract_json)
        current = self.descriptor()
        current["sha256"] = stages.sha(current)
        if current != expected:
            raise ValueError("Prepared continuation phase changed")
        return expected


def prepare_phase(contexts, phase, *, clip, video_vae, audio_vae, prompt, length,
                  context_audio="video_and_audio", **options):
    _phase(phase)
    if type(contexts) is not ContinuationContexts:
        raise ValueError("Connect authenticated continuation contexts")
    if set(options) - OPTIONS:
        raise ValueError("Phase options cannot override accepted source/geometry")
    if type(length) is not int or length <= contexts.source.binding["request"]["context_frames"]:
        raise ValueError("Continuation must leave newly generated frames")
    with chain_guard(contexts.source.root):
        contexts.verify()
        request = contexts.source.binding["request"]
        width, height = ((request["low_width"], request["low_height"]) if phase == "low"
                         else (request["width"], request["height"]))
        result = list(long_video.build_long_video_conditioning(clip=clip, video_vae=video_vae, audio_vae=audio_vae,
            context=contexts.low if phase == "low" else contexts.high, segment_index=request["segment_index"],
            context_frames=request["context_frames"], context_audio=context_audio, prompt=prompt,
            width=width, height=height, length=length, return_details=True, **options))
        if phase == "high":
            result[1], _ = lock_high_video_prefix(result[1], contexts.high,
                chain_id=request["chain_id"], segment_index=request["segment_index"], context_frames=request["context_frames"])
        provisional = ContinuationPhase(contexts, phase, tuple(result), "")
        descriptor = provisional.descriptor()
        descriptor["sha256"] = stages.sha(descriptor)
        result = ContinuationPhase(contexts, phase, tuple(result), stages.canonical(descriptor))
        result.verify()
        return result


def build_plan(contexts, length, sigmas, low_evaluations=4, low_scale=.5):
    """Geometry/schedule only: no HIGH prompt, encoding or sampling dependency."""
    contexts.verify()
    request = contexts.source.binding["request"]
    if type(length) is not int or length <= request["context_frames"]:
        raise ValueError("Continuation must leave newly generated frames")
    template, _ = core.empty_av_latent(request["width"], request["height"], length)
    plan = stages.build_plan(template, sigmas, low_evaluations, low_scale, "t2va", "initialized_av_exp")
    if (plan.low_width, plan.low_height) != (request["low_width"], request["low_height"]):
        raise ValueError("LOW scale disagrees with the accepted-picture context canvas")
    return plan


def _selected_conditions(prepared, positive=None, negative=None):
    """Allow external text/Bridge/Relay while preserving authenticated motion guides and frame grid."""
    positive = prepared.result[0] if positive is None else positive
    negative = positive if negative is None else negative
    def guides(value):
        if not isinstance(value, list) or not value:
            raise ValueError("Expected native continuation CONDITIONING")
        return stages.snapshot([{key: metadata.get(key) for key in PROTECTED} for _, metadata in value])
    if guides(positive) != guides(prepared.result[0]):
        raise ValueError("External conditioning changed authenticated continuation guides/grid")
    return positive, negative


def _checked_phase(prepared, phase, plan):
    if type(prepared) is not ContinuationPhase or prepared.phase != phase:
        raise ValueError("Connect the matching prepared continuation phase")
    prepared.verify()
    stages._geometry(prepared.result[1], plan, phase)
    request = prepared.contexts.source.binding["request"]
    if (plan.low_width, plan.low_height, plan.target_width, plan.target_height) != (
            request["low_width"], request["low_height"], request["width"], request["height"]):
        raise ValueError("Continuation plan changed its accepted-parent canvas")


def sample_low(prepared, model, sampler, plan, noise, *, positive=None, negative=None, seed,
               callback=None, reserve_vram_mib=1024):
    _checked_phase(prepared, "low", plan)
    with chain_guard(prepared.contexts.source.root):
        _checked_phase(prepared, "low", plan)
        pos, neg = _selected_conditions(prepared, positive, negative)
        selected = long_video.patch_long_video_model(model)
        boundary = stages.sample_low(selected, sampler, plan, prepared.result[1], pos, neg, noise,
            seed=seed, cfg=1., callback=callback, reserve_vram_mib=reserve_vram_mib)
        phase = prepared.verify()
        receipt = boundary.verify()
        receipt["request"][KEY] = {"contexts": prepared.contexts.verify(), "low_phase": phase,
                                  "clean_audio": stages._input_identity(prepared.result[1]["samples"].unbind()[1])}
        receipt["request_sha256"] = stages.sha(receipt["request"])
        receipt["portable_identity"] = bool(stages.portable(receipt["request"])
            and receipt["execution"]["actual_apply_calls"] >= plan.low_evaluations)
        receipt.pop("receipt_sha256")
        receipt["receipt_sha256"] = stages.sha(receipt)
        result = stages.ProgressiveBoundary(boundary.tensors, stages.canonical(receipt))
        result.verify()
        return result


def prepare_high(prepared, boundary, model, sampler, lifted_av, video_noise, *, high_sigmas=None):
    receipt = boundary.verify()
    plan = stages.plan_from_dict(receipt["request"]["plan"])
    _checked_phase(prepared, "high", plan)
    with chain_guard(prepared.contexts.source.root):
        descriptor = prepared.verify()
        low = receipt["request"].get(KEY)
        audio_id = stages._input_identity(prepared.result[1]["samples"].unbind()[1])
        if not isinstance(low, dict) or low["contexts"] != prepared.contexts.verify() or low["clean_audio"] != audio_id:
            raise ValueError("Frozen LOW has a different accepted-parent context or clean audio")
        source = {**prepared.result[1], KEY: {"contexts_sha256": low["contexts"]["sha256"],
            "source_sha256": prepared.contexts.source.sha256, "high_phase": descriptor,
            "delivery": {"accepted_source": prepared.contexts.source.binding,
                "mux_audio": prepared.result[2], "conditioned_prompt": prepared.result[3],
                "media_map_json": prepared.result[4], "conditioning_report": prepared.result[5]}}}
        result = stages.prepare_high(boundary, model, sampler, lifted_av, source, video_noise, high_sigmas=high_sigmas)
        prepared.verify()
        return result


def sample_high(prepared, restart, model, sampler, *, positive=None, negative=None, seed,
                callback=None, reserve_vram_mib=1024):
    _checked_phase(prepared, "high", restart.plan)
    with chain_guard(prepared.contexts.source.root):
        phase = prepared.verify()
        restart.verify()
        marker = restart.metadata.get(KEY)
        if (not isinstance(marker, dict) or marker.get("high_phase") != phase
                or marker.get("contexts_sha256") != prepared.contexts.verify()["sha256"]
                or marker.get("source_sha256") != prepared.contexts.source.sha256):
            raise ValueError("Continuation HIGH restart belongs to a different prepared phase")
        pos, neg = _selected_conditions(prepared, positive, negative)
        selected = long_video.patch_long_video_model(model)
        result, report = high_results.sample_high_result(restart, selected, sampler, pos, neg,
            seed=seed, cfg=1., callback=callback, reserve_vram_mib=reserve_vram_mib)
        prepared.verify()
        return result, report


def deliver(result, source):
    """Revalidate the selected parent and expose the completed segment unchanged.

No source re-encoding, sampling, candidate write, auto-accept, timeline append
or compose. The parent job SHA describes the selected old parent, not proof
that today's edited MODEL/recipe is the same job. Durable queue integration is
a separate layer which must bind the actual new stage receipts.
"""
    if type(result) is not high_results.ProgressiveHighResult or type(source) is not legacy_source.ProgressiveContinuationSource:
        raise ValueError("Continuation delivery requires completed HIGH and an authenticated accepted parent")
    with chain_guard(source.root):
        receipt = result.verify()
        binding = source.revalidate()
        marker = result.output.get(KEY)
        if (not isinstance(marker, dict) or marker.get("source_sha256") != source.sha256
                or marker.get("delivery", {}).get("accepted_source") != binding):
            raise ValueError("Completed continuation HIGH belongs to another accepted parent")
        policy = marker["delivery"]
        # Validate the actual selected PCM again. None retains the native
        # decode-generated-audio policy; do not silently replace model audio.
        if policy["mux_audio"] is not None:
            core.validate_audio(policy["mux_audio"])
            stages.masks._finite(policy["mux_audio"]["waveform"], "continuation mux audio")
        report = {"schema": "t8.modular-sampling.continuation-delivery.v1",
            "high_receipt_sha256": receipt["receipt_sha256"], "accepted_source_sha256": source.sha256,
            "audio_policy": "explicit_selected_pcm" if policy["mux_audio"] is not None else "decode_generated_audio",
            "selected_pcm": stages.snapshot(policy["mux_audio"]), "sampling_calls": 0, "vae_calls": 0,
            "candidate_written": False, "accepted": False, "composed": False, "quality_accepted": False,
            "boundary": "Selected previous job is source provenance, not today's edited recipe equivalence."}
        return result.output, policy["mux_audio"], policy["conditioned_prompt"], policy["media_map_json"], stages.canonical(report)
