"""Authenticated accepted-parent contexts for the separate HyperFlow P7 body.

The historical P7 runner and its accepted/cache formats are never rewritten.
Its LOW context is a separate candidate artifact whose sampling_summary is the
job SHA; the accepted HIGH context has the ordinary readable summary instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re

import folder_paths
import torch

from .. import long_video, long_video_delivery as delivery
from .. import long_video_dual_picture_context as picture
from .. import sampling
from ..learned_latent_upscale_advanced import PIXELS_PER_H3_LATENT, learned_upscale_h3_av_latent
from ..hyperflow_long_video_exp.runner import RECIPE
from ..long_video_in_node_loop_effects_advanced import _load_effects_audit
from .continuation import chain_guard
from .progressive import snapshot
from .results import StageResult, _input_identity, canonical, sha

SOURCE_SCHEMA = "t8.modular-sampling.hyperflow-p7-parent.v1"
CONTEXT_SCHEMA = "t8.modular-sampling.hyperflow-p7-contexts.v1"
PHASE_SCHEMA = "t8.modular-sampling.hyperflow-p7-phase.v1"
INITIAL_SCHEMA = "t8.modular-sampling.hyperflow-p7-initial.v1"
LOW_SOURCE_SCHEMA = "t8.modular-sampling.hyperflow-p7-low-source.v1"
LOW_RESULT_SCHEMA = "t8.modular-sampling.hyperflow-p7-low-result.v1"
LOW_SOURCE_KEY = "t8_hyperflow_p7_low_source"
LIFT_SCHEMA = "t8.modular-sampling.hyperflow-p7-lift.v1"
HIGH_SOURCE_SCHEMA = "t8.modular-sampling.hyperflow-p7-high-source.v1"
HIGH_RESULT_SCHEMA = "t8.modular-sampling.hyperflow-p7-high-result.v1"
HIGH_SOURCE_KEY = "t8_hyperflow_p7_high_source"


def _digest(value, name):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA256")
    return value


def _geometry_request(chain_id, context_frames, width, height, low_width, low_height):
    if type(chain_id) is not str or long_video.sanitize_chain_id(chain_id) != chain_id:
        raise ValueError("Use the exact normalized P7 chain_id")
    if type(context_frames) is not int or context_frames not in long_video.CONTEXT_FRAME_STEPS:
        raise ValueError("P7 context_frames must be 5, 22, or 39")
    for name, value in (("width", width), ("height", height),
                        ("low_width", low_width), ("low_height", low_height)):
        if type(value) is not int or value < 32 or value % 32:
            raise ValueError(f"P7 {name} must be a positive 32-pixel grid")
    if low_width >= width or low_height >= height:
        raise ValueError("P7 LOW canvas must be smaller in both dimensions")
    return dict(chain_id=chain_id, context_frames=context_frames, width=width, height=height,
                low_width=low_width, low_height=low_height)


def _request(chain_id, segment_index, parent_candidate_id, parent_revision,
             previous_job_sha256, context_frames, width, height, low_width, low_height):
    geometry = _geometry_request(chain_id, context_frames, width, height, low_width, low_height)
    if (type(parent_candidate_id) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", parent_candidate_id) is None):
        raise ValueError("Select an exact P7 accepted candidate ID")
    for name, value in (("segment_index", segment_index), ("parent_revision", parent_revision)):
        if type(value) is not int or value < 1:
            raise ValueError(f"P7 {name} must be a positive integer")
    _digest(previous_job_sha256, "Previous P7 job")
    return dict(**geometry, segment_index=segment_index, parent_candidate_id=parent_candidate_id,
                parent_revision=parent_revision, previous_job_sha256=previous_job_sha256)


def _capture(root, request):
    """Read and validate the original P7 chain under its exclusive loop lease."""
    chain_id, index, parent = (request[name] for name in
                               ("chain_id", "segment_index", "parent_candidate_id"))
    manifest_path = delivery._resolve_inside(root, root / delivery.MANIFEST_NAME)
    raw_manifest = manifest_path.read_bytes()
    manifest = delivery._validate_manifest(json.loads(raw_manifest), chain_id)
    if manifest["revision"] != request["parent_revision"] or len(manifest["segments"]) < index:
        raise ValueError("P7 accepted manifest revision or predecessor changed")
    entry = manifest["segments"][index - 1]
    if entry["candidate_id"] != parent or entry.get("is_final_segment"):
        raise ValueError("P7 selected parent is not the immediate nonfinal predecessor")
    if (entry["width"], entry["height"], entry["fps"]) != (
            request["width"], request["height"], 24):
        raise ValueError("P7 accepted parent canvas/fps changed")
    candidate_path = delivery._resolve_inside(root, root / "candidates"
        / f"segment_{index - 1:05d}" / parent / "candidate.json")
    raw_candidate = candidate_path.read_bytes()
    candidate = json.loads(raw_candidate)
    if (candidate.get("candidate_id"), candidate.get("chain_id"), candidate.get("index")) != (
            parent, chain_id, index - 1):
        raise ValueError("P7 candidate is not the selected immediate predecessor")
    if (candidate.get("video_sha256") != entry["video_sha256"]
            or candidate.get("context_sha256") != entry.get("context_sha256")):
        raise ValueError("P7 accepted media/context differs from candidate")
    candidate_media, picture_source = picture.accepted_source(root, parent, index, chain_id)
    accepted_media = delivery._resolve_inside(root, entry["video_path"])
    if (picture_source["source_media_sha256"] != entry["video_sha256"]
            or delivery._sha256_file(accepted_media) != entry["video_sha256"]):
        raise ValueError("P7 accepted copy differs from selected candidate movie")
    accepted_context_path = delivery._resolve_inside(root, entry["context_path"])
    candidate_context_path = delivery._resolve_inside(root, candidate["context_path"])
    if (delivery._sha256_file(accepted_context_path) != entry["context_sha256"]
            or delivery._sha256_file(candidate_context_path) != entry["context_sha256"]):
        raise ValueError("P7 accepted context differs from selected candidate")
    audit = _load_effects_audit(candidate_path,
        contract_sha256=request["previous_job_sha256"],
        segment_index=index - 1, candidate_id=parent)
    plan = audit.get("sampling_plan", {})
    if (plan.get("mode") != RECIPE or plan.get("hyperflow", {}).get("intervals") != [[0, 4], [4, 8]]):
        raise ValueError("Selected parent did not execute the P7 partial4+fresh4 recipe")
    record = plan.get("dual_model", {}).get("low_context")
    if not isinstance(record, dict) or record.get("audio_source") != "completed_second_pass_output":
        raise ValueError("P7 accepted parent has no completed-audio LOW context")
    low_path = delivery._resolve_inside(root, record["path"])
    if delivery._sha256_file(low_path) != record.get("sha256"):
        raise ValueError("P7 LOW context checksum changed")
    low, _ = delivery._load_accepted_context_file(low_path, chain_id, index - 1, index)
    high, _ = delivery._load_accepted_context_file(accepted_context_path, chain_id, index - 1, index)
    long_video._validate_context(low, index, request["context_frames"],
                                 request["low_width"], request["low_height"])
    long_video._validate_context(high, index, request["context_frames"],
                                 request["width"], request["height"])
    if (low["metadata"]["sampling_summary"] != request["previous_job_sha256"]
            or high["metadata"]["sampling_summary"] != entry["sampling_summary"]):
        raise ValueError("P7 LOW/HIGH contexts have incompatible source contracts")
    if (low["audio_tail"].shape != high["audio_tail"].shape
            or not torch.equal(low["audio_tail"], high["audio_tail"])):
        raise ValueError("P7 accepted LOW/HIGH completed audio differs")
    if manifest_path.read_bytes() != raw_manifest or candidate_path.read_bytes() != raw_candidate:
        raise ValueError("P7 accepted parent changed during validation")
    binding = {"schema": SOURCE_SCHEMA, "request": request,
               "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
               "candidate_sha256": hashlib.sha256(raw_candidate).hexdigest(),
               "audit_sha256": audit["audit_sha256"], "picture": picture_source,
               "accepted_video_sha256": entry["video_sha256"],
               "accepted_context_sha256": entry["context_sha256"],
               "low_context_sha256": record["sha256"],
               "completed_audio_sha256": picture.tensor_sha(low["audio_tail"])}
    return binding, low, high, candidate_media, picture_source


@dataclass(frozen=True)
class P7Parent:
    root: Path
    binding_json: str

    @property
    def binding(self):
        return json.loads(self.binding_json)

    @property
    def sha256(self):
        return hashlib.sha256(self.binding_json.encode()).hexdigest()

    def revalidate(self):
        with chain_guard(self.root):
            current, *_ = _capture(self.root, self.binding["request"])
        if canonical(current) != self.binding_json:
            raise ValueError("P7 accepted parent changed; do not reuse stage result")
        return current

    def prepare_contexts(self, video_vae):
        with chain_guard(self.root):
            binding, low, high, media, source = _capture(self.root, self.binding["request"])
            if canonical(binding) != self.binding_json:
                raise ValueError("P7 accepted parent changed before VAE preparation")
            before_audio = picture.tensor_sha(low["audio_tail"])
            high_video = picture.tensor_sha(high["video_tail"])
            request = binding["request"]
            prepared, report = picture.prepare_context(low, media, source, video_vae,
                request["low_width"], request["low_height"])
            if (prepared["audio_tail"] is not low["audio_tail"]
                    or picture.tensor_sha(prepared["audio_tail"]) != before_audio
                    or picture.tensor_sha(high["video_tail"]) != high_video):
                raise RuntimeError("P7 accepted-picture preparation changed completed HIGH/AV context")
            after, *_ = _capture(self.root, request)
            if canonical(after) != self.binding_json:
                raise ValueError("P7 accepted parent changed during VAE preparation")
        provisional = P7Contexts(self, prepared, high, report, "")
        contract = provisional.descriptor()
        contract["sha256"] = sha(contract)
        value = P7Contexts(self, prepared, high, report, canonical(contract))
        value.verify()
        return value


@dataclass(frozen=True)
class P7Contexts:
    parent: P7Parent
    low: dict
    high: dict
    preparation: dict
    contract_json: str

    def descriptor(self):
        binding = self.parent.revalidate()
        request = binding["request"]
        for phase, context in (("low", self.low), ("high", self.high)):
            width, height = ((request["low_width"], request["low_height"]) if phase == "low"
                             else (request["width"], request["height"]))
            long_video._validate_context(context, request["segment_index"],
                                         request["context_frames"], width, height)
        if picture.tensor_sha(self.low["audio_tail"]) != binding["completed_audio_sha256"]:
            raise ValueError("P7 LOW completed audio changed")
        if not torch.equal(self.low["audio_tail"], self.high["audio_tail"]):
            raise ValueError("P7 LOW/HIGH completed audio changed")
        return {"schema": CONTEXT_SCHEMA, "parent_sha256": self.parent.sha256,
                "low": snapshot(self.low), "high": snapshot(self.high),
                "preparation": self.preparation}

    def verify(self):
        expected = json.loads(self.contract_json)
        current = self.descriptor()
        current["sha256"] = sha(current)
        if current != expected:
            raise ValueError("P7 accepted contexts changed after preparation")
        return expected


def capture_parent(root, *, chain_id, segment_index, parent_candidate_id,
                   parent_revision, previous_job_sha256, context_frames,
                   width, height, low_width, low_height):
    request = _request(chain_id, segment_index, parent_candidate_id, parent_revision,
        previous_job_sha256, context_frames, width, height, low_width, low_height)
    root = Path(root).resolve(strict=True)
    with chain_guard(root):
        binding, *_ = _capture(root, request)
    return P7Parent(root, canonical(binding))


def _initial_binding(root, request):
    """Inspect a new/empty chain without creating its directory or manifest."""
    if root.exists() and not root.is_dir():
        raise ValueError("P7 initial chain path is not a directory")
    manifests = {}
    for name in (delivery.MANIFEST_NAME, delivery.MANIFEST_BACKUP_NAME):
        path = root / name
        if path.exists():
            if not path.is_file():
                raise ValueError("P7 initial manifest path is not a file")
            raw = path.read_bytes()
            accepted = delivery._validate_manifest(json.loads(raw), request["chain_id"])
            if accepted["segments"]:
                raise ValueError("P7 segment 0 cannot reuse a chain with accepted segments")
            manifests[name] = hashlib.sha256(raw).hexdigest()
        else:
            manifests[name] = None
    return {"schema": INITIAL_SCHEMA, "request": request, "manifests": manifests}


@dataclass(frozen=True)
class P7InitialContexts:
    root: Path
    binding_json: str

    @property
    def request(self):
        return json.loads(self.binding_json)["request"]

    @property
    def low(self):
        return long_video._empty_context(self.request["chain_id"], 0)

    @property
    def high(self):
        return long_video._empty_context(self.request["chain_id"], 0)

    def verify(self):
        current = _initial_binding(self.root, self.request)
        if canonical(current) != self.binding_json:
            raise ValueError("P7 initial chain changed after selection")
        return {**current, "sha256": sha(current)}


def capture_initial(root, *, chain_id, context_frames, width, height, low_width, low_height):
    request = _geometry_request(chain_id, context_frames, width, height, low_width, low_height)
    request["segment_index"] = 0
    request["continuation_context_frames"] = request["context_frames"]
    request["context_frames"] = 0  # The original Segment Planner passes zero on segment 0.
    root = Path(root).resolve()
    value = P7InitialContexts(root, canonical(_initial_binding(root, request)))
    value.verify()
    return value


@dataclass(frozen=True)
class P7Phase:
    contexts: P7Contexts | P7InitialContexts
    phase: str
    result: tuple
    contract_json: str

    def descriptor(self):
        if self.phase not in ("low", "high") or type(self.contexts) not in (P7Contexts, P7InitialContexts):
            raise ValueError("Expected authenticated P7 LOW or HIGH phase")
        contexts = self.contexts.verify()
        if type(self.result) is not tuple or len(self.result) != 7:
            raise ValueError("Expected complete native P7 conditioning details")
        from . import progressive as stages

        stages.masks._av_parts(self.result[1]["samples"], "P7 " + self.phase)
        return {"schema": PHASE_SCHEMA, "phase": self.phase,
                "contexts_sha256": contexts["sha256"], "result": snapshot(self.result)}

    def verify(self):
        current = self.descriptor()
        current["sha256"] = sha(current)
        if current != json.loads(self.contract_json):
            raise ValueError("Prepared P7 phase changed")
        return current


def prepare_phase(contexts, phase, *, clip, video_vae, audio_vae, prompt, length):
    """Exact original P7 T2VA/native conditions with independently chosen prompt.

    Model patching, Relay, EAV and sampling belong to explicit downstream nodes.
    """
    if type(contexts) not in (P7Contexts, P7InitialContexts) or phase not in ("low", "high"):
        raise ValueError("Connect authenticated P7 contexts and select LOW or HIGH")
    request = (contexts.parent.binding["request"] if type(contexts) is P7Contexts else contexts.request)
    if type(length) is not int or length <= request["context_frames"]:
        raise ValueError("P7 continuation must leave newly generated frames")
    before = contexts.verify()
    guard = chain_guard(contexts.parent.root) if type(contexts) is P7Contexts else nullcontext()
    with guard:
        width, height = ((request["low_width"], request["low_height"]) if phase == "low"
                         else (request["width"], request["height"]))
        result = list(long_video.build_long_video_conditioning(
            clip=clip, video_vae=video_vae, audio_vae=audio_vae,
            context=contexts.low if phase == "low" else contexts.high,
            segment_index=request["segment_index"], context_frames=request["context_frames"],
            context_audio="video_and_audio", prompt=prompt, width=width, height=height,
            length=length, task_type="T2VA", audio_mode="native", audio_denoise_strength=.35,
            add_source_as_reference=False, prompt_primary_audio_ordinal=0,
            strict_prompt_tags=True, ref_image_size="match",
            reference_video_policy="official_2_to_15s", first_frame_reuse="segment0_only",
            persistent_identity_strategy="single_reference", persistent_identity_interval=1,
            return_details=True))
    if contexts.verify() != before:
        raise ValueError("P7 accepted contexts changed during phase preparation")
    provisional = P7Phase(contexts, phase, tuple(result), "")
    descriptor = provisional.descriptor()
    descriptor["sha256"] = sha(descriptor)
    value = P7Phase(contexts, phase, tuple(result), canonical(descriptor))
    value.verify()
    return value


def low_stage_source(phase):
    if type(phase) is not P7Phase or phase.phase != "low":
        raise ValueError("Select the authenticated P7 LOW phase")
    descriptor = phase.verify()
    source = phase.result[1]
    if LOW_SOURCE_KEY in source:
        raise ValueError("P7 LOW source marker was already present")
    marker = {"schema": LOW_SOURCE_SCHEMA, "phase_sha256": descriptor["sha256"],
              "contexts_sha256": descriptor["contexts_sha256"],
              "segment_index": (phase.contexts.parent.binding["request"]["segment_index"]
                                if type(phase.contexts) is P7Contexts else 0)}
    return {**source, LOW_SOURCE_KEY: marker}


def setup_low(phase, model, positive=None, negative=None):
    """Only original P7 LOW model patch and absolute 0:4 setup, no sampling."""
    from . import continuation, hyperflow_fresh as fresh

    source = low_stage_source(phase)
    positive, negative = continuation._selected_conditions(phase, positive, negative)
    patched = long_video.patch_long_video_model(model)
    selected, sampler, sigmas, stage_context, setup_report = fresh.build_stage(
        patched, source, "hyperflow_low_partial4")
    report = {"schema": LOW_SOURCE_SCHEMA, "source": source[LOW_SOURCE_KEY],
              "fresh_setup": json.loads(setup_report), "sampling_calls": 0,
              "boundary": "Connect optional external EAV/paired Relay and ONE Stage Sampler; "
                          "bind its actual result back to this P7 LOW phase."}
    return selected, sampler, sigmas, source, stage_context, positive, negative, canonical(report)


@dataclass(frozen=True)
class P7LowResult:
    phase: P7Phase
    sampled: StageResult
    contract_json: str

    def descriptor(self):
        source = low_stage_source(self.phase)
        if type(self.sampled) is not StageResult:
            raise ValueError("P7 LOW requires an actual typed Stage Result")
        receipt = self.sampled.verify()
        stage = receipt["request"]["stage_context"]
        if (stage["stage"] != "hyperflow_low_partial4" or stage["start"] != 0
                or stage["end"] != 4 or not receipt["verified_recipe_completion"]
                or receipt["request"]["source"] != _input_identity(source)):
            raise ValueError("Completed LOW result is not from this P7 source/phase")
        return {"schema": LOW_RESULT_SCHEMA, "phase_sha256": self.phase.verify()["sha256"],
                "source": source[LOW_SOURCE_KEY],
                "stage_receipt_sha256": receipt["receipt_sha256"],
                "selected_socket": "denoised_output", "sampling_calls": 0}

    def verify(self):
        current = self.descriptor()
        current["sha256"] = sha(current)
        if current != json.loads(self.contract_json):
            raise ValueError("P7 LOW binding changed after completion")
        return current


def bind_low(phase, sampled):
    provisional = P7LowResult(phase, sampled, "")
    descriptor = provisional.descriptor()
    descriptor["sha256"] = sha(descriptor)
    value = P7LowResult(phase, sampled, canonical(descriptor))
    value.verify()
    return value


@dataclass(frozen=True)
class P7Lift:
    low: P7LowResult
    latent: dict
    report_json: str
    contract_json: str

    def descriptor(self):
        low = self.low.verify()
        report = json.loads(self.report_json)
        request = (self.low.phase.contexts.parent.binding["request"]
                   if type(self.low.phase.contexts) is P7Contexts else self.low.phase.contexts.request)
        low_video, low_audio = sampling.nested_av_parts(self.low.sampled.denoised_output)
        video, audio = sampling.nested_av_parts(self.latent)
        geometry = report.get("geometry", {})
        model = report.get("model", {})
        if (report.get("status") != "ok" or report.get("node") != "MiniMaxH3LearnedLatentUpscaleT8Advanced"
                or geometry.get("size_mode") != "target_dimensions"
                or geometry.get("aspect_policy") != "honor_dimensions_exp"
                or (geometry.get("source_width"), geometry.get("source_height")) !=
                    (request["low_width"], request["low_height"])
                or (geometry.get("output_width"), geometry.get("output_height")) !=
                    (request["width"], request["height"])
                or tuple(video.shape[:3]) != tuple(low_video.shape[:3])
                or tuple(video.shape[-2:]) != (request["height"] // PIXELS_PER_H3_LATENT,
                                                request["width"] // PIXELS_PER_H3_LATENT)
                or not torch.equal(audio, low_audio) or report.get("audio_preserved") is not True
                or model.get("precision") != "fp16" or report.get("release_policy") != "offload_after"):
            raise ValueError("P7 learned lift differs from original dimensions/audio/policy")
        path = Path(folder_paths.get_full_path_or_raise("latent_upscale_models", model.get("name")))
        if str(path) != model.get("path") or delivery._sha256_file(path) != _digest(model.get("sha256"), "P7 upscaler"):
            raise ValueError("P7 learned upscaler file changed")
        return {"schema": LIFT_SCHEMA, "low_sha256": low["sha256"],
                "upscaler_sha256": model["sha256"], "report": report,
                "latent": _input_identity(self.latent), "sampling_calls": 0}

    def verify(self):
        current = self.descriptor()
        current["sha256"] = sha(current)
        if current != json.loads(self.contract_json):
            raise ValueError("P7 learned lift changed")
        return current


def lift_low(low, upscaler_model):
    """One standalone call to the exact historical P7 learned3D settings."""
    if type(low) is not P7LowResult:
        raise ValueError("P7 lift requires a completed authenticated LOW result")
    low.verify()
    request = (low.phase.contexts.parent.binding["request"]
               if type(low.phase.contexts) is P7Contexts else low.phase.contexts.request)
    latent, width, height, report = learned_upscale_h3_av_latent(
        low.sampled.denoised_output, upscaler_model, "target_dimensions", 2.0, 1.0,
        request["width"], request["height"], "honor_dimensions_exp", 1.05, "fp16", "offload_after")
    if (width, height) != (request["width"], request["height"]):
        raise ValueError("P7 learned upscaler returned another canvas")
    provisional = P7Lift(low, latent, report, "")
    descriptor = provisional.descriptor()
    descriptor["sha256"] = sha(descriptor)
    value = P7Lift(low, latent, report, canonical(descriptor))
    value.verify()
    return value


@dataclass(frozen=True)
class P7HighInput:
    lift: P7Lift
    phase: P7Phase
    source: dict
    positive: list
    negative: list
    reconcile_json: str
    contract_json: str

    def descriptor(self):
        from . import continuation

        lift = self.lift.verify()
        high = self.phase.verify()
        low_phase = self.lift.low.phase
        if (self.phase.phase != "high" or self.phase.contexts is not low_phase.contexts
                or self.phase.result[1]["samples"].unbind()[0].shape[2] !=
                    low_phase.result[1]["samples"].unbind()[0].shape[2]):
            raise ValueError("P7 HIGH must share LOW segment, contexts and frame grid")
        if HIGH_SOURCE_KEY not in self.source:
            raise ValueError("P7 HIGH source marker is missing")
        if self.source[HIGH_SOURCE_KEY] != {"schema": HIGH_SOURCE_SCHEMA,
                "lift_sha256": lift["sha256"], "high_phase_sha256": high["sha256"]}:
            raise ValueError("P7 HIGH source is not this LOW lift and HIGH phase")
        # The reconciler intentionally changes the HIGH positive's audio masks.
        # Preserve its exact resulting CONDITIONING, not a second generic guide comparison.
        continuation._selected_conditions(self.phase, self.phase.result[0], self.negative)
        video, audio = sampling.nested_av_parts(self.source)
        expected_video, expected_audio = sampling.nested_av_parts(self.phase.result[1])
        if video.shape != expected_video.shape or audio.shape != expected_audio.shape:
            raise ValueError("P7 HIGH reconciled AV geometry changed")
        return {"schema": HIGH_SOURCE_SCHEMA, "lift_sha256": lift["sha256"],
                "high_phase_sha256": high["sha256"], "source": _input_identity(self.source),
                "positive": snapshot(self.positive), "negative": snapshot(self.negative),
                "reconcile": json.loads(self.reconcile_json), "sampling_calls": 0}

    def verify(self):
        current = self.descriptor()
        current["sha256"] = sha(current)
        if current != json.loads(self.contract_json):
            raise ValueError("P7 HIGH handoff changed")
        return current


def handoff_high(lift, phase, positive=None, negative=None):
    from . import continuation, native_dual

    if type(lift) is not P7Lift or type(phase) is not P7Phase or phase.phase != "high":
        raise ValueError("Connect a verified P7 lift and HIGH phase")
    lift.verify()
    phase.verify()
    if phase.contexts is not lift.low.phase.contexts:
        raise ValueError("P7 HIGH contexts must match the completed LOW segment")
    selected_positive, selected_negative = continuation._selected_conditions(phase, positive, negative)
    prepared, bound, report = native_dual.reconcile(lift.latent, phase.result[1], selected_positive,
        first_pass_steps=4, second_audio_source="auto", second_audio_strength=0.)
    marker = {"schema": HIGH_SOURCE_SCHEMA, "lift_sha256": lift.verify()["sha256"],
              "high_phase_sha256": phase.verify()["sha256"]}
    if HIGH_SOURCE_KEY in prepared:
        raise ValueError("P7 HIGH source marker was already present")
    value = P7HighInput(lift, phase, {**prepared, HIGH_SOURCE_KEY: marker}, bound,
                        selected_negative, report, "")
    descriptor = value.descriptor()
    descriptor["sha256"] = sha(descriptor)
    value = P7HighInput(lift, phase, value.source, bound, selected_negative, report, canonical(descriptor))
    value.verify()
    return value


def setup_high(handoff, model):
    from . import hyperflow_fresh as fresh

    if type(handoff) is not P7HighInput:
        raise ValueError("P7 HIGH setup requires the explicit learned/reconciled handoff")
    descriptor = handoff.verify()
    patched = long_video.patch_long_video_model(model)
    selected, sampler, sigmas, stage_context, setup_report = fresh.build_stage(
        patched, handoff.source, "hyperflow_high_after_partial4")
    report = {"schema": HIGH_SOURCE_SCHEMA, "handoff_sha256": descriptor["sha256"],
              "fresh_setup": json.loads(setup_report), "sampling_calls": 0}
    return selected, sampler, sigmas, handoff.source, stage_context, handoff.positive, handoff.negative, canonical(report)


@dataclass(frozen=True)
class P7HighResult:
    handoff: P7HighInput
    sampled: StageResult
    contract_json: str

    def descriptor(self):
        handoff = self.handoff.verify()
        if type(self.sampled) is not StageResult:
            raise ValueError("P7 HIGH requires an actual typed Stage Result")
        receipt = self.sampled.verify()
        stage = receipt["request"]["stage_context"]
        if (stage["stage"] != "hyperflow_high_after_partial4" or stage["start"] != 4
                or stage["end"] != 8 or not receipt["verified_recipe_completion"]
                or receipt["request"]["source"] != _input_identity(self.handoff.source)):
            raise ValueError("Completed HIGH result is not from this P7 handoff")
        return {"schema": HIGH_RESULT_SCHEMA, "handoff_sha256": handoff["sha256"],
                "stage_receipt_sha256": receipt["receipt_sha256"],
                "selected_socket": "output", "completed_audio": "second_pass_joint_av"}

    def verify(self):
        current = self.descriptor()
        current["sha256"] = sha(current)
        if current != json.loads(self.contract_json):
            raise ValueError("P7 HIGH binding changed")
        return current


def bind_high(handoff, sampled):
    provisional = P7HighResult(handoff, sampled, "")
    descriptor = provisional.descriptor()
    descriptor["sha256"] = sha(descriptor)
    value = P7HighResult(handoff, sampled, canonical(descriptor))
    value.verify()
    return value
