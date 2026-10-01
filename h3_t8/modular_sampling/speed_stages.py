"""Explicit one-stage SPEED execution; the old whole-chain runner is untouched.

Each invocation prepares or samples exactly one planned canvas.  The live
result is intentionally not a portable cache receipt; disk restoration needs
its own source/model/effect identity contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.samplers
from comfy_extras.nodes_custom_sampler import Guider_Basic

from ..core import nested_av_parts
from ..sampling import setup_dual_clock_sampling
from ..speed_advanced import (
    SPEED_PLAN_SCHEMA,
    SPEED_SOURCE_SCHEMA,
    StageConditioning,
    _build_empty_t2va_stage,
    _build_stage,
    _ensure_native_h3_model,
    _profile_binding,
    _task_support,
    _weight_patch_contract,
    _apply_speed_scoped_headroom,
    _restore_speed_scoped_headroom,
    H3ModalityStableNoise,
)
from .contracts import StageContext
from . import speed_effects, native_explicit
from .speed_transition import transition_speed_stage
from .results import _conditions, _input_identity, canonical, implementation_identity, selected_model_identity, sha
from ..patch_stack_policy import UnverifiedModelStack


def _plan_sha(plan: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _tensor_version(tensor: torch.Tensor) -> int | None:
    # Core executes nodes under inference_mode, whose tensors have no version
    # counter. The full output-content identity below still detects mutation.
    return None if tensor.is_inference() else tensor._version


@dataclass(frozen=True)
class SpeedStageSpec:
    plan: dict[str, Any]
    plan_sha256: str
    index: int
    seed: int
    shift_audio: float
    scope: str
    source_ref: dict[str, Any]
    source_prompt: str
    stage: StageConditioning
    stage_context: StageContext
    sampling_object: Any
    audio_scale: float
    noise_scale: float
    support_reason: str
    profile_binding: dict[str, Any]
    weight_patch_report: dict[str, Any]

    @property
    def nfe(self) -> int:
        return int(self.plan["segments"][self.index]["nfe"])


@dataclass(frozen=True)
class FrozenSpeedSpec:
    """Only the authenticated fields needed for an explicit next-stage rebuild."""
    plan: dict[str, Any]
    plan_sha256: str
    index: int
    seed: int
    shift_audio: float
    source_prompt: str
    stage: StageConditioning
    stage_context: StageContext
    audio_scale: float
    noise_scale: float

    @property
    def nfe(self) -> int:
        return int(self.plan["segments"][self.index]["nfe"])


@dataclass(frozen=True)
class SpeedStageResult:
    spec: SpeedStageSpec
    external_output: Any
    video_version: int | None
    audio_version: int | None
    receipt_json: str

    def verify_live(self) -> dict[str, Any]:
        if _plan_sha(self.spec.plan) != self.spec.plan_sha256:
            raise ValueError("Completed SPEED plan changed after sampling")
        receipt = json.loads(self.receipt_json)
        if receipt.get("schema") != "t8.modular-sampling.speed-result.v1":
            raise ValueError("Unknown SPEED completed-stage receipt")
        unsigned = dict(receipt)
        claimed = unsigned.pop("receipt_sha256", None)
        if claimed != sha(unsigned) or receipt.get("request_sha256") != sha(receipt.get("request")):
            raise ValueError("SPEED completed-stage receipt integrity failed")
        request = receipt["request"]
        if (request.get("plan_sha256") != self.spec.plan_sha256
                or request.get("stage_index") != self.spec.index
                or request.get("seed") != self.spec.seed
                or request.get("shift_audio") != self.spec.shift_audio
                or request.get("audio_scale") != self.spec.audio_scale
                or request.get("noise_scale") != self.spec.noise_scale
                or request.get("source_prompt") != self.spec.source_prompt
                or request.get("text_template") != _input_identity(_text_template(self.spec.stage))
                or request.get("stage_context") != self.spec.stage_context.to_dict()):
            raise ValueError("SPEED completed-stage receipt belongs to another stage")
        video, audio = tuple(self.external_output.unbind())
        if (_tensor_version(video) != self.video_version
                or _tensor_version(audio) != self.audio_version):
            raise ValueError("Completed SPEED output was modified after sampling")
        shape = self.spec.plan["stages"][self.spec.index]
        if tuple(video.shape[-2:]) != (shape["latent_height"], shape["latent_width"]):
            raise ValueError("Completed SPEED output canvas changed")
        if receipt.get("output") != _input_identity(self.external_output):
            raise ValueError("Completed SPEED output content differs from its receipt")
        if receipt.get("verified_recipe_completion") is not (
            receipt["execution"]["sampler_known"]
            and receipt["execution"]["callbacks"] == list(range(self.spec.nfe))
        ):
            raise ValueError("SPEED completed-stage callback evidence disagrees")
        effects = request["model"].get("stage_effects")
        effect_verified = _effect_execution_verified(
            effects, receipt["execution"].get("effects"), self.spec.stage_context, self.spec.nfe,
        )
        expected_portable = (receipt["verified_recipe_completion"]
                             and request["model"].get("portable_cache_reuse", True)
                             and request["noise_provider"]["portable"]
                             and "unverified" not in request["conditions"]
                             and effect_verified)
        if receipt.get("portable_identity") is not bool(expected_portable):
            raise ValueError("SPEED portable execution/effect evidence disagrees")
        return receipt


@dataclass(frozen=True)
class FrozenSpeedStageResult:
    spec: FrozenSpeedSpec
    external_output: Any
    receipt_json: str

    def verify_live(self) -> dict[str, Any]:
        video, audio = tuple(self.external_output.unbind())
        return SpeedStageResult(
            self.spec, self.external_output, _tensor_version(video), _tensor_version(audio), self.receipt_json,
        ).verify_live()


class FixedSpeedNoise:
    """NOISE port for the next segment; refuses an unrelated target layout."""

    def __init__(self, value: Any, seed: int, source_receipt_sha256: str,
                 source_portable: bool, source_stage_index: int):
        self.value = value
        self.seed = int(seed)
        self.source_receipt_sha256 = source_receipt_sha256
        self.source_portable = source_portable
        self.source_stage_index = source_stage_index
        self.shapes = tuple(tuple(part.shape) for part in value.unbind())

    def generate_noise(self, input_latent):
        parts = tuple(input_latent["samples"].unbind())
        if tuple(tuple(part.shape) for part in parts) != self.shapes:
            raise ValueError("SPEED solved noise belongs to a different target AV layout")
        return self.value


def _speed_noise_identity(noise, index):
    if (index == 0 and type(noise) is H3ModalityStableNoise
            and set(vars(noise)) == {"seed"}):
        return {"provider": "h3_modality_stable_v1", "portable": True, "seed": noise.seed}
    if (index > 0 and type(noise) is FixedSpeedNoise
            and set(vars(noise)) == {"value", "seed", "source_receipt_sha256",
                                     "source_portable", "source_stage_index", "shapes"}
            and noise.source_stage_index == index - 1):
        return {"provider": "speed_dct_solved_v1", "portable": noise.source_portable,
                "seed": noise.seed, "source_receipt_sha256": noise.source_receipt_sha256,
                "source_stage_index": noise.source_stage_index,
                "value": _input_identity(noise.value)}
    return {"provider": "unverified_connected_noise", "portable": False}


def _implementation_identity():
    from .. import speed_advanced
    from . import speed_transition, effect_identity
    modules = (speed_advanced, speed_transition, speed_effects, effect_identity)
    return {"core_and_stage": implementation_identity(), **{
        Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in modules
    }, "speed_stages.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _condition_identity(positive):
    try:
        return _conditions(positive), True
    except UnverifiedModelStack as error:
        return {"unverified": str(error)}, False


def _text_template(stage):
    return {"positive": stage.positive, "mux_audio": stage.mux_audio,
            "conditioned_prompt": stage.conditioned_prompt, "media_map": stage.media_map,
            "report": stage.report}


def _relay_counts(model, effects):
    if effects is None or "relay" not in effects or "eav" in effects:
        return None
    from ..prompt_relay_advanced import prompt_relay_model_contract
    counts = prompt_relay_model_contract(model)["execution_counts"]
    return {name: int(counts[name]) for name in ("completed_forwards", "routed_attention_calls")}


def _effect_execution(model, effects, relay_before):
    if effects is None:
        return None
    from . import eav
    blocks = len(model.get_model_object("diffusion_model").blocks)
    if "eav" in effects:
        runtime = model.get_attachment(eav.KEY)
        if type(runtime) is not eav.StageEAVRuntime:
            return {"kind": "unverified"}
        audit = runtime.snapshot()
        return {"kind": "eav", "status": audit["status"], "mode": audit["config"]["mode"],
                "stage_context": audit["stage_context"], "blocks": blocks,
                "completed_forwards": audit["completed_forwards"],
                "planned_forwards": audit["planned_forwards"],
                "selector_calls": audit["selector_calls"],
                "sparse_producer_calls": audit["sparse_producer_calls"],
                "relay_attention_calls": audit["relay_attention_calls"],
                "relay_required": audit["relay_required"],
                "clock_match": audit["clock_match"],
                "aborted": bool(audit["feta"]["aborted"])}
    if "relay" in effects and relay_before is not None:
        after = _relay_counts(model, effects)
        return {"kind": "relay", "blocks": blocks,
                "completed_forwards": after["completed_forwards"] - relay_before["completed_forwards"],
                "routed_attention_calls": after["routed_attention_calls"] - relay_before["routed_attention_calls"]}
    return {"kind": "unverified"}


def _effect_execution_verified(effects, evidence, context, nfe):
    if effects is None:
        return evidence is None
    if type(effects) is not dict or type(evidence) is not dict:
        return False
    if evidence.get("kind") == "eav" and "eav" in effects:
        blocks = evidence.get("blocks")
        mode = effects["eav"].get("mode")
        status = evidence.get("status")
        return (type(blocks) is int and blocks > 0 and blocks == effects.get("blocks")
                and mode in ("report_only", "apply_exp") and evidence.get("mode") == mode
                and evidence.get("stage_context") == context.to_dict()
                and evidence.get("planned_forwards") == nfe
                and evidence.get("completed_forwards") == nfe
                and evidence.get("clock_match") is True and evidence.get("aborted") is False
                and evidence.get("selector_calls", -1) + evidence.get("sparse_producer_calls", -1) >= nfe * blocks
                and evidence.get("relay_required") is ("relay" in effects)
                and ("relay" not in effects or evidence.get("relay_attention_calls", -1) >= nfe * blocks)
                and status in (("observed_report_only", "observed_no_steps_in_effect_window")
                               if mode == "report_only" else
                               ("observed_apply_exp", "observed_no_steps_in_effect_window")))
    if evidence.get("kind") == "relay" and set(effects) == {"schema", "relay"}:
        blocks = evidence.get("blocks")
        return (type(blocks) is int and blocks > 0
                and evidence.get("completed_forwards") == nfe
                and evidence.get("routed_attention_calls", -1) >= nfe * blocks)
    return False


def prepare_speed_stage(
    model,
    plan: dict[str, Any],
    source: dict[str, Any],
    *,
    stage_index: int,
    shift_audio: float,
    seed: int,
    execution_scope: str,
    previous_spec: SpeedStageSpec | FrozenSpeedSpec | None = None,
    reuse_t2va_text: bool = True,
):
    if plan.get("schema") != SPEED_PLAN_SCHEMA or source.get("schema") != SPEED_SOURCE_SCHEMA:
        raise ValueError("SPEED Stage requires the published Plan and Source contracts")
    if len(plan["stages"]) < 2 or plan.get("status") != "planned":
        raise ValueError("A single-stage SPEED passthrough is not a multi-stage split")
    if type(stage_index) is not int or not 0 <= stage_index < len(plan["stages"]):
        raise ValueError("SPEED stage index is out of range")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("SPEED seed must be an unsigned 64-bit integer")
    if not math.isfinite(shift_audio) or shift_audio <= 0:
        raise ValueError("SPEED audio shift must be finite and positive")
    _ensure_native_h3_model(model)
    binding = _profile_binding(plan, source)
    supported, support_reason = _task_support(
        source, execution_scope, int(plan["steps"]), float(plan["shift_video"]), float(shift_audio)
    )
    _, _, weight_report = _weight_patch_contract(model, execution_scope)
    if not supported:
        # The old whole-chain fallback silently collapses to one full-size
        # stage.  A graph advertised as separated stages must never do that.
        raise ValueError(f"SPEED split cannot use a one-stage fallback: {support_reason}")
    digest = _plan_sha(plan)
    if previous_spec is not None and (
        type(previous_spec) not in (SpeedStageSpec, FrozenSpeedSpec)
        or previous_spec.plan_sha256 != digest
        or previous_spec.index + 1 != stage_index
        or previous_spec.seed != seed
        or previous_spec.shift_audio != float(shift_audio)
    ):
        raise ValueError("Previous SPEED stage does not belong to this next stage")
    shape = plan["stages"][stage_index]
    if stage_index and execution_scope in {"strict_t2va_stock20", "turbo8_t2va_research_exp"} and reuse_t2va_text:
        if previous_spec is None:
            raise ValueError("Strict SPEED text reuse needs the preceding stage spec")
        if source.get("prompt") != previous_spec.source_prompt:
            raise ValueError("Edited SPEED text requires reuse_t2va_text=false and a stage rebuild")
        stage = _build_empty_t2va_stage(source, shape["width"], shape["height"], previous_spec.stage)
    else:
        stage = _build_stage(source, shape["width"], shape["height"])
    sampler_name = "euler" if hasattr(comfy.model_sampling, "ModelSamplingAV") else "dual_clock_euler"
    prepared, sampler, full_sigmas = setup_dual_clock_sampling(
        model, stage.latent, int(plan["steps"]), float(plan["shift_video"]), float(shift_audio),
        sampler_name, "native_flow",
    )
    expected_full = torch.tensor(plan["full_sigmas"], dtype=torch.float32)
    if not torch.allclose(full_sigmas.cpu(), expected_full, atol=1e-7, rtol=0.0):
        raise RuntimeError("Native H3 sigma construction changed since this SPEED plan")
    segment = plan["segments"][stage_index]
    if segment["stage_index"] != stage_index or len(segment["sigmas"]) != segment["nfe"] + 1:
        raise ValueError("SPEED stage segment is inconsistent")
    sigmas = torch.tensor(segment["sigmas"], dtype=torch.float32)
    prepared, stage_context = speed_effects.bind(
        prepared, plan, stage_index, sigmas, stage.latent, shift_audio, execution_scope
    )
    sampling_object = prepared.get_model_object("model_sampling")
    video, audio = nested_av_parts(stage.latent)
    if tuple(video.shape[-2:]) != (shape["latent_height"], shape["latent_width"]):
        raise ValueError("Rebuilt SPEED video canvas differs from the plan")
    spec = SpeedStageSpec(
        plan, digest, stage_index, seed, float(shift_audio), execution_scope,
        source, str(source.get("prompt", "")), stage, stage_context, sampling_object,
        float(getattr(sampling_object, "audio_scale", 1.0)),
        float(getattr(sampling_object, "noise_scale", 1.0)), support_reason, binding, weight_report,
    )
    report = {
        "schema": "t8.modular-sampling.speed-stage.v1",
        "stage_index": stage_index,
        "canvas": [shape["width"], shape["height"]],
        "video_shape": list(video.shape),
        "audio_shape": list(audio.shape),
        "sigmas": list(segment["sigmas"]),
        "nfe_planned": spec.nfe,
        "conditioning_route": stage.route,
        "plan_sha256": digest,
        "sampled": False,
        "persistent_resume_authorized": False,
    }
    return (prepared, stage.positive, stage.latent, sampler, sigmas, spec,
            stage.mux_audio, stage.conditioned_prompt, stage.media_map, report, stage_context)


@torch.no_grad()
def sample_speed_stage(model, positive, av_latent, sampler, sigmas, noise, spec: SpeedStageSpec):
    if type(spec) is not SpeedStageSpec or spec.plan_sha256 != _plan_sha(spec.plan):
        raise ValueError("SPEED stage spec changed")
    if model.get_model_object("model_sampling") is not spec.sampling_object:
        raise ValueError("SPEED MODEL lost its prepared AV sampling object")
    speed_effects.validate_stage(model, sigmas, av_latent, spec.stage_context)
    expected = torch.tensor(spec.plan["segments"][spec.index]["sigmas"], dtype=torch.float32)
    if sigmas.shape != expected.shape or not torch.allclose(sigmas.cpu(), expected, atol=1e-7, rtol=0.0):
        raise ValueError("SPEED stage SIGMAS differ from the plan")
    video, audio = nested_av_parts(av_latent)
    original_video, original_audio = nested_av_parts(spec.stage.latent)
    if video.shape != original_video.shape or audio.shape != original_audio.shape:
        raise ValueError("SPEED stage AV source layout differs from the prepared stage")
    if getattr(noise, "seed", None) != spec.seed or not callable(getattr(noise, "generate_noise", None)):
        raise ValueError("SPEED stage NOISE must use the plan seed")
    actual_noise = noise.generate_noise(av_latent)
    noise_video, noise_audio = tuple(actual_noise.unbind())
    if noise_video.shape != video.shape or noise_audio.shape != audio.shape:
        raise ValueError("SPEED stage NOISE layout differs from its AV source")
    model_identity = selected_model_identity(model)
    noise_identity = _speed_noise_identity(noise, spec.index)
    source_identity = _input_identity(av_latent)
    actual_noise_identity = _input_identity(actual_noise)
    conditions_identity, conditions_portable = _condition_identity(positive)
    implementation = _implementation_identity()
    sampler_kind = native_explicit._sampler_kind(
        model, sampler, spec.stage_context, speed_effects.capture_owner(model),
    )
    effects = model_identity.get("stage_effects")
    relay_before = _relay_counts(model, effects)
    callbacks = []

    def record(step, _denoised, _current, total_steps):
        if int(total_steps) != spec.nfe:
            raise ValueError("SPEED sampler callback reported another stage length")
        callbacks.append(int(step))

    guider = Guider_Basic(model)
    guider.set_conds(positive)
    token = None
    try:
        token, headroom = _apply_speed_scoped_headroom()
        output_nested = guider.sample(
            actual_noise, av_latent["samples"], sampler, sigmas,
            denoise_mask=av_latent.get("noise_mask"), callback=record,
            disable_pbar=False, seed=spec.seed,
        )
    finally:
        _restore_speed_scoped_headroom(token)
    out_video, out_audio = tuple(output_nested.unbind())
    if out_video.shape != video.shape or out_audio.shape != audio.shape:
        raise RuntimeError("SPEED stage sampler returned an incompatible AV layout")
    if (_input_identity(av_latent) != source_identity or _condition_identity(positive)[0] != conditions_identity
            or selected_model_identity(model) != model_identity
            or _speed_noise_identity(noise, spec.index) != noise_identity
            or _implementation_identity() != implementation):
        raise ValueError("SPEED source, MODEL, controls or implementation changed during sampling")
    output = av_latent.copy()
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = output_nested
    completed = sampler_kind != "unadapted" and callbacks == list(range(spec.nfe))
    effect_execution = _effect_execution(model, effects, relay_before)
    effect_verified = _effect_execution_verified(effects, effect_execution, spec.stage_context, spec.nfe)
    portable = (completed and model_identity.get("portable_cache_reuse", True)
                and noise_identity["portable"] and conditions_portable
                and effect_verified)
    request = {"plan_sha256": spec.plan_sha256, "stage_index": spec.index,
               "stage_context": spec.stage_context.to_dict(), "seed": spec.seed,
               "shift_audio": spec.shift_audio,
               "audio_scale": spec.audio_scale, "noise_scale": spec.noise_scale,
               "source_prompt": spec.source_prompt,
               "text_template": _input_identity(_text_template(spec.stage)),
               "source": source_identity, "conditions": conditions_identity,
               "model": model_identity, "noise_provider": noise_identity,
               "actual_noise": actual_noise_identity, "sigmas": _input_identity(sigmas),
               "sampler_kind": sampler_kind, "implementation": implementation}
    receipt = {"schema": "t8.modular-sampling.speed-result.v1", "request": request,
               "request_sha256": sha(request), "output": _input_identity(output_nested),
               "execution": {"callbacks": callbacks, "sampler_known": sampler_kind != "unadapted",
                             "effects": effect_execution},
               "verified_recipe_completion": completed, "portable_identity": bool(portable),
               "automatic_cache_reuse": False}
    receipt["receipt_sha256"] = sha(receipt)
    result = SpeedStageResult(
        spec, output_nested, _tensor_version(out_video), _tensor_version(out_audio), canonical(receipt),
    )
    result.verify_live()
    report = {
        "schema": "t8.modular-sampling.speed-stage-result.v1",
        "stage_index": spec.index,
        "plan_sha256": spec.plan_sha256,
        "nfe_planned": spec.nfe,
        "sampling_calls": 1,
        "verified_recipe_completion": completed,
        "portable_identity": portable,
        "effect_execution_verified": effect_verified,
        "receipt_sha256": receipt["receipt_sha256"],
        "scoped_vram_headroom": headroom,
        "quality_validated": False,
        "persistent_resume_authorized": False,
    }
    return output, result, report


def handoff_speed_stage(result: SpeedStageResult | FrozenSpeedStageResult,
                        next_spec: SpeedStageSpec, *, dct_chunk_size: int = 64):
    if type(result) not in (SpeedStageResult, FrozenSpeedStageResult) or type(next_spec) is not SpeedStageSpec:
        raise TypeError("SPEED transition needs completed and prepared typed stages")
    receipt = result.verify_live()
    previous = result.spec
    if _plan_sha(next_spec.plan) != next_spec.plan_sha256:
        raise ValueError("Next SPEED plan changed after preparation")
    if previous.plan_sha256 != next_spec.plan_sha256 or previous.index + 1 != next_spec.index or previous.seed != next_spec.seed:
        raise ValueError("SPEED transition stages have different plans, order, or seeds")
    if previous.audio_scale != next_spec.audio_scale or previous.noise_scale != next_spec.noise_scale:
        raise ValueError("SPEED transition changed the AV sampling scale between stages")
    external_output = result.external_output
    if type(result) is FrozenSpeedStageResult:
        # Save/Load deliberately stores tensors on CPU, while Core's live
        # sampler returns its stage output on the diffusion load device.
        # The next-stage empty latent remains on the intermediate (usually
        # CPU) device, so it cannot identify the DCT compute device.
        # Restore the completed state to Core's diffusion device before the
        # DCT handoff, matching an uninterrupted live result.
        # Keep the verified frozen state unchanged for its receipt identity.
        compute_device = comfy.model_management.get_torch_device()
        video, audio = tuple(external_output.unbind())
        if video.device != compute_device or audio.device != compute_device:
            external_output = comfy.nested_tensor.NestedTensor((
                video.to(device=compute_device),
                audio.to(device=compute_device),
            ))
    value, report = transition_speed_stage(
        previous.plan, previous.index, external_output, next_spec.stage.latent,
        audio_scale=previous.audio_scale, noise_scale=previous.noise_scale,
        seed=previous.seed, dct_chunk_size=dct_chunk_size,
    )
    return FixedSpeedNoise(value, previous.seed, receipt["receipt_sha256"],
                           receipt["portable_identity"], previous.index), report
