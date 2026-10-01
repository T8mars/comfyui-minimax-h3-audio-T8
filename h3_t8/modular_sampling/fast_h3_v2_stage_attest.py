"""Bind one completed V2 stage to the live editable graph, never accept media.

The current recipe and a StageResult are different claims. This audit checks
their shared branch against the actual sampler inputs, including the original
MODEL before stage setup. It does not certify a parent segment or save a
candidate; both stages must be independently audited before delivery.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

from ..core import align_frame_count, temporal_shape
from ..fast_h3_v2_advanced import AUDIO_SHIFT, VIDEO_SHIFT, dmd_sigmas
from ..long_video_dual_identity import content_identity, stage_model_identity, _v2_stage_adapter
from ..prompt_relay_advanced import _validate_plan
from ..prompt_relay_long_video_advanced import PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY
from .contracts import StageContext
from .continuation_identity import project as project_continuation
from .eav import EAVConfig, KEY as EAV_KEY
from .effect_identity import project_stage_effects
from .fast_h3_v2 import RECIPE
from .fast_h3_v2_conditioning import verify_condition_provenance
from .fast_h3_v2_job import FastH3V2CurrentRecipe
from .noise import operator_identity
from .results import (StageResult, _input_identity, canonical, implementation_identity,
                      request_conditions, selected_model_identity)


SCHEMA = "t8.modular-sampling.fasth3-v2-stage-current-attestation.v1"


@dataclass(frozen=True)
class FastH3V2StageAttestation:
    payload_json: str
    sha256: str

    def verify(self):
        payload = json.loads(self.payload_json)
        if payload.get("schema") != SCHEMA or canonical(payload) != self.payload_json:
            raise ValueError("Unknown or noncanonical FastH3 V2 stage attestation")
        if hashlib.sha256(self.payload_json.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("FastH3 V2 stage attestation fingerprint changed")
        return payload


def _raw_model_from_stage(model, *, _omit_dormant_sampling=False):
    """Undo only source-authenticated stage/effect wrappers on read-only clones."""
    view, _ = project_stage_effects(model)
    # The live Long Video owner is separately checked against today's exact
    # global/projected plan before this projection is called. Remove only its
    # inert descriptor on the read-only clone, never from the executing MODEL.
    if view.get_attachment(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY) is not None:
        view.remove_attachments(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY)
    view, _ = project_continuation(view)
    if view.get_attachment("t8_fasth3_v2_owner_v1") is None:
        raise ValueError("Stage MODEL lost its authenticated V2 owner")
    raw_view, _ = _v2_stage_adapter(view)
    options = dict(raw_view.model_options.get("transformer_options", {}))
    for name, expected in (("minimax_h3_sigma_shift_video", VIDEO_SHIFT),
                           ("minimax_h3_sigma_shift_audio", AUDIO_SHIFT)):
        if options.pop(name, None) != expected:
            raise ValueError("V2 stage lost its exact installed AV shift")
    raw_view.model_options = {**raw_view.model_options, "transformer_options": options}
    return stage_model_identity(raw_view, _omit_dormant_sampling=_omit_dormant_sampling)


def attest_current_stage(current_recipe, stage_result, raw_model, stage_model,
                         guider, noise, sigmas, source_latent, stage_context,
                         global_plan, projected_plan, eav_config, segment_index,
                         phase, condition_receipt=None, conditioned_latent=None):
    if type(current_recipe) is not FastH3V2CurrentRecipe:
        raise ValueError("Current stage needs the typed V2 recipe fingerprint")
    recipe = current_recipe.verify()
    if type(stage_result) is not StageResult:
        raise ValueError("Current stage needs an actual typed StageResult")
    receipt = stage_result.verify()
    if (receipt.get("verified_recipe_completion") is not True
            or receipt.get("portable_identity") is not True):
        raise ValueError("Current stage needs a completed portable sampler receipt")
    if phase not in ("low_0_4", "high_4_8") or type(segment_index) is not int or segment_index not in (0, 1):
        raise ValueError("Current V2 pilot supports only stage LOW/HIGH in segment 0/1")
    context = StageContext.from_dict(receipt["request"]["stage_context"])
    if type(stage_context) is not StageContext or stage_context.to_dict() != context.to_dict():
        raise ValueError("Current stage descriptor differs from its sampled receipt")
    start, end = (0, 4) if phase == "low_0_4" else (4, 8)
    if (context.recipe != RECIPE or context.stage != phase or context.profile != "dense_compat_exp"
            or (context.start, context.end) != (start, end)
            or context.trajectory_sigmas != tuple(float(value) for value in dmd_sigmas().tolist())
            or (context.video_shift, context.audio_shift) != (VIDEO_SHIFT, AUDIO_SHIFT)):
        raise ValueError("Current stage is not the exact dense FastH3 V2 absolute sampler window")
    frames = (124 if segment_index == 0 else
              recipe["window"].get("continuation_render_frames", 90))
    if align_frame_count(frames) != frames:
        raise RuntimeError("Current V2 pilot window is no longer native-aligned")
    width_key, height_key = ("low_width", "low_height") if phase == "low_0_4" else ("width", "height")
    width, height = recipe["geometry"][width_key], recipe["geometry"][height_key]
    _, video_tokens, audio_tokens = temporal_shape(frames)
    if (context.video_shape != (1, 24, video_tokens, height // 16, width // 16)
            or context.audio_shape != (1, 32, 2, audio_tokens)):
        raise ValueError("Current stage AV geometry differs from the recipe window")
    request = receipt["request"]
    if (request["model"] != selected_model_identity(stage_model)
            or request["source"] != _input_identity(source_latent)
            or request["conditions"] != request_conditions(guider.original_conds)
            or request["sigmas"] != _input_identity(sigmas)
            or request["noise_operator"] != operator_identity(noise)
            or request["seed"] != int(noise.seed)
            or request["implementation"] != implementation_identity()):
        raise ValueError("Current sampler inputs differ from the completed stage receipt")
    if getattr(guider, "model_patcher", None) is not stage_model:
        raise ValueError("Current guider is not connected to the attested stage MODEL")
    if request["seed"] != recipe["window"]["first_seed"] + segment_index:
        raise ValueError("Current stage seed differs from the two-window recipe")
    expected_raw = recipe["model_pass1" if phase == "low_0_4" else "model_pass2"]
    if stage_model_identity(raw_model, _omit_dormant_sampling=True) != expected_raw:
        raise ValueError("Current raw MODEL/LoRA branch differs from the editable recipe")
    if type(eav_config) is not EAVConfig:
        raise ValueError("Current stage needs its native external EAV config")
    expected_eav = recipe["eav"]["low" if phase == "low_0_4" else "high"]
    effects = request["model"].get("stage_effects")
    if type(effects) is dict and "continuation" in effects:
        effects = effects.get("effects")
    if asdict(eav_config) != expected_eav or type(effects) is not dict:
        raise ValueError("Current stage EAV setting differs from the editable recipe")
    if eav_config.mode == "disabled":
        if stage_model.get_attachment(EAV_KEY) is not None or "eav" in effects:
            raise ValueError("Disabled stage EAV unexpectedly installed an executable effect")
    elif (effects.get("eav") != expected_eav
          or effects.get("stage_context") != context.to_dict()):
        raise ValueError("Current stage EAV effect differs from the editable recipe")
    global_checked, projected_checked = _validate_plan(global_plan), _validate_plan(projected_plan)
    expected_global = recipe["relay"]["low_global_plan_sha256" if phase == "low_0_4"
                                       else "high_global_plan_sha256"]
    attachment = stage_model.get_attachment(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY)
    relay = effects.get("relay")
    if type(attachment) is not dict:
        raise ValueError("Current stage Relay owner projection is missing")
    if type(relay) is not dict:
        raise ValueError("Current stage Relay receipt binding is missing")
    relay_checks = {
        "global_plan": global_checked["plan_hash"] == expected_global,
        "owner_global": attachment.get("global_plan_hash") == expected_global,
        "projection_global": projected_checked.get("global_plan_hash") == expected_global,
        "owner_projection": attachment.get("projected_plan_hash") == projected_checked["plan_hash"],
        "segment": attachment.get("segment_index") == segment_index,
        "sampling_binding": relay.get("binding", {}).get("plan_hash") == projected_checked["plan_hash"],
    }
    failures = [name for name, passed in relay_checks.items() if not passed]
    if failures:
        raise ValueError("Current stage Relay owner/plan differs: " + ", ".join(failures))
    if ("long_video_projection" in relay
            and relay["long_video_projection"] != content_identity(attachment)):
        raise ValueError("Current stage Relay receipt projection changed")
    raw_from_stage = _raw_model_from_stage(stage_model, _omit_dormant_sampling=True)
    raw_current = stage_model_identity(raw_model, _omit_dormant_sampling=True)
    if raw_from_stage != raw_current:
        from .results import _identity_difference_path
        raise ValueError("Current sampled MODEL is not derived from the recipe's raw MODEL/LoRA branch at "
                         + str(_identity_difference_path(raw_from_stage, raw_current)))
    if (condition_receipt is None) != (conditioned_latent is None):
        raise ValueError("Current V2 condition provenance needs both receipt and conditioned latent")
    condition_sha = None
    if condition_receipt is not None:
        condition_sha = verify_condition_provenance(condition_receipt, recipe, projected_plan,
            phase, segment_index, guider, conditioned_latent, source_latent)
    payload = {"schema": SCHEMA, "current_recipe_sha256": current_recipe.sha256,
               "segment_index": segment_index, "stage": phase,
               "stage_request_sha256": receipt["request_sha256"],
               "stage_receipt_sha256": receipt["receipt_sha256"],
               "raw_model_sha256": expected_raw["sha256"],
               "projected_plan_sha256": projected_checked["plan_hash"],
               "source": request["source"], "output": receipt["outputs"]["output"],
               "boundary": "One current stage and its live inputs only; no accepted parent, "
                           "candidate save, cross-segment assembly or human approval"}
    if condition_sha is not None:
        payload["condition_receipt_sha256"] = condition_sha
    encoded = canonical(payload)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    attestation = FastH3V2StageAttestation(encoded, digest)
    attestation.verify()
    report = {"schema": SCHEMA, "status": ("current_stage_with_condition_provenance"
                                           if condition_sha is not None else "current_stage_attested"),
              "stage": phase, "segment_index": segment_index,
              "current_recipe_sha256": current_recipe.sha256,
              "attestation_sha256": digest, "boundary": payload["boundary"]}
    return attestation, digest, json.dumps(report, ensure_ascii=False, sort_keys=True)
