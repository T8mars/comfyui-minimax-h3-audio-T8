"""Read-only provenance check for the V2 LOW x0 -> learned lift -> HIGH port.

This audit replays only the deterministic reconcile/prefix operations. It does
not run a second upscaler or sampler, and never accepts a candidate.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import comfy.sampler_helpers

from .. import learned_latent_upscale_advanced, long_video
from ..long_video_dual_identity import content_identity
from ..learned_latent_upscale_advanced import reconcile_two_pass_h3_latent
from .continuation import ContinuationContexts
from .fast_h3_v2_conditioning import FastH3V2ConditionReceipt
from .fast_h3_v2_continuation import lock_accepted_high_prefix, reconcile_accepted_high
from .fast_h3_v2_job import FastH3V2CurrentRecipe, _upscale_contract
from .fast_h3_v2_stage_attest import FastH3V2StageAttestation
from .fast_h3_v2_upscale import FastH3V2UpscaleReceipt, implementation_sha256 as upscale_implementation
from .results import StageResult, _input_identity, canonical, request_conditions


SCHEMA = "t8.modular-sampling.fasth3-v2-current-handoff.v1"


def implementation_sha256():
    sources = {"handoff": __file__,
               "reconcile": learned_latent_upscale_advanced.__file__,
               "context": long_video.__file__}
    files = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
             for name, path in sources.items()}
    return hashlib.sha256(canonical(files).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FastH3V2CurrentHandoff:
    payload_json: str
    sha256: str

    def verify(self):
        payload = json.loads(self.payload_json)
        if payload.get("schema") != SCHEMA or canonical(payload) != self.payload_json:
            raise ValueError("Unknown or noncanonical V2 current handoff")
        if hashlib.sha256(self.payload_json.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("V2 current handoff fingerprint changed")
        if payload.get("implementation_sha256") != implementation_sha256():
            raise ValueError("V2 current handoff implementation changed")
        return payload


def attest_current_handoff(current_recipe, low_result, high_result,
                           low_attestation, high_attestation,
                           low_condition_receipt, high_condition_receipt,
                           upscale_receipt, upscaled_latent, upscale_report_json,
                           high_template, high_positive, high_source_latent,
                           segment_index, contexts=None):
    if type(current_recipe) is not FastH3V2CurrentRecipe:
        raise ValueError("HIGH handoff requires the typed current recipe")
    recipe = current_recipe.verify()
    if type(segment_index) is not int or segment_index not in (0, 1):
        raise ValueError("HIGH handoff supports only segment 0 or 1")
    if (type(low_result) is not StageResult or type(high_result) is not StageResult
            or type(low_attestation) is not FastH3V2StageAttestation
            or type(high_attestation) is not FastH3V2StageAttestation):
        raise ValueError("HIGH handoff requires both completed typed stage attestations")
    stage_results = (low_result.verify(), high_result.verify())
    attestations = (low_attestation.verify(), high_attestation.verify())
    conditions = []
    for receipt in (low_condition_receipt, high_condition_receipt):
        if type(receipt) is not FastH3V2ConditionReceipt:
            raise ValueError("HIGH handoff needs both actual conditioning receipts")
        payload = receipt.verify()
        if payload.get("portable_identity") is not True:
            raise ValueError("HIGH handoff conditioning identity is not portable")
        conditions.append(payload)
    for result, attestation, condition_receipt, stage in zip(
            stage_results, attestations, (low_condition_receipt, high_condition_receipt),
            ("low_0_4", "high_4_8"), strict=True):
        if (result.get("verified_recipe_completion") is not True
                or result.get("portable_identity") is not True
                or attestation.get("current_recipe_sha256") != current_recipe.sha256
                or attestation.get("segment_index") != segment_index
                or attestation.get("stage") != stage
                or attestation.get("stage_request_sha256") != result["request_sha256"]
                or attestation.get("stage_receipt_sha256") != result["receipt_sha256"]
                or attestation.get("condition_receipt_sha256") != condition_receipt.sha256):
            raise ValueError("HIGH handoff stages do not share this completed current recipe")
    if (type(upscale_receipt) is not FastH3V2UpscaleReceipt
            or type(upscale_report_json) is not str):
        raise ValueError("HIGH handoff needs the actual learned-upscale receipt and report")
    upscale = upscale_receipt.verify()
    if (upscale.get("portable_identity") is not True
            or upscale.get("implementation_sha256") != upscale_implementation()
            or upscale.get("input") != stage_results[0]["outputs"]["denoised_output"]
            or upscale.get("output") != _input_identity(upscaled_latent)
            or upscale.get("report_sha256") != hashlib.sha256(upscale_report_json.encode()).hexdigest()
            or (upscale.get("width"), upscale.get("height")) !=
               (recipe["geometry"]["width"], recipe["geometry"]["height"])):
        raise ValueError("Learned 3D call is not bound to completed LOW x0 and HIGH geometry")
    expected_settings = {"model_name": recipe["learned_upscale"]["model_name"],
                         "size_mode": "target_dimensions", "scale_by": 2.0,
                         "target_megapixels": 0.46,
                         "target_width": recipe["geometry"]["width"],
                         "target_height": recipe["geometry"]["height"],
                         "aspect_policy": "honor_dimensions_exp", "max_anisotropy": 1.05,
                         "precision": "fp16", "release_policy": "offload_after"}
    if upscale.get("settings") != expected_settings:
        raise ValueError("Learned 3D execution settings differ from current V2 pilot")
    if _upscale_contract(upscale_report_json, recipe["learned_upscale"]["geometry"]) != recipe["learned_upscale"]:
        raise ValueError("Learned 3D model/report differs from current recipe")
    if conditions[1]["outputs"]["conditioned_latent"] != _input_identity(high_template):
        raise ValueError("HIGH template is not the actual authenticated condition output")
    if conditions[1]["outputs"]["positive"] != request_conditions({
            "positive": comfy.sampler_helpers.convert_cond(high_positive)}):
        raise ValueError("HIGH reconcile positive is not the actual condition output")
    accepted_source_sha = None
    if segment_index == 0:
        if contexts is not None:
            raise ValueError("First V2 window cannot claim an accepted parent")
        empty_context = content_identity(long_video._empty_context(recipe["chain_id"], 0))
        if (conditions[0]["inputs"]["context"] != empty_context
                or conditions[1]["inputs"]["context"] != empty_context):
            raise ValueError("First V2 window must use the same empty segment-0 context")
        expected_source, selected_positive, _ = reconcile_two_pass_h3_latent(
            upscaled_latent, high_template, high_positive, "first_pass",
            second_pass_audio_source="legacy_policy", second_pass_audio_strength=0.0)
    else:
        if type(contexts) is not ContinuationContexts:
            raise ValueError("Continuation HIGH handoff requires the accepted-parent contexts")
        descriptor = contexts.verify()
        request = contexts.source.binding["request"]
        if (request["segment_index"] != segment_index
                or request["chain_id"] != recipe["chain_id"]
                or conditions[0]["inputs"]["context"] != content_identity(contexts.low)
                or conditions[1]["inputs"]["context"] != content_identity(contexts.high)):
            raise ValueError("Continuation condition calls differ from the accepted parent contexts")
        reconciled, selected_positive, _ = reconcile_accepted_high(
            contexts, upscaled_latent, high_template, high_positive)
        expected_source, _ = lock_accepted_high_prefix(
            contexts, reconciled, mode="high_native_mask_ramp_exp")
        contexts.verify()
        accepted_source_sha = descriptor["source_sha256"]
    if (_input_identity(expected_source) != _input_identity(high_source_latent)
            or stage_results[1]["request"]["source"] != _input_identity(high_source_latent)):
        raise ValueError("HIGH sampler source differs from deterministic learned handoff")
    if request_conditions({"positive": comfy.sampler_helpers.convert_cond(selected_positive)}) != conditions[1]["outputs"]["positive"]:
        raise ValueError("HIGH handoff changed the authenticated positive conditioning")
    payload = {"schema": SCHEMA, "implementation_sha256": implementation_sha256(),
               "current_recipe_sha256": current_recipe.sha256, "segment_index": segment_index,
               "low_stage_attestation_sha256": low_attestation.sha256,
               "high_stage_attestation_sha256": high_attestation.sha256,
               "high_stage_receipt_sha256": stage_results[1]["receipt_sha256"],
               "upscale_receipt_sha256": upscale_receipt.sha256,
               "low_x0": upscale["input"], "upscaled": upscale["output"],
               "high_output": stage_results[1]["outputs"]["output"],
               "high_source": stage_results[1]["request"]["source"],
               "accepted_source_sha256": accepted_source_sha,
               "boundary": "Completed LOW/upscale/HIGH input handoff only; no candidate save, acceptance or assembly"}
    encoded = canonical(payload)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    receipt = FastH3V2CurrentHandoff(encoded, digest)
    receipt.verify()
    report = {"schema": SCHEMA, "status": "current_handoff_attested",
              "segment_index": segment_index, "handoff_sha256": digest,
              "boundary": payload["boundary"]}
    return receipt, digest, json.dumps(report, ensure_ascii=False, sort_keys=True)
