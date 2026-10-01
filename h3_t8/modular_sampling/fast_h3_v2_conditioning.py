"""Opt-in provenance for the exact native Long Video Relay conditioning call.

This wrapper calls the original builder once and adds an inert receipt. It does
not alter conditioning, the source latent, Relay, or the sampler's AV clocks.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import uuid

import comfy.sampler_helpers

from .. import core, long_video, nodes_long_video_dual_model, prompt_relay_advanced
from .. import prompt_relay_long_video_advanced
from ..long_video_dual_identity import content_identity
from ..nodes_long_video_dual_model import _component_identity
from ..prompt_relay_advanced import _validate_plan
from ..prompt_relay_long_video_advanced import build_prompt_relay_long_video_conditioning
from .results import _input_identity, canonical, request_conditions


SCHEMA = "t8.modular-sampling.fasth3-v2-condition-provenance.v1"
SETTING_DEFAULTS = {
    "first_frame_reuse": "segment0_only", "persistent_identity_strategy": "single_reference",
    "persistent_identity_interval": 1,
}
SETTINGS = ("segment_index", "context_frames", "context_audio", "width", "height",
            "length", "task_type", "audio_mode", "audio_denoise_strength",
            "add_source_as_reference", "prompt_primary_audio_ordinal", "strict_prompt_tags",
            "ref_image_size", "reference_video_policy", "execution_mode",
            "query_chunk_rows", "first_frame_reuse", "persistent_identity_strategy",
            "persistent_identity_interval")
OTHER_MEDIA = ("drive_audio", "final_audio", "last_frame", "ref_images", "ref_videos",
               "ref_video_audios", "ref_audios", "persistent_identity_image", "semantic_bridge")


def implementation_sha256():
    sources = {"v2_condition_provenance": __file__,
               "long_video": long_video.__file__,
               "relay_long_video": prompt_relay_long_video_advanced.__file__,
               "relay_plan": prompt_relay_advanced.__file__,
               "component_identity": nodes_long_video_dual_model.__file__,
               "core_geometry": core.__file__,
               "core_condition_convert": comfy.sampler_helpers.__file__}
    files = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
             for name, path in sources.items()}
    return hashlib.sha256(canonical(files).encode("utf-8")).hexdigest()


def _other_media_identity(kwargs):
    values = {}
    for name in OTHER_MEDIA:
        value = kwargs.get(name)
        if name.startswith("ref_") and type(value) in (dict, list, tuple) and not value:
            value = None  # Core Autogrow sends an empty container for disconnected slots.
        values[name] = value
    return content_identity(values)


@dataclass(frozen=True)
class FastH3V2ConditionReceipt:
    payload_json: str
    sha256: str

    def verify(self):
        payload = json.loads(self.payload_json)
        if payload.get("schema") != SCHEMA or canonical(payload) != self.payload_json:
            raise ValueError("Unknown or noncanonical V2 conditioning provenance")
        if hashlib.sha256(self.payload_json.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("V2 conditioning provenance fingerprint changed")
        return payload


def condition_with_provenance(**kwargs):
    # Capture the actual arguments of this call, not separately editable audit
    # sockets. Unknown component adapters can still execute the original node,
    # but their receipt never upgrades to portable provenance.
    try:
        component_ids = {name: _component_identity(kwargs[name])
                         for name in ("clip", "video_vae", "audio_vae")}
        inputs = {
            "components": component_ids,
            "first_frame": content_identity(kwargs.get("first_frame")),
            "context": content_identity(kwargs["context"]),
            "projected_plan_sha256": _validate_plan(kwargs["prompt_relay_plan"])["plan_hash"],
            "settings": {name: kwargs.get(name, SETTING_DEFAULTS.get(name)) for name in SETTINGS},
            "other_media": _other_media_identity(kwargs),
        }
        portable = all(item.get("portable_cache_reuse") is not False
                       for item in component_ids.values())
    except Exception as error:
        # Identity failures are never admission failures for the original
        # conditioner. Sampling may proceed, but portable audit cannot.
        inputs = {"unverified": f"{type(error).__name__}: {error}",
                  "execution_nonce": uuid.uuid4().hex}
        portable = False
    result = build_prompt_relay_long_video_conditioning(**kwargs)
    model, positive, latent = result[:3]
    del model  # The separate stage audit authenticates the executed MODEL.
    try:
        converted = comfy.sampler_helpers.convert_cond(positive)
        outputs = {"positive": request_conditions({"positive": converted}),
                   "conditioned_latent": _input_identity(latent),
                   "report_sha256": hashlib.sha256(result[6].encode("utf-8")).hexdigest()}
    except Exception as error:
        outputs = {"unverified": f"{type(error).__name__}: {error}",
                   "execution_nonce": uuid.uuid4().hex}
        portable = False
    payload = {
        "schema": SCHEMA, "implementation_sha256": implementation_sha256(),
        "portable_identity": portable, "inputs": inputs, "outputs": outputs,
        "boundary": "One conditioning call only; no HIGH handoff, accepted parent or candidate acceptance",
    }
    encoded = canonical(payload)
    receipt = FastH3V2ConditionReceipt(encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest())
    receipt.verify()
    return (*result, receipt)


def verify_condition_provenance(receipt, recipe, projected_plan, phase, segment_index,
                                guider, conditioned_latent, source_latent):
    if type(receipt) is not FastH3V2ConditionReceipt:
        raise ValueError("Current V2 condition proof needs the typed call receipt")
    payload = receipt.verify()
    if (payload.get("portable_identity") is not True
            or payload.get("implementation_sha256") != implementation_sha256()):
        raise ValueError("Current V2 condition call has no portable matching implementation")
    inputs, outputs = payload["inputs"], payload["outputs"]
    if inputs["components"] != recipe["components"]:
        raise ValueError("Current V2 CLIP/VAEs differ from the actual condition call")
    expected_frame = recipe["first_frame"] if segment_index == 0 else content_identity(None)
    if inputs["first_frame"] != expected_frame:
        raise ValueError("Current V2 first-frame use differs from the editable recipe")
    if inputs["projected_plan_sha256"] != _validate_plan(projected_plan)["plan_hash"]:
        raise ValueError("Current V2 condition call used a different projected Relay plan")
    width_key, height_key = (("low_width", "low_height") if phase == "low_0_4"
                             else ("width", "height"))
    expected_settings = {
        "segment_index": segment_index, "context_frames": 0 if segment_index == 0 else 22,
        "context_audio": "video_and_audio",
        "width": recipe["geometry"][width_key], "height": recipe["geometry"][height_key],
        "length": (124 if segment_index == 0 else
                   recipe["window"].get("continuation_render_frames", 90)),
        "task_type": "I2VA", "audio_mode": "native", "audio_denoise_strength": 1.,
        "add_source_as_reference": False, "prompt_primary_audio_ordinal": 0,
        "strict_prompt_tags": True, "ref_image_size": "match",
        "reference_video_policy": "official_2_to_15s", "execution_mode": "apply_exp",
        "query_chunk_rows": 256, "first_frame_reuse": "segment0_only",
        "persistent_identity_strategy": "single_reference", "persistent_identity_interval": 1,
    }
    if inputs["settings"] != expected_settings:
        raise ValueError("Current V2 condition call settings differ from the pilot recipe")
    if inputs["other_media"] != _other_media_identity({}):
        raise ValueError("Current V2 condition call has media outside the pilot recipe")
    if outputs["conditioned_latent"] != _input_identity(conditioned_latent):
        raise ValueError("Current V2 conditioned latent differs from its call receipt")
    if phase == "low_0_4" and _input_identity(source_latent) != outputs["conditioned_latent"]:
        raise ValueError("Current V2 LOW sampler did not use its authenticated conditioning latent")
    if outputs["positive"] != request_conditions(guider.original_conds):
        raise ValueError("Current V2 Guider did not use the authenticated condition output")
    return receipt.sha256
