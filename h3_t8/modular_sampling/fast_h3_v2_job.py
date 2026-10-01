"""Stable current-recipe identity for explicit FastH3 V2 long-video stages.

The parent candidate SHA authenticates a *previous* execution. This identity
binds the current editable inputs, but it does not by itself attest that either
stage ran or authorize candidate acceptance. A separate stage attestation must
compare real completed StageResult receipts before delivery can use it.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import torch

from .. import long_video
from ..learned_latent_upscale_advanced import learned_upscale_geometry
from ..long_video_dual_identity import content_identity, stage_model_identity
from ..nodes_long_video_dual_model import _component_identity
from ..prompt_relay_advanced import _validate_plan
from .eav import EAVConfig
from .results import canonical, implementation_identity


SCHEMA = "t8.modular-sampling.fasth3-v2-current-recipe.v1"


@dataclass(frozen=True)
class FastH3V2CurrentRecipe:
    payload_json: str
    sha256: str

    def verify(self):
        payload = json.loads(self.payload_json)
        if payload.get("schema") != SCHEMA or canonical(payload) != self.payload_json:
            raise ValueError("Unknown or noncanonical FastH3 V2 current recipe")
        if hashlib.sha256(self.payload_json.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("FastH3 V2 current recipe fingerprint changed")
        return payload


def _identity(value, label):
    if not isinstance(value, dict) or type(value.get("sha256")) is not str:
        raise ValueError(f"{label} did not provide a content-bound model identity")
    digest = value["sha256"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} model identity digest is malformed")
    return value


def _upscale_contract(report_json, geometry):
    report = json.loads(report_json)
    if (report.get("schema_version") != 1
            or report.get("node") != "MiniMaxH3LearnedLatentUpscaleT8Advanced"
            or report.get("status") != "ok" or report.get("geometry") != geometry
            or report.get("audio_preserved") is not True):
        raise ValueError("Current recipe needs the actual completed learned 3D upscale report")
    model = report.get("model")
    if (type(model) is not dict or type(model.get("sha256")) is not str
            or len(model["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in model["sha256"])
            or not str(model.get("name", "")).endswith(".safetensors")
            or model.get("precision") != "fp16"
            or type(model.get("contract")) is not dict):
        raise ValueError("Learned 3D upscale model content identity is missing")
    return {"model_name": model["name"], "model_sha256": model["sha256"],
            "precision": model["precision"], "model_contract": model["contract"],
            "geometry": geometry, "policy": "target_dimensions_honor_dimensions_exp"}


def make_current_recipe(model_pass1, model_pass2, clip, video_vae, audio_vae,
                        first_frame, low_global_plan, high_global_plan,
                        low_eav_config, high_eav_config, upscale_report_json,
                        chain_id, low_width=256, low_height=384,
                        width=512, height=768, total_accepted_frames=192,
                        first_seed=2609152201, continuation_render_frames=90):
    safe_chain = long_video.sanitize_chain_id(chain_id)
    if any(type(item) is not int for item in
           (low_width, low_height, width, height, total_accepted_frames, first_seed,
            continuation_render_frames)):
        raise ValueError("FastH3 V2 recipe geometry, duration and seed must be integers")
    if (low_width < 32 or low_height < 32 or width <= low_width or height <= low_height
            or any(item % 32 for item in (low_width, low_height, width, height))):
        raise ValueError("FastH3 V2 LOW/HIGH geometry must be distinct 32-pixel multiples")
    if total_accepted_frames != 192 or first_seed < 0 or first_seed > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("This current-recipe contract supports only the explicit 192-frame V2 pilot")
    if continuation_render_frames not in (90, 124):
        raise ValueError("Current V2 continuation must select the compact 90 or old fixed 124 render window")
    if (not isinstance(first_frame, torch.Tensor) or first_frame.ndim != 4
            or first_frame.shape[0] != 1 or first_frame.shape[-1] not in (3, 4)
            or not torch.isfinite(first_frame).all()):
        raise ValueError("Current V2 I2VA recipe needs one finite RGB/RGBA first frame")
    if type(low_eav_config) is not EAVConfig or type(high_eav_config) is not EAVConfig:
        raise ValueError("Connect both native external Stage EAV Config nodes")
    low_plan = _validate_plan(low_global_plan)
    high_plan = _validate_plan(high_global_plan)
    if low_plan["frame_count"] < total_accepted_frames or high_plan["frame_count"] < total_accepted_frames:
        raise ValueError("Both global Relay plans must cover the complete accepted timeline")
    geometry = learned_upscale_geometry(low_width // 16, low_height // 16,
        "target_dimensions", 2., 1., width, height, "honor_dimensions_exp", 1.05)
    if (geometry["output_width"], geometry["output_height"]) != (width, height):
        raise ValueError("Learned V2 upscale cannot represent the chosen HIGH geometry")
    upscale = _upscale_contract(upscale_report_json, geometry)
    # Core may replace its dormant model_sampling module on first model load.
    # The V2 stage selects and attests its own sampling patch, so bind the
    # stable raw diffusion/LoRA branch here and compare that same projection
    # after execution. The selected V2 sampler remains bound by StageContext
    # and the live stage MODEL identity.
    low_model = _identity(stage_model_identity(model_pass1, _omit_dormant_sampling=True), "LOW")
    high_model = _identity(stage_model_identity(model_pass2, _omit_dormant_sampling=True), "HIGH")
    window = {"total_accepted_frames": total_accepted_frames,
              "first_render_frames": 124, "continuation_context_frames": 22,
              "first_seed": first_seed, "seed_policy": "increment"}
    # Omit the new optional field for the original 90-frame recipe so saved
    # graphs retain their old payload shape and effective default.
    if continuation_render_frames != 90:
        window["continuation_render_frames"] = continuation_render_frames
    payload = {"schema": SCHEMA, "chain_id": safe_chain,
        "model_pass1": low_model, "model_pass2": high_model,
        "geometry": {"low_width": low_width, "low_height": low_height,
                     "width": width, "height": height},
        "components": {"clip": _component_identity(clip),
                       "video_vae": _component_identity(video_vae),
                       "audio_vae": _component_identity(audio_vae)},
        "first_frame": content_identity(first_frame),
        "relay": {"low_global_plan_sha256": low_plan["plan_hash"],
                  "high_global_plan_sha256": high_plan["plan_hash"],
                  "mode": "apply_exp", "backend": "dense_compat_exp"},
        "eav": {"low": asdict(low_eav_config), "high": asdict(high_eav_config)},
        "learned_upscale": upscale,
        "window": window,
        "conditioning": {"task": "I2VA", "audio_mode": "native",
                         "audio_denoise_strength": 1., "context_audio": "video_and_audio",
                         "first_frame_reuse": "segment0_only",
                         "add_source_as_reference": False},
        "handoff": {"second_audio_source": "legacy_policy",
                    "low_context_source": "accepted_picture_low_context_v1",
                    "high_context_mode": "high_native_mask_ramp_exp"},
        "implementation": implementation_identity()}
    encoded = canonical(payload)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    recipe = FastH3V2CurrentRecipe(encoded, digest)
    recipe.verify()
    model_id = low_model["sha256"][:16] + ":" + high_model["sha256"][:16]
    report = {"schema": SCHEMA, "chain_id": safe_chain,
              "current_recipe_sha256": digest, "model_id": model_id,
              "boundary": "Current editable recipe only; no stage execution, accepted parent, "
                          "candidate save or chain acceptance attested"}
    return recipe, digest, model_id, json.dumps(report, ensure_ascii=False, sort_keys=True)
