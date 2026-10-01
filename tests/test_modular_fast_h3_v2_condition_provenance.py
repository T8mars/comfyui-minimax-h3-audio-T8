"""An opt-in V2 conditioner binds actual component inputs to sampler conditions."""

from dataclasses import replace
import hashlib
from types import SimpleNamespace

import comfy.sampler_helpers
import pytest
import torch

from h3_audio_t8_pkg.long_video_dual_identity import content_identity
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_conditioning as origin
from h3_audio_t8_pkg.modular_sampling import node_classes
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2_job_nodes import (
    CONDITION_RECEIPT_TYPE, MiniMaxH3FastH3V2ConditionProvenanceEXPT8,
)
from h3_audio_t8_pkg.prompt_relay_advanced import build_prompt_relay_plan


def _hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(origin, "_component_identity", lambda item: {"sha256": _hash(item)})
    plan = build_prompt_relay_plan("One scene.", "She walks.\nShe stops.", 193,
                                   "auto_equal", "", "paper_v1", .1, False, False)[0]
    positive = [[torch.ones(1, 2, 3), {"model_conds": {}}]]
    latent = {"samples": torch.zeros(1, 2, 3)}
    calls = []

    def original(**kwargs):
        calls.append(kwargs)
        return ("original-model", positive, latent, None, "prompt", "{}", "{\"status\":\"ok\"}")

    monkeypatch.setattr(origin, "build_prompt_relay_long_video_conditioning", original)
    first = torch.zeros(1, 32, 32, 3)
    kwargs = dict(model="raw", clip="clip", video_vae="video", audio_vae="audio",
                  context={"video": None, "audio": None}, prompt_relay_plan=plan,
                  segment_index=0, context_frames=0, context_audio="video_and_audio",
                  width=256, height=384, length=124, task_type="I2VA", audio_mode="native",
                  audio_denoise_strength=1., add_source_as_reference=False,
                  prompt_primary_audio_ordinal=0, strict_prompt_tags=True,
                  ref_image_size="match", reference_video_policy="official_2_to_15s",
                  execution_mode="apply_exp", query_chunk_rows=256, first_frame=first,
                  first_frame_reuse="segment0_only")
    recipe = {"components": {name: {"sha256": _hash(value)}
                             for name, value in (("clip", "clip"), ("video_vae", "video"),
                                                 ("audio_vae", "audio"))},
              "first_frame": content_identity(first),
              "geometry": {"low_width": 256, "low_height": 384,
                           "width": 512, "height": 768}}
    guider = SimpleNamespace(original_conds={"positive": comfy.sampler_helpers.convert_cond(positive)})
    return kwargs, recipe, guider, latent, calls


def test_opt_in_condition_call_proves_actual_inputs_and_guider(case):
    kwargs, recipe, guider, latent, calls = case
    result = origin.condition_with_provenance(**kwargs)
    receipt = result[7]
    assert len(calls) == 1
    assert result[0] == "original-model" and result[2] is latent
    assert result[3:7] == (None, "prompt", "{}", "{\"status\":\"ok\"}")
    assert receipt.verify()["portable_identity"] is True
    assert origin.verify_condition_provenance(
        receipt, recipe, kwargs["prompt_relay_plan"], "low_0_4", 0,
        guider, latent, latent) == receipt.sha256
    assert MiniMaxH3FastH3V2ConditionProvenanceEXPT8 in node_classes()
    assert MiniMaxH3FastH3V2ConditionProvenanceEXPT8.GET_NODE_INFO_V1()["output"][-1] == CONDITION_RECEIPT_TYPE


@pytest.mark.parametrize("change,match", [
    ({"clip": "other-clip"}, "CLIP/VAEs"),
    ({"first_frame": torch.ones(1, 32, 32, 3)}, "first-frame"),
    ({"audio_denoise_strength": .5}, "settings"),
    ({"final_audio": {"waveform": torch.zeros(1)}}, "media outside"),
])
def test_provenance_rejects_recipe_divergence(case, change, match):
    kwargs, recipe, guider, latent, _ = case
    receipt = origin.condition_with_provenance(**{**kwargs, **change})[7]
    with pytest.raises(ValueError, match=match):
        origin.verify_condition_provenance(receipt, recipe, kwargs["prompt_relay_plan"],
                                           "low_0_4", 0, guider, latent, latent)


def test_provenance_rejects_changed_condition_or_source_and_tampered_receipt(case):
    kwargs, recipe, guider, latent, _ = case
    receipt = origin.condition_with_provenance(**kwargs)[7]
    wrong_guider = SimpleNamespace(original_conds={"positive": comfy.sampler_helpers.convert_cond(
        [[torch.zeros(1, 2, 3), {"model_conds": {}}]])})
    with pytest.raises(ValueError, match="Guider"):
        origin.verify_condition_provenance(receipt, recipe, kwargs["prompt_relay_plan"],
                                           "low_0_4", 0, wrong_guider, latent, latent)
    with pytest.raises(ValueError, match="LOW sampler"):
        origin.verify_condition_provenance(receipt, recipe, kwargs["prompt_relay_plan"],
                                           "low_0_4", 0, guider, latent,
                                           {"samples": torch.ones(1, 2, 3)})
    with pytest.raises(ValueError, match="fingerprint changed"):
        replace(receipt, payload_json=receipt.payload_json.replace("One", "Two")).verify()


def test_unverified_component_still_calls_original_but_receipt_cannot_attest(case, monkeypatch):
    kwargs, recipe, guider, latent, calls = case
    monkeypatch.setattr(origin, "_component_identity",
                        lambda _: (_ for _ in ()).throw(ValueError("foreign component")))
    receipt = origin.condition_with_provenance(**kwargs)[7]
    assert len(calls) == 1 and receipt.verify()["portable_identity"] is False
    with pytest.raises(ValueError, match="no portable"):
        origin.verify_condition_provenance(receipt, recipe, kwargs["prompt_relay_plan"],
                                           "low_0_4", 0, guider, latent, latent)


def test_condition_receipt_rejects_changed_implementation(case, monkeypatch):
    kwargs, recipe, guider, latent, _ = case
    receipt = origin.condition_with_provenance(**kwargs)[7]
    monkeypatch.setattr(origin, "implementation_sha256", lambda: "0" * 64)
    with pytest.raises(ValueError, match="matching implementation"):
        origin.verify_condition_provenance(receipt, recipe, kwargs["prompt_relay_plan"],
                                           "low_0_4", 0, guider, latent, latent)
