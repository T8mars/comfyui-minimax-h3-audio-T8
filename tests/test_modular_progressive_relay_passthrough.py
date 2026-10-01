"""Original Relay's zero/single/report-only output must remain a true no-op.

Text/VAE encoding uses the existing explicitly labelled CPU doubles; actual
conditioning preparation and native tiny stage sampling are not substituted.
"""
from copy import deepcopy
from dataclasses import replace
import json

import comfy.conds
import pytest
import torch

from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.modular_sampling import progressive as stages
from h3_audio_t8_pkg.modular_sampling import progressive_effects as effects
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
from test_modular_progressive_stages import inputs, low_only, restart
from test_modular_progressive_effects import sample
from test_prompt_relay_advanced import NativeLikeFakeClip, FakeVideoVAE, FakeAudioVAE, _plan
from test_progressive_sampling_runtime import tiny_model
import test_progressive_sampling_runtime as fixtures

stub_lifter = fixtures.stub_lifter


def original_passthrough(task, event_count, execution_mode):
    plan, *_ = _plan(global_prompt="A person walks.", local_prompts="\n".join(
        f"Event {i}." for i in range(event_count)), length=22 if event_count > 1 else 5)
    model = tiny_model()
    extras = {"first_frame": torch.zeros(1, 64, 128, 3)} if task == "i2va" else {}
    result = relay.build_prompt_relay_conditioning(model=model, clip=NativeLikeFakeClip(),
        video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(), prompt_relay_plan=plan,
        width=128, height=64, task_type=task.upper(), audio_mode="native", audio_denoise_strength=1.,
        add_source_as_reference=False, prompt_primary_audio_ordinal=0, strict_prompt_tags=True,
        ref_image_size="match", reference_video_policy="official_2_to_15s",
        execution_mode=execution_mode, query_chunk_rows=64, **extras)
    returned, positive, latent, _, _, _, report = result
    assert returned is model and not json.loads(report)["attention_patch_installed"]
    case = inputs(source=latent, model=model)
    case["task"] = task
    case["plan"] = replace(case["plan"], task=task)
    case["positive"] = positive
    case["lp"], case["hp"] = stages.legacy.prepare_stage_conditioning(positive, case["plan"], positive=True)
    return case, positive


@pytest.mark.parametrize("task", ["t2va", "i2va"])
@pytest.mark.parametrize("phase", ["low", "high"])
@pytest.mark.parametrize("stage_mode", ["disabled", "apply_exp"])
@pytest.mark.parametrize("event_count,source_mode", [(0, "apply_exp"), (1, "apply_exp"), (2, "report_only")])
def test_real_original_conditioning_passthrough_equals_plain_stage(stub_lifter, task, phase, stage_mode, event_count, source_mode):
    case, positive = original_passthrough(task, event_count, source_mode)
    before = stages.snapshot(positive)
    selected, pos, neg = effects.apply_relay(case["model"], positive, positive, case["plan"], phase, mode=stage_mode)
    assert selected is case["model"] and effects.owner(selected) is None
    index = int(phase == "high")
    expected_pos = stages.legacy.prepare_stage_conditioning(positive, case["plan"], positive=True)[index]
    expected_neg = stages.legacy.prepare_stage_conditioning(positive, case["plan"], positive=False)[index]
    assert stages.snapshot((pos, neg)) == stages.snapshot((expected_pos, expected_neg))
    assert stages.snapshot(positive) == before
    state = restart(case, low_only(case)) if phase == "high" else None
    actual = sample(case, phase, selected, pos, neg, state=state)
    expected = sample(case, phase, selected, expected_pos, expected_neg, state=state)
    a = actual.tensors.values() if phase == "low" else actual.output["samples"].unbind()
    b = expected.tensors.values() if phase == "low" else expected.output["samples"].unbind()
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert effects.audit(actual, phase)["status"] == "no_external_stage_effects"


@pytest.mark.parametrize("marker", ["positive_binding", "negative_binding", "payload", "wrapper", "selector"])
def test_missing_model_contract_with_residual_relay_marker_is_not_a_passthrough(marker):
    case = inputs()
    positive, negative = deepcopy(case["positive"]), deepcopy(case["negative"])
    if marker == "positive_binding":
        positive[0][1][relay.PROMPT_RELAY_BINDING_KEY] = None
    elif marker == "negative_binding":
        negative[0][1][relay.PROMPT_RELAY_BINDING_KEY] = {"binding_hash": "missing-model"}
    elif marker == "payload":
        negative[0][1]["model_conds"] = {relay.PROMPT_RELAY_PAYLOAD_KEY: comfy.conds.CONDConstant("stale")}
    elif marker == "wrapper":
        case["model"].add_wrapper_with_key("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY,
                                          lambda executor, *a, **k: executor(*a, **k))
    else:
        def stale_selector(executor, *a, **k):
            return executor(*a, **k)
        stale_selector._t8_prompt_relay_binding_hash = "stale"
        case["model"].model_options["transformer_options"]["optimized_attention_override"] = stale_selector
    with pytest.raises(ValueError, match="[Uu]npaired"):
        effects.apply_relay(case["model"], positive, negative, case["plan"], "low")


@pytest.mark.parametrize("event_count,source_mode", [(0, "apply_exp"), (1, "apply_exp"), (2, "report_only")])
def test_passthrough_can_still_enable_independent_eav(event_count, source_mode):
    case, positive = original_passthrough("t2va", event_count, source_mode)
    model, pos, neg = effects.apply_relay(case["model"], positive, positive, case["plan"], "low")
    model = effects.apply_eav(model, EAVConfig("apply_exp", .2, 0., 1.), case["plan"], "low", case["low_source"])
    result = sample(case, "low", model, pos, neg)
    report = effects.audit(result, "low")
    assert "relay" not in report and report["eav"]["model_forward_count"] == 2


def test_unrelated_user_wrapper_and_lora_are_not_stripped_by_passthrough():
    case = inputs()
    calls = []
    def user(executor, *args, **kwargs):
        calls.append(True)
        return executor(*args, **kwargs)
    case["model"].add_wrapper_with_key("apply_model", "user-retained", user)
    key, weight = next(iter(dict(case["model"].model.named_parameters()).items()))
    case["model"].add_patches({key: ("diff", (torch.full_like(weight, .01),))})
    model, pos, neg = effects.apply_relay(case["model"], case["positive"], case["negative"], case["plan"], "low")
    assert model is case["model"] and len(model.patches[key]) == 1
    result = sample(case, "low", model, pos, neg)
    assert len(calls) == 2 and effects.audit(result, "low")["status"] == "no_external_stage_effects"
