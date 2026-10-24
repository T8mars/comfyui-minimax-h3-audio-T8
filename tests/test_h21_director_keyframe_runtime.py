"""Real tiny Core/Encoder D execution, with explicit synthetic text tokens.

The CLIP below is NativeLikeFakeClip's byte/tag contract (text_dim=8), not
Qwen weights. Video latents are produced by an actual Core video Encoder;
DiT and two-step CPU sampler calls are real, random-weight, not media quality.
"""
from copy import deepcopy
import json
import types

import pytest
import torch

from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.multikeyframe_advanced import (
    ACTUAL_FRAME_INDEX, MULTIKEYFRAME_SCHEMA_KEY, append_keyframe_plan,
    build_multikeyframe_conditioning,
)
from h3_audio_t8_pkg.nodes_director_keyframes import MiniMaxH3DirectorKeyframeConditioningEXPT8
from h3_audio_t8_pkg.prompt_relay_advanced import (
    PROMPT_RELAY_BINDING_KEY, PROMPT_RELAY_PAYLOAD_KEY, PROMPT_RELAY_RUNTIME_KEY,
    PROMPT_RELAY_WRAPPER_KEY, _layout_contract, build_prompt_relay_conditioning,
    build_prompt_relay_plan, prompt_relay_model_contract,
)
from test_fast_h3_v2_core_sampler import model as real_tiny_model
from test_h21_control_regions import RealTinyEncoderOwner
from test_prompt_relay_advanced import NativeLikeFakeClip
from helpers import FakeAudioVAE


class CountingNativeLikeFakeClip(NativeLikeFakeClip):
    """Synthetic text only: observe actual top-level fresh visual encoding calls."""
    def __init__(self):
        super().__init__()
        self.tokenize_calls = []
        self.encode_calls = []

    def tokenize(self, prompt, **kwargs):
        tokens = super().tokenize(prompt, **kwargs)
        self.tokenize_calls.append((prompt, kwargs, tokens))
        return tokens

    def encode_from_tokens_scheduled(self, tokens):
        self.encode_calls.append(tokens)
        return super().encode_from_tokens_scheduled(tokens)


@pytest.fixture(scope="module")
def real_encoder():
    return RealTinyEncoderOwner()


@pytest.fixture
def case(monkeypatch, real_encoder):
    # Select the real PyTorch attention kernel on CPU, not a stub output.
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    bare = real_tiny_model()
    real_encoder.calls.clear()
    image = torch.linspace(.05, .95, 32 * 32 * 3).reshape(1, 32, 32, 3)
    clip = CountingNativeLikeFakeClip()
    return types.SimpleNamespace(bare=bare, vae=real_encoder, clip=clip, image=image,
                                 image_before=image.clone(),
                                 options_before=deepcopy(bare.model_options),
                                 weights_before=[value.detach().clone() for value in bare.model.parameters()])


def keyframes(case, count=1, *, nonuniform=False):
    result = None
    for index in range(count):
        result = append_keyframe_plan(result, case.image * (1 - index * .03), "frame", index + 1,
                                      .92 if nonuniform and index == 0 else .999, "stretch", True)
    return result


def args(case, count=1, *, nonuniform=False):
    return dict(model=case.bare, clip=case.clip, video_vae=case.vae, audio_vae=FakeAudioVAE(),
        prompt="A person moves.", width=32, height=32, length=22, task_type="FL2VA",
        audio_mode="native", audio_denoise_strength=.35, add_source_as_reference=False,
        prompt_primary_audio_ordinal=0, strict_prompt_tags=True, ref_image_size="match",
        reference_video_policy="official_2_to_15s", first_frame_noise_aug=.999,
        last_frame_noise_aug=.999, reference_visual_noise_aug=.999,
        first_frame=case.image, last_frame=case.image * .8,
        keyframe_plan=keyframes(case, count, nonuniform=nonuniform))


def relay_plan(events=2):
    return build_prompt_relay_plan("A person moves.", "Raise hand.\nTurn." if events == 2 else "Raise hand.",
        22, "auto_equal", "", "paper_v1", .1, False, False)[0]


def build(case, *, relay=False, count=1, nonuniform=False, execution_mode="apply_exp", events=2):
    inputs = args(case, count, nonuniform=nonuniform)
    if relay:
        return MiniMaxH3DirectorKeyframeConditioningEXPT8.execute(
            **inputs, prompt_relay_plan=relay_plan(events), execution_mode=execution_mode,
            query_chunk_rows=64).result
    return MiniMaxH3DirectorKeyframeConditioningEXPT8.execute(**inputs).result


def actual_payload(model, positive, latent):
    context, metadata = positive[0]
    return model.get_model_object("extra_conds")(
        device=torch.device("cpu"), noise=latent["samples"], cross_attn=context,
        latent_shapes=[list(value.shape) for value in latent["samples"].unbind()],
        **metadata)["minimax_payload"].cond


def fresh_once(case, image_count):
    assert len(case.clip.tokenize_calls) == len(case.clip.encode_calls) == 1
    _, kwargs, tokens = case.clip.tokenize_calls[0]
    assert case.clip.encode_calls[0] is tokens
    assert len(kwargs.get("images", kwargs.get("minimax_ref_items", []))) == image_count
    assert len(case.vae.calls) == image_count
    assert all(tuple(value.shape) == (1, 32, 32, 3) for value in case.vae.calls)


def sample_twice(case, built):
    model, positive, latent = built[:3]
    original_latent = [value.clone() for value in latent["samples"].unbind()]
    original_model_patches = dict(case.bare.object_patches)
    physical_layouts = []

    def observe(_module, _positional, keyword):
        physical_layouts.append(_layout_contract(keyword["minimax_payload"]["layout"]))

    # An observer on the actual nn.Module, not mocked sampler/kernel counts.
    handle = model.model.diffusion_model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        prepared, sampler, sigmas = sampling.setup_dual_clock_sampling(model, latent, 2, 12., 3.)
        guider = BasicGuider.execute(prepared, positive).result[0]
        noise = RandomNoise.execute(21102190).result[0]
        output = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent).result[0]
    finally:
        handle.remove()
    assert len(physical_layouts) == 2
    assert all(torch.isfinite(value).all() for value in output["samples"].unbind())
    assert [tuple(value.shape) for value in output["samples"].unbind()] == [tuple(value.shape) for value in original_latent]
    assert all(torch.equal(before, value) for before, value in zip(original_latent, latent["samples"].unbind(), strict=True))
    assert case.bare.object_patches == original_model_patches
    assert case.bare.model_options == case.options_before
    assert all(torch.equal(before, value) for before, value in
               zip(case.weights_before, case.bare.model.parameters(), strict=True))
    assert torch.equal(case.image, case.image_before)
    assert not torch.cuda.is_initialized()
    return output, physical_layouts


@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_actual_tiny_core_and_encoder_two_nfe_keep_fresh_pair_and_originals(case, relay, nonuniform):
    before = dict(case.bare.object_patches)
    result = build(case, relay=relay, nonuniform=nonuniform)
    assert len(result) == 7
    model, positive, latent = result[:3]
    fresh_once(case, 3)
    payload = actual_payload(model, positive, latent)
    assert len(payload["keyframes"]) == len(payload["cond_video_latents"]) == 3
    assert [item[ACTUAL_FRAME_INDEX] for item in payload["keyframes"]] == [0, 1, 21]
    assert MULTIKEYFRAME_SCHEMA_KEY in payload
    if relay:
        contract = prompt_relay_model_contract(model)
        assert _layout_contract(payload["layout"]) == contract["binding"]["layout_contract"]
        assert positive[0][1]["model_conds"][PROMPT_RELAY_PAYLOAD_KEY].cond == contract["binding_hash"]
    else:
        assert PROMPT_RELAY_BINDING_KEY not in positive[0][1]
        assert model.get_attachment(PROMPT_RELAY_WRAPPER_KEY) is None
        assert PROMPT_RELAY_PAYLOAD_KEY not in positive[0][1]
        assert PROMPT_RELAY_PAYLOAD_KEY not in positive[0][1].get("model_conds", {})
        assert json.loads(result[-1])["per_condition_forward_patch"] is nonuniform
    _, layouts = sample_twice(case, result)
    if relay:
        assert all(item == contract["binding"]["layout_contract"] for item in layouts)
        assert prompt_relay_model_contract(model)["execution_counts"]["completed_forwards"] == 2
        assert PROMPT_RELAY_RUNTIME_KEY not in model.model_options["transformer_options"]
    assert case.bare.object_patches == before == {}
    assert not case.bare.wrappers


@pytest.mark.parametrize("relay", [False, True])
def test_eight_middle_images_are_actually_encoded_and_sampled_not_truncated(case, relay):
    result = build(case, relay=relay, count=8)
    fresh_once(case, 10)
    payload = actual_payload(result[0], result[1], result[2])
    assert len(payload["cond_video_latents"]) == 10
    assert [item[ACTUAL_FRAME_INDEX] for item in payload["keyframes"]] == [0, *range(1, 9), 21]
    assert len(result[1][0][1]["minimax_keyframes"]) == 10
    sample_twice(case, result)


def test_opt_in_details_have_authoritative_tokens_and_default_arity_is_unchanged(case):
    ordinary = build_multikeyframe_conditioning(**args(case))
    assert len(ordinary) == 8
    case.clip.tokenize_calls.clear()
    case.clip.encode_calls.clear()
    case.vae.calls.clear()
    detailed = build_multikeyframe_conditioning(**args(case), return_details=True)
    assert len(detailed) == 9
    details = detailed[-1]
    assert details["tokens"] is case.clip.encode_calls[0]
    assert details["keyframes"] is detailed[1][0][1]["minimax_keyframes"]
    assert details["multikeyframe_layout"] and details["frame_count"] == 22
    assert details["resolved_task"] == "fl2va"
    fresh_once(case, 3)


@pytest.mark.parametrize("events,execution_mode", [(1, "apply_exp"), (2, "report_only")])
def test_relay_passthrough_retains_actual_middle_model_instead_of_erasing_it(case, events, execution_mode):
    result = build(case, relay=True, nonuniform=True, events=events, execution_mode=execution_mode)
    report = json.loads(result[-1])
    assert report["status"] == ("passthrough_single_event" if events == 1 else "report_only")
    assert result[0].get_attachment(PROMPT_RELAY_WRAPPER_KEY) is None
    assert "extra_conds" in result[0].object_patches
    assert "diffusion_model._forward" in result[0].object_patches
    fresh_once(case, 3)
    sample_twice(case, result)


def test_unknown_real_forward_is_preserved_executes_and_does_not_claim_strength_owner(case):
    original = case.bare.get_model_object("diffusion_model._forward")
    calls = []

    def foreign(_self, *positional, **keyword):
        calls.append(1)
        return original(*positional, **keyword)

    selected = types.MethodType(foreign, case.bare.model.diffusion_model)
    case.bare.add_object_patch("diffusion_model._forward", selected)
    result = build(case, nonuniform=True)
    assert result[0].get_model_object("diffusion_model._forward") is selected
    report = json.loads(result[-1])
    assert report["per_condition_forward_requested"] and not report["per_condition_forward_patch"]
    assert any("not installed" in item for item in report["warnings"])
    sample_twice(case, result)
    assert len(calls) == 2 and case.bare.get_model_object("diffusion_model._forward") is selected


def test_real_relay_mispaired_conditioning_still_errors_then_correct_next_request_runs(case):
    first = build(case, relay=True)
    second_inputs = args(case)
    second_plan = build_prompt_relay_plan("Another room.", "Sit.\nStand.", 22,
        "auto_equal", "", "paper_v1", .1, False, False)[0]
    second = MiniMaxH3DirectorKeyframeConditioningEXPT8.execute(**second_inputs,
        prompt_relay_plan=second_plan, execution_mode="apply_exp", query_chunk_rows=64).result
    model, positive, latent = first[:3]
    wrong = (model, second[1], latent, *first[3:])
    with pytest.raises(RuntimeError, match="paired outputs|same node"):
        sample_twice(case, wrong)
    assert not case.bare.object_patches
    sample_twice(case, (model, positive, latent, *first[3:]))


def test_empty_optional_middle_plan_keeps_existing_relay_factory_outputs(case):
    common = args(case, count=0)
    common.pop("prompt")
    common.pop("length")
    for key in ("first_frame_noise_aug", "last_frame_noise_aug", "reference_visual_noise_aug"):
        common.pop(key)
    common.pop("keyframe_plan")
    expected = build_prompt_relay_conditioning(**common, prompt_relay_plan=relay_plan(),
        execution_mode="apply_exp", query_chunk_rows=64)
    actual = build_prompt_relay_conditioning(**common, prompt_relay_plan=relay_plan(),
        execution_mode="apply_exp", query_chunk_rows=64, keyframe_plan=())
    assert actual[4:6] == expected[4:6]
    assert actual[1][0][0].shape == expected[1][0][0].shape
    assert torch.equal(actual[1][0][0], expected[1][0][0])
    assert all(torch.equal(a, b) for a, b in zip(actual[2]["samples"].unbind(), expected[2]["samples"].unbind(), strict=True))
    assert _layout_contract(actual_payload(actual[0], actual[1], actual[2])["layout"]) == prompt_relay_model_contract(actual[0])["binding"]["layout_contract"]
    assert deepcopy(case.bare.object_patches) == {}


@pytest.mark.parametrize("relay", [False, True])
def test_actual_hybrid_reference_rows_keep_native_order_and_same_relay_physical_grid(case, relay):
    inputs = args(case)
    inputs.update(task_type="Hybrid", ref_images={"ref_image_0": case.image * .6})
    result = MiniMaxH3DirectorKeyframeConditioningEXPT8.execute(**inputs,
        **({"prompt_relay_plan": relay_plan(), "execution_mode": "apply_exp", "query_chunk_rows": 64} if relay else {})).result
    fresh_once(case, 4)
    payload = actual_payload(result[0], result[1], result[2])
    assert len(payload["cond_video_latents"]) == 4
    assert len(payload["keyframes"]) == 3
    assert sum(item.get("kind") == "image" for item in payload["refs"]) == 1
    _, layouts = sample_twice(case, result)
    if relay:
        bound = prompt_relay_model_contract(result[0])["binding"]["layout_contract"]
        assert _layout_contract(payload["layout"]) == bound
        assert all(item == bound for item in layouts)


def test_selected_legacy_position_repair_branch_binds_the_actual_repaired_grid(case, monkeypatch):
    from h3_audio_t8_pkg import multikeyframe_advanced

    # Explicit branch instrumentation, not a claim of running an old Core:
    # actual current Core constructor, Encoder and sampler remain untouched.
    monkeypatch.setattr(multikeyframe_advanced, "native_middle_keyframe_support", lambda: False)
    result = build(case, relay=True)
    payload = actual_payload(result[0], result[1], result[2])
    bound = prompt_relay_model_contract(result[0])["binding"]["layout_contract"]
    assert payload["keyframes"][1]["resolved_frame_index"] == 0
    assert payload["keyframes"][1][ACTUAL_FRAME_INDEX] == 1
    assert _layout_contract(payload["layout"]) == bound
    _, layouts = sample_twice(case, result)
    assert all(item == bound for item in layouts)
