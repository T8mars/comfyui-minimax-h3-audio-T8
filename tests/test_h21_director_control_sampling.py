"""Real CPU Director source/Apply/Audit consumption, not trained media quality.

DiT, video Encoder, official/compat ControlDiT towers and samplers are real.
Text8 embeddings, tiny HyperFlow adapter weights and FakeAudioVAE are explicit
synthetic fixtures. No model loader, residual, sampler or kernel is stubbed.
"""
import json
from pathlib import Path
import uuid

import comfy.ops
import comfy.sample
from comfy.ldm.minimax.controlnet import MiniMaxH3FunControl
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from comfy.model_patcher import ModelPatcher
from comfy.nested_tensor import NestedTensor
from comfy.patcher_extension import WrappersMP
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
import pytest
import torch

from h3_audio_t8_pkg import h3_fun_control_advanced as fun, sampling
from h3_audio_t8_pkg import hyperflow_runtime_advanced as hf_runtime
from h3_audio_t8_pkg.control_region_runtime import REGION_ATTACHMENT
from h3_audio_t8_pkg.control_regions import tensor_sha
from h3_audio_t8_pkg.director_controls import normalize_controls
from h3_audio_t8_pkg.modular_sampling import hyperflow as separate
from h3_audio_t8_pkg.modular_sampling import hyperflow_effects as effects
from h3_audio_t8_pkg.modular_sampling import hyperflow_fresh as fresh
from h3_audio_t8_pkg.modular_sampling import hyperflow_storage as storage
from h3_audio_t8_pkg.multikeyframe_advanced import append_keyframe_plan
from h3_audio_t8_pkg.nodes_control_region import (
    MiniMaxH3ControlRegionAuditEXPT8, MiniMaxH3FunControlRegionApplyEXPT8,
)
from h3_audio_t8_pkg.nodes_director_controls import (
    MiniMaxH3DirectorControlHeadAuditEXPT8, MiniMaxH3DirectorControlSourceEXPT8,
)
from h3_audio_t8_pkg.nodes_director_keyframes import MiniMaxH3DirectorKeyframeConditioningEXPT8
from h3_audio_t8_pkg.prompt_relay_advanced import build_prompt_relay_plan
from h3_audio_t8_pkg.progressive_sampling_runtime import _restore_stage_objects
from helpers import FakeAudioVAE
from test_fast_h3_v2_core_sampler import model as ordinary_model
from test_h21_control_regions import RealTinyEncoderOwner, initialized, source_frames
from test_h21_director_keyframe_runtime import CountingNativeLikeFakeClip
from test_hyperflow_advanced import _tiny_native_model, _tiny_weights
from test_progressive_sampling_runtime import tiny_model


@pytest.fixture(scope="module")
def encoder():
    return RealTinyEncoderOwner()


def conditioned(base, encoder, *, relay, width=32, weights=None):
    if weights is not None:
        base = fresh.install(base, weights)[0]
    image = torch.linspace(.05, .95, 32 * 32 * 3).reshape(1, 32, 32, 3)
    plan = append_keyframe_plan(None, image * .9, "frame", 1, .999, "stretch", True)
    inputs = dict(model=base, clip=CountingNativeLikeFakeClip(), video_vae=encoder,
        audio_vae=FakeAudioVAE(), prompt="A person moves.", width=width, height=width,
        length=22, task_type="FL2VA", audio_mode="native", audio_denoise_strength=.35,
        add_source_as_reference=False, prompt_primary_audio_ordinal=0, strict_prompt_tags=True,
        ref_image_size="match", reference_video_policy="official_2_to_15s",
        first_frame_noise_aug=.999, last_frame_noise_aug=.999, reference_visual_noise_aug=.999,
        first_frame=image, last_frame=image * .8, keyframe_plan=plan)
    if relay:
        inputs.update(prompt_relay_plan=build_prompt_relay_plan("A person moves.",
            "Raise hand.\nTurn.", 22, "auto_equal", "", "paper_v1", .1, False, False)[0],
            execution_mode="apply_exp", query_chunk_rows=64)
    return MiniMaxH3DirectorKeyframeConditioningEXPT8.execute(**inputs).result


def bundle(backend, *, hyperflow, seed):
    settings = dict(control_in_dim=49, injection_layers=(0,),
        hidden_size=8 if hyperflow else 24, num_attention_heads=2 if hyperflow else 3,
        attention_head_dim=8 if hyperflow else 128, ffn_hidden_size=16 if hyperflow else 32,
        time_embed_dim=4 if hyperflow else 24, dtype=torch.float32, device="cpu",
        operations=comfy.ops.disable_weight_init, use_adaln_curves=False)
    cls = MiniMaxH3FunControl if backend == "official_model_patch" else fun._MiniMaxH3FunControl
    tower = initialized(cls(**settings), seed)
    patcher = ModelPatcher(tower, torch.device("cpu"), torch.device("cpu"))
    holder = patcher if backend == "official_model_patch" else {"model": tower, "patcher": patcher}
    return fun.H3FunControlBundle(backend, holder, f"tiny-{seed}.safetensors", "", {}), tower


def controlled(built, encoder, backend, *, stage="SINGLE", hyperflow=False, shared_bundle=False):
    model, positive, latent = built[:3]
    width, height = int(latent["samples"].unbind()[0].shape[-1]) * 16, int(latent["samples"].unbind()[0].shape[-2]) * 16
    raw = source_frames(1, width=96, height=64)
    before = raw.clone()
    states, observations, handles = [], [], []
    shared = None
    for index in range(2):
        spec = normalize_controls({"version": 1, "items": [{"id": str(uuid.uuid4()),
            "control_net_name": "user-choice.safetensors", "strength": .4 + index * .2,
            "start_percent": 0., "end_percent": 1.,
            "region": {"mode": "rect", "xyxy_normalized": [.13 + index * .15, .1, .87, .9]},
            "crop": {"enabled": True, "xyxy_normalized": [-.1, .1, .8, 1.1],
                     "fit": "stretch", "flip_x": bool(index)}}]})["items"][0]
        frames, mask, crop, source_report = MiniMaxH3DirectorControlSourceEXPT8.execute(
            raw, json.dumps(spec), width, height, 22, "image", 24., stage).result
        source_report = json.loads(source_report)
        assert frames.shape == (1, 64, 96, 3) and mask.shape == (1, height, width)
        assert json.loads(crop)["source_size"] == [96, 64]
        assert source_report["geometry"]["stage"] == stage
        if shared is None or not shared_bundle:
            shared = bundle(backend, hyperflow=hyperflow, seed=21102170 + index)
        selected, tower = shared
        calls = []
        handles.append(tower.control_blocks[0].after_proj.register_forward_hook(
            lambda _module, _args, value, calls=calls: calls.append(tuple(value.shape))))
        model, positive, state, _ = MiniMaxH3FunControlRegionApplyEXPT8.execute(
            model, positive, selected, encoder, frames, width, height, 22,
            "pose", "exact", spec["strength"], 0., 1., mask, crop, stage).result
        states.append(state)
        observations.append(calls)
    assert torch.equal(raw, before) and states[0] is not states[1]
    assert model.get_attachment(REGION_ATTACHMENT) is states[-1]
    assert states[0].geometry["geometry_sha256"] != states[1].geometry["geometry_sha256"]
    return (model, positive, latent), states, observations, handles


def inspect_states(states, observations, *, nfe, width, height, stage, shared_bundle=False):
    for state, calls in zip(states, observations, strict=True):
        report = state.report()
        assert state.run > 0 and len(calls) == nfe * (2 if shared_bundle else 1)
        assert report["residual_calls"] == report["gated_calls"] == nfe
        assert report["stage"] == stage and report["geometry"]["target"] == {
            "width": width, "height": height, "length": 22}
        assert report["target_row_evaluations"] == nfe * 7 * (height // 32) * (width // 32)
        assert report["crop_consumed_by_observed_control"] and report["region_applied"]
        assert not report["portable_cache_identity"] and not state.gate._cache
        assert report["errors"] == []


def source_identity(latent):
    return [tensor_sha(value) for value in latent["samples"].unbind()]


def ordinary_sample(built):
    model, positive, latent = built
    branch, sampler, sigmas = sampling.setup_dual_clock_sampling(model, latent, 2, 12., 3.)
    sigmas_before = sigmas.clone()
    noise = RandomNoise.execute(21102171).result[0]
    before = source_identity(latent)
    try:
        result = SamplerCustomAdvanced.execute(noise,
            BasicGuider.execute(branch, positive).result[0], sampler, sigmas, latent).result[0]
    finally:
        _restore_stage_objects(branch)
    assert torch.equal(sigmas, sigmas_before) and source_identity(latent) == before
    assert all(torch.isfinite(value).all() for value in result["samples"].unbind())
    return result


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("shared_bundle", [False, True])
def test_two_control_runtimes_are_actually_consumed_and_reset_in_ordinary_sampling(monkeypatch, encoder, backend, relay, shared_bundle):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    base = ordinary_model()
    built = conditioned(base, encoder, relay=relay)
    selected, states, observations, handles = controlled(built, encoder, backend, shared_bundle=shared_bundle)
    try:
        output = ordinary_sample(selected)
        inspect_states(states, observations, nfe=2, width=32, height=32, stage="SINGLE", shared_bundle=shared_bundle)
        for state in states:
            audited = MiniMaxH3ControlRegionAuditEXPT8.execute(state, output).result
            assert audited[0] is output and not json.loads(audited[1])["portable_cache_identity"]
        runs = [state.run for state in states]
        for observed in observations:
            observed.clear()
        retried = ordinary_sample(selected)
        inspect_states(states, observations, nfe=2, width=32, height=32, stage="SINGLE", shared_bundle=shared_bundle)
        assert all(state.run > count for state, count in zip(states, runs, strict=True))
        assert source_identity(retried) == source_identity(output)
    finally:
        for handle in handles:
            handle.remove()
    assert not base.wrappers and not base.object_patches and not torch.cuda.is_initialized()


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("shared_bundle", [False, True])
def test_two_controls_hyperflow_head_tail_use_typed_original_boundary_without_portable_claim(monkeypatch, tmp_path, encoder, backend, relay, masked, shared_bundle):
    _, diffusion = _tiny_native_model(monkeypatch)
    base = tiny_model()
    base.model.diffusion_model = diffusion
    weights = _tiny_weights(diffusion)
    low = conditioned(base, encoder, relay=relay, weights=weights)
    high = conditioned(base, encoder, relay=relay, weights=weights)
    low, low_states, low_calls, low_handles = controlled(low, encoder, backend, stage="LOW", hyperflow=True, shared_bundle=shared_bundle)
    high, high_states, high_calls, high_handles = controlled(high, encoder, backend, stage="HIGH", hyperflow=True, shared_bundle=shared_bundle)
    model, positive, source = low
    if masked:
        video, audio = source["samples"].unbind()
        video_mask, audio_mask = torch.ones_like(video), torch.ones_like(audio)
        video_mask[:, :, 0] = 0
        video_mask[:, :, 1] = .5
        source["noise_mask"] = NestedTensor((video_mask, audio_mask))
    before = source_identity(source)
    before_mask = source_identity({"samples": source["noise_mask"]}) if masked else None
    seed, calls = 21102172, []
    noise = comfy.sample.prepare_noise(source["samples"], seed)
    noise_before = source_identity({"samples": noise})
    def callback(step, _prediction, _state, total):
        calls.append((step, total))
    try:
        head_model, head_positive, head_negative = effects.bind_head(model, source, positive, [], 4)[:3]
        boundary = separate.sample_head(head_model, source, head_positive, head_negative,
            noise, seed=seed, split_interval=4, callback=callback)
        receipt = boundary.verify()
        assert (boundary.inpaint is not None) is masked
        assert receipt["execution"]["portable_identity"] is False
        assert receipt["request"]["plan"]["absolute_interval"] == [0, 4]
        inspect_states(low_states, low_calls, nfe=4, width=32, height=32, stage="LOW", shared_bundle=shared_bundle)
        for state in low_states:
            audited = MiniMaxH3DirectorControlHeadAuditEXPT8.execute(state, boundary).result
            assert audited[0] is boundary and boundary.verify() == receipt
            report = json.loads(audited[1])
            assert report["sampling_calls"] == 0 and not report["portable_cache_identity"]
        artifact, digest, archived = storage.save_boundary(boundary, tmp_path, "observed-head")
        assert (Path(tmp_path) / artifact).is_file()
        assert json.loads(archived)["status"] == "archived_unverified_identity"
        with pytest.raises(ValueError, match="Unverified executable identity"):
            storage.load_boundary(tmp_path, artifact, digest)
        tail_model, tail_positive, tail_negative = effects.bind_tail(boundary, high[0], high[1], [])[:3]
        output, text = separate.sample_tail(boundary, tail_model, tail_positive, tail_negative,
            seed=seed, callback=callback)
        tail = json.loads(text)
        assert tail["execution"]["portable_identity"] is False
        assert tail["request"]["plan"]["absolute_interval"] == [4, 8]
        assert not tail["head_executed"] and not tail["fresh_noise"] and not tail["learned_upscale"]
        inspect_states(high_states, high_calls, nfe=4, width=32, height=32, stage="HIGH", shared_bundle=shared_bundle)
        assert calls == [(index, 8) for index in range(8)]
        assert source_identity({"samples": noise}) == noise_before and source_identity(source) == before
        if masked:
            assert source_identity({"samples": source["noise_mask"]}) == before_mask
        else:
            assert "noise_mask" not in source
        assert all(torch.isfinite(value).all() for value in output["samples"].unbind())
        assert hf_runtime.active_step() is None and hf_runtime._ACTIVE_FORWARD.get() is None
    finally:
        for handle in low_handles + high_handles:
            handle.remove()
    assert not base.object_patches and not base.wrappers and not torch.cuda.is_initialized()


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_actual_low_high_canvases_keep_source_crop_and_external_forward_owner(monkeypatch, encoder, backend):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    base = ordinary_model()
    calls = []

    def foreign(executor, *args, **kwargs):
        calls.append(True)
        return executor(*args, **kwargs)

    base.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "real_foreign_delegate", foreign)
    for stage, width in [("LOW", 32), ("HIGH", 64)]:
        built = conditioned(base, encoder, relay=True, width=width)
        selected, states, observations, handles = controlled(built, encoder, backend, stage=stage)
        try:
            ordinary_sample(selected)
            inspect_states(states, observations, nfe=2, width=width, height=width, stage=stage)
            assert selected[0].get_wrappers(WrappersMP.DIFFUSION_MODEL, "real_foreign_delegate") == [foreign]
            for state in states:
                assert state.geometry["source"] == {"width": 96, "height": 64, "frames": 1}
        finally:
            for handle in handles:
                handle.remove()
    assert len(calls) == 4 and not torch.cuda.is_initialized()


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_second_real_control_failure_does_not_leave_first_state_dirty_on_retry(monkeypatch, encoder, backend):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    selected, states, observations, handles = controlled(
        conditioned(ordinary_model(), encoder, relay=True), encoder, backend)
    baseline = ordinary_sample(selected)
    failures, consume = [], states[1].consume

    def fail_after_actual_residual(skip, args, **kwargs):
        actual = consume(skip, args, **kwargs)
        failures.append(tuple(actual.shape))
        raise RuntimeError("actual second regional residual failure")

    try:
        monkeypatch.setattr(states[1], "consume", fail_after_actual_residual)
        with pytest.raises(RuntimeError, match="actual second regional residual failure"):
            ordinary_sample(selected)
        assert len(failures) == 1 and all(not state.gate._cache for state in states)
        monkeypatch.setattr(states[1], "consume", consume)
        for observation in observations:
            observation.clear()
        recovered = ordinary_sample(selected)
        inspect_states(states, observations, nfe=2, width=32, height=32, stage="SINGLE")
        assert source_identity(recovered) == source_identity(baseline)
    finally:
        for handle in handles:
            handle.remove()
    assert not torch.cuda.is_initialized()
