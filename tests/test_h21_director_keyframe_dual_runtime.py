"""LOW/full learned3D/HIGH CPU chain: real loader, no upscaler doubles.

Actual installed 690MB checkpoint, random tiny DiT, real miniature Core
Encoders and synthetic text_dim8 CLIP tokens. Two complete 2-NFE descents
test composition, not the production partial4+4 grid or trained media quality.
"""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.director_project import file_sha
from h3_audio_t8_pkg.multikeyframe_advanced import ACTUAL_FRAME_INDEX, append_keyframe_plan
from h3_audio_t8_pkg.nodes_director_keyframes import MiniMaxH3DirectorKeyframeConditioningEXPT8
from h3_audio_t8_pkg.prompt_relay_advanced import (
    _layout_contract, build_prompt_relay_plan, prompt_relay_model_contract,
)
from test_fast_h3_v2_core_sampler import model as real_tiny_model
from test_h21_control_regions import RealTinyEncoderOwner
from test_h21_director_keyframe_runtime import CountingNativeLikeFakeClip, actual_payload
from helpers import FakeAudioVAE


MODEL_NAME = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
LENGTH = 22  # Actual 17n+5 H3 frame grid; two Relay events each have >=5 frames.


def tensor_sha(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


@pytest.fixture(scope="module")
def full_learned_checkpoint():
    # Actual production file lookup, SHA, safetensors read, state validation,
    # complete 512-channel network and CoreModelPatcher. No patched loader or
    # model-management function. Reuse the real cache between the two cases.
    entry, _ = learned._load_cached_model(MODEL_NAME, "fp32")
    path = Path(entry.path)
    assert path.is_file() and path.stat().st_size == 690_592_672
    assert entry.sha256 == file_sha(path)
    assert entry.contract["state_contract_match"] is True
    assert isinstance(entry.patcher.model, learned.MiniMaxH3LearnedResizer3D)
    assert entry.patcher.model.conv_in.out_channels == 512
    calls = []

    def actual_forward(_module, positional, output):
        calls.append({"input_shape": list(positional[0].shape),
                      "scale": float(positional[1]), "target": list(positional[2]),
                      "output_shape": list(output.shape), "output_sha256": tensor_sha(output)})

    handle = entry.patcher.model.register_forward_hook(actual_forward)
    try:
        yield entry, path, calls
    finally:
        handle.remove()
        assert file_sha(path) == entry.sha256
        learned.clear_learned_resizer_cache()
        assert not torch.cuda.is_initialized()


def conditioning(bare, vae, image, size, relay):
    clip = CountingNativeLikeFakeClip()
    plan = append_keyframe_plan(None, image * .9, "frame", 1, .93, "stretch", True)
    kwargs = dict(model=bare.clone(), clip=clip, video_vae=vae, audio_vae=FakeAudioVAE(),
        prompt="A person moves.", width=size, height=size, length=LENGTH, task_type="FL2VA",
        audio_mode="native", audio_denoise_strength=.35, add_source_as_reference=False,
        prompt_primary_audio_ordinal=0, strict_prompt_tags=True, ref_image_size="match",
        reference_video_policy="official_2_to_15s", first_frame_noise_aug=.999,
        last_frame_noise_aug=.999, reference_visual_noise_aug=.999,
        first_frame=image, last_frame=image * .8, keyframe_plan=plan)
    if relay:
        kwargs.update(prompt_relay_plan=build_prompt_relay_plan("A person moves.", "Raise hand.\nTurn.",
            LENGTH, "auto_equal", "", "paper_v1", .1, False, False)[0],
            execution_mode="apply_exp", query_chunk_rows=64)
    result = MiniMaxH3DirectorKeyframeConditioningEXPT8.execute(**kwargs).result
    assert len(clip.tokenize_calls) == len(clip.encode_calls) == 1
    assert len(vae.calls) == 3
    assert all(tuple(frames.shape) == (1, size, size, 3) for frames in vae.calls)
    assert [item[ACTUAL_FRAME_INDEX] for item in result[1][0][1]["minimax_keyframes"]] == [0, 1, LENGTH - 1]
    assert all(tuple(item["latent"].shape[-2:]) == (size // 16, size // 16)
               for item in result[1][0][1]["minimax_keyframes"])
    return result


def sample(model, positive, latent, seed):
    observed = []

    def actual_forward(_module, _positional, keyword):
        observed.append(_layout_contract(keyword["minimax_payload"]["layout"]))

    handle = model.model.diffusion_model.register_forward_pre_hook(actual_forward, with_kwargs=True)
    source = [part.clone() for part in latent["samples"].unbind()]
    try:
        prepared, sampler, sigmas = sampling.setup_dual_clock_sampling(model, latent, 2, 12., 3.)
        guider = BasicGuider.execute(prepared, positive).result[0]
        result = SamplerCustomAdvanced.execute(RandomNoise.execute(seed).result[0], guider, sampler, sigmas, latent).result
    finally:
        handle.remove()
    assert len(observed) == 2
    assert all(torch.equal(before, after) for before, after in zip(source, latent["samples"].unbind(), strict=True))
    assert all(torch.isfinite(part).all() for part in result[0]["samples"].unbind())
    return result, observed


@pytest.mark.parametrize("relay", [False, True])
def test_installed_full_learned_network_between_independent_real_low_high_conditions(
        full_learned_checkpoint, monkeypatch, request, relay):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    entry, weight_path, lift_calls = full_learned_checkpoint
    lift_before = len(lift_calls)
    bare = real_tiny_model()
    original_weights = [value.detach().clone() for value in bare.model.parameters()]
    image = torch.linspace(.05, .95, 32 * 32 * 3).reshape(1, 32, 32, 3)
    original_image = image.clone()
    low_vae, high_vae = RealTinyEncoderOwner(), RealTinyEncoderOwner()
    low = conditioning(bare, low_vae, image, 32, relay)
    low_source = [value.clone() for value in low[2]["samples"].unbind()]
    low_outputs, low_layouts = sample(low[0], low[1], low[2], 21102191)
    low_x0 = low_outputs[1]
    low_x0_before = [value.clone() for value in low_x0["samples"].unbind()]
    lifted, width, height, raw = learned.learned_upscale_h3_av_latent(
        low_x0, MODEL_NAME, "target_dimensions", 2., .004096, 64, 64,
        "preserve_source", 1.05, "fp32", "offload_after")
    report = json.loads(raw)
    assert (width, height) == (64, 64)
    assert len(lift_calls) == lift_before + 1
    assert lift_calls[-1]["input_shape"] == [1, 24, 7, 2, 2]
    assert lift_calls[-1]["output_shape"] == [1, 24, 7, 4, 4]
    assert report["model"]["sha256"] == entry.sha256
    assert report["model"]["contract"]["state_contract_match"] is True
    assert lifted["samples"].unbind()[1] is low_x0["samples"].unbind()[1]
    assert all(torch.equal(before, value) for before, value in zip(low_x0_before, low_x0["samples"].unbind(), strict=True))
    high = conditioning(bare, high_vae, image, width, relay)
    assert high[0] is not low[0] and high_vae is not low_vae
    assert low[0].model is high[0].model is bare.model
    with pytest.raises(ValueError, match="not rebuilt"):
        learned.reconcile_two_pass_h3_latent(lifted, high[2], low[1], "auto")
    reconciled, high_positive, reconciled_raw = learned.reconcile_two_pass_h3_latent(lifted, high[2], high[1], "auto")
    assert reconciled["samples"].unbind()[1] is lifted["samples"].unbind()[1]
    assert json.loads(reconciled_raw)["audio_source"] == "first_pass"
    high_inputs = [value.clone() for value in reconciled["samples"].unbind()]
    high_outputs, high_layouts = sample(high[0], high_positive, reconciled, 21102192)
    assert high_outputs[0]["samples"].unbind()[0].shape == (1, 24, 7, 4, 4)
    assert len(low_vae.calls) == len(high_vae.calls) == 3
    if relay:
        low_binding = prompt_relay_model_contract(low[0])["binding"]["layout_contract"]
        high_binding = prompt_relay_model_contract(high[0])["binding"]["layout_contract"]
        assert low_binding != high_binding
        assert all(value == low_binding for value in low_layouts)
        assert all(value == high_binding for value in high_layouts)
        assert _layout_contract(actual_payload(high[0], high_positive, reconciled)["layout"]) == high_binding
    assert not bare.object_patches and not bare.wrappers
    assert all(torch.equal(before, value) for before, value in zip(original_weights, bare.model.parameters(), strict=True))
    assert all(torch.equal(before, value) for before, value in zip(low_source, low[2]["samples"].unbind(), strict=True))
    assert all(torch.equal(before, value) for before, value in zip(high_inputs, reconciled["samples"].unbind(), strict=True))
    assert torch.equal(image, original_image)
    assert not torch.cuda.is_initialized()
    request.node.user_properties.append(("actual_dual_evidence", json.dumps({
        "route": "relay" if relay else "ordinary", "low_nfe": 2, "high_nfe": 2,
        "qualification": "two_complete_two_step_descents_not_published_partial4plus4",
        "full_weight": {"path": str(weight_path), "sha256": entry.sha256, "bytes": weight_path.stat().st_size},
        "actual_lift": lift_calls[-1],
        "source_image_sha256": tensor_sha(image),
        "low_x0_video_sha256": tensor_sha(low_x0["samples"].unbind()[0]),
        "low_x0_audio_sha256": tensor_sha(low_x0["samples"].unbind()[1]),
        "lifted_video_sha256": tensor_sha(lifted["samples"].unbind()[0]),
        "lifted_audio_sha256": tensor_sha(lifted["samples"].unbind()[1]),
        "high_output_video_sha256": tensor_sha(high_outputs[0]["samples"].unbind()[0]),
        "high_output_audio_sha256": tensor_sha(high_outputs[0]["samples"].unbind()[1]),
        "low_condition_latent_sha256": [tensor_sha(item["latent"]) for item in low[1][0][1]["minimax_keyframes"]],
        "high_condition_latent_sha256": [tensor_sha(item["latent"]) for item in high[1][0][1]["minimax_keyframes"]],
        "low_physical_layout": low_layouts[0], "high_physical_layout": high_layouts[0],
        "trained_text_or_dit_quality": False, "cuda_initialized": False}, sort_keys=True)))
