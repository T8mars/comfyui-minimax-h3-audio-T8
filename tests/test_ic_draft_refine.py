import json

import pytest
import torch
from safetensors.torch import save_file

from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg.ic_draft_refine import compose_ic_target, ic_recipe


def asset(tmp_path, length=207):
    raw = (
        torch.linspace(-0.3, 0.5, 2 * 32 * length)
        .reshape(2, 32, length)
        .to(torch.bfloat16)
    )
    path = tmp_path / "renamed-silence.safetensors"
    save_file({"audio_latents": raw}, str(path))
    return path, raw


def test_124_native_target_consumes_exact_nonzero_silence_and_keeps_mask_metadata(
    tmp_path,
):
    target, count = empty_av_latent(64, 32, 124)
    assert count == 124
    video, audio = target["samples"].unbind()
    path, raw = asset(tmp_path)
    target["test_clock_metadata"] = {"source_origin": 7.5}
    video = torch.ones_like(video) * 0.2
    mask = torch.ones_like(video)
    mask[:, :, :3] = 0.0
    coarse = {"samples": video, "noise_mask": mask}
    output, report = compose_ic_target(coarse, target, path)
    actual_video, actual_audio = output["samples"].unbind()
    vm, am = output["noise_mask"].unbind()
    assert actual_video is video and vm is mask and not am.count_nonzero()
    assert torch.equal(actual_audio, raw.permute(1, 0, 2).unsqueeze(0).to(audio))
    assert output["test_clock_metadata"] is target["test_clock_metadata"]
    assert target.get("noise_mask") is None and not audio.count_nonzero()
    assert json.loads(report)["audio_rows_added"] == 0
    assert json.loads(report)["quality_accepted"] is False


@pytest.mark.parametrize("length", [141, 107])
def test_different_duration_requires_explicit_fitting_not_silent_zero_padding(
    tmp_path, length
):
    target, _ = empty_av_latent(64, 32, length)
    path, raw = asset(tmp_path)
    with pytest.raises(ValueError, match="Silence time rows"):
        compose_ic_target(target, target, path)
    output, text = compose_ic_target(target, target, path, "trim_or_zero_pad_exp")
    actual = output["samples"].unbind()[1]
    expected = target["samples"].unbind()[1]
    report = json.loads(text)
    assert actual.shape == expected.shape
    assert report["audio_rows_added"] + report["audio_rows_removed"] == abs(
        expected.shape[-1] - 207
    )
    common = min(207, expected.shape[-1])
    assert torch.equal(
        actual[..., :common], raw.permute(1, 0, 2).unsqueeze(0)[..., :common].to(actual)
    )
    if expected.shape[-1] > 207:
        assert not actual[..., 207:].count_nonzero()


def test_actual_shape_or_corrupt_asset_error_is_not_a_model_name_gate(tmp_path):
    target, _ = empty_av_latent(64, 32, 124)
    bad, _ = empty_av_latent(96, 32, 124)
    path, _ = asset(tmp_path)
    with pytest.raises(ValueError, match="same latent"):
        compose_ic_target(bad, target, path)
    save_file({"not_audio_latents": torch.ones(2, 32, 207)}, str(path))
    with pytest.raises(ValueError, match="audio_latents"):
        compose_ic_target(target, target, path)


def test_unpinned_is_explicit_and_does_not_clear_video_mask(tmp_path):
    source, _ = empty_av_latent(64, 32, 124)
    path, _ = asset(tmp_path)
    output, text = compose_ic_target(source, source, path, pin_audio=False)
    assert output["noise_mask"].unbind()[1].all()
    assert not json.loads(text)["audio_pinned"]


def test_recipe_keeps_literal_caption_and_already_shifted_sigmas():
    caption = "  <d>你好，今天不剪辑。</d>  \nKeep <Video 1>."
    prompt, grid, text = ic_recipe(caption)
    assert prompt.endswith("Scene description: " + caption)
    torch.testing.assert_close(
        grid, torch.tensor([0.834942, 0.696852, 0.468753, 0.0]), rtol=0, atol=0
    )
    assert not json.loads(text)["additional_shift_applied"]
    _, custom, _ = ic_recipe(caption, custom_sigmas="1.2, 0.4; 0")
    assert custom.tolist() == torch.tensor([1.2, 0.4, 0.0]).tolist()


@pytest.mark.parametrize("values", [".9,.5", ".9,.5,.6,0", ".9,nan,0", "1,0,0"])
def test_invalid_sigma_data_is_not_automatically_changed(values):
    with pytest.raises(ValueError, match="sigmas"):
        ic_recipe("same video", custom_sigmas=values)


def test_node_schemas_do_not_load_models():
    from h3_audio_t8_pkg.nodes_ic_draft_refine import NODES, SAMPLER_NODES

    assert len(NODES) == 2 and len(SAMPLER_NODES) == 1
    for node in [*NODES, *SAMPLER_NODES]:
        assert node.define_schema().node_id.startswith("MiniMaxH3IC")


@pytest.mark.parametrize(
    "preset,steps,start", [("up3", 3, 5), ("up4", 4, 4), ("hf8", 8, 0)]
)
def test_real_Core_IC_sampler_uses_actual_two_time_intervals_and_pinned_audio(
    monkeypatch, preset, steps, start
):
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider,
        RandomNoise,
        SamplerCustomAdvanced,
    )
    from comfy.nested_tensor import NestedTensor
    from test_fast_h3_v2_core_sampler import model
    from test_progressive_sampling_runtime import conditioning, latent
    from test_hyperflow_advanced import _tiny_native_model, _tiny_weights
    from h3_audio_t8_pkg.hyperflow_runtime_advanced import (
        install_hyperflow,
        active_step,
    )
    from h3_audio_t8_pkg.ic_draft_refine import setup_ic_sampler

    selected = model()
    _, diffusion = _tiny_native_model(monkeypatch)
    selected.model.diffusion_model = diffusion
    selected, _, _ = install_hyperflow(selected, _tiny_weights(diffusion))
    source = latent()
    video, audio = source["samples"].unbind()
    audio.fill_(0.25)
    source["noise_mask"] = NestedTensor(
        (torch.ones_like(video), torch.zeros_like(audio))
    )
    prepared, sampler, sigmas, report = setup_ic_sampler(selected, source, preset)
    observed = []
    handle = diffusion.register_forward_pre_hook(
        lambda *_: observed.append(active_step().absolute_index)
    )
    try:
        guider = BasicGuider.execute(prepared, conditioning()).result[0]
        output = SamplerCustomAdvanced.execute(
            RandomNoise.execute(44).result[0], guider, sampler, sigmas, source
        ).result[0]
    finally:
        handle.remove()
    assert observed == list(range(start, 8))
    assert len(sigmas) == steps + 1 and sigmas[-1] == 0
    assert json.loads(report)["absolute_interval"] == [start, 8]
    assert all(torch.isfinite(x).all() for x in output["samples"].unbind())
    # Core latent_format normalization roundtrip is not byte-exact float32.
    # Delivery uses the original draft PCM, not this fixed condition's decode.
    torch.testing.assert_close(output["samples"].unbind()[1], audio, rtol=0, atol=1e-6)
    assert (audio == 0.25).all()
    assert active_step() is None


@pytest.mark.parametrize("memory", [False, True])
def test_HyperFlow_Sharp_consumes_expanded_current_times_with_independent_delta_oracle(
    monkeypatch,
    memory,
):
    import torch.nn.functional as F
    from comfy.nested_tensor import NestedTensor
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider,
        RandomNoise,
        SamplerCustomAdvanced,
    )
    from test_fast_h3_v2_core_sampler import model
    from test_hyperflow_advanced import _tiny_native_model, _tiny_weights
    from test_open_regenerate_recipe import factors
    from test_progressive_sampling_runtime import conditioning, latent
    from h3_audio_t8_pkg.hyperflow_runtime_advanced import (
        install_hyperflow,
        active_time_values,
    )
    from h3_audio_t8_pkg.open_regenerate_recipe import attach_curves
    from h3_audio_t8_pkg.ic_draft_refine import setup_ic_sampler

    selected = model()
    _, network = _tiny_native_model(monkeypatch)
    selected.model.diffusion_model = network
    if memory:
        from h3_audio_t8_pkg.h3_memory_advanced import (
            configure_low_vram_attention,
            configure_chunk_feed_forward,
        )

        selected = configure_low_vram_attention(selected, 2)[0]
        selected = configure_chunk_feed_forward(selected, 2, 256)[0]
    selected, _, _ = install_hyperflow(selected, _tiny_weights(network))
    table, branches = factors(network)
    selected, runtime = attach_curves(selected, table, branches, 1.1)
    memory_attention_calls = []
    memory_ffn_calls = []
    if memory:
        from h3_audio_t8_pkg import h3_memory_advanced as memory_runtime

        original_attention = memory_runtime._attention_impl

        def observed_attention(*args, **kwargs):
            memory_attention_calls.append(True)
            return original_attention(*args, **kwargs)

        monkeypatch.setattr(memory_runtime, "_attention_impl", observed_attention)
        network.blocks[0].mlp.fc1.register_forward_hook(
            lambda *_: memory_ffn_calls.append(True)
        )
    source = latent()
    condition = conditioning("i2va")
    if memory:
        # More than 256 actual packed tokens, so the selected FFN split runs.
        old_audio = source["samples"].unbind()[1]
        source["samples"] = NestedTensor((torch.zeros(1, 24, 2, 16, 32), old_audio))
        condition[0][1]["minimax_keyframes"][0]["latent"] = (
            torch.ones(1, 24, 1, 16, 32) * 0.2
        )
    video, audio = source["samples"].unbind()
    source["noise_mask"] = NestedTensor(
        (torch.ones_like(video), torch.zeros_like(audio))
    )
    prepared, sampler, sigmas, _ = setup_ic_sampler(selected, source, "up3")
    checked = []

    def oracle(module, inputs, output):
        times = active_time_values()
        a, b = branches["blocks.0.adaln_proj.linear"]
        # Independently evaluate interpolation then both factors at all actual
        # current times. Expanded duplicates have equal t and distinct r.
        positions = times.clamp(0, 1) * (len(table) - 1)
        lo = positions.floor().long().clamp(max=len(table) - 2)
        coords = table[lo] + (table[lo + 1] - table[lo]) * (positions - lo).unsqueeze(
            -1
        )
        expected = F.linear(inputs[0], module.weight, module.bias) + 1.1 * F.linear(
            F.linear(coords, a), b
        )
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-7)
        checked.append(times.clone())

    # Registered inside the scoped consumer after its own delta hook.
    original = runtime.apply_delta

    def checked_delta(value, output, name):
        result = original(value, output, name)
        oracle(network.blocks[0].adaln_proj.linear, (value,), result)
        return result

    runtime.apply_delta = checked_delta
    guider = BasicGuider.execute(prepared, condition).result[0]
    result = SamplerCustomAdvanced.execute(
        RandomNoise.execute(76).result[0], guider, sampler, sigmas, source
    ).result[0]
    assert len(checked) == 3 and all(len(t) > len(torch.unique(t)) for t in checked)
    assert runtime.completed_forwards == 3 and runtime.failed_forwards == 0
    assert all(
        r["clock_source"] == "actual_HyperFlow_expanded_current_time_rows"
        for r in runtime.forwards
    )
    assert all(torch.isfinite(t).all() for t in result["samples"].unbind())
    assert active_time_values() is None
    if memory:
        assert len(memory_attention_calls) == 3 * len(network.blocks)
        assert len(memory_ffn_calls) == 6  # 2 real chunks for each of 3 forwards.
