import json

import pytest
import torch
from comfy_extras.nodes_custom_sampler import (
    BasicGuider,
    RandomNoise,
    SamplerCustomAdvanced,
)

from h3_audio_t8_pkg.audio_tail_release import (
    carry_audio_prefix,
    setup_audio_tail_release,
    KEY,
)
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import latent, conditioning


def source():
    target = latent()
    target["retained_metadata"] = {"source": "same target"}
    previous = latent()
    previous["samples"].unbind()[1].fill_(0.25)
    carried, report = carry_audio_prefix(target, previous, 2)
    assert json.loads(report)["actual_target_audio_mask_pinned"]
    return target, previous, carried


def test_carry_is_real_target_audio_pin_and_preserves_source_video_metadata():
    target, previous, carried = source()
    video, audio = carried["samples"].unbind()
    assert video is target["samples"].unbind()[0]
    assert carried["retained_metadata"] is target["retained_metadata"]
    assert (audio[..., :2] == 0.25).all() and not audio[..., 2:].count_nonzero()
    assert not carried["noise_mask"].unbind()[1][..., :2].count_nonzero()
    assert carried["noise_mask"].unbind()[1][..., 2:].all()
    assert not target["samples"].unbind()[1].count_nonzero()
    assert (previous["samples"].unbind()[1] == 0.25).all()


def test_disabled_modes_are_exact_noops_without_inventing_a_pin():
    selected = model()
    target = latent()
    assert carry_audio_prefix(target, None, 0)[0] is target
    assert setup_audio_tail_release(selected, target)[0] is selected
    with pytest.raises(ValueError, match="voice-reference"):
        setup_audio_tail_release(selected, target, "release_after_step", 2, 2)


@pytest.mark.parametrize("release_step", [0, 2, 20])
def test_actual_Core_loop_inpaint_and_DiT_audio_mask_release_together(release_step):
    _, _, carried = source()
    original_audio_mask = carried["noise_mask"].unbind()[1].clone()
    selected, sampler, sigmas = setup_dual_clock_sampling(
        model(), carried, 4, 12.0, 3.0
    )
    delegates = []

    def prior(apply, options):
        delegates.append(True)
        return apply(options["input"], options["timestep"], **options["c"])

    selected.set_model_unet_function_wrapper(prior)
    selected, report = setup_audio_tail_release(
        selected, carried, "release_after_step", 2, release_step
    )
    runtime = selected.get_attachment(KEY)[0]
    observed = []

    def inspect(_module, _args, kwargs):
        mask = kwargs["audio_denoise_mask"]
        observed.append(float(mask[..., :2].min()))

    handle = selected.model.diffusion_model.register_forward_pre_hook(
        inspect, with_kwargs=True
    )
    outputs = []
    try:
        guider = BasicGuider.execute(selected, conditioning()).result[0]
        for _ in range(2):
            output = SamplerCustomAdvanced.execute(
                RandomNoise.execute(412).result[0], guider, sampler, sigmas, carried
            ).result[0]
            outputs.append(output)
            assert len(runtime.calls) == len(runtime.mask_calls) == 4
    finally:
        handle.remove()
    expected = [float(step >= release_step) for step in range(4)]
    assert observed == expected * 2 and len(delegates) == 8
    assert torch.equal(carried["noise_mask"].unbind()[1], original_audio_mask)
    assert all(torch.isfinite(t).all() for t in outputs[0]["samples"].unbind())
    for a, b in zip(outputs[0]["samples"].unbind(), outputs[1]["samples"].unbind()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    audio = outputs[0]["samples"].unbind()[1]
    if release_step >= 4:
        torch.testing.assert_close(
            audio[..., :2], carried["samples"].unbind()[1][..., :2], rtol=0, atol=1e-6
        )
    else:
        assert (
            audio[..., :2] - carried["samples"].unbind()[1][..., :2]
        ).abs().max() > 1e-5


def test_actual_failure_does_not_mutate_input_or_prevent_cached_guider_retry():
    _, _, carried = source()
    original = carried["noise_mask"].unbind()[1].clone()
    selected, sampler, sigmas = setup_dual_clock_sampling(
        model(), carried, 4, 12.0, 3.0
    )
    fail = [True]
    mask_calls = []

    def mask(sigma, value, extra_options=None):
        mask_calls.append(True)
        return value.clone()

    def wrapper(apply, options):
        if fail[0]:
            fail[0] = False
            raise RuntimeError("actual user delegate failure")
        return apply(options["input"], options["timestep"], **options["c"])

    selected.set_model_denoise_mask_function(mask)
    selected.set_model_unet_function_wrapper(wrapper)
    selected, _ = setup_audio_tail_release(
        selected, carried, "release_after_step", 2, 0
    )
    runtime = selected.get_attachment(KEY)[0]
    guider = BasicGuider.execute(selected, conditioning()).result[0]
    noise = RandomNoise.execute(412).result[0]
    with pytest.raises(RuntimeError, match="actual user delegate failure"):
        SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, carried)
    assert torch.equal(carried["noise_mask"].unbind()[1], original)
    output = SamplerCustomAdvanced.execute(
        noise, guider, sampler, sigmas, carried
    ).result[0]
    assert len(runtime.calls) == len(runtime.mask_calls) == 4
    assert len(mask_calls) == 5 and all(
        torch.isfinite(v).all() for v in output["samples"].unbind()
    )
    assert torch.equal(carried["noise_mask"].unbind()[1], original)


@pytest.mark.parametrize("broken", ["interval", "nonfinite", "not_pinned", "delegate"])
def test_actual_input_errors_are_specific_and_original_objects_remain_unchanged(broken):
    target, previous, carried = source()
    if broken == "interval":
        with pytest.raises(ValueError, match="interval exceeds"):
            carry_audio_prefix(target, previous, 2, 1)
    elif broken == "nonfinite":
        previous["samples"].unbind()[1][..., -1] = float("nan")
        with pytest.raises(ValueError, match="nonfinite"):
            carry_audio_prefix(target, previous, 2)
    elif broken == "not_pinned":
        carried["noise_mask"].unbind()[1][..., :2] = 0.5
        with pytest.raises(ValueError, match="hard-pinned"):
            setup_audio_tail_release(model(), carried, "release_after_step", 2, 0)
    else:
        selected = model()
        selected.model_options["denoise_mask_function"] = 7
        with pytest.raises(TypeError, match="must be callable"):
            setup_audio_tail_release(selected, carried, "release_after_step", 2, 0)


def test_node_schemas_expose_noop_defaults_and_do_not_download_or_load_models():
    from h3_audio_t8_pkg.nodes_audio_tail_release import NODES

    assert len({cls.define_schema().node_id for cls in NODES}) == 3
    target = latent()
    assert NODES[0].execute(target).result[0] is target
    selected = model()
    assert NODES[1].execute(selected, target).result[0] is selected
    assert NODES[2].execute(selected, target).result[0] is target
