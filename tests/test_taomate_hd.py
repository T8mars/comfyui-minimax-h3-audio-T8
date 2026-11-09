"""Teacher recipe and real tiny Core execution; not trained GPU quality."""
import json

import pytest
import torch
from comfy.nested_tensor import NestedTensor

from h3_audio_t8_pkg import taomate_hd_math as math_hd
from h3_audio_t8_pkg.taomate_hd_runtime import sample_taomate_hd, conditioning_at_size
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning


def source():
    return {"samples": NestedTensor((torch.zeros(1, 24, 2, 4, 6), torch.zeros(1, 32, 2, 8))),
            "user_metadata": {"keep": False}}


def test_teacher_exact_states_noise_and_lift_covariance():
    high = math_hd.ProgressiveCanvas(96, 64)
    low = math_hd.derive_low_canvas(high, .625)
    v, h = math_hd.coupled_video_noise(low_canvas=low, high_canvas=high, temporal_length=2, seed=8301)
    py = math_hd._orthonormal_lowpass_projection(low.latent_height, high.latent_height)
    px = math_hd._orthonormal_lowpass_projection(low.latent_width, high.latent_width)
    torch.testing.assert_close(py @ py.T, torch.eye(py.shape[0]))
    torch.testing.assert_close(px @ px.T, torch.eye(px.shape[0]))
    target = torch.einsum("kw,bctlw->bctlk", px, torch.einsum("lh,bcthw->bctlw", py, h))
    assert torch.equal(v, target)
    sigmas = math_hd.teacher_sigmas()
    grid = torch.linspace(1., 0., 50)
    assert sigmas == tuple(float((12*grid/(1+11*grid))[i]) for i in (0,16,33,49))
    # A pure sigma*LOW-noise state lifts back to sigma*HIGH-noise.
    result = math_hd.lift_video_latent(low_video=v*sigmas[2], low_noise=v, high_noise=h,
        sigma=sigmas[2], low_canvas=low, high_canvas=high)
    torch.testing.assert_close(result, h*sigmas[2], rtol=2e-6, atol=5e-7)


@pytest.mark.parametrize("steps", [3,4,6])
def test_real_core_exact_call_count_finite_audio_and_preserved_source(steps):
    src = source()
    original = [p.clone() for p in src["samples"].unbind()]
    selected = model()
    out, report = sample_taomate_hd(selected, conditioning(), src, steps=steps)
    report = json.loads(report)
    assert report["total_denoiser_calls"] == steps
    assert report["official_evaluation_count"] == (steps == 3)
    assert report["stages"][0]["video_shape"][-2:] == [2,4]
    assert report["stages"][2]["video_shape"][-2:] == [4,6]
    assert out["user_metadata"] == {"keep":False}
    assert all(torch.isfinite(p).all() for p in out["samples"].unbind())
    assert all(torch.equal(p,q) for p,q in zip(src["samples"].unbind(),original))
    assert not selected.object_patches and not selected.wrappers


def test_multiple_keyframes_resized_only_spatially_and_refs_remain_same():
    positive = conditioning()
    refs = [{"type":"image", "latent":torch.ones(1,24,1,4,6)}]
    frames = [{"resolved_frame_index":i, "latent":torch.randn(1,24,1,4,6)} for i in (0,4)]
    positive[0][1].update(minimax_keyframes=frames, minimax_refs=refs, custom=False)
    low = conditioning_at_size(positive,2,4)
    assert low[0][1]["minimax_refs"] is refs
    assert low[0][1]["custom"] is False
    assert [f["resolved_frame_index"] for f in low[0][1]["minimax_keyframes"]] == [0,4]
    assert all(f["latent"].shape == (1,24,1,2,4) for f in low[0][1]["minimax_keyframes"])
    assert all(f["latent"].shape == (1,24,1,4,6) for f in frames)


def test_native_mask_preserves_locked_audio_and_mask_metadata():
    src = source()
    generator = torch.Generator().manual_seed(12)
    src["samples"].unbind()[1].copy_(torch.randn((1,32,2,8),generator=generator)*.1)
    src["noise_mask"] = NestedTensor((torch.ones(1,1,2,4,6),torch.zeros(1,1,2,8)))
    expected = src["samples"].unbind()[1].clone()
    result, _ = sample_taomate_hd(model(), conditioning(), src)
    torch.testing.assert_close(result["samples"].unbind()[1],expected,rtol=0,atol=1e-6)
    assert result["noise_mask"] is src["noise_mask"]


def test_seed_reproducible_independent_high_and_unknown_model_class_delegates():
    src = source()
    selected = model()
    # Alternate class name must not become a new model whitelist.
    class AnotherCompatibleCore(type(selected)):
        pass
    selected.__class__ = AnotherCompatibleCore
    a, report = sample_taomate_hd(selected,conditioning(),src,high_model=model())
    b, _ = sample_taomate_hd(model(),conditioning(),source())
    assert json.loads(report)["independent_high_model"]
    for x,y in zip(a["samples"].unbind(),b["samples"].unbind()):
        assert torch.equal(x,y)


def test_actual_core_audio_only_guide_survives_low_resize_and_sampling():
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide
    from test_nov05_reference_slices import RealAudioEncoderOwner
    src = source()
    sound = {"sample_rate": 32000, "waveform": torch.zeros(1, 2, 6400)}
    positive = MiniMaxH3AddGuide.execute(conditioning(), src, 0,
        audio_vae=RealAudioEncoderOwner(), audio=sound).result[0]
    original = positive[0][1]["minimax_keyframes"][0]
    assert set(original) == {"resolved_frame_index", "audio_latent"}
    low = conditioning_at_size(positive, 2, 4)
    assert low[0][1]["minimax_keyframes"][0]["audio_latent"] is original["audio_latent"]
    output, raw = sample_taomate_hd(model(), positive, src)
    assert json.loads(raw)["actual_model_apply_calls"] == 3
    assert all(bool(torch.isfinite(value).all()) for value in output["samples"].unbind())
    assert "latent" not in original


@pytest.mark.parametrize("cfg", [0., 1., 2.])
def test_report_separates_solver_evaluations_from_actual_model_calls(cfg):
    selected = model()
    seen = []
    hook = selected.model.diffusion_model.register_forward_pre_hook(lambda *a: seen.append(True))
    try:
        _, raw = sample_taomate_hd(selected, conditioning(), source(), cfg=cfg, negative=conditioning())
    finally:
        hook.remove()
    report = json.loads(raw)
    assert report["total_solver_evaluations"] == report["total_denoiser_calls"] == 3
    assert report["actual_model_apply_calls"] == len(seen)
    assert report["actual_model_apply_calls"] == sum(row["actual_model_apply_calls"] for row in report["stages"])
    assert report["official_evaluation_count"] is (cfg == 1.)
    if cfg == 2.:
        assert len(seen) == 6
    assert not selected.wrappers


@pytest.mark.parametrize("locked_audio", [False, True])
def test_actual_raw_audio_clock_model_uses_physical_audio_delta_and_keeps_selected_owner(locked_audio):
    from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling, time_shift_sigma
    src = source()
    if locked_audio:
        video, audio = src["samples"].unbind()
        audio.fill_(.1)
        src["noise_mask"] = NestedTensor((torch.ones_like(video), torch.zeros_like(audio)))
    selected, _, _ = setup_dual_clock_sampling(model(), src, 3, 12., 3.)
    sampling = selected.get_model_object("model_sampling")
    class AnotherCompatibleAudioClock(type(sampling)):
        pass
    sampling.__class__ = AnotherCompatibleAudioClock
    seen = []
    def observe(module, args, kwargs, output):
        seen.append((args[0][1].clone(), float(args[1].flatten()[0])/1000., output[1].clone()))
    hook = selected.model.diffusion_model.register_forward_hook(observe, with_kwargs=True)
    try:
        result, raw = sample_taomate_hd(selected, conditioning(), src)
    finally:
        hook.remove()
    report = json.loads(raw)
    assert all(row["audio_clock"]["mode"] == "raw_audio_velocity_dual_clock" for row in report["stages"])
    assert selected.get_model_object("model_sampling") is sampling
    assert not selected.wrappers
    if locked_audio:
        torch.testing.assert_close(result["samples"].unbind()[1], src["samples"].unbind()[1], rtol=0, atol=1e-6)
    else:
        current, sigma, velocity = seen[0]
        following, next_sigma, _ = seen[1]
        expected = current + velocity * (time_shift_sigma(next_sigma, 12., 3.) - time_shift_sigma(sigma, 12., 3.))
        torch.testing.assert_close(following, expected, rtol=1e-5, atol=1e-6)


def test_independent_high_clock_is_reported_without_rewriting_user_sampling():
    from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
    high, _, _ = setup_dual_clock_sampling(model(), source(), 3, 6., 2., sampler_name="euler")
    selected = high.get_model_object("model_sampling")
    _, raw = sample_taomate_hd(model(), conditioning(), source(), high_model=high)
    report = json.loads(raw)
    assert report["stages"][2]["audio_clock"] == {
        "mode": "native_packed_flow_av", "audio_scale": 3., "video_shift": 6., "audio_shift": 2.}
    assert high.get_model_object("model_sampling") is selected


def test_call_observer_keeps_existing_user_wrapper_and_source_selection():
    selected = model()
    calls = []
    def user_apply(executor, *args, **kwargs):
        calls.append(True)
        return executor(*args, **kwargs)
    selected.add_wrapper_with_key("apply_model", "user_callback", user_apply)
    sampling = selected.get_model_object("model_sampling")
    _, raw = sample_taomate_hd(selected, conditioning(), source())
    assert len(calls) == json.loads(raw)["actual_model_apply_calls"] == 3
    assert selected.get_wrappers("apply_model", "user_callback") == [user_apply]
    assert selected.get_model_object("model_sampling") is sampling
