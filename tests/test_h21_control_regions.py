"""Independent H21-C geometry contracts; real tower probes are added after freeze.

Coverage is target-canvas geometry, never a promise of final-pixel/audio locking.
All numerical checks run on CPU and preserve the original fit implementation.
"""
from copy import deepcopy
import json

import pytest
import torch
from types import SimpleNamespace

import comfy.ops
from comfy.ldm.minimax import model as native
from comfy.ldm.minimax.controlnet import MiniMaxH3FunControl
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
from comfy.ldm.modules import attention
from comfy.model_patcher import ModelPatcher

from h3_audio_t8_pkg import h3_fun_control_advanced as fun
from h3_audio_t8_pkg.conditioning import build_packed_layout
from test_fast_h3_v2_core_sampler import model as tiny_base


def geometry(frames, *, width, height, length, mask=None, crop=None, fit="exact", fit_video=None):
    from h3_audio_t8_pkg.control_regions import prepare_control_geometry
    return prepare_control_geometry(
        frames, width=width, height=height, length=length, fit_mode=fit,
        fit_video=fun._fit_control_video if fit_video is None else fit_video,
        region_mask=mask, crop_plan_json="" if crop is None else json.dumps(crop))


def source_frames(count=3, width=64, height=32):
    x = torch.arange(width, dtype=torch.float32).expand(height, width)/(width-1)
    y = torch.arange(height, dtype=torch.float32).view(height, 1).expand(height, width)/(height-1)
    return torch.stack([torch.stack((torch.full_like(x, index/max(1, count-1)), x, y), dim=-1)
                        for index in range(count)])


def frame_groups(length):
    groups, start, index = [], 0, 0
    while start < length:
        end = start + (1, 4, 4, 4, 4)[index % 5]
        assert end <= length
        groups.append((start, end))
        start, index = end, index + 1
    return groups


@pytest.mark.parametrize("fit,width,height", [("exact", 64, 32), ("stretch", 96, 64), ("center_crop", 96, 64)])
def test_no_explicit_crop_calls_original_fit_without_changing_legacy_values(fit, width, height):
    source = source_frames(2)
    before, calls = source.clone(), []

    def observed_fit(value, **kwargs):
        calls.append((value is source, kwargs))
        return fun._fit_control_video(value, **kwargs)

    actual, gate, info = geometry(source, width=width, height=height, length=5, fit=fit,
                                  fit_video=observed_fit)
    expected = fun._fit_control_video(source, width=width, height=height, length=5, fit_mode=fit)
    assert calls == [(True, {"width": width, "height": height, "length": 5, "fit_mode": fit})]
    assert torch.equal(actual, expected) and gate is None and isinstance(info, dict)
    assert torch.equal(source, before)


@pytest.mark.parametrize("length,latent_t", [(5, 2), (22, 7), (39, 12)])
def test_moving_region_uses_real_temporal_bins_and_asymmetric_spatial_raster(length, latent_t):
    source = torch.zeros(length, 64, 96, 3)
    cells = torch.arange(6, dtype=torch.float32).reshape(2, 3)/5
    mask = torch.arange(length, dtype=torch.float32).view(-1, 1, 1)/(length-1)*cells
    before = mask.clone()
    _, gate, _ = geometry(source, width=96, height=64, length=length, mask=mask)
    rows = gate.rows(latent_t, 4, 6)
    expected = torch.stack([mask[start:end].mean(0) for start, end in frame_groups(length)]).reshape(-1, 1)
    assert rows.shape == (latent_t*2*3, 1) and rows.dtype == torch.float32 and rows.device.type == "cpu"
    torch.testing.assert_close(rows, expected, rtol=0, atol=1e-7)
    assert torch.equal(mask, before)


def test_spatial_partial_cell_is_area_coverage_not_binary_center_or_max():
    mask = torch.zeros(1, 64, 96)
    mask[:, :16, :16] = 1.
    _, gate, _ = geometry(torch.zeros(5, 64, 96, 3), width=96, height=64, length=5, mask=mask)
    expected = torch.zeros(2, 2, 3)
    expected[:, 0, 0] = .25
    torch.testing.assert_close(gate.rows(2, 4, 6).reshape(2, 2, 3), expected, rtol=0, atol=0)


def test_low_high_map_same_original_region_independently_and_refuse_wrong_stage_canvas():
    source = torch.zeros(5, 64, 96, 3)
    mask = torch.zeros(1, 64, 96)
    mask[:, :32, :32] = 1.
    _, low, _ = geometry(source, width=96, height=64, length=5, mask=mask)
    _, high, _ = geometry(source, width=192, height=128, length=5, mask=mask, fit="stretch")
    low_rows, high_rows = low.rows(2, 4, 6).reshape(2, 2, 3), high.rows(2, 8, 12).reshape(2, 4, 6)
    expected_low, expected_high = torch.zeros_like(low_rows), torch.zeros_like(high_rows)
    expected_low[:, 0, 0], expected_high[:, :2, :2] = 1., 1.
    assert torch.equal(low_rows, expected_low) and torch.equal(high_rows, expected_high)
    assert low_rows.mean() == high_rows.mean()
    with pytest.raises(ValueError):
        low.rows(2, 8, 12)
    with pytest.raises(ValueError):
        high.rows(2, 4, 6)


@pytest.mark.parametrize("flip_x,flip_y", [(False, False), (True, False), (False, True), (True, True)])
def test_explicit_crop_and_flips_have_exact_half_open_source_frame_mapping(flip_x, flip_y):
    source = source_frames()
    before = source.clone()
    indices = [2, 0, 1, 2, 0]
    crop = {"schema": "t8.h3.control_crop.v1", "source_size": [64, 32], "frame_indices": indices,
            "boxes": [[16, 0, 48, 32]], "fit": "exact", "flip_x": flip_x, "flip_y": flip_y}
    frozen = deepcopy(crop)
    actual, gate, _ = geometry(source, width=32, height=32, length=5, crop=crop)
    expected = source[indices, :, 16:48, :]
    if flip_x:
        expected = expected.flip(2)
    if flip_y:
        expected = expected.flip(1)
    assert torch.equal(actual, expected) and gate is None
    assert torch.equal(source, before) and crop == frozen


def test_crop_does_not_crop_or_flip_independent_target_region_mask():
    source = source_frames()
    mask = torch.zeros(3, 32, 64)
    mask[:, :, :16] = torch.tensor([0., .5, 1.]).view(3, 1, 1)
    crop = {"frame_indices": [2, 0, 1, 2, 0], "boxes": [[0, 0, 32, 32]],
            "fit": "exact", "flip_x": True}
    actual, gate, _ = geometry(source, width=32, height=32, length=5, crop=crop, mask=mask)
    assert torch.equal(actual, source[[2, 0, 1, 2, 0], :, :32].flip(2))
    # Mask refers to the whole original target canvas: a quarter, not the half
    # obtained by incorrectly cutting the mask with the control-image crop.
    torch.testing.assert_close(gate.rows(2, 2, 2).flatten(), torch.tensor([.25, .09375]), rtol=0, atol=0)


def test_explicit_region_frame_indices_resolve_other_mask_lengths_without_guessing():
    source = source_frames()
    mask = torch.stack((torch.zeros(32, 32), torch.ones(32, 32)))
    crop = {"frame_indices": [0, 0, 1, 2, 2], "region_frame_indices": [1, 0, 0, 0, 0]}
    _, gate, _ = geometry(source, width=64, height=32, length=5, crop=crop, mask=mask)
    expected = torch.tensor([[1., 1.], [0., 0.]]).reshape(-1, 1)
    assert torch.equal(gate.rows(2, 2, 4), expected)
    with pytest.raises(ValueError):
        geometry(source, width=64, height=32, length=5, mask=mask)


@pytest.mark.parametrize("box,zero", [([-16, -16, 16, 16], False), ([80, 40, 112, 72], True)])
def test_out_of_source_crop_is_explicit_zero_padding_not_wrapping(box, zero):
    actual, _, _ = geometry(torch.ones(1, 32, 64, 3), width=32, height=32, length=5,
                            crop={"boxes": [box], "fit": "exact"})
    expected = torch.zeros_like(actual)
    if not zero:
        expected[:, 16:, 16:, :] = 1.
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("fit", ["contain", "cover", "stretch"])
def test_explicit_fit_preserves_constant_pixels_and_declares_padding_geometry(fit):
    actual, _, info = geometry(torch.ones(1, 32, 64, 3), width=32, height=32, length=5,
                              crop={"fit": fit})
    expected = torch.ones_like(actual)
    if fit == "contain":
        expected[:, :8] = 0.
        expected[:, 24:] = 0.
    assert torch.equal(actual, expected) and isinstance(info, dict)


@pytest.mark.parametrize("plan", [
    [], {"schema": "other"}, {"source_size": [65, 32]},
    {"frame_indices": [0]}, {"frame_indices": [0, 0, 0, 0, 3]},
    {"frame_indices": [0, 0, 0, 0, True]}, {"boxes": [[0, 0, 0, 32]]},
    {"boxes": [[32, 0, 0, 32]]}, {"boxes": [[0, 0, float("nan"), 32]]},
    {"fit": "guess"}, {"flip_x": "false"}, {"region_frame_indices": [0]},
])
def test_invalid_crop_fields_are_real_input_errors_not_implicit_choices(plan):
    with pytest.raises(ValueError):
        geometry(source_frames(), width=64, height=32, length=5, crop=plan, mask=torch.ones(1, 32, 64))


@pytest.mark.parametrize("fault", ["nan", "inf", "negative", "above_one", "empty", "wrong_rank"])
def test_invalid_mask_values_and_shape_are_rejected_without_mutation(fault):
    mask = torch.ones(1, 32, 64)
    if fault in {"nan", "inf", "negative", "above_one"}:
        mask.flatten()[0] = {"nan": float("nan"), "inf": float("inf"), "negative": -.01, "above_one": 1.01}[fault]
    elif fault == "empty":
        mask = mask[:0]
    else:
        mask = mask.unsqueeze(-1)
    with pytest.raises(ValueError):
        geometry(source_frames(), width=64, height=32, length=5, mask=mask)


def test_full_region_has_finite_float32_ones_for_each_real_target_row():
    _, gate, _ = geometry(torch.zeros(22, 64, 96, 3), width=96, height=64, length=22,
                          mask=torch.ones(1, 11, 17))
    rows = gate.rows(7, 4, 6)
    assert torch.equal(rows, torch.ones(42, 1)) and rows.dtype == torch.float32 and rows.device.type == "cpu"
    assert not torch.cuda.is_initialized()


def initialized(module, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator)*.035)
    return module.eval().requires_grad_(False)


class RealTinyEncoderOwner:
    """Only adapts IMAGE channel/range to a real miniature Core video encoder."""
    def __init__(self):
        self.core = initialized(MiniMaxH3VideoVAE(ch=32, ch_mult=(1, 1, 1, 1, 1, 1),
                                num_res_blocks=0, decoder=False, tiling=False), 21102131)
        self.calls = []

    def encode(self, frames):
        self.calls.append(frames.clone())
        with torch.no_grad():
            return self.core.encode(frames.permute(3, 0, 1, 2).unsqueeze(0)*2-1)

    @staticmethod
    def spacial_compression_encode():
        return 16


@pytest.fixture(scope="module")
def real_video_owner():
    return RealTinyEncoderOwner()


@pytest.fixture
def tower_case(monkeypatch, real_video_owner):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    selected = tiny_base()
    source = source_frames(5)
    generator = torch.Generator().manual_seed(21102132)
    x = [torch.randn((1, 24, 2, 2, 4), generator=generator),
         torch.randn((1, 32, 2, 8), generator=generator)]
    keys = [{"resolved_frame_index": 0, "latent": torch.randn((1, 24, 1, 2, 4), generator=generator)}]
    refs = [{"kind": "image", "latent_h": 2, "latent_w": 4,
             "latent": torch.randn((1, 24, 1, 2, 4), generator=generator)},
            {"kind": "audio", "ref_audio_t": 3,
             "audio_latent": torch.randn((1, 32, 2, 3), generator=generator)}]
    layout = build_packed_layout(2, 2, 2, 4, 8, keyframes=keys, refs=refs, frame_count=5)
    payload = {"layout": layout, "frame_count": 5, "cond_video_latents": [keys[0]["latent"], refs[0]["latent"]],
               "cond_audio_latents": [refs[1]["audio_latent"]]}
    return SimpleNamespace(base=selected, source=source, vae=real_video_owner, x=x, payload=payload, keys=keys, refs=refs,
                           context=torch.randn((1, 2, 8), generator=generator), positive=[[torch.zeros(1, 2, 8), {}]])


def real_bundle(backend, seed=21102133, layers=(0,)):
    if backend == "official_model_patch":
        tower = MiniMaxH3FunControl(control_in_dim=49, injection_layers=layers, hidden_size=24,
            num_attention_heads=3, attention_head_dim=128, ffn_hidden_size=32, time_embed_dim=24,
            dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
    else:
        tower = fun._MiniMaxH3FunControl(control_in_dim=49, injection_layers=layers, hidden_size=24,
            num_attention_heads=3, attention_head_dim=128, ffn_hidden_size=32, time_embed_dim=24,
            use_adaln_curves=False, dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
    initialized(tower, seed)
    patcher = ModelPatcher(tower, torch.device("cpu"), torch.device("cpu"))
    holder = patcher if backend == "official_model_patch" else {"model": tower, "patcher": patcher}
    return fun.H3FunControlBundle(backend, holder, "user-control.safetensors", "", {}), tower


def apply_control(case, bundle, *, mask=None, old=False, selected=None, strength=.7,
                  start=0., end=1., crop="", stage="SINGLE"):
    args = (case.base if selected is None else selected, case.positive, bundle, case.vae,
            case.source, 64, 32, 5, "pose", "exact", strength, start, end)
    if old:
        model, positive, report = fun.apply_h3_fun_control(*args)
        return model, positive, None, report
    from h3_audio_t8_pkg.control_region_runtime import apply_h3_fun_control_region
    return apply_h3_fun_control_region(*args, region_mask=mask, crop_plan_json=crop, stage=stage)


def actual_forward(case, selected, sigma=.5):
    # Match Core's request boundary before its official Control wrapper saves
    # the currently loaded models. Direct tiny probes otherwise retain dead
    # weakref entries from earlier independent tests.
    comfy.model_management.cleanup_models_gc()
    selected.pre_run()
    selected.prepare_state(torch.tensor([sigma]), selected.model_options)
    options = {**selected.model_options.get("transformer_options", {}), "wrappers": selected.wrappers,
               "sigmas": torch.tensor([sigma]), "sample_sigmas": torch.tensor([1., .5, 0.])}
    try:
        with torch.no_grad():
            output = selected.model.diffusion_model([value.clone() for value in case.x],
                torch.tensor([sigma*1000]), case.context.clone(), transformer_options=options,
                minimax_payload=case.payload)
        assert all(torch.isfinite(value).all() for value in output)
        return output
    finally:
        selected.cleanup()
        assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_actual_tiny_core_and_real_control_tower_full_region_equals_old_control(tower_case, backend):
    case = tower_case
    bundle, _ = real_bundle(backend)
    old, _, _, _ = apply_control(case, bundle, old=True)
    regional, positive, runtime, raw = apply_control(case, bundle, mask=torch.ones(1, 32, 64))
    assert positive is case.positive and runtime.report()["status"] == "not_executed"
    assert not json.loads(raw)["regional_control"]["region_applied"]
    baseline, actual = actual_forward(case, old), actual_forward(case, regional)
    assert all(torch.equal(left, right) for left, right in zip(baseline, actual, strict=True))
    report = runtime.report()
    assert report["residual_calls"] == report["gated_calls"] == 1 and report["target_row_evaluations"] == 4
    assert report["region_applied"] and not report["portable_cache_identity"] and not report["quality_accepted"]


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_real_residual_coverage_changes_only_actual_target_rows_and_retains_foreign_owner(tower_case, backend):
    case = tower_case
    bundle, _ = real_bundle(backend)
    foreign_calls = []

    def foreign(args, extra):
        foreign_calls.append(args["img"].shape)
        return extra["original_block"](args)

    case.base.set_model_patch_replace(foreign, "dit", "double_block", 0)
    mask = torch.zeros(1, 32, 64)
    mask[..., :32] = 1.
    selected, _, runtime, _ = apply_control(case, bundle, mask=mask)
    captured, consume = [], runtime.consume

    def observe_real_skip(skip, args, **kwargs):
        before = skip.clone()
        after = consume(skip, args, **kwargs)
        captured.append((before, after.clone(), kwargs))
        return after

    runtime.consume = observe_real_skip
    actual_forward(case, selected)
    assert len(captured) == len(foreign_calls) == 1
    before, after, kwargs = captured[0]
    start, stop, _ = next(row for row in case.payload["layout"].segments if row[2] == "video")
    weights = torch.tensor([1., 0., 1., 0.]).view(-1, 1)
    if kwargs.get("full_packed"):
        expected = before.clone()
        expected[start:stop] *= weights
        assert torch.equal(before[:start], after[:start]) and torch.equal(before[stop:], after[stop:])
        # This genuine tower has nonzero text/ref skips; preserving them is not
        # the same thing as clearing all packed rows outside target video.
        audio_rows = case.payload["layout"].audio_pos
        not_audio_or_target = torch.ones(before.shape[0], dtype=torch.bool)
        not_audio_or_target[audio_rows] = False
        not_audio_or_target[start:stop] = False
        assert torch.count_nonzero(before[not_audio_or_target]) > 0
    else:
        expected = before * weights
    assert torch.equal(after, expected) and torch.count_nonzero(before) > 0
    assert runtime.report()["gated_calls"] == 1
    assert case.base.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)] is foreign


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_multiple_real_controls_have_independent_runtime_and_full_mask_old_parity(tower_case, backend):
    case = tower_case
    first, _ = real_bundle(backend)
    second, _ = real_bundle(backend, 21102134)
    old1 = apply_control(case, first, old=True)[0]
    old2 = apply_control(case, second, old=True, selected=old1)[0]
    new1, _, r1, _ = apply_control(case, first, mask=torch.ones(1, 32, 64), stage="LOW")
    new2, _, r2, _ = apply_control(case, second, mask=torch.ones(1, 32, 64), selected=new1, stage="HIGH")
    baseline, actual = actual_forward(case, old2), actual_forward(case, new2)
    assert all(torch.equal(left, right) for left, right in zip(baseline, actual, strict=True))
    assert r1 is not r2 and r1.report()["stage"] == "LOW" and r2.report()["stage"] == "HIGH"
    assert r1.report()["gated_calls"] == r2.report()["gated_calls"] == 1


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_inactive_sigma_does_not_count_configured_control_as_executed(tower_case, backend):
    case = tower_case
    bundle, tower = real_bundle(backend)
    after_proj_calls = []
    handle = tower.control_blocks[0].after_proj.register_forward_hook(lambda *args: after_proj_calls.append(True))
    try:
        selected, _, runtime, _ = apply_control(case, bundle, mask=torch.ones(1, 32, 64), end=.5)
        actual_forward(case, selected, sigma=.01)
    finally:
        handle.remove()
    assert after_proj_calls == [] and runtime.report()["residual_calls"] == 0
    assert not runtime.report()["region_applied"] and runtime.report()["status"] == "not_executed"


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_real_tower_exception_propagates_once_then_fresh_request_resets_state(tower_case, backend):
    case = tower_case
    bundle, tower = real_bundle(backend)
    selected, _, runtime, _ = apply_control(case, bundle, mask=torch.ones(1, 32, 64))
    expected = actual_forward(case, selected)
    failures = []

    def fail_after_actual_projection(module, inputs, output):
        failures.append(output.clone())
        raise RuntimeError("injected actual post-projection owner failure")

    handle = tower.control_blocks[0].after_proj.register_forward_hook(fail_after_actual_projection)
    try:
        with pytest.raises(RuntimeError, match="actual post-projection owner failure"):
            actual_forward(case, selected)
    finally:
        handle.remove()
    assert len(failures) == 1 and runtime.report()["status"] == "failed"
    recovered = actual_forward(case, selected)
    assert all(torch.equal(left, right) for left, right in zip(expected, recovered, strict=True))
    assert runtime.report()["errors"] == [] and runtime.report()["gated_calls"] == 1


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_new_strength_zero_is_same_model_conditioning_no_real_vae_or_tower_execution(tower_case, backend):
    case = tower_case
    bundle, _ = real_bundle(backend)
    before = len(case.vae.calls)
    selected, positive, runtime, raw = apply_control(case, bundle, mask=torch.ones(1, 32, 64), strength=0.)
    assert selected is case.base and positive is case.positive and len(case.vae.calls) == before
    assert runtime.report()["bypass_reason"] == "strength_zero" and runtime.report()["residual_calls"] == 0
    assert json.loads(raw)["status"] == "bypass"


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_actual_ordinary_stage_sampler_keeps_region_callbacks_executable_without_portable_claim(tower_case, backend):
    from comfy.nested_tensor import NestedTensor
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
    from h3_audio_t8_pkg import sampling
    from h3_audio_t8_pkg.modular_sampling import native_explicit
    from h3_audio_t8_pkg.modular_sampling.results import sample_stage, selected_model_identity

    case = tower_case
    case.positive = [[case.context.clone(), {"minimax_token_tags": torch.ones(2, dtype=torch.long),
        "minimax_keyframes": case.keys, "minimax_refs": case.refs, "minimax_frame_count": 5}]]
    source = {"samples": NestedTensor(tuple(value.clone() for value in case.x))}
    source_before = [value.clone() for value in source["samples"].unbind()]
    bundle, _ = real_bundle(backend)
    selected, positive, runtime, _ = apply_control(case, bundle, mask=torch.ones(1, 7, 17))
    prepared, sampler, sigmas = sampling.setup_dual_clock_sampling(selected, source, 2, 12., 3.)
    prepared, sampler, sigmas, context, _ = native_explicit.bind_stage(prepared, sampler, sigmas, source)
    identity = selected_model_identity(prepared)
    if backend == "compatibility":
        assert identity["schema"] == "t8.modular-sampling.model-effects.v1"
        contract = identity["stage_effects"]["regional_control"]
        assert contract["schema"] == "t8.h3.control-owner-content.v1"
        assert len(contract["controls"]) == 1
        assert contract["controls"][0]["geometry"] == runtime.geometry
        assert contract["controls"][0]["encoded_rows"] is not None
        assert "execution_selection" not in identity and identity.get("portable_cache_reuse") is not False
    else:
        assert identity["portable_cache_reuse"] is False
    noise = RandomNoise.execute(21102135).result[0]
    guider = BasicGuider.execute(prepared, positive).result[0]
    output, _, stage_result, _ = sample_stage(noise, guider, sampler, sigmas, source, context)
    receipt = stage_result.verify()
    assert receipt["execution"]["callbacks"] == [0, 1]
    assert receipt["execution"]["denoiser_evaluations"] == 2
    assert receipt["verified_recipe_completion"]
    assert receipt["portable_identity"] is (backend == "compatibility")
    assert not receipt["cache_reuse_authorized"] and runtime.report()["gated_calls"] == 2
    assert all(torch.isfinite(value).all() for value in output["samples"].unbind())
    assert all(torch.equal(before, actual) for before, actual in zip(source_before, source["samples"].unbind(), strict=True))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_actual_crop_reaches_real_vae_and_control_without_mask_or_output_compositing(tower_case, backend):
    case = tower_case
    bundle, _ = real_bundle(backend)
    crop = {"boxes": [[0, 0, 32, 32]], "fit": "stretch", "flip_x": True}
    prepared_frames, gate, geometry_report = geometry(case.source, width=64, height=32, length=5, crop=crop)
    assert gate is None and not geometry_report["output_rgb_compositing"]
    original = case.source.clone()
    manual_case = SimpleNamespace(**{**vars(case), "source": prepared_frames})
    old = apply_control(manual_case, bundle, old=True)[0]
    selected, _, runtime, _ = apply_control(case, bundle, crop=json.dumps(crop))
    assert runtime.report()["crop_prepared"] and not runtime.report()["crop_consumed_by_observed_control"]
    baseline, actual = actual_forward(case, old), actual_forward(case, selected)
    assert all(torch.equal(left, right) for left, right in zip(baseline, actual, strict=True))
    assert runtime.report()["crop_consumed_by_observed_control"] and runtime.report()["residual_calls"] == 1
    assert not runtime.report()["region_requested"] and not runtime.report()["region_applied"]
    assert torch.equal(case.source, original)


def test_real_compat_tower_cannot_reuse_previous_forward_stream_when_first_hook_is_bypassed(tower_case):
    case = tower_case
    network = case.base.model.diffusion_model
    network.blocks.append(deepcopy(network.blocks[0]))
    bundle, tower = real_bundle("compatibility", layers=(0, 1))
    selected, _, runtime, _ = apply_control(case, bundle, mask=torch.ones(1, 32, 64))
    foreign_calls, projections = [], []
    handles = [block.after_proj.register_forward_hook(lambda *args: projections.append(True))
               for block in tower.control_blocks]

    def forward_without_request_cleanup(model):
        options = {**model.model_options.get("transformer_options", {}), "wrappers": model.wrappers,
                   "sigmas": torch.tensor([.5]), "sample_sigmas": torch.tensor([1., .5, 0.])}
        with torch.no_grad():
            return network([value.clone() for value in case.x], torch.tensor([500.]), case.context.clone(),
                           transformer_options=options, minimax_payload=case.payload)

    def bypass_first_control(args, extra):
        foreign_calls.append(True)
        return extra["original_block"](args)

    selected.pre_run()
    try:
        forward_without_request_cleanup(selected)
        assert runtime.report()["gated_calls"] == 2 and len(projections) == 2
        selected.set_model_patch_replace(bypass_first_control, "dit", "double_block", 0)
        second = forward_without_request_cleanup(selected)
        baseline = forward_without_request_cleanup(case.base)
        assert len(foreign_calls) == 1 and len(projections) == 2
        assert runtime.report()["gated_calls"] == 2  # no extra residual from stale state
        assert all(torch.equal(left, right) for left, right in zip(second, baseline, strict=True))
    finally:
        selected.cleanup()
        for handle in handles:
            handle.remove()
    assert not torch.cuda.is_initialized()
