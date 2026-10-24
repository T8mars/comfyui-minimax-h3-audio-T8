"""Actual native packed-row gating and append-only UI/API contracts, CPU only."""
import asyncio
import json
from pathlib import Path

import pytest
import torch
from comfy.ldm.minimax.model import PackedLayout

import h3_audio_t8_pkg as package
from h3_audio_t8_pkg import h3_fun_control_advanced as fun
from h3_audio_t8_pkg.control_regions import prepare_control_geometry
from h3_audio_t8_pkg.control_region_runtime import ControlRegionRuntime, apply_h3_fun_control_region, target_span
from h3_audio_t8_pkg.nodes_control_region import NODES, MiniMaxH3ControlRegionAuditEXPT8
from h3_audio_t8_pkg.nodes_h3_fun_control_advanced import MiniMaxH3FunControlApplyT8Advanced


def layout():
    return PackedLayout(3, 2, 4, 6, 5,
        keyframes=[dict(resolved_frame_index=0, latent=torch.zeros(1, 24, 1, 4, 6))],
        refs=[dict(kind="image", latent_h=4, latent_w=6), dict(kind="audio", ref_audio_t=3)])


def state(mask):
    _, gate, geometry = prepare_control_geometry(torch.zeros(5, 64, 96, 3),
        width=96, height=64, length=5, fit_mode="exact", fit_video=fun._fit_control_video, region_mask=mask)
    return ControlRegionRuntime(stage="LOW", gate=gate, geometry=geometry)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_full_region_preserves_actual_packed_skip_bitwise(dtype):
    native = layout()
    runtime = state(torch.ones(1, 64, 96))
    skip = torch.randn(native.seq_len, 7).to(dtype)
    before = skip.clone()
    result = runtime.consume(skip, {"layout": native}, full_packed=True)
    assert result is skip and torch.equal(result, before)
    assert runtime.report()["gated_calls"] == 1
    assert runtime.report()["target_row_evaluations"] == 12
    assert not runtime.report()["portable_cache_identity"]


@pytest.mark.parametrize("height,width", [(7, 17), (13, 19), (108, 191), (1, 1)])
def test_full_region_is_bitwise_identity_even_when_original_mask_does_not_divide_token_grid(height, width):
    native = layout()
    runtime = state(torch.ones(1, height, width))
    skip = torch.randn(native.seq_len, 7)
    before = skip.clone()
    runtime.consume(skip, {"layout": native}, full_packed=True)
    assert torch.equal(skip, before)


def test_large_finite_virtual_crop_is_zero_padding_not_nonfinite_pixels():
    source = torch.ones(1, 32, 64, 3)
    frames, _, _ = prepare_control_geometry(source, width=64, height=32, length=5,
        fit_mode="stretch", fit_video=fun._fit_control_video,
        crop_plan_json=json.dumps({"boxes": [[1e300, 1e300, 2e300, 2e300]], "fit": "stretch"}))
    assert torch.isfinite(frames).all() and torch.count_nonzero(frames) == 0


@pytest.mark.parametrize("strength", [float("nan"), float("inf")])
def test_nonfinite_control_strength_is_a_real_input_error_not_success(strength):
    bundle = fun.H3FunControlBundle("compatibility", {}, "user", "user", {})
    with pytest.raises(ValueError, match="finite.*strength|strength.*finite"):
        apply_h3_fun_control_region(object(), [], bundle, object(), torch.zeros(5, 32, 32, 3),
            32, 32, 5, "custom", "exact", strength, 0., 1.)


def test_zero_region_changes_only_target_video_not_old_text_reference_or_audio_skip():
    native = layout()
    runtime = state(torch.zeros(1, 64, 96))
    skip = torch.arange(native.seq_len * 3, dtype=torch.float32).reshape(-1, 3)
    before = skip.clone()
    start, stop, *_ = target_span({"layout": native})
    runtime.consume(skip, {"layout": native}, full_packed=True)
    assert torch.count_nonzero(skip[start:stop]) == 0
    assert torch.equal(skip[:start], before[:start])
    assert torch.equal(skip[stop:], before[stop:])


def test_compatibility_local_skip_uses_same_non_square_spatial_rows():
    native = layout()
    mask = torch.zeros(1, 64, 96)
    mask[:, :32, :32] = 1
    runtime = state(mask)
    start, stop, *_ = target_span({"layout": native})
    skip = torch.ones(stop - start, 4)
    out = runtime.consume(skip, {"layout": native}, span=(start, stop))
    expected = torch.tensor([1, 0, 0, 0, 0, 0] * 2).float()[:, None].expand(-1, 4)
    assert torch.equal(out, expected) and torch.equal(skip, torch.ones_like(skip))


@pytest.mark.parametrize("fault", ["span", "update_mask", "rows", "canvas"])
def test_real_malformed_layout_or_target_shape_still_errors(fault):
    native = layout()
    runtime = state(torch.ones(1, 64, 96))
    start, stop, *_ = target_span({"layout": native})
    skip = torch.ones(stop - start, 4)
    span = (start, stop)
    if fault == "span":
        span = (start + 1, stop + 1)
    elif fault == "update_mask":
        native.img_update[0] = True
    elif fault == "rows":
        skip = skip[:-1]
    else:
        runtime.gate.width = 128
    with pytest.raises(ValueError, match="region|Region|residual"):
        runtime.consume(skip, {"layout": native}, span=span)


@pytest.mark.parametrize("options,expected_reason", [
    ({}, "target_layout_unavailable_region_not_applied"),
    ({"sol_morton": True}, "non_raster_Morton_region_not_applied"),
])
def test_unknown_layout_or_selected_morton_keeps_original_control_not_fake_gate(options, expected_reason):
    runtime = state(torch.zeros(1, 64, 96))
    skip = torch.ones(12, 4)
    assert runtime.consume(skip, {"transformer_options": options}) is skip
    assert torch.all(skip == 1)
    report = runtime.report()
    assert report["region_requested"] and not report["region_applied"]
    assert report["preserved_reasons"] == {expected_reason: 1}


def test_strength_zero_keeps_both_objects_and_does_not_encode_or_parse_unused_crop():
    class VAE:
        def encode(self, *_):
            raise AssertionError("A zero strength must never encode")
    model, positive = object(), [[torch.zeros(1), {}]]
    bundle = fun.H3FunControlBundle("compatibility", {}, "user model", "user model", {})
    result, cond, runtime, text = apply_h3_fun_control_region(
        model, positive, bundle, VAE(), torch.zeros(5, 32, 32, 3), 32, 32, 5,
        "custom", "exact", 0., 0., 1., region_mask=torch.ones(1, 1, 1), crop_plan_json="invalid unused draft")
    assert result is model and cond is positive
    assert runtime.report()["bypass_reason"] == "strength_zero"
    assert json.loads(text)["model_identity_preserved"]


def test_native_graph_schema_is_additive_and_audit_is_actual_dependency_passthrough():
    current = asyncio.run(package.comfy_entrypoint().get_node_list())
    fixture = json.loads((Path(__file__).parent / "fixtures/h21_control_node_prefix.json").read_text(encoding="utf8"))
    assert [node.__name__ for node in current[:len(fixture["node_ids"])]] == fixture["node_ids"]
    assert current[len(fixture["node_ids"]):len(fixture["node_ids"]) + len(NODES)] == NODES
    assert len({node.__name__ for node in current}) == len(current)
    old = MiniMaxH3FunControlApplyT8Advanced.define_schema().get_v1_info(MiniMaxH3FunControlApplyT8Advanced)
    info = NODES[0].define_schema().get_v1_info(NODES[0])
    assert list(info.input["required"])[:len(old.input["required"])] == list(old.input["required"])
    expected = {key: (value[0], dict(value[1])) for key, value in old.input["required"].items()}
    for name in ("width", "height", "strength"):
        expected[name][1].pop("max", None)
    expected["strength"][1].pop("min", None)
    assert {key: info.input["required"][key] for key in old.input["required"]} == expected
    assert MiniMaxH3FunControlApplyT8Advanced.define_schema().get_v1_info(
        MiniMaxH3FunControlApplyT8Advanced).input == old.input
    assert old.input["required"]["strength"][1]["min"] == 0.
    assert old.input["required"]["strength"][1]["max"] == 2.
    assert info.input["optional"]["region_mask"][0] == "MASK"
    runtime = state(torch.ones(1, 64, 96))
    sample = {"samples": torch.ones(1)}
    output = MiniMaxH3ControlRegionAuditEXPT8.execute(runtime, sample).result
    assert output[0] is sample and json.loads(output[1])["status"] == "not_executed"
    assert not torch.cuda.is_initialized()


def test_runtime_reset_and_cleanup_preserve_counters_until_next_request():
    runtime = state(torch.ones(1, 64, 96))
    callbacks = []
    runtime.cleanup_callbacks.append(lambda: callbacks.append("clean"))
    runtime.pre_run()
    native = layout()
    runtime.consume(torch.ones(native.seq_len, 2), {"layout": native}, full_packed=True)
    runtime.errors.append("actual failed kernel")
    runtime.cleanup()
    assert runtime.calls == runtime.gated_calls == 1 and runtime.report()["status"] == "failed"
    runtime.pre_run()
    assert runtime.run == 2 and not runtime.errors and runtime.calls == 0
    assert len(callbacks) == 3
