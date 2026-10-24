"""Only new R2VA contracts; no downloads, GPU, old media matrix or weights."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from comfy.ldm.minimax.model import PackedLayout
from comfy.ldm.modules import attention as core_attention

import h3_audio_t8_pkg as package
from h3_audio_t8_pkg import nodes_veda_r2va as nodes
from h3_audio_t8_pkg import veda_r2va_bundle as bundle
from h3_audio_t8_pkg import veda_r2va_runtime as runtime
from h3_audio_t8_pkg import veda_r2va_flex as flex
from h3_audio_t8_pkg import veda_r2va_identity as identity
from h3_audio_t8_pkg.veda_r2va_layout import describe, segment_grid, select_plan, Layout, Span, PlanChoice
from h3_audio_t8_pkg.veda_vendor.veda import mask, plan, tiling, predictor
from test_veda_sparse_exp import _fake_h3_model


def metadata():
    p = {"geometry": "16x9_t37", "grid": [37, 24, 42], "shapes": ["2x8x8"],
         "head_shape": [[0]*56 for _ in range(50)]}
    return dict(format=bundle.FORMAT, num_layers="50", num_heads="56", head_dim="128",
                dtype="float8_e4m3fn", tile_conditions="true",
                target_budget_kind="tiles", target_budget_value="32.0",
                ref_budget_kind="tiles", ref_budget_value="32.0", plans=json.dumps({p["geometry"]: p}))


def fake_bundle(tmp_path, monkeypatch):
    file = tmp_path / "predictor.safetensors"
    file.write_bytes(b"not-a-real-predictor-test-only")
    ref = bundle.R2VABundleRef(file, bundle.file_sha(file), metadata())
    monkeypatch.setattr(bundle, "inspect_bundle", lambda path: ref)
    return ref


def native_layout():
    return PackedLayout(3, 2, 4, 4, 2,
        keyframes=[dict(resolved_frame_index=0, latent=torch.zeros(1, 24, 1, 4, 4),
                        audio_latent=torch.zeros(1, 32, 2))],
        refs=[dict(kind="image", latent_h=4, latent_w=6),
              dict(kind="video_audio", latent_h=4, latent_w=4, latent_t=2, ref_audio_t=2),
              dict(kind="audio", ref_audio_t=3)])


def test_append_only_three_isolated_nodes_and_safe_defaults():
    all_nodes = asyncio.run(package.comfy_entrypoint().get_node_list())
    # These were the tail when R2VA shipped; later append-only node groups must
    # not make the original contiguous registration contract fail forever.
    predecessor = package._h07_delivery_node_classes[-1]
    start = all_nodes.index(predecessor) + 1
    assert all_nodes[start:start + 3] == nodes.NODES
    assert start > all_nodes.index(package._veda_sparse_node_classes[-1])
    assert len({n.__name__ for n in all_nodes}) == len(all_nodes)
    info = nodes.MiniMaxH3VedaR2VAApplyEXPT8.define_schema().get_v1_info(nodes.MiniMaxH3VedaR2VAApplyEXPT8)
    assert info.input["required"]["mode"][1]["default"] == "report_only"
    assert info.input["required"]["plan_mode"][1]["default"] == "strict_exact"
    assert info.input["required"]["bundle"][0] == "T8_VEDA_R2VA_BUNDLE"
    from h3_audio_t8_pkg.nodes_veda_sparse_exp import MiniMaxH3VedaApplyEXPT8
    old = MiniMaxH3VedaApplyEXPT8.define_schema().get_v1_info(MiniMaxH3VedaApplyEXPT8)
    assert old.input["required"]["bundle"][0] == "T8_VEDA_BUNDLE"
    assert old.input["required"]["keep_ratio"][1]["default"] == .1


def test_metadata_without_legacy_keep_ratio_and_two_budgets():
    m = metadata()
    ref = bundle.R2VABundleRef(Path("r2va.safetensors"), bundle.OFFICIAL_SHA256, m)
    assert ref.report()["trained_reference_budget"] == dict(ratio=None, tiles=32.)
    assert runtime.budget("trained", m, "target") == mask.Budget(tiles=32)
    assert runtime.budget("0%", m, "ref").keeps_all
    assert not runtime.budget("32", m, "ref").keeps_all
    for text in ("-1", "0", "nan", "10%", "4097"):
        with pytest.raises(ValueError):
            runtime.budget(text, m, "ref")
    with pytest.raises(ValueError):
        bundle.trained_budget({**m, "target_budget_value": "nan"}, "target")
    damaged = json.loads(m["plans"])
    damaged["16x9_t37"]["shapes"] = ["-2x-8x8"]
    with pytest.raises(ValueError, match="geometry/heads"):
        bundle.read_plans({**m, "plans": json.dumps(damaged)})


def test_real_native_image_video_audio_and_keyframe_layout():
    layout = native_layout()
    original = layout.position_ids.clone()
    spec = describe(layout, layout.seq_len)
    assert len(spec.references) == 3 and not spec.skipped
    assert [s.grid for s in spec.references] == [(1, 2, 2), (1, 2, 3), (2, 2, 2)]
    tiles = flex.tile_layout(spec, tiling.TileShape(2, 8, 8), mask.Budget(tiles=32), "cpu", {})
    visual_rows = set(tiles.perm[:tiles.n_video_tiles*128].tolist()) - {-1}
    for start, stop, kind in layout.segments:
        if "audio" in kind or kind == "text":
            assert not (set(range(start, stop)) & visual_rows)
    assert torch.equal(layout.position_ids, original)
    dense_ref = flex.tile_layout(spec, tiling.TileShape(2, 8, 8), mask.Budget(ratio=1), "cpu", {})
    assert dense_ref.n_ref_tiles == 0 and tiles.n_ref_tiles > 0


def test_unverifiable_reference_is_global_without_mutating_order():
    layout = native_layout()
    start, stop, _ = next(r for r in layout.segments if r[2] == "ref_img")
    layout.position_ids[start:stop] = layout.position_ids[start:stop].flip(0)
    spec = describe(layout, layout.seq_len)
    assert len(spec.references) == 2 and len(spec.skipped) == 1
    assert spec.skipped[0][-1] == "non_row_major_position_grid"
    with pytest.raises(ValueError, match="row_major"):
        segment_grid(layout.position_ids[start:stop])


def test_reference_above32_has_own_cap_even_when_target_scores_win():
    # Exactly 32 ref would miss the independent-selection bug; use >32.
    spans = [tiling.TiledSpan(3, (2, 64, 64), tiling.TileShape(2, 8, 8)),
             tiling.TiledSpan(8195, (2, 64, 64), tiling.TileShape(2, 8, 8))]
    tiles = tiling.build_tile_layout(spans, 16387, 16387)
    scores = torch.zeros(1, tiles.n_video_tiles, tiles.n_video_tiles)
    scores[:, :, tiles.n_ref_tiles:] = 1000
    selected = mask.dense_block_mask(mask.select_video_blocks(scores, tiles,
        mask.column_blocks(tiles, mask.Budget(tiles=32), mask.Budget(tiles=32))), tiles)
    assert tiles.n_ref_tiles == 64
    assert (selected[:, :tiles.n_video_tiles, :64].sum(-1) == 32).all()
    assert (selected[:, :tiles.n_video_tiles, 64:128].sum(-1) == 32).all()
    assert selected[:, 128:].all() and selected[:, :, 128:].all()
    assert selected[0].diagonal()[:128].all()


def test_nearest_is_explicit_and_keeps_actual_grid():
    table = bundle.read_plans(metadata())
    with pytest.raises(ValueError, match="exact plan"):
        select_plan(table, (37, 15, 27), "strict_exact")
    choice = select_plan(table, (37, 15, 27), "nearest_exp")
    assert not choice.exact and choice.actual_grid == (37, 15, 27)
    assert tuple(choice.plan.grid) == (37, 24, 42)
    assert select_plan(table, (37, 24, 42), "strict_exact").exact


def test_report_clones_and_mask_and_partial_query_keep_exact_math(tmp_path, monkeypatch):
    ref = fake_bundle(tmp_path, monkeypatch)
    original = _fake_h3_model()
    patched, state = runtime.apply_r2va(original, ref, plan_mode="nearest_exp")
    assert patched is not original
    assert "optimized_attention_override" not in original.model_options["transformer_options"]
    owner = patched.model_options["transformer_options"]["optimized_attention_override"]
    layout = native_layout()
    q, k, v = (torch.randn(1, 56, layout.seq_len, 128) for _ in range(3))
    opts = dict(minimax_h3_layout=layout, block_index=0)
    for partial, bias in ((False, None), (False, torch.zeros(layout.seq_len, layout.seq_len)),
                          (True, torch.randn(3, layout.seq_len)*.1)):
        query = q[:, :, :3] if partial else q
        kwargs = dict(skip_reshape=True, transformer_options=opts, mask=bias)
        actual = owner(core_attention.optimized_attention, query, k, v, 56, **kwargs)
        expected = core_attention.optimized_attention(query, k, v, 56, **kwargs)
        assert torch.equal(actual, expected)
    assert state.delegates == {"report_only": 1, "mask_preserved": 1, "partial_query_preserved": 1}
    assert state.sparse_calls == 0
    result = nodes.MiniMaxH3VedaR2VAAuditEXPT8.execute({"samples": q}, state)
    assert '"sparse_calls": 0' in result[1]


def test_foreign_owner_is_not_replaced_and_does_not_require_kernel(tmp_path, monkeypatch):
    original = _fake_h3_model()
    def foreign(*args, **kwargs):
        return args[0]
    original.model_options["transformer_options"]["optimized_attention_override"] = foreign
    monkeypatch.setattr(runtime, "require_flex_runtime", lambda _: pytest.fail("must preserve foreign owner"))
    patched, state = runtime.apply_r2va(original, fake_bundle(tmp_path, monkeypatch), mode="apply_exp")
    assert patched.model_options["transformer_options"]["optimized_attention_override"] is foreign
    assert state.snapshot()["status"] == "bypassed_owner_preserved"
    assert state.snapshot()["sparse_calls"] == 0


def test_multispan_flex_dense_limit_and_padding_match_actual_sdpa(monkeypatch):
    from torch.nn.attention.flex_attention import flex_attention
    monkeypatch.setattr(flex.veda_flex, "compiled_flex", lambda: flex_attention)
    spec = Layout(30, Span(14, (1, 4, 4)), (Span(2, (1, 3, 4)),), ())
    p = plan.TilePlan("tiny-not-H3-inference", (1, 4, 4), [tiling.TileShape(1, 8, 16)], [[0, 0]])
    choice = PlanChoice(p, spec.target.grid, True)
    net = predictor.TileScorePredictor(1, 2, 8).eval()
    q, k, v = (torch.randn(30, 2, 8) for _ in range(3))
    actual = flex.attend(q, k, v, layer_index=0, spec=spec, choice=choice, predictor=net,
        generated_budget=mask.Budget(ratio=1), reference_budget=mask.Budget(tiles=32),
        max_copy_mib=1, tile_cache={}, statistics={})
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None])[0].transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("damage", ["missing_scale", "scale_shape", "scale_dtype"])
def test_fp8_descriptor_rejects_bad_scales_before_loading(tmp_path, monkeypatch, damage):
    file = tmp_path / "bad.safetensors"
    file.write_bytes(b"test-only-header-stub")
    keys = {f"layers.{i}.proj_{side}": ([56, 384, 128], "F8_E4M3")
            for i in range(50) for side in ("q", "k")}
    keys.update({k + bundle.SCALE_SUFFIX: ([56], "F32") for k in list(keys)})
    name = "layers.0.proj_q.__scale"
    if damage == "missing_scale":
        keys.pop(name)
    else:
        keys[name] = ([55], "F32") if damage == "scale_shape" else ([56], "BF16")
    class Header:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def metadata(self):
            return metadata()
        def keys(self):
            return keys.keys()
        def get_slice(self, key):
            return SimpleNamespace(get_shape=lambda: keys[key][0], get_dtype=lambda: keys[key][1])
        def get_tensor(self, key):
            pytest.fail("bad headers must be rejected without loading tensors")
    monkeypatch.setattr(bundle, "safe_open", lambda *args, **kwargs: Header())
    with pytest.raises(ValueError, match="tensor/scale|shape/dtype"):
        bundle.inspect_bundle(file)


def test_sparse_route_passes_new_geometry_budgets_and_owns_execution_audit(tmp_path, monkeypatch):
    ref = fake_bundle(tmp_path, monkeypatch)
    net = torch.nn.Linear(1, 1, bias=False).requires_grad_(False).eval()
    loaded = bundle.LoadedR2VA(net, bundle.read_plans(ref.metadata), ref.metadata, identity.parameter_hash(net))
    monkeypatch.setattr(bundle, "load", lambda _: loaded)
    monkeypatch.setattr(runtime, "require_flex_runtime", lambda _: None)
    import comfy.cli_args
    monkeypatch.setattr(comfy.cli_args, "enables_dynamic_vram", lambda: False)
    calls = []
    def attend(q, k, v, **kwargs):
        calls.append(kwargs)
        return q
    monkeypatch.setattr(runtime, "attend", attend)
    patched, state = runtime.apply_r2va(_fake_h3_model(), ref, mode="apply_exp", plan_mode="nearest_exp")
    layout = native_layout()
    q = torch.randn(1, 56, layout.seq_len, 128)
    owner = patched.model_options["transformer_options"]["optimized_attention_override"]
    actual = owner(core_attention.optimized_attention, q, q, q, 56, skip_reshape=True,
                   transformer_options=dict(minimax_h3_layout=layout, block_index=0))
    assert torch.equal(actual, q.transpose(1, 2).reshape(1, layout.seq_len, 56*128))
    assert calls[0]["reference_budget"] == calls[0]["generated_budget"] == mask.Budget(tiles=32)
    assert calls[0]["spec"].references and calls[0]["choice"].actual_grid == (2, 2, 2)
    report = state.snapshot()
    assert report["sparse_calls"] == 0 and report["delegated_calls"] == 0
    assert report["sparse_attempts"] == report["unverified_sparse_calls"] == 1
    assert report["status"] == "unverified_executable_preserved" and report["executed_sparse_kernel"] is None
    assert "replaced_runtime_attend" in report["execution_identity"]["unverified_executable_reasons"]
    assert report["execution_identity"]["observed_layouts"]
    assert report["execution_identity"]["portable_cache_reuse"] is False
    from h3_audio_t8_pkg import veda_identity
    assert veda_identity.project(patched) == (patched, None)
    with torch.no_grad():
        net.weight.add_(1)
    with pytest.raises(ValueError, match="parameters changed"):
        state.snapshot()


def test_execution_audit_rejects_changed_settings_and_derived_cache(tmp_path, monkeypatch):
    _, state = runtime.apply_r2va(_fake_h3_model(), fake_bundle(tmp_path, monkeypatch))
    state.generated = mask.Budget(tiles=17)
    with pytest.raises(ValueError, match="configuration/implementation"):
        state.snapshot()
    spec = Layout(30, Span(14, (1, 4, 4)), (Span(2, (1, 3, 4)),), ())
    cache = {}
    tiles = flex.tile_layout(spec, tiling.TileShape(1, 8, 16), mask.Budget(tiles=32), "cpu", cache)
    assert identity.cache_records(cache)[0]["content_sha256"]
    tiles.valid_count[0] -= 1
    with pytest.raises(ValueError, match="tile cache changed"):
        identity.cache_records(cache)


@pytest.mark.parametrize("mutation", ["positions", "segments"])
def test_audit_rejects_mutated_live_native_layout(tmp_path, monkeypatch, mutation):
    patched, state = runtime.apply_r2va(_fake_h3_model(), fake_bundle(tmp_path, monkeypatch), plan_mode="nearest_exp")
    layout = native_layout()
    q = torch.randn(1, 56, layout.seq_len, 128)
    owner = patched.model_options["transformer_options"]["optimized_attention_override"]
    owner(core_attention.optimized_attention, q, q, q, 56, skip_reshape=True,
          transformer_options=dict(minimax_h3_layout=layout, block_index=0))
    state.snapshot()
    if mutation == "positions":
        layout.position_ids[:, 0] += 1
    else:
        layout.segments = [(start, stop, "unverified_ref" if kind == "ref_img" else kind)
                           for start, stop, kind in layout.segments]
    with pytest.raises(ValueError, match="native layout/position_ids changed"):
        state.snapshot()


def test_predictor_hook_and_compiled_owner_are_explicitly_unverified(monkeypatch):
    with torch.device("meta"):
        network = predictor.TileScorePredictor(50, 56, 128).eval().requires_grad_(False)
    loaded = SimpleNamespace(predictor=network)
    monkeypatch.setattr(flex.veda_flex, "_COMPILED", None)
    assert identity.executable_issues(loaded) == []
    network.layers[0].register_forward_hook(lambda *args: None)
    assert "unverified_predictor_callable_or_hook" in identity.executable_issues(loaded)
    monkeypatch.setattr(flex.veda_flex, "_COMPILED", lambda *args, **kwargs: args[0])
    assert "unverified_compiled_Flex_owner" in identity.executable_issues(loaded, require_compiled=True)
