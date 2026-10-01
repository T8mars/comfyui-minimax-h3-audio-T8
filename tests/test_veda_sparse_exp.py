"""Opt-in Veda nodes: pinned plans, Core layout guard and tile-mask semantics."""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from comfy.model_patcher import ModelPatcher

import h3_audio_t8_pkg as package
from h3_audio_t8_pkg.veda_flex import (
    _windows_inductor_text_encoding_supported, flex_block_mask,
)
from h3_audio_t8_pkg import veda_runtime
from h3_audio_t8_pkg import veda_flex
from h3_audio_t8_pkg.nodes_veda_sparse_exp import MiniMaxH3VedaAuditEXPT8
from h3_audio_t8_pkg import nodes_veda_sparse_exp as veda_nodes
from h3_audio_t8_pkg.veda_runtime import VedaBundleRef, VedaRuntime, _t2va_layout, exact_plan
from h3_audio_t8_pkg.veda_vendor.h3.geometry import Geometry
from h3_audio_t8_pkg.veda_vendor.veda import mask, plan, tiling


@pytest.mark.parametrize("filenames,expected", [
    (["old.safetensors", ".cache/.gitignore", "CACHEDIR.TAG", ".cache/model.metadata",
      "nested/new.SAFETENSORS", "predictor.pt", "readme.md"],
     ["old.safetensors", "nested/new.SAFETENSORS"]),
    ([], ["放入 models/veda_scorers"]),
    ([".cache/.gitignore", "config.json"], ["放入 models/veda_scorers"]),
])
def test_bundle_menu_keeps_only_supported_weights_without_changing_category(monkeypatch, filenames, expected):
    categories = []
    def available(category):
        categories.append(category)
        return filenames
    monkeypatch.setattr(veda_nodes.folder_paths, "get_filename_list", available)
    node = veda_nodes.MiniMaxH3VedaBundleEXPT8
    info = node.define_schema().get_v1_info(node)
    assert info.input["required"]["bundle_name"][1]["options"] == expected
    assert categories == ["veda_scorers"]


@pytest.mark.parametrize(
    ("platform_name", "encoding", "supported"),
    [("nt", "cp936", False), ("nt", "UTF-8", True),
     ("nt", "utf8", True), ("posix", "cp936", True)],
)
def test_windows_inductor_template_requires_utf8_default_encoding(
    platform_name, encoding, supported,
):
    assert _windows_inductor_text_encoding_supported(platform_name, encoding) is supported


def test_flex_compile_is_dynamic_from_first_call_and_reused(monkeypatch):
    calls = []
    compiled = object()

    def compile_once(function, **options):
        calls.append((function, options))
        return compiled

    monkeypatch.setattr(veda_flex, "_COMPILED", None)
    monkeypatch.setattr(veda_flex.torch, "compile", compile_once)
    assert veda_flex.compiled_flex() is compiled
    assert veda_flex.compiled_flex() is compiled
    assert calls == [(veda_flex.flex_attention, {"fullgraph": True, "dynamic": True})]


def _fake_h3_model():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = SimpleNamespace(heads=56, head_dim=128)

    root = torch.nn.Module()
    root.diffusion_model = torch.nn.Module()
    root.diffusion_model.blocks = torch.nn.ModuleList(Block() for _ in range(50))
    return ModelPatcher(root, torch.device("cpu"), torch.device("cpu"))


def _fake_bundle(tmp_path, monkeypatch):
    file = tmp_path / "bundle.safetensors"
    file.write_bytes(b"test-bundle")
    reference = VedaBundleRef(file, veda_runtime._sha256(file), {})
    loaded = SimpleNamespace(plans=SimpleNamespace(plans={}), keep_ratio=0.1,
                             predictor=object())
    monkeypatch.setattr(veda_runtime.veda_bundle, "load", lambda path, device: loaded)
    return reference


def test_veda_nodes_are_append_only_tail():
    names = [node.__name__ for node in asyncio.run(
        package.comfy_entrypoint().get_node_list())]
    assert names[567:570] == [
        "MiniMaxH3VedaBundleEXPT8",
        "MiniMaxH3VedaApplyEXPT8",
        "MiniMaxH3VedaAuditEXPT8",
    ]
    assert len(names) == len(set(names))
    for node in package._veda_sparse_node_classes:
        schema = node.define_schema().get_v1_info(node)
        assert schema.name == node.__name__


def test_exact_plan_does_not_resize_or_choose_nearest():
    geo = Geometry("16:9", 1344, 768, 124, 37, 48, 84, 200)
    selected = plan.TilePlan.uniform(geo, tiling.TileShape(8, 8, 2), 50, 56)
    table = plan.PlanTable([selected])
    assert exact_plan(table, (37, 24, 42)) is selected
    with pytest.raises(ValueError, match="exact live video grid"):
        exact_plan(table, (7, 9, 16))


def test_t2va_only_native_layout_contract():
    layout = SimpleNamespace(
        seq_len=64 + 80 + 1008,
        signature=(64, 7, 18, 32, 40),
        segments=[(0, 64, "text"), (64, 144, "audio"),
                  (144, 1152, "video")],
    )
    assert _t2va_layout(layout, 1152) == (144, (7, 9, 16))
    with pytest.raises(RuntimeError, match="T2VA-only"):
        _t2va_layout(SimpleNamespace(
            **{**vars(layout), "segments": [(0, 64, "text"),
                                             (64, 80, "cond"),
                                             (80, 144, "audio"),
                                             (144, 1152, "video")]}), 1152)


def test_author_selection_keeps_global_and_diagonal():
    target = tiling.TiledSpan(16, (2, 8, 8), tiling.TileShape(2, 8, 8))
    tiles = tiling.build_tile_layout([target], 144, 144, torch.device("cpu"))
    scores = torch.zeros(1, 1, tiles.n_video_tiles)
    selection = mask.select_video_blocks(
        scores, tiles, mask.column_blocks(tiles, mask.Budget(ratio=0.1)))
    selected = mask.dense_block_mask(selection, tiles)
    assert selected.shape == (1, 2, 2)
    assert selected.all()


def test_flex_block_mask_respects_partial_key_tile():
    selected = torch.ones(1, 2, 2, dtype=torch.bool)
    counts = torch.tensor([128, 16], dtype=torch.int32)
    flex = flex_block_mask(selected, counts)
    assert flex.BLOCK_SIZE == (128, 128)
    assert bool(flex.to_dense().all())
    assert flex.full_kv_num_blocks[0, 0].tolist() == [1, 1]
    assert flex.kv_num_blocks[0, 0].tolist() == [1, 1]


def test_apply_report_mode_clones_only_selected_model(tmp_path, monkeypatch):
    model = _fake_h3_model()
    reference = _fake_bundle(tmp_path, monkeypatch)
    patched, runtime = veda_runtime.apply_veda(
        model, reference, mode="report_only", fused_tile_io=True)
    assert patched is not model
    assert "optimized_attention_override" not in model.model_options["transformer_options"]
    override = patched.model_options["transformer_options"]["optimized_attention_override"]
    assert callable(override)
    assert runtime.snapshot()["status"] == "not_executed"
    assert runtime.snapshot()["tile_io"] == "not_run_report_only"
    from comfy.ldm.modules import attention as core_attention

    torch.manual_seed(20260929)
    q = torch.randn(1, 56, 4, 128)
    k, v = torch.randn_like(q), torch.randn_like(q)
    kwargs = {"skip_reshape": True, "transformer_options": {}}
    actual = override(core_attention.optimized_attention, q, k, v, 56, **kwargs)
    expected = core_attention.optimized_attention(q, k, v, 56, **kwargs)
    assert torch.equal(actual, expected)
    assert runtime.calls == 0


def test_selected_masked_call_preserves_dense_attention_and_audits_no_sparse(
    tmp_path, monkeypatch,
):
    import comfy.cli_args
    from comfy.ldm.modules import attention as core_attention

    monkeypatch.setattr(comfy.cli_args, "enables_dynamic_vram", lambda: False)
    monkeypatch.setattr(veda_runtime, "require_flex_runtime", lambda device: None)
    monkeypatch.setattr(veda_runtime, "exact_plan", lambda plans, grid: SimpleNamespace(geometry="test"))
    source = _fake_h3_model()
    bundle = _fake_bundle(tmp_path, monkeypatch)
    patched, runtime = veda_runtime.apply_veda(source, bundle, mode="apply_exp")
    owner = patched.model_options["transformer_options"]["optimized_attention_override"]
    layout = SimpleNamespace(
        seq_len=8, signature=(3, 1, 2, 2, 4),
        segments=[(0, 3, "text"), (3, 7, "audio"), (7, 8, "video")],
    )
    q = torch.randn(1, 56, 8, 128)
    k, v = torch.randn_like(q), torch.randn_like(q)
    mask = torch.ones(8, 8, dtype=torch.bool)
    options = {"minimax_h3_layout": layout, "block_index": 0}
    actual = owner(core_attention.optimized_attention, q, k, v, 56,
                   mask=mask, skip_reshape=True, transformer_options=options)
    expected = core_attention.optimized_attention(
        q, k, v, 56, mask=mask, skip_reshape=True,
        transformer_options=options)
    assert torch.equal(actual, expected)
    report = runtime.snapshot()
    assert report["calls"] == report["delegated_calls"] == 1
    assert report["sparse_calls"] == 0
    assert report["delegate_reason"] == "mask_preserved"
    assert report["status"] == "observed_dense_delegate"
    assert report["kernel"] == "original_dense_delegate"
    assert report["tile_io"] == "not_run_dense_delegate"


def test_relay_style_partial_query_preserves_bias_and_reports_dense_delegate(
    tmp_path, monkeypatch,
):
    import comfy.cli_args
    from comfy.ldm.modules import attention as core_attention

    monkeypatch.setattr(comfy.cli_args, "enables_dynamic_vram", lambda: False)
    monkeypatch.setattr(veda_runtime, "require_flex_runtime", lambda device: None)
    patched, runtime = veda_runtime.apply_veda(
        _fake_h3_model(), _fake_bundle(tmp_path, monkeypatch), mode="apply_exp")
    owner = patched.model_options["transformer_options"]["optimized_attention_override"]
    layout = SimpleNamespace(seq_len=8)
    q = torch.randn(1, 56, 3, 128)
    k, v = torch.randn(1, 56, 8, 128), torch.randn(1, 56, 8, 128)
    bias = torch.randn(3, 8) * 0.1
    options = {"minimax_h3_layout": layout, "block_index": 0}
    kwargs = {"mask": bias, "skip_reshape": True, "transformer_options": options}
    actual = owner(core_attention.optimized_attention, q, k, v, 56, **kwargs)
    expected = core_attention.optimized_attention(q, k, v, 56, **kwargs)
    assert torch.equal(actual, expected)
    report = runtime.snapshot()
    assert report["status"] == "observed_dense_delegate"
    assert report["calls"] == report["delegated_calls"] == 1
    assert report["sparse_calls"] == 0
    assert report["delegate_reason"] == "partial_query_preserved"
    assert report["kernel"] == "original_dense_delegate"


def test_actual_relay_router_over_veda_keeps_temporal_bias_but_not_sparse(
    tmp_path, monkeypatch,
):
    import comfy.cli_args
    from comfy.ldm.modules import attention as core_attention

    from h3_audio_t8_pkg.prompt_relay_advanced import (
        PROMPT_RELAY_RUNTIME_KEY, make_prompt_relay_bias, route_prompt_relay_attention,
    )
    from h3_audio_t8_pkg.relay_sol_backend import capture_composed_backend

    monkeypatch.setattr(comfy.cli_args, "enables_dynamic_vram", lambda: False)
    monkeypatch.setattr(veda_runtime, "require_flex_runtime", lambda device: None)
    monkeypatch.setattr(
        core_attention, "optimized_attention",
        lambda q, k, v, heads, **kwargs: core_attention.attention_pytorch(
            q, k, v, heads, **{**kwargs, "_inside_attn_wrapper": True}),
    )
    patched, runtime = veda_runtime.apply_veda(
        _fake_h3_model(), _fake_bundle(tmp_path, monkeypatch), mode="apply_exp")
    owner = patched.model_options["transformer_options"]["optimized_attention_override"]
    relay_backend = capture_composed_backend(owner)
    event = {"text_key_start": 1, "text_key_end": 3,
             "midpoint": 0.5, "window": 0.25, "sigma": 0.75}
    route = {"seq_len": 8, "query_segments": (
        {"kind": "video", "start": 7, "end": 8,
         "query_times": torch.tensor([0.0])},
    ), "events": (event,)}
    options = {"minimax_h3_layout": SimpleNamespace(seq_len=8),
               "block_index": 0, PROMPT_RELAY_RUNTIME_KEY: route}
    q, k, v = (torch.randn(1, 56, 8, 128) for _ in range(3))
    actual = route_prompt_relay_attention(
        q, k, v, 56, skip_reshape=True, transformer_options=options,
        query_chunk_rows=2, relay_backend=relay_backend)
    full_bias = torch.zeros(8, 8)
    full_bias[7:] = make_prompt_relay_bias(
        route["query_segments"][0]["query_times"], 8, route["events"], dtype=q.dtype)
    expected = core_attention.attention_pytorch(
        q, k, v, 56, mask=full_bias, skip_reshape=True,
        _inside_attn_wrapper=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    report = runtime.snapshot()
    assert report["calls"] == report["delegated_calls"] == 2
    assert report["sparse_calls"] == 0
    assert report["delegate_reason"] == "partial_query_preserved"
    assert relay_backend.report()["completed_calls"]["delegate:completed"] == 2


def test_audit_exposes_report_to_canvas_history(tmp_path):
    bundle = VedaBundleRef(tmp_path / "bundle.safetensors", "test-sha", {})
    loaded = SimpleNamespace(keep_ratio=0.1)
    runtime = VedaRuntime(bundle, "report_only", 0.1, 64, loaded, None)
    latent = {"samples": torch.zeros(1, 1, 1, 1)}
    result = MiniMaxH3VedaAuditEXPT8.execute(latent, runtime)
    assert result[0] is latent
    assert result.ui["text"] == (result[1],)
    assert '"status": "not_executed"' in result[1]
    assert '"kernel": "original_dense_delegate"' in result[1]


def test_apply_refuses_verified_windows_dynamic_vram_crash(tmp_path, monkeypatch):
    import comfy.cli_args

    monkeypatch.setattr(comfy.cli_args, "enables_dynamic_vram", lambda: True)
    bundle = _fake_bundle(tmp_path, monkeypatch)
    model = _fake_h3_model()
    with pytest.raises(RuntimeError, match="--disable-dynamic-vram"):
        veda_runtime.apply_veda(model, bundle, mode="apply_exp")
    assert "optimized_attention_override" not in model.model_options["transformer_options"]


def test_unverified_attention_owner_is_preserved_and_audited(tmp_path, monkeypatch):
    import comfy.cli_args

    model = _fake_h3_model()
    bundle = _fake_bundle(tmp_path, monkeypatch)
    calls = []

    def previous(original, q, k, v, heads, **kwargs):
        calls.append((heads, kwargs.get("skip_reshape")))
        return original(q, k, v, heads, **kwargs)

    model.model_options["transformer_options"]["optimized_attention_override"] = previous
    monkeypatch.setattr(comfy.cli_args, "enables_dynamic_vram", lambda: True)
    monkeypatch.setattr(veda_runtime, "require_flex_runtime",
                        lambda device: pytest.fail("bypassed owner must not require Flex"))
    patched, runtime = veda_runtime.apply_veda(model, bundle, mode="apply_exp")
    assert patched is not model
    assert patched.model_options["transformer_options"]["optimized_attention_override"] is previous
    assert model.model_options["transformer_options"]["optimized_attention_override"] is previous
    report = runtime.snapshot()
    assert report["status"] == "bypassed_unverified_owner_preserved"
    assert report["sparse_calls"] == report["calls"] == 0
    assert report["bypass_reason"] == "unverified_attention_owner"
    assert report["tile_io"] == "not_run_owner_preserved"
    from comfy.ldm.modules import attention as core_attention

    q = torch.randn(1, 56, 4, 128)
    expected = core_attention.optimized_attention(q, q, q, 56, skip_reshape=True)
    actual = patched.model_options["transformer_options"]["optimized_attention_override"](
        core_attention.optimized_attention, q, q, q, 56, skip_reshape=True)
    assert torch.equal(actual, expected)
    assert calls == [(56, True)]


def test_existing_dit_patch_is_preserved_without_sparse_claim(tmp_path, monkeypatch):
    model = _fake_h3_model()
    bundle = _fake_bundle(tmp_path, monkeypatch)
    def hook(args, extra):
        return extra["original_block"](args)
    patches = {"dit": {("double_block", 0): hook}}
    model.model_options["transformer_options"]["patches_replace"] = patches
    patched, runtime = veda_runtime.apply_veda(model, bundle, mode="report_only")
    assert patched.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)] is hook
    assert patched.model_options["transformer_options"].get("optimized_attention_override") is None
    assert runtime.snapshot()["bypass_reason"] == "existing_DiT_patch_owner"
    assert runtime.snapshot()["sparse_calls"] == 0


def test_fused_tile_io_requires_boolean_before_model_change(tmp_path):
    bundle = VedaBundleRef(tmp_path / "not-read.safetensors", "test-sha", {})
    with pytest.raises(TypeError, match="must be a Boolean"):
        veda_runtime.apply_veda(object(), bundle, mode="report_only", fused_tile_io=1)
