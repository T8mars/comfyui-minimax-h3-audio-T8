"""Per-MODEL tile-64 heuristic owner, native geometry and honest audits."""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from comfy.model_patcher import ModelPatcher

import h3_audio_t8_pkg as package
from h3_audio_t8_pkg import veda_heuristic_runtime as runtime_module
from h3_audio_t8_pkg.nodes_veda_heuristic_exp import MiniMaxH3VedaHeuristicAuditEXPT8


def _model():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = SimpleNamespace(heads=56, head_dim=128)

    root = torch.nn.Module()
    root.diffusion_model = torch.nn.Module()
    root.diffusion_model.blocks = torch.nn.ModuleList(Block() for _ in range(50))
    return ModelPatcher(root, torch.device("cpu"), torch.device("cpu"))


def _layout():
    return SimpleNamespace(
        seq_len=7, signature=(2, 1, 2, 2, 1),
        segments=[(0, 2, "text"), (2, 4, "ref_img"),
                  (4, 6, "audio"), (6, 7, "video")],
    )


def test_nodes_append_after_trained_veda_and_schema():
    names = [item.__name__ for item in asyncio.run(package.comfy_entrypoint().get_node_list())]
    assert names[567:572] == [
        "MiniMaxH3VedaBundleEXPT8", "MiniMaxH3VedaApplyEXPT8",
        "MiniMaxH3VedaAuditEXPT8", "MiniMaxH3VedaHeuristicApplyEXPT8",
        "MiniMaxH3VedaHeuristicAuditEXPT8",
    ]
    assert len(names) == len(set(names))
    for node in package._veda_heuristic_node_classes:
        assert node.define_schema().get_v1_info(node).name == node.__name__
    schema = package._veda_heuristic_node_classes[0].define_schema().get_v1_info(
        package._veda_heuristic_node_classes[0])
    assert {"start_percent", "end_percent", "sink_conditioning"} <= set(
        schema.input["optional"])
    assert all(name not in schema.input["required"] for name in (
        "start_percent", "end_percent", "sink_conditioning"))


def test_old_seven_widget_apply_call_keeps_new_optional_defaults():
    apply_node = package._veda_heuristic_node_classes[0]
    result = apply_node.execute(
        _model(), "report_only", 5.0, "dual_fast", "4,4,4;8,4,2",
        "triplet", 4096, 16)
    runtime = result[1]
    assert runtime.start_percent == 0.0 and runtime.end_percent == 1.0
    assert runtime.sigma_start is runtime.sigma_end is None
    assert runtime.sink_conditioning == "exact_kv_and_rows"


def test_native_layout_with_reference_rows_keeps_exact_target():
    assert runtime_module.target_grid(_layout(), 7) == (6, (1, 1, 1))
    with pytest.raises(RuntimeError, match="native final target"):
        runtime_module.target_grid(SimpleNamespace(
            seq_len=7, signature=(2, 1, 2, 2, 1),
            segments=[(0, 2, "text"), (2, 7, "audio")]), 7)


def test_report_only_delegates_without_touching_original_model():
    model = _model()
    patched, runtime = runtime_module.apply_heuristic(model, mode="report_only")
    assert patched is not model
    assert "optimized_attention_override" not in model.model_options["transformer_options"]
    override = patched.model_options["transformer_options"]["optimized_attention_override"]
    from comfy.ldm.modules import attention as core_attention

    q = torch.randn(1, 56, 7, 128)
    options = {"minimax_h3_layout": _layout(), "block_index": 0}
    actual = override(core_attention.optimized_attention, q, q, q, 56,
                      skip_reshape=True, transformer_options=options)
    expected = core_attention.optimized_attention(
        q, q, q, 56, skip_reshape=True, transformer_options=options)
    assert torch.equal(actual, expected)
    assert runtime.snapshot()["calls"] == 1
    assert runtime.snapshot()["delegated_calls"] == 1
    assert runtime.snapshot()["sparse_calls"] == 0
    assert runtime.snapshot()["trained_predictor_used"] is False


def test_partial_query_preserves_relay_bias_and_does_not_claim_flex(monkeypatch):
    import comfy.cli_args
    from comfy.ldm.modules import attention as core_attention

    monkeypatch.setattr(comfy.cli_args, "enables_dynamic_vram", lambda: False)
    monkeypatch.setattr(runtime_module, "require_flex_runtime", lambda device: None)
    patched, runtime = runtime_module.apply_heuristic(
        _model(), mode="apply_exp", min_tokens=0)
    owner = patched.model_options["transformer_options"]["optimized_attention_override"]
    q = torch.randn(1, 56, 3, 128)
    k, v = torch.randn(1, 56, 7, 128), torch.randn(1, 56, 7, 128)
    bias = torch.randn(3, 7) * 0.1
    kwargs = {"mask": bias, "skip_reshape": True,
              "transformer_options": {"minimax_h3_layout": _layout(), "block_index": 0}}
    actual = owner(core_attention.optimized_attention, q, k, v, 56, **kwargs)
    expected = core_attention.optimized_attention(q, k, v, 56, **kwargs)
    assert torch.equal(actual, expected)
    report = runtime.snapshot()
    assert report["status"] == "observed_dense_delegate"
    assert report["kernel"] == "original_dense_delegate"
    assert report["calls"] == report["delegated_calls"] == 1
    assert report["sparse_calls"] == 0
    assert report["delegate_reason"] == "partial_query_preserved"


def test_two_independent_model_branches_do_not_share_runtime_or_owner():
    source = _model()
    low, low_runtime = runtime_module.apply_heuristic(
        source, mode="report_only", keep_percent=5)
    high, high_runtime = runtime_module.apply_heuristic(
        source, mode="report_only", keep_percent=10)
    assert low is not high and low is not source and high is not source
    assert low_runtime is not high_runtime
    assert low_runtime.keep_ratio == pytest.approx(0.05)
    assert high_runtime.keep_ratio == pytest.approx(0.1)
    original = source.model_options["transformer_options"]
    assert "optimized_attention_override" not in original
    low_owner = low.model_options["transformer_options"]["optimized_attention_override"]
    high_owner = high.model_options["transformer_options"]["optimized_attention_override"]
    assert low_owner is not high_owner
    from comfy.ldm.modules import attention as core_attention

    q = torch.randn(1, 56, 7, 128)
    low_owner(core_attention.optimized_attention, q, q, q, 56,
              skip_reshape=True,
              transformer_options={"minimax_h3_layout": _layout(), "block_index": 0})
    assert low_runtime.snapshot()["calls"] == 1
    assert high_runtime.snapshot()["calls"] == 0


def test_existing_eav_delegate_capture_preserves_prior_heuristic_owner():
    from comfy.ldm.modules import attention as core_attention
    from h3_audio_t8_pkg.relay_sol_backend import capture_composed_backend

    patched, runtime = runtime_module.apply_heuristic(_model(), mode="report_only")
    owner = patched.model_options["transformer_options"]["optimized_attention_override"]
    backend = capture_composed_backend(owner)
    assert backend.report()["kind"] == "user_selected_unverified_delegate"
    q = torch.randn(1, 56, 7, 128)
    options = {"minimax_h3_layout": _layout(), "block_index": 0}
    actual = backend.attention(q, q, q, 56, skip_reshape=True,
                               transformer_options=options)
    expected = core_attention.optimized_attention(
        q, q, q, 56, skip_reshape=True, transformer_options=options)
    assert torch.equal(actual, expected)
    assert runtime.snapshot()["calls"] == 1
    assert runtime.snapshot()["sparse_calls"] == 0


def test_unknown_owner_keeps_callable_and_skips_flex_and_dynamic_guard(monkeypatch):
    import comfy.cli_args

    model = _model()
    calls = []

    def owner(original, q, k, v, heads, **kwargs):
        calls.append(heads)
        return original(q, k, v, heads, **kwargs)

    model.model_options["transformer_options"]["optimized_attention_override"] = owner
    monkeypatch.setattr(comfy.cli_args, "enables_dynamic_vram", lambda: True)
    monkeypatch.setattr(runtime_module, "require_flex_runtime",
                        lambda device: pytest.fail("foreign owner should bypass Flex"))
    patched, runtime = runtime_module.apply_heuristic(model, mode="apply_exp")
    assert patched.model_options["transformer_options"]["optimized_attention_override"] is owner
    assert runtime.snapshot()["status"] == "bypassed_unverified_owner_preserved"
    assert runtime.snapshot()["sparse_calls"] == 0
    from comfy.ldm.modules import attention as core_attention

    q = torch.randn(1, 56, 3, 128)
    patched.model_options["transformer_options"]["optimized_attention_override"](
        core_attention.optimized_attention, q, q, q, 56, skip_reshape=True)
    assert calls == [56]


def test_audit_is_history_visible_and_never_claims_trained_predictor():
    runtime = runtime_module.HeuristicRuntime("report_only", 0.05, "dual_fast",
                                              "triplet", 4096, 16, None)
    latent = {"samples": torch.zeros(1, 1, 1, 1)}
    result = MiniMaxH3VedaHeuristicAuditEXPT8.execute(latent, runtime)
    assert result[0] is latent
    assert result.ui["text"] == (result[1],)
    assert '"trained_predictor_used": false' in result[1]


def test_invalid_resource_settings_fail_before_model_mutation():
    model = _model()
    with pytest.raises(ValueError, match="resource or keep"):
        runtime_module.apply_heuristic(model, keep_percent=0.5)
    assert "optimized_attention_override" not in model.model_options["transformer_options"]


def test_sigma_window_delegates_outside_and_applies_inside(monkeypatch):
    import comfy.cli_args
    from comfy.ldm.modules import attention as core_attention

    model = _model()
    model.model.model_sampling = SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent)
    monkeypatch.setattr(comfy.cli_args, "enables_dynamic_vram", lambda: False)
    monkeypatch.setattr(runtime_module, "require_flex_runtime", lambda device: None)
    seen = []

    def fake_attend(q, k, v, **kwargs):
        seen.append(kwargs["sink_conditioning"])
        return torch.zeros_like(q)

    monkeypatch.setattr(runtime_module, "attend", fake_attend)
    patched, runtime = runtime_module.apply_heuristic(
        model, mode="apply_exp", min_tokens=0, start_percent=0.2,
        end_percent=0.8, sink_conditioning="off")
    owner = patched.model_options["transformer_options"]["optimized_attention_override"]
    q = torch.randn(1, 56, 7, 128)
    options = {"minimax_h3_layout": _layout(), "block_index": 0,
               "sigmas": torch.tensor([0.9])}
    actual = owner(core_attention.optimized_attention, q, q, q, 56,
                   skip_reshape=True, transformer_options=options)
    expected = core_attention.optimized_attention(
        q, q, q, 56, skip_reshape=True, transformer_options=options)
    assert torch.equal(actual, expected)
    assert runtime.snapshot()["window_delegated_calls"] == 1
    assert runtime.snapshot()["delegate_reason"] == "outside_sigma_window"
    assert not seen
    options["sigmas"] = torch.tensor([0.5])
    owner(core_attention.optimized_attention, q, q, q, 56,
          skip_reshape=True, transformer_options=options)
    assert seen == ["off"]
    assert runtime.calls == 2 and runtime.sparse_calls == 1
    assert runtime.delegated_calls == 1
    assert runtime.snapshot()["sigma_start"] == pytest.approx(0.8)
    assert runtime.snapshot()["sigma_end"] == pytest.approx(0.2)
    assert "optimized_attention_override" not in model.model_options["transformer_options"]


def test_sigma_window_rejects_missing_live_sigma_and_invalid_bounds(monkeypatch):
    import comfy.cli_args
    from comfy.ldm.modules import attention as core_attention

    model = _model()
    model.model.model_sampling = SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent)
    with pytest.raises(ValueError, match="sigma window"):
        runtime_module.apply_heuristic(model, start_percent=0.8, end_percent=0.2)
    with pytest.raises(ValueError, match="conditioning sink"):
        runtime_module.apply_heuristic(model, sink_conditioning="unknown")
    monkeypatch.setattr(comfy.cli_args, "enables_dynamic_vram", lambda: False)
    monkeypatch.setattr(runtime_module, "require_flex_runtime", lambda device: None)
    patched, runtime = runtime_module.apply_heuristic(
        model, mode="apply_exp", min_tokens=0, start_percent=0.2, end_percent=0.8)
    owner = patched.model_options["transformer_options"]["optimized_attention_override"]
    q = torch.randn(1, 56, 7, 128)
    with pytest.raises(RuntimeError, match="live sampler sigmas"):
        owner(core_attention.optimized_attention, q, q, q, 56,
              skip_reshape=True,
              transformer_options={"minimax_h3_layout": _layout(), "block_index": 0})
    assert runtime.failure is not None
    assert "optimized_attention_override" not in model.model_options["transformer_options"]
