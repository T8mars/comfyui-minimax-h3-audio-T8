"""Saved synthetic model contracts and explicit stacks; not trained video quality."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from h3_audio_t8_pkg import semantic_bridge as sb
from h3_audio_t8_pkg import semantic_bridge_profiles as profiles
from h3_audio_t8_pkg import nodes_semantic_bridge as nodes
from h3_audio_t8_pkg.semantic_bridge_trans import TransSpec, TransBridge


@pytest.fixture
def saved_models(tmp_path, monkeypatch):
    generator = torch.Generator().manual_seed(1930)
    paths = {}
    for name in ("original", "bunny"):
        state = {key: torch.randn(shape, generator=generator) * .02 for key, shape in sb.SHAPES.items()}
        path = tmp_path / (name + ".safetensors")
        save_file(state, path)
        paths[name] = str(path)
    with torch.random.fork_rng(devices=[]):
        model = TransBridge(TransSpec(64, 1, 4, 64, True, 1., None)).eval()
    contract = {"schema": 1, "alpha_min": 1., "alpha_max": 1., "magnitude_match": "per_token",
                "token_span": "all", "tail_ratio": 1., "chunk_tokens": 0,
                "auto_alpha": False, "guard": False, "allow_dim_mismatch": False}
    state = {"net." + key: tensor for key, tensor in model.state_dict().items()}
    metadata = {"arch": "trans", "dim": "5120", "hidden": "64", "layers": "1", "heads": "4",
                "max_tokens": "64", "residual_skip": "True", "residual_scale": "1."}
    for name, extra in (("comic", {"application_contract": contract}), ("unidentified", {})):
        path = tmp_path / (name + ".safetensors")
        save_file(state, path, metadata={**metadata, "extra_json": json.dumps(extra)})
        paths[name] = str(path)
    monkeypatch.setattr(nodes, "model_paths", lambda: paths)
    return paths, contract, state, metadata


def cfg(path, **kwargs):
    return sb.BridgeConfig(path, sb.file_sha(path), **kwargs)


def native():
    return [[torch.randn(1, 5, 5120, generator=torch.Generator().manual_seed(7)),
             {"minimax_token_tags": torch.tensor([1, 0, 1, 0, 1]), "start_percent": .2}]]


def test_auto_reads_contract_not_name_and_never_changes_manual_defaults(saved_models):
    paths, *_ = saved_models
    paths["misleading-alpha-0.1-name"] = paths["comic"]
    result = nodes.MiniMaxH3SemanticBridgeAutoConfigT8.execute("misleading-alpha-0.1-name")
    config, report = result.result
    assert config.alpha == 1. and config.chunk_tokens == 0
    assert json.loads(report)["effective_parameters"]["alpha"] == 1.
    original = nodes.MiniMaxH3SemanticBridgeConfigT8.execute("original").result[0]
    assert original.alpha == .10 and original.chunk_tokens == 256


def test_wushu_preset_is_sha_pinned_not_all_transformers(saved_models, monkeypatch):
    paths, *_ = saved_models
    with pytest.raises(ValueError, match="No reliable"):
        nodes.MiniMaxH3SemanticBridgeAutoConfigT8.execute("unidentified")
    monkeypatch.setattr(profiles, "WUSHU_V1_SHA", sb.file_sha(paths["unidentified"]))
    config = nodes.MiniMaxH3SemanticBridgeAutoConfigT8.execute("unidentified").result[0]
    assert config.alpha == .12 and config.chunk_tokens == 0
    assert config.alpha != 1.
    manual, report = nodes.MiniMaxH3SemanticBridgeConfigT8.execute("unidentified", alpha=.2).result
    assert manual.alpha == .2 and json.loads(report)["warnings"]


@pytest.mark.parametrize("setting", [{"alpha": .1}, {"chunk_tokens": 256},
    {"magnitude_match": "none"}, {"token_scope": "text_only_preserve_reference"}])
def test_wrong_fixed_settings_fail_in_config_and_before_loop_sampling(saved_models, setting):
    paths, *_ = saved_models
    args = {"alpha": 1., "chunk_tokens": 0, **setting}
    with pytest.raises(ValueError, match="application contract mismatch"):
        nodes.MiniMaxH3SemanticBridgeConfigT8.execute("comic", **args)
    assert nodes.MiniMaxH3SemanticBridgeConfigT8.validate_inputs("comic", **args) is not True
    with pytest.raises(ValueError, match="application contract mismatch"):
        sb.preflight_bridge(cfg(paths["comic"], **args))


@pytest.mark.parametrize("fault", [{"alpha_min": True}, {"alpha_max": float("nan")},
    {"guard": 0}, {"schema": True}, {"chunk_tokens": "0"}, {"allow_dim_mismatch": True}])
def test_malformed_contract_never_becomes_an_auto_preset(saved_models, fault, tmp_path):
    _, contract, state, metadata = saved_models
    path = tmp_path / "bad-contract.safetensors"
    save_file(state, path, metadata={**metadata, "extra_json": json.dumps({"application_contract": {**contract, **fault}})})
    with pytest.raises(ValueError, match="contract"):
        profiles.inspect_bridge_profile(path)


def test_direct_legacy_zero_chunk_whole_item_is_supported(saved_models):
    paths, *_ = saved_models
    actual, _ = sb.apply_bridge(native(), cfg(paths["original"], chunk_tokens=0), cancel=lambda: None)
    expected, _ = sb.apply_bridge(native(), cfg(paths["original"], chunk_tokens=99), cancel=lambda: None)
    assert torch.equal(actual[0][0], expected[0][0])


def test_content_replacement_after_auto_config_is_rejected(saved_models):
    paths, *_ = saved_models
    config = nodes.MiniMaxH3SemanticBridgeAutoConfigT8.execute("original").result[0]
    Path(paths["original"]).write_bytes(Path(paths["bunny"]).read_bytes())
    with pytest.raises(ValueError, match="content changed"):
        sb.preflight_bridge(config)


def test_disabled_auto_does_not_resolve_or_hash(monkeypatch):
    monkeypatch.setattr(nodes, "model_paths", lambda: pytest.fail("bypass inspected model paths"))
    config = nodes.MiniMaxH3SemanticBridgeAutoConfigT8.execute("missing", enabled=False).result[0]
    assert not config.active and nodes.MiniMaxH3SemanticBridgeAutoConfigT8.validate_inputs("missing", False) is True


def test_explicit_stack_matches_two_ordered_independent_applications(saved_models):
    paths, *_ = saved_models
    first, second = cfg(paths["original"], alpha=.1), cfg(paths["bunny"], alpha=.2)
    original = native()
    intermediate, _ = sb.apply_bridge(original, first, cancel=lambda: None)
    fresh = [[intermediate[0][0], original[0][1]]]
    expected, _ = sb.apply_bridge(fresh, second, cancel=lambda: None)
    stack = sb.compose_bridges(first, second)
    actual, report = sb.apply_bridge(original, stack, cancel=lambda: None)
    assert torch.equal(actual[0][0], expected[0][0])
    assert sb.RECEIPT_KEY not in original[0][1]
    assert torch.equal(original[0][0], native()[0][0])
    assert actual[0][1]["minimax_token_tags"] is original[0][1]["minimax_token_tags"]
    receipt = report["items"][0]
    assert len(receipt["stages"]) == 2 and receipt["stages"][0]["output_sha256"] == receipt["stages"][1]["input_sha256"]
    assert receipt["receipt_sha256"] == sb.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    assert "path" not in sb.canonical(stack.identity())
    with pytest.raises(ValueError, match="already applied"):
        sb.apply_bridge(actual, stack, cancel=lambda: None)


def test_stack_order_and_individual_strengths_are_identity_bound(saved_models):
    paths, *_ = saved_models
    first, second = cfg(paths["original"]), cfg(paths["bunny"], alpha=.2)
    a, b = sb.compose_bridges(first, second), sb.compose_bridges(second, first)
    assert a.identity() != b.identity()
    assert a.identity() != sb.compose_bridges(replace(first, alpha=.3), second).identity()
    x, _ = sb.apply_bridge(native(), a, cancel=lambda: None)
    y, _ = sb.apply_bridge(native(), b, cancel=lambda: None)
    assert not torch.equal(x[0][0], y[0][0])
    disabled = sb.compose_bridges(first, second, enabled=False)
    incoming = native()
    result, report = sb.apply_bridge(incoming, disabled, cancel=lambda: pytest.fail("bypass cancelled"))
    assert result is incoming and not report["applied"]


def test_flattening_duplicate_limit_and_disabled_stack_never_reactivates(saved_models):
    paths, *_ = saved_models
    first, second = cfg(paths["original"]), cfg(paths["bunny"])
    with pytest.raises(ValueError, match="selected twice"):
        sb.compose_bridges(first, first)
    many = [replace(first, sha256=str(i)) for i in range(9)]
    with pytest.raises(ValueError, match="1 to 8"):
        sb.BridgeStack(tuple(many))
    merged = sb.compose_bridges(sb.compose_bridges(first, second, enabled=False), first)
    assert merged.bridges == (first,)


def test_stack_fixed_contract_preflight_all_models_before_first_application(saved_models, monkeypatch):
    paths, *_ = saved_models
    stack = sb.compose_bridges(cfg(paths["original"]), cfg(paths["comic"], alpha=.1, chunk_tokens=0))
    monkeypatch.setattr(sb, "_project", lambda *_: pytest.fail("first stage applied before second was preflighted"))
    with pytest.raises(ValueError, match="application contract mismatch"):
        sb.apply_bridge(native(), stack, cancel=lambda: None)


def test_stack_cancellation_after_first_stage_preserves_original(saved_models):
    paths, *_ = saved_models
    source = native()
    stack = sb.compose_bridges(cfg(paths["original"], chunk_tokens=0), cfg(paths["bunny"], chunk_tokens=0))
    calls = []
    def cancel():
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError("cancelled second stage")
    with pytest.raises(RuntimeError, match="second stage"):
        sb.apply_bridge(source, stack, cancel=cancel)
    assert torch.equal(source[0][0], native()[0][0]) and sb.RECEIPT_KEY not in source[0][1]


def test_stack_invalid_conditioning_rejected_before_model_reads(saved_models, monkeypatch):
    paths, *_ = saved_models
    stack = sb.compose_bridges(cfg(paths["original"]), cfg(paths["bunny"]))
    monkeypatch.setattr(sb, "_weights", lambda *_: pytest.fail("invalid stack input read model"))
    with pytest.raises(ValueError, match="finite raw H3"):
        sb.apply_bridge([[torch.ones(1, 4, 256), {}]], stack, cancel=lambda: None)


def test_stack_text_only_keeps_reference_rows_exactly(saved_models):
    paths, *_ = saved_models
    source = native()
    stack = sb.compose_bridges(cfg(paths["original"], token_scope="text_only_preserve_reference"),
                               cfg(paths["bunny"], token_scope="text_only_preserve_reference"))
    actual, _ = sb.apply_bridge(source, stack, cancel=lambda: None)
    assert torch.equal(actual[0][0][:, [1, 3]], source[0][0][:, [1, 3]])


def test_transformer_loading_and_application_do_not_consume_cpu_rng(saved_models):
    paths, *_ = saved_models
    initial = torch.random.get_rng_state().clone()
    sb.read_weights(paths["comic"])
    sb.apply_bridge(native(), cfg(paths["comic"], alpha=1., chunk_tokens=0), cancel=lambda: None)
    assert torch.equal(initial, torch.random.get_rng_state())


def test_stack_relay_binding_carries_one_composite_receipt(saved_models):
    from test_prompt_relay_advanced import NativeLikeFakeClip
    from h3_audio_t8_pkg.prompt_relay_advanced import build_prompt_relay_plan, build_prompt_relay_binding
    paths, *_ = saved_models
    clip = NativeLikeFakeClip()
    plan = build_prompt_relay_plan("room", "left\nright", 73, "auto_equal", "", "paper_v1", .1, False, False)[0]
    tokens = clip.tokenize(plan["compiled_prompt"])
    count = len(tokens["qwen3vl_32b"][0])
    source = [[torch.ones(1, count, 5120), {"minimax_token_tags": torch.ones(count, dtype=torch.long)}]]
    plain = build_prompt_relay_binding(clip, plan, plan["compiled_prompt"], source, tokens)
    bridged, report = sb.apply_bridge(source, sb.compose_bridges(cfg(paths["original"]), cfg(paths["bunny"])), cancel=lambda: None)
    bound = build_prompt_relay_binding(clip, plan, plan["compiled_prompt"], bridged, tokens)
    assert bound["events"] == plain["events"] and bound["binding_hash"] != plain["binding_hash"]
    assert bound["semantic_bridge_receipts"] == [report["items"][0]["receipt_sha256"]]


def test_real_core_rejects_wrong_contract_and_accepts_auto(saved_models, monkeypatch):
    import folder_paths
    core = Path(folder_paths.__file__).resolve().parent
    monkeypatch.syspath_prepend(str(core))
    import execution
    import nodes as core_nodes
    assert Path(core_nodes.__file__).resolve().parent == core
    for node in (nodes.MiniMaxH3SemanticBridgeConfigT8, nodes.MiniMaxH3SemanticBridgeAutoConfigT8):
        monkeypatch.setitem(core_nodes.NODE_CLASS_MAPPINGS, node.__name__, node)
    def validate(cls, inputs):
        graph = {"1": {"class_type": cls.__name__, "inputs": inputs}}
        return asyncio.run(execution.validate_inputs("bridge-contract", graph, "1", {}))[0]
    manual = {"model_name": "comic", "enabled": True, "alpha": .1, "magnitude_match": "per_token",
              "token_scope": "all_tokens", "chunk_tokens": 256, "device": "auto"}
    assert validate(nodes.MiniMaxH3SemanticBridgeConfigT8, manual) is False
    assert validate(nodes.MiniMaxH3SemanticBridgeConfigT8, {**manual, "alpha": 1., "chunk_tokens": 0}) is True
    assert validate(nodes.MiniMaxH3SemanticBridgeAutoConfigT8, {"model_name": "comic", "enabled": True, "device": "auto"}) is True
    assert not torch.cuda.is_initialized()


def test_new_nodes_append_after_all_current_old_ids():
    from h3_audio_t8_pkg import comfy_entrypoint
    registered = asyncio.run(comfy_entrypoint().get_node_list())
    ids = [node.define_schema().node_id for node in registered]
    # Preserve the released two nodes at their original577-prefix positions;
    # later append-only additions must not move them or force them to stay last.
    assert ids[575:577] == [node.__name__ for node in nodes.SEMANTIC_BRIDGE_EXTRA_NODE_CLASSES]
    assert len(ids) == len(set(ids))
    manual = nodes.MiniMaxH3SemanticBridgeConfigT8.define_schema()
    assert [item.id for item in manual.inputs] == ["model_name", "enabled", "alpha", "magnitude_match",
        "token_scope", "device", "chunk_tokens"]


def test_stack_native_conditioning_internal_external_parity_audio_unchanged(saved_models):
    from test_semantic_bridge import RawH3Clip
    from helpers import FakeVideoVAE, FakeAudioVAE
    from h3_audio_t8_pkg.conditioning import build_conditioning
    paths, *_ = saved_models
    stack = sb.compose_bridges(cfg(paths["original"]), cfg(paths["bunny"], alpha=.2))
    args = dict(clip=RawH3Clip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(), prompt="two cups",
                width=128, height=128, length=73, audio_mode="native")
    plain = build_conditioning(**args)
    internal = build_conditioning(**args, semantic_bridge=stack)
    external, _ = sb.apply_bridge(plain[0], stack, cancel=lambda: None)
    assert torch.equal(internal[0][0][0], external[0][0])
    assert internal[3:5] == plain[3:5]
    for before, after in zip(plain[1]["samples"].unbind(), internal[1]["samples"].unbind()):
        assert torch.equal(before, after)
