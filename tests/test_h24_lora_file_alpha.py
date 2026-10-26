"""Explicit file-alpha interpretation through real safetensors/Core forwards."""
import json

import pytest
import torch
import torch.nn.functional as functional
from safetensors.torch import save_file

from h3_audio_t8_pkg import h3_lora_compat_advanced as compat
from h3_audio_t8_pkg.nodes_h3_lora_compat_advanced import MiniMaxH3LoRACompatibilityLoaderT8Advanced as Loader
from test_h21_lora_consumption import FUSED, KEY, actual_projection, split_qkv
from test_progressive_sampling_runtime import tiny_model


POLICY = "metadata_alpha_per_rank"


def direct_pair(*, alpha=None, prefix=""):
    generator = torch.Generator().manual_seed(2410)
    a = torch.randn(16, 8, generator=generator) * .03
    b = torch.randn(48, 16, generator=generator) * .03
    module = prefix + FUSED
    state = {module + ".lora_A.weight": a, module + ".lora_B.weight": b}
    if alpha is not None:
        state[module + ".alpha"] = torch.tensor(alpha)
    return state, b @ a


def check_forward(patched, base, delta, strength=1.):
    layer = base.model.diffusion_model.blocks[0].attn.qkv_proj
    original = layer.weight.detach().clone()
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 29
    expected = functional.linear(x, original + strength * delta, layer.bias)
    torch.testing.assert_close(actual_projection(patched, x), expected, rtol=1e-6, atol=1e-7)
    assert torch.equal(layer.weight, original)
    assert not base.patches and not base.backup
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("prefix", ("", "diffusion_model."))
@pytest.mark.parametrize("alpha", (None, 8., 0., -8.))
def test_explicit_file_alpha_and_per_key_precedence(tmp_path, prefix, alpha):
    state, delta = direct_pair(alpha=alpha, prefix=prefix)
    path = tmp_path / "selected.safetensors"
    save_file(state, str(path), metadata={"ss_network_alpha": "8", "ss_network_dim": "16"})
    base = tiny_model()
    legacy, _ = compat.load_minimax_h3_lora_model(base, path, .7)
    check_forward(legacy, base, delta * (1. if alpha is None else alpha / 16), .7)
    patched, text = compat.load_minimax_h3_lora_model(base, path, .7, file_alpha_policy=POLICY)
    report = json.loads(text)
    check_forward(patched, base, delta * ((8 if alpha is None else alpha) / 16), .7)
    assert report["selected_strength"] == .7
    assert report["alpha_interpretation"]["requested_policy"] == POLICY
    assert report["alpha_interpretation"]["injected_module_count"] == (1 if alpha is None else 0)
    assert report["alpha_interpretation"]["training_scale_verified"] is False


@pytest.mark.parametrize("metadata", ({}, {"ss_network_dim": "16"}))
def test_missing_file_alpha_retains_legacy_even_when_explicit(tmp_path, metadata):
    state, delta = direct_pair()
    path = tmp_path / "legacy.safetensors"
    save_file(state, str(path), metadata=metadata)
    base = tiny_model()
    patched, text = compat.load_minimax_h3_lora_model(base, path, file_alpha_policy=POLICY)
    check_forward(patched, base, delta)
    assert json.loads(text)["alpha_interpretation"]["injected_module_count"] == 0


@pytest.mark.parametrize("file_alpha", ("0", "-8"))
def test_explicit_file_alpha_is_literal_not_a_producer_zero_guess(tmp_path, file_alpha):
    state, delta = direct_pair()
    path = tmp_path / "literal.safetensors"
    save_file(state, str(path), metadata={"ss_network_alpha": file_alpha, "ss_network_dim": "16"})
    base = tiny_model()
    patched, _ = compat.load_minimax_h3_lora_model(base, path, file_alpha_policy=POLICY)
    check_forward(patched, base, delta * (float(file_alpha) / 16))


@pytest.mark.parametrize("metadata", (
    {"ss_network_alpha": "nan"}, {"ss_network_alpha": "inf"},
    {"ss_network_alpha": "broken"},
    {"ss_network_alpha": "8", "ss_network_dim": "0"},
    {"ss_network_alpha": "8", "ss_network_dim": "3.5"},
    {"ss_network_alpha": "8", "ss_network_dim": "8"},
))
def test_unusable_selected_metadata_does_not_silently_change_scale(tmp_path, metadata):
    state, delta = direct_pair()
    path = tmp_path / "bad-metadata.safetensors"
    save_file(state, str(path), metadata=metadata)
    base = tiny_model()
    # Metadata is not a model-selection gate on the unchanged default path.
    legacy, _ = compat.load_minimax_h3_lora_model(base, path)
    check_forward(legacy, base, delta)
    with pytest.raises(ValueError, match="(?i)metadata.*alpha|metadata.*rank"):
        compat.load_minimax_h3_lora_model(base, path, file_alpha_policy=POLICY)


def test_unused_bad_metadata_cannot_override_explicit_tensor_alpha(tmp_path):
    state, delta = direct_pair(alpha=8)
    path = tmp_path / "per-key.safetensors"
    save_file(state, str(path), metadata={"ss_network_alpha": "nan", "ss_network_dim": "not-used"})
    base = tiny_model()
    patched, _ = compat.load_minimax_h3_lora_model(base, path, file_alpha_policy=POLICY)
    check_forward(patched, base, delta * .5)


def test_file_alpha_is_applied_before_unequal_rank_qkv_fusion(tmp_path):
    state, _ = split_qkv(alpha=False)
    # Explicit Q alpha wins; alpha-free K/V use 8 divided by their own rank.
    state["transformer_blocks.0.attn.to_q.alpha"] = torch.tensor(-1.)
    expected = torch.cat([
        state[f"transformer_blocks.0.attn.to_{name}.lora_B.weight"]
        @ state[f"transformer_blocks.0.attn.to_{name}.lora_A.weight"] * (alpha / rank)
        for name, rank, alpha in (("q", 1, -1.), ("k", 2, 8.), ("v", 3, 8.))
    ])
    path = tmp_path / "split.safetensors"
    # No uniform rank declaration: the actual per-projection ranks are used.
    save_file(state, str(path), metadata={"ss_network_alpha": "8"})
    base = tiny_model()
    patched, text = compat.load_minimax_h3_lora_model(base, path, file_alpha_policy=POLICY)
    check_forward(patched, base, expected)
    assert json.loads(text)["alpha_interpretation"]["injected_module_count"] == 2


def test_reloading_file_does_not_rewrite_or_double_scale_its_factors(tmp_path):
    state, delta = direct_pair()
    path = tmp_path / "repeat.safetensors"
    save_file(state, str(path), metadata={"ss_network_alpha": "8", "ss_network_dim": "16"})
    original_bytes = path.read_bytes()
    base = tiny_model()
    first, _ = compat.load_minimax_h3_lora_model(base, path, .7, file_alpha_policy=POLICY)
    second, _ = compat.load_minimax_h3_lora_model(first, path, -.2, file_alpha_policy=POLICY)
    assert [patch[0] for patch in second.patches[KEY]] == [.7, -.2]
    check_forward(second, base, delta * .5, .5)
    assert path.read_bytes() == original_bytes


def test_unknown_pairs_and_special_normalization_keep_original_metadata_semantics(tmp_path):
    state = {"future.proj.lora_A.weight": torch.ones(2, 8),
             "future.proj.lora_B.weight": torch.ones(48, 2)}
    path = tmp_path / "unknown.safetensors"
    save_file(state, str(path), metadata={"ss_network_alpha": "nan"})
    patched, text = compat.load_minimax_h3_lora_model(tiny_model(), path, file_alpha_policy=POLICY)
    report = json.loads(text)
    assert report["status"] == "no_compatible_patches" and not patched.patches
    assert report["alpha_interpretation"]["injected_module_count"] == 0
    assert set(report["mapping_diagnostics"]["converted_unmapped_keys"]) == set(state)


@pytest.mark.parametrize("extra", ("different_rank", "bad_shape", "incomplete_unused"))
def test_same_alias_multiple_formats_leave_native_priority_unchanged(tmp_path, extra):
    state, _ = direct_pair()
    # The regular up/down format takes priority over PEFT A/B in real Core.
    state[FUSED + ".lora_down.weight"] = torch.full((16, 8), .1)
    state[FUSED + ".lora_up.weight"] = torch.full((48, 16), .2)
    if extra == "different_rank":
        state[FUSED + ".lora_A.weight"] = torch.ones(8, 8)
        state[FUSED + ".lora_B.weight"] = torch.ones(48, 8)
    elif extra == "bad_shape":
        state[FUSED + ".lora_B.weight"] = torch.ones(7, 2)
    else:
        del state[FUSED + ".lora_B.weight"]
    path = tmp_path / "mixed-formats.safetensors"
    save_file(state, str(path), metadata={"ss_network_alpha": "nan", "ss_network_dim": "16"})
    base = tiny_model()
    legacy, _ = compat.load_minimax_h3_lora_model(base, path)
    selected, text = compat.load_minimax_h3_lora_model(base, path, file_alpha_policy=POLICY)
    x = torch.ones(3, 8)
    assert torch.equal(actual_projection(selected, x), actual_projection(legacy, x))
    assert json.loads(text)["alpha_interpretation"]["injected_module_count"] == 0


def test_unconverted_fastvideo_direct_pair_is_not_a_consumed_alpha_candidate(tmp_path):
    state = {"transformer_blocks.0.ff.net.0.proj.lora_A.weight": torch.ones(2, 8),
             "transformer_blocks.0.ff.net.0.proj.lora_B.weight": torch.ones(32, 2)}
    path = tmp_path / "ffn-only.safetensors"
    save_file(state, str(path), metadata={"ss_network_alpha": "nan", "ss_network_dim": "bad"})
    selected, text = compat.load_minimax_h3_lora_model(tiny_model(), path, file_alpha_policy=POLICY)
    report = json.loads(text)
    assert not selected.patches and report["status"] == "no_compatible_patches"
    assert report["alpha_interpretation"]["injected_module_count"] == 0


def test_ambiguous_target_aliases_do_not_interpret_shadowed_metadata(tmp_path):
    state, _ = direct_pair()
    other, _ = direct_pair(alpha=8., prefix="diffusion_model.")
    state.update(other)
    path = tmp_path / "two-aliases.safetensors"
    save_file(state, str(path), metadata={"ss_network_alpha": "nan", "ss_network_dim": "bad"})
    base = tiny_model()
    legacy, _ = compat.load_minimax_h3_lora_model(base, path)
    selected, text = compat.load_minimax_h3_lora_model(base, path, file_alpha_policy=POLICY)
    assert torch.equal(actual_projection(selected, torch.ones(3, 8)), actual_projection(legacy, torch.ones(3, 8)))
    report = json.loads(text)["alpha_interpretation"]
    assert report["injected_module_count"] == 0
    assert all(row["reason"] == "ambiguous_target_aliases_not_interpreted" for row in report["skipped_modules"])


@pytest.mark.parametrize(("down", "up"), (("_lora.down.weight", "_lora.up.weight"),
    (".lora_linear_layer.down.weight", ".lora_linear_layer.up.weight")))
def test_other_core_formats_are_included_in_alias_ambiguity(tmp_path, down, up):
    state, _ = direct_pair()
    alias = "diffusion_model." + FUSED
    state[alias + down] = torch.full((8, 8), .1)
    state[alias + up] = torch.full((48, 8), .2)
    state[alias + ".alpha"] = torch.tensor(8.)
    path = tmp_path / "other-format.safetensors"
    save_file(state, str(path), metadata={"ss_network_alpha": "nan", "ss_network_dim": "bad"})
    base = tiny_model()
    legacy, _ = compat.load_minimax_h3_lora_model(base, path)
    selected, text = compat.load_minimax_h3_lora_model(base, path, file_alpha_policy=POLICY)
    assert torch.equal(actual_projection(selected, torch.ones(3, 8)), actual_projection(legacy, torch.ones(3, 8)))
    assert json.loads(text)["alpha_interpretation"]["injected_module_count"] == 0


@pytest.mark.parametrize("suffix", (".diff", ".w_norm", ".set_weight"))
def test_native_dense_override_does_not_use_shadowed_file_alpha(tmp_path, suffix):
    state, _ = direct_pair()
    state[FUSED + suffix] = torch.full((48, 8), .2)
    path = tmp_path / "dense-override.safetensors"
    save_file(state, str(path), metadata={"ss_network_alpha": "nan", "ss_network_dim": "bad"})
    base = tiny_model()
    legacy, _ = compat.load_minimax_h3_lora_model(base, path)
    selected, text = compat.load_minimax_h3_lora_model(base, path, file_alpha_policy=POLICY)
    assert torch.equal(actual_projection(selected, torch.ones(3, 8)), actual_projection(legacy, torch.ones(3, 8)))
    assert json.loads(text)["alpha_interpretation"]["injected_module_count"] == 0


def test_disabled_loader_preserves_old_api_and_does_not_open_a_file(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Disabled must not access an adapter file")
    monkeypatch.setattr(compat.comfy.utils, "load_torch_file", forbidden)
    base = tiny_model()
    assert Loader.execute(base, "disabled", .4).result[0] is base
    assert Loader.execute(base, "disabled", .4, file_alpha_policy=POLICY).result[0] is base


def test_new_policy_is_optional_appended_and_defaults_to_unchanged_legacy():
    schema = Loader.define_schema()
    assert [item.id for item in schema.inputs][:3] == ["model", "lora_name", "strength_model"]
    option = next(item for item in schema.inputs if item.id == "file_alpha_policy")
    assert option.id == "file_alpha_policy" and option.optional is True
    assert option.default == "legacy_tensor_only"
    prefix = schema.inputs[-1]
    assert prefix.id == "prefix_policy" and prefix.optional is True and prefix.default == "legacy"


@pytest.mark.parametrize("file_policy", ("legacy_tensor_only", POLICY))
def test_explicit_double_prefix_alias_uses_whole_pair_and_keeps_default_unchanged(tmp_path, file_policy):
    state, delta = direct_pair(prefix="diffusion_model.diffusion_model.")
    path = tmp_path / "double-prefix.safetensors"
    save_file(state, str(path), metadata={"ss_network_alpha": "8", "ss_network_dim": "16"})
    original_bytes = path.read_bytes()
    base = tiny_model()
    old, old_report = compat.load_minimax_h3_lora_model(base, path)
    assert not old.patches and json.loads(old_report)["status"] == "no_compatible_patches"
    selected, text = compat.load_minimax_h3_lora_model(base, path,
        prefix_policy="unique_double_prefix", file_alpha_policy=file_policy)
    check_forward(selected, base, delta * (.5 if file_policy == POLICY else 1.))
    report = json.loads(text)["prefix_interpretation"]
    assert len(report["added_aliases"]) == 1 and not report["source_keys_rewritten"]
    assert path.read_bytes() == original_bytes


def test_double_prefix_collision_keeps_the_existing_source_not_last_wins(tmp_path):
    state, delta = direct_pair()
    doubled, _ = direct_pair(prefix="diffusion_model.diffusion_model.")
    doubled = {key: value * 9 for key, value in doubled.items()}
    state.update(doubled)
    path = tmp_path / "collision.safetensors"
    save_file(state, str(path))
    base = tiny_model()
    selected, text = compat.load_minimax_h3_lora_model(base, path, prefix_policy="unique_double_prefix")
    check_forward(selected, base, delta)
    report = json.loads(text)["prefix_interpretation"]
    assert not report["added_aliases"] and report["not_applied"]


@pytest.mark.parametrize("other_namespace", ("future.", "diffusion_model."))
def test_prefix_policy_does_not_join_incomplete_namespaces(tmp_path, other_namespace):
    state, _ = direct_pair(prefix="diffusion_model.diffusion_model.")
    b = state.pop("diffusion_model.diffusion_model." + FUSED + ".lora_B.weight")
    state[other_namespace + FUSED + ".lora_B.weight"] = b
    base = tiny_model()
    mapping, _ = compat.build_minimax_h3_lora_key_map(base.model)
    original = dict(mapping)
    if other_namespace == "diffusion_model.":
        # The original native parser reports the missing mapped A. The prefix
        # option must preserve that genuine error, not splice in the other A.
        with pytest.raises(KeyError, match="diffusion_model.*lora_A.weight"):
            compat.explicit_prefix_aliases(state, mapping, base.model, "unique_double_prefix")
    else:
        actual, report = compat.explicit_prefix_aliases(state, mapping, base.model, "unique_double_prefix")
        assert actual == mapping == original and not report["added_aliases"]


def test_bad_repeated_pair_shape_is_still_a_real_error(tmp_path):
    state, _ = direct_pair(prefix="diffusion_model.diffusion_model.")
    state["diffusion_model.diffusion_model." + FUSED + ".lora_B.weight"] = torch.ones(47, 16)
    path = tmp_path / "wrong-shape.safetensors"
    save_file(state, str(path))
    with pytest.raises(ValueError, match="does not match actual target"):
        compat.load_minimax_h3_lora_model(tiny_model(), path, prefix_policy="unique_double_prefix")


@pytest.mark.parametrize("suffix", (".diff", ".w_norm", ".set_weight"))
def test_double_prefix_does_not_overwrite_an_existing_dense_consumer(tmp_path, suffix):
    state, _ = direct_pair(prefix="diffusion_model.diffusion_model.")
    state[FUSED + suffix] = torch.full((48, 8), .2)
    path = tmp_path / "dense-prefix-collision.safetensors"
    save_file(state, str(path))
    base = tiny_model()
    legacy, _ = compat.load_minimax_h3_lora_model(base, path)
    selected, text = compat.load_minimax_h3_lora_model(base, path, prefix_policy="unique_double_prefix")
    assert torch.equal(actual_projection(selected, torch.ones(3, 8)), actual_projection(legacy, torch.ones(3, 8)))
    report = json.loads(text)["prefix_interpretation"]
    assert not report["added_aliases"] and report["not_applied"]


def test_double_prefix_does_not_overwrite_a_converted_split_qkv_consumer(tmp_path):
    state, expected = split_qkv(alpha=False)
    doubled, _ = direct_pair(prefix="diffusion_model.diffusion_model.")
    state.update(doubled)
    path = tmp_path / "split-prefix-collision.safetensors"
    save_file(state, str(path))
    base = tiny_model()
    selected, text = compat.load_minimax_h3_lora_model(base, path, prefix_policy="unique_double_prefix")
    check_forward(selected, base, expected)
    report = json.loads(text)["prefix_interpretation"]
    assert not report["added_aliases"] and report["not_applied"]
