"""Native selector content, derived-cache integrity and read-only projection."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg import veda_identity as identity, veda_heuristic_runtime as heuristic
from h3_audio_t8_pkg import veda_runtime as trained
from h3_audio_t8_pkg.modular_sampling import results
from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from h3_audio_t8_pkg.veda_vendor.h3.geometry import Geometry
from h3_audio_t8_pkg.veda_vendor.veda import bundle, plan, predictor, tiling
from test_veda_sparse_exp import _fake_h3_model


def _heuristic(**kwargs):
    original = _fake_h3_model()
    patched, runtime = heuristic.apply_heuristic(original, **kwargs)
    return original, patched, runtime


def _tiny_bundle(tmp_path):
    # Exact native file/code owners but tiny predictor storage for identity
    # tests only: not an H3 inference, loaded model or quality qualification.
    network = predictor.TileScorePredictor(1, 1, 4)
    geo = Geometry("1:1", 32, 32, 22, 7, 2, 2, 40)
    table = plan.PlanTable([plan.TilePlan.uniform(geo, tiling.TileShape(2, 8, 8), 1, 1)])
    file = tmp_path / "identity-only.safetensors"
    bundle.save(str(file), network.state_dict(), table, num_layers=1, num_heads=1,
                head_dim=4, keep_ratio=.1, source="unit-test", source_weights="live", step=0)
    return trained.VedaBundleRef(file, trained._sha256(file), bundle.read_metadata(str(file)))


def test_heuristic_projection_is_read_only_and_telemetry_is_not_numerical():
    original, patched, runtime = _heuristic()
    selected = patched.model_options["transformer_options"]["optimized_attention_override"]
    view, before = identity.project(patched)
    assert view is not patched and "optimized_attention_override" not in view.model_options["transformer_options"]
    assert patched.model_options["transformer_options"]["optimized_attention_override"] is selected
    assert "optimized_attention_override" not in original.model_options["transformer_options"]
    runtime.calls, runtime.delegated_calls, runtime.grid = 50, 50, (7, 2, 2)
    runtime.attention_wall_seconds = 1.2
    assert identity.project(patched)[1] == before


def test_keep_and_custom_shapes_bind_operator_content():
    _, first, _ = _heuristic(keep_percent=5)
    _, second, _ = _heuristic(keep_percent=10)
    assert identity.project(first)[1] != identity.project(second)[1]
    _, custom, _ = _heuristic(head_tiling="custom", custom_tiling="4,4,4;2,4,8")
    assert identity.project(custom)[1]["configuration"]["shapes"]["items"] != \
        identity.project(first)[1]["configuration"]["shapes"]["items"]


def test_heuristic_derived_cache_verified_but_not_in_pre_post_identity():
    _, patched, runtime = _heuristic()
    before = identity.project(patched)[1]
    key = ((4, 4, 4), (1, 2, 2), 3, 7, torch.device("cpu"))
    runtime.tile_cache[key] = identity.veda_heuristic.build_layout((1, 2, 2), 3, 7, (4, 4, 4))
    assert identity.project(patched)[1] == before
    runtime.tile_cache[key].gather_index[0] = 0
    with pytest.raises(ValueError, match="derived tile cache changed"):
        identity.project(patched)


@pytest.mark.parametrize("field,value", [("mode", "apply_exp"), ("max_copy_mib", 32),
                                         ("sink_conditioning", "off")])
def test_changed_bound_settings_fail_without_unpatching(field, value):
    _, patched, runtime = _heuristic()
    selected = patched.model_options["transformer_options"]["optimized_attention_override"]
    setattr(runtime, field, value)
    with pytest.raises(ValueError, match="captured numerical settings"):
        identity.project(patched)
    assert patched.model_options["transformer_options"]["optimized_attention_override"] is selected


def test_unknown_owner_is_not_projected_or_granted_portability():
    original = _fake_h3_model()
    foreign = lambda *args, **kwargs: args[0]  # noqa: E731
    original.model_options["transformer_options"]["optimized_attention_override"] = foreign
    view, contract = identity.project(original)
    assert view is original and contract is None
    patched, runtime = heuristic.apply_heuristic(original)
    assert runtime.bypass_reason == "unverified_attention_owner"
    assert identity.project(patched) == (patched, None)


def test_unknown_runtime_hook_and_kernel_owner_are_not_authenticated(monkeypatch):
    _, patched, runtime = _heuristic()
    runtime.foreign_callback = lambda: None
    with pytest.raises(UnverifiedModelStack, match="runtime ownership"):
        identity.project(patched)
    del runtime.foreign_callback
    monkeypatch.setattr(heuristic, "attend", lambda *args, **kwargs: None)
    with pytest.raises(UnverifiedModelStack, match="another callable owner"):
        identity.project(patched)


def test_foreign_compiled_kernel_is_not_granted_portable_identity(monkeypatch):
    _, patched, _ = _heuristic()
    monkeypatch.setattr(identity.veda_flex, "_COMPILED", lambda *args, **kwargs: None)
    with pytest.raises(UnverifiedModelStack, match="compiled Flex kernel"):
        identity.project(patched)


def test_trained_bundle_actual_weights_and_plans_bind_without_live_mutation(tmp_path):
    reference = _tiny_bundle(tmp_path)
    model = _fake_h3_model()
    patched, runtime = trained.apply_veda(model, reference)
    before = identity.project(patched)[1]
    runtime.loaded.plans.plans[next(iter(runtime.loaded.plans.plans))].head_groups(0, "cpu")
    assert identity.project(patched)[1] == before
    with torch.no_grad():
        runtime.loaded.predictor.layers[0].proj_q[0, 0, 0] += 1
    changed = identity.project(patched)[1]
    assert changed != before
    assert "optimized_attention_override" in patched.model_options["transformer_options"]
    assert "optimized_attention_override" not in model.model_options["transformer_options"]


def test_trained_changed_file_metadata_plan_and_cached_heads_rejected(tmp_path):
    reference = _tiny_bundle(tmp_path)
    patched, runtime = trained.apply_veda(_fake_h3_model(), reference)
    selected = next(iter(runtime.loaded.plans.plans.values()))
    selected.head_groups(0, "cpu")[0].heads[0] = 9
    with pytest.raises(ValueError, match="head assignment"):
        identity.project(patched)
    selected._groups.clear()
    original = deepcopy(selected.head_shape)
    selected.head_shape[0][0] = 1
    with pytest.raises(ValueError, match="tile plans differ"):
        identity.project(patched)
    selected.head_shape = original
    runtime.bundle = replace(reference, metadata={**reference.metadata, "step": "99"})
    with pytest.raises(ValueError, match="bound metadata changed"):
        identity.project(patched)


def test_modular_projection_includes_operator_without_changing_legacy_empty(monkeypatch):
    original, patched, _runtime = _heuristic()
    monkeypatch.setattr(results, "_selected_base_identity", lambda model: {
        "portable_cache_reuse": True, "sha256": "base",
        "remaining_override": "optimized_attention_override" in model.model_options["transformer_options"],
    })
    # The fake MODEL is only an identity fixture, not a sampler completion.
    plain = results.selected_model_identity(original)
    selected = results.selected_model_identity(patched)
    assert plain["remaining_override"] is False
    assert "stage_effects" not in plain
    assert selected["stage_effects"]["veda_operator"]["algorithm"] == "model_free_tripool64_flex"
    assert selected["remaining_override"] is False


def test_non_native_loaded_predictor_stays_nonportable(tmp_path):
    reference = _tiny_bundle(tmp_path)
    patched, runtime = trained.apply_veda(_fake_h3_model(), reference)
    runtime.loaded.predictor = SimpleNamespace(layers=[])
    with pytest.raises(UnverifiedModelStack, match="predictor has another"):
        identity.project(patched)


@pytest.mark.parametrize("mutate", [False, True])
def test_actual_tiny_stage_uses_the_same_veda_effect_projection_before_and_after(monkeypatch, mutate):
    # Real CPU Core stage execution, but an inert operator descriptor seam.
    # Native Veda ownership itself is covered above; not a trained GPU test.
    from test_modular_native_explicit import inputs

    args, _runtime, _original = inputs(effects="eav")
    projections = []

    def project(model):
        projections.append(model)
        keep = .2 if mutate and len(projections) > 1 else .1
        return model, {"algorithm": "test-inert-operator", "keep": keep}

    monkeypatch.setattr(identity, "project", project)
    if mutate:
        with pytest.raises(ValueError, match="Stage effect identity changed"):
            results.sample_stage(*args)
    else:
        receipt = results.sample_stage(*args)[2].verify()
        assert receipt["portable_identity"] is receipt["verified_recipe_completion"] is True
        assert receipt["request"]["model"]["stage_effects"]["veda_operator"]["keep"] == .1
    assert len(projections) == 2
