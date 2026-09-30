"""Real Core pre-execution validation, not direct execute-only bypass tests."""
import asyncio
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg import nodes_semantic_bridge as bridge


@pytest.fixture
def validate(monkeypatch, tmp_path):
    import folder_paths
    core = Path(folder_paths.__file__).resolve().parent
    # conftest adds h3_t8 for legacy tests; "nodes" here must be Core's,
    # not the project's package-relative node registry of the same name.
    monkeypatch.syspath_prepend(str(core))
    import execution
    import nodes
    assert Path(nodes.__file__).resolve().parent == core
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "MiniMaxH3SemanticBridgeConfigT8",
                        bridge.MiniMaxH3SemanticBridgeConfigT8)
    monkeypatch.setattr(bridge, "model_paths", lambda: {"present.safetensors": str(tmp_path / "present")})
    monkeypatch.setattr(bridge, "file_sha", lambda *_: pytest.fail("validation hashed/loaded a model"))
    # This fixture tests Core range/enum and path-validation dispatch only.
    # Real metadata/fixed-contract validation has separate saved-model tests.
    monkeypatch.setattr(bridge, "inspect_bridge_profile", lambda *_: {"fixed_contract": None, "settings": None})

    def run(**overrides):
        inputs = dict(model_name="missing.safetensors", enabled=True, alpha=.1,
                      magnitude_match="per_token", token_scope="all_tokens",
                      device="auto", chunk_tokens=256)
        inputs.update(overrides)
        graph = {"1": {"class_type": "MiniMaxH3SemanticBridgeConfigT8", "inputs": inputs}}
        result = asyncio.run(execution.validate_inputs("bridge-validation", graph, "1", {}))
        assert not torch.cuda.is_initialized()
        return result
    return run


@pytest.mark.parametrize("overrides", [{"enabled": False}, {"alpha": 0.0},
                                     {"enabled": False, "alpha": 0.0}])
def test_missing_model_bypass_passes_actual_core(validate, overrides):
    assert validate(**overrides)[0] is True


def test_active_missing_rejected_and_listed_model_accepted(validate):
    assert validate()[0] is False
    assert validate(model_name="present.safetensors")[0] is True


@pytest.mark.parametrize("value", [-.1, 1.01, float("nan"), float("inf")])
@pytest.mark.parametrize("enabled", [False, True])
def test_range_and_finite_checks_not_lost_to_custom_validation(validate, value, enabled):
    assert validate(alpha=value, enabled=enabled)[0] is False


@pytest.mark.parametrize("overrides", [
    {"chunk_tokens": -1}, {"chunk_tokens": 65537}, {"magnitude_match": "typo"},
    {"token_scope": "unknown"}, {"device": "unknown"},
])
def test_other_native_validation_not_swallowed_on_bypass(validate, overrides):
    assert validate(enabled=False, **overrides)[0] is False


@pytest.mark.parametrize("overrides", [{"enabled": None}, {"alpha": None}, {"model_name": None}])
def test_unknown_linked_values_defer_until_execution(overrides):
    values = dict(model_name="missing.safetensors", enabled=True, alpha=.1)
    values.update(overrides)
    assert bridge.MiniMaxH3SemanticBridgeConfigT8.validate_inputs(**values) is True


def test_deferred_active_model_is_still_checked_at_execution(monkeypatch):
    monkeypatch.setattr(bridge, "model_paths", lambda: {})
    with pytest.raises(FileNotFoundError):
        bridge.MiniMaxH3SemanticBridgeConfigT8.execute("missing.safetensors")
