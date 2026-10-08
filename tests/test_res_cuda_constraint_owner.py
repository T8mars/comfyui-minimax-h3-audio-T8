"""Actual native CUDA constraint declarations, without registration or launch."""
from dataclasses import replace
import importlib

import pytest
import torch

from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from h3_audio_t8_pkg.res_kitchen_identity import _SourceAudit, _constraint
from h3_audio_t8_pkg.res_rope_identity import _PINS


def test_actual_CUDA_RoPE_constraint_function_uses_its_source_owner_without_registration():
    cuda = importlib.import_module("comfy_kitchen.backends.cuda")
    constraints = importlib.import_module("comfy_kitchen.constraints")
    registry = importlib.import_module("comfy_kitchen.registry").registry
    before = (dict(registry._backends), dict(registry._constraints), set(registry._disabled))
    # This installed builder only creates literal constraint objects. Do not
    # register a fake available CUDA backend or execute the validation rule.
    rules = cuda._build_constraints()
    actual = rules["rms_rope_split_half_"]
    assert actual is rules["rms_rope_split_half"]
    assert actual.call_rules == (cuda._validate_cuda_rms_rope,)
    assert cuda._validate_cuda_rms_rope.__globals__ is vars(cuda)
    audit = _SourceAudit(_PINS)
    observed = _constraint(audit, actual, constraints)
    assert observed["state"]["call_rules"] == [audit.functions[cuda.__name__ + ":_validate_cuda_rms_rope"]]
    assert cuda.__name__ + ":_native_rms_rope_layout" in audit.functions
    assert "comfy_kitchen._rope_utils:detect_rms_rope_bnhd" in audit.functions
    assert audit.sources[cuda.__name__] == audit.source_pins[cuda.__name__]
    assert before == (dict(registry._backends), dict(registry._constraints), set(registry._disabled))
    assert not torch.cuda.is_initialized()


def test_unknown_CUDA_constraint_rule_or_helper_remains_unverified_unexecuted(monkeypatch):
    cuda = importlib.import_module("comfy_kitchen.backends.cuda")
    constraints = importlib.import_module("comfy_kitchen.constraints")
    actual = cuda._build_constraints()["rms_rope_split_half_"]
    calls = []

    def foreign(*args, **kwargs):
        calls.append(True)
        raise AssertionError("inspection must not execute a foreign rule")

    with pytest.raises(UnverifiedModelStack, match="constraint call rule"):
        _constraint(_SourceAudit(_PINS), replace(actual, call_rules=(foreign,)), constraints)
    with monkeypatch.context() as changed:
        changed.setattr(cuda, "_native_rms_rope_layout", foreign)
        with pytest.raises(UnverifiedModelStack, match="outside the audited package"):
            _constraint(_SourceAudit(_PINS), actual, constraints)
        assert cuda._native_rms_rope_layout is foreign
    assert not calls and not torch.cuda.is_initialized()
