from __future__ import annotations

import gc
import math
from pathlib import Path
import sys
from types import ModuleType
import weakref

import pytest
import torch

from h3_audio_t8_pkg.skin_finish_vretoucher_runtime import (
    MinimalConvModule,
    _forward_autocast,
    _pop_module_tree,
    _restore_module_tree,
    PureTorchFusedLeakyReLU,
    VRetoucherRuntimeSession,
    VRetouchRuntimeUnavailable,
    pure_torch_fused_leaky_relu,
    pure_torch_upfirdn2d,
    load_vretoucher_model,
    runtime_capability_report,
    run_vretoucher_context,
    unload_vretoucher_model,
    VRETOUCHER_PINNED_FILES,
    VRETOUCHER_STATE_STRUCTURE_SHA256,
    bundled_vretoucher_source_root,
    construct_vretoucher_model,
    verify_vretoucher_source,
)


class _ObservedContext:
    def __init__(self, entered: list[bool]):
        self.entered = entered

    def __enter__(self):
        self.entered.append(True)

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        return False


def test_temporary_generic_module_tree_is_restored_without_leaking_replacement():
    root_name = "h3_t8_runtime_tree_test"
    original_root = ModuleType(root_name)
    original_child = ModuleType(f"{root_name}.child")
    sys.modules[root_name] = original_root
    sys.modules[f"{root_name}.child"] = original_child
    previous = _pop_module_tree(root_name)
    assert previous == {root_name: original_root, f"{root_name}.child": original_child}
    replacement = ModuleType(root_name)
    sys.modules[root_name] = replacement
    try:
        _restore_module_tree(root_name, previous)
        assert sys.modules[root_name] is original_root
        assert sys.modules[f"{root_name}.child"] is original_child
    finally:
        sys.modules.pop(root_name, None)
        sys.modules.pop(f"{root_name}.child", None)


def test_pure_fused_leaky_relu_matches_explicit_formula_and_keys():
    value = torch.tensor([[[[-2.0, 1.5]], [[0.5, -0.25]]]])
    bias = torch.tensor([0.25, -0.5])
    expected = math.sqrt(2.0) * torch.nn.functional.leaky_relu(
        value + bias.view(1, 2, 1, 1), negative_slope=0.2
    )
    actual = pure_torch_fused_leaky_relu(value, bias)
    assert torch.allclose(actual, expected)
    module = PureTorchFusedLeakyReLU(2)
    assert list(module.state_dict()) == ["bias"]


def test_pure_upfirdn_is_deterministic_and_has_upstream_geometry():
    value = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    kernel = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    first = pure_torch_upfirdn2d(value, kernel, up=2, down=1, pad=(1, 0))
    second = pure_torch_upfirdn2d(value, kernel, up=2, down=1, pad=(1, 0))
    assert first.shape == (1, 1, 8, 8)
    assert torch.equal(first, second)
    upsampled = torch.zeros((8, 8), dtype=torch.float32)
    upsampled[::2, ::2] = value[0, 0]
    padded = torch.nn.functional.pad(upsampled, (1, 0, 1, 0))
    flipped = torch.flip(kernel, (0, 1))
    expected_2d = torch.empty((8, 8), dtype=torch.float32)
    for output_y in range(8):
        for output_x in range(8):
            expected_2d[output_y, output_x] = (
                padded[output_y : output_y + 2, output_x : output_x + 2] * flipped
            ).sum()
    expected = expected_2d.view(1, 1, 8, 8)
    assert torch.equal(first, expected)


def test_minimal_conv_module_preserves_mmcv_weight_key_contract():
    module = MinimalConvModule(
        in_channels=8,
        out_channels=32,
        kernel_size=7,
        stride=1,
        padding=3,
        norm_cfg=None,
        act_cfg={"type": "ReLU"},
    )
    assert list(module.state_dict()) == ["conv.weight", "conv.bias"]
    assert module(torch.zeros((1, 8, 9, 9))).shape == (1, 32, 9, 9)
    with pytest.raises(ValueError):
        MinimalConvModule(8, 32, 7, 1, 3, norm_cfg={"type": "BN"})


def test_missing_or_modified_source_fails_before_model_construction(tmp_path: Path):
    assert "op/fused_act.py" in VRETOUCHER_PINNED_FILES
    assert "op/upfirdn2d.py" in VRETOUCHER_PINNED_FILES
    with pytest.raises(VRetouchRuntimeUnavailable) as error:
        verify_vretoucher_source(tmp_path)
    assert error.value.status == "ABSTAIN_PINNED_SOURCE_MISSING"


def test_bundled_source_is_pinned_and_meta_constructs_exact_graph():
    root = bundled_vretoucher_source_root()
    source = verify_vretoucher_source()
    assert source["status"] == "PINNED_SOURCE_PASS"
    assert source["bundled_source"] is True
    assert source["root"] == str(root.resolve())
    assert source["source_hash_mode"] == "sha256_after_crlf_to_lf_normalization"
    model, report = construct_vretoucher_model()
    try:
        assert report["status"] == "PINNED_MODEL_STRUCTURE_PASS"
        assert report["source"]["bundled_source"] is True
        assert report["state_structure"]["tensor_count"] == 411
        assert report["state_structure"]["sha256"] == VRETOUCHER_STATE_STRUCTURE_SHA256
        assert next(model.parameters()).device.type == "meta"
    finally:
        del model


def test_runtime_weight_identity_gates_fail_before_deserialization(tmp_path: Path):
    missing = tmp_path / "missing.pth"
    with pytest.raises(VRetouchRuntimeUnavailable) as error:
        load_vretoucher_model(
            tmp_path,
            missing,
            expected_checkpoint_sha256="not-a-hash",
            device="cpu",
        )
    assert error.value.status == "ABSTAIN_TRUSTED_CHECKPOINT_SHA256_REQUIRED"
    with pytest.raises(VRetouchRuntimeUnavailable) as error:
        load_vretoucher_model(
            tmp_path,
            missing,
            expected_checkpoint_sha256="0" * 64,
            device="cpu",
        )
    assert error.value.status == "ABSTAIN_CHECKPOINT_MISSING"
    undersized = tmp_path / "undersized.pth"
    undersized.write_bytes(b"not a checkpoint")
    with pytest.raises(VRetouchRuntimeUnavailable) as error:
        load_vretoucher_model(
            tmp_path,
            undersized,
            expected_checkpoint_sha256="0" * 64,
            device="cpu",
        )
    assert error.value.status == "ABSTAIN_CHECKPOINT_SIZE_MISMATCH"


class _NoRunModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        raise AssertionError("invalid context must fail before model forward")


class _NonFiniteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        del inputs
        result = torch.full((1, 3, 512, 512), float("nan"))
        masks = [torch.zeros((1, 1, 256, 256)) for _ in range(6)]
        flows = torch.zeros((5, 1, 64, 64, 2))
        return result, masks, flows


def test_runtime_context_contract_rejects_wrong_geometry_before_forward():
    with pytest.raises(VRetouchRuntimeUnavailable) as error:
        run_vretoucher_context(_NoRunModel(), torch.zeros((5, 3, 512, 512)))
    assert error.value.status == "ABSTAIN_CONTEXT_SHAPE_MISMATCH"


def test_runtime_rejects_meta_structure_before_forward():
    model = _NoRunModel().to(device="meta")
    with pytest.raises(VRetouchRuntimeUnavailable) as error:
        run_vretoucher_context(model, torch.zeros((6, 3, 512, 512)))
    assert error.value.status == "ABSTAIN_MODEL_NOT_CHECKPOINT_BACKED"


def test_runtime_rejects_nonfinite_values_after_completed_forward():
    with pytest.raises(VRetouchRuntimeUnavailable) as error:
        run_vretoucher_context(
            _NonFiniteModel(),
            torch.zeros((6, 3, 512, 512), dtype=torch.float32),
        )
    assert error.value.status == "ABSTAIN_RUNTIME_OUTPUT_NONFINITE_AFTER_FORWARD"
    assert error.value.model_forward_completed is True
    assert "786432 non-finite" in str(error.value)


def test_cuda_half_forward_context_enables_matching_autocast(monkeypatch):
    calls: list[dict[str, object]] = []
    entered: list[bool] = []

    def fake_autocast(*, device_type, dtype, enabled):
        calls.append({"device_type": device_type, "dtype": dtype, "enabled": enabled})
        return _ObservedContext(entered)

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    with _forward_autocast("cuda", torch.float16):
        pass
    assert calls == [
        {"device_type": "cuda", "dtype": torch.float16, "enabled": True}
    ]
    assert entered == [True]
    calls.clear()
    with _forward_autocast("cpu", torch.float32):
        pass
    assert calls == []


def test_unload_is_selective_and_capability_report_makes_boundaries_explicit():
    model = _NoRunModel()
    report = unload_vretoucher_model(model)
    assert report["global_comfy_models_unloaded"] is False
    assert report["caller_reference_must_be_released"] is True
    assert next(model.parameters()).item() == 1.0
    capability = runtime_capability_report()
    assert capability["requires_mmcv"] is False
    assert capability["requires_turtle_or_tkinter"] is False
    assert capability["requires_external_spynet_checkpoint"] is False
    assert capability["owner_scoped_runtime_session"] is True
    assert capability["registered_node"] is False


def test_owner_scoped_session_releases_its_only_model_reference_and_is_idempotent():
    session = VRetoucherRuntimeSession(_NoRunModel(), {"status": "TEST_MODEL"})
    assert session.closed is False
    assert session.load_report == {"status": "TEST_MODEL"}
    with session:
        pass
    assert session.closed is True
    assert session.close_report["status"] == "VRETOUCHER_OWNER_CLEARED_OBJECT_RELEASED"
    replay = session.close()
    assert replay["status"] == "VRETOUCHER_OWNER_CLEARED_OBJECT_RELEASED"
    assert replay["replay_status"] == "VRETOUCHER_RUNTIME_SESSION_ALREADY_CLOSED"
    assert replay["idempotent_replay"] is True
    with pytest.raises(VRetouchRuntimeUnavailable) as error:
        session.run(None)
    assert error.value.status == "ABSTAIN_RUNTIME_SESSION_CLOSED"


def test_owner_scoped_session_reports_external_reference_without_global_unload():
    externally_held = _NoRunModel()
    session = VRetoucherRuntimeSession(externally_held, {})
    report = session.close()
    assert report["status"] == "VRETOUCHER_OWNER_CLEARED_OBJECT_STILL_REFERENCED"
    assert report["object_still_referenced_elsewhere"] is True
    assert report["global_comfy_models_unloaded"] is False
    assert next(externally_held.parameters()).item() == 1.0


def test_closed_session_rechecks_temporary_external_reference_release():
    externally_held = _NoRunModel()
    model_reference = weakref.ref(externally_held)
    session = VRetoucherRuntimeSession(externally_held, {})
    first = session.close()
    assert first["object_still_referenced_elsewhere"] is True
    del externally_held
    gc.collect()
    assert model_reference() is None
    replay = session.close()
    assert replay["status"] == "VRETOUCHER_OWNER_CLEARED_OBJECT_RELEASED"
    assert replay["object_still_referenced_elsewhere"] is False
    assert replay["idempotent_replay"] is True


def test_owner_scoped_session_closes_on_inference_path_exception():
    session = VRetoucherRuntimeSession(_NoRunModel(), {})
    with pytest.raises(RuntimeError, match="simulated inference failure"):
        with session:
            raise RuntimeError("simulated inference failure")
    assert session.closed is True
    assert session.close_report["status"] == "VRETOUCHER_OWNER_CLEARED_OBJECT_RELEASED"
