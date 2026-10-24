"""Independent actual Core PTD lifecycle/foreign-owner review, CPU only."""

import comfy.model_management as mm
import comfy.ops as ops
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
import pytest
import torch
import torch.nn.functional as F

from h3_audio_t8_pkg import ptd_advanced as ptd
from h3_audio_t8_pkg import ptd_checkpoint as files
from test_fast_h3_v2_core_sampler import model
from test_h21_projection_model import sample as plain
from test_h21_ptd_advanced import experts, sample


@pytest.fixture(autouse=True)
def actual_CPU(monkeypatch):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    with torch.inference_mode():
        yield
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("damage", ["finite", "nan"])
def test_last_real_forward_cannot_publish_completion_after_selected_expert_mutation(damage):
    base = model()
    expected_plain = plain(base)
    selected, _ = ptd.configure_ptd(base, experts(base), files.metadata_for("h3_4step"))
    controller = ptd.get_controller(selected)
    factor = controller.bank["blocks.0.attn.out_proj"][0][0].a
    original = factor.clone()
    calls = []

    def mutate_last(_module, _args, _output):
        if controller.current()[0] == 3:
            calls.append(3)
            if damage == "nan":
                factor.view(-1)[0] = float("nan")
            else:
                factor.add_(.25)

    handle = base.model.diffusion_model.final_layer.register_forward_hook(mutate_last)
    try:
        with pytest.raises(ValueError):
            sample(selected, controller)
    finally:
        handle.remove()
        factor.copy_(original)
    assert calls == [3]
    assert controller.report()["status"] == "failed"
    assert controller.selection.get() is None and controller.request.get() is None
    actual_plain = plain(base)
    for actual, expected in zip(actual_plain["samples"].unbind(), expected_plain["samples"].unbind()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    sample(selected, controller)
    assert controller.report()["last_request"]["hops"] == [0, 1, 2, 3]


def test_actual_unknown_modulation_helper_retains_original_execution(monkeypatch):
    base = model()
    original, calls = native._mod_gate, []

    def foreign(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(native, "_mod_gate", foreign)
    selected, _ = ptd.configure_ptd(base, experts(base), files.metadata_for("h3_4step"))
    controller = ptd.get_controller(selected)
    sample(selected, controller)
    assert calls == [1] * 8  # Two real residual applications per complete hop.
    assert controller.report()["partial"]


def test_actual_unknown_fused_module_delegate_retains_original_execution(monkeypatch):
    base = model()
    original, calls = ops.linear_input_act, []

    def foreign(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ops, "linear_input_act", foreign)
    selected, _ = ptd.configure_ptd(base, experts(base), files.metadata_for("h3_4step"))
    controller = ptd.get_controller(selected)
    sample(selected, controller)
    assert calls == [1] * 4
    assert controller.report()["partial"]


def test_global_actual_adaln_pre_hook_input_is_not_erased_by_precision_hook():
    base = model()
    root = base.model.diffusion_model
    controller = ptd.PTDController(root, experts(base), files.metadata_for("h3_4step"))
    layer = root.blocks[0].adaln_proj
    calls = []

    def user_pre_hook(module, args):
        if module is layer.linear:
            calls.append(1)
            return (args[0] + .5, *args[1:])

    handle = torch.nn.modules.module.register_module_forward_pre_hook(user_pre_hook)
    try:
        labels = torch.tensor([1 - .999], dtype=torch.float32)
        embedded = root.time_embedder(labels)
        expected = layer(embedded)
        with controller.activate(0, .999), controller.forward_scope():
            actual = layer(root.time_embedder(labels))
        assert calls == [1, 1]
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
    finally:
        handle.remove()
    assert not layer.linear._forward_pre_hooks


def test_actual_Core_interrupt_restores_ordinary_next_request_and_selected_retry(monkeypatch):
    base = model()
    expected = plain(base)
    selected, _ = ptd.configure_ptd(base, experts(base), files.metadata_for("h3_4step"))
    controller = ptd.get_controller(selected)
    original, cancellations = mm.throw_exception_if_processing_interrupted, []

    def cancel_once():
        if ptd._FRAME.get() is not None and not cancellations:
            cancellations.append(1)
            raise mm.InterruptProcessingException()
        return original()

    monkeypatch.setattr(mm, "throw_exception_if_processing_interrupted", cancel_once)
    with pytest.raises(mm.InterruptProcessingException):
        sample(selected, controller)
    assert cancellations == [1]
    assert controller.report()["status"] == "failed"
    assert controller.selection.get() is None and controller.request.get() is None
    assert all(not layer._forward_hooks and not layer._forward_pre_hooks
               for layer in base.model.diffusion_model.modules())
    after = plain(base)
    for actual, reference in zip(after["samples"].unbind(), expected["samples"].unbind()):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    sample(selected, controller)
    assert controller.report()["last_request"]["hops"] == [0, 1, 2, 3]


@pytest.mark.parametrize("layout", ["different_hop", "independent_qkv"])
def test_actual_compatible_nonuniform_expert_ranks_are_not_a_model_gate(layout):
    base = model()
    tensors = experts(base)
    source = ("transformer.transformer_blocks.0.attn.to_out.0"
              if layout == "different_hop" else "transformer.transformer_blocks.0.attn.to_q")
    hops = [1] if layout == "different_hop" else [0, 1, 2, 3]
    generator = torch.Generator().manual_seed(21102401)
    for hop in hops:
        a_key, b_key = (f"{source}.lora_{part}.{hop}.weight" for part in ("A", "B"))
        a, b = tensors[a_key], tensors[b_key]
        tensors[a_key] = torch.cat((a, torch.randn((1, a.shape[1]), generator=generator) * .01))
        tensors[b_key] = torch.cat((b, torch.randn((b.shape[0], 1), generator=generator) * .01), dim=1)
        probe = torch.randn((3, a.shape[1]), generator=generator)
        actual_delta = F.linear(F.linear(probe, tensors[a_key]), tensors[b_key])
        assert actual_delta.shape == (3, b.shape[0]) and bool(torch.isfinite(actual_delta).all())
    selected, _ = ptd.configure_ptd(base, tensors, files.metadata_for("h3_4step"))
    controller = ptd.get_controller(selected)
    path = "blocks.0.attn.out_proj" if layout == "different_hop" else "blocks.0.attn.qkv_proj"
    target_qkv = None if layout == "different_hop" else 0
    for hop in hops:
        actual_expert = next(e for e in controller.bank[path][hop] if e.qkv_index == target_qkv)
        assert actual_expert.a.shape[0] == 3 and actual_expert.b.shape[1] == 3
        torch.testing.assert_close(actual_expert.a, tensors[f"{source}.lora_A.{hop}.weight"], rtol=0, atol=0)
    output = sample(selected, controller)
    assert all(bool(torch.isfinite(v).all()) for v in output["samples"].unbind())
    report = controller.report()
    assert report["last_request"]["hops"] == [0, 1, 2, 3]
    assert all(f"{hop}:{path}" in report["last_request"]["expert_calls"] for hop in hops)
    assert report["warnings"] and not report["portable_cache_identity"]


def test_last_forward_original_error_is_not_replaced_by_bank_seal_and_retry_recovers():
    base = model()
    selected, _ = ptd.configure_ptd(base, experts(base), files.metadata_for("h3_4step"))
    controller = ptd.get_controller(selected)
    factor = controller.bank["blocks.0.attn.out_proj"][0][0].a
    original, calls = factor.clone(), []

    def fail_last(_module, _args, _output):
        if controller.current()[0] == 3:
            calls.append(3)
            factor.add_(.25)
            raise TypeError("actual PTD selected last-forward failure")

    handle = base.model.diffusion_model.final_layer.register_forward_hook(fail_last)
    try:
        with pytest.raises(TypeError, match="actual PTD selected last-forward failure"):
            sample(selected, controller)
    finally:
        handle.remove()
        factor.copy_(original)
    assert calls == [3] and controller.report()["status"] == "failed"
    assert controller.selection.get() is None and controller.request.get() is None
    sample(selected, controller)
    assert controller.report()["last_request"]["hops"] == [0, 1, 2, 3]


def test_instance_actual_adaln_pre_hook_input_is_not_erased_by_precision_hook():
    base = model()
    root = base.model.diffusion_model
    controller = ptd.PTDController(root, experts(base), files.metadata_for("h3_4step"))
    layer = root.blocks[0].adaln_proj
    calls = []

    def user_pre_hook(module, args):
        calls.append(1)
        return (args[0] + .5, *args[1:])

    handle = layer.register_forward_pre_hook(user_pre_hook)
    try:
        labels = torch.tensor([1 - .999], dtype=torch.float32)
        expected = layer(root.time_embedder(labels))
        with controller.activate(0, .999), controller.forward_scope():
            actual = layer(root.time_embedder(labels))
        assert calls == [1, 1]
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
    finally:
        handle.remove()
    assert not layer._forward_pre_hooks and not layer.linear._forward_pre_hooks


def test_actual_unknown_RoPE_execution_owner_retains_original_attention(monkeypatch):
    base = model()
    kitchen = native.comfy.quant_ops.ck
    original, calls = kitchen.rms_rope_split_half_, []

    def foreign(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(kitchen, "rms_rope_split_half_", foreign)
    selected, _ = ptd.configure_ptd(base, experts(base), files.metadata_for("h3_4step"))
    controller = ptd.get_controller(selected)
    output = sample(selected, controller)
    assert all(bool(torch.isfinite(value).all()) for value in output["samples"].unbind())
    assert calls == [1] * 4
    assert controller.report()["partial"]
