"""Real tiny Core VDN numerical baseline before the modular adapter.

Only downloaded-asset/audit/loading gates are replaced by deterministic tiny
weights. The original composer, additional-model lifecycle, block/layout route,
grouped softmax, linear recurrence and Core AV sampler execute on CPU.
This is NOT a pretrained OpenVDN, learned-upscaler or media qualification.
"""
import json
from pathlib import Path

import pytest
import torch
import comfy.nested_tensor
import comfy.model_patcher
import comfy.ops
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import vdn_h3_advanced as vdn, vdn_two_pass
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning


def tiny_vdn(stage="stage_dmd_8nfe"):
    bare = model()
    block = vdn.VDNBlockBranch.__new__(vdn.VDNBlockBranch)
    torch.nn.Module.__init__(block)
    block.linear_attention = vdn.BidirectionalLinearBranch(24, 3, 128, dtype=torch.float32)
    block.softmax_gate = vdn.OutputGate(24, 3, dtype=torch.float32)
    block.to_out_linear = comfy.ops.manual_cast.Linear(384, 24, bias=False, dtype=torch.float32)
    network = vdn.VDNBranchModel.__new__(vdn.VDNBranchModel)
    torch.nn.Module.__init__(network)
    network.blocks = torch.nn.ModuleList([block])
    generator = torch.Generator().manual_seed(863)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * .03)
    branch = comfy.model_patcher.ModelPatcher(network, torch.device("cpu"), torch.device("cpu"))
    audit = {"root": ".", "base_provenance_exact": False, "base_variant": "tiny_test_only",
             "adapter_strategy": "native_full_width", "license": "test_fixture_no_downloaded_weights"}
    key, weight = next((key, value) for key, value in bare.model_state_dict().items()
        if ".blocks.0.attn." in key and key.endswith("weight") and value.ndim == 2)
    patches = {key: ("diff", (torch.ones_like(weight) * .0001,))}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(vdn, "audit_vdn_runtime", lambda *a, **kw: (True, audit))
        patch.setattr(vdn, "_stage_assets", lambda *a: {
            "linear_branch": Path("tiny"), "default_adapter": Path("tiny"), "turbo_adapter": Path("tiny")})
        patch.setattr(vdn, "load_branch_model", lambda *a: (branch, {"test_boundary": "tiny_random_branch"}))
        patch.setattr(vdn, "_load_adapter_patches", lambda *a: (
            patches, patches, patches, {}, {"checked_targets": 1, "bias_diff_targets": 0}))
        prepared, _ = vdn.compose_vdn_model(bare, ".", stage)
    return prepared, branch


def source(frames=2):
    return {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, frames, 4, 4), torch.zeros(1, 32, 2, 8)))}


def run(prepared, sampler, sigmas, latent, seed=43):
    return SamplerCustomAdvanced.execute(RandomNoise.execute(seed).result[0],
        BasicGuider.execute(prepared, conditioning()).result[0], sampler, sigmas, latent).result


@pytest.mark.parametrize("stage,frames,tail", [
    ("stage_dmd_8nfe", 2, 4), ("stage_dmd_8nfe", 16, 4), ("stage_b_50nfe", 2, 5)])
def test_real_original_vdn_full_then_own_tail_executes_its_actual_backend(monkeypatch, stage, frames, tail):
    prepared, branch = tiny_vdn(stage)
    calls = {"window": 0, "linear": 0, "selector": 0}
    old_window = vdn.window_softmax_sdpa

    def window(*args, **kwargs):
        calls["window"] += 1
        return old_window(*args, **kwargs)

    def selector(function, *args, **kwargs):
        calls["selector"] += 1
        return function(*args, **kwargs)

    monkeypatch.setattr(vdn, "window_softmax_sdpa", window)
    handle = branch.model.blocks[0].linear_attention.register_forward_hook(
        lambda *args: calls.__setitem__("linear", calls["linear"] + 1))
    prepared.set_model_optimized_attention(selector)
    latent = source(frames)
    low, sampler, full, _ = vdn.setup_vdn_execution(prepared, latent)
    first = run(low, sampler, full, latent)[0]
    high, second_sampler, sigmas, raw = vdn_two_pass.setup_vdn_refine(prepared, first, first, tail)
    second = run(high, second_sampler, sigmas, first, seed=44)[0]
    handle.remove()
    steps = vdn.STAGES[stage]["steps"]
    assert json.loads(raw)["total_nfe"] == steps + tail
    assert torch.equal(sigmas, full[-tail-1:])
    assert calls == {"window": steps + tail, "linear": steps + tail if frames > 10 else 0, "selector": 0}
    for output in (first, second):
        assert all(torch.isfinite(value).all() for value in output["samples"].unbind())
        assert [tuple(value.shape) for value in output["samples"].unbind()] == [
            tuple(value.shape) for value in latent["samples"].unbind()]
    assert not torch.equal(first["samples"].unbind()[0], second["samples"].unbind()[0])
    assert prepared.additional_models[vdn.ADDITIONAL_MODEL_KEY][0] is branch
    assert not torch.cuda.is_initialized()
