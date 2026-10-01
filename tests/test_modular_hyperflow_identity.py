"""Real 50+2 native tiny structure/closures, synthetic adapter; no GPU claim."""
from dataclasses import replace

import comfy.patcher_extension
import pytest
import torch

from h3_audio_t8_pkg.modular_sampling import hyperflow_identity as identity
from h3_audio_t8_pkg import hyperflow_runtime_advanced as runtime
from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from h3_audio_t8_pkg.vdn_attention_compat import _factory_closure
from test_modular_hyperflow import inputs, head


def captured(model):
    forward = model.object_patches["diffusion_model.time_embedder.forward"]
    return _factory_closure(forward, runtime._install_two_time, "time_forward")


def test_distinct_loader_and_rebuilt_base_have_identical_portable_identity(monkeypatch):
    low, high, _, _ = inputs(monkeypatch)
    other, _, _, _ = inputs(monkeypatch)
    assert low.model is not other.model
    assert low.get_attachment(runtime.ATTACHMENT_KEY).owner != high.get_attachment(runtime.ATTACHMENT_KEY).owner
    first = identity.model_identity(low)
    assert first["portable_cache_reuse"] is True
    assert first == identity.model_identity(high) == identity.model_identity(other)


def test_actual_stage_cleanup_and_warm_cache_do_not_change_identity(monkeypatch):
    low, _, source, positive = inputs(monkeypatch)
    before = identity.model_identity(low)
    head(low, source, positive)
    assert identity.model_identity(low) == before


def test_content_lora_is_independent_but_not_omitted(monkeypatch):
    low, high, _, _ = inputs(monkeypatch, distinct=True)
    original = dict(low.object_patches)
    left, right = identity.model_identity(low), identity.model_identity(high)
    assert left["sha256"] != right["sha256"]
    assert left["common_base_sha256"] == right["common_base_sha256"]
    assert left["adapter_sha256"] == right["adapter_sha256"]
    assert low.object_patches == original
    assert len(low.object_patches) == 52


@pytest.mark.parametrize("fault", ["wrapper", "native_delegate", "owner", "block_default", "remap", "endpoint_cache",
                                   "endpoint_weight", "snapshot", "patch_removed", "binding", "extra_attribute"])
def test_unknown_or_mutated_owner_cannot_claim_portable(monkeypatch, fault):
    low, _, source, positive = inputs(monkeypatch)
    time = captured(low)
    if fault == "wrapper":
        low.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL,
                                 "user", lambda executor, *a, **kw: executor(*a, **kw))
    elif fault == "native_delegate":
        time["source_forward"].__self__.forward = lambda *a: a
    elif fault == "owner":
        low.object_patches["diffusion_model.blocks.0.forward"]._t8_hyperflow_owner = "foreign"
    elif fault == "block_default":
        low.object_patches["diffusion_model.blocks.0.forward"].__kwdefaults__["_inner"] = lambda *a: a
    elif fault == "remap":
        block = low.object_patches["diffusion_model.blocks.0.forward"]
        for name, cell in zip(block.__code__.co_freevars, block.__closure__):
            if name == "remap":
                cell.cell_contents = lambda *a: a
    elif fault in {"endpoint_cache", "snapshot"}:
        head(low, source, positive)
        endpoint = _factory_closure(time["endpoint_forward"], runtime._endpoint_forward, "forward")
        with torch.inference_mode():
            if fault == "endpoint_cache":
                values = endpoint["cache"][torch.device("cpu")]
                endpoint["cache"][torch.device("cpu")] = (values[0] + .2, *values[1:])
            else:
                endpoint["wi"].add_(.2)
    elif fault == "endpoint_weight":
        time["weights"].endpoint["proj_in"] = (*time["weights"].endpoint["proj_in"][:2], 999.)
    elif fault == "patch_removed":
        low.patches.pop(next(iter(low.patches)))
    elif fault == "binding":
        low.set_attachments(runtime.ATTACHMENT_KEY, replace(low.get_attachment(runtime.ATTACHMENT_KEY), gate=.2))
    else:
        time["endpoint_forward"].foreign = True
    with pytest.raises(UnverifiedModelStack):
        identity.model_identity(low)


def test_changed_base_changes_common_identity(monkeypatch):
    low, _, _, _ = inputs(monkeypatch)
    before = identity.model_identity(low)
    with torch.no_grad():
        low.model.diffusion_model.blocks[0].attn.qkv_proj.weight.add_(.1)
    after = identity.model_identity(low)
    assert after["common_base_sha256"] != before["common_base_sha256"]
    assert after["sha256"] != before["sha256"]
