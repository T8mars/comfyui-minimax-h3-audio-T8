"""Actual Core native base/storage and own fit, CPU only; no model-file fiction."""
from dataclasses import replace

import comfy.patcher_extension
import pytest
import torch

from h3_audio_t8_pkg.modular_sampling import hyperflow_curve_identity as identity
from h3_audio_t8_pkg import hyperflow_curve_runtime_exp as runtime
from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from test_hyperflow_curve_runtime_exp import prepare, evaluate


def setup(tmp_path, monkeypatch):
    base, diffusion, fit, weights, files = prepare(tmp_path, monkeypatch, portable=True)
    low, _, _ = runtime.install_curve(base, weights, fit, files["base"])
    high, _, _ = runtime.install_curve(base, weights, fit, files["base"])
    return base, diffusion, low, high, files


def test_portable_native_pending_warm_restored_and_rebuilt_content_equal(tmp_path, monkeypatch):
    base, diffusion, low, high, files = setup(tmp_path, monkeypatch)
    before_files = {name: runtime.fitting._file(path) for name, path in files.items()}
    first = identity.model_identity(low)
    assert first["portable_cache_reuse"] is True and first["model_filename_trusted"] is False
    assert first == identity.model_identity(high)
    patch_descriptors = dict(low.object_patches)
    low.patch_model()
    try:
        assert identity.model_identity(low) == first
        evaluate(low, diffusion, low.get_attachment(runtime.KEY))
        assert identity.model_identity(low) == first
        assert identity.model_identity(high) == first
    finally:
        low.unpatch_model()
    assert identity.model_identity(low) == first
    other_path = tmp_path / "other"
    other_path.mkdir()
    other_base, _, _, _, _ = prepare(other_path, monkeypatch, portable=True)
    details = low.get_attachment(runtime.DETAILS_KEY)
    other, _, _ = runtime.install_curve(other_base, details.weights, details.fit, files["base"])
    assert other.model is not low.model
    assert identity.model_identity(other) == first
    assert low.object_patches == patch_descriptors
    assert base.get_attachment(runtime.KEY) is None
    assert {name: runtime.fitting._file(path) for name, path in files.items()} == before_files
    assert not torch.cuda.is_initialized()


def test_prior_and_per_stage_lora_kept_in_selected_not_common_base(tmp_path, monkeypatch):
    _, diffusion, low, high, _ = setup(tmp_path, monkeypatch)
    key = "diffusion_model.blocks.0.attn.out_proj.weight"
    low.add_patches({key: ("diff", (torch.full_like(diffusion.blocks[0].attn.out_proj.weight, .001),))}, .7)
    left, right = identity.model_identity(low), identity.model_identity(high)
    assert left["sha256"] != right["sha256"]
    assert left["common_base_sha256"] == right["common_base_sha256"]
    assert left["adapter_sha256"] == right["adapter_sha256"]
    assert len(low.patches[key]) == 2


@pytest.mark.parametrize("fault", ["owner", "delegate", "fit", "adapter", "patch", "binding", "wrapper", "protocol", "latent"])
def test_unknown_or_modified_execution_cannot_claim_portable(tmp_path, monkeypatch, fault):
    _, diffusion, low, _, _ = setup(tmp_path, monkeypatch)
    before = dict(low.object_patches)
    details = low.get_attachment(runtime.DETAILS_KEY)
    binding = low.get_attachment(runtime.KEY)
    if fault == "owner":
        low.object_patches["diffusion_model.blocks.0.forward"]._t8_curve_owner = "other"
    elif fault == "delegate":
        low.object_patches["diffusion_model.blocks.0.forward"].__kwdefaults__["_inner"] = lambda *a, **kw: a
    elif fault == "fit":
        details.fit.generated[0, 0, 0] += .1
    elif fault == "adapter":
        details.weights.endpoint["proj_in"][0][0, 0] += .1
    elif fault == "patch":
        low.patches.pop(next(iter(low.patches)))
    elif fault == "binding":
        low.set_attachments(runtime.KEY, replace(binding, gate=.1))
    elif fault == "wrapper":
        low.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL,
            "user", lambda executor, *a, **kw: executor(*a, **kw))
    elif fault == "protocol":
        low.model.process_latent_in = lambda latent: latent
    else:
        low.model.latent_format.process_in = lambda latent: latent
    with pytest.raises((UnverifiedModelStack, ValueError)):
        identity.model_identity(low)
    assert low.object_patches == before
    assert diffusion.use_adaln_curves is True
    assert not hasattr(diffusion, "time_embedder")


def test_actual_backbone_content_changes_common_identity_not_fit_certificate(tmp_path, monkeypatch):
    _, diffusion, low, _, _ = setup(tmp_path, monkeypatch)
    before = identity.model_identity(low)
    with torch.no_grad():
        diffusion.blocks[0].attn.out_proj.weight.add_(.1)
    after = identity.model_identity(low)
    assert before["common_base_sha256"] != after["common_base_sha256"]
    assert before["sha256"] != after["sha256"]


@pytest.mark.parametrize("target", ["table", "weight", "bias"])
def test_modified_actual_curve_basis_rejects_portable_and_forward(tmp_path, monkeypatch, target):
    _, diffusion, low, _, _ = setup(tmp_path, monkeypatch)
    tensor = (diffusion.adaln_t_table if target == "table" else
        getattr(diffusion.blocks[0].adaln_proj.linear, target))
    with torch.no_grad():
        tensor.flatten()[0] += .1
    with pytest.raises(UnverifiedModelStack, match="basis changed"):
        identity.model_identity(low)
    low.patch_model()
    try:
        with pytest.raises(ValueError, match="basis changed"):
            evaluate(low, diffusion, low.get_attachment(runtime.KEY))
        assert runtime.ACTIVE.get() is None
    finally:
        low.unpatch_model()


@pytest.mark.parametrize("target", ["time", "block", "final"])
def test_live_foreign_marker_is_not_a_normalizable_sibling(tmp_path, monkeypatch, target):
    _, diffusion, low, high, _ = setup(tmp_path, monkeypatch)
    low.patch_model()
    try:
        def fake(*args, **kwargs):
            return args
        fake._t8_curve_owner = low.get_attachment(runtime.KEY).owner
        fake._t8_curve_inner = diffusion.blocks[0].forward
        if target == "time":
            diffusion.time_embedder = fake
        elif target == "block":
            diffusion.blocks[0].forward = fake
        else:
            diffusion.final_layer.forward = fake
        with pytest.raises(UnverifiedModelStack):
            identity.model_identity(high)
    finally:
        low.unpatch_model()
