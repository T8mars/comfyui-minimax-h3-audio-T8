"""Actual native mixed precision loading; no post-load restoration or tolerance."""
import json

import comfy.ops
import pytest
import torch
from safetensors.torch import load_file, save_file

from h3_audio_t8_pkg import hyperflow_curve_loader_exp as loader
from h3_audio_t8_pkg import hyperflow_curve_runtime_exp as runtime
from h3_audio_t8_pkg import nodes_hyperflow_curve_exp as nodes
from h3_audio_t8_pkg.modular_sampling import hyperflow_curve_identity as identity
from test_hyperflow_curve_runtime_exp import prepare, evaluate


@pytest.mark.parametrize("assign", [False, True])
def test_actual_native_constructor_FP32_island_ordinary_BF16_and_lora_lifecycle(assign):
    native = comfy.ops.mixed_precision_ops({}, torch.bfloat16, disabled={"int8_tensorwise"})
    policy = loader.precision_operations(native)
    weight = torch.tensor([[.1001, -.3313], [.7311, .00173]], dtype=torch.float16)
    bias = torch.tensor([.02631, -.8237], dtype=torch.float16)
    broken = native.Linear(2, 2, dtype=torch.float32, device="cpu")
    corrected = policy.Linear(2, 2, dtype=torch.float32, device="cpu")
    ordinary = policy.Linear(2, 2, dtype=torch.bfloat16, device="cpu")
    for module in (broken, corrected, ordinary):
        module.load_state_dict({"weight": weight.clone(), "bias": bias.clone()}, assign=assign)
    assert type(corrected) is type(broken) is native.Linear
    assert not torch.equal(broken.weight.float(), weight.float())
    assert torch.equal(corrected.weight.float(), weight.float())
    assert torch.equal(corrected.bias.float(), bias.float())
    assert corrected.factory_kwargs["dtype"] == corrected.weight_comfy_model_dtype == corrected.bias_comfy_model_dtype == torch.float32
    assert ordinary.factory_kwargs["dtype"] == native._compute_dtype == torch.bfloat16
    assert torch.equal(ordinary.weight, broken.weight)
    assert native.Linear is type(broken) and native._compute_dtype == torch.bfloat16
    from comfy.model_patcher import ModelPatcher
    root = torch.nn.Module()
    root.layer = corrected
    root.device = torch.device("cpu")
    model = ModelPatcher(root, torch.device("cpu"), torch.device("cpu"))
    model.add_patches({"layer.weight": ("diff", (torch.full_like(corrected.weight, .003),))}, .7)
    original = corrected.weight.detach().clone()
    snapshots = []
    for _ in range(2):
        model.patch_model()
        snapshots.append(corrected.weight.detach().clone())
        assert not torch.equal(corrected.weight, original)
        assert torch.equal(corrected(torch.ones(1, 2)), torch.nn.functional.linear(torch.ones(1, 2), corrected.weight, corrected.bias.float()))
        model.unpatch_model()
        assert torch.equal(corrected.weight, original)
    assert torch.equal(*snapshots) and torch.equal(broken.weight.float(), weight.bfloat16().float())
    assert not torch.cuda.is_initialized()


def _real_files(tmp_path, monkeypatch):
    _, _, _, weights, files = prepare(tmp_path, monkeypatch, portable=True)
    state = load_file(str(files["base"]))
    for name in loader.ISLANDS:
        for suffix in ("weight", "bias"):
            state[name + "." + suffix] = state[name + "." + suffix].half()
    state["condition_proj.weight"] = torch.round(state["condition_proj.weight"] / .001).to(torch.int8)
    state["condition_proj.weight_scale"] = torch.tensor(.001)
    state["condition_proj.comfy_quant"] = torch.tensor(list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8)
    save_file(state, str(files["base"]))
    new_fit = tmp_path / "mixed-fit.safetensors"
    fit = runtime.fitting.build_fit(files["base"], files["teacher"], files["adapter"], new_fit)
    return fit, weights, files


def test_fresh_actual_Core_model_reconstruction_fit_prior_lora_native_dispatch(tmp_path, monkeypatch):
    import comfy.model_management as management
    fit, weights, files = _real_files(tmp_path, monkeypatch)
    # CPU executes a tiny BF16 backbone with the same mixed constructor which
    # caused the actual GPU failure. No hardware capability is falsely claimed.
    monkeypatch.setattr(management, "unet_manual_cast", lambda *_args: torch.bfloat16)
    monkeypatch.setattr(management, "unet_dtype", lambda **_kwargs: torch.bfloat16)
    first, report = loader.load_curve_base(files["base"])
    diffusion = first.model.diffusion_model
    assert diffusion.dtype == torch.bfloat16
    assert report["fit_checked"] is False and report["quality_accepted"] is False
    assert isinstance(diffusion.condition_proj.weight, comfy.ops.QuantizedTensor)
    runtime.verify_basis(first, fit, files["base"])
    factory, arguments, index = first.cached_patcher_init
    second = factory(*arguments, disable_dynamic=True)[index]
    assert first.model is not second.model and not first.backup and not second.backup
    assert first.model.diffusion_model.blocks[0] is not second.model.diffusion_model.blocks[0]
    before = {name: value.clone() for name, value in second.model.diffusion_model.named_parameters()
              if not isinstance(value, comfy.ops.QuantizedTensor)}
    trunk = "diffusion_model.blocks.0.attn.out_proj.weight"
    first.add_patches({trunk: ("diff", (torch.full_like(diffusion.blocks[0].attn.out_proj.weight, .001),))}, .7)
    first.add_patches({"diffusion_model.blocks.0.adaln_proj.linear.weight":
                      ("diff", (torch.full_like(diffusion.blocks[0].adaln_proj.linear.weight, .0001),))}, .3)
    patched, binding, audit = runtime.install_curve(first, weights, fit, files["base"])
    selected = identity.model_identity(patched)
    assert selected["portable_cache_reuse"] is True
    for _ in range(2):
        patched.patch_model()
        try:
            evaluate(patched, diffusion, binding)
            assert identity.model_identity(patched) == selected
        finally:
            patched.unpatch_model()
        runtime.verify_basis(first, fit, files["base"])
    assert audit["forwards"][-1]["verified_native_projections"] == list(range(51))
    assert all(torch.equal(value, dict(second.model.diffusion_model.named_parameters())[name]) for name, value in before.items())
    layer = diffusion.blocks[0].adaln_proj.linear
    layer.weight_comfy_model_dtype = torch.bfloat16
    assert identity.model_identity(patched)["common_base_sha256"] != selected["common_base_sha256"]
    layer.weight_comfy_model_dtype = torch.float32
    assert identity.model_identity(patched) == selected
    diffusion.blocks[0].adaln_proj.linear.weight.data[0, 0] += .000001
    with pytest.raises(ValueError, match="loaded curve basis"):
        runtime.verify_basis(first, fit, files["base"])
    assert not torch.cuda.is_initialized()


def test_preflight_rejects_quantized_basis_before_file_hash_and_reconstruction_changed_bytes(tmp_path, monkeypatch):
    fit, _, files = _real_files(tmp_path, monkeypatch)
    first, _ = loader.load_curve_base(files["base"])
    state = load_file(str(files["base"]))
    state["blocks.0.adaln_proj.linear.comfy_quant"] = state["condition_proj.comfy_quant"].clone()
    save_file(state, str(files["base"]))
    def forbidden(_path):
        raise AssertionError("Invalid precision islands must be refused before full hashing")
    original = loader.fitting._file
    monkeypatch.setattr(loader.fitting, "_file", forbidden)
    with pytest.raises(ValueError, match="unquantized"):
        loader.load_curve_base(files["base"])
    monkeypatch.setattr(loader.fitting, "_file", original)
    state.pop("blocks.0.adaln_proj.linear.comfy_quant")
    state["blocks.0.attn.out_proj.weight"][0, 0] += .001
    save_file(state, str(files["base"]))
    factory, arguments, index = first.cached_patcher_init
    with pytest.raises(ValueError, match="changed before native reconstruction"):
        factory(*arguments)[index]
    assert fit.metadata["base"]["sha256"] != loader.fitting._file(files["base"])["sha256"]


def test_public_loader_only_selects_fresh_base_no_loaded_model_or_fit_input():
    cls = nodes.MiniMaxH3HyperFlowCurveBaseLoaderEXPT8
    schema = cls.define_schema().get_v1_info(cls)
    assert set(schema.input["required"]) == {"base_file"}
    assert schema.output == ["MODEL", "STRING"]
    assert loader.precision_operations(comfy.ops.manual_cast) is comfy.ops.manual_cast
