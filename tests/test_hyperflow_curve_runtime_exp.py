"""New owner tested on real tiny native pruned Core, not mocked execution."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
import comfy.ops
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from comfy.model_patcher import ModelPatcher

from h3_audio_t8_pkg import hyperflow_curve_fit_exp as fitting
from h3_audio_t8_pkg import hyperflow_curve_runtime_exp as runtime
from h3_audio_t8_pkg.hyperflow_weights_advanced import load_hyperflow_original, plan_source_keys
from h3_audio_t8_pkg.hyperflow_runtime_advanced import HyperFlowStep, push_step, pop_step, active_step, ATTACHMENT_KEY
from h3_audio_t8_pkg.hyperflow_sampling_advanced import build_hyperflow_plan
from test_hyperflow_advanced import _metadata, _synthetic_source

def prepare(tmp_path, monkeypatch, *, portable=False):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    torch.manual_seed(321)
    options = dict(hidden_size=8, num_layers=50, token_refiner_num_layers=2,
        num_attention_heads=2, attention_head_dim=8, ffn_hidden_size=16,
        text_dim=8, timestep_input_dim=4, time_embed_hidden_size=8,
        time_embed_dim=4, rope_inv_freq_len=1,
        dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
    teacher = native.MiniMaxH3Model(**options)
    options.update(time_embed_dim=8, adaln_curve_grid=1025)
    pruned = native.MiniMaxH3Model(**options)
    for network in (teacher, pruned):
        for parameter in network.parameters():
            torch.nn.init.normal_(parameter, std=0.03)
        network.rope.inv_freq.fill_(1.)
        network.requires_grad_(False)
    torch.nn.init.normal_(pruned.adaln_t_table, std=0.03)
    source = _synthetic_source()
    plan = plan_source_keys(list(source))
    modules = dict(teacher.named_modules())
    for target, entry in plan.items():
        module = modules.get(target.replace("endpoint_time_embedder", "time_embedder"))
        output, width = module.weight.shape
        if target.endswith(".attn.qkv_proj"):
            for pair in entry["parts"].values():
                source[pair["A"]] = torch.full((2, width), 0.003)
                source[pair["B"]] = torch.full((output // 3, 2), 0.003)
        else:
            source[entry["A"]] = torch.full((2, width), 0.003)
            source[entry["B"]] = torch.full((output, 2), 0.003)
    files = {name: tmp_path / (name + ".safetensors") for name in ("base", "teacher", "adapter", "fit")}
    save_file(pruned.state_dict(), str(files["base"]))
    save_file(teacher.state_dict(), str(files["teacher"]))
    save_file(source, str(files["adapter"]), metadata=_metadata())
    fit = fitting.build_fit(files["base"], files["teacher"], files["adapter"], files["fit"])
    weights = load_hyperflow_original(files["adapter"])

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = pruned
            self.device = torch.device("cpu")

    base = ModelPatcher(Base(), torch.device("cpu"), torch.device("cpu"))
    if portable:
        from test_progressive_sampling_runtime import tiny_model
        base = tiny_model()
        base.model.diffusion_model = pruned
        base.model.model_config.unet_config.update({key: value for key, value in options.items()
            if key not in {"operations", "device"}})
    return base, pruned, fit, weights, files


def evaluate(patched, diffusion, binding, index=0, *, soft_mask=False, masks=None, payload=None):
    plan = runtime.build_plan(patched)
    grid = plan.video_segment
    token = push_step(HyperFlowStep(binding.owner, index, float(grid[index]), float(grid[index + 1])))
    try:
        kwargs = {}
        if soft_mask:
            kwargs["denoise_mask"] = torch.full((1, 1, 2, 4, 4), 0.5)
        kwargs.update(masks or {})
        with torch.no_grad():
            return diffusion([torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3)],
                torch.tensor([float(grid[index]) * 1000]), torch.randn(1, 2, 8, dtype=diffusion.dtype),
                transformer_options={"sample_sigmas": grid, "wrappers": patched.wrappers,
                                     "minimax_h3_sigma_shift_video": 12., "minimax_h3_sigma_shift_audio": 3.},
                minimax_payload={"layout": native.PackedLayout(2, 2, 4, 4, 3), **(payload or {})}, **kwargs)
    finally:
        pop_step(token)


def test_actual_native50_each_of_8_intervals_and_restore(tmp_path, monkeypatch):
    base, diffusion, fit, weights, files = prepare(tmp_path, monkeypatch)
    source = {name: fitting._file(path) for name, path in files.items()}
    native_forward = diffusion.blocks[0].forward
    called = []

    def user_delegate(*args, **kwargs):
        called.append(1)
        return native_forward(*args, **kwargs)

    diffusion.blocks[0].forward = user_delegate
    patch_key = "diffusion_model.blocks.0.attn.out_proj.weight"
    base.add_patches({patch_key: ("diff", (torch.zeros_like(diffusion.blocks[0].attn.out_proj.weight),))}, .17)
    prior = deepcopy(base.patches)
    patched, binding, audit = runtime.install_curve(base, weights, fit, files["base"])
    assert len(patched.patches) == 208
    assert patched.patches[patch_key][0][0] == prior[patch_key][0][0]
    assert base.get_attachment(runtime.KEY) is None
    assert patched.get_attachment(ATTACHMENT_KEY) is None
    with pytest.raises(ValueError, match="dedicated HyperFlow Loader"):
        build_hyperflow_plan(patched)
    patched.patch_model()
    try:
        for index in range(8):
            output = evaluate(patched, diffusion, binding, index)
            assert all(torch.isfinite(part).all() for part in output)
            assert runtime.ACTIVE.get() is None and active_step() is None
        assert len(called) == 8
        assert [row["interval"] for row in audit["forwards"]] == list(range(8))
        assert all(row["invoked_projection_wrappers"] == list(range(51)) for row in audit["forwards"])
        assert all(row["verified_native_projections"] == list(range(1, 51)) for row in audit["forwards"])
        assert all(row["complete_51"] is False for row in audit["forwards"])
        assert audit["portable_cache_reuse"] is False
        assert audit["full_backbone_file_identity_certified"] is False
    finally:
        patched.unpatch_model()
    assert diffusion.use_adaln_curves is True
    assert not hasattr(diffusion, "time_embedder")
    assert diffusion.blocks[0].forward is user_delegate
    assert all(fitting._file(path) == source[name] for name, path in files.items())
    assert not torch.cuda.is_initialized()


def test_actual_unsupported_soft_mask_rejected_not_single_time_fallback(tmp_path, monkeypatch):
    base, diffusion, fit, weights, files = prepare(tmp_path, monkeypatch)
    patched, binding, audit = runtime.install_curve(base, weights, fit, files["base"])
    patched.patch_model()
    try:
        with pytest.raises(ValueError, match="off-grid/soft-mask"):
            evaluate(patched, diffusion, binding, soft_mask=True)
        assert runtime.ACTIVE.get() is None and active_step() is None
        assert audit["forwards"] == []
    finally:
        patched.unpatch_model()


def test_pair_selection_pinned_endpoints_and_no_shared_51_vector(tmp_path, monkeypatch):
    _, _, fit, weights, _ = prepare(tmp_path, monkeypatch)
    pairs = [(float(fit.t[0]), float(fit.r[0])), (0.999, 0.999), (1., 1.)]
    selected = runtime._coordinates(fit, pairs, torch.device("cpu"))
    assert selected.shape == (51, 3, 8)
    assert torch.equal(selected[:, 0], fit.generated[:, 0])
    assert torch.equal(selected[:, 2], fit.pinned[:, -1])
    assert not torch.equal(selected[0], selected[49])
    bad = deepcopy(fit.metadata)
    bad["raw_sigmas"][1] -= .01
    with pytest.raises(ValueError, match="in-memory metadata"):
        runtime._recipe(weights, replace(fit, metadata=bad))


def test_changed_actual_loaded_basis_wrong_file_or_plan_does_not_execute(tmp_path, monkeypatch):
    base, diffusion, fit, weights, files = prepare(tmp_path, monkeypatch)
    diffusion.blocks[0].adaln_proj.linear.weight.data[0, 0] += .01
    with pytest.raises(ValueError, match="loaded curve basis"):
        runtime.install_curve(base, weights, fit, files["base"])
    assert base.get_attachment(runtime.KEY) is None
    assert base.patches == {}


def test_actual_bf16_backbone_keeps_pruned_projection_coordinates_fp32(tmp_path, monkeypatch):
    base, diffusion, fit, weights, files = prepare(tmp_path, monkeypatch)
    diffusion.dtype = torch.bfloat16
    for name, parameter in diffusion.named_parameters():
        if ".adaln_proj." not in name and not name.startswith((
                "video_patch_proj.", "audio_patch_proj.", "final_layer.video_out.", "final_layer.audio_out.")):
            parameter.data = parameter.data.to(torch.bfloat16)
    original = diffusion.blocks[0].forward
    observed = []

    def delegate(x, t_emb, *args, **kwargs):
        observed.append(t_emb.dtype)
        return original(x, t_emb, *args, **kwargs)

    diffusion.blocks[0].forward = delegate
    patched, binding, audit = runtime.install_curve(base, weights, fit, files["base"])
    patched.patch_model()
    try:
        output = evaluate(patched, diffusion, binding)
        assert all(torch.isfinite(value).all() for value in output)
        assert observed == [torch.float32]
        assert audit["forwards"][0]["complete_51"] is False
        assert audit["forwards"][0]["verified_native_projections"] == list(range(1, 51))
    finally:
        patched.unpatch_model()


@pytest.mark.parametrize("failure", [False, True])
def test_pending_native_object_patches_are_delegated_and_restored(tmp_path, monkeypatch, failure):
    base, diffusion, fit, weights, files = prepare(tmp_path, monkeypatch)
    calls = []
    originals = {"block": diffusion.blocks[0].forward, "final": diffusion.final_layer.forward}
    def block(*args, **kwargs):
        calls.append("block")
        return originals["block"](*args, **kwargs)
    def final(*args, **kwargs):
        calls.append("final")
        if failure:
            raise RuntimeError("real pending final failure")
        return originals["final"](*args, **kwargs)
    base.add_object_patch("diffusion_model.blocks.0.forward", block)
    base.add_object_patch("diffusion_model.final_layer.forward", final)
    # A user's similar marker must not authorize stripping their callable.
    block._t8_curve_owner, block._t8_curve_inner = "user-marker", originals["block"]
    pending = dict(base.object_patches)
    patched, binding, audit = runtime.install_curve(base, weights, fit, files["base"])
    patched.patch_model()
    try:
        if failure:
            with pytest.raises(RuntimeError, match="real pending final failure"):
                evaluate(patched, diffusion, binding)
            assert audit["forwards"] == []
        else:
            evaluate(patched, diffusion, binding)
            assert audit["forwards"][0]["complete_51"] is False
            assert audit["forwards"][0]["verified_native_projections"] == list(range(1, 50))
        assert calls == ["block", "final"]
        assert runtime.ACTIVE.get() is None and active_step() is None
    finally:
        patched.unpatch_model()
    assert base.object_patches == pending
    assert diffusion.blocks[0].forward == originals["block"]
    assert diffusion.final_layer.forward == originals["final"]
    assert not hasattr(diffusion, "time_embedder")


@pytest.mark.parametrize("stream", ["video", "audio"])
@pytest.mark.parametrize("soft", [False, True])
def test_actual_binary_or_soft_masks_use_core_pins_not_low_augmentation(tmp_path, monkeypatch, stream, soft):
    base, diffusion, fit, weights, files = prepare(tmp_path, monkeypatch)
    if stream == "video":
        mask = torch.ones(1, 1, 2, 4, 4)
        mask[..., :2, :2] = .5 if soft else 0.
        masks = {"denoise_mask": mask}
    else:
        masks = {"audio_denoise_mask": torch.tensor([[[[.5 if soft else 0., 1., 1.],
                                                      [.5 if soft else 0., 1., 1.]]]])}
    patched, binding, audit = runtime.install_curve(base, weights, fit, files["base"])
    patched.patch_model()
    try:
        if soft:
            with pytest.raises(ValueError, match="off-grid/soft-mask"):
                evaluate(patched, diffusion, binding, masks=masks,
                    payload={"visual_cond_noise_aug": .1, "audio_cond_noise_aug": .1})
            assert audit["forwards"] == []
        else:
            output = evaluate(patched, diffusion, binding, masks=masks,
                payload={"visual_cond_noise_aug": .1, "audio_cond_noise_aug": .1})
            assert all(torch.isfinite(x).all() for x in output)
            assert audit["forwards"][0]["complete_51"] is True
            pairs = audit["pairs"][0]["pairs"]
            pin = native.VISUAL_COND_TIMESTEP if stream == "video" else native.AUDIO_COND_TIMESTEP
            assert (runtime.full_runtime._f32(pin), runtime.full_runtime._f32(pin)) in pairs
        assert runtime.ACTIVE.get() is None and active_step() is None
    finally:
        patched.unpatch_model()


def test_nonzero_interval_cannot_restart_with_only_continuation_boolean(tmp_path, monkeypatch):
    from comfy.nested_tensor import NestedTensor
    base, _, fit, weights, files = prepare(tmp_path, monkeypatch)
    patched, _, _ = runtime.install_curve(base, weights, fit, files["base"])
    plan = runtime.build_plan(patched, 4, 8)
    latent = {"samples": NestedTensor((torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 3)))}
    with pytest.raises(ValueError, match="captured x_sigma provider"):
        runtime.setup_sampler(patched, latent, plan, continuation=True)


def test_unknown_delegate_can_skip_native_without_false_projection_certificate(tmp_path, monkeypatch):
    base, diffusion, fit, weights, files = prepare(tmp_path, monkeypatch)
    calls = []
    def skipped(x, *_args, **_kwargs):
        calls.append(1)
        return x
    base.add_object_patch("diffusion_model.blocks.0.forward", skipped)
    patched, binding, audit = runtime.install_curve(base, weights, fit, files["base"])
    patched.patch_model()
    try:
        output = evaluate(patched, diffusion, binding)
        assert calls == [1] and all(torch.isfinite(part).all() for part in output)
        row = audit["forwards"][0]
        assert row["invoked_projection_wrappers"] == list(range(51))
        assert row["verified_native_projections"] == list(range(1, 51))
        assert row["complete_51"] is False
    finally:
        patched.unpatch_model()
