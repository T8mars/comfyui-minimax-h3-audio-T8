"""Synthetic full-key/tiny actual-Core qualification, not trained GPU quality."""

from copy import deepcopy
from contextlib import contextmanager
import json

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import ptd_checkpoint as old_files
from h3_audio_t8_pkg import ptd_compact as compact
from h3_audio_t8_pkg import ptd_compact_checkpoint as files
from h3_audio_t8_pkg import ptd_sampling as old_clock
from h3_audio_t8_pkg import ptd_advanced as ptd
from h3_audio_t8_pkg import nodes_ptd_advanced as nodes
from h3_audio_t8_pkg import ptd_continuous as continuous
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent


@pytest.fixture(autouse=True)
def actual_cpu(monkeypatch):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    with torch.inference_mode():
        yield
    assert not torch.cuda.is_initialized()


def stacked(base, hops=4, steps=32, *, quantized=False, alpha=2.):
    root, tensors = base.model.diffusion_model, {}
    generator = torch.Generator().manual_seed(241402)
    for prefix, count in (("transformer_blocks", len(root.blocks)),
                          ("token_refiner.refiner_blocks", len(root.token_refiner.blocks))):
        for index in range(count):
            for part in old_files.PARTS:
                source = f"{prefix}.{index}.{part}"
                target, _, _, qkv, _, _, _ = old_files.map_factor(f"transformer.{source}.lora_A.0.weight")
                module = root.get_submodule(target)
                width = module.out_features // (3 if qkv is not None else 1)
                tensors[source + ".lora_down"] = torch.randn(hops, 2, module.in_features, generator=generator) * .02
                tensors[source + ".lora_up"] = torch.randn(hops, width, 2, generator=generator) * .02
    for prefix, path in files.HEADS.items():
        module = root.get_submodule(path)
        tensors[prefix + ".weight"] = torch.stack([
            module.weight.detach().clone().float() + (step + 1) * .0001 for step in range(steps)])
        if module.bias is not None:
            tensors[prefix + ".bias"] = torch.stack([
                module.bias.detach().clone().float() + (step + 1) * .00001 for step in range(steps)])
    metadata = {"ptd_num_hops": str(hops), "pdd_num_steps": str(steps), "lora_rank": "2", "lora_alpha": str(alpha)}
    if quantized:
        # Independent test producer: fixed a50419a7 pdd.to_int8 formula.
        for key in list(tensors):
            if key.endswith((".lora_down", ".lora_up")):
                x = tensors[key].float()
                scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
                tensors[key] = (x / scale).round().clamp(-127, 127).to(torch.int8)
                tensors[key + "_scale"] = scale.squeeze(-1)
        metadata["ptd_compact"] = json.dumps({"schema": 1, "quant": "int8 per-row"})
    return tensors, metadata


def sample(selected, controller, seed=241402):
    guider = BasicGuider.execute(selected, conditioning()).result[0]
    return SamplerCustomAdvanced.execute(RandomNoise.execute(seed).result[0], guider,
        compact.make_compact_sampler(controller), compact.compact_sigmas(controller), latent()).result[0]


@pytest.mark.parametrize("hops", [4, 8])
def test_distinct_checkpoint_grid_fine_head_plans_and_audio_chord(hops):
    grid = compact.compact_grid(hops, 32)
    fine = 12 * torch.linspace(1., 0., 33, dtype=torch.float64)
    fine /= 1 + 11 * torch.linspace(1., 0., 33, dtype=torch.float64)
    expected = torch.tensor([.999 * float(fine[h * (32 // hops)]) for h in range(hops)] + [0.], dtype=torch.float64)
    assert torch.equal(grid, expected)
    assert grid[0] == .999 and grid[-1] == 0
    if hops == 4:
        assert not torch.equal(grid, old_clock.ptd_sigmas("h3_4step"))
    for shift, plan in zip((12., 3.), compact.compact_plans(hops, 32)):
        assert plan.shape == (hops, 32)
        for hop, row in enumerate(plan):
            start, stop = hop * (32 // hops), (hop + 1) * (32 // hops)
            assert torch.count_nonzero(row[:start]) == 0 and torch.count_nonzero(row[stop:]) == 0
            sigma = [shift * ((32-i)/32) / (1+(shift-1)*((32-i)/32)) for i in range(33)]
            times = [1. - value for value in sigma]
            widths = [times[i+1] - times[i] for i in range(start, stop)]
            reference = torch.tensor([value/sum(widths) for value in widths], dtype=torch.float32)
            torch.testing.assert_close(row[start:stop], reference, rtol=1e-7, atol=0)
            assert float(row.sum()) == pytest.approx(1., abs=1e-7)
    for start, stop, factor in zip(grid[:-1], grid[1:], compact.audio_chord_corrections(grid)):
        actual = factor * old_clock.audio_sigma_prime(float(start))
        expected_chord = (old_clock.audio_sigma(float(start))-old_clock.audio_sigma(float(stop))) / float(start-stop)
        assert actual == pytest.approx(expected_chord, rel=1e-14)


@pytest.mark.parametrize("scale", [0., -0., -2., .01])
def test_schema1_signed_zero_and_negative_scales_follow_literal_producer_formula(scale):
    q = torch.tensor([[[2, -3, 127], [0, -127, 3]]], dtype=torch.int8)
    s = torch.full((1, 2), scale, dtype=torch.float32)
    before_q, before_s = q.clone(), s.clone()
    actual = files.decode_rows(q, s, "factor")
    expected = (q.float() * s.unsqueeze(-1)).to(torch.bfloat16)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(q, before_q) and torch.equal(s.view(torch.uint8), before_s.view(torch.uint8))


@pytest.mark.parametrize("quantized", [False, True])
def test_full_key_loader_does_not_mutate_source_and_keeps_all_refiner_and_head_contracts(quantized):
    base = model()
    tensors, metadata = stacked(base, quantized=quantized)
    before = {key: value.clone() for key, value in tensors.items()}
    parsed = files.parse_compact_checkpoint(tensors, metadata)
    assert (parsed.hops, parsed.steps, parsed.rank, parsed.alpha) == (4, 32, 2, 2.)
    assert len(parsed.pairs) == 6 * (parsed.blocks + parsed.refiners)
    assert set(parsed.heads) == {"final_layer.video_out", "final_layer.audio_out"}
    for key, original in before.items():
        assert torch.equal(tensors[key], original)
    first = parsed.pairs[0]
    first.a.zero_()
    assert all(torch.equal(tensors[key], original) for key, original in before.items())


@pytest.mark.parametrize("damage", ["missing", "orphan-scale", "no-schema", "schema", "rank", "hop", "nan", "head", "extra", "collision", "noncontiguous"])
def test_real_storage_and_topology_errors_are_not_guessed_or_name_banned(damage):
    tensors, metadata = stacked(model(), quantized=True)
    key = next(k for k in tensors if k.endswith(".lora_down"))
    if damage == "missing":
        tensors.pop(key)
    elif damage == "orphan-scale":
        tensors[key + "_scale"] = torch.ones(1)
    elif damage == "no-schema":
        metadata.pop("ptd_compact")
    elif damage == "schema":
        metadata["ptd_compact"] = '{"schema":true,"quant":"int8 per-row"}'
    elif damage == "rank":
        metadata["lora_rank"] = "0"
    elif damage == "hop":
        tensors[key] = tensors[key][:3]
    elif damage == "nan":
        tensors[key + "_scale"].flatten()[0] = float("nan")
    elif damage == "head":
        tensors["proj_out.weight"] = tensors["proj_out.weight"][:31]
    elif damage == "extra":
        tensors["ordinary_lora.weight"] = torch.ones(2, 2)
    elif damage == "collision":
        tensors[key.replace(".0.", ".0._checkpoint_wrapped_module.")] = tensors[key]
    else:
        tensors = {k.replace("transformer_blocks.0", "transformer_blocks.2"): v for k,v in tensors.items()}
    with pytest.raises(ValueError):
        files.parse_compact_checkpoint(tensors, metadata)


@pytest.mark.parametrize("alpha", [0., -4., 6.])
def test_actual_expert_alpha_once_and_swiglu_half_order(alpha):
    base = model()
    tensors, metadata = stacked(base, alpha=alpha)
    selected, _ = compact.configure_compact(base, files.parse_compact_checkpoint(tensors, metadata), lora_strength=-3.)
    controller = selected.model_options[compact.KEY]
    module = base.model.diffusion_model.blocks[0].attn.qkv_proj
    value = torch.randn(5, module.in_features)
    expected, width = module(value), module.out_features // 3
    for index, part in enumerate(("to_q", "to_k", "to_v")):
        source = "transformer_blocks.0.attn." + part
        delta = F.linear(F.linear(value, tensors[source + ".lora_down"][2]), tensors[source + ".lora_up"][2]) * (alpha/2 * -3.)
        expected[:, index*width:(index+1)*width] += delta
    with controller.activate(2, controller.grid[2]), controller.forward_scope():
        actual = module(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    raw_b = tensors["transformer_blocks.0.ff.net.0.proj.lora_up"][0]
    assert torch.equal(controller.bank["blocks.0.mlp.fc1"][0][0].b, torch.cat(raw_b.chunk(2)[::-1]))


def test_absolute_audio_head_chord_precedes_existing_physical_derivative():
    base = model()
    tensors, metadata = stacked(base)
    selected, _ = compact.configure_compact(base, files.parse_compact_checkpoint(tensors, metadata))
    controller = selected.model_options[compact.KEY]
    module = base.model.diffusion_model.final_layer.audio_out
    value = torch.randn(7, module.in_features)
    weight, bias = controller.heads["final_layer.audio_out"]
    factor = controller.chord_corrections[0] * old_clock.audio_sigma_prime(controller.grid[0])
    expected = F.linear(value, weight[0], None if bias is None else bias[0]) * controller.chord_corrections[0]
    expected *= old_clock.audio_sigma_prime(controller.grid[0])
    with controller.activate(0, controller.grid[0]), controller.forward_scope():
        actual = module(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert factor != 4.  # no accidental tangent-only/double correction.


@pytest.mark.parametrize("hops,quantized", [(4, False), (8, True)])
def test_actual_Core_hop_head_refiner_consumption_and_next_plain_restore(hops, quantized):
    from test_h21_projection_model import sample as plain
    base = model()
    before = plain(base)
    methods = {name: vars(module).get("forward") for name, module in base.model.diffusion_model.named_modules()}
    tensors, metadata = stacked(base, hops=hops, quantized=quantized)
    selected, setup = compact.configure_compact(base, files.parse_compact_checkpoint(tensors, metadata))
    controller = selected.model_options[compact.KEY]
    assert setup["status"] == "selected_not_executed" and not setup["runtime_int8"]
    output = sample(selected, controller)
    report = controller.report()
    assert report["last_request"]["hops"] == list(range(hops))
    assert len(report["last_request"]["head_calls"]) == 2*hops
    assert len(report["last_request"]["expert_calls"]) == len(controller.bank)*hops
    assert report["status"] == "observed_complete" and not report["partial"]
    assert all(torch.isfinite(part).all() for part in output["samples"].unbind())
    assert controller.request.get() is None and controller.selection.get() is None
    for name, module in base.model.diffusion_model.named_modules():
        assert vars(module).get("forward") is methods[name]
        assert not module._forward_hooks and not module._forward_pre_hooks
    after = plain(base)
    for actual, expected in zip(after["samples"].unbind(), before["samples"].unbind()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_bad_grid_failure_and_private_head_tamper_are_real_errors():
    base = model()
    tensors, metadata = stacked(base)
    selected, _ = compact.configure_compact(base, files.parse_compact_checkpoint(tensors, metadata))
    controller = selected.model_options[compact.KEY]
    with pytest.raises(ValueError, match="checkpoint-derived"):
        compact.sample_compact(lambda *_: None, torch.zeros(1, 2), old_clock.ptd_sigmas("h3_4step"), controller=controller)
    controller.heads["final_layer.video_out"][0].flatten()[0] += 1
    with pytest.raises(ValueError, match="contents changed"):
        controller.begin_request()
    assert controller.request.get() is None


def test_unknown_head_hook_preserved_and_reported_partial_not_rejected():
    base = model()
    tensors, metadata = stacked(base)
    selected, _ = compact.configure_compact(base, files.parse_compact_checkpoint(tensors, metadata))
    controller = selected.model_options[compact.KEY]
    seen = []
    module = base.model.diffusion_model.final_layer.video_out
    handle = module.register_forward_hook(lambda *_: seen.append(1))
    try:
        sample(selected, controller)
        assert seen == [1]*4
        assert handle.id in module._forward_hooks
        report = controller.report()
        assert report["partial"] and any("unknown_head" in item for item in report["last_request"]["delegated"])
    finally:
        handle.remove()


def test_actual_raw_refiner_fc2_consumed_every_hop_not_precomputed():
    base = model()
    root = base.model.diffusion_model
    root.token_refiner = native.TokenRefiner(1, 24, 3, 128, 32, 1e-6, 1e-6, 1e-6,
        dtype=torch.float32, device="cpu", operations=__import__("comfy.ops", fromlist=["disable_weight_init"]).disable_weight_init)
    generator = torch.Generator().manual_seed(241403)
    for parameter in root.token_refiner.parameters():
        parameter.copy_(torch.randn(parameter.shape, generator=generator)*.03)
    seen = []
    handle = root.token_refiner.register_forward_hook(lambda *_: seen.append(1))
    tensors, metadata = stacked(base, hops=8, quantized=True)
    selected, _ = compact.configure_compact(base, files.parse_compact_checkpoint(tensors, metadata))
    controller = selected.model_options[compact.KEY]
    try:
        sample(selected, controller)
    finally:
        handle.remove()
    assert seen == [1]*8
    report = controller.report()
    assert all(f"{hop}:token_refiner.blocks.0.mlp.fc2" in report["last_request"]["expert_calls"] for hop in range(8))
    assert not report["last_request"]["delegated"]


def test_actual_error_restores_ordinary_request_and_keeps_user_wrapper_and_lora_order():
    from test_h21_projection_model import sample as plain
    base = model()
    key = "diffusion_model.blocks.0.attn.out_proj.weight"
    weight = base.model_state_dict()[key]
    for strength in (-.25, .6):
        base.add_patches({key: ("diff", (torch.ones_like(weight)*.001,))}, strength_patch=strength)
    patches = deepcopy(base.patches)
    before = plain(base)
    calls = []

    def previous(function, payload):
        calls.append(1)
        return function(payload["input"], payload["timestep"], **payload["c"])

    base.set_model_unet_function_wrapper(previous)
    tensors, metadata = stacked(base)
    selected, _ = compact.configure_compact(base, files.parse_compact_checkpoint(tensors, metadata))
    controller = selected.model_options[compact.KEY]

    def fail(*_args):
        raise TypeError("real compact projection failure")

    module = base.model.diffusion_model.blocks[0].mlp.fc1
    handle = module.register_forward_hook(fail)
    try:
        with pytest.raises(TypeError, match="real compact projection failure"):
            sample(selected, controller)
    finally:
        handle.remove()
    assert calls == [1]
    assert controller.report()["status"] == "failed"
    assert controller.request.get() is None and controller.selection.get() is None
    assert selected.model_options["model_function_wrapper"].previous is previous
    assert list(selected.patches) == list(patches)
    assert [row[0] for row in selected.patches[key]] == [-.25, .6]
    for module in base.model.diffusion_model.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks and "forward" not in vars(module)
    after = plain(base)
    for actual, expected in zip(after["samples"].unbind(), before["samples"].unbind()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_safe_file_loading_old_EPFL_unchanged_and_no_pickle(tmp_path, monkeypatch):
    base = model()
    tensors, metadata = stacked(base, quantized=True)
    path = tmp_path / "one.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: pytest.fail("pickle called"))
    assert files.is_compact_file(path)
    selected, sampler, sigmas, report = compact.load_compact_setup(base, path)
    assert sampler.extra_options["controller"] is selected.model_options[compact.KEY]
    assert report["file"]["sha256"] and not report["file"]["pickle_executed"]
    assert torch.equal(sigmas, compact.compact_grid(4, 32))
    assert tuple(old_clock.ptd_sigmas("h3_6step").tolist()) == (.999,.968,.923,.857,.750,.545,0.)
    with pytest.raises(ValueError, match="DCP/pickle"):
        files.load_compact_file(tmp_path / "untrusted.pt")


@pytest.mark.parametrize("hops,quantized", [(4, False), (8, True)])
def test_existing_setup_sampler_audit_dispatch_by_actual_metadata_not_filename(tmp_path, hops, quantized):
    base = model()
    tensors, metadata = stacked(base, hops=hops, quantized=quantized)
    # Deliberately name the stacked file like the legacy EPFL recipe.
    path = tmp_path / "ptd_h3_6step.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    selected, setup_sampler, setup_grid, setup_json = nodes.MiniMaxH3PTDSetupEXPT8.execute(base, str(path)).result
    controller = ptd.get_controller(selected)
    sampler, grid = nodes.MiniMaxH3PTDSamplerEXPT8.execute(selected).result
    assert type(controller) is compact.CompactPTDController
    assert setup_sampler.sampler_function is sampler.sampler_function is compact.sample_compact
    assert torch.equal(grid, setup_grid) and torch.equal(grid, compact.compact_grid(hops, 32))
    assert json.loads(setup_json)["status"] == "selected_not_executed"
    output = SamplerCustomAdvanced.execute(
        RandomNoise.execute(241402).result[0], BasicGuider.execute(selected, conditioning()).result[0],
        sampler, grid, latent()).result[0]
    audit = json.loads(nodes.MiniMaxH3PTDAuditEXPT8.execute(selected, output).result[0])
    assert audit["status"] == "observed_complete" and audit["last_request"]["hops"] == list(range(hops))
    assert not audit["continuous_stage_protocol_implemented"]
    with pytest.raises(ValueError, match="recipe override"):
        ptd.load_ptd_setup(base, path, recipe="h3_4step")


@pytest.mark.parametrize("recipe", ["h3_4step", "h3_6step"])
def test_legacy_file_dispatch_keeps_EPFL_sampler_grid_and_controller(tmp_path, recipe):
    from test_h21_ptd_advanced import experts

    base = model()
    path = tmp_path / "compact_ref2va_8step.safetensors"
    save_file(experts(base, recipe), str(path), metadata=old_files.metadata_for(recipe))
    selected, sampler, grid, _ = ptd.load_ptd_setup(base, path)
    controller = ptd.get_controller(selected)
    assert type(controller) is ptd.PTDController
    assert sampler.sampler_function is old_clock.sample_ptd
    assert torch.equal(grid, old_clock.ptd_sigmas(recipe))
    node_sampler, node_grid = nodes.MiniMaxH3PTDSamplerEXPT8.execute(selected).result
    assert node_sampler.sampler_function is old_clock.sample_ptd
    assert torch.equal(node_grid, grid)


def test_compact_continuous_stage_reports_missing_protocol_before_legacy_grid(monkeypatch):
    base = model()
    tensors, metadata = stacked(base)
    selected, _ = compact.configure_compact(base, files.parse_compact_checkpoint(tensors, metadata))
    monkeypatch.setattr(old_clock, "ptd_sigmas", lambda *_: pytest.fail("legacy grid used for Compact"))
    with pytest.raises(ValueError, match="absolute-head bank identity.*not implemented"):
        continuous.run_head(RandomNoise.execute(241402).result[0],
                            BasicGuider.execute(selected, conditioning()).result[0], latent())
    assert ptd.get_controller(selected).request.get() is None


@pytest.mark.parametrize("order", ["before", "after"])
def test_head_weight_LoRA_before_or_after_setup_is_consumed_and_reported_partial(order):
    outputs = []
    path = "final_layer.video_out"
    key = "diffusion_model." + path + ".weight"
    for strength in (0., .8):
        base = model()
        tensors, metadata = stacked(base)
        patch = {key: ("diff", (torch.ones_like(base.model_state_dict()[key]) * .02,))}
        if order == "before":
            base.add_patches(patch, strength_patch=strength)
        selected, _ = compact.configure_compact(base, files.parse_compact_checkpoint(tensors, metadata))
        if order == "after":
            selected = selected.clone()
            selected.add_patches(patch, strength_patch=strength)
        controller = ptd.get_controller(selected)
        outputs.append(sample(selected, controller)["samples"].unbind()[0].clone())
        report = controller.report()
        assert report["status"] == "observed_partial"
        assert report["last_request"]["retained_head_paths"] == (path,)
        assert [row[0] for row in selected.patches[key]] == [strength]
        assert all(f"{hop}:{path}" not in report["last_request"].get("head_calls", {}) for hop in range(4))
        assert len(report["last_request"]["head_calls"]) == 4  # audio bank still runs
    assert not torch.equal(outputs[0], outputs[1])


def test_unknown_head_class_callable_executes_and_is_not_overwritten():
    base = model()
    head = base.model.diffusion_model.final_layer.video_out
    inherited = type(head)
    calls = []

    class UserProjection(inherited):
        def forward(self, value):
            calls.append(1)
            return super().forward(value) + .5

    head.__class__ = UserProjection
    tensors, metadata = stacked(base)
    selected, _ = compact.configure_compact(base, files.parse_compact_checkpoint(tensors, metadata))
    controller = ptd.get_controller(selected)
    value = torch.randn(3, head.in_features)
    expected = head(value)
    controller.begin_request()
    with controller.activate(0, controller.grid[0]), controller.forward_scope():
        actual = head(value)
    controller.end_request(False)
    assert calls == [1, 1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert any("unknown_head" in item for item in controller.report()["last_request"]["delegated"])


def test_head_hooks_are_installed_only_after_shared_lease_and_removed_after_failure(monkeypatch):
    base = model()
    tensors, metadata = stacked(base)
    selected, _ = compact.configure_compact(base, files.parse_compact_checkpoint(tensors, metadata))
    controller = ptd.get_controller(selected)
    seen = []
    original = compact.shared_network_lease

    @contextmanager
    def lease(network):
        seen.append([len(network.get_submodule(path)._forward_hooks) for path in controller.heads])
        with original(network) as acquired:
            yield acquired

    monkeypatch.setattr(compact, "shared_network_lease", lease)
    with pytest.raises(RuntimeError, match="head scope interruption"):
        with controller.activate(0, controller.grid[0]), controller.forward_scope():
            raise RuntimeError("head scope interruption")
    assert seen == [[0, 0]]
    assert all(not base.model.diffusion_model.get_submodule(path)._forward_hooks for path in controller.heads)
