"""Real tiny Core PTD plumbing, synthetic experts: not trained AV qualification."""

from contextlib import contextmanager
from copy import deepcopy
import json
import threading

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from comfy_extras.nodes_custom_sampler import (
    BasicGuider,
    RandomNoise,
    SamplerCustomAdvanced,
)

from h3_audio_t8_pkg import ptd_checkpoint as files
from h3_audio_t8_pkg import ptd_sampling as clocks
from h3_audio_t8_pkg import ptd_advanced as ptd
from h3_audio_t8_pkg import nodes_ptd_advanced as nodes
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent


@pytest.fixture(autouse=True)
def actual_cpu(monkeypatch):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    with torch.inference_mode():
        yield
    assert not torch.cuda.is_initialized()


def experts(base, recipe="h3_4step"):
    root = base.model.diffusion_model
    result = {}
    generator = torch.Generator().manual_seed(212104)
    for source in sorted(
        files.expected_sources(len(root.blocks), len(root.token_refiner.blocks))
    ):
        for hop in range(len(files.recipe_grid(recipe)) - 1):
            a_key, b_key = (
                f"{source}.lora_{factor}.{hop}.weight" for factor in ("A", "B")
            )
            path, _, _, qkv, _, dense, _ = files.map_factor(a_key)
            module = root.get_submodule(path)
            out = module.out_features // (3 if qkv is not None else 1)
            rank = out if dense else 2
            result[a_key] = (
                torch.randn((rank, module.in_features), generator=generator) * 0.01
            )
            result[b_key] = (
                torch.randn((out, rank), generator=generator) * 0.01 * (hop + 1)
            )
    return result


def sample(selected, controller, seed=212104):
    guider = BasicGuider.execute(selected, conditioning()).result[0]
    return SamplerCustomAdvanced.execute(
        RandomNoise.execute(seed).result[0],
        guider,
        clocks.make_sampler(controller),
        clocks.ptd_sigmas(controller.recipe),
        latent(),
    ).result[0]


@pytest.mark.parametrize(
    "recipe,grid",
    [
        ("h3_4step", (0.999, 0.947, 0.857, 0.667, 0.0)),
        ("h3_6step", (0.999, 0.968, 0.923, 0.857, 0.750, 0.545, 0.0)),
    ],
)
def test_published_grid_and_distinct_physical_audio_clock(recipe, grid):
    assert tuple(clocks.ptd_sigmas(recipe).tolist()) == grid
    for time in grid[:-1]:
        audio = clocks.audio_sigma(time)
        shift = clocks.native_audio_shift(time)
        native_sigma = shift * time / (1 + (shift - 1) * time)
        assert native_sigma == pytest.approx(audio, abs=2e-15)
        if time != 0.999:
            eps = 1e-7
            derivative = (
                clocks.audio_sigma(time + eps) - clocks.audio_sigma(time - eps)
            ) / (2 * eps)
            assert derivative == pytest.approx(clocks.audio_sigma_prime(time), rel=1e-8)
    assert clocks.audio_sigma(0.999) == 0.999 and clocks.audio_sigma_prime(
        0.999
    ) == pytest.approx(4.0)
    assert clocks.audio_sigma(0.857) != pytest.approx(
        0.25 * 0.857 / (1 - 0.75 * 0.857), abs=1e-6
    )


@pytest.mark.parametrize("recipe", files.GRIDS)
def test_exact_FP32_recurrence_uses_one_expert_per_hop_and_raw_noise_once(recipe):
    seen = []

    class Controller:
        def __init__(self):
            self.recipe = recipe

        def begin_request(self):
            seen.append("begin")

        @contextmanager
        def activate(self, hop, time):
            self.hop = hop
            seen.append((hop, time))
            yield

        def completed_hop(self, hop):
            seen.append(("done", hop))

        def end_request(self, success):
            seen.append(success)

    controller = Controller()
    noise = torch.tensor([[1.1234567, -0.5533421]], dtype=torch.float32)

    class Denoiser:
        def __init__(self):
            self.noise = noise

        def __call__(self, state, time, **kwargs):
            assert state.dtype is torch.float32 and time.dtype is torch.float64
            return state - time.float().reshape(-1, 1) * (controller.hop + 1)

    callbacks = []
    actual = clocks.sample_ptd(
        Denoiser(),
        noise * 0.123,
        clocks.ptd_sigmas(recipe),
        controller=controller,
        callback=callbacks.append,
    )
    expected = (noise.double() * 0.999).float()
    for hop, (start, end) in enumerate(
        zip(files.GRIDS[recipe][:-1], files.GRIDS[recipe][1:])
    ):
        # Reference retains denoised float32 rounding, not an ideal-real shortcut.
        predicted = expected - torch.tensor(start).float() * (hop + 1)
        velocity = (expected - predicted) / start
        expected = expected + torch.tensor(end - start, dtype=torch.float32) * velocity
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert [item["i"] for item in callbacks] == list(
        range(len(files.GRIDS[recipe]) - 1)
    )
    assert seen[0] == "begin" and seen[-1] is True


@pytest.mark.parametrize("recipe", files.GRIDS)
def test_actual_tiny_Core_complete_sampler_and_next_plain_request(recipe):
    from test_h21_projection_model import sample as plain

    base = model()
    expected = plain(base)
    original = {
        path: vars(module).get("forward")
        for path, module in base.model.diffusion_model.named_modules()
    }
    selected, setup = ptd.configure_ptd(
        base, experts(base, recipe), files.metadata_for(recipe)
    )
    controller = ptd.get_controller(selected)
    assert setup["status"] == "selected_not_executed"
    assert (
        base.model is selected.model
        and not base.object_patches
        and ptd.KEY not in base.model_options
    )
    output = sample(selected, controller)
    assert all(
        torch.isfinite(value).all() and value.dtype is torch.float32
        for value in output["samples"].unbind()
    )
    report = controller.report()
    count = len(files.GRIDS[recipe]) - 1
    assert report["last_request"]["hops"] == list(range(count))
    assert [item["hop"] for item in report["last_request"]["forwards"]] == list(
        range(count)
    )
    assert len(report["last_request"]["expert_calls"]) == len(controller.bank) * count
    assert not report["last_request"]["delegated"]
    assert not report["portable_cache_identity"] and not report["trained_GPU_qualified"]
    assert controller.selection.get() is None and controller.request.get() is None
    for path, module in base.model.diffusion_model.named_modules():
        assert vars(module).get("forward") is original[path]
        assert not module._forward_hooks and not module._forward_pre_hooks
    after = plain(base)
    for actual, reference in zip(
        after["samples"].unbind(), expected["samples"].unbind()
    ):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_qkv_independent_factors_scale_one_and_SwiGLU_half_order():
    base = model()
    tensors = experts(base)
    selected, _ = ptd.configure_ptd(base, tensors, files.metadata_for("h3_4step"))
    controller = ptd.get_controller(selected)
    root = base.model.diffusion_model
    qkv = root.blocks[0].attn.qkv_proj
    x = torch.randn((5, qkv.in_features))
    expected = qkv(x)
    width = qkv.out_features // 3
    for index, part in enumerate(("to_q", "to_k", "to_v")):
        prefix = f"transformer.transformer_blocks.0.attn.{part}"
        delta = F.linear(
            F.linear(x, tensors[f"{prefix}.lora_A.2.weight"]),
            tensors[f"{prefix}.lora_B.2.weight"],
        )
        expected[..., index * width : (index + 1) * width] += delta
    with controller.activate(2, 0.857), controller.forward_scope():
        actual = qkv(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expert = controller.bank["blocks.0.mlp.fc1"][0][0]
    b = tensors["transformer.transformer_blocks.0.ff.net.0.proj.lora_B.0.weight"]
    torch.testing.assert_close(
        expert.b,
        torch.cat((b[b.shape[0] // 2 :], b[: b.shape[0] // 2])),
        rtol=0,
        atol=0,
    )
    # Controller owns private storage, not mutable aliases into the caller dict.
    tensors["transformer.transformer_blocks.0.ff.net.0.proj.lora_B.0.weight"].zero_()
    assert bool(torch.count_nonzero(expert.b))


def test_safe_loader_actual_safetensors_metadata_and_raw_pickle_never_called(
    tmp_path, monkeypatch
):
    base = model()
    path = tmp_path / "experts.safetensors"
    save_file(experts(base), str(path), metadata=files.metadata_for("h3_4step"))
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("unsafe torch.load reached")
    )
    selected, sampler, sigmas, report = ptd.load_ptd_setup(base, path)
    assert report["file"]["sha256"] and not report["file"]["pickle_executed"]
    assert sampler.extra_options["controller"] is ptd.get_controller(selected)
    assert torch.equal(sigmas, clocks.ptd_sigmas("h3_4step"))
    with pytest.raises(ValueError, match="DCP/pickle"):
        files.load_ptd_file(tmp_path / "untrusted.pt")
    path.write_bytes(b"bad safetensors bytes")
    with pytest.raises(Exception, match="header|Header|deserialize"):
        files.load_ptd_file(path)


@pytest.mark.parametrize(
    "damage", ["hop", "missing", "rank", "nan", "unknown", "heads"]
)
def test_real_bad_checkpoint_dimensions_topology_and_values_fail(damage):
    base = model()
    tensors = experts(base)
    key = next(iter(tensors))
    if damage == "hop":
        tensors[key.replace(".0.weight", ".9.weight")] = tensors.pop(key)
    elif damage == "missing":
        tensors.pop(key)
    elif damage == "rank":
        tensors[key] = tensors[key][:, :-1]
    elif damage == "nan":
        tensors[key].view(-1)[0] = float("nan")
    elif damage == "unknown":
        tensors["ordinary_lora.lora_A.weight"] = torch.ones(2, 3)
    else:
        tensors["transformer.proj_out.orig_weight"] = torch.ones(2, 3)
    with pytest.raises(ValueError):
        ptd.configure_ptd(base, tensors, files.metadata_for("h3_4step"))
    assert not base.object_patches and not base.model_options.get(ptd.KEY)


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([0.999, 0.947, 0.0]),
        torch.tensor([0.999, 0.947, 0.857, 0.667, float("nan")]),
        torch.tensor([1.0, 0.947, 0.857, 0.667, 0.0]),
        torch.tensor([0.999, 0.947, 0.857, 0.667, 0.0], dtype=torch.float16),
    ],
)
def test_bad_grid_not_silently_recomputed_or_split(bad):
    with pytest.raises(ValueError, match="PTD requires"):
        clocks.validate_grid(bad, "h3_4step")


def test_unknown_model_class_wrappers_and_attention_are_not_name_banned():
    base = model()
    root = base.model.diffusion_model
    original = root.blocks[0].attn.forward
    seen = []

    def foreign(*args, **kwargs):
        seen.append("attention")
        return original(*args, **kwargs)

    root.blocks[0].attn.forward = foreign
    wrapper_calls = []

    def previous(function, payload):
        wrapper_calls.append(1)
        return function(payload["input"], payload["timestep"], **payload["c"])

    base.set_model_unet_function_wrapper(previous)
    selected, _ = ptd.configure_ptd(base, experts(base), files.metadata_for("h3_4step"))
    sample(selected, ptd.get_controller(selected))
    assert seen == ["attention"] * 4 and wrapper_calls == [1] * 4
    report = ptd.get_controller(selected).report()
    assert (
        report["partial"]
        and "unknown_Attention.forward_original_forward_retained"
        in report["last_request"]["delegated"]
    )
    assert root.blocks[0].attn.forward is foreign


def test_failure_removes_actual_hooks_and_next_request_uses_original_base():
    from test_h21_projection_model import sample as plain

    base = model()
    before = plain(base)
    selected, _ = ptd.configure_ptd(base, experts(base), files.metadata_for("h3_4step"))
    controller = ptd.get_controller(selected)
    seen = []

    def fail(_module, _args, _output):
        seen.append(1)
        raise TypeError("actual selected projection error")

    handle = base.model.diffusion_model.blocks[0].mlp.fc1.register_forward_hook(fail)
    try:
        with pytest.raises(TypeError, match="actual selected projection error"):
            sample(selected, controller)
    finally:
        handle.remove()
    assert seen == [1] and controller.report()["status"] == "failed"
    assert controller.selection.get() is None and controller.request.get() is None
    assert all(
        not module._forward_hooks
        and not module._forward_pre_hooks
        and "forward" not in vars(module)
        for module in base.model.diffusion_model.modules()
    )
    after = plain(base)
    for a, b in zip(after["samples"].unbind(), before["samples"].unbind()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    sample(selected, controller)
    assert controller.report()["last_request"]["hops"] == [0, 1, 2, 3]


def test_actual_ordered_LoRA_and_user_metadata_choices_remain():
    base = model()
    key = "diffusion_model.blocks.0.attn.out_proj.weight"
    weight = base.model_state_dict()[key]
    base.add_patches(
        {key: ("diff", (torch.ones_like(weight) * 0.001,))}, strength_patch=-0.25
    )
    base.add_patches(
        {key: ("diff", (torch.ones_like(weight) * 0.002,))}, strength_patch=0.6
    )
    original = deepcopy(base.patches)
    metadata = files.metadata_for("h3_4step")
    metadata["ptd_base_model"] = "user-selected-other-name"
    selected, _ = ptd.configure_ptd(base, experts(base), metadata)
    assert [value[0] for value in selected.patches[key]] == [-0.25, 0.6]
    assert list(selected.patches) == list(original)
    sample(selected, ptd.get_controller(selected))
    assert ptd.get_controller(selected).report()["warnings"]
    assert [value[0] for value in base.patches[key]] == [-0.25, 0.6]


def test_explicit_node_schemas_and_audit_order_do_not_manufacture_execution():
    assert len(nodes.NODES) == 3
    assert all(node.define_schema().is_experimental for node in nodes.NODES)
    selected, _ = ptd.configure_ptd(
        model(), experts(model()), files.metadata_for("h3_4step")
    )
    report = json.loads(
        nodes.MiniMaxH3PTDAuditEXPT8.execute(selected, object()).result[0]
    )
    assert (
        report["status"] == "selected_not_executed" and report["last_request"] is None
    )
    with pytest.raises(ValueError, match="within models/ptd"):
        nodes.checkpoint_path("../outside.safetensors")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.0])
def test_invalid_active_time_never_selects_an_expert(invalid):
    base = model()
    controller = ptd.PTDController(
        base.model.diffusion_model, experts(base), files.metadata_for("h3_4step")
    )
    with pytest.raises(ValueError, match="trained grid time"):
        with controller.activate(0, invalid):
            pytest.fail("invalid clock selected a PTD expert")
    assert controller.selection.get() is None


def test_unknown_preprocess_callable_is_retained_not_erased_for_optional_PTD():
    base = model()
    network = base.model.diffusion_model
    original = network.preprocess_text_embeds
    seen = []

    def foreign(value):
        seen.append(1)
        return original(value)

    network.preprocess_text_embeds = foreign
    selected, _ = ptd.configure_ptd(base, experts(base), files.metadata_for("h3_4step"))
    sample(selected, ptd.get_controller(selected))
    assert seen == [1]
    assert network.preprocess_text_embeds is foreign
    report = ptd.get_controller(selected).report()
    assert report["partial"] and any(
        "preprocess" in message for message in report["warnings"]
    )


def test_actual_token_refiner_is_recomputed_with_each_hop_not_cached_before_sampling():
    base = model()
    root = base.model.diffusion_model
    root.token_refiner = native.TokenRefiner(
        1,
        24,
        3,
        128,
        32,
        1e-6,
        1e-6,
        1e-6,
        dtype=torch.float32,
        device="cpu",
        operations=__import__(
            "comfy.ops", fromlist=["disable_weight_init"]
        ).disable_weight_init,
    )
    generator = torch.Generator().manual_seed(212105)
    for parameter in root.token_refiner.parameters():
        parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.03)
    seen = []
    handle = root.token_refiner.register_forward_hook(lambda *_: seen.append(1))
    selected, _ = ptd.configure_ptd(base, experts(base), files.metadata_for("h3_4step"))
    try:
        output = sample(selected, ptd.get_controller(selected))
    finally:
        handle.remove()
    assert seen == [1] * 4
    report = ptd.get_controller(selected).report()
    assert all(
        f"{hop}:token_refiner.blocks.0.mlp.fc2"
        in report["last_request"]["expert_calls"]
        for hop in range(4)
    )
    assert not report["last_request"]["delegated"]
    assert all(torch.isfinite(value).all() for value in output["samples"].unbind())


def test_actual_BF16_Core_state_stays_FP32_and_heads_remain_FP32():
    base = model()
    root = base.model.diffusion_model
    for name, module in root.named_modules():
        if isinstance(module, torch.nn.Linear) and not name.startswith(
            (
                "time_embedder.",
                "video_patch_proj",
                "audio_patch_proj",
                "final_layer.video_out",
                "final_layer.audio_out",
            )
        ):
            module.to(dtype=torch.bfloat16)
    for name, module in root.named_modules():
        if isinstance(module, torch.nn.RMSNorm):
            module.to(dtype=torch.bfloat16)
    # BaseModel asks its model configuration for inference dtype.
    base.model.manual_cast_dtype = torch.bfloat16
    selected, _ = ptd.configure_ptd(base, experts(base), files.metadata_for("h3_4step"))
    output = sample(selected, ptd.get_controller(selected))
    assert root.final_layer.audio_out.weight.dtype is torch.float32
    assert root.final_layer.video_out.weight.dtype is torch.float32
    assert all(
        value.dtype is torch.float32 and torch.isfinite(value).all()
        for value in output["samples"].unbind()
    )
    assert ptd.get_controller(selected).report()["last_request"]["hops"] == [0, 1, 2, 3]


def test_actual_audio_expert_is_added_before_physical_clock_derivative():
    base = model()
    tensors = experts(base)
    selected, _ = ptd.configure_ptd(base, tensors, files.metadata_for("h3_4step"))
    controller = ptd.get_controller(selected)
    module = base.model.diffusion_model.final_layer.audio_out
    x = torch.randn((7, module.in_features))
    prefix = "transformer.audio_proj_out.main"
    original = module(x)
    delta = F.linear(
        F.linear(x, tensors[f"{prefix}.lora_A.0.weight"]),
        tensors[f"{prefix}.lora_B.0.weight"],
    )
    expected = (original + delta) * torch.tensor(
        clocks.audio_sigma_prime(0.999), dtype=original.dtype
    )
    with controller.activate(0, 0.999), controller.forward_scope():
        actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, original * 4 + delta)


def test_real_plain_other_thread_and_nested_hop_do_not_inherit_experts():
    base = model()
    root = base.model.diffusion_model
    tensors = experts(base)
    outer = ptd.PTDController(root, tensors, files.metadata_for("h3_4step"))
    inner = ptd.PTDController(root, tensors, files.metadata_for("h3_4step"))
    module = root.blocks[0].attn.qkv_proj
    x = torch.randn((5, module.in_features))
    expected = module(x)
    answers, errors = [], []

    def plain_worker():
        try:
            with torch.inference_mode():
                answers.append(module(x))
        except BaseException as error:
            errors.append(error)

    def adapted(hop):
        value = expected.clone()
        width = module.out_features // 3
        for index, part in enumerate(("to_q", "to_k", "to_v")):
            prefix = f"transformer.transformer_blocks.0.attn.{part}"
            value[..., index * width : (index + 1) * width] += F.linear(
                F.linear(x, tensors[f"{prefix}.lora_A.{hop}.weight"]),
                tensors[f"{prefix}.lora_B.{hop}.weight"],
            )
        return value

    with outer.activate(0, 0.999), outer.forward_scope():
        a = module(x)
        thread = threading.Thread(target=plain_worker)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        with inner.activate(2, 0.857), inner.forward_scope():
            b = module(x)
        c = module(x)
    assert not errors and len(answers) == 1
    torch.testing.assert_close(answers[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(a, adapted(0), rtol=0, atol=0)
    torch.testing.assert_close(b, adapted(2), rtol=0, atol=0)
    torch.testing.assert_close(c, a, rtol=0, atol=0)
    assert not module._forward_hooks


def test_integrator_without_selected_MODEL_execution_cannot_claim_expert_completion():
    base = model()
    controller = ptd.PTDController(
        base.model.diffusion_model, experts(base), files.metadata_for("h3_4step")
    )
    seen = []

    def foreign(state, sigma, **kwargs):
        seen.append(1)
        return state

    actual = clocks.sample_ptd(
        foreign,
        torch.ones(1, 5),
        clocks.ptd_sigmas(controller.recipe),
        controller=controller,
    )
    assert seen == [1] * 4 and torch.equal(actual, torch.ones(1, 5))
    report = controller.report()
    assert report["status"] == "observed_partial" and report["partial"]
    assert (
        not report["last_request"]["forwards"]
        and not report["last_request"]["expert_calls"]
    )
    assert len(report["last_request"]["delegated"]) == len(controller.bank) * 4
