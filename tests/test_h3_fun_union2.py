"""Real native Core on small CPU weights; not qualification of the 13GB model."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import comfy.ops
import comfy.patcher_extension
from comfy.ldm.minimax.controlnet import MiniMaxH3FunControl
from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlBlockPatch

from h3_audio_t8_pkg import h3_fun_union2 as union
from h3_audio_t8_pkg.h3_fun_control_advanced import H3FunControlBundle
from h3_audio_t8_pkg.h3_fun_union2_runtime import AlignedUnion2Patch
from h3_audio_t8_pkg.nodes_h3_fun_union2 import (
    MiniMaxH3FunUnion2ApplyEXPT8,
    MiniMaxH3FunUnion2LoaderEXPT8,
)


@pytest.fixture
def control():
    result = MiniMaxH3FunControl(
        control_in_dim=49,
        injection_layers=union.LAYERS,
        inpaint_post_norm=True,
        hidden_size=4,
        num_attention_heads=1,
        attention_head_dim=4,
        ffn_hidden_size=8,
        time_embed_dim=2688,
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    with torch.no_grad():
        for index, parameter in enumerate(result.parameters()):
            parameter.fill_(0.001 * (index + 1))
    return result


def raw_names(state):
    result = {}
    for key, value in state.items():
        if key.endswith(".attn.qkv_proj.weight"):
            prefix = key[: -len("qkv_proj.weight")]
            for suffix, part in zip(("q", "k", "v"), value.chunk(3), strict=True):
                result[prefix + f"to_{suffix}.weight"] = part.clone()
        elif key.endswith(".mlp.fc1.weight"):
            first, second = value.chunk(2)
            result[key.replace(".mlp.fc1.", ".ff.net.0.proj.")] = torch.cat(
                [second, first]
            )
        else:
            name = key.replace(".attn.q_norm.", ".attn.norm_q.").replace(
                ".attn.k_norm.", ".attn.norm_k."
            )
            name = name.replace(".attn.out_proj.", ".attn.to_out.0.").replace(
                ".mlp.fc2.", ".ff.net.2."
            )
            result[name] = value.clone()
    return result


@pytest.mark.parametrize("raw", [False, True])
def test_full_native_small_state_conversion_and_strict_load(control, raw):
    original = control.state_dict()
    selected = raw_names(original) if raw else original
    converted, structure = union.union_structure(selected, {"format": "pt"})
    assert structure["block_count"] == 10 and structure["control_in_dim"] == 49
    assert (
        structure["time_embed_dim"] == 2688 and structure["raw_names_converted"] is raw
    )
    assert structure["injection_layers"] == tuple(range(0, 50, 5))
    assert set(converted) == set(original)
    for name in original:
        assert torch.equal(converted[name], original[name]), name
    loaded = control.load_state_dict(converted, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys


@pytest.mark.parametrize(
    "metadata",
    [
        {"control_blocks_places": "[0,10,20,30,40]"},
        {"control_blocks_places": "[0,5,10,15,20,25,30,35,40,45.0]"},
        {"inpaint_masked_pixel_mode": "pre_norm"},
        {"control_apply_audio": "true"},
        {"control_apply_audio": 0},
        {"minimax_h3_fun_controlnet": "adaln_basis"},
    ],
)
def test_explicit_conflicting_metadata_rejected(control, metadata):
    with pytest.raises(ValueError):
        union.union_structure(control.state_dict(), metadata)


@pytest.mark.parametrize(
    "corruption", ["missing_block", "extra_block", "channels", "width8"]
)
def test_wrong_actual_structure_is_not_filename_policy(control, corruption):
    state = dict(control.state_dict())
    if corruption == "missing_block":
        state = {
            name: value
            for name, value in state.items()
            if not name.startswith("control_blocks.6.")
        }
    elif corruption == "extra_block":
        state["control_blocks.11.after_proj.weight"] = state[
            "control_blocks.9.after_proj.weight"
        ]
    elif corruption == "channels":
        state["control_proj_in.weight"] = torch.zeros(4, 192)
    else:
        state["control_blocks.0.adaln_proj.linear.weight"] = torch.zeros(72, 8)
    with pytest.raises(ValueError):
        union.union_structure(state, {})


@pytest.mark.parametrize("raw", [False, True])
def test_loader_uses_real_safetensors_and_actual_core_modules(
    control, tmp_path, monkeypatch, raw
):
    state = raw_names(control.state_dict()) if raw else control.state_dict()
    path = tmp_path / "user_selected_name.safetensors"
    save_file(state, str(path), metadata={"format": "pt"})
    monkeypatch.setattr(
        union, "_resolve_fun_control_path", lambda name: (str(path), "controlnet")
    )
    bundle, report_json = union.load_union2(path.name)
    assert type(bundle.control.model) is MiniMaxH3FunControl
    assert bundle.control.model.injection_layers == union.LAYERS
    assert bundle.control.model.inpaint_post_norm is True
    assert len(bundle.control.model.control_blocks) == 10
    assert bundle.control.model.control_blocks[0].adaln_proj.linear.in_features == 2688
    report = json.loads(report_json)
    assert (
        report["profile"] == union.PROFILE
        and report["training_identity_inferred"] is False
    )
    assert report["structure"]["raw_names_converted"] is raw
    assert not torch.cuda.is_initialized()


class Model:
    def __init__(self, width=2688, blocks=50):
        projection = SimpleNamespace(
            linear=SimpleNamespace(weight=torch.zeros(72, width))
        )
        self.model = SimpleNamespace(
            diffusion_model=SimpleNamespace(
                blocks=[SimpleNamespace(adaln_proj=projection) for _ in range(blocks)]
            )
        )
        self.model_options = {"transformer_options": {}}
        self.wrappers = {}
        self.attachments = {}

    def get_model_object(self, name):
        assert name == "model_sampling"
        return SimpleNamespace(percent_to_sigma=lambda value: 1 - value)

    def clone(self):
        result = copy.copy(self)
        result.model_options = copy.deepcopy(self.model_options)
        result.wrappers = {key: list(values) for key, values in self.wrappers.items()}
        result.attachments = dict(self.attachments)
        return result

    def add_wrapper(self, kind, wrapper):
        self.wrappers.setdefault(kind, []).append(wrapper)

    def set_model_patch_replace(self, value, family, name, index):
        self.model_options["transformer_options"].setdefault(
            "patches_replace", {}
        ).setdefault(family, {})[(name, index)] = value

    def set_attachments(self, name, value):
        self.attachments[name] = value


def bundle(control):
    return H3FunControlBundle(
        "official_model_patch",
        SimpleNamespace(model=control),
        "arbitrary.safetensors",
        "not_a_provenance_gate",
        {"profile": union.PROFILE},
    )


def apply(control, **changes):
    args = dict(
        model=Model(),
        positive=[[torch.zeros(1), {"old": "unchanged"}]],
        bundle=bundle(control),
        vae=object(),
        source_video=torch.full((5, 32, 32, 3), 0.9),
        regen_mask=torch.zeros(5, 32, 32),
        width=32,
        height=32,
        length=5,
    )
    args.update(changes)
    return union.apply_union2(**args)


def test_actual_native_patch_register_preserves_previous_callable_and_wrappers(control):
    model = Model()
    calls = []

    def previous(args, extra):
        calls.append("previous")
        return extra["original_block"](args)

    def sentinel(*args):
        return None
    model.wrappers[comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL] = [sentinel]
    model.set_model_patch_replace(previous, "dit", "double_block", 5)
    positive = [[torch.zeros(1), {"old": "unchanged"}]]
    patched, conditioning, report = apply(control, model=model, positive=positive)
    assert patched is not model and conditioning is positive
    assert model.wrappers[comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL] == [
        sentinel
    ]
    wrappers = patched.wrappers[comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL]
    assert wrappers[0] is sentinel and type(wrappers[1].__self__) is AlignedUnion2Patch
    owner = wrappers[1].__self__
    assert owner.expected_shape == (1, 24, 2, 2, 2) and owner.control_latent is None
    replacing = patched.model_options["transformer_options"]["patches_replace"]["dit"]
    assert set(replacing) == {("double_block", index) for index in union.LAYERS}
    assert type(replacing[("double_block", 5)]) is MiniMaxH3FunControlBlockPatch
    assert replacing[("double_block", 5)].previous is previous
    out = replacing[("double_block", 5)](
        {"img": torch.ones(1)}, {"original_block": lambda args: {"img": args["img"]}}
    )
    assert calls == ["previous"] and torch.equal(out["img"], torch.ones(1))
    assert json.loads(report)["portable_cache_certified"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"source_video": torch.zeros(4, 32, 32, 3)},
        {"source_video": torch.zeros(5, 64, 32, 3)},
        {"source_video": torch.full((5, 32, 32, 3), float("nan"))},
        {"control_video": torch.zeros(6, 32, 32, 3)},
        {"regen_mask": torch.zeros(1, 32, 32)},
        {"regen_mask": torch.ones(5, 32, 32) * -1},
        {"width": 33},
        {"length": 6},
        {"length": True},
        {"broadcast_single_mask": 1},
        {"strength": float("nan")},
        {"start_percent": 0.8, "end_percent": 0.2},
        {"model": Model(width=8)},
        {"model": Model(blocks=45)},
    ],
)
def test_wrong_media_canvas_or_live_base_is_rejected(control, changes):
    with pytest.raises((ValueError, RuntimeError)):
        apply(control, **changes)


def test_single_mask_is_only_broadcast_when_explicit(control):
    patched, _, _ = apply(
        control, regen_mask=torch.ones(1, 32, 32), broadcast_single_mask=True
    )
    owner = patched.wrappers[comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL][
        -1
    ].__self__
    assert owner.mask.shape == (5, 32, 32) and torch.equal(
        owner.mask, torch.ones(5, 32, 32)
    )


def test_strength_zero_preserves_exact_objects_no_requirements_or_encoding():
    model, positive = object(), object()
    result = union.apply_union2(
        model, positive, None, None, None, None, None, None, None, strength=0
    )
    assert result[0] is model and result[1] is positive
    assert json.loads(result[2])["vae_encoded"] is False


class VAE:
    def __init__(self):
        self.inputs = []

    def spacial_compression_encode(self):
        return 16

    def encode(self, value):
        self.inputs.append(value.clone())
        # Deterministic stand-in encoder; actual Core builds every channel/keep/fill.
        scalar = value.mean()
        return torch.ones(1, 24, 2, 2, 2) * scalar


@pytest.mark.parametrize("with_hint", [False, True])
def test_native_post_norm_fill_keep_and_49_channel_order(
    control, monkeypatch, with_hint
):
    from comfy.ldm.minimax.vae import IMAGENET_MEAN

    monkeypatch.setattr(
        union.comfy.model_management, "loaded_models", lambda **kwargs: ["previous"]
    )
    reloaded = []
    monkeypatch.setattr(
        union.comfy.model_management,
        "load_models_gpu",
        lambda models: reloaded.append(models),
    )
    vae = VAE()
    mask = torch.zeros(5, 32, 32)
    mask[:, :, :16] = 1
    frames = torch.ones(5, 32, 32, 3) * 0.2 if with_hint else None
    patched, _, _ = apply(control, vae=vae, regen_mask=mask, control_video=frames)
    patch = patched.wrappers[comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL][
        -1
    ].__self__
    patch.prepare_control_latent((1, 24, 2, 2, 2))
    expected = torch.full((5, 32, 32, 3), 0.9)
    expected[:, :, :16] = torch.tensor(IMAGENET_MEAN)
    assert torch.equal(vae.inputs[-1], expected)
    assert len(vae.inputs) == (2 if with_hint else 1)
    latent = patch.control_latent
    assert latent.shape == (1, 49, 2, 2, 2)
    assert torch.equal(latent[:, 24:25, :, :, 0], torch.zeros(1, 1, 2, 2))
    assert torch.equal(latent[:, 24:25, :, :, 1], torch.ones(1, 1, 2, 2))
    assert torch.equal(latent[:, 25:], torch.ones(1, 24, 2, 2, 2) * expected.mean())
    if with_hint:
        assert torch.equal(vae.inputs[0], frames)
        assert torch.equal(latent[:, :24], torch.ones(1, 24, 2, 2, 2) * frames.mean())
    else:
        assert torch.count_nonzero(latent[:, :24]) == 0
    patch.prepare_control_latent((1, 24, 2, 2, 2))
    assert len(vae.inputs) == (2 if with_hint else 1) and reloaded == [["previous"]]
    assert not torch.cuda.is_initialized()


def test_actual_sampling_shape_mismatch_cannot_trigger_core_resize_or_pad(control):
    vae = VAE()
    patched, _, _ = apply(control, vae=vae)
    patch = patched.wrappers[comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL][
        -1
    ].__self__
    with pytest.raises(ValueError, match="actual sampler"):
        patch.prepare_control_latent((1, 24, 7, 2, 2))
    assert vae.inputs == []
    vae.spacial_compression_encode = lambda: 8
    with pytest.raises(ValueError, match="16x"):
        patch.prepare_control_latent((1, 24, 2, 2, 2))
    assert vae.inputs == []


def test_native_audio_skip_zero_is_not_joint_audio_identity_claim(control):
    patched, _, _ = apply(control)
    patch = patched.wrappers[comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL][
        -1
    ].__self__
    patch.active = True
    patch.control_latent = torch.zeros(1, 49, 2, 2, 2)
    patch.pristine_stream = torch.zeros(5, 4)
    control.init_stream = lambda *args: torch.ones(5, 4)
    control.step = lambda *args, **kwargs: (torch.ones(5, 4), torch.ones(5, 4) * 2)
    incoming = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    original = incoming.clone()
    args = {
        "layout": SimpleNamespace(audio_pos=torch.tensor([1, 3])),
        "t_emb": None,
        "mod_segments": None,
        "rope_freqs": None,
        "transformer_options": {},
    }
    out = patch.after_block(0, args, {"img": incoming})
    assert torch.equal(out["img"][[1, 3]], original[[1, 3]])
    assert torch.equal(out["img"][[0, 2, 4]], original[[0, 2, 4]] + 1.4)


def test_distinct_loader_socket_and_explicit_external_apply_schema():
    loader = MiniMaxH3FunUnion2LoaderEXPT8.define_schema().get_v1_info(
        MiniMaxH3FunUnion2LoaderEXPT8
    )
    applier = MiniMaxH3FunUnion2ApplyEXPT8.define_schema().get_v1_info(
        MiniMaxH3FunUnion2ApplyEXPT8
    )
    assert loader.output[0] == union.CONTROL_TYPE
    assert applier.input["required"]["union_control"][0] == union.CONTROL_TYPE
    assert applier.input["required"]["broadcast_single_mask"][1]["default"] is False
    assert set(applier.input["optional"]) == {"control_video"}
    assert not any(
        "relay" in name.lower() or "eav" in name.lower()
        for name in applier.input["required"]
    )
