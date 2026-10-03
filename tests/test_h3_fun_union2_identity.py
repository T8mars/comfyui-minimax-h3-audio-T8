"""Native owner/weight/producer-bound inspection, not inference qualification."""

import copy
from types import MethodType

import pytest
import torch
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrappersMP
from comfy.ldm.minimax.model import DiTBlock
import comfy.ops

from h3_audio_t8_pkg import h3_fun_union2 as union
from h3_audio_t8_pkg import h3_fun_union2_identity as identity
from h3_audio_t8_pkg.modular_sampling import results
from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from test_h3_fun_union2 import control as _control_fixture
from test_progressive_producers import component
from test_relay_kj_memory import small_model

control = _control_fixture


def scene(control):
    base = small_model()
    # Tiny native architecture shell with the actual full-width AdaLN and 50 positions.
    base.model.diffusion_model.blocks = torch.nn.ModuleList(
        [
            DiTBlock(
                4,
                1,
                4,
                8,
                2688,
                1e-5,
                1e-5,
                dtype=torch.float32,
                device="cpu",
                operations=comfy.ops.disable_weight_init,
            )
            for _ in range(50)
        ]
    )
    with torch.no_grad():
        for parameter in base.model.parameters():
            parameter.fill_(0.01)
    vae = component("video_vae")
    patcher = ModelPatcher(control, torch.device("cpu"), torch.device("cpu"))
    bundle = union.H3FunControlBundle(
        "official_model_patch",
        patcher,
        "arbitrary.safetensors",
        "arbitrary",
        {"profile": union.PROFILE},
    )
    positive = [[torch.zeros(1), {}]]
    patched, returned, _ = union.apply_union2(
        base,
        positive,
        bundle,
        vae,
        torch.ones(5, 32, 32, 3) * 0.9,
        torch.zeros(5, 32, 32),
        32,
        32,
        5,
    )
    assert returned is positive
    owner = patched.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)[-1].__self__
    return base, patched, owner, vae


def test_projection_removes_only_own_runtime_on_clone_and_binds_native_content(control):
    base, patched, owner, _ = scene(control)
    before = copy.deepcopy(patched.attachments)
    original_wrappers = patched.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)
    view, contract = identity.project(patched)
    assert view is not patched and view.get_attachment(union.ATTACHMENT_KEY) is None
    assert not view.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)
    assert "patches_replace" not in view.model_options["transformer_options"]
    assert (
        patched.attachments == before
        and patched.get_all_wrappers(WrappersMP.DIFFUSION_MODEL) == original_wrappers
    )
    assert (
        len(patched.model_options["transformer_options"]["patches_replace"]["dit"])
        == 10
    )
    assert contract == identity.project(patched)[1]
    assert contract["vae"]["role"] == "video_vae" and contract["control"]["state"]
    assert (
        results.selected_model_identity(patched).get("portable_cache_reuse")
        is not False
    )
    assert (
        results.selected_model_identity(patched)["stage_effects"]["fun_union2_operator"]
        == contract
    )
    assert results.selected_model_identity(base) != results.selected_model_identity(
        patched
    )
    owner.active = True
    assert identity.project(patched)[1] == contract


@pytest.mark.parametrize(
    "change", ["source", "mask", "control_weight", "vae_weight", "hole_mean", "sigma"]
)
def test_every_actual_numerical_input_invalidates_content(control, change, monkeypatch):
    _, model, owner, vae = scene(control)
    before = identity.project(model)[1]
    if change == "source":
        owner.source_video[0, 0, 0, 0] += 0.01
    elif change == "mask":
        owner.mask[0, 0, 0] = 1
    elif change == "control_weight":
        with torch.no_grad():
            owner.model_patch.model.control_blocks[3].after_proj.weight[0, 0] += 0.01
    elif change == "vae_weight":
        with torch.no_grad():
            vae.first_stage_model.probe[0] += 0.01
    elif change == "hole_mean":
        monkeypatch.setattr(identity, "IMAGENET_MEAN", [0.5, 0.5, 0.5])
    else:
        owner.sigma_end += 0.01
    assert identity.project(model)[1] != before


@pytest.mark.parametrize(
    "change",
    [
        "hook",
        "control_method",
        "owner_field",
        "replace",
        "wrapper",
        "producer",
        "patch",
    ],
)
def test_unknown_operators_do_not_get_fabricated_portability(control, change):
    _, model, owner, vae = scene(control)
    if change == "hook":
        control.register_forward_pre_hook(lambda *args: None)
    elif change == "control_method":
        control.step = MethodType(lambda self, *args: args[0], control)
    elif change == "owner_field":
        owner.foreign = object()
    elif change == "replace":
        model.set_model_patch_replace(lambda *args: None, "dit", "double_block", 5)
    elif change == "wrapper":
        owner.diffusion_model_wrapper = MethodType(lambda self, *args: None, owner)
        model.wrappers[WrappersMP.DIFFUSION_MODEL][None][-1] = (
            owner.diffusion_model_wrapper
        )
    elif change == "producer":
        vae.encode = lambda value: value
    else:
        owner.model_patch.add_patches(
            {
                "control_proj_in.weight": (
                    "diff",
                    (torch.ones_like(control.control_proj_in.weight),),
                )
            }
        )
    with pytest.raises(UnverifiedModelStack):
        identity.project(model)
    # Public sampler path preserves execution and reports nonportable, no policy ban.
    selected = results.selected_model_identity(model)
    assert selected["portable_cache_reuse"] is False
    assert model.get_attachment(union.ATTACHMENT_KEY) is not None


def test_prior_unknown_delegate_is_restored_on_inspection_clone_not_certified(control):
    base, _, _, vae = scene(control)
    calls = []

    def previous(args, extra):
        calls.append(1)
        return extra["original_block"](args)

    base.set_model_patch_replace(previous, "dit", "double_block", 5)
    bundle = union.H3FunControlBundle(
        "official_model_patch",
        ModelPatcher(control, torch.device("cpu"), torch.device("cpu")),
        "name",
        "path",
        {"profile": union.PROFILE},
    )
    patched, _, _ = union.apply_union2(
        base,
        [],
        bundle,
        vae,
        torch.zeros(5, 32, 32, 3),
        torch.zeros(5, 32, 32),
        32,
        32,
        5,
    )
    view, _ = identity.project(patched)
    assert (
        view.model_options["transformer_options"]["patches_replace"]["dit"][
            ("double_block", 5)
        ]
        is previous
    )
    assert results.selected_model_identity(patched)["portable_cache_reuse"] is False
    entry = patched.model_options["transformer_options"]["patches_replace"]["dit"][
        ("double_block", 5)
    ]
    entry({"img": torch.ones(1)}, {"original_block": lambda args: {"img": args["img"]}})
    assert calls == [1]


def test_derived_cache_content_verified_not_an_independent_stage_input(control):
    _, model, owner, _ = scene(control)
    before = identity.project(model)[1]
    owner.control_latent = torch.zeros(1, 49, 2, 2, 2)
    owner.control_latent_shape = (1, 24, 2, 2, 2)
    owner.derived_content = identity.content_identity(owner.control_latent)
    assert identity.project(model)[1] == before
    owner.control_latent[0, 48, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="derived.*changed"):
        results.selected_model_identity(model)
    owner.cleanup()
    assert identity.project(model)[1] == before


def test_no_attachment_has_exact_legacy_projection_identity(control):
    base, _, _, _ = scene(control)
    view, contract = identity.project(base)
    assert view is base and contract is None
