"""Actual tiny Core AV continuous PTD; synthetic experts, never GPU quality."""
from copy import deepcopy
from collections import UserDict
import json

import pytest
import torch
import comfy.model_management as mm
import comfy.sample
import comfy.utils
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise

from h3_audio_t8_pkg import ptd_advanced as ptd
from h3_audio_t8_pkg import ptd_checkpoint as files
from h3_audio_t8_pkg import ptd_sampling as clocks
from h3_audio_t8_pkg import ptd_continuous as continuous
from h3_audio_t8_pkg import nodes_ptd_continuous as nodes
from test_fast_h3_v2_core_sampler import model
from test_h21_ptd_advanced import experts, sample
from test_h21_projection_model import sample as plain
from test_progressive_sampling_runtime import conditioning, latent


@pytest.fixture(autouse=True)
def CPU(monkeypatch):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    with torch.inference_mode():
        yield
    assert not torch.cuda.is_initialized()


def selected(recipe="h3_4step", base=None):
    base = model() if base is None else base
    chosen, _ = ptd.configure_ptd(base, experts(base, recipe), files.metadata_for(recipe))
    return base, chosen


def guider(chosen):
    return BasicGuider.execute(chosen, conditioning()).result[0]


def head(chosen, source=None):
    return continuous.run_head(RandomNoise.execute(212104).result[0], guider(chosen),
                               latent() if source is None else source)


@pytest.mark.parametrize("recipe", files.GRIDS)
def test_actual_Core_two_AV_streams_split_is_bitexact_to_original_complete(recipe):
    base, chosen = selected(recipe)
    expected = sample(chosen, ptd.get_controller(chosen))
    before_plain = plain(base)
    source = latent()
    source_before = deepcopy(source)
    prefix, head_report = head(chosen, source)
    middle = (len(files.GRIDS[recipe]) - 1) // 2
    assert head_report["global_hops"] == list(range(middle))
    assert head_report["next_global_hop"] == middle
    assert head_report["range_completed"] and not head_report["whole_recipe_completed"]
    assert [row["hop"] for row in head_report["observed_network_forwards"]] == list(range(middle))
    assert prefix.payload["state"].dtype is torch.float32
    # This is Core's real flat internal pack, not an x0/result LATENT alias.
    packed, shapes = comfy.utils.pack_latents(source["samples"].unbind())
    assert prefix.payload["state"].shape == packed.shape
    internal_streams = comfy.utils.unpack_latents(prefix.payload["state"], shapes)
    assert len(internal_streams) == 2 and bool(torch.count_nonzero(internal_streams[1]))
    actual, completion, tail_report = continuous.run_tail(guider(chosen), prefix)
    assert tail_report["global_hops"] == list(range(middle, len(files.GRIDS[recipe]) - 1))
    assert tail_report["all_global_hops"] == list(range(len(files.GRIDS[recipe]) - 1))
    assert tail_report["whole_recipe_completed"] and not tail_report["portable_cache_identity"]
    assert [row["hop"] for row in tail_report["observed_network_forwards"]] == tail_report["global_hops"]
    assert len(tail_report["actual_expert_calls"]) == len(ptd.get_controller(chosen).bank) * middle
    for actual_part, expected_part in zip(actual["samples"].unbind(), expected["samples"].unbind()):
        torch.testing.assert_close(actual_part, expected_part, rtol=0, atol=0)
    assert continuous.inspect_stage(completion) == tail_report
    for value, before in zip(source["samples"].unbind(), source_before["samples"].unbind()):
        torch.testing.assert_close(value, before, rtol=0, atol=0)
    after_plain = plain(base)
    for value, before in zip(after_plain["samples"].unbind(), before_plain["samples"].unbind()):
        torch.testing.assert_close(value, before, rtol=0, atol=0)
    root = base.model.diffusion_model
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in root.modules())
    assert ptd.get_controller(chosen).request.get() is None


def test_TAIL_does_not_generate_new_random_noise_or_change_global_RNG(monkeypatch):
    _, chosen = selected()
    prefix, _ = head(chosen)
    random_before = torch.random.get_rng_state().clone()
    monkeypatch.setattr(comfy.sample, "prepare_noise", lambda *a, **k: pytest.fail("TAIL generated new noise"))
    output, _, _ = continuous.run_tail(guider(chosen), prefix)
    assert torch.equal(torch.random.get_rng_state(), random_before)
    assert all(bool(torch.isfinite(part).all()) for part in output["samples"].unbind())


@pytest.mark.parametrize("damage", ["finite", "nan", "shape", "index", "grid", "template", "report"])
def test_HEAD_actual_content_and_completion_cannot_be_mutated(damage):
    _, chosen = selected()
    prefix, _ = head(chosen)
    payload = prefix.payload
    if damage == "finite":
        payload["state"].add_(.1)
    elif damage == "nan":
        payload["state"].reshape(-1)[0] = float("nan")
    elif damage == "shape":
        payload["state"] = payload["state"].reshape(-1)
    elif damage == "index":
        payload["next_hop"] = 0
    elif damage == "grid":
        payload["full_grid"] = (1., .5, 0.)
    elif damage == "template":
        payload["template"]["samples"].unbind()[0].add_(.1)
    else:
        payload["report"]["range_completed"] = False
    with pytest.raises(ValueError, match="changed|finite"):
        continuous.run_tail(guider(chosen), prefix)


def test_actual_complete_output_mutation_invalidates_its_own_audit():
    _, chosen = selected()
    prefix, _ = head(chosen)
    output, completion, _ = continuous.run_tail(guider(chosen), prefix)
    output["samples"].unbind()[1].add_(.25)
    with pytest.raises(ValueError, match="contents changed"):
        continuous.inspect_stage(completion)


def test_real_Core_UserDict_provider_is_snapshotted_without_upstream_tensor_alias():
    _, chosen = selected()
    expected = sample(chosen, ptd.get_controller(chosen))
    source = UserDict(latent())
    prefix, _ = head(chosen, source)
    source["samples"].unbind()[0].add_(.5)
    assert not bool(torch.count_nonzero(prefix.payload["template"]["samples"].unbind()[0]))
    actual, _, report = continuous.run_tail(guider(chosen), prefix)
    for value, before in zip(actual["samples"].unbind(), expected["samples"].unbind()):
        torch.testing.assert_close(value, before, rtol=0, atol=0)
    assert report["whole_recipe_completed"] and not report["portable_cache_identity"]


@pytest.mark.parametrize("kind", [continuous.PTDHead, continuous.PTDCompletion])
def test_plain_LATENT_JSON_or_constructed_object_cannot_fake_actual_completion(kind):
    value = kind()
    value.payload = {"report": {"whole_recipe_completed": True}}
    with pytest.raises(ValueError, match="not issued"):
        continuous.inspect_stage(value)
    with pytest.raises(ValueError, match="actual typed"):
        continuous.inspect_stage({"report": {"whole_recipe_completed": True}})


def test_foreign_guider_that_never_calls_integrator_cannot_issue_HEAD(monkeypatch):
    _, chosen = selected()
    target = guider(chosen)
    monkeypatch.setattr(target, "sample", lambda noise, original, *a, **k: original)
    with pytest.raises(ValueError, match="did not execute"):
        continuous.run_head(RandomNoise.execute(3).result[0], target, latent())


def test_foreign_attention_and_existing_wrapper_are_kept_through_both_phases():
    base = model()
    layer = base.model.diffusion_model.blocks[0].attn
    original, observed, wrappers = layer.forward, [], []

    def foreign(*args, **kwargs):
        observed.append(1)
        return original(*args, **kwargs)

    def previous(function, payload):
        wrappers.append(1)
        return function(payload["input"], payload["timestep"], **payload["c"])

    layer.forward = foreign
    base.set_model_unet_function_wrapper(previous)
    _, chosen = selected(base=base)
    prefix, low = head(chosen)
    _, _, high = continuous.run_tail(guider(chosen), prefix)
    assert observed == [1] * 4 and wrappers == [1] * 4
    assert low["partial"] and high["partial"]
    assert any("unknown_Attention" in item for item in high["delegated"])
    assert layer.forward is foreign


def test_each_stage_keeps_its_actual_ordered_LoRA_and_different_choices():
    base = model()
    key = "diffusion_model.blocks.0.attn.out_proj.weight"
    weight = base.model_state_dict()[key]
    first, second = base.clone(), base.clone()
    first.add_patches({key: ("diff", (torch.ones_like(weight) * .1,))}, strength_patch=-.25)
    second.add_patches({key: ("diff", (torch.ones_like(weight) * .2,))}, strength_patch=.6)
    second.add_patches({key: ("diff", (torch.ones_like(weight) * .1,))}, strength_patch=-.1)
    _, low = selected(base=first)
    _, high = selected(base=second)
    prefix, _ = head(low)
    expected, _, _ = continuous.run_tail(guider(low), prefix)
    actual, _, report = continuous.run_tail(guider(high), prefix)
    assert [row[0] for row in first.patches[key]] == [-.25]
    assert [row[0] for row in second.patches[key]] == [.6, -.1]
    assert [row[0] for row in high.patches[key]] == [.6, -.1]
    assert not torch.equal(actual["samples"].unbind()[0], expected["samples"].unbind()[0])
    assert report["all_global_hops"] == [0, 1, 2, 3] and not report["portable_cache_identity"]


def test_real_Core_cancel_then_same_HEAD_retry_and_next_plain_restore(monkeypatch):
    base, chosen = selected()
    reference = plain(base)
    prefix, _ = head(chosen)
    original = mm.throw_exception_if_processing_interrupted
    calls = []

    def interrupt():
        if ptd._FRAME.get() is not None and not calls:
            calls.append(1)
            raise mm.InterruptProcessingException()
        return original()

    monkeypatch.setattr(mm, "throw_exception_if_processing_interrupted", interrupt)
    with pytest.raises(mm.InterruptProcessingException):
        continuous.run_tail(guider(chosen), prefix)
    assert calls == [1]
    assert ptd.get_controller(chosen).request.get() is None
    continuous.inspect_stage(prefix)
    continuous.run_tail(guider(chosen), prefix)
    actual = plain(base)
    for value, expected in zip(actual["samples"].unbind(), reference["samples"].unbind()):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_actual_last_forward_error_not_hidden_by_seal_and_retry_works():
    base, chosen = selected()
    prefix, _ = head(chosen)
    controller = ptd.get_controller(chosen)
    factor = controller.bank["blocks.0.attn.out_proj"][0][0].a
    saved = factor.clone()

    def fail(_module, _args, _out):
        if controller.current()[0] == 3:
            factor.add_(.1)
            raise TypeError("original TAIL user failure")

    handle = base.model.diffusion_model.final_layer.register_forward_hook(fail)
    try:
        with pytest.raises(TypeError, match="original TAIL user failure"):
            continuous.run_tail(guider(chosen), prefix)
    finally:
        handle.remove()
        factor.copy_(saved)
    continuous.run_tail(guider(chosen), prefix)


def test_complete_PTD_entry_and_old_full_grid_contract_unchanged():
    _, chosen = selected()
    with pytest.raises(ValueError, match="complete unmodified"):
        clocks.validate_grid(clocks.ptd_sigmas("h3_4step")[:3], "h3_4step")
    sample(chosen, ptd.get_controller(chosen))
    assert ptd.get_controller(chosen).report()["last_request"]["hops"] == [0, 1, 2, 3]


def test_real_callback_global_indices_full_grid_and_noisy_audio_mutation_match_full(monkeypatch):
    import latent_preview

    _, chosen = selected()
    seen = []

    def make_callback(_model, _total, _x0_output):
        def callback(step, _x0, noisy, total):
            seen.append((step, total))
            # A real preview callback may mutate the live state. Preserve its
            # original placement after velocity and before the state update.
            noisy.unbind()[1].add_(.0001 * (step + 1))
        return callback

    monkeypatch.setattr(latent_preview, "prepare_callback", make_callback)
    expected = sample(chosen, ptd.get_controller(chosen))
    assert seen == [(step, 4) for step in range(4)]
    seen.clear()
    prefix, _ = head(chosen)
    actual, _, _ = continuous.run_tail(guider(chosen), prefix)
    assert seen == [(step, 4) for step in range(4)]
    for value, reference in zip(actual["samples"].unbind(), expected["samples"].unbind()):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_new_unregistered_node_schemas_and_actual_typed_audit():
    assert len(nodes.NODES) == 3 and all(node.define_schema().is_experimental for node in nodes.NODES)
    _, chosen = selected()
    prefix, text = nodes.MiniMaxH3PTDContinuousHEAD_EXPT8.execute(
        RandomNoise.execute(212104).result[0], guider(chosen), latent()).result
    assert json.loads(text)["phase"] == "HEAD"
    out, completion, text = nodes.MiniMaxH3PTDContinuousTAIL_EXPT8.execute(guider(chosen), prefix).result
    assert len(out["samples"].unbind()) == 2
    audited = json.loads(nodes.MiniMaxH3PTDContinuousAuditEXPT8.execute(completion).result[0])
    assert audited == json.loads(text) and audited["whole_recipe_completed"]
