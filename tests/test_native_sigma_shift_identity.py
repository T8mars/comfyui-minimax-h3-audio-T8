"""Actual Core Shift + original ER-SDE, not a reimplementation of sampling math."""
from copy import deepcopy
import importlib
import inspect
from pathlib import Path
import sys
from types import FunctionType

import pytest
import torch
import comfy.model_sampling as core_sampling
from comfy_extras.nodes_custom_sampler import (
    BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced,
)
from h3_audio_t8_pkg import long_video_dual_identity as identity
from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.modular_sampling import native_explicit
from h3_audio_t8_pkg.modular_sampling.face_stage import (
    bind_parity_face_stage, bind_multiface_face_stage, bind_window_face_stage,
)
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import load_stage, save_stage
from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from test_fast_h3_v2_core_sampler import model
from test_face_refine_parity_advanced import _locked_av
from test_modular_face_multiface import _job as multiface_job
from test_modular_face_parity import _inputs as parity_inputs
from test_modular_face_window import _job as window_job
from test_progressive_sampling_runtime import conditioning


def _actual_core_node_module():
    # conftest adds the nodepack's h3_t8/ directory. Core's native node module
    # imports root "nodes", which must be the real Core module, not this pack.
    core = Path(core_sampling.__file__).resolve().parents[1]
    original_path = list(sys.path)
    sys.path.insert(0, str(core))
    try:
        actual_nodes = importlib.import_module("nodes")
        assert Path(actual_nodes.__file__).resolve() == core / "nodes.py"
        native = importlib.import_module("comfy_extras.nodes_minimax_h3")
        assert Path(native.__file__).resolve() == core / "comfy_extras/nodes_minimax_h3.py"
        return native
    finally:
        sys.path[:] = original_path


native_h3 = _actual_core_node_module()


def _configured(video_shift=12., audio_shift=3.):
    bare = model()
    before = deepcopy(bare.model_options)
    original_sampling = bare.get_model_object("model_sampling")
    patched = native_h3.MiniMaxH3SigmaShift.execute(bare, video_shift, audio_shift).result[0]
    assert bare.model_options == before and bare.object_patches == {}
    assert bare.get_model_object("model_sampling") is original_sampling
    return patched


def _job(variant):
    if variant == "parity":
        frames, plan, latent, *_ = parity_inputs(per_frame=True)
        return latent, lambda patched, sampler, sigmas: bind_parity_face_stage(
            plan, frames, patched, sampler, sigmas, latent)
    if variant == "multiface":
        parent, frames, plan = multiface_job()
        latent = _locked_av()
        return latent, lambda patched, sampler, sigmas: bind_multiface_face_stage(
            plan, frames, parent, patched, sampler, sigmas, latent)
    parent, audio, window_plan, frames, window_audio, mapping, plan = window_job()
    latent = _locked_av(frame_count=22)
    return latent, lambda patched, sampler, sigmas: bind_window_face_stage(
        plan, frames, parent, window_plan, mapping, audio, window_audio,
        patched, sampler, sigmas, latent)


@pytest.mark.parametrize("variant", ("parity", "multiface", "window"))
@pytest.mark.parametrize("denoise", (.45, 1.))
def test_native_shift_direct_stage_matches_stock_sampling_and_saved_result(variant, denoise, tmp_path):
    latent, bind = _job(variant)
    original, candidate = _configured(), _configured()
    first, second = original.model.state_dict(), candidate.model.state_dict()
    assert first.keys() == second.keys() and original.model is not candidate.model
    assert all(torch.equal(value, second[key]) for key, value in first.items())
    native_sigmas = BasicScheduler.execute(original, "simple", 8, denoise).result[0]
    sigmas = BasicScheduler.execute(candidate, "simple", 8, denoise).result[0]
    assert torch.equal(native_sigmas, sigmas)
    sampler = KSamplerSelect.execute("er_sde").result[0]
    seed = 26091001
    torch.manual_seed(seed)
    expected = SamplerCustomAdvanced.execute(RandomNoise.execute(seed).result[0],
        BasicGuider.execute(original, conditioning()).result[0], sampler, native_sigmas, latent).result
    bound, selected, table, context, _ = bind(candidate, sampler, sigmas)
    assert selected is sampler and table is sigmas and context.steps == 8
    torch.manual_seed(seed)
    result = sample_stage(RandomNoise.execute(seed).result[0],
        BasicGuider.execute(bound, conditioning()).result[0], selected, table, latent, context)[2]
    for actual, baseline in zip((result.output, result.denoised_output), expected, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), baseline["samples"].unbind(), strict=True))
    assert result.verify()["portable_identity"] is True
    assert result.verify()["verified_recipe_completion"] is True
    path, digest, _ = save_stage(result, tmp_path, "core-native-shift")
    _, _, loaded_context, loaded, _ = load_stage(tmp_path, path, digest, context.stage)
    assert loaded_context == context and loaded.verify() == result.verify()
    with pytest.raises(ValueError, match="SHA"):
        load_stage(tmp_path, path, "0" * 64, context.stage)


def test_existing_T8_identity_format_and_new_core_content_binding():
    t8_model, _, _ = sampling.setup_dual_clock_sampling(model(), _locked_av(), 8, 12., 3., "er_sde", "simple")
    old = identity._v2_sampling_identity(t8_model.get_model_object("model_sampling"))
    assert old["protocol"] == "native_av_carrier"
    assert set(old) == {"schema", "protocol", "configuration", "implementation", "methods"}
    selected = _configured().get_model_object("model_sampling")
    first = identity._v2_sampling_identity(selected)
    assert first["protocol"] == "core_minimax_h3_sigma_shift"
    assert first["factory"]["source_sha256"] == identity.CORE_SIGMA_SHIFT_SOURCE_SHA256
    selected.set_parameters(shift=7.5, audio_shift=2.25)
    second = identity._v2_sampling_identity(selected)
    assert first != second and first["factory"] == second["factory"]
    selected.sigmas[3] += .001
    assert identity._v2_sampling_identity(selected) != second


@pytest.mark.parametrize("mutation", ("parameters", "buffer", "clock", "replacement"))
def test_bound_native_shift_owner_rejects_changed_selected_state(mutation):
    latent, bind = _job("parity")
    configured = _configured()
    sigmas = BasicScheduler.execute(configured, "simple", 8, .45).result[0]
    bound, *_ = bind(configured, KSamplerSelect.execute("er_sde").result[0], sigmas)
    native_explicit.capture_owner(bound)
    selected = bound.get_model_object("model_sampling")
    if mutation == "parameters":
        selected.set_parameters(shift=7.5, audio_shift=2.25)
    elif mutation == "buffer":
        selected.sigmas[3] += .001
    elif mutation == "clock":
        bound.model_options["transformer_options"]["minimax_h3_sigma_shift_audio"] = 9.
    else:
        bound.object_patches["model_sampling"] = _configured().get_model_object("model_sampling")
    with pytest.raises(ValueError, match="sampling configuration changed|clock shifts|sampling object was replaced"):
        native_explicit.capture_owner(bound)


@pytest.mark.parametrize("mutation", ("method", "instance", "hook", "metaclass", "foreign_name", "foreign_bases"))
def test_unknown_sampling_execution_is_not_portably_certified(mutation):
    selected = _configured().get_model_object("model_sampling")
    cls = type(selected)
    if mutation == "method":
        cls.sigma = lambda self, value: value
    elif mutation == "instance":
        selected.forward = lambda value: value
    elif mutation == "hook":
        selected.register_forward_pre_hook(lambda module, values: values)
    elif mutation == "metaclass":
        class ForeignMeta(type):
            pass
        selected = ForeignMeta("ModelSamplingAdvanced", cls.__bases__,
            {"__module__": cls.__module__, "__qualname__": cls.__qualname__})()
    elif mutation == "foreign_name":
        cls.__module__ = __name__
    else:
        selected = type("ModelSamplingAdvanced", (core_sampling.ModelSamplingDiscreteFlow, core_sampling.CONST),
            {"__module__": cls.__module__, "__qualname__": cls.__qualname__})()
    with pytest.raises(UnverifiedModelStack):
        identity._v2_sampling_identity(selected)


def test_unknown_core_source_epoch_is_not_certified(monkeypatch):
    selected = _configured().get_model_object("model_sampling")
    monkeypatch.setattr(identity, "CORE_SIGMA_SHIFT_SOURCE_SHA256", "0" * 64)
    with pytest.raises(UnverifiedModelStack, match="audited sampling factory"):
        identity._v2_sampling_identity(selected)


def test_live_factory_bytecode_replacement_is_not_certified(monkeypatch):
    selected = _configured().get_model_object("model_sampling")
    factory = inspect.unwrap(native_h3.MiniMaxH3SigmaShift.execute)
    altered = FunctionType(factory.__code__.replace(co_consts=(*factory.__code__.co_consts, "private-mutation")),
        factory.__globals__, factory.__name__, factory.__defaults__)
    altered.__qualname__ = factory.__qualname__
    monkeypatch.setattr(native_h3.MiniMaxH3SigmaShift, "execute", classmethod(altered))
    with pytest.raises(UnverifiedModelStack, match="bytecode differs"):
        identity._v2_sampling_identity(selected)
