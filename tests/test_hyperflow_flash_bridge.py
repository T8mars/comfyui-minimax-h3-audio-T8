import json
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg import hyperflow_flash_bridge as bridge
from h3_audio_t8_pkg.nodes_hyperflow_flash_bridge import NODES


def test_external_provider_receives_exact_selection_and_model_is_not_rewritten(monkeypatch):
    model = SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(
        hyperflow_sigmas=torch.tensor([1., .8, .3, 0.]))))
    calls = []
    class ForeignProvider:
        FUNCTION = "load"
        def load(self, unet_name, weights_in):
            calls.append((unet_name, weights_in))
            return (model,)
    monkeypatch.setitem(bridge.core_nodes.NODE_CLASS_MAPPINGS, "HyperFlowLoader", ForeignProvider)
    result, report = bridge.load_flash("my-renamed-checkpoint.safetensors", "disk")
    assert result is model and calls == [("my-renamed-checkpoint.safetensors", "disk")]
    assert json.loads(report)["checkpoint_schedule"] == model.model.diffusion_model.hyperflow_sigmas.tolist()
    assert json.loads(report)["quality_validated"] is False
    sigma = bridge.checkpoint_schedule(model)
    assert torch.equal(sigma, model.model.diffusion_model.hyperflow_sigmas)
    assert sigma.data_ptr() != model.model.diffusion_model.hyperflow_sigmas.data_ptr()


def test_missing_runtime_is_a_specific_dependency_error(monkeypatch):
    monkeypatch.delitem(bridge.core_nodes.NODE_CLASS_MAPPINGS, "HyperFlowLoader", raising=False)
    with pytest.raises(ImportError, match="official ComfyUI-HyperFlow-Flash"):
        bridge.load_flash("selected.safetensors")


@pytest.mark.parametrize("schedule", [torch.tensor([1., .3]), torch.tensor([1., .2, .4, 0.]),
                                     torch.tensor([1., float('nan'), 0.])])
def test_corrupt_schedule_is_not_guessed_or_shifted(schedule):
    model = SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(hyperflow_sigmas=schedule)))
    with pytest.raises(ValueError, match="hyperflow_sigmas"):
        bridge.checkpoint_schedule(model)


def test_registration_schema_does_not_require_optional_provider_to_be_installed():
    for cls in NODES:
        assert cls.define_schema().node_id.startswith("MiniMaxH3")


@pytest.mark.parametrize("steps", [4, 6, 8, 5])
def test_presets_delegate_without_a_second_shift_or_a_T8_step_whitelist(monkeypatch, steps):
    selected = object()
    schedule = torch.tensor([1., .75, 0.])
    calls = []
    class ScheduleProvider:
        FUNCTION = "sigmas"
        def sigmas(self, model, steps):
            calls.append((model, steps))
            return (schedule,)
    monkeypatch.setitem(bridge.core_nodes.NODE_CLASS_MAPPINGS,"HyperFlowSigmas",ScheduleProvider)
    assert bridge.flash_schedule(selected,str(steps)) is schedule
    assert calls==[(selected,steps)]


def test_provider_errors_are_not_replaced_by_an_ordinary_scheduler(monkeypatch):
    class BrokenProvider:
        def sigmas(self, model, steps):
            raise RuntimeError("actual selected provider error")
    monkeypatch.setitem(bridge.core_nodes.NODE_CLASS_MAPPINGS,"HyperFlowSigmas",BrokenProvider)
    with pytest.raises(RuntimeError,match="actual selected provider"):
        bridge.flash_schedule(object(),"4")
