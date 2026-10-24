"""Independent actual tiny Core checks, not trained/GPU/portable qualification."""
from collections import OrderedDict
import json

import pytest
import torch
import comfy.model_management as mm
from comfy.nested_tensor import NestedTensor
from comfy.samplers import CFGGuider
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention

from h3_audio_t8_pkg import ptd_continuous as continuous
from h3_audio_t8_pkg import ptd_sampling as clocks
from test_h21_ptd_continuous import selected, guider, head
from test_progressive_sampling_runtime import conditioning, latent


@pytest.fixture(autouse=True)
def CPU(monkeypatch):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    with torch.inference_mode():
        yield
    assert not torch.cuda.is_initialized()


class ProviderLatent(dict):
    """An ordinary upstream provider's dictionary subclass, accepted by Core."""


@pytest.mark.parametrize("container", [dict, OrderedDict, ProviderLatent])
def test_actual_HEAD_snapshots_accepted_source_mapping_before_later_canvas_edit(container, tmp_path):
    _, chosen = selected()
    source = container(latent())
    original_shapes = [tuple(part.shape) for part in source["samples"].unbind()]
    prefix, _ = head(chosen, source)
    video, audio = source["samples"].unbind()
    # A legitimate subsequent provider/canvas edit keeps total packed length,
    # but must not retroactively reinterpret already sampled HEAD coordinates.
    source["samples"] = NestedTensor((video.transpose(-1, -2).contiguous(), audio))
    actual, _, report = continuous.run_tail(guider(chosen), prefix)
    (tmp_path / "actual-layout-observation.json").write_text(json.dumps({
        "input_container": container.__name__, "original_shapes": original_shapes,
        "actual_output_shapes": [list(part.shape) for part in actual["samples"].unbind()],
        "HEAD_template_is_original_source": prefix.payload["template"] is source,
        "raw_state_shape": list(prefix.payload["state"].shape),
        "global_hops": report["all_global_hops"], "whole_recipe_completed": report["whole_recipe_completed"],
        "partial": report["partial"], "portable_cache_identity": report["portable_cache_identity"],
        "warnings": report["warnings"], "CUDA_initialized": torch.cuda.is_initialized(),
    }, indent=2, allow_nan=False), encoding="utf8")
    assert [tuple(part.shape) for part in actual["samples"].unbind()] == original_shapes
    assert prefix.payload["template"] is not source
    assert report["all_global_hops"] == [0, 1, 2, 3]
    assert report["portable_cache_identity"] is False


def test_changed_TAIL_conditioning_runs_actual_new_branch_without_portable_claim():
    _, chosen = selected()
    prefix, _ = head(chosen)
    original, _, _ = continuous.run_tail(guider(chosen), prefix)
    changed = conditioning()
    changed[0][0].add_(.75)
    new_guider = BasicGuider.execute(chosen, changed).result[0]
    actual, completion, report = continuous.run_tail(new_guider, prefix)
    assert any(not torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), original["samples"].unbind()))
    assert report["whole_recipe_completed"] and report["all_global_hops"] == [0, 1, 2, 3]
    assert not report["portable_cache_identity"]
    assert continuous.inspect_stage(completion) == report


def test_opaque_auxiliary_preserved_and_reported_partial_without_claiming_its_contents():
    class Auxiliary:
        pass
    _, chosen = selected()
    source = latent()
    source["external_auxiliary"] = Auxiliary()
    prefix, report = head(chosen, source)
    source["external_auxiliary"].counter = 2
    actual, completion, final = continuous.run_tail(guider(chosen), prefix)
    assert actual["external_auxiliary"] is source["external_auxiliary"]
    assert report["partial"] and final["partial"]
    assert "opaque_auxiliary_value_process_local_only" in final["warnings"]
    assert not final["portable_cache_identity"]
    assert continuous.inspect_stage(completion) == final


def test_real_CFG_positive_negative_complete_and_split_match_exactly():
    _, chosen = selected()
    positive, negative = conditioning(), conditioning()
    negative[0][0].sub_(.3)
    def selected_guider():
        result = CFGGuider(chosen)
        result.set_conds(positive, negative)
        result.set_cfg(2.0)
        return result
    noise = RandomNoise.execute(212104).result[0]
    controller = continuous._controller(selected_guider())
    expected = SamplerCustomAdvanced.execute(noise, selected_guider(), clocks.make_sampler(controller),
                                            clocks.ptd_sigmas(controller.recipe), latent()).result[0]
    prefix, head_report = continuous.run_head(noise, selected_guider(), latent())
    actual, _, report = continuous.run_tail(selected_guider(), prefix)
    for left, right in zip(actual["samples"].unbind(), expected["samples"].unbind()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert head_report["global_hops"] == [0, 1] and report["global_hops"] == [2, 3]
    assert controller.request.get() is None
    assert mm.get_torch_device().type == "cpu"
