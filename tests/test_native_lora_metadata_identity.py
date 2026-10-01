"""Stock Core's inert LoRA metadata is content-bound, never an execution bypass."""
from copy import deepcopy

import comfy.sd
import comfy.supported_models
import pytest
import torch

from h3_audio_t8_pkg.long_video_dual_identity import stage_model_identity
from test_long_video_dual_identity import small_model


def test_actual_core_loader_attaches_metadata_without_turning_native_H3_nonportable():
    original = small_model()
    original.model.model_config = comfy.supported_models.MiniMaxH3({})
    baseline = stage_model_identity(original)
    metadata = {"format": "comfy", "training": "EMA B", "中文": "参数"}
    before = deepcopy(metadata)
    loaded, clip = comfy.sd.load_lora_for_models(original, None, {}, .75, 0., lora_metadata=metadata)
    assert clip is None and loaded is not original
    assert loaded.get_attachment("lora_metadata") == before
    assert original.get_attachment("lora_metadata") is None
    first = stage_model_identity(loaded)
    assert first.get("portable_cache_reuse", True) is True
    assert first["sha256"] != baseline["sha256"]
    assert first == stage_model_identity(loaded) and metadata == before
    assert stage_model_identity(original) == baseline


def test_core_and_T8_metadata_remain_independently_content_bound():
    model = small_model()
    original = stage_model_identity(model)
    model.set_attachments("lora_metadata", {"training": "Core"})
    first = stage_model_identity(model)
    model.set_attachments("t8_h3_lora_metadata", {"training": "T8"})
    second = stage_model_identity(model)
    model.set_attachments("lora_metadata", {"training": "changed"})
    third = stage_model_identity(model)
    assert len({entry["sha256"] for entry in (original, first, second, third)}) == 4
    model.remove_attachments("lora_metadata")
    model.remove_attachments("t8_h3_lora_metadata")
    assert stage_model_identity(model) == original


@pytest.mark.parametrize("metadata", (["data"], {"callback": lambda: None}, {"weight": torch.ones(1)},
                                     {"nested": {"training": "not a string"}}, {1: "bad key"}, False))
def test_core_metadata_cannot_hide_execution_or_non_string_state(metadata):
    model = small_model()
    model.set_attachments("lora_metadata", metadata)
    with pytest.raises(ValueError, match="Core LoRA metadata.*plain safetensors string map"):
        stage_model_identity(model)


def test_valid_metadata_does_not_admit_other_unknown_attachments_or_patches():
    model = small_model()
    model.set_attachments("lora_metadata", {"training": "Core"})
    model.set_attachments("foreign_runtime", {"claimed_safe": True})
    unknown = stage_model_identity(model)
    assert unknown["portable_cache_reuse"] is False and unknown["opaque_internal_state_verified"] is False
    model.remove_attachments("foreign_runtime")
    original = model.model.apply_model
    model.model.apply_model = lambda *args, **kwargs: original(*args, **kwargs)
    assert stage_model_identity(model)["portable_cache_reuse"] is False


def test_actual_core_native_weight_bytes_and_strength_still_change_identity():
    model = small_model()
    model.set_attachments("lora_metadata", {"training": "Core"})
    key = next(name for name, value in model.model_state_dict().items() if value.ndim == 2)
    weight = torch.zeros_like(model.model_state_dict()[key])
    model.add_patches({key: ("diff", (weight,))}, .75)
    first = stage_model_identity(model)
    weight.reshape(-1)[0] = .125
    second = stage_model_identity(model)
    assert second["sha256"] != first["sha256"]
    model.patches[key][0] = (.5, *model.patches[key][0][1:])
    assert stage_model_identity(model)["sha256"] != second["sha256"]
