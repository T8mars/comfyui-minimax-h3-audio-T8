"""Explicit filesystem drift review is not schema or model qualification."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest

from legacy_schema_review import assert_reviewed_legacy_schemas


UPSCALER_FILE_MENUS = [
    ("MiniMaxH3ChunkedTwoPassPlanT8Advanced", "model_name"),
    ("MiniMaxH3ChunkedTwoPassGlobalNoisePlanT8Advanced", "model_name"),
    ("MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced", "model_name"),
    ("MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced", "model_name"),
    ("MiniMaxH3ProgressiveSamplerEXPT8", "upscaler_model"),
    ("MiniMaxH3DualModelLongVideoEXPT8", "upscaler_model"),
    ("MiniMaxH3ProgressiveLongVideoEXPT8", "upscaler_model"),
    ("MiniMaxH3FastH3V2DualModelLongVideoEXPT8", "upscaler_model"),
    ("MiniMaxH3AvatarProgressiveEXPT8", "upscaler_model"),
    ("MiniMaxH3HyperFlowLongVideoEXPT8", "upscaler_model"),
]


@pytest.mark.parametrize("change", ["valid", "delete_positive", "delete_negative", "positive_type", "negative_type",
                                   "positive_metadata", "old_negative", "old_order", "new_order", "interposed", "foreign"])
def test_named_D_HIGH_conditioning_append_does_not_hide_any_other_socket_change(tmp_path, change):
    node_id = "MiniMaxH3HyperFlowSplitT8Advanced"
    old = {"input": {"required": {"positive": ["CONDITIONING", {}]},
                     "optional": {"negative": ["CONDITIONING", {}]}},
           "input_order": {"required": ["positive"], "optional": ["negative"]},
           "output": ["LATENT", "STRING"]}
    new = deepcopy(old)
    new["input"]["optional"].update(positive_high=["CONDITIONING", {}], negative_high=["CONDITIONING", {}])
    new["input_order"]["optional"] += ["positive_high", "negative_high"]
    if change == "delete_positive":
        del new["input"]["optional"]["positive_high"]
    elif change == "delete_negative":
        del new["input"]["optional"]["negative_high"]
    elif change == "positive_type":
        new["input"]["optional"]["positive_high"][0] = "MODEL"
    elif change == "negative_type":
        new["input"]["optional"]["negative_high"][0] = "LATENT"
    elif change == "positive_metadata":
        new["input"]["optional"]["positive_high"][1]["default"] = "unexpected"
    elif change == "old_negative":
        new["input"]["optional"]["negative"][0] = "MODEL"
    elif change == "old_order":
        old["input_order"]["optional"].insert(0, "not_old_negative")
    elif change == "new_order":
        new["input_order"]["optional"].reverse()
    elif change == "interposed":
        new["input"]["optional"]["other"] = ["CONDITIONING", {}]
        new["input_order"]["optional"].insert(1, "other")
    elif change == "foreign":
        node_id = "UnreviewedConditioningNode"
    entries, info = [{"id": node_id, "info": old}], {node_id: new}
    before = deepcopy((entries, info))
    if change == "valid":
        assert assert_reviewed_legacy_schemas(entries, info, tmp_path) == [
            {"id": node_id, "review": "exact_D_HIGH_conditioning_append",
             "optional_appended": ["positive_high", "negative_high"]}]
    else:
        with pytest.raises(AssertionError):
            assert_reviewed_legacy_schemas(entries, info, tmp_path)
    assert (entries, info) == before


@pytest.mark.parametrize("node_id,field", UPSCALER_FILE_MENUS)
@pytest.mark.parametrize("change", ["valid", "missing", "default", "socket", "reordered", "sentinel_removed", "static_enum"])
def test_only_explicit_upscaler_owner_file_menu_drift_is_reviewed(tmp_path, node_id, field, change):
    root = tmp_path / "latent_upscale_models"
    root.mkdir()
    for name in ("first.safetensors", "second.safetensors", "new.safetensors"):
        if change != "missing" or name != "new.safetensors":
            (root / name).write_bytes(b"Physical eligibility fixture, not trained weight certification")
    old = {"input": {"required": {
        field: ["COMBO", {"options": ["first.safetensors", "second.safetensors"]}],
        "precision": ["COMBO", {"options": ["fp16", "bf16"], "default": "fp16"}],
    }}, "input_order": {"required": [field, "precision"]}, "output": ["LATENT"]}
    new = deepcopy(old)
    new["input"]["required"][field][1]["options"].insert(1, "new.safetensors")
    if change == "default":
        new["input"]["required"][field][1]["default"] = "new.safetensors"
    elif change == "socket":
        new["input"]["required"][field][0] = "LATENT"
    elif change == "reordered":
        new["input"]["required"][field][1]["options"].reverse()
    elif change == "sentinel_removed":
        old["input"]["required"][field][1]["options"].insert(0, "auto")
    elif change == "static_enum":
        new["input"]["required"]["precision"][1]["options"].append("fp32")
    entries, info = [{"id": node_id, "info": old}], {node_id: new}
    before = deepcopy((entries, info))
    if change == "valid":
        changes = assert_reviewed_legacy_schemas(entries, info, tmp_path, review_missing_files=True)
        assert changes == [{"id": node_id, "field": field, "category": "latent_upscale_models",
                            "added": ["new.safetensors"], "removed_absent_files": [],
                            "review": "explicit_filesystem_drift_not_schema_or_model_certificate"}]
    else:
        with pytest.raises((AssertionError, ValueError)):
            assert_reviewed_legacy_schemas(entries, info, tmp_path, review_missing_files=True)
    assert (entries, info) == before


@pytest.mark.parametrize("change", ["absent", "present", "sentinel", "reordered", "default",
                                   "escape", "unknown_owner", "symlink", "empty_added"])
@pytest.mark.parametrize("node_id,category,field", [
    ("MiniMaxH3HybridPairInspectorT8Advanced", "diffusion_models", "quality_base"),
    ("MiniMaxH3LearnedLatentUpscaleT8Advanced", "latent_upscale_models", "model_name"),
])
def test_missing_asset_review_preserves_present_choices_order_defaults_and_owner(tmp_path, change, node_id, category, field):
    root = tmp_path / category
    root.mkdir()
    for name in ("old.safetensors", "second.safetensors"):
        (root / name).write_bytes(b"Actual file-presence fixture, not model content qualification")
    retired = "retired.safetensors"
    if change == "sentinel":
        retired = "auto"
    elif change == "escape":
        retired = "../outside.safetensors"
    elif change == "present":
        (root / retired).write_bytes(b"Still installed: its menu removal must fail")
    old = {"input": {"required": {field: ["COMBO", {
        "options": [retired, "old.safetensors", "second.safetensors"],
        "default": "old.safetensors", "tooltip": "Unchanged metadata",
    }]}}, "output": ["MODEL"]}
    new = deepcopy(old)
    new["input"]["required"][field][1]["options"] = ["old.safetensors", "new.safetensors", "second.safetensors"]
    if change == "symlink":
        target = tmp_path / "external-real-model.bin"
        target.write_bytes(b"Explicit logical model-root symlink fixture")
        (root / "new.safetensors").symlink_to(target)
    else:
        (root / "new.safetensors").write_bytes(b"" if change == "empty_added" else b"Real new file")
    if change == "reordered":
        new["input"]["required"][field][1]["options"].reverse()
    elif change == "default":
        new["input"]["required"][field][1]["default"] = "new.safetensors"
    elif change == "unknown_owner":
        node_id = "UnreviewedStaticChoiceNode"
    entries, live = [{"id": node_id, "info": old}], {node_id: new}
    original = deepcopy((entries, live))
    if change in ("absent", "symlink"):
        report = assert_reviewed_legacy_schemas(entries, live, tmp_path, review_missing_files=True)
        assert report == [{"id": node_id, "field": field, "category": category,
                           "added": ["new.safetensors"], "removed_absent_files": [retired],
                           "review": "explicit_filesystem_drift_not_schema_or_model_certificate"}]
    else:
        with pytest.raises((AssertionError, ValueError)):
            assert_reviewed_legacy_schemas(entries, live, tmp_path, review_missing_files=True)
    assert (entries, live) == original
    # This module's independent --noconftest scope is filesystem/metadata only.
    if "h3_audio_t8_pkg" not in sys.modules:
        assert "torch" not in sys.modules


def test_strict_default_still_rejects_even_genuinely_removed_model_file(tmp_path):
    node_id = "MiniMaxH3HybridPairInspectorT8Advanced"
    old = {"input": {"required": {"quality_base": ["COMBO", {"options": ["gone.safetensors"]}]}}}
    new = deepcopy(old)
    new["input"]["required"]["quality_base"][1]["options"] = []
    with pytest.raises(AssertionError):
        assert_reviewed_legacy_schemas([{"id": node_id, "info": old}], {node_id: new}, tmp_path)


@pytest.mark.parametrize("field,category,token", [("unet_name", "diffusion_models", "minimax_h3"),
                                                ("lora_name", "loras", "raven")])
@pytest.mark.parametrize("change", ["absent", "still_present", "source_changed", "wrong_default"])
def test_only_unchanged_source_computed_default_may_follow_actual_missing_file(tmp_path, field, category, token, change):
    root = tmp_path / category
    root.mkdir()
    retired, remaining = f"old_{token}.safetensors", f"new_{token}.safetensors"
    (root / remaining).write_bytes(b"Actual file, not a model certificate")
    if change == "still_present":
        (root / retired).write_bytes(b"Cannot remove an installed option/default")
    node_id = "MiniMaxH3RavenGuardedLoaderT8Advanced"
    old = {"input": {"required": {field: ["COMBO", {"options": [retired, remaining], "default": retired}]}}}
    live = deepcopy(old)
    live["input"]["required"][field][1].update(options=[remaining], default=remaining)
    # This pure test is also copied to a dedicated path outside the plugin, so
    # bind the fixture to the helper's actual repository rather than test cwd.
    import legacy_schema_review
    source = Path(legacy_schema_review.__file__).resolve().parents[1] / "h3_t8/nodes_raven_streaming_advanced.py"
    entry = {"id": node_id, "info": old, "source": "h3_t8/nodes_raven_streaming_advanced.py",
             "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    if change == "source_changed":
        entry["source_sha256"] = "0" * 64
    elif change == "wrong_default":
        live["input"]["required"][field][1]["default"] = "unselected.safetensors"
    before = deepcopy(([entry], {node_id: live}))
    if change == "absent":
        reviewed = assert_reviewed_legacy_schemas([entry], {node_id: live}, tmp_path, review_missing_files=True)
        assert reviewed[0]["default_from_unchanged_source_after_file_removal"] is True
    else:
        with pytest.raises(AssertionError):
            assert_reviewed_legacy_schemas([entry], {node_id: live}, tmp_path, review_missing_files=True)
    assert ([entry], {node_id: live}) == before
