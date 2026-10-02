"""Exact additive template diff; scope is saved graphs, not media acceptance."""
from copy import deepcopy
import hashlib
import json

import pytest

from tools import build_reviewed_audio_followup_workflows as builder


@pytest.mark.parametrize("filename", sorted(builder.SOURCES))
def test_saved_template_changes_only_the_approved_explicit_policy(filename):
    relative, expected = builder.SOURCES[filename]
    path = builder.WORKFLOWS / relative
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == expected
    old = json.loads(content)
    new = builder.generated()[builder.DESTINATION / filename]
    assert json.loads((builder.DESTINATION / filename).read_text(encoding="utf8")) == new
    assert new["id"] != old["id"]
    restored = deepcopy(new)
    restored["id"] = old["id"]
    restored["extra"] = old["extra"]
    old_by_id = {node["id"]: node for node in old["nodes"]}
    for node in restored["nodes"]:
        original = old_by_id[node["id"]]
        if node["type"] == "MarkdownNote":
            assert builder.COMMON_NOTE in node["widgets_values"][0]
            assert "不能删除原audio rebase" not in node["widgets_values"][0]
            node["widgets_values"] = original["widgets_values"]
        elif node["type"] == builder.JOINT_RF:
            assert node["widgets_values"] == original["widgets_values"]
            node["type"] = original["type"]
            node["title"] = original["title"]
            node["properties"] = original["properties"]
        elif node["type"] == builder.AUDIO_PLAN:
            assert node["widgets_values"] == [4, .35, 2608290022]
            node["widgets_values"] = original["widgets_values"]
            node["title"] = original["title"]
    assert restored == old  # includes ALL wires, masks, effects and guards
    assert path.read_bytes() == content


@pytest.mark.parametrize("filename", sorted(builder.SOURCES))
def test_cold_templates_keep_explicit_placeholders_and_no_hidden_first_pass(filename):
    graph = builder.generated()[builder.DESTINATION / filename]
    kinds = [node["type"] for node in graph["nodes"]]
    if filename.startswith(("C1_", "C2_")):
        assert kinds.count(builder.JOINT_RF) == 1
        assert builder.OLD_RF not in kinds
        assert "MiniMaxH3StageEAVApplyEXPT8" in kinds
        assert "MiniMaxH3PromptRelayPlanT8Advanced" in kinds
        if "Resume" in filename:
            load = builder.single(graph, "MiniMaxH3StageLoadEXPT8")
            assert load["widgets_values"] == [
                "REPLACE_WITH_RF_BASE_PATH/manifest.json", "0" * 64, "rf_base"]
            assert "MiniMaxH3RFBaseStageSetupEXPT8" not in kinds
        else:
            assert kinds.count("MiniMaxH3RFBaseStageSetupEXPT8") == 1
    elif "Resume" in filename:
        load = builder.single(graph, "MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8")
        assert load["widgets_values"][:3] == ["", "PASTE_EXACT_SAVE_MANIFEST_JSON", "0" * 64]
        assert kinds.count("SamplerCustomAdvanced") == 1
        assert "MiniMaxH3StageEAVApplyEXPT8" in kinds
        gate = builder.single(graph, "MiniMaxH3AudioRefineQualityGateT8Advanced")
        assert gate["widgets_values"][0] is False
    else:
        assert builder.AUDIO_PLAN not in kinds
        save = builder.single(graph, "MiniMaxH3NativeLatentCheckpointSaveT8Advanced")
        assert save["widgets_values"][2] is False


def test_template_writer_does_not_overwrite_modified_graph_or_partially_write(tmp_path):
    changed, missing = tmp_path / "changed.json", tmp_path / "missing.json"
    changed.write_bytes(b"user-modified")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        builder.write_missing({missing: {}, changed: {}}, write=True)
    assert changed.read_bytes() == b"user-modified"
    assert not missing.exists()


def test_b8_rejects_other_strength_steps_or_seed():
    filename = "B8_LongRelay_035_Resume_Audio_EAV_EXP.json"
    relative, _ = builder.SOURCES[filename]
    original = json.loads((builder.WORKFLOWS / relative).read_text(encoding="utf8"))
    for changed in ([8, .50, 2608290022], [4, .35, 2608290022], [4, .50, 1]):
        graph = deepcopy(original)
        builder.single(graph, builder.AUDIO_PLAN)["widgets_values"] = changed
        with pytest.raises(ValueError, match="contract changed"):
            builder.transform(graph, filename)
