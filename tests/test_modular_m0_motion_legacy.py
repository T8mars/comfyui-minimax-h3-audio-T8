"""The M0 Motion real-GPU probe must preserve frozen old/split sources."""
import hashlib

import pytest

from tools.run_modular_m0_motion_legacy_gpu import SOURCE_SHA, probe_graphs, source_files


@pytest.mark.parametrize("variant,stage", [("Fullclip", "18"), ("Windowed", "19")])
def test_legacy_motion_probe_keeps_old_graph_and_source_bound_split(variant, stage):
    old_path, split_path = source_files(variant)
    old_bytes, split_bytes = old_path.read_bytes(), split_path.read_bytes()
    old, new = probe_graphs(variant)
    assert hashlib.sha256(old_bytes).hexdigest() == SOURCE_SHA[variant]["legacy"]
    assert hashlib.sha256(split_bytes).hexdigest() == SOURCE_SHA[variant]["split"]
    assert old_path.read_bytes() == old_bytes and split_path.read_bytes() == split_bytes
    assert old["9"]["class_type"] == new["9"]["class_type"] == "SamplerCustomAdvanced"
    assert old[stage]["class_type"] == "SamplerCustomAdvanced"
    assert new[stage]["class_type"] == "MiniMaxH3StageSamplerEXPT8"
    assert old["5"]["inputs"] == new["5"]["inputs"]
    assert old["12"]["inputs"] == new["12"]["inputs"]
    assert old["6"]["inputs"] == new["6"]["inputs"]
    assert old["16" if variant == "Windowed" else "15"]["inputs"] == (
        new["16" if variant == "Windowed" else "15"]["inputs"])
    assert not any(node["class_type"].startswith("MiniMaxH3StageEAV")
                   for node in new.values())
    assert not any("PromptRelay" in node["class_type"] for node in new.values())


def test_legacy_motion_probe_refuses_changed_source_hash(monkeypatch):
    expected = SOURCE_SHA["Fullclip"]
    monkeypatch.setitem(SOURCE_SHA, "Fullclip", {**expected, "legacy": "0" * 64})
    with pytest.raises(ValueError, match="Frozen legacy Motion source changed"):
        probe_graphs("Fullclip")
