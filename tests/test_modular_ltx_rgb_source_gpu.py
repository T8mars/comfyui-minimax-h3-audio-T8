"""Integrity of the trained probe; these tests do not claim trained inference."""
from copy import deepcopy
import json

import pytest

from tools import run_modular_ltx_rgb_source_gpu as probe


@pytest.mark.parametrize("route", ("ordinary", "identity"))
@pytest.mark.parametrize("variant", probe.VARIANTS)
def test_saved_graph_probe_changes_only_explicit_test_settings(route, variant):
    saved = probe.saved_graphs(route)
    original = saved[variant][1]
    frozen = deepcopy(original)
    graph = probe.execution_graph(original, variant, prompt="test", width=1024, height=576,
                                   receipt={"path": "one/manifest.json", "sha": "a" * 64})
    permitted = {"1": {"file"}, "3": {"target_width", "target_height"}, "10": {"attention_backend"},
                 "12": {"text"}, "20": {"filename_prefix"},
                 "29": {"artifact_path", "artifact_sha256", "confirm_save", "filename_prefix"}}
    for key, node in original.items():
        assert graph[key]["class_type"] == node["class_type"]
        changed = {k for k in node["inputs"].keys() | graph[key]["inputs"].keys()
                   if node["inputs"].get(k) != graph[key]["inputs"].get(k)}
        assert changed <= permitted.get(key, set())
    assert original == frozen
    if variant != "freeze_source":
        assert graph["16"] == original["16"]
        assert graph["9"]["inputs"]["strength_model"] == .8
        assert graph["26"] == original["26"]
        assert graph["51"]["inputs"]["audio"] == ["29", 1]
    if variant == "resume_ltx":
        assert not any(graph.get(str(i)) for i in range(1, 8))


@pytest.mark.parametrize("variant", probe.VARIANTS)
def test_phase_checks_require_real_node_execution_and_sampler_callbacks(variant):
    ids = {"29", "1", "3", "5", "7"} if variant == "freeze_source" else {"29", "16", "28", "17", "20", "50"}
    phase = {"terminal": {"type": "execution_success"}, "events": [
        {"type": "executing", "node": key} for key in ids]}
    if variant != "freeze_source":
        phase["events"] += [{"type": "progress", "node": "16"}] * 3
    assert all(probe.phase_checks(phase, variant).values())
    phase["events"] = []
    assert not all(probe.phase_checks(phase, variant).values())
    phase["terminal"]["type"] = "execution_error"
    assert not probe.phase_checks(phase, variant)["terminal_success"]


def test_cold_copy_is_sha_checked_and_does_not_overwrite(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    state = source / "state.safetensors"
    state.write_bytes(b"inert test data")
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps({"test": True}), encoding="utf8")
    receipt = {"path": "one/manifest.json", "sha": probe.sha(manifest), "state_sha": probe.sha(state),
               "manifest_file": str(manifest), "state_file": str(state)}
    owned = tmp_path / "owned"
    probe.copy_source(receipt, owned)
    assert probe.sha(owned / "output/MiniMaxH3/ltx_rgb_sources/one/state.safetensors") == receipt["state_sha"]
    with pytest.raises(ValueError, match="must be new"):
        probe.copy_source(receipt, owned)
    with pytest.raises(ValueError, match="must be new"):
        probe.copy_source(dict(receipt, path="../../../../escape/manifest.json"), owned)
    state.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed before"):
        probe.copy_source(receipt, tmp_path / "second")


def test_invalid_geometry_or_missing_receipt_cannot_start_a_graph():
    original = probe.saved_graphs("ordinary")["resume_ltx"][1]
    with pytest.raises(ValueError, match="32-aligned"):
        probe.execution_graph(original, "resume_ltx", prompt="test", width=1023, height=576)
    with pytest.raises(ValueError, match="explicit source receipt"):
        probe.execution_graph(original, "resume_ltx", prompt="test", width=1024, height=576)
    with pytest.raises(ValueError, match="explicitly saved"):
        probe.saved_graphs("ordinary", isolated_media=True)


def test_media_qualification_rejects_decode_failures_and_wrong_geometry():
    prep = {"source": {"width": 512, "height": 288}, "target": {"width": 1024, "height": 576, "frames": 113},
            "fps": 24., "output_duration_seconds": 113 / 24}
    files = {key: {"fully_decoded": True, "decoded_sha": {"video": "v", "audio": "a"},
        "streams": [{"codec_type": "video", "width": 512 if key.startswith("source_") else 1024,
                     "height": 288 if key.startswith("source_") else 576, "nb_frames": "113",
                     "avg_frame_rate": "24/1", "duration": "4.708333"}]}
        for key in ("full", "cold", "source_full", "source_cold")}
    assert all(probe.media_checks(files, prep).values())
    files["full"]["fully_decoded"] = files["cold"]["fully_decoded"] = False
    files["full"]["decoded_sha"]["video"] = files["cold"]["decoded_sha"]["video"] = None
    checks = probe.media_checks(files, prep)
    assert not checks["exact_full_cold_media"] and checks["original_audio_bypassed"]
    files["cold"]["streams"][0]["nb_frames"] = "112"
    assert not probe.media_checks(files, prep)["cold_geometry_duration"]
