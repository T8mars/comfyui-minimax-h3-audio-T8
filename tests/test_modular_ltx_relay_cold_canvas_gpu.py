"""Cold canvas must visibly bind one checked source without reintroducing H3 prep."""

from copy import deepcopy
from argparse import Namespace
import json

import pytest

from tools import run_modular_ltx_relay_cold_canvas_gpu as cold
from tools import run_modular_ltx_relay_gpu as relay


RECEIPT = {"path": "source-123/manifest.json", "sha": "a" * 64}


@pytest.mark.parametrize("route", ["ordinary", "identity"])
@pytest.mark.parametrize("combined", [False, True])
def test_each_cold_canvas_route_only_edits_visible_checkpoint_and_effects(route, combined):
    _, original, _ = relay._saved(route, combined, relay.SAVED, variant="resume_ltx")
    before = deepcopy(original)
    prepared = cold.prepared_frontend(original, RECEIPT)
    assert original == before
    assert prepared["links"] == original["links"]
    old = {node["id"]: node for node in original["nodes"]}
    new = {node["id"]: node for node in prepared["nodes"]}
    changed_types = {new[node_id]["type"] for node_id in old if old[node_id] != new[node_id]}
    expected_types = {"MiniMaxH3LTXRGBSourceLoadEXPT8", "MiniMaxH3SaveVideoIsolatedEXPT8",
                      relay.builder.PLAN, relay.builder.APPLY}
    if route == "ordinary":  # The Identity saved graph already selects dense_reference.
        expected_types.add("MiniMaxH3SolEngineLTXRefinerSetupT8Advanced")
    assert changed_types == expected_types
    nodes = {node["type"]: node for node in prepared["nodes"]}
    assert nodes["MiniMaxH3LTXRGBSourceLoadEXPT8"]["widgets_values"] == [
        RECEIPT["path"], RECEIPT["sha"]]
    assert nodes[relay.builder.PLAN]["widgets_values"][:2] == [cold.full_canvas.GLOBAL,
                                                                cold.full_canvas.LOCAL]
    assert nodes[relay.builder.APPLY]["widgets_values"][0] == "apply_exp"
    assert nodes["MiniMaxH3SaveVideoIsolatedEXPT8"]["widgets_values"][0] == cold.MEDIA_PREFIX
    assert not {"LoadVideo", "VAEEncode", "LTXVLatentUpsampler"}.intersection(nodes)
    if combined:
        assert nodes["MiniMaxH3StageEAVConfigEXPT8"]["widgets_values"][0] == "report_only"
    setup = next(node for node in prepared["nodes"] if node["type"] in relay.builder.SETUPS)
    assert setup["widgets_values"][3 if route == "identity" else 1] == "dense_reference"
    assert setup["widgets_values"][6 if route == "identity" else 4] is False


def test_cold_canvas_rejects_existing_checkpoint_or_source_preparation():
    _, original, _ = relay._saved("ordinary", False, relay.SAVED, variant="resume_ltx")
    existing = deepcopy(original)
    next(node for node in existing["nodes"] if node["type"] ==
         "MiniMaxH3LTXRGBSourceLoadEXPT8")["widgets_values"][0] = "old/manifest.json"
    with pytest.raises(ValueError, match="already contains"):
        cold.prepared_frontend(existing, RECEIPT)
    source_node = deepcopy(original)
    source_node["nodes"].append({"id": 999, "type": "LoadVideo", "widgets_values": ["source.mp4"]})
    with pytest.raises(ValueError, match="source preparation"):
        cold.prepared_frontend(source_node, RECEIPT)


def test_cold_canvas_rejects_changed_verbose_or_enabled_eav():
    _, original, _ = relay._saved("identity", True, relay.SAVED, variant="resume_ltx")
    verbose = deepcopy(original)
    next(node for node in verbose["nodes"] if node["type"] in
         relay.builder.SETUPS)["widgets_values"][6] = True
    with pytest.raises(ValueError, match="widget layout"):
        cold.prepared_frontend(verbose, RECEIPT)
    eav = deepcopy(original)
    next(node for node in eav["nodes"] if node["type"] ==
         "MiniMaxH3StageEAVConfigEXPT8")["widgets_values"][0] = "apply_exp"
    with pytest.raises(ValueError, match="report-only"):
        cold.prepared_frontend(eav, RECEIPT)


def test_api_control_requires_unchanged_checkpoint_and_both_complete_media(tmp_path, monkeypatch):
    root = tmp_path / "artifacts/development/modular-ltx-relay-20260928/trained/control"
    root.mkdir(parents=True)
    files = {}
    for name in ("manifest.json", "state.safetensors", "full.mp4", "cold.mp4"):
        path = root / name
        path.write_bytes(name.encode())
        files[name] = {"path": str(path), "sha": cold.source.sha(path)}
    report = {"status": "full_cold_relay_media_mechanical_not_canvas_or_quality",
              "checks": {"all": True},
              "source_receipt": {"path": "source/manifest.json", "sha": files["manifest.json"]["sha"],
                                 "state_sha": files["state.safetensors"]["sha"],
                                 "manifest_file": files["manifest.json"]["path"],
                                 "state_file": files["state.safetensors"]["path"]},
              "media": {name: {**files[name + ".mp4"], "fully_decoded": True,
                               "decoded_sha": {"video": "v", "audio": "a"}}
                        for name in ("full", "cold")}}
    (root / "report.json").write_text(json.dumps(report), encoding="utf8")
    monkeypatch.setattr(cold, "ROOT", tmp_path)
    monkeypatch.setattr(cold, "API_BASELINES", {("ordinary", False): "control"})
    path, loaded, receipt = cold.control("ordinary", False)
    assert path == root / "report.json" and loaded == report and receipt == report["source_receipt"]
    (root / "state.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        cold.control("ordinary", False)
    (root / "state.safetensors").write_bytes(b"state.safetensors")
    report["media"]["cold"]["fully_decoded"] = False
    (root / "report.json").write_text(json.dumps(report), encoding="utf8")
    with pytest.raises(ValueError, match="media control changed"):
        cold.control("ordinary", False)


def test_preflight_allows_absent_user_core_but_records_its_state(tmp_path, monkeypatch):
    model = tmp_path / "models/diffusion_models/model.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")
    monkeypatch.setattr(cold.shared, "gpu_memory_mib", lambda: {"available": True, "free_mib": 13000})
    monkeypatch.setattr(cold.shared, "port_is_listening", lambda *_: False)
    monkeypatch.setattr(cold.full_canvas, "_port_owner", lambda port: None)
    monkeypatch.setattr(cold.shutil, "which", lambda tool: tool)
    args = Namespace(comfy_root=tmp_path, port=8958, min_free_vram_mib=12000, python=python)
    api = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": model.name}}}
    report = {"media": {"cold": {"streams": [
        {"codec_type": "video", "width": 1024, "height": 576, "nb_frames": "113"}
    ]}}}
    readiness = cold.preflight(args, api, report)
    assert readiness["ready"] is True
    assert readiness["user_8940_owner_before"] is None
    monkeypatch.setattr(cold.full_canvas, "_port_owner", lambda port: 8812)
    assert cold.preflight(args, api, report)["user_8940_owner_before"] == 8812


def test_cold_input_allows_only_core_empty_3d_directory(tmp_path):
    assert cold.no_original_video_mounted(tmp_path)
    auto_directory = tmp_path / "3d"
    auto_directory.mkdir()
    assert cold.no_original_video_mounted(tmp_path)
    (auto_directory / "source.mp4").write_bytes(b"video")
    assert not cold.no_original_video_mounted(tmp_path)
    (auto_directory / "source.mp4").unlink()
    (tmp_path / "source.mp4").write_bytes(b"video")
    assert not cold.no_original_video_mounted(tmp_path)
