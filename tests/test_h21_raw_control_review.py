"""Independent real CPU raw-graph asset isolation regressions; no inference."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import uuid
import wave

import pytest

from test_h21_raw_control_probe import CORE, probe
import test_h21_raw_control_probe as raw_tests

raw_config = raw_tests.raw_config


@pytest.fixture(scope="module")
def stable_source(tmp_path_factory):
    """Self-contained owned copy of selected runtime and seven controller tools.

    This is a file-reader contract fixture, not current whole-tree qualification.
    Concurrent R2/R3 development must not invalidate an unrelated file-read test.
    """
    target = tmp_path_factory.mktemp("immutable-test-source")
    current = Path(__file__).resolve().parents[1]
    identities = {}
    for source in probe.runtime_sources(current):
        relative = source.relative_to(current).as_posix()
        identities[relative] = probe.copy_exact(source, target / relative)
    tools = target / "tools"
    tools.mkdir(exist_ok=True)
    for name in probe.TOOLS:
        identities["tools/" + name] = probe.copy_exact(current / "tools" / name, tools / name)
    (target / "fixture-provenance.json").write_text(json.dumps({
        "source_root": str(current), "test_source": str(target), "source_files": identities,
        "scope": "real Core file-reader bindings only; no model execution or single-time full-tree qualification",
    }, indent=2), encoding="utf-8")
    return target


@pytest.mark.parametrize("kind", ["LoadImage", "LoadImageMask"])
def test_real_core_rejects_external_absolute_image_without_our_extra_binding_guard(raw_config, kind):
    import folder_paths
    import nodes

    root, _, _, _ = raw_config
    source = root / "owned.png"
    before = probe.file_identity(source)
    # Actual current Core's realpath containment rejects this existing file.
    # This must not be misreported as an executable unowned-file read.
    owned = root / (kind + "-core-input")
    owned.mkdir()
    previous = folder_paths.get_input_directory()
    try:
        folder_paths.set_input_directory(str(owned))
        assert "Invalid image file" in nodes.NODE_CLASS_MAPPINGS[kind].VALIDATE_INPUTS(str(source))
    finally:
        folder_paths.set_input_directory(previous)
    assert probe.file_identity(source) == before


@pytest.mark.parametrize("kind", ["LoadImage", "LoadImageMask"])
def test_second_image_consumer_must_explicitly_bind_even_existing_frozen_asset(raw_config, stable_source, kind):
    """A provenance-slot gap, not a claim of external file escape."""
    root, _, original_config, original_graph = raw_config
    config, graph = deepcopy(original_config), deepcopy(original_graph)
    asset = next(iter(config["assets"].values()))
    # Node4 remains explicitly bound/copied. Node10 reads its private literal
    # path without declaring node10/image in that same resource's bindings.
    graph["10"] = {"class_type": kind, "inputs": {"image": asset["private_relative"]}}
    if kind == "LoadImageMask":
        graph["10"]["inputs"]["channel"] = "alpha"
    graph["11"] = {"class_type": "PreviewAny", "inputs": {"source": ["10", 0]}}
    api = root / (kind + "-second-unbound-api.json")
    api.write_text(json.dumps(graph), encoding="utf-8")
    config["raw_api_cases"][0].update(
        api=str(api), api_sha256=probe.file_identity(api)["sha256"]
    )
    config_file = root / (kind + "-second-unbound-config.json")
    config_file.write_text(json.dumps(config), encoding="utf-8")
    run = root / (kind + "-second-unbound-prepared")
    with pytest.raises(RuntimeError, match="CPU graph preparation failed"):
        probe.prepare(config_file, run, CORE, source_root=stable_source, python=sys.executable)
    assert "explicit private asset binding" in (run / "prepare.stderr.log").read_text(
        encoding="utf-8"
    )
    assert not (run / "manifest.json").exists()
    assert not (run / "attempts").exists()


@pytest.mark.parametrize("second_unbound", [False, True])
def test_actual_core_LoadAudio_registration_and_explicit_private_consumers(raw_config, stable_source, second_unbound):
    import folder_paths
    import torch
    from comfy_extras.nodes_audio import LoadAudio

    root, _, original_config, original_graph = raw_config
    config, graph = deepcopy(original_config), deepcopy(original_graph)
    source = root / ("audio-" + str(second_unbound) + ".wav")
    with wave.open(str(source), "wb") as outgoing:
        outgoing.setnchannels(1)
        outgoing.setsampwidth(2)
        outgoing.setframerate(32000)
        outgoing.writeframes(b"\x01\x00" * 128)
    # Execute the real Core audio reader on CPU as well as its actual graph schema.
    previous = folder_paths.get_input_directory()
    try:
        folder_paths.set_input_directory(str(root))
        audio, = LoadAudio.execute(source.name)
        assert audio["sample_rate"] == 32000
        assert audio["waveform"].shape == (1, 1, 128)
        assert torch.isfinite(audio["waveform"]).all() and not torch.cuda.is_initialized()
    finally:
        folder_paths.set_input_directory(previous)
    aid = str(uuid.uuid4())
    relative = "raw/" + aid + "/source.wav"
    config["assets"][aid] = {"source": str(source), "sha256": probe.file_identity(source)["sha256"],
                              "private_relative": relative, "nodes": [{"node": "10", "input": "audio"}]}
    graph["10"] = {"class_type": "LoadAudio", "inputs": {"audio": source.name}}
    graph["11"] = {"class_type": "PreviewAny", "inputs": {"source": ["10", 0]}}
    if second_unbound:
        graph["12"] = {"class_type": "LoadAudio", "inputs": {"audio": relative}}
        graph["13"] = {"class_type": "PreviewAny", "inputs": {"source": ["12", 0]}}
    api = root / ("audio-" + str(second_unbound) + "-api.json")
    api.write_text(json.dumps(graph), encoding="utf-8")
    config["raw_api_cases"][0].update(api=str(api), api_sha256=probe.file_identity(api)["sha256"])
    config_file = root / ("audio-" + str(second_unbound) + "-config.json")
    config_file.write_text(json.dumps(config), encoding="utf-8")
    run = root / ("audio-" + str(second_unbound) + "-prepared")
    if second_unbound:
        with pytest.raises(RuntimeError, match="CPU graph preparation failed"):
            probe.prepare(config_file, run, CORE, source_root=stable_source, python=sys.executable)
        assert "explicit private asset binding: 12/audio" in (run / "prepare.stderr.log").read_text(encoding="utf-8")
        assert not (run / "manifest.json").exists()
    else:
        plan = probe.prepare(config_file, run, CORE, source_root=stable_source, python=sys.executable)
        assert probe.read_json(run / plan["cases"][0]["validation"])["node_errors"] == {}
        assert probe.read_json(run / plan["cases"][0]["base_prompt"])["10"]["inputs"]["audio"] == relative
        assert (run / "input" / relative).read_bytes() == source.read_bytes()
        assert not plan["model_execution"] and not plan["gpu_started"]
