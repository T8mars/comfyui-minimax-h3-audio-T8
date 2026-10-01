"""Probe integrity only; actual trained output evidence is separate."""
from copy import deepcopy

import pytest

from tools import run_modular_ltx_media_replay as replay


@pytest.mark.parametrize("route", ("ordinary", "identity"))
def test_replay_reuses_decode_and_audio_but_never_reruns_sampling(route):
    original = replay.prior.execution_graph(replay.prior.saved_graphs(route)["resume_ltx"][1],
        "resume_ltx", prompt="test", width=1024, height=576, receipt={"path": "x/manifest.json", "sha": "a" * 64})
    before = deepcopy(original)
    graph = replay.replay_graph(original)
    assert original == before
    for key in ("25", "29", "19", "51", "52"):
        assert graph[key] == original[key]
    for key in ("17", "18"):
        expected = deepcopy(original[key])
        expected["inputs"]["latent" if key == "17" else "audio"] = ["55", 0] if key == "17" else ["29", 1]
        assert graph[key] == expected
    assert not any(node["class_type"] in {"SamplerCustomAdvanced", "CLIPLoader", "UNETLoader", "VAEEncode", "LTXVLatentUpsampler"}
                   for node in graph.values())
    assert graph["20"]["class_type"] == graph["53"]["class_type"] == replay.WRITER
    assert graph["55"]["inputs"] == {"latent": "candidate.latent"}
    owned = replay.replay_graph(original, isolated=True)
    for key in graph.keys() - {"20", "53"}:
        assert owned[key] == graph[key]
    for key in ("20", "53"):
        assert owned[key]["class_type"] == replay.ISOLATED_WRITER
        assert owned[key]["inputs"]["video"] == graph[key]["inputs"]["video"]
        assert owned[key]["inputs"]["timeout_seconds"] == 300


def test_probe_requires_all_nodes_executed_not_merely_submission_success():
    graph = replay.read_graph()
    phase = {"terminal": {"type": "execution_success"},
             "events": [{"type": "executing", "node": key} for key in graph]}
    assert all(replay.phase_checks(phase, graph).values())
    phase["events"].pop()
    assert not replay.phase_checks(phase, graph)["every_explicit_node_executed"]
    phase["terminal"]["type"] = "execution_error"
    assert not replay.phase_checks(phase, graph)["terminal_success"]


def test_media_audit_rejects_external_or_uncommitted_paths(tmp_path):
    with pytest.raises(ValueError, match="completed owned"):
        replay.inspect_media(tmp_path / "outside/video.mp4", tmp_path / "output")
    with pytest.raises(ValueError, match="completed owned"):
        replay.inspect_media(tmp_path / "output/video.partial.mp4", tmp_path / "output")
