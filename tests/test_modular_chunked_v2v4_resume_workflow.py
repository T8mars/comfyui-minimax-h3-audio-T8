"""Chunked full-clip freeze/resume drafts keep frontend and API execution aligned."""
from collections import Counter
from copy import deepcopy
import hashlib

import pytest

from tools.build_modular_chunked_v2v4_resume_workflow import (
    AUDIT_TYPE, NATIVE_LOAD, NATIVE_SAVE, PASS_TYPE, PREPARE_TYPE,
    SOURCE_TYPE, SOURCES, _assert_wires_match, _frontend_source, build_pair,
)


def _only(graph, kind):
    matches = [node for node in graph["nodes"] if node["type"] == kind]
    assert len(matches) == 1
    return matches[0]


def _origin(graph, node, name):
    pin = next(pin for pin in node["inputs"] if pin["name"] == name)
    link = next(link for link in graph["links"] if link[0] == pin["link"])
    return link[1:3]


def _assert_links(graph):
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    assert len(nodes) == len(graph["nodes"])
    assert len(links) == len(graph["links"])
    for link_id, source, source_slot, target, target_slot, _ in graph["links"]:
        assert link_id in nodes[source]["outputs"][source_slot]["links"]
        assert nodes[target]["inputs"][target_slot]["link"] == link_id
    for node in graph["nodes"]:
        for pin in node["inputs"]:
            assert pin["link"] is None or pin["link"] in links


@pytest.mark.parametrize("variant", ("v2", "v3", "v4"))
def test_freeze_resume_retain_independent_high_effects_and_remove_low_sampler(variant):
    path, digest = SOURCES[variant]
    before = path.read_bytes()
    assert hashlib.sha256(before).hexdigest() == digest
    source = _frontend_source(variant)
    frozen_source = deepcopy(source)
    (freeze, freeze_api), (resume, resume_api) = build_pair(variant)
    assert source == frozen_source
    assert path.read_bytes() == before
    for frontend, api in ((freeze, freeze_api), (resume, resume_api)):
        _assert_links(frontend)
        _assert_wires_match(frontend, api)
        assert {str(node["id"]) for node in frontend["nodes"]} == set(api)
        for node in frontend["nodes"]:
            if node["type"] == "VHS_VideoCombine":
                names = ("frame_rate", "loop_count", "filename_prefix", "format",
                         "pingpong", "save_output")
            else:
                names = tuple(name for name, value in api[str(node["id"])]["inputs"].items()
                              if not (isinstance(value, list) and len(value) == 2 and
                                      isinstance(value[0], str) and value[0] in api and
                                      isinstance(value[1], int)))
            values = list(node.get("widgets_values") or [])
            if node["type"] in ("MiniMaxH3AudioConditioningT8", "RandomNoise"):
                values.pop()
            assert values == [api[str(node["id"])]["inputs"][name] for name in names]

    freeze_kinds = Counter(node["type"] for node in freeze["nodes"])
    resume_kinds = Counter(node["type"] for node in resume["nodes"])
    assert freeze_kinds["SamplerCustomAdvanced"] == freeze_kinds[NATIVE_SAVE] == 1
    assert freeze_kinds[PASS_TYPE] == freeze_kinds[AUDIT_TYPE] == 0
    assert resume_kinds["SamplerCustomAdvanced"] == resume_kinds[NATIVE_SAVE] == 0
    assert resume_kinds[NATIVE_LOAD] == resume_kinds[PASS_TYPE] == 1
    assert resume_kinds["MiniMaxH3ChunkedPass2RelayBindEXPT8"] == 1
    assert resume_kinds["MiniMaxH3ChunkedPass2EAVApplyEXPT8"] == 1
    assert resume_kinds[AUDIT_TYPE] == resume_kinds["VHS_VideoCombine"] == 1
    assert _origin(freeze, _only(freeze, NATIVE_SAVE), "av_latent") == [12, 1]
    assert _only(freeze, NATIVE_SAVE)["widgets_values"][2] is False
    for kind in (SOURCE_TYPE, PREPARE_TYPE):
        assert _origin(resume, _only(resume, kind), "first_pass_latent") == [
            _only(resume, NATIVE_LOAD)["id"], 0]
    assert _only(resume, NATIVE_LOAD)["widgets_values"] == ["", "", "", 8]
    assert "12" not in resume_api
    assert resume_api[str(_only(resume, NATIVE_LOAD)["id"])]["inputs"] == {
        "checkpoint_path": "", "expected_manifest_json": "",
        "expected_file_sha256": "", "hash_chunk_megabytes": 8,
    }
    if variant == "v2":
        assert freeze_api["7"]["inputs"]["width"] == 416
        assert freeze_api["7"]["inputs"]["length"] == 22
        assert resume_api["14"]["inputs"]["target_width"] == 832
        assert resume_api["29"]["inputs"]["length"] == 22
    else:
        assert freeze_api["8"]["inputs"]["width"] == 576
        assert freeze_api["8"]["inputs"]["length"] == 124
        assert resume_api["14"]["inputs"]["target_width"] == 1152


def test_frontend_api_wire_parity_rejects_divergence():
    (freeze, api), _ = build_pair("v2")
    changed = deepcopy(api)
    save = _only(freeze, NATIVE_SAVE)
    changed[str(save["id"])]["inputs"]["av_latent"] = ["12", 0]
    with pytest.raises(ValueError, match="wire mismatch"):
        _assert_wires_match(freeze, changed)


def test_source_hash_pin_rejects_changed_candidate(monkeypatch, tmp_path):
    path, digest = SOURCES["v2"]
    altered = tmp_path / path.name
    altered.write_bytes(path.read_bytes() + b"\n")
    monkeypatch.setitem(SOURCES, "v2", (altered, digest))
    with pytest.raises(ValueError, match="source candidate changed"):
        build_pair("v2")
