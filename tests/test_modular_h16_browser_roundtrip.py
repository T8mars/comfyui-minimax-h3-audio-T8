"""Fail-closed native H16 browser normalization and source guards."""

from copy import deepcopy

import pytest

from tools.run_modular_h16_browser_roundtrip import (
    BRIDGE_FIELDS,
    BRIDGE_KIND,
    BRIDGE_PLACEHOLDERS,
    CASES,
    GEOMETRY_CASES,
    _bridge_connected_placeholders,
    _normalize_preview_any_input,
    _seed_control_after_generate,
    _sha,
)
from tools.serve_modular_h16_browser import PRIVATE, SOURCES, GEOMETRY_SOURCES


def _bridge_pair(node_id=53):
    fields = ("av_latent", *BRIDGE_FIELDS)
    pins = [{"name": name, "link": index}
            for index, name in enumerate(fields, start=1)]
    old = {"nodes": [{"id": node_id, "type": BRIDGE_KIND,
                      "inputs": pins, "widgets_values": []}]}
    new = deepcopy(old)
    new["nodes"][0]["widgets_values"] = BRIDGE_PLACEHOLDERS.copy()
    new["nodes"][0]["widgets_values_named"] = dict(zip(
        BRIDGE_FIELDS, BRIDGE_PLACEHOLDERS, strict=True))
    schema = {BRIDGE_KIND: {"info": {
        "input_order": {"required": list(fields)}}}}
    return old, new, schema


@pytest.mark.parametrize("node_id", [53, 90])
def test_h16_bridge_only_accepts_exact_connected_placeholders(node_id):
    old, new, schema = _bridge_pair(node_id)
    assert _bridge_connected_placeholders(old, new, schema, node_id) == BRIDGE_PLACEHOLDERS
    damaged = deepcopy(new)
    damaged["nodes"][0]["inputs"][1]["link"] = 999
    with pytest.raises(ValueError, match="connected-widget"):
        _bridge_connected_placeholders(old, damaged, schema, node_id)
    damaged = deepcopy(new)
    damaged["nodes"][0]["widgets_values"][1] = "not inert"
    with pytest.raises(ValueError, match="connected-widget"):
        _bridge_connected_placeholders(old, damaged, schema, node_id)


def _seed_pair():
    names = [f"setting_{index}" for index in range(18)] + ["restart_seed"]
    values = list(range(19))
    kind = "MiniMaxH3TwoPassDetailMixerT8Advanced"
    old = {"nodes": [{"id": 16, "type": kind, "widgets_values": values}]}
    new = deepcopy(old)
    new["nodes"][0]["widgets_values"] = [*values, "randomize"]
    new["nodes"][0]["widgets_values_named"] = dict(zip(
        [*names, "control_after_generate"], [*values, "randomize"], strict=True))
    schema = {kind: {"info": {"input_order": {
        "required": ["model", "av_latent", "refine_sigmas", *names]}}}}
    return old, new, schema


def test_h16_seed_frontend_control_rejects_changed_seed_or_mode():
    old, new, schema = _seed_pair()
    assert _seed_control_after_generate(old, new, schema) == ["randomize"]
    damaged = deepcopy(new)
    damaged["nodes"][0]["widgets_values"][18] += 1
    with pytest.raises(ValueError, match="seed control"):
        _seed_control_after_generate(old, damaged, schema)
    damaged = deepcopy(new)
    damaged["nodes"][0]["widgets_values"][-1] = "increment"
    with pytest.raises(ValueError, match="seed control"):
        _seed_control_after_generate(old, damaged, schema)


def _preview_pair():
    old = {"nodes": [{"id": 81, "type": "PreviewAny", "inputs": [
        {"name": "source", "type": "STRING", "link": 369}], "outputs": []}]}
    new = deepcopy(old)
    new["nodes"][0]["inputs"][0]["type"] = "*"
    new["nodes"][0]["outputs"] = [{"name": "STRING", "type": "STRING", "links": None}]
    schema = {"PreviewAny": {"info": {
        "input_order": {"required": ["source"]},
        "input": {"required": {"source": ["*", {}]}},
        "output": ["STRING"], "output_name": ["STRING"]}}}
    return old, new, schema


def test_h16_preview_normalization_requires_disconnected_output():
    old, new, schema = _preview_pair()
    assert _normalize_preview_any_input(old, new, schema) == [81]
    assert new["nodes"][0]["inputs"][0]["type"] == "STRING"
    assert new["nodes"][0]["outputs"] == []
    _, damaged, schema = _preview_pair()
    damaged["nodes"][0]["outputs"][0]["links"] = [7]
    with pytest.raises(ValueError, match="PreviewAny"):
        _normalize_preview_any_input(old, damaged, schema)


def test_h16_four_browser_sources_are_sha_pinned():
    for cases, sources in ((CASES, SOURCES), (GEOMETRY_CASES, GEOMETRY_SOURCES)):
        assert len(sources) == len(cases) == 4
        for (_, expected_sha, _, _, _), (directory, name, _) in zip(
                cases, sources, strict=True):
            assert _sha(PRIVATE / directory / name) == expected_sha
