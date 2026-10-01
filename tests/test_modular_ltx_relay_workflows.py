"""Saved opt-in Relay graphs preserve the old LTX sampling and audio routes."""
import hashlib
from copy import deepcopy

import pytest

from tools import build_modular_ltx_eav_workflows as eav
from tools import build_modular_ltx_relay_workflows as relay
from tools import build_modular_ltx_rgb_source_workflows as source
from tools.build_modular_ltx_rgb_workflows import sources as old_sources


def _edge(graph, kind, field):
    node = next(node for node in graph["nodes"] if node["type"] == kind)
    item = next(item for item in node["inputs"] if item["name"] == field)
    link = next(link for link in graph["links"] if link[0] == item["link"])
    return tuple(link[1:3])


@pytest.mark.parametrize("name,graph", list(relay.generated().items()))
def test_eight_additive_relay_graphs_keep_native_sampling_audio_and_source(name, graph):
    combined = name.endswith("_external_eav_relay")
    base = (eav.generated()[name.removesuffix("_relay")] if combined else
            source.generated(isolated_media=True)[name.removesuffix("_external_relay")])
    old_nodes = {node["id"]: node for node in base["nodes"]}
    nodes = {node["type"]: node for node in graph["nodes"]}
    old_positive = next(node for node in base["nodes"] if node["type"] == "CLIPTextEncode"
                        and _edge(base, "LTXVConditioning", "positive") == (node["id"], 0))
    assert {node["id"] for node in graph["nodes"]} - old_nodes.keys() == {
        nodes[kind]["id"] for kind in (relay.PLAN, relay.ENCODE, relay.APPLY, relay.RELAY_AUDIT)}
    assert old_nodes.keys() - {node["id"] for node in graph["nodes"]} == {old_positive["id"]}
    for node in graph["nodes"]:
        if node["id"] in old_nodes:
            assert node["type"] == old_nodes[node["id"]]["type"]
            assert node["widgets_values"] == old_nodes[node["id"]]["widgets_values"]
    store = nodes[source.LOAD if "resume_ltx" in name else source.SAVE]
    setup = next(nodes[kind] for kind in relay.SETUPS if kind in nodes)
    upstream_model = (nodes[eav.APPLY]["id"], 0) if combined else (setup["id"], 0)
    upstream_candidate = ((nodes[eav.EFFECT_AUDIT]["id"], 0) if combined else
                          (nodes[relay.AUDIT]["id"], 0))
    assert _edge(graph, relay.PLAN, "ltx_latent") == (store["id"], 4)
    assert _edge(graph, relay.ENCODE, "clip") == (nodes["CLIPLoader"]["id"], 0)
    assert _edge(graph, relay.ENCODE, "ltx_relay_plan") == (nodes[relay.PLAN]["id"], 0)
    assert _edge(graph, relay.APPLY, "model") == upstream_model
    assert _edge(graph, relay.APPLY, "ltx_latent") == (store["id"], 4)
    assert _edge(graph, relay.APPLY, "sigmas") == (setup["id"], 2)
    assert _edge(graph, relay.APPLY, "positive") == (nodes[relay.ENCODE]["id"], 0)
    assert _edge(graph, relay.APPLY, "text_binding") == (nodes[relay.ENCODE]["id"], 1)
    for kind in (relay.BIND, "CFGGuider", relay.AUDIT):
        assert _edge(graph, kind, "model") == (nodes[relay.APPLY]["id"], 0)
    if combined:
        assert _edge(graph, eav.EFFECT_AUDIT, "model") == (nodes[relay.APPLY]["id"], 0)
        assert _edge(graph, eav.EFFECT_AUDIT, "candidate_latent") == (nodes[relay.AUDIT]["id"], 0)
    assert _edge(graph, "LTXVConditioning", "positive") == (nodes[relay.APPLY]["id"], 1)
    assert _edge(graph, "LTXVConditioning", "negative") == _edge(base, "LTXVConditioning", "negative")
    assert _edge(graph, relay.RELAY_AUDIT, "candidate_latent") == upstream_candidate
    assert _edge(graph, "MiniMaxH3SolEngineTAEHVDecodeT8Advanced", "latent") == (
        nodes[relay.RELAY_AUDIT]["id"], 0)
    assert _edge(graph, "MiniMaxH3OutputTrimT8", "audio") == _edge(
        base, "MiniMaxH3OutputTrimT8", "audio")
    assert nodes[relay.APPLY]["widgets_values"] == ["report_only", 32]
    assert nodes["SamplerCustomAdvanced"]["widgets_values"] == (
        next(node for node in base["nodes"] if node["type"] == "SamplerCustomAdvanced")["widgets_values"])


def test_old_source_eav_and_public_workflows_are_not_mutated():
    public_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in old_sources()}
    old_source = deepcopy(source.generated(isolated_media=True))
    old_eav = deepcopy(eav.generated())
    assert len(relay.generated()) == 8
    assert source.generated(isolated_media=True) == old_source
    assert eav.generated() == old_eav
    assert public_hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in old_sources()}
