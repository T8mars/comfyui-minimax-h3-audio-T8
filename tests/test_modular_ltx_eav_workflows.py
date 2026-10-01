"""Opt-in effect graphs cannot silently change H3/LTX source or sampler."""
import hashlib
from copy import deepcopy

import pytest

from tools import build_modular_ltx_eav_workflows as effect
from tools import build_modular_ltx_rgb_source_workflows as source
from tools.build_modular_ltx_rgb_workflows import sources as old_sources


@pytest.mark.parametrize("name,graph", list(effect.generated().items()))
def test_four_effect_graphs_only_insert_eav_and_route_model_candidate(name, graph):
    baseline_name = name.removesuffix("_external_eav")
    original = source.generated(isolated_media=True)[baseline_name]
    old_nodes = {node["id"]: node for node in original["nodes"]}
    by_kind = {node["type"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    def edge(kind, field):
        node = by_kind[kind]
        item = next(item for item in node["inputs"] if item["name"] == field)
        return tuple(links[item["link"]][1:3])
    assert {node["id"] for node in graph["nodes"]} - old_nodes.keys() == {
        by_kind[kind]["id"] for kind in (effect.CONFIG, effect.APPLY, effect.EFFECT_AUDIT)}
    assert edge(effect.APPLY, "model") == (by_kind[next(k for k in effect.SETUPS if k in by_kind)]["id"], 0)
    assert edge(effect.APPLY, "ltx_latent") == (
        by_kind[source.LOAD if name.endswith("resume_ltx_external_eav") else source.SAVE]["id"], 4)
    for kind in (effect.BIND, "CFGGuider", effect.AUDIT):
        assert edge(kind, "model") == (by_kind[effect.APPLY]["id"], 0)
    assert edge(effect.EFFECT_AUDIT, "candidate_latent") == (by_kind[effect.AUDIT]["id"], 0)
    assert edge("MiniMaxH3SolEngineTAEHVDecodeT8Advanced", "latent") == (
        by_kind[effect.EFFECT_AUDIT]["id"], 0)
    assert by_kind[effect.CONFIG]["widgets_values"][0] == "report_only"
    assert by_kind["SamplerCustomAdvanced"]["widgets_values"] == (
        next(n for n in original["nodes"] if n["type"] == "SamplerCustomAdvanced")["widgets_values"])
    assert by_kind["MiniMaxH3OutputTrimT8"]["widgets_values"] == (
        next(n for n in original["nodes"] if n["type"] == "MiniMaxH3OutputTrimT8")["widgets_values"])


def test_existing_six_graphs_and_two_public_sources_unchanged():
    old = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in old_sources()}
    baseline = deepcopy(source.generated(isolated_media=True))
    assert len(effect.generated()) == 4
    assert source.generated(isolated_media=True) == baseline
    assert old == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in old_sources()}
    assert not any(name.endswith("freeze_source_external_eav") for name in effect.generated())
