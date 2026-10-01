"""Explicit frontend profile preserves old graph values and named sampler wires."""
from copy import deepcopy
import json

import pytest

from tools import build_modular_ltx_load_policy_workflows as builder
from tools import build_modular_ltx_rgb_source_workflows as source
from tools.build_modular_h16_storage_workflow import Draft
from tools.build_modular_ltx_rgb_workflows import _source, _unique


def originals():
    return {**{key: graph for key, graph in source.generated(isolated_media=True).items()
               if not key.endswith("freeze_source")},
            **builder.relay.eav.generated(), **builder.relay.generated()}


@pytest.mark.parametrize("name,original", list(originals().items()))
def test_all_existing_full_cold_variants_only_add_explicit_MODEL_load_profile(name, original):
    before = deepcopy(original)
    old, new = Draft(original), Draft(builder.build(original))
    assert original == before
    assert set(old.nodes) <= set(new.nodes)
    setup = next(node for node in old.nodes.values() if node["type"] in builder.relay.SETUPS)
    policy, observe = _unique(new, builder.POLICY), _unique(new, builder.OBSERVE)
    assert policy["widgets_values"] == ["consistent_streaming_exp"]
    assert _source(new, policy, "model") == ((setup["id"], 0), "MODEL")
    for key, node in old.nodes.items():
        assert new.nodes[key]["type"] == node["type"]
        assert new.nodes[key].get("widgets_values") == node.get("widgets_values")
        for item in node.get("inputs", []):
            if item["link"] is None:
                assert next(value for value in new.nodes[key]["inputs"] if value["name"] == item["name"])["link"] is None
                continue
            edge, dtype = _source(old, node, item["name"])
            expected = ((policy["id"], 0), "MODEL") if item["name"] == "model" and edge == (setup["id"], 0) else (edge, dtype)
            assert _source(new, new.nodes[key], item["name"]) == expected
    sampler = _unique(new, "SamplerCustomAdvanced")
    assert sampler["widgets_values"] == old.nodes[sampler["id"]]["widgets_values"]
    assert _source(new, observe, "runtime") == ((policy["id"], 1), builder.RUNTIME)
    decoder = _unique(new, "MiniMaxH3SolEngineTAEHVDecodeT8Advanced")
    assert _source(new, observe, "candidate_latent") == _source(new, decoder, "latent")


def test_four_delivery_graphs_keep_explicit_cold_receipt_and_original_effect_defaults():
    graphs = builder.generated()
    assert len(graphs) == 4
    assert sum("Identity_Preserve" in name for name in graphs) == 2
    assert sum("resume_ltx" in name for name in graphs) == 2
    for graph in graphs.values():
        draft = Draft(graph)
        assert _unique(draft, builder.relay.APPLY)["widgets_values"][0] == "report_only"
        assert _unique(draft, builder.relay.eav.CONFIG)["widgets_values"][0] == "report_only"
        with pytest.raises(ValueError, match="original graph"):
            builder.build(graph)


def test_delivered_additive_JSONs_match_the_generator_without_old_graph_rewrites():
    for name, expected in builder.generated().items():
        path = source.ROOT / builder.TARGET / (name + ".json")
        assert json.loads(path.read_text(encoding="utf8")) == expected
