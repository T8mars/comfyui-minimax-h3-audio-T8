"""EAV drafts add only explicit tail branches; frozen/public originals stay exact."""
import hashlib
import json

import pytest

from tools import build_modular_audio_tail_effect_workflows as builder
from tools.build_formal_audio_refine_split_workflows import generated as old_graphs
from tools.build_modular_audio_refine_resume_all import _single, _source
from tools.build_modular_h16_storage_workflow import Draft


@pytest.mark.parametrize("path", sorted(builder.generated()))
def test_all_ten_new_tail_drafts_preserve_sampler_noise_sigmas_av_and_old_gate(path):
    old = old_graphs()[path]
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    current = builder.generated()[path]
    original, updated = Draft(old), Draft(current)
    old_sampler = _single(original, {"SamplerCustomAdvanced"})
    sampler = _single(updated, {"SamplerCustomAdvanced"})
    assert sampler["id"] == old_sampler["id"]
    for field in ("noise", "sampler", "sigmas", "latent_image"):
        a, a_slot, a_type = _source(original, old_sampler, field)
        b, b_slot, b_type = _source(updated, sampler, field)
        assert (a["id"], a_slot, a_type) == (b["id"], b_slot, b_type)
    actual_guider, slot, dtype = _source(updated, sampler, "guider")
    assert (actual_guider["type"], slot, dtype) == (builder.GUIDER, 0, "GUIDER")
    gate_type = "MiniMaxH3AudioRefineQualityGateT8Advanced"
    old_gate, gate = _single(original, {gate_type}), _single(updated, {gate_type})
    assert gate == old_gate
    for kind in (builder.BIND, builder.GUIDER, builder.APPLY, builder.CONFIG, builder.AUDIT):
        assert _single(updated, {kind})
    assert len(current["nodes"]) == len(old["nodes"]) + 5
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert json.loads(path.read_text(encoding="utf8")) == old


def test_long_tail_effect_context_uses_the_actual_planner_window():
    graphs = builder.generated()
    long_graph = next(graph for path, graph in graphs.items() if "Long_Video" in path.name)
    draft = Draft(long_graph)
    bind = _single(draft, {builder.BIND})
    cond = _single(draft, {"MiniMaxH3PromptRelayLongVideoConditioningT8Advanced"})
    for field in ("segment_index", "context_frames"):
        a, a_slot, a_type = _source(draft, bind, field)
        b, b_slot, b_type = _source(draft, cond, field)
        assert (a["id"], a_slot, a_type) == (b["id"], b_slot, b_type)


def test_builder_rejects_an_already_modified_tail_guider():
    graph = next(iter(builder.generated().values()))
    with pytest.raises(ValueError, match="original bound Setup guider"):
        builder.add_tail_effects(graph)


@pytest.mark.parametrize("effect", ["relay", "combined"])
@pytest.mark.parametrize("path", sorted(builder.generated("relay")))
def test_fresh_relay_is_independent_paired_and_does_not_replace_frozen_av(path, effect):
    old = old_graphs()[path]
    draft = Draft(builder.generated(effect)[path])
    original = Draft(old)
    old_cond = _single(original, {"MiniMaxH3AudioConditioningT8"})
    assert draft.nodes[old_cond["id"]]["widgets_values"] == old_cond["widgets_values"]
    cond = _single(draft, {builder.RELAY_COND})
    plan = _single(draft, {builder.RELAY_PLAN})
    route = _single(draft, {builder.RELAY_ROUTE})
    bind = _single(draft, {builder.BIND})
    guider = _single(draft, {builder.GUIDER})
    sampler = _single(draft, {"SamplerCustomAdvanced"})
    old_sampler = _single(original, {"SamplerCustomAdvanced"})
    assert _source(draft, cond, "model")[0]["id"] == bind["id"]
    assert _source(draft, cond, "prompt_relay_plan")[0]["id"] == route["id"]
    assert _source(draft, route, "prompt_relay_plan")[0]["id"] == plan["id"]
    assert _source(draft, guider, "positive") == (cond, 1, "CONDITIONING")
    assert cond["outputs"][2]["links"] == []  # Never use Relay's fresh empty AV.
    for field in ("noise", "sampler", "sigmas", "latent_image"):
        a, sa, ta = _source(original, old_sampler, field)
        b, sb, tb = _source(draft, sampler, field)
        assert (a["id"], sa, ta) == (b["id"], sb, tb)
    for item in old_cond["inputs"]:
        if item.get("link") is None:
            continue
        a, sa, ta = _source(original, old_cond, item["name"])
        target, field = (plan, "length") if item["name"] == "length" else (cond, item["name"])
        b, sb, tb = _source(draft, target, field)
        assert (a["id"], sa, ta) == (b["id"], sb, tb)
    assert plan["widgets_values"][0] == old_cond["widgets_values"][0]
    assert plan["widgets_values"][1] == plan["widgets_values"][4] == ""
    assert cond["widgets_values"][-2:] == ["report_only", 256]
    if effect == "relay":
        assert not any(node["type"] in {builder.CONFIG, builder.APPLY, builder.AUDIT}
                       for node in draft.nodes.values())
        assert _source(draft, guider, "model") == (cond, 0, "MODEL")
    else:
        apply = _single(draft, {builder.APPLY})
        assert _source(draft, apply, "model") == (cond, 0, "MODEL")
    # Every serialized edge still agrees with both input and output indexes.
    for link in draft.graph["links"]:
        assert draft.nodes[link[3]]["inputs"][link[4]]["link"] == link[0]
        assert link[0] in draft.nodes[link[1]]["outputs"][link[2]]["links"]
    gate = _single(draft, {"MiniMaxH3AudioRefineQualityGateT8Advanced"})
    assert gate == _single(original, {gate["type"]})
    assert json.loads(path.read_text(encoding="utf8")) == old


def test_matrix_has_all_ten_eav_and_eight_fresh_relay_and_combined_variants():
    matrix = builder.generated_matrix()
    assert len(matrix) == 26
    assert {effect: sum(path.stem.endswith("_" + effect) for path in matrix)
            for effect in ("eav", "relay", "combined")} == {"eav": 10, "relay": 8, "combined": 8}
    for path, graph in old_graphs().items():
        if "_resume_audio_" in path.name and any(node["type"] == builder.RELAY_PLAN for node in graph["nodes"]):
            with pytest.raises(ValueError, match="existing independent tail Relay"):
                builder.add_tail_effects(graph, effect="combined")
