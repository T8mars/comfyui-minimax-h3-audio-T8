"""The native canvas probe may only prefill a private S27 Relay copy."""

from copy import deepcopy

import pytest

from tools import run_modular_ltx_relay_canvas_gpu as canvas
from tools import run_modular_ltx_relay_gpu as relay


def test_prepared_canvas_copy_retains_structure_and_changes_only_expected_widgets():
    _, original, _ = relay._saved("ordinary", False, relay.SAVED)
    before = deepcopy(original)
    prepared = canvas.prepared_frontend(original, width=1024, height=576)
    assert original == before
    assert prepared["links"] == original["links"]
    assert [(node["id"], node["type"]) for node in prepared["nodes"]] == [
        (node["id"], node["type"]) for node in original["nodes"]]
    old = {node["id"]: node for node in original["nodes"]}
    new = {node["id"]: node for node in prepared["nodes"]}
    changed = {node_id for node_id in old if old[node_id] != new[node_id]}
    assert changed == {1, 3, 10, 20, 30, 32}
    assert new[1]["widgets_values"][0] == "source.mp4"
    assert new[3]["widgets_values"][:2] == [1024, 576]
    assert new[10]["widgets_values"][1] == "dense_reference"
    assert new[20]["widgets_values"][0] == canvas.MEDIA_PREFIX
    assert new[30]["widgets_values"][:2] == [canvas.GLOBAL, canvas.LOCAL]
    assert new[32]["widgets_values"][0] == "apply_exp"
    assert new[29]["widgets_values"][-1] is False


def test_prepared_canvas_rejects_missing_effect_node():
    _, original, _ = relay._saved("ordinary", False, relay.SAVED)
    broken = deepcopy(original)
    broken["nodes"] = [node for node in broken["nodes"] if node["type"] != relay.builder.APPLY]
    with pytest.raises(ValueError, match="inventory changed"):
        canvas.prepared_frontend(broken, width=1024, height=576)


@pytest.mark.parametrize("route", ["ordinary", "identity"])
@pytest.mark.parametrize("combined", [False, True])
def test_each_full_canvas_route_has_visible_relay_and_optional_independent_eav(route, combined):
    _, original, _ = relay._saved(route, combined, relay.SAVED)
    prepared = canvas.prepared_frontend(original, width=1024, height=576)
    nodes = {node["type"]: node for node in prepared["nodes"]}
    assert nodes[relay.builder.PLAN]["widgets_values"][:2] == [canvas.GLOBAL, canvas.LOCAL]
    assert nodes[relay.builder.APPLY]["widgets_values"][0] == "apply_exp"
    if combined:
        assert nodes["MiniMaxH3StageEAVConfigEXPT8"]["widgets_values"][0] == "report_only"
    else:
        assert "MiniMaxH3StageEAVConfigEXPT8" not in nodes
    setup = next(node for node in prepared["nodes"] if node["type"] in relay.builder.SETUPS)
    if route == "identity":
        assert setup["widgets_values"][3] == "dense_reference"
        assert setup["widgets_values"][6] is False
    else:
        assert setup["widgets_values"][1] == "dense_reference"
        assert setup["widgets_values"][4] is False
