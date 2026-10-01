"""Native browser-save audit preserves semantic graph edges and widget values."""
from copy import deepcopy

import pytest

from tools.audit_modular_browser_roundtrip import audit_roundtrip


def _pair():
    original = {
        "nodes": [
            {"id": 1, "type": "Source", "mode": 0, "inputs": [],
             "outputs": [{"name": "model", "type": "MODEL", "links": [1]}],
             "widgets_values": ["old"]},
            {"id": 2, "type": "Sink", "mode": 0,
             "inputs": [{"name": "model", "type": "MODEL", "link": 1}],
             "outputs": [], "widgets_values": [3.0, False]},
        ],
        "links": [[1, 1, 0, 2, 0, "MODEL"]],
    }
    saved = deepcopy(original)
    saved["nodes"][0]["widgets_values"] = ["edited"]
    saved["nodes"][0]["widgets_values_named"] = {"prompt": "edited"}
    saved["nodes"][1]["inputs"].insert(
        0, {"name": "strength", "type": "FLOAT", "link": None,
            "widget": {"name": "strength"}}
    )
    saved["nodes"][1]["widgets_values"] = [3, False, "auto"]
    saved["nodes"][1]["widgets_values_named"] = {
        "strength": 3, "enabled": False, "codec": "auto"
    }
    saved["links"][0][4] = 1
    return original, saved


def test_native_widget_socket_insertion_preserves_named_edge():
    original, saved = _pair()
    assert audit_roundtrip(original, saved, edits={(1, 0): "edited"},
                           appended_widgets={2: ["auto"]}) == {
        "nodes": 2, "named_edges": 1, "widgets": 4,
        "edits": ["1:0"], "browser_normalized_optional_widgets": [2],
    }


@pytest.mark.parametrize("damage", ["reroute", "widget", "mode", "named", "owner"])
def test_roundtrip_rejects_unexpected_changes(damage):
    original, saved = _pair()
    if damage == "reroute":
        saved["nodes"][1]["inputs"][1]["name"] = "other_model"
    elif damage == "widget":
        saved["nodes"][1]["widgets_values"][1] = 0
    elif damage == "mode":
        saved["nodes"][1]["mode"] = 2
    elif damage == "named":
        saved["nodes"][1]["widgets_values_named"].pop("codec")
    else:
        saved["nodes"][0]["outputs"][0]["links"] = []
    with pytest.raises(ValueError):
        audit_roundtrip(original, saved, edits={(1, 0): "edited"},
                        appended_widgets={2: ["auto"]})
