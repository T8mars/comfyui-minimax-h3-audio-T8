"""S22 native-save exceptions are narrow and fail closed."""

from copy import deepcopy

import pytest

from tools.run_modular_s22_browser_roundtrip import (
    WINDOW_PROFILES,
    _bridge_connected_placeholders,
    _local_url,
    _sha,
)
from tools.serve_modular_s22_browser import PROFILE_NAMES, WORKFLOWS


FIELDS = ("resume_verified", "checkpoint_id", "content_sha256",
          "file_sha256", "manifest_json", "report_json")
PLACEHOLDERS = [False, "", "", "", "", ""]


def _bridge_pair(node_id=49):
    inputs = [{"name": name, "link": index}
              for index, name in enumerate(("av_latent", *FIELDS), start=1)]
    old = {"nodes": [{"id": node_id,
                      "type": "MiniMaxH3ChunkedV5VerifiedNativeSourceEXPT8",
                      "inputs": inputs, "widgets_values": []}]}
    new = deepcopy(old)
    new["nodes"][0]["widgets_values"] = PLACEHOLDERS.copy()
    new["nodes"][0]["widgets_values_named"] = dict(zip(FIELDS, PLACEHOLDERS, strict=True))
    schema = {"MiniMaxH3ChunkedV5VerifiedNativeSourceEXPT8": {
        "info": {"input_order": {"required": ["av_latent", *FIELDS]}}}}
    return old, new, schema


@pytest.mark.parametrize("node_id", [49, 54])
def test_connected_bridge_native_placeholders_are_exact(node_id):
    old, new, schema = _bridge_pair(node_id)
    assert _bridge_connected_placeholders(old, new, schema,
                                          node_id=node_id) == PLACEHOLDERS


@pytest.mark.parametrize("window_count,ids", [(3, (33, 44, 49)),
                                               (4, (34, 49, 54))])
def test_browser_sources_and_service_mapping_are_sha_pinned(window_count, ids):
    candidate, cases, plan_id, eav_id, bridge_id = WINDOW_PROFILES[window_count]
    assert (plan_id, eav_id, bridge_id) == ids
    service_candidate, *source_names = WORKFLOWS[window_count]
    assert candidate.name == service_candidate
    assert [case[1] for case in cases] == source_names
    assert [case[2] for case in cases] == list(PROFILE_NAMES[window_count])
    for _, source_name, _, expected_sha, _, _ in cases:
        assert _sha(candidate / source_name) == expected_sha


@pytest.mark.parametrize("damage", ["link", "value", "named", "schema"])
def test_connected_bridge_rejects_non_native_change(damage):
    old, new, schema = _bridge_pair()
    if damage == "link":
        new["nodes"][0]["inputs"][1]["link"] = 999
    elif damage == "value":
        new["nodes"][0]["widgets_values"][1] = "tampered"
    elif damage == "named":
        new["nodes"][0]["widgets_values_named"]["checkpoint_id"] = "tampered"
    else:
        schema["MiniMaxH3ChunkedV5VerifiedNativeSourceEXPT8"]["info"]["input_order"]["required"][1] = "other"
    with pytest.raises(ValueError, match="connected-widget placeholders"):
        _bridge_connected_placeholders(old, new, schema)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8189", "http://localhost:8242", "https://127.0.0.1:8242",
    "http://127.0.0.1:8242/other", "http://127.0.0.1:8242/?q=1",
])
def test_browser_qa_rejects_user_or_nonlocal_core(url):
    with pytest.raises(ValueError, match="isolated"):
        _local_url(url)
