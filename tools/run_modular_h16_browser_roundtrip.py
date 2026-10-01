"""Native UI import, Save As and reopen of four H16 freeze/resume candidates.

This does not queue inference. The only edits are visible Relay text and the
final-window EAV mode in the effects pair's isolated QA copies.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time

from playwright.sync_api import sync_playwright

from tools.audit_modular_browser_roundtrip import audit_roundtrip
from tools.audit_modular_legacy_browser_save import (
    normalize_known_ui_changes,
    schema_default_appends,
)
from tools.audit_modular_sampling_compat import write_new
from tools.run_modular_s22_browser_roundtrip import (
    _edit_final_eav,
    _edit_visible_plan,
    _json_get,
    _local_url,
    _open,
    _queue_empty,
    _save_as,
)
from tools.serve_modular_h16_browser import PRIVATE, PROJECT, SOURCE_SETS


CASES = (
    ("plain_freeze", "44a44417b541afed42c46956472573610dde77a11ef4fff02ca58636e1223d1d", 31, 89, None),
    ("plain_resume", "efcf6a38ed5fbb263ada5013dee36e9a459582eded2749df4025d05c6c215a1b", 34, 109, 53),
    ("effect_freeze", "5e6d6a16ed918ba4e34cd7d3df79f9676e55d584c1cd740ae1cea0e29e73169f", 48, 173, None),
    ("effect_resume", "d3430b0e4632039ec1360aaff3b8ec91264ccc1b900f27feb0f7950407e661ea", 56, 219, 90),
)
GEOMETRY_CASES = (
    CASES[0],
    ("plain_resume", "736636204c8064aa39570ed95d5a2c71750176c435889eeff4c32533df0bc407", 34, 109, 53),
    CASES[2],
    ("effect_resume", "06429413859c7a84155a7d748b18009ea229a7009e5b077f413e2d22deacd93f", 56, 219, 90),
)
CASE_SETS = {"historical": CASES, "geometry": GEOMETRY_CASES}
BRIDGE_KIND = "MiniMaxH3H16VerifiedNativeSourceEXPT8"
BRIDGE_FIELDS = ("resume_verified", "checkpoint_id", "content_sha256",
                 "file_sha256", "manifest_json", "report_json")
BRIDGE_PLACEHOLDERS = [False, "", "", "", "", ""]
MARKER = " [H16 browser QA; do not queue]"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bridge_connected_placeholders(before: dict, after: dict,
                                   current_nodes: dict, node_id: int) -> list:
    """Accept only six inert current-UI values for connected required pins."""
    old = next(node for node in before["nodes"] if node["id"] == node_id)
    new = next(node for node in after["nodes"] if node["id"] == node_id)
    fields = ("av_latent", *BRIDGE_FIELDS)
    if (old["type"] != BRIDGE_KIND or new["type"] != BRIDGE_KIND
            or old.get("widgets_values") != []
            or current_nodes[BRIDGE_KIND]["info"]["input_order"]["required"] != list(fields)
            or len(old["inputs"]) != len(fields) or len(new["inputs"]) != len(fields)
            or [pin["name"] for pin in old["inputs"]] != list(fields)
            or [pin["name"] for pin in new["inputs"]] != list(fields)
            or any(old_pin.get("link") is None or new_pin.get("link") != old_pin["link"]
                   for old_pin, new_pin in zip(old["inputs"], new["inputs"], strict=True))
            or new.get("widgets_values") != BRIDGE_PLACEHOLDERS
            or new.get("widgets_values_named") != dict(zip(
                BRIDGE_FIELDS, BRIDGE_PLACEHOLDERS, strict=True))):
        raise ValueError("H16 native bridge connected-widget placeholders changed")
    return BRIDGE_PLACEHOLDERS.copy()


def _seed_control_after_generate(before: dict, after: dict,
                                 current_nodes: dict, node_id: int = 16) -> list:
    """Accept only the current frontend's inert appended seed UI control."""
    old = next(node for node in before["nodes"] if node["id"] == node_id)
    new = next(node for node in after["nodes"] if node["id"] == node_id)
    kind = "MiniMaxH3TwoPassDetailMixerT8Advanced"
    old_values = old.get("widgets_values")
    new_values = new.get("widgets_values")
    named = new.get("widgets_values_named")
    order = current_nodes[kind]["info"]["input_order"]
    if (old["type"] != kind or new["type"] != kind
            or not isinstance(old_values, list) or len(old_values) != 19
            or not isinstance(new_values, list) or new_values != [*old_values, "randomize"]
            or not isinstance(named, dict)
            or list(named) != [*order["required"][-19:], "control_after_generate"]
            or list(named.values()) != new_values
            or order.get("optional", []) != []
            or order["required"][-1] != "restart_seed"
            or len(order["required"]) != len(old_values) + 3):
        raise ValueError("H16 DetailMixer frontend seed control changed")
    return ["randomize"]


def _normalize_preview_any_input(before: dict, after: dict,
                                 current_nodes: dict) -> list[int]:
    """Current PreviewAny schema widens a saved STRING pin to wildcard."""
    old_nodes = {node["id"]: node for node in before["nodes"]}
    info = current_nodes["PreviewAny"]["info"]
    if (info["input_order"]["required"] != ["source"]
            or info["input"]["required"]["source"][0] != "*"
            or info["output"] != ["STRING"]
            or info["output_name"] != ["STRING"]):
        raise ValueError("Current PreviewAny input schema changed")
    normalized = []
    for node in after["nodes"]:
        old = old_nodes[node["id"]]
        if old["type"] != "PreviewAny":
            continue
        old_pins, new_pins = old.get("inputs"), node.get("inputs")
        if (node["type"] != "PreviewAny" or len(old_pins) != 1
                or len(new_pins) != 1
                or old.get("outputs") != []
                or len(node.get("outputs", [])) != 1
                or node["outputs"][0]["name"] != "STRING"
                or node["outputs"][0]["type"] != "STRING"
                or node["outputs"][0].get("links")
                or old_pins[0]["name"] != "source"
                or old_pins[0]["type"] != "STRING"
                or old_pins[0].get("link") is None
                or new_pins[0]["name"] != "source"
                or new_pins[0]["type"] != "*"
                or new_pins[0].get("link") != old_pins[0]["link"]):
            raise ValueError("H16 PreviewAny pin changed beyond current-schema widening")
        new_pins[0]["type"] = "STRING"
        node["outputs"] = []
        normalized.append(node["id"])
    return normalized


def _audit(item: dict, profile: Path, current_nodes: dict) -> dict:
    original = item["source"]
    saved = item["saved"]
    if (_sha(original) != item["sha"] or not saved.is_file()
            or not original.is_relative_to(PRIVATE.resolve())
            or not saved.is_relative_to(profile)):
        raise ValueError("Pinned H16 source or isolated native save changed")
    before, after = json.loads(original.read_bytes()), json.loads(saved.read_bytes())
    normalized, ui_changes = normalize_known_ui_changes(
        before, after, current_nodes,
        labels_sha256=_sha(PROJECT / "web/task_type_labels.js"))
    preview_nodes = _normalize_preview_any_input(before, normalized, current_nodes)
    for node_id in preview_nodes:
        ui_changes[str(node_id)] = {
            "kind": "current_schema_preview_any", "original_type": "STRING",
            "added_disconnected_output": "STRING"}
    seed_appended = _seed_control_after_generate(before, normalized, current_nodes)
    for_schema = deepcopy(normalized)
    seed_node = next(node for node in for_schema["nodes"] if node["id"] == 16)
    seed_node["widgets_values"] = seed_node["widgets_values"][:-1]
    seed_node["widgets_values_named"].pop("control_after_generate")
    bridge_id = item["bridge_id"]
    if bridge_id is not None:
        placeholders = _bridge_connected_placeholders(
            before, normalized, current_nodes, bridge_id)
        next(node for node in for_schema["nodes"] if node["id"] == bridge_id)["widgets_values"] = []
        ui_changes[str(bridge_id)] = {
            "kind": "connected_bridge_placeholders", "fields": list(BRIDGE_FIELDS)}
    appends = schema_default_appends(before, for_schema, current_nodes)
    appends[16] = seed_appended
    ui_changes["16"] = {"kind": "current_ui_seed_control", "value": "randomize"}
    if bridge_id is not None:
        appends[bridge_id] = placeholders
    semantic = audit_roundtrip(before, normalized, edits=item["edits"],
                               appended_widgets=appends)
    return {"case": item["label"],
            "source": original.relative_to(PROJECT).as_posix(),
            "source_sha256": item["sha"],
            "saved": saved.relative_to(profile).as_posix(),
            "saved_sha256": _sha(saved),
            "semantic_audit": semantic,
            "ui_normalizations": ui_changes,
            "appended_connected_or_optional_widgets": {
                str(node_id): values for node_id, values in appends.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-set", choices=tuple(CASE_SETS),
                        default="historical")
    parser.add_argument("--reopen-existing", action="store_true",
                        help="Audit native saves made by an earlier interrupted QA pass")
    args = parser.parse_args()
    profile, output = args.profile.resolve(), args.output.resolve()
    base_url = _local_url(args.base_url)
    if (not profile.is_relative_to(PRIVATE.resolve()) or not (profile / "ready.json").is_file()
            or not output.is_relative_to(PRIVATE.resolve()) or output.exists()
            or not re.fullmatch(r"QA_[A-Za-z0-9_-]{1,45}", args.prefix)):
        parser.error("Use an existing isolated private profile, new report and QA prefix")
    ready = json.loads((profile / "ready.json").read_text(encoding="utf8"))
    cases_spec, sources = CASE_SETS[args.candidate_set], SOURCE_SETS[args.candidate_set]
    if (ready.get("url") != base_url or not ready.get("cpu_only")
            or ready.get("candidate_set", "historical") != args.candidate_set
            or len(ready.get("copies", [])) != len(cases_spec)
            or not _queue_empty(base_url) or _json_get(base_url + "/history")):
        parser.error("This is not the expected idle isolated CPU Core")

    workflow_dir = profile / "user/default/workflows"
    cases = []
    for (label, expected_sha, nodes, links, bridge_id), (directory, name, copy_name) in zip(
            cases_spec, sources, strict=True):
        source = (PRIVATE / directory / name).resolve()
        copied = workflow_dir / copy_name
        saved = workflow_dir / f"{args.prefix}_{label}.json"
        if (_sha(source) != expected_sha or not copied.is_file()
                or _sha(copied) != expected_sha
                or saved.is_file() != args.reopen_existing):
            parser.error(f"Pinned H16 candidate or isolated copy changed: {label}")
        graph = json.loads(source.read_bytes())
        edits = {}
        if label.startswith("effect_"):
            plan = next(node for node in graph["nodes"] if node["id"] == 51)["widgets_values"][0]
            edits[(51, 0)] = plan + MARKER
        if label == "effect_resume":
            eav = next(node for node in graph["nodes"] if node["id"] == 78)
            if eav["widgets_values"][0] != "report_only":
                parser.error("Pinned last-window EAV mode changed")
            edits[(78, 0)] = "apply_exp"
        cases.append({"label": label, "sha": expected_sha,
                      "source": source, "copied": copied, "saved": saved,
                      "nodes": nodes, "links": links,
                      "bridge_id": bridge_id, "edits": edits})

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--disable-gpu"])
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 960})
            page.goto(base_url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_function("() => window.app && app.graph && app.vueAppReady", timeout=30000)
            page.wait_for_timeout(1000)
            for item in cases:
                if not args.reopen_existing:
                    _open(page, item["copied"].stem, item["nodes"], item["links"])
                    if item["label"].startswith("effect_"):
                        original_plan = next(node for node in json.loads(
                            item["source"].read_bytes())["nodes"] if node["id"] == 51)["widgets_values"][0]
                        if _edit_visible_plan(page, original_plan, node_id=51,
                                              marker=MARKER) != item["edits"][(51, 0)]:
                            raise ValueError("Visible H16 Relay edit differs")
                    if item["label"] == "effect_resume":
                        _edit_final_eav(page, node_id=78)
                    _save_as(page, item["saved"].stem)
                    deadline = time.monotonic() + 10
                    while not item["saved"].is_file() and time.monotonic() < deadline:
                        page.wait_for_timeout(100)
                    if not item["saved"].is_file():
                        raise ValueError("Native H16 Save As did not create the isolated copy")
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("() => window.app && app.graph && app.vueAppReady", timeout=30000)
                    page.wait_for_timeout(1000)
                _open(page, item["saved"].stem, item["nodes"], item["links"])
                if item["label"].startswith("effect_"):
                    actual = page.evaluate("id => app.graph.getNodeById(id).widgets[0].value", 51)
                    if actual != item["edits"][(51, 0)]:
                        raise ValueError("Native reopen lost H16 Relay text edit")
                if item["label"] == "effect_resume":
                    actual = page.evaluate("id => app.graph.getNodeById(id).widgets[0].value", 78)
                    if actual != "apply_exp":
                        raise ValueError("Native reopen lost H16 final-window EAV edit")
        finally:
            browser.close()

    if not _queue_empty(base_url) or _json_get(base_url + "/history"):
        raise ValueError("H16 browser QA unexpectedly queued inference")
    object_info = _json_get(base_url + "/object_info")
    current_nodes = {name: {"id": name, "info": info}
                     for name, info in object_info.items()}
    audited = [_audit(item, profile, current_nodes) for item in cases]
    report = {"schema": "t8.modular-sampling.h16-native-browser-roundtrip.v1",
              "status": "pass_browser_serialization_only",
              "profile": profile.relative_to(PROJECT).as_posix(),
              "candidate_set": args.candidate_set,
              "base_url": base_url, "queue_and_history_empty_before_after": True,
              "reopen_existing_native_saves": args.reopen_existing,
              "cases": audited,
              "qualification": "Native headless import, visible effect edits, Save As, reload/reopen, "
              "nodes/named edges/widgets and source SHA only. Never queue QA copies; "
              "no model, media, GPU, cache or human-quality qualification."}
    write_new(output, report)
    print(json.dumps({"status": report["status"], "cases": [
        {"case": item["case"], "semantic_audit": item["semantic_audit"]}
        for item in audited]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
