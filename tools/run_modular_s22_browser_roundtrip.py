"""Exercise S22's saved three/four-window candidates in an isolated native UI.

This opens the frontend graphs, edits the visible Relay text and final-window
EAV control, uses ComfyUI Save As, reloads the browser, and audits the files.
It never queues a prompt or treats UI serialization as model qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlparse
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

from tools.audit_modular_browser_roundtrip import audit_roundtrip
from tools.audit_modular_legacy_browser_save import (
    normalize_known_ui_changes,
    schema_default_appends,
)
from tools.audit_modular_sampling_compat import write_new


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "artifacts/development"
BASE = PRIVATE / "modular-sampling-m4-chunked-v5-storage-20260924"
THREE_CASES = (
    ("freeze", "01_freeze_window_1.json", "S22_freeze_window_1.json",
     "23af579e5b6a1f6e49b18a777c6acaa4a3f7e292816aab4ac8d20bcb238ffbe1", 33, 107),
    ("resume", "02_resume_window_2_DRAFT.json", "S22_resume_window_2_DRAFT.json",
     "be109aa6eb0f54f265d9729cb3152dba9ac94ef5e51cd6fe52f7ea5419a2a450", 29, 81),
)
FOUR_CASES = (
    ("freeze", "01_freeze_window_2.json", "S22_freeze_window_2.json",
     "1faa383d9e8cbb2a1d935d569cd847369190d684a4d8776c4b164eb6669c6d41", 38, 140),
    ("resume", "02_resume_window_3_DRAFT.json", "S22_resume_window_3_DRAFT.json",
     "0508c270c81de4d79d173a39ca8dd7b2bdaac6bcbb026f39191d579cde893ed7", 29, 81),
)
WINDOW_PROFILES = {
    3: (BASE / "candidate-v4-three-windows", THREE_CASES, 33, 44, 49),
    4: (BASE / "candidate-v6-four-windows", FOUR_CASES, 34, 49, 54),
}
MARKER = " [S22 browser QA; do not queue]"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_url(value: str) -> str:
    parsed = urlparse(value)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port
            or parsed.username or parsed.password or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment or parsed.port == 8189):
        raise ValueError("Use the explicit isolated 127.0.0.1 Core URL, never user Core 8189")
    return f"http://127.0.0.1:{parsed.port}"


def _json_get(url: str) -> dict:
    with urlopen(url, timeout=20) as response:
        return json.load(response)


def _queue_empty(base_url: str) -> bool:
    queue = _json_get(base_url + "/queue")
    return not queue.get("queue_running") and not queue.get("queue_pending")


def _open(page, name: str, nodes: int, links: int) -> None:
    if page.get_by_role("dialog").count():
        raise ValueError(f"Unexpected native dialog before opening {name}: "
                         f"{page.get_by_role('dialog').all_text_contents()}")
    sidebar = page.get_by_test_id("workflows-sidebar")
    if not sidebar.is_visible():
        page.get_by_test_id("workflows-tab-button").click()
    sidebar.get_by_text(name, exact=True).click()
    page.wait_for_function("n => window.app && app.graph && app.graph._nodes.length === n", arg=nodes)
    counts = page.evaluate("() => ({nodes: app.graph._nodes.length, links: Object.keys(app.graph.links || {}).length})")
    if counts != {"nodes": nodes, "links": links} or page.get_by_role("dialog").count():
        raise ValueError(f"Native graph import differs or warned: {name}: {counts}")


def _edit_visible_plan(page, original: str, node_id: int = 33,
                       marker: str = MARKER) -> str:
    changed = original + marker
    page.evaluate("id => app.canvas.centerOnNode(app.graph.getNodeById(id))", node_id)
    page.wait_for_function("""([id, original]) => {
        const element = app.graph.getNodeById(id).widgets[0].element;
        return element?.getClientRects().length && element.value === original;
    }""", arg=[node_id, original])
    widget = page.locator('textarea[placeholder="global_prompt"]:visible')
    if widget.count() != 1 or widget.input_value() != original:
        raise ValueError("The visible Relay Plan text widget is not the saved source")
    if not page.evaluate("""id => {
        const widget = app.graph.getNodeById(id).widgets[0].element;
        const visible = [...document.querySelectorAll('textarea[placeholder="global_prompt"]')]
            .filter(element => element.getClientRects().length);
        return visible.length === 1 && visible[0] === widget;
    }""", node_id):
        raise ValueError(f"The visible textarea does not belong to Relay Plan node {node_id}")
    widget.fill(changed)
    if page.evaluate("id => app.graph.getNodeById(id).widgets[0].value", node_id) != changed:
        raise ValueError("Native Relay widget ignored the visible edit")
    return changed


def _edit_final_eav(page, node_id: int = 44) -> None:
    page.evaluate("""id => {
        const node = app.graph.getNodeById(id);
        if (!node || node.type !== 'MiniMaxH3StageEAVConfigEXPT8'
                || node.widgets[0].value !== 'report_only') throw Error('EAV source changed');
        app.canvas.centerOnNode(node);
    }""", node_id)
    page.wait_for_timeout(250)
    point = page.evaluate("""id => {
        const node = app.graph.getNodeById(id);
        const ds = app.canvas.ds;
        return {x: (node.pos[0] + node.size[0] * .75 + ds.offset[0]) * ds.scale,
                y: (node.pos[1] + node.widgets[0].y + 10 + ds.offset[1]) * ds.scale};
    }""", node_id)
    page.mouse.click(point["x"], point["y"])
    page.get_by_text("apply_exp", exact=True).click()
    if page.evaluate("id => app.graph.getNodeById(id).widgets[0].value", node_id) != "apply_exp":
        raise ValueError("Native final-window EAV widget ignored the visible edit")


def _save_as(page, name: str) -> None:
    page.get_by_role("button", name="图形模式，工作流操作").click()
    page.get_by_role("menuitem", name="另存为").click()
    dialog = page.get_by_role("dialog")
    dialog.locator("input").fill(name)
    dialog.get_by_role("button", name="确认").click()
    dialog.wait_for(state="hidden")


def _bridge_connected_placeholders(before: dict, after: dict, current_nodes: dict,
                                   node_id: int = 49) -> list:
    """Accept only current UI's inert values for six *connected* bridge pins."""
    old = next(node for node in before["nodes"] if node["id"] == node_id)
    new = next(node for node in after["nodes"] if node["id"] == node_id)
    kind = "MiniMaxH3ChunkedV5VerifiedNativeSourceEXPT8"
    fields = ("resume_verified", "checkpoint_id", "content_sha256",
              "file_sha256", "manifest_json", "report_json")
    placeholders = [False, "", "", "", "", ""]
    info = current_nodes[kind]["info"]
    if (old["type"] != kind or new["type"] != kind or old.get("widgets_values") != []
            or info["input_order"]["required"] != ["av_latent", *fields]
            or any(old_pin.get("link") is None or new_pin.get("link") != old_pin["link"]
                   for old_pin, new_pin in zip(old["inputs"], new["inputs"], strict=True))
            or [pin["name"] for pin in old["inputs"]] != ["av_latent", *fields]
            or new.get("widgets_values") != placeholders
            or new.get("widgets_values_named") != dict(zip(fields, placeholders, strict=True))):
        raise ValueError("Native bridge connected-widget placeholders changed")
    return placeholders


def _audit_s22_case(item: dict, profile: Path, current_nodes: dict) -> dict:
    case = item["case"]
    original = (ROOT / case["original"]).resolve()
    saved = (profile / "user/default/workflows" / case["saved"]).resolve()
    if (not original.is_relative_to(PRIVATE.resolve()) or not saved.is_relative_to(profile)
            or _sha(original) != case["original_sha256"] or not saved.is_file()):
        raise ValueError("SHA-pinned source or isolated native save is missing")
    before, after = json.loads(original.read_bytes()), json.loads(saved.read_bytes())
    labels_sha = _sha(ROOT / "web/task_type_labels.js")
    normalized, ui_changes = normalize_known_ui_changes(
        before, after, current_nodes, labels_sha256=labels_sha)
    if item["label"] == "resume":
        # A connected required BOOLEAN/STRING input has no source widget value.
        # Current native save serializes six inert placeholders. Check exact
        # values, order and all six links before excluding them from the generic
        # optional-widget detector; no other required additions are permitted.
        bridge_id = item["bridge_id"]
        placeholders = _bridge_connected_placeholders(
            before, normalized, current_nodes, node_id=bridge_id)
        for_schema = deepcopy(normalized)
        next(node for node in for_schema["nodes"] if node["id"] == bridge_id)["widgets_values"] = []
        appends = schema_default_appends(before, for_schema, current_nodes)
        appends[bridge_id] = placeholders
        ui_changes[str(bridge_id)] = {"kind": "connected_bridge_placeholders",
                            "fields": list(normalized_node["name"] for normalized_node in
                                           next(node for node in normalized["nodes"]
                                                if node["id"] == bridge_id)["inputs"][1:])}
    else:
        appends = schema_default_appends(before, normalized, current_nodes)
    edits = {(edit["node_id"], edit["widget_index"]): edit["after"]
             for edit in case["edits"]}
    semantic = audit_roundtrip(before, normalized, edits=edits, appended_widgets=appends)
    return {"original": case["original"], "original_sha256": case["original_sha256"],
            "saved": saved.relative_to(profile).as_posix(), "saved_sha256": _sha(saved),
            "semantic_audit": semantic, "ui_normalizations": ui_changes,
            "appended_connected_or_optional_widgets": {
                str(node_id): values for node_id, values in appends.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--window-count", type=int, choices=(3, 4), default=3)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile, output = args.profile.resolve(), args.output.resolve()
    candidate, source_cases, plan_id, final_eav_id, bridge_id = WINDOW_PROFILES[args.window_count]
    base_url = _local_url(args.base_url)
    if (not profile.is_relative_to(PRIVATE.resolve()) or not (profile / "ready.json").is_file()
            or not output.is_relative_to(PRIVATE.resolve()) or output.exists()
            or not re.fullmatch(r"QA_[A-Za-z0-9_-]{1,50}", args.prefix)):
        parser.error("Use an existing isolated private profile, a new private report and QA prefix")
    ready = json.loads((profile / "ready.json").read_text(encoding="utf8"))
    if ready.get("url") != base_url or not ready.get("cpu_only") or not _queue_empty(base_url):
        parser.error("This is not the expected idle isolated CPU Core")

    cases = []
    for label, source_name, profile_name, expected_sha, nodes, links in source_cases:
        source = candidate / source_name
        saved_name = f"{args.prefix}_{label}.json"
        target = profile / "user/default/workflows" / saved_name
        profile_copy = profile / "user/default/workflows" / profile_name
        if (_sha(source) != expected_sha or not profile_copy.is_file()
                or _sha(profile_copy) != expected_sha):
            parser.error(f"SHA-pinned candidate or profile copy changed: {source_name}")
        if target.exists():
            parser.error(f"Native QA save already exists: {target}")
        graph = json.loads(source.read_bytes())
        plan = next(node for node in graph["nodes"] if node["id"] == plan_id)["widgets_values"][0]
        edits = [{"node_id": plan_id, "widget_index": 0, "before": plan, "after": plan + MARKER}]
        if label == "resume":
            edits.append({"node_id": final_eav_id, "widget_index": 0,
                          "before": "report_only", "after": "apply_exp"})
        cases.append({"label": label, "profile_name": profile_name, "saved": saved_name,
                      "nodes": nodes, "links": links, "plan_id": plan_id,
                      "final_eav_id": final_eav_id, "bridge_id": bridge_id, "case": {
                          "original": source.relative_to(ROOT).as_posix(),
                          "original_sha256": expected_sha, "saved": saved_name, "edits": edits}})

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--disable-gpu"])
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 960})
            page.goto(base_url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_function("() => window.app && app.graph && app.vueAppReady", timeout=30000)
            page.wait_for_timeout(1000)
            for item in cases:
                _open(page, item["profile_name"].removesuffix(".json"), item["nodes"], item["links"])
                _edit_visible_plan(page, item["case"]["edits"][0]["before"],
                                   node_id=item["plan_id"])
                if item["label"] == "resume":
                    _edit_final_eav(page, node_id=item["final_eav_id"])
                _save_as(page, item["saved"].removesuffix(".json"))
                saved_file = profile / "user/default/workflows" / item["saved"]
                deadline = time.monotonic() + 10
                while not saved_file.is_file() and time.monotonic() < deadline:
                    page.wait_for_timeout(100)
                if not saved_file.is_file():
                    raise ValueError("Native Save As did not write into the isolated profile")
                page.reload(wait_until="domcontentloaded")
                page.wait_for_function("() => window.app && app.graph && app.vueAppReady", timeout=30000)
                page.wait_for_timeout(1000)
                _open(page, item["saved"].removesuffix(".json"), item["nodes"], item["links"])
                actual = page.evaluate("""([planId, eavId]) => ({
                    plan:app.graph.getNodeById(planId).widgets[0].value,
                    eav:app.graph.getNodeById(eavId)?.widgets[0].value})""",
                                       [item["plan_id"], item["final_eav_id"]])
                if actual["plan"] != item["case"]["edits"][0]["after"]:
                    raise ValueError("Native reopen lost the Relay text edit")
                if item["label"] == "resume" and actual["eav"] != "apply_exp":
                    raise ValueError("Native reopen lost the final-window EAV edit")
        finally:
            browser.close()

    if not _queue_empty(base_url):
        raise ValueError("Browser QA unexpectedly queued generation")
    object_info = _json_get(base_url + "/object_info")
    current_nodes = {name: {"id": name, "info": info} for name, info in object_info.items()}
    audited = [_audit_s22_case(item, profile=profile, current_nodes=current_nodes)
               for item in cases]
    report = {"schema": "t8.modular-sampling.s22-native-browser-roundtrip.v2",
              "status": "pass_browser_serialization_only", "profile": profile.relative_to(ROOT).as_posix(),
              "base_url": base_url, "window_count": args.window_count,
              "queue_empty_before_after": True,
              "cases": audited, "qualification": "Native headless browser import, visible edits, Save As, reload/reopen, "
              "nodes/named edges/widgets and source SHA only. QA copies must not be queued; "
              "no model, media, GPU, cache or human-quality qualification."}
    write_new(output, report)
    print(json.dumps({"status": report["status"], "cases": [
        {"saved": item["saved"], "semantic_audit": item["semantic_audit"]}
        for item in audited]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
