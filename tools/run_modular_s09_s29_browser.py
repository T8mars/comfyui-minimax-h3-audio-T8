"""Native open/edit/Save As/reload audit for all current S09 and S29 split drafts.

The frozen-stage resume paths are placeholders. Never queue these QA copies.
"""

from __future__ import annotations

import argparse
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
from tools.run_modular_hyperflow_fresh_browser_matrix import _open_stable
from tools.run_modular_s22_browser_roundtrip import (
    _edit_final_eav,
    _json_get,
    _local_url,
    _queue_empty,
    _save_as,
)
from tools.serve_modular_s09_s29_browser import PRIVATE, PROJECT, SOURCES

EDIT = frozenset({
    "S09_Manual_Pass_save_effects_FreeNoise_EXP.json",
    "S09_Manual_Pass_resume_effects_FreeNoise_EXP.json",
    "S29_RF_two_pass_detail_mixer_save_effects_EXP.json",
    "S29_RF_two_pass_detail_mixer_resume_effects_EXP.json",
})
SOURCE_NAMES: dict[str, frozenset[str]] | None = None
EDIT_MARKER = "S09-S29 QA"
REPORT_SCHEMA = "t8.modular-sampling.s09-s29-native-browser.v1"
REPORT_LIMITS = "Placeholder resume artifacts were not run by this browser audit; no trained GPU, media or human review."
RESUME_MARKERS = ("resume_effects",)
EAV_ORIGINAL = "report_only"
EAV_EDITED = "apply_exp"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _edit_node_plan(page, *, node_id: int, original: str, marker: str) -> None:
    """Fill the visible textarea owned by one Plan, even if other Plans are visible."""
    page.evaluate("id => app.canvas.centerOnNode(app.graph.getNodeById(id))", node_id)
    page.wait_for_function("""id => {
        const element = app.graph.getNodeById(id).widgets[0].element;
        return element?.getClientRects().length && element.value ===
            app.graph.getNodeById(id).widgets[0].value;
    }""", arg=node_id)
    index = page.evaluate("""id => {
        const element = app.graph.getNodeById(id).widgets[0].element;
        return [...document.querySelectorAll('textarea[placeholder="global_prompt"]')]
            .filter(item => item.getClientRects().length).indexOf(element);
    }""", node_id)
    textareas = page.locator('textarea[placeholder="global_prompt"]:visible')
    if index < 0 or index >= textareas.count():
        raise ValueError(f"Visible Relay textarea for node {node_id} was not found")
    widget = textareas.nth(index)
    if widget.input_value() != original:
        raise ValueError(f"Relay Plan {node_id} source text changed before QA edit")
    widget.fill(original + marker)
    if page.evaluate("id => app.graph.getNodeById(id).widgets[0].value", node_id) != original + marker:
        raise ValueError(f"Relay Plan {node_id} ignored its visible text edit")


def _edit_eav_mode(page, *, node_id: int) -> None:
    if (EAV_ORIGINAL, EAV_EDITED) == ("report_only", "apply_exp"):
        _edit_final_eav(page, node_id=node_id)
        return
    page.evaluate("""([id, expected]) => {
        const node = app.graph.getNodeById(id);
        if (!node || node.type !== 'MiniMaxH3StageEAVConfigEXPT8'
                || node.widgets[0].value !== expected) throw Error('EAV source changed');
        app.canvas.centerOnNode(node);
    }""", [node_id, EAV_ORIGINAL])
    page.wait_for_timeout(250)
    point = page.evaluate("""id => {
        const node = app.graph.getNodeById(id);
        const ds = app.canvas.ds;
        return {x: (node.pos[0] + node.size[0] * .75 + ds.offset[0]) * ds.scale,
                y: (node.pos[1] + node.widgets[0].y + 10 + ds.offset[1]) * ds.scale};
    }""", node_id)
    page.mouse.click(point["x"], point["y"])
    option = page.get_by_text(EAV_EDITED, exact=True)
    try:
        option.wait_for(state="visible", timeout=3000)
    except Exception as error:
        menus = page.locator('[role="menu"]').all_text_contents()
        current = page.evaluate("id => app.graph.getNodeById(id).widgets[0].value", node_id)
        geometry = page.evaluate("""([id, point]) => {
            const node = app.graph.getNodeById(id);
            const canvas = app.canvas.canvas;
            const hit = document.elementFromPoint(point.x, point.y);
            return {nodePos: node.pos, nodeSize: node.size,
                    widgetY: node.widgets[0].y, scale: app.canvas.ds.scale,
                    offset: app.canvas.ds.offset,
                    canvasRect: canvas.getBoundingClientRect().toJSON(),
                    hitTag: hit?.tagName, hitClass: hit?.className};
        }""", [node_id, point])
        raise ValueError(f"EAV menu did not expose {EAV_EDITED}: node={node_id}, "
                         f"current={current}, point={point}, menus={menus}, "
                         f"geometry={geometry}") from error
    option.click()
    if page.evaluate("id => app.graph.getNodeById(id).widgets[0].value", node_id) != EAV_EDITED:
        raise ValueError(f"EAV {node_id} ignored its visible mode edit")


def _edits(graph: dict, copy_name: str) -> tuple[list[dict], dict]:
    if copy_name not in EDIT:
        return [], {}
    plans = sorted((node for node in graph["nodes"]
                    if node["type"] == "MiniMaxH3PromptRelayPlanT8Advanced"),
                   key=lambda node: node["id"])
    eav = sorted((node for node in graph["nodes"]
                  if node["type"] == "MiniMaxH3StageEAVConfigEXPT8"),
                 key=lambda node: node["id"])
    expected = (1 if any(marker in copy_name for marker in RESUME_MARKERS) else
                3 if "two_pass_detail_mixer" in copy_name else 2)
    if len(plans) != expected or len(eav) != expected:
        raise ValueError(f"Independent Relay/EAV stage count changed: {copy_name}")
    actions, changes = [], {}
    for plan in plans:
        original = plan["widgets_values"][0]
        marker = f" [{EDIT_MARKER} node {plan['id']}; do not queue]"
        actions.append({"kind": "relay", "node_id": plan["id"],
                        "original": original, "marker": marker})
        changes[(plan["id"], 0)] = original + marker
    last = eav[-1]
    if last["widgets_values"][0] != EAV_ORIGINAL:
        raise ValueError(f"Expected a {EAV_ORIGINAL} final-stage EAV setting")
    actions.append({"kind": "eav", "node_id": last["id"]})
    changes[(last["id"], 0)] = EAV_EDITED
    return actions, changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="Audit already saved copies and continue an interrupted owned QA profile")
    options = parser.parse_args()
    profile, output = options.profile.resolve(), options.output.resolve()
    url = _local_url(options.base_url)
    if (not profile.is_relative_to(PRIVATE.resolve()) or not (profile / "ready.json").is_file()
            or not output.is_relative_to(PRIVATE.resolve()) or output.exists()
            or not re.fullmatch(r"QA_[A-Za-z0-9_-]{1,50}", options.prefix)):
        parser.error("Use an isolated private profile, unused report, and QA prefix")
    ready = json.loads((profile / "ready.json").read_text(encoding="utf-8"))
    if ready.get("url") != url or not ready.get("cpu_only") or not _queue_empty(url):
        parser.error("Expected the matching idle CPU Core")
    recorded = {entry["profile_copy"]: entry for entry in ready["copies"]}
    cases = []
    for route, directory, pattern, expected in SOURCES:
        sources = tuple(sorted(path for path in directory.glob(pattern)
                               if SOURCE_NAMES is None or path.name in SOURCE_NAMES[route]))
        if len(sources) != expected:
            parser.error(f"Expected {expected} sealed {route} candidates")
        for source in sources:
            copy_name = f"{route}_{source.name}"
            copy = profile / "user/default/workflows" / copy_name
            saved_name = f"{options.prefix}_{copy_name}"
            saved = profile / "user/default/workflows" / saved_name
            receipt = recorded.get(str(copy))
            if (not receipt or receipt["source"] != str(source)
                    or receipt["source_sha256"] != _sha(source)
                    or receipt["copy_sha256"] != _sha(source)
                    or not copy.is_file() or _sha(copy) != _sha(source)
                    or (saved.exists() and not options.resume)):
                parser.error(f"Candidate/profile copy changed: {copy_name}")
            graph = json.loads(source.read_bytes())
            actions, changes = _edits(graph, copy_name)
            cases.append({"route": route, "source": source, "source_sha256": _sha(source),
                          "copy": copy_name, "saved": saved_name,
                          "nodes": len(graph["nodes"]), "links": len(graph["links"]),
                          "actions": actions, "edits": changes})
    if {case["copy"] for case in cases if case["actions"]} != EDIT:
        parser.error("The independent edit coverage changed")

    with sync_playwright() as playwright:
        for offset in range(0, len(cases), 5):
            browser = playwright.chromium.launch(headless=True, args=["--disable-gpu"])
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 960})
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_function("() => window.app && app.graph && app.vueAppReady", timeout=30000)
                page.wait_for_timeout(600)
                template = page.locator(
                    '[role="dialog"][aria-labelledby="global-workflow-template-selector"]')
                if template.count() == 1 and template.is_visible():
                    page.keyboard.press("Escape")
                page.get_by_role("dialog").wait_for(state="hidden", timeout=15000)
                for index in range(offset, min(offset + 5, len(cases))):
                    case = cases[index]
                    if options.resume and (profile / "user/default/workflows" / case["saved"]).is_file():
                        print(f"native_browser_existing={index + 1}/{len(cases)}", flush=True)
                        continue
                    _open_stable(page, case["copy"].removesuffix(".json"),
                                 case["nodes"], case["links"])
                    for action in case["actions"]:
                        if action["kind"] == "relay":
                            _edit_node_plan(page, node_id=action["node_id"],
                                            original=action["original"], marker=action["marker"])
                        else:
                            _edit_eav_mode(page, node_id=action["node_id"])
                    _save_as(page, case["saved"].removesuffix(".json"))
                    saved = profile / "user/default/workflows" / case["saved"]
                    deadline = time.monotonic() + 10
                    while not saved.is_file() and time.monotonic() < deadline:
                        page.wait_for_timeout(100)
                    if not saved.is_file():
                        raise ValueError(f"Native Save As missed {case['copy']}")
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("() => window.app && app.graph && app.vueAppReady",
                                           timeout=30000)
                    _open_stable(page, case["saved"].removesuffix(".json"),
                                 case["nodes"], case["links"])
                    for action in case["actions"]:
                        actual = page.evaluate("id => app.graph.getNodeById(id).widgets[0].value",
                                               action["node_id"])
                        if actual != case["edits"][(action["node_id"], 0)]:
                            raise ValueError(f"Native reopen lost {case['copy']} edit {action['node_id']}")
                    print(f"native_browser_roundtrip={index + 1}/{len(cases)}", flush=True)
            finally:
                browser.close()

    if not _queue_empty(url) or _json_get(url + "/history"):
        raise ValueError("Browser QA unexpectedly queued inference")
    info = _json_get(url + "/object_info")
    current = {name: {"id": name, "info": schema} for name, schema in info.items()}
    audited = []
    for case in cases:
        saved = profile / "user/default/workflows" / case["saved"]
        before, after = json.loads(case["source"].read_bytes()), json.loads(saved.read_bytes())
        normalized, changes = normalize_known_ui_changes(
            before, after, current, labels_sha256=_sha(PROJECT / "web/task_type_labels.js"))
        appends = schema_default_appends(before, normalized, current)
        semantic = audit_roundtrip(before, normalized, edits=case["edits"],
                                   appended_widgets=appends)
        audited.append({"route": case["route"], "source": str(case["source"]),
                        "source_sha256": case["source_sha256"], "saved": str(saved),
                        "saved_sha256": _sha(saved), "semantic_audit": semantic,
                        "ui_normalizations": changes,
                        "schema_default_appends": {str(key): value for key, value in appends.items()}})
    report = {"schema": REPORT_SCHEMA,
              "status": "pass_browser_serialization_only", "url": url,
              "candidate_count": len(audited), "edited_case_count": len(EDIT),
              "queue_and_history_empty": True, "cases": audited,
              "limits": REPORT_LIMITS}
    write_new(output, report)
    print(json.dumps({"status": report["status"], "cases": len(audited),
                      "edited": len(EDIT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
