"""Native save/reopen QA for all 81 S13 external EAV/Relay stage candidates.

Six combined/both save/resume graphs also receive visible independent edits.
Neither the HEAD artifact placeholders nor any QA copy may be queued.
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
    _edit_visible_plan,
    _json_get,
    _local_url,
    _queue_empty,
    _save_as,
)


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "artifacts/development"
CANDIDATE = PRIVATE / "modular-sampling-m3-hyperflow-effects-20260923/candidate-v1"
EDIT_NAMES = frozenset(
    f"HyperFlow_{split}plus{8-split}_combined_both_{variant}_EXP.json"
    for split in (1, 4, 7)
    for variant in ("save", "resume_tail")
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_edits(graph: dict, source_name: str) -> tuple[list[dict], dict]:
    if source_name not in EDIT_NAMES:
        return [], {}
    plans = sorted((node for node in graph["nodes"]
                    if node["type"] == "MiniMaxH3PromptRelayPlanT8Advanced"),
                   key=lambda node: node["id"])
    eav_nodes = sorted((node for node in graph["nodes"]
                        if node["type"] == "MiniMaxH3StageEAVConfigEXPT8"),
                       key=lambda node: node["id"])
    expected = 1 if "resume_tail" in source_name else 2
    if len(plans) != expected or len(eav_nodes) != expected:
        raise ValueError(f"Unexpected independent Relay/EAV node count: {source_name}")
    edits: dict[tuple[int, int], str] = {}
    actions = []
    for plan in plans:
        original = plan["widgets_values"][0]
        marker = f" [S13 node {plan['id']} browser QA; do not queue]"
        edits[(plan["id"], 0)] = original + marker
        actions.append({"kind": "relay", "node_id": plan["id"],
                        "before": original, "marker": marker})
    tail_eav = eav_nodes[-1]
    if tail_eav["widgets_values"][0] != "report_only":
        raise ValueError("Unexpected original TAIL EAV mode")
    edits[(tail_eav["id"], 0)] = "apply_exp"
    actions.append({"kind": "eav", "node_id": tail_eav["id"]})
    return actions, edits


def _audit(case: dict, profile: Path, current_nodes: dict) -> dict:
    source = CANDIDATE / case["source"]
    saved = profile / "user/default/workflows" / case["saved"]
    if _sha(source) != case["sha"] or not saved.is_file():
        raise ValueError("Source changed or native saved graph missing")
    before, after = json.loads(source.read_bytes()), json.loads(saved.read_bytes())
    normalized, ui_changes = normalize_known_ui_changes(
        before, after, current_nodes, labels_sha256=_sha(ROOT / "web/task_type_labels.js"))
    appends = schema_default_appends(before, normalized, current_nodes)
    semantic = audit_roundtrip(before, normalized, edits=case["edits"],
                               appended_widgets=appends)
    return {"source": source.relative_to(ROOT).as_posix(),
            "source_sha256": case["sha"],
            "saved": saved.relative_to(profile).as_posix(),
            "saved_sha256": _sha(saved),
            "semantic_audit": semantic,
            "ui_normalizations": ui_changes,
            "schema_default_appends": {str(key): value for key, value in appends.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile, output = args.profile.resolve(), args.output.resolve()
    base_url = _local_url(args.base_url)
    if (not profile.is_relative_to(PRIVATE.resolve()) or not (profile / "ready.json").is_file()
            or not output.is_relative_to(PRIVATE.resolve()) or output.exists()
            or not re.fullmatch(r"QA_[A-Za-z0-9_-]{1,50}", args.prefix)):
        parser.error("Use isolated private profile, new report and QA prefix")
    ready = json.loads((profile / "ready.json").read_text(encoding="utf8"))
    if ready.get("url") != base_url or not ready.get("cpu_only") or not _queue_empty(base_url):
        parser.error("Expected an idle isolated CPU Core")

    sources = tuple(sorted(CANDIDATE.glob("HyperFlow_*_EXP.json")))
    if len(sources) != 81 or set(EDIT_NAMES) - {path.name for path in sources}:
        parser.error("Expected all 81 S13 frontend candidates")
    copies = {entry["source"]: entry for entry in ready.get("copies", [])}
    if len(copies) != len(sources):
        parser.error("The isolated profile did not receive every candidate copy")
    cases = []
    for source in sources:
        sha = _sha(source)
        copy_name = "S13_" + source.name
        copy = profile / "user/default/workflows" / copy_name
        saved_name = f"{args.prefix}_{source.name}"
        saved = profile / "user/default/workflows" / saved_name
        record = copies.get(str(source))
        if (not record or record.get("source_sha256") != sha
                or record.get("copy_sha256") != sha
                or not copy.is_file() or _sha(copy) != sha or saved.exists()):
            parser.error(f"Candidate/profile copy changed: {source.name}")
        graph = json.loads(source.read_bytes())
        actions, edits = _selected_edits(graph, source.name)
        cases.append({"source": source.name, "sha": sha, "copy": copy_name,
                      "saved": saved_name, "nodes": len(graph["nodes"]),
                      "links": len(graph["links"]), "actions": actions, "edits": edits})

    with sync_playwright() as playwright:
        for offset in range(0, len(cases), 8):
            browser = playwright.chromium.launch(headless=True, args=["--disable-gpu"])
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 960})
                page.goto(base_url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_function("() => window.app && app.graph && app.vueAppReady",
                                       timeout=30000)
                page.wait_for_timeout(1000)
                template = page.locator(
                    '[role="dialog"][aria-labelledby="global-workflow-template-selector"]')
                if template.count() == 1 and template.is_visible():
                    page.keyboard.press("Escape")
                page.get_by_role("dialog").wait_for(state="hidden", timeout=15000)
                for index in range(offset, min(offset + 8, len(cases))):
                    case = cases[index]
                    _open_stable(page, case["copy"].removesuffix(".json"),
                                 case["nodes"], case["links"])
                    for action in case["actions"]:
                        if action["kind"] == "relay":
                            _edit_visible_plan(page, action["before"],
                                               node_id=action["node_id"], marker=action["marker"])
                        else:
                            _edit_final_eav(page, node_id=action["node_id"])
                    _save_as(page, case["saved"].removesuffix(".json"))
                    target = profile / "user/default/workflows" / case["saved"]
                    deadline = time.monotonic() + 10
                    while not target.is_file() and time.monotonic() < deadline:
                        page.wait_for_timeout(100)
                    if not target.is_file():
                        raise ValueError(f"Native Save As missed {case['saved']}")
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("() => window.app && app.graph && app.vueAppReady",
                                           timeout=30000)
                    page.wait_for_timeout(500)
                    _open_stable(page, case["saved"].removesuffix(".json"),
                                 case["nodes"], case["links"])
                    for action in case["actions"]:
                        actual = page.evaluate("id => app.graph.getNodeById(id).widgets[0].value",
                                               action["node_id"])
                        expected = case["edits"][(action["node_id"], 0)]
                        if actual != expected:
                            raise ValueError(
                                f"Native reopen lost edit {case['source']}:{action['node_id']}")
                    if (index + 1) % 5 == 0 or index + 1 == len(cases):
                        print(f"native_browser_roundtrip={index + 1}/{len(cases)}", flush=True)
            finally:
                browser.close()

    if not _queue_empty(base_url) or _json_get(base_url + "/history"):
        raise ValueError("Browser QA unexpectedly queued inference")
    object_info = _json_get(base_url + "/object_info")
    current_nodes = {name: {"id": name, "info": info} for name, info in object_info.items()}
    audited = [_audit(case, profile, current_nodes) for case in cases]
    report = {"schema": "t8.modular-sampling.s13-native-browser-matrix.v1",
              "status": "pass_browser_serialization_only",
              "profile": profile.relative_to(ROOT).as_posix(),
              "base_url": base_url, "queue_and_history_empty": True,
              "candidate_count": len(audited), "edited_case_count": len(EDIT_NAMES),
              "cases": audited,
              "qualification": "All 81 S13 native frontend candidates opened, saved, browser-reloaded, "
              "reopened and audited for node IDs, named edges and widget values. Six combined/both "
              "save/resume graphs had visible independent Relay/EAV edits. Draft HEAD paths "
              "are unbound; no graph was queued. No trained GPU/media/human-quality claim."}
    write_new(output, report)
    print(json.dumps({"status": report["status"], "cases": len(audited),
                      "edited": len(EDIT_NAMES)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
