"""Native open/Save As/reload/reopen of the 51 in-scope frozen old graphs.

Every source and private copy is SHA-bound to the original M0 baseline. An
individual browser failure is reported, not silently excluded or retried.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

from tools.audit_modular_browser_roundtrip import audit_roundtrip
from tools.audit_modular_legacy_browser_save import audit_pair
from tools.audit_modular_sampling_compat import write_new
from tools.run_modular_hyperflow_fresh_browser_matrix import _open_stable
from tools.run_modular_s22_browser_roundtrip import _local_url, _queue_empty, _save_as
from tools.serve_modular_m0_legacy_browser import BASELINE, PRIVATE, ROOT, SCOPE, selected

REPORT_SCHEMA = "t8.modular-sampling.m0-legacy-browser-matrix.v1"
SCOPE_LABEL = "51-graph"
BRIDGES = ("legacy_markdown_note_widgets.js", "legacy_m0_widget_slots.js")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get(url: str) -> dict:
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def _audit_reopened(saved: dict, reopened: dict) -> dict:
    before, after = deepcopy(saved), deepcopy(reopened)
    previous = {node["id"]: node for node in before["nodes"]}
    for node in after["nodes"]:
        old = previous.get(node["id"], {})
        if list(old.get("widgets_values_named") or {}) != list(node.get("widgets_values_named") or {}):
            raise ValueError(f"Reopen changed widget names: {node['id']}")
    # VHS intentionally stores a named dictionary, not a positional array.
    # Convert both exact dictionaries to ordered values for the array auditor.
    for graph in (before, after):
        for node in graph["nodes"]:
            values = node.get("widgets_values")
            if isinstance(values, dict):
                if node["type"] != "VHS_VideoCombine" or list(values) != list(
                        node.get("widgets_values_named") or {}):
                    raise ValueError("Unexpected named widget dictionary on reopen")
                node["widgets_values"] = list(values.values())
    return audit_roundtrip(before, after)


def _open_one(page, case: dict) -> dict:
    case["phase"] = "initial_navigation"
    page.goto(page.url if page.url != "about:blank" else case["url"],
              wait_until="domcontentloaded", timeout=30000)
    case["phase"] = "initial_app_ready"
    page.wait_for_function("() => window.app && app.graph && app.vueAppReady", timeout=30000)
    page.wait_for_timeout(500)
    template = page.locator('[role="dialog"][aria-labelledby="global-workflow-template-selector"]')
    if template.count() == 1 and template.is_visible():
        page.keyboard.press("Escape")
    page.get_by_role("dialog").wait_for(state="hidden", timeout=15000)
    case["phase"] = "open_frozen_graph"
    _open_stable(page, case["copy_name"].removesuffix(".json"),
                 case["nodes"], case["links"])
    case["phase"] = "native_save_as"
    _save_as(page, case["saved_name"].removesuffix(".json"))
    saved = case["saved_path"]
    deadline = time.monotonic() + 10
    while not saved.is_file() and time.monotonic() < deadline:
        page.wait_for_timeout(100)
    if not saved.is_file():
        raise ValueError("Native Save As did not create the isolated old-graph copy")
    page.reload(wait_until="domcontentloaded", timeout=30000)
    case["phase"] = "reloaded_app_ready"
    page.wait_for_function("() => window.app && app.graph && app.vueAppReady", timeout=30000)
    _open_stable(page, case["saved_name"].removesuffix(".json"),
                 case["nodes"], case["links"])
    case["phase"] = "reopened_semantic_audit"
    # Native Save As serializes the graph, whereas graphToPrompt deliberately
    # removes disconnected widget sockets. Compare the same representation.
    reopened = page.evaluate("() => app.graph.serialize()")
    semantic = _audit_reopened(json.loads(saved.read_bytes()), reopened)
    resources = page.evaluate("() => performance.getEntriesByType('resource').map(item => item.name)")
    served = {name: sorted({url for url in resources
                           if url.split("?", 1)[0].endswith("/" + name)})
              for name in BRIDGES}
    served_hashes = {}
    for name, urls in served.items():
        if len(urls) != 1:
            raise ValueError(f"Native browser did not load the bridge URL: {name}")
        # Separate read-only HTTP check of those exact observed URLs. This is
        # server-byte provenance, not a claim to capture browser response bodies.
        with urlopen(urls[0], timeout=10) as response:
            served_hashes[name] = hashlib.sha256(response.read()).hexdigest()
        if served_hashes[name] != _sha(ROOT / "web" / name):
            raise ValueError(f"Served bridge differs from local source: {name}")
    return {"reopened_semantic_audit": semantic,
            "loaded_bridge_resource_urls": served, "served_bridge_sha256": served_hashes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--browser-observation", type=Path)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--browser-channel", choices=("headless-shell", "chrome"),
                        default="headless-shell")
    parser.add_argument("--browser-executable", type=Path,
                        help="Explicit existing Chromium binary; never downloads a browser")
    options = parser.parse_args()
    executable = options.browser_executable.resolve() if options.browser_executable else None
    if executable and (not executable.is_file() or options.browser_channel != "headless-shell"):
        parser.error("Use an existing executable without a simultaneous browser channel")
    profile, output = options.profile.resolve(), options.output.resolve()
    url = _local_url(options.base_url)
    ready_path = profile / "ready.json"
    if (not profile.is_relative_to(PRIVATE) or not ready_path.is_file()
            or not output.is_relative_to(PRIVATE) or output.exists() or not _queue_empty(url)):
        parser.error("Use an idle isolated CPU profile and unused private report")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    full_expected = selected()
    if (ready.get("url") != url or ready.get("cpu_only") is not True
            or ready.get("baseline_sha256") != _sha(BASELINE)
            or ready.get("scope_sha256") != _sha(SCOPE)
            or ready.get("cases") != full_expected or _get(url + "/history")):
        parser.error(f"Isolated Core or frozen {SCOPE_LABEL} inventory changed")
    if options.indices:
        chosen = set(options.indices)
        if len(chosen) != len(options.indices) or not chosen.issubset(
                range(1, len(full_expected) + 1)):
            parser.error(f"Use distinct 1–{len(full_expected)} frozen case indices")
        expected = [item for item in full_expected if item["index"] in chosen]
    else:
        expected = full_expected
    cases = []
    workflows = profile / "user/default/workflows"
    observation = None
    if options.audit_only:
        if options.browser_observation is None:
            parser.error("Audit-only requires the prior native browser observation")
        observed_path = options.browser_observation.resolve()
        if not observed_path.is_relative_to(PRIVATE) or not observed_path.is_file():
            parser.error("Browser observation must be an existing private receipt")
        observation = json.loads(observed_path.read_text(encoding="utf-8"))
        if (observation.get("schema") != REPORT_SCHEMA
                or observation.get("url") != url or observation.get("in_scope") != len(expected)
                or observation.get("queue_and_history_empty") is not True
                or [item.get("original") for item in observation.get("cases", [])]
                != [item["relative"] for item in expected]):
            parser.error(f"Prior browser observation does not cover this exact {SCOPE_LABEL} scope")
    elif options.browser_observation is not None:
        parser.error("--browser-observation is only for --audit-only")
    for item in expected:
        copy = workflows / item["copy_name"]
        saved = workflows / item["saved_name"]
        if (not copy.is_file() or _sha(copy) != item["sha256"]
                or saved.exists() != options.audit_only):
            parser.error("Private legacy graph copy changed or saved target exists")
        graph = json.loads(copy.read_bytes())
        cases.append({**item, "url": url, "links": len(graph["links"]),
                      "saved_path": saved})
    info = _get(url + "/object_info")
    current = {"nodes": [{"id": name, "info": schema} for name, schema in info.items()]}
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    audit_source_sha256 = _sha(ROOT / "tools/audit_modular_legacy_browser_save.py")
    runner_source_sha256 = _sha(Path(__file__))
    bridge_source_sha256 = {name: _sha(ROOT / "web" / name) for name in BRIDGES}
    audited = []
    def audit_case(case: dict, browser_evidence: dict | None = None) -> None:
        try:
            result = audit_pair(ROOT, profile, baseline, current,
                                case["relative"], case["saved_name"])
            audited.append({"status": "pass", "index": case["index"], **result,
                            **(browser_evidence or {})})
        except Exception as error:
            audited.append({"status": "fail", "index": case["index"],
                            "original": case["relative"],
                            "error_type": type(error).__name__, "reason": str(error)[:800],
                            "traceback": "".join(traceback.format_exception(error))[-2000:]})
        print(f"legacy_native_roundtrip={len(audited)}/{len(cases)} index={case['index']} "
              f"{audited[-1]['status']}", flush=True)

    if options.audit_only:
        for case in cases:
            audit_case(case)
    else:
        with sync_playwright() as playwright:
            for case in cases:
                browser = None
                page = None
                errors = []
                try:
                    browser = playwright.chromium.launch(
                        headless=True, args=["--disable-gpu"],
                        channel="chrome" if options.browser_channel == "chrome" else None,
                        executable_path=str(executable) if executable else None)
                    page = browser.new_page(viewport={"width": 1440, "height": 960})
                    page.add_init_script("performance.setResourceTimingBufferSize(8192)")
                    page.on("pageerror", lambda error: errors.append(str(error)[:1000]))
                    browser_evidence = _open_one(page, case)
                    browser_evidence["browser_version"] = browser.version
                    audit_case(case, browser_evidence)
                except Exception as error:
                    diagnostics = {"phase": case.get("phase"), "page_errors": errors[-10:]}
                    try:
                        screenshot = output.with_name(output.stem + f"-case-{case['index']}.png")
                        page.screenshot(path=str(screenshot), timeout=3000)
                        diagnostics["screenshot"] = str(screenshot)
                    except Exception as capture_error:
                        diagnostics["screenshot_error"] = str(capture_error)[:200]
                    audited.append({"status": "fail", "index": case["index"],
                                    "original": case["relative"],
                                    "error_type": type(error).__name__, "reason": str(error)[:800],
                                    "diagnostics": diagnostics})
                    print(f"legacy_native_roundtrip={len(audited)}/{len(cases)} "
                          f"index={case['index']} fail", flush=True)
                finally:
                    if browser is not None:
                        try:
                            browser.close()
                        except Exception as close_error:
                            audited[-1].update(status="fail", close_error=str(close_error)[:500])
                    write_new(output.with_suffix("").with_name(output.stem + "-cases") /
                              f"case-{case['index']:03}.json", audited[-1])
    queue_idle = _queue_empty(url) and not _get(url + "/history")
    sources_stable = (audit_source_sha256 == _sha(ROOT / "tools/audit_modular_legacy_browser_save.py")
                      and runner_source_sha256 == _sha(Path(__file__))
                      and bridge_source_sha256 == {
                          name: _sha(ROOT / "web" / name) for name in BRIDGES})
    report = {"schema": REPORT_SCHEMA,
              "status": "pass_browser_serialization_only" if queue_idle and sources_stable and all(
                  item["status"] == "pass" for item in audited) else "fail",
              "url": url, "baseline_sha256": _sha(BASELINE),
              "scope_sha256": _sha(SCOPE), "in_scope": len(cases),
              "audit_only": options.audit_only,
              "browser_channel": None if options.audit_only else options.browser_channel,
              "driver_node_override": os.environ.get("PLAYWRIGHT_NODEJS_PATH"),
              "browser_executable": str(executable) if executable else None,
              "browser_executable_sha256": _sha(executable) if executable else None,
              "audit_source_sha256": audit_source_sha256,
              "runner_source_sha256": runner_source_sha256,
              "bridge_source_sha256": bridge_source_sha256,
              "sources_unchanged_during_run": sources_stable,
              "browser_observation_sha256": _sha(options.browser_observation.resolve())
              if observation else None,
              "passed": sum(item["status"] == "pass" for item in audited),
              "failed": sum(item["status"] == "fail" for item in audited),
              "out_of_scope_frontend": 306 - len(cases),
              "queue_and_history_empty": queue_idle, "cases": audited,
              "limits": f"Native browser serialization of this {SCOPE_LABEL} scope only; "
                        "not GPU execution, media, numerical parity, human review, or all 306 old graphs."}
    write_new(output, report)
    print(json.dumps({"status": report["status"], "passed": report["passed"],
                      "failed": report["failed"]}, ensure_ascii=False))
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
