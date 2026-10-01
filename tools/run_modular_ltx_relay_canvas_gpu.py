"""Queue one private real-weight S27 Relay workflow through the native canvas.

Read-only preflight unless --confirm-run. This never queues on the user Core,
alters a public workflow, or claims subjective picture/audio quality.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlparse
import urllib.request

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from tools import run_modular_ltx_relay_gpu as relay  # noqa: E402
from tools import run_modular_ltx_rgb_source_gpu as source  # noqa: E402
from tools.run_candidate_cpu_regression import source_snapshot  # noqa: E402
from tools.run_modular_hyperflow_fresh_browser_matrix import _open_stable  # noqa: E402
from tools.run_modular_s26_pdd_resume_gpu import shared  # noqa: E402
from tools.run_modular_s22_browser_roundtrip import _json_get, _queue_empty  # noqa: E402
from tools.vdn_probe_environment import probe_resource_config  # noqa: E402

GLOBAL = "A woman in a red raincoat cycles through a rainy city street, one continuous cinematic shot."
LOCAL = "She pedals past shop windows.\nShe rounds the corner and rides away."
MEDIA_PREFIX = "MiniMaxH3/S27_Canvas_Relay/candidate"


def prepared_frontend(original: dict, *, width: int, height: int) -> dict:
    """Fill only visible workflow widgets; leave wiring and old sources alone."""
    graph = deepcopy(original)
    nodes = {node["type"]: node for node in graph["nodes"]}
    setups = [node for node in graph["nodes"] if node["type"] in relay.builder.SETUPS]
    if len(setups) != 1:
        raise ValueError("The saved S27 Relay graph's setup inventory changed")
    setup = setups[0]
    required = {"LoadVideo", "MiniMaxH3SolEngineDraftToLTXT8Advanced",
                setup["type"], relay.builder.PLAN,
                relay.builder.APPLY, source.builder.SAVE, source.builder.ISOLATED_WRITER}
    if (not required.issubset(nodes) or any(
            sum(node["type"] == kind for node in graph["nodes"]) != 1 for kind in required)):
        raise ValueError("The saved S27 Relay graph's visible node inventory changed")
    nodes["LoadVideo"]["widgets_values"][0] = "source.mp4"
    nodes["MiniMaxH3SolEngineDraftToLTXT8Advanced"]["widgets_values"][:2] = [width, height]
    backend_index = (1 if setup["type"] == "MiniMaxH3SolEngineLTXRefinerSetupT8Advanced"
                     else 3 if setup["type"] == "MiniMaxH3SolEngineLTXIdentityRefinerSetupT8Advanced"
                     else None)
    if backend_index is None:
        raise ValueError("Unknown LTX setup widget layout")
    verbose_index = 4 if backend_index == 1 else 6
    if setup["widgets_values"][verbose_index] is not False:
        raise ValueError("Saved LTX setup verbose control changed")
    setup["widgets_values"][backend_index] = "dense_reference"
    nodes[relay.builder.PLAN]["widgets_values"][:2] = [GLOBAL, LOCAL]
    nodes[relay.builder.APPLY]["widgets_values"][0] = "apply_exp"
    eav = [node for node in graph["nodes"] if node["type"] == "MiniMaxH3StageEAVConfigEXPT8"]
    if len(eav) > 1 or (eav and eav[0]["widgets_values"][0] != "report_only"):
        raise ValueError("Combined canvas EAV must remain report-only")
    if nodes[source.builder.SAVE]["widgets_values"][-1] is not False:
        raise ValueError("Private canvas run must not publish a source checkpoint")
    nodes[source.builder.ISOLATED_WRITER]["widgets_values"][0] = MEDIA_PREFIX
    if graph["links"] != original["links"] or len(graph["nodes"]) != len(original["nodes"]):
        raise ValueError("Preparing widgets unexpectedly changed workflow structure")
    return graph


def _history(url: str, prompt_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = _json_get(url + "/history/" + prompt_id).get(prompt_id)
        if result is not None:
            return result
        time.sleep(2)
    raise TimeoutError("Native canvas prompt did not reach history before its deadline")


def normalize_empty_preview(prequeue: dict, posted: dict) -> dict:
    """Allow only Core's empty, non-executing LoadVideo preview display field."""
    result = deepcopy(posted)
    if set(result) != set(prequeue) or result.get("1", {}).get("class_type") != "LoadVideo":
        raise ValueError("Submitted node set or LoadVideo class differs")
    inputs = result["1"].get("inputs")
    if not isinstance(inputs, dict) or inputs.pop("video-preview", None) != "":
        raise ValueError("LoadVideo preview was not the exact empty frontend field")
    if result != prequeue:
        raise ValueError("Canvas POST changed another executable input or edge")
    return result


def _port_owner(port: int) -> int | None:
    import psutil
    owners = {row.pid for row in psutil.net_connections(kind="tcp")
              if row.laddr and row.laddr.port == port and row.status == psutil.CONN_LISTEN}
    if len(owners) != 1:
        return None
    return next(iter(owners))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("ordinary", "identity"), default="ordinary")
    parser.add_argument("--combined-eav", action="store_true")
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--port", type=int, default=8958)
    parser.add_argument("--comfy-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    args.source_video = args.source_video.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.host = "127.0.0.1"
    args.use_pytorch_cross_attention = True
    if args.port in (8188, 8189, 8940):
        parser.error("Never use a user ComfyUI port for canvas qualification")
    name, original, api = relay._saved(args.route, args.combined_eav, relay.SAVED)
    ready = source.preflight(args, {"full_save": (original, api)})
    print(json.dumps({"schema": "t8.s27.relay-canvas.preflight.v1", **ready}), flush=True)
    if not args.confirm_run or not ready["ready"]:
        return 0 if ready["ready"] else 2

    prepared = prepared_frontend(original, width=args.width, height=args.height)
    run = ROOT / "artifacts/development/modular-ltx-relay-20260928/canvas" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.route
        + ("-eav-relay" if args.combined_eav else "-relay"))
    run.mkdir(parents=True, exist_ok=False)
    print("RUN_ROOT=" + str(run), flush=True)
    before = source_snapshot(ROOT)
    relay._write(run / "sources-before.json", before)
    relay._write(run / "prepared.frontend.json", prepared)
    args.input_directory = run / "input"
    args.input_directory.mkdir()
    source_sha = source.sha(args.source_video)
    shutil.copyfile(args.source_video, args.input_directory / "source.mp4")
    if source.sha(args.input_directory / "source.mp4") != source_sha:
        raise ValueError("Owned input video copy differs")
    paths = probe_resource_config(args.comfy_root, ROOT)
    paths["t8_runtime_models"]["taehv"] = str(args.comfy_root / "models/taehv")
    args.extra_model_paths_config = run / "paths.json"
    relay._write(args.extra_model_paths_config, paths)
    workflow_name = "QA_S27_Relay_Canvas_" + args.route + (
        "_EAV_Full_Save" if args.combined_eav else "_Full_Save")
    workflow = run / "user/default/workflows" / (workflow_name + ".json")
    workflow.parent.mkdir(parents=True)
    relay._write(workflow, prepared)
    user_port_before = _port_owner(8940)
    report = {"schema": "t8.s27.relay-native-canvas-gpu.v1", "status": "started",
              "saved_graph": name, "route": args.route, "combined_eav": args.combined_eav,
              "run_root": str(run), "source_sha256": source_sha,
              "preflight": ready, "checks": {}, "boundary":
              "One private ordinary full-save canvas-queued real-weight media run; not cold canvas, "
              "all routes, long-video, multi-source or subjective quality acceptance."}
    server = shared.IsolatedServer(args, run, "canvas")
    try:
        with server:
            url = f"http://127.0.0.1:{args.port}"
            with urllib.request.urlopen(url + "/object_info", timeout=30) as response:
                info = json.load(response)
            expected = source.builder.split_api(prepared, info)
            relay._write(run / "prepared.api.json", expected)
            if not _queue_empty(url) or _json_get(url + "/history"):
                raise ValueError("Owned Core must start with an empty queue and history")
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True, args=["--disable-gpu"])
                try:
                    page = browser.new_page(viewport={"width": 1440, "height": 960})
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_function("() => window.app && app.graph && app.vueAppReady", timeout=30000)
                    page.wait_for_timeout(600)
                    dialog = page.locator('[role="dialog"][aria-labelledby="global-workflow-template-selector"]')
                    if dialog.count() == 1 and dialog.is_visible():
                        page.keyboard.press("Escape")
                    page.get_by_role("dialog").wait_for(state="hidden", timeout=15000)
                    _open_stable(page, workflow_name, len(prepared["nodes"]), len(prepared["links"]))
                    generated = page.evaluate("async () => await app.graphToPrompt()")
                    output = generated["output"]
                    if output != expected:
                        raise ValueError("Native canvas serialization differs from prepared live-schema API")
                    relay._write(run / "canvas-before-queue.api.json", output)
                    page.screenshot(path=str(run / "canvas-before-queue.png"), full_page=True)
                    button = page.get_by_test_id("queue-button")
                    if button.count() != 1 or not button.is_visible() or not button.is_enabled():
                        raise ValueError("Native queue button is not uniquely visible and enabled")
                    posted, responses = [], []
                    def is_prompt(request):
                        return request.method == "POST" and urlparse(request.url).path.endswith("/prompt")
                    page.on("request", lambda request: posted.append(request) if is_prompt(request) else None)
                    page.on("response", lambda response: responses.append(response)
                            if is_prompt(response.request) else None)
                    button.click()
                    deadline = time.monotonic() + 30
                    while not responses and time.monotonic() < deadline:
                        page.wait_for_timeout(100)
                    if len(posted) != 1 or len(responses) != 1:
                        raise ValueError("Native Queue click did not yield exactly one prompt request/response: "
                                         + repr([item.url for item in posted]))
                    response = responses[0]
                    submitted = response.json()
                    post = json.loads(posted[0].post_data)
                    report["canvas"] = {"queue_button_testid": "queue-button", "request_url": posted[0].url,
                                        "response_status": response.status,
                                        "response": submitted, "pid": server.process.pid,
                                        "workflow_name": workflow_name}
                    relay._write(run / "canvas-post.json", post)
                    try:
                        normalize_empty_preview(output, post.get("prompt") or {})
                        report["checks"]["browser_prompt_matches_preflight"] = True
                    except ValueError:
                        report["checks"]["browser_prompt_matches_preflight"] = False
                    if not report["checks"]["browser_prompt_matches_preflight"]:
                        report["canvas"]["differing_prompt_nodes"] = sorted(
                            key for key in set(output) | set(post.get("prompt") or {})
                            if output.get(key) != (post.get("prompt") or {}).get(key))
                    if response.status != 200:
                        raise ValueError("Native canvas prompt rejected; inspect saved response")
                    prompt_id = submitted["prompt_id"]
                    report["canvas"]["prompt_id"] = prompt_id
                    print("CANVAS_PROMPT_ID=" + prompt_id, flush=True)
                    history = _history(url, prompt_id, args.timeout_seconds)
                    relay._write(run / "history.json", history)
                finally:
                    browser.close()
            status = history.get("status") or {}
            report["checks"]["native_canvas_completed"] = (
                status.get("status_str") == "success" and status.get("completed") is True)
            report["checks"]["history_prompt_exact"] = (
                isinstance(history.get("prompt"), list) and len(history["prompt"]) >= 3
                and history["prompt"][1] == prompt_id and history["prompt"][2] == post["prompt"])
            report["checks"]["owned_queue_empty"] = _queue_empty(url)
            if not all(report["checks"].values()):
                raise RuntimeError("Native canvas run or history failed its execution checks")
        report["media"] = source.media(run, MEDIA_PREFIX, isolated=True)
        video = next(s for s in report["media"]["streams"] if s["codec_type"] == "video")
        report["checks"].update(strict_complete_av=report["media"]["fully_decoded"],
                                expected_geometry_frames=(video["width"], video["height"],
                                                          int(video.get("nb_frames", 0))) ==
                                (args.width, args.height, 113),
                                original_audio_sha=(report["media"]["decoded_sha"]["audio"] ==
                                                    "9e6844a213817e8a712e6d44a4b28bcc590b115f01a8457605c0d2cbecc68adc"))
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["checks"].update(source_stable=source_snapshot(ROOT) == before,
                                source_video_stable=source.sha(args.source_video) == source_sha,
                                user_8940_untouched=(user_port_before is not None
                                                    and user_port_before == _port_owner(8940)),
                                owned_service_stopped=(server.process is None or server.process.poll() is not None),
                                owned_port_closed=not shared.port_is_listening(args.host, args.port))
        report["status"] = ("pass_native_canvas_single_route_mechanical_not_quality" if
                            "error" not in report and all(report["checks"].values()) else "fail")
        relay._write(run / "report.json", report)
        print(json.dumps({"run_root": str(run), "status": report["status"],
                          "failed": [key for key, value in report["checks"].items() if not value]}), flush=True)
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
