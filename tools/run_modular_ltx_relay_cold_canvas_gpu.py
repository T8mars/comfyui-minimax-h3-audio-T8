"""Queue one frozen-input LTX Relay tail through an owned native canvas.

Read-only preflight unless --confirm-run. The historical API checkpoint is
copied by content identity; no source video is mounted in the owned Core.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.parse import urlparse
import urllib.request

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from tools import run_modular_ltx_relay_canvas_gpu as full_canvas  # noqa: E402
from tools import run_modular_ltx_relay_gpu as relay  # noqa: E402
from tools import run_modular_ltx_rgb_source_gpu as source  # noqa: E402
from tools.audit_modular_ltx_relay_canvas_gpu import API_BASELINES, ORIGINAL_AUDIO_SHA  # noqa: E402
from tools.run_candidate_cpu_regression import source_snapshot  # noqa: E402
from tools.run_modular_hyperflow_fresh_browser_matrix import _open_stable  # noqa: E402
from tools.run_modular_s22_browser_roundtrip import _json_get, _queue_empty  # noqa: E402
from tools.run_modular_s26_pdd_resume_gpu import shared  # noqa: E402
from tools.vdn_probe_environment import probe_resource_config  # noqa: E402

MEDIA_PREFIX = "MiniMaxH3/S27_Canvas_Relay/cold_candidate"


def no_original_video_mounted(input_directory: Path) -> bool:
    """Allow only Core's empty 3d input directory, never any mounted file."""
    entries = list(input_directory.iterdir())
    if not entries:
        return True
    return (len(entries) == 1 and entries[0].name == "3d"
            and entries[0].is_dir() and not entries[0].is_symlink()
            and not any(entries[0].iterdir()))


def control(route: str, combined: bool) -> tuple[Path, dict, dict]:
    path = (ROOT / "artifacts/development/modular-ltx-relay-20260928/trained"
            / API_BASELINES[(route, combined)] / "report.json")
    report = json.loads(path.read_text(encoding="utf8"))
    if (report["status"] != "full_cold_relay_media_mechanical_not_canvas_or_quality"
            or not all(report["checks"].values())):
        raise ValueError("The frozen API control is not fully mechanically qualified")
    receipt = report["source_receipt"]
    for key, digest in (("manifest_file", "sha"), ("state_file", "state_sha")):
        if source.sha(Path(receipt[key])) != receipt[digest]:
            raise ValueError("The API source checkpoint changed: " + key)
    for phase in ("full", "cold"):
        media = report["media"][phase]
        if not media["fully_decoded"] or source.sha(Path(media["path"])) != media["sha"]:
            raise ValueError("The API media control changed: " + phase)
    if report["media"]["full"]["decoded_sha"] != report["media"]["cold"]["decoded_sha"]:
        raise ValueError("The API full/cold decoded media are not equal")
    return path, report, receipt


def prepared_frontend(original: dict, receipt: dict) -> dict:
    """Edit only cold graph's visible checkpoint, setup, Relay and output widgets."""
    graph = deepcopy(original)
    nodes = {node["type"]: node for node in graph["nodes"]}
    setups = [node for node in graph["nodes"] if node["type"] in relay.builder.SETUPS]
    if len(setups) != 1:
        raise ValueError("The cold Relay setup inventory changed")
    setup = setups[0]
    required = {source.builder.LOAD, setup["type"], relay.builder.PLAN,
                relay.builder.APPLY, source.builder.ISOLATED_WRITER}
    if (not required.issubset(nodes) or any(
            sum(node["type"] == kind for node in graph["nodes"]) != 1 for kind in required)
            or any(node["type"] in {"LoadVideo", "VAEEncode", "LTXVLatentUpsampler"}
                   for node in graph["nodes"])):
        raise ValueError("The cold Relay graph still contains source preparation or lost a required node")
    load = nodes[source.builder.LOAD]
    if load["widgets_values"] != ["", ""]:
        raise ValueError("The saved cold graph already contains a source checkpoint")
    load["widgets_values"][:] = [receipt["path"], receipt["sha"]]
    backend_index = (1 if setup["type"] == "MiniMaxH3SolEngineLTXRefinerSetupT8Advanced"
                     else 3 if setup["type"] == "MiniMaxH3SolEngineLTXIdentityRefinerSetupT8Advanced"
                     else None)
    if backend_index is None or setup["widgets_values"][4 if backend_index == 1 else 6] is not False:
        raise ValueError("The saved LTX setup widget layout changed")
    setup["widgets_values"][backend_index] = "dense_reference"
    nodes[relay.builder.PLAN]["widgets_values"][:2] = [full_canvas.GLOBAL, full_canvas.LOCAL]
    nodes[relay.builder.APPLY]["widgets_values"][0] = "apply_exp"
    eav = [node for node in graph["nodes"] if node["type"] == "MiniMaxH3StageEAVConfigEXPT8"]
    if len(eav) > 1 or (eav and eav[0]["widgets_values"][0] != "report_only"):
        raise ValueError("Combined cold canvas EAV must remain report-only")
    nodes[source.builder.ISOLATED_WRITER]["widgets_values"][0] = MEDIA_PREFIX
    if graph["links"] != original["links"] or len(graph["nodes"]) != len(original["nodes"]):
        raise ValueError("Cold widget preparation changed workflow structure")
    return graph


def preflight(args: argparse.Namespace, api: dict, control_report: dict) -> dict:
    assets = {}
    folders = {"unet_name": "diffusion_models", "vae_name": "vae",
               "clip_name": "text_encoders", "lora_name": "loras"}
    for key, node in api.items():
        for field, folder in folders.items():
            if field in node["inputs"]:
                assets[key] = args.comfy_root / "models" / folder / node["inputs"][field]
        if node["class_type"] == "MiniMaxH3SolEngineTAEHVLoaderT8Advanced":
            assets[key] = args.comfy_root / "models/taehv" / node["inputs"]["model_name"]
    gpu = shared.gpu_memory_mib()
    reference = control_report["media"]["cold"]["streams"]
    video = next(item for item in reference if item["codec_type"] == "video")
    checks = {"owned_port_free": args.port not in (8188, 8189, 8940)
              and not shared.port_is_listening("127.0.0.1", args.port),
              "gpu_ready": gpu.get("available") and gpu.get("free_mib", 0) >= args.min_free_vram_mib,
              "cold_assets_exist": bool(assets) and all(path.is_file() for path in assets.values()),
              "python_exists": args.python.is_file(),
              "media_tools": all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")),
              "baseline_geometry": (video["width"], video["height"], int(video["nb_frames"]))
              == (1024, 576, 113)}
    return {"checks": checks, "ready": all(checks.values()), "gpu": gpu,
            "user_8940_owner_before": full_canvas._port_owner(8940),
            "assets": {key: str(path) for key, path in assets.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("ordinary", "identity"), default="ordinary")
    parser.add_argument("--combined-eav", action="store_true")
    parser.add_argument("--port", type=int, default=8958)
    parser.add_argument("--comfy-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    args.comfy_root = args.comfy_root.resolve()
    args.host, args.use_pytorch_cross_attention = "127.0.0.1", True
    name, original, api = relay._saved(args.route, args.combined_eav, relay.SAVED,
                                       variant="resume_ltx")
    baseline_path, baseline, receipt = control(args.route, args.combined_eav)
    ready = preflight(args, api, baseline)
    print(json.dumps({"schema": "t8.s27.relay-cold-canvas.preflight.v1", **ready}), flush=True)
    if not args.confirm_run or not ready["ready"]:
        return 0 if ready["ready"] else 2

    prepared = prepared_frontend(original, receipt)
    run = ROOT / "artifacts/development/modular-ltx-relay-20260928/canvas" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.route
        + ("-eav-relay-cold" if args.combined_eav else "-relay-cold"))
    run.mkdir(parents=True, exist_ok=False)
    print("RUN_ROOT=" + str(run), flush=True)
    before = source_snapshot(ROOT)
    relay._write(run / "sources-before.json", before)
    relay._write(run / "prepared.frontend.json", prepared)
    args.input_directory = run / "input"
    args.input_directory.mkdir()
    source.copy_source(receipt, run)
    copied_manifest = run / "output/MiniMaxH3/ltx_rgb_sources" / receipt["path"]
    copied_state = copied_manifest.parent / "state.safetensors"
    paths = probe_resource_config(args.comfy_root, ROOT)
    paths["t8_runtime_models"]["taehv"] = str(args.comfy_root / "models/taehv")
    args.extra_model_paths_config = run / "paths.json"
    relay._write(args.extra_model_paths_config, paths)
    workflow_name = "QA_S27_Relay_Canvas_" + args.route + (
        "_EAV_Cold" if args.combined_eav else "_Cold")
    workflow = run / "user/default/workflows" / (workflow_name + ".json")
    workflow.parent.mkdir(parents=True)
    relay._write(workflow, prepared)
    user_port_before = full_canvas._port_owner(8940)
    if user_port_before != ready["user_8940_owner_before"]:
        raise RuntimeError("User Core 8940 changed between preflight and isolated startup")
    report = {"schema": "t8.s27.relay-native-cold-canvas-gpu.v1", "status": "started",
              "saved_graph": name, "route": args.route, "combined_eav": args.combined_eav,
              "run_root": str(run), "api_control": str(baseline_path), "source_receipt": receipt,
              "preflight": ready, "checks": {}, "boundary":
              "One historical frozen source and one cold native Queue; no source preparation, "
              "new H3 first pass, multi-source, human quality or full dual-pass qualification."}
    server = shared.IsolatedServer(args, run, "cold_canvas")
    try:
        with server:
            url = f"http://127.0.0.1:{args.port}"
            with urllib.request.urlopen(url + "/object_info", timeout=30) as response:
                info = json.load(response)
            expected = source.builder.split_api(prepared, info)
            relay._write(run / "prepared.api.json", expected)
            if (not _queue_empty(url) or _json_get(url + "/history")
                    or any(node["class_type"] in {"LoadVideo", "VAEEncode", "LTXVLatentUpsampler"}
                           for node in expected.values())):
                raise ValueError("Cold Core must start empty without original video preparation")
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
                        raise ValueError("Cold native canvas differs from the live-schema prepared API")
                    relay._write(run / "canvas-before-queue.api.json", output)
                    page.screenshot(path=str(run / "canvas-before-queue.png"), full_page=True)
                    button = page.get_by_test_id("queue-button")
                    if button.count() != 1 or not button.is_visible() or not button.is_enabled():
                        raise ValueError("Native cold Queue button is not uniquely enabled")
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
                        raise ValueError("Native cold Queue click did not produce exactly one prompt")
                    response = responses[0]
                    submitted = response.json()
                    post = json.loads(posted[0].post_data)
                    relay._write(run / "canvas-post.json", post)
                    if response.status != 200 or post.get("prompt") != output:
                        raise ValueError("Cold browser POST changed the visible API or was rejected")
                    prompt_id = submitted["prompt_id"]
                    report["canvas"] = {"queue_button_testid": "queue-button", "pid": server.process.pid,
                                        "prompt_id": prompt_id, "request_url": posted[0].url,
                                        "response_status": response.status,
                                        "workflow_name": workflow_name}
                    print("CANVAS_PROMPT_ID=" + prompt_id, flush=True)
                    history = full_canvas._history(url, prompt_id, args.timeout_seconds)
                    relay._write(run / "history.json", history)
                finally:
                    browser.close()
            report["checks"].update(
                native_canvas_completed=history.get("status", {}).get("status_str") == "success"
                and history.get("status", {}).get("completed") is True,
                history_prompt_exact=(isinstance(history.get("prompt"), list)
                                      and len(history["prompt"]) >= 3
                                      and history["prompt"][1] == prompt_id
                                      and history["prompt"][2] == post["prompt"]),
                owned_queue_empty=_queue_empty(url),
                no_original_video_preparation=not any(
                    node["class_type"] in {"LoadVideo", "VAEEncode", "LTXVLatentUpsampler"}
                    for node in post["prompt"].values()))
            if not all(report["checks"].values()):
                raise RuntimeError("Native cold canvas or history failed its execution checks")
        media = source.media(run, MEDIA_PREFIX, isolated=True)
        report["media"] = media
        video = next(item for item in media["streams"] if item["codec_type"] == "video")
        history_output = history.get("outputs", {}).get("20", {}).get("images") or []
        relative = Path(media["path"]).resolve().relative_to((run / "output").resolve()).as_posix()
        logs = (run / "logs/cold_canvas.stderr.log").read_text(encoding="utf8", errors="replace")
        report["checks"].update(
            strict_complete_av=media["fully_decoded"],
            expected_geometry_frames=(video["width"], video["height"], int(video["nb_frames"]))
            == (1024, 576, 113) and video["avg_frame_rate"] == "24/1",
            original_audio_bypass=media["decoded_sha"]["audio"] == ORIGINAL_AUDIO_SHA,
            independent_api_cold_media_parity=(media["decoded_sha"]
                                               == baseline["media"]["cold"]["decoded_sha"]),
            native_history_names_media=(len(history_output) == 1 and history_output[0].get("filename")
                                        == "video.mp4" and relative ==
                                        history_output[0].get("subfolder", "").replace("\\", "/")
                                        + "/video.mp4"),
            three_real_sampler_steps_logged=bool(re.search(r"100%.*3/3", logs)))
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["checks"].update(
            source_stable=source_snapshot(ROOT) == before,
            checkpoint_manifest_unchanged=(Path(receipt["manifest_file"]).is_file()
                                           and source.sha(Path(receipt["manifest_file"])) == receipt["sha"]),
            checkpoint_state_unchanged=(Path(receipt["state_file"]).is_file()
                                        and source.sha(Path(receipt["state_file"])) == receipt["state_sha"]),
            copied_checkpoint_unchanged=(copied_manifest.is_file() and copied_state.is_file()
                                         and source.sha(copied_manifest) == receipt["sha"]
                                         and source.sha(copied_state) == receipt["state_sha"]),
            no_original_video_mounted=no_original_video_mounted(args.input_directory),
            user_8940_untouched=user_port_before == full_canvas._port_owner(8940),
            owned_service_stopped=server.process is None or server.process.poll() is not None,
            owned_port_closed=not shared.port_is_listening(args.host, args.port))
        report["status"] = ("pass_native_cold_canvas_mechanical_not_quality"
                            if "error" not in report and all(report["checks"].values()) else "fail")
        relay._write(run / "report.json", report)
        print(json.dumps({"run_root": str(run), "status": report["status"],
                          "failed": [key for key, value in report["checks"].items() if not value]}), flush=True)
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
