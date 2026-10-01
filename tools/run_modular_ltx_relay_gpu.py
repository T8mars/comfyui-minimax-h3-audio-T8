"""Owned real-weight LTX Relay sampling probe; no media export or user service.

Read-only preflight by default. Explicit --confirm-run starts one isolated
ComfyUI, executes the saved opt-in graph with output video saving disconnected,
and retains a latent plus Plan/Encode/Apply/Audit reports. This is API sampling,
not canvas, complete media, cold-resume or subjective-quality qualification.
"""
import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from tools import build_modular_ltx_relay_workflows as builder  # noqa: E402
from tools import build_modular_ltx_eav_workflows as eav  # noqa: E402
from tools import build_modular_ltx_rgb_source_workflows as source  # noqa: E402
from tools import run_modular_ltx_rgb_source_gpu as source_runner  # noqa: E402
from tools.run_modular_s26_pdd_resume_gpu import shared, pdd  # noqa: E402
from tools.run_candidate_cpu_regression import source_snapshot  # noqa: E402
from tools.vdn_probe_environment import probe_resource_config  # noqa: E402

SAVED = ROOT / "artifacts/development/modular-ltx-relay-20260928/candidate-v1"


def _write(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _one(api, kind):
    matches = [key for key, node in api.items() if node["class_type"] == kind]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {kind}, found {matches}")
    return matches[0]


def _saved(route, combined, directory, *, variant="full_save"):
    if route not in ("ordinary", "identity"):
        raise ValueError("Unknown LTX route")
    if variant not in ("full_save", "resume_ltx"):
        raise ValueError("Unknown saved LTX graph variant")
    suffix = "_external_eav_relay" if combined else "_external_relay"
    names = [name for name in builder.generated() if name.endswith(variant + suffix)
             and ("Identity_Preserve" in name) == (route == "identity")]
    if len(names) != 1:
        raise ValueError(f"Expected one saved {route}/{suffix} graph")
    name = names[0]
    frontend = json.loads((directory / (name + ".json")).read_text(encoding="utf8"))
    if frontend != builder.generated()[name]:
        raise ValueError("Saved frontend differs from current Relay builder")
    api = json.loads((directory / (name + ".api.json")).read_text(encoding="utf8"))
    return name, frontend, api


def _probe_graph(api, *, width, height, global_prompt, local_prompts, mode,
                 include_media=False):
    if min(width, height) < 32 or width % 32 or height % 32:
        raise ValueError("Probe geometry must be positive and divisible by 32")
    if len([line for line in local_prompts.splitlines() if line.strip()]) < 2:
        raise ValueError("A real two-event Relay probe needs at least two local prompts")
    graph = deepcopy(api)
    load = _one(graph, "LoadVideo")
    prep = _one(graph, "MiniMaxH3SolEngineDraftToLTXT8Advanced")
    store = _one(graph, source.SAVE)
    setup = next(key for key, node in graph.items() if node["class_type"] in builder.SETUPS)
    plan = _one(graph, builder.PLAN)
    encode = _one(graph, builder.ENCODE)
    apply = _one(graph, builder.APPLY)
    audit = _one(graph, builder.RELAY_AUDIT)
    sampler = _one(graph, "SamplerCustomAdvanced")
    exporter = _one(graph, source.ISOLATED_WRITER)
    if not include_media:
        graph.pop(exporter)
    graph[load]["inputs"]["file"] = "source.mp4"
    graph[prep]["inputs"].update(target_width=width, target_height=height)
    graph[store]["inputs"]["confirm_save"] = False
    graph[setup]["inputs"]["attention_backend"] = "dense_reference"
    graph[plan]["inputs"].update(global_prompt=global_prompt, local_prompts=local_prompts)
    graph[apply]["inputs"]["mode"] = mode
    next_id = max(map(int, graph)) + 1
    save_id = str(next_id)
    graph[save_id] = {"class_type": "SaveLatent", "inputs": {
        "samples": [audit, 0], "filename_prefix": "ltx_relay_candidate"}}
    reports = {}
    for offset, (label, node, slot) in enumerate((("plan", plan, 4), ("encode", encode, 2),
                                                   ("apply", apply, 3), ("audit", audit, 1)), 1):
        preview_id = str(next_id + offset)
        graph[preview_id] = {"class_type": "PreviewAny", "inputs": {"source": [node, slot]}}
        reports[label] = preview_id
    if any(node["class_type"] == eav.EFFECT_AUDIT for node in graph.values()):
        eav_audit = _one(graph, eav.EFFECT_AUDIT)
        preview_id = str(next_id + len(reports) + 1)
        graph[preview_id] = {"class_type": "PreviewAny", "inputs": {"source": [eav_audit, 1]}}
        reports["eav"] = preview_id
    return graph, {"load": load, "store": store, "setup": setup, "plan": plan,
                   "encode": encode, "apply": apply, "audit": audit,
                   "sampler": sampler, "save": save_id, "reports": reports,
                   "exporter": exporter if include_media else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("ordinary", "identity"), required=True)
    parser.add_argument("--combined-eav", action="store_true")
    parser.add_argument("--mode", choices=("report_only", "apply_exp"), default="apply_exp")
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--global-prompt", required=True)
    parser.add_argument("--local-prompts", required=True)
    parser.add_argument("--saved-graphs", type=Path, default=SAVED)
    parser.add_argument("--comfy-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--port", type=int, default=8956)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    args.comfy_root = args.comfy_root.resolve()
    args.source_video = args.source_video.resolve()
    args.saved_graphs = args.saved_graphs.resolve()
    name, frontend, api = _saved(args.route, args.combined_eav, args.saved_graphs)
    args.host = "127.0.0.1"
    args.use_pytorch_cross_attention = True
    ready = source_runner.preflight(args, {"full_save": (frontend, api)})
    ready["checks"]["saved_graph_bound"] = bool(name)
    ready["ready"] = all(ready["checks"].values())
    print(json.dumps({"schema": "t8.s27.ltx-relay-real-probe.preflight.v1", **ready}), flush=True)
    if not args.confirm_run or not ready["ready"]:
        return 0 if ready["ready"] else 2
    run = ROOT / "artifacts/development/modular-ltx-relay-20260928/trained" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.route +
        ("-eav-relay" if args.combined_eav else "-relay") + "-" + args.mode)
    run.mkdir(parents=True, exist_ok=False)
    print("RUN_ROOT=" + str(run), flush=True)
    before = source_snapshot(ROOT)
    _write(run / "sources-before.json", before)
    (run / "input").mkdir()
    args.input_directory = run / "input"
    shutil.copyfile(args.source_video, run / "input/source.mp4")
    source_sha = source_runner.sha(args.source_video)
    if source_runner.sha(run / "input/source.mp4") != source_sha:
        raise ValueError("Owned source copy differs")
    config = probe_resource_config(args.comfy_root, ROOT)
    config["t8_runtime_models"]["taehv"] = str(args.comfy_root / "models/taehv")
    args.extra_model_paths_config = run / "paths.json"
    _write(args.extra_model_paths_config, config)
    report = {"schema": "t8.s27.ltx-relay-real-probe.v1", "status": "started", "saved_graph": name,
              "mode": args.mode,
              "source_sha256": source_sha, "preflight": ready, "checks": {}, "boundary":
              "Real trained-weight API sampling only; not canvas, full/cold parity, media or human quality."}
    server = shared.IsolatedServer(args, run, "relay")
    try:
        with server:
            with urllib.request.urlopen(f"http://{args.host}:{args.port}/object_info", timeout=30) as response:
                info = json.load(response)
            serialized = source.split_api(frontend, info)
            _one(serialized, "LoadVideo")
            serialized[_one(serialized, "LoadVideo")]["inputs"]["file"] = "0.6.mp4"
            if serialized != api:
                raise ValueError("Live saved Relay graph/schema differs")
            graph, pins = _probe_graph(api, width=args.width, height=args.height,
                                       global_prompt=args.global_prompt, local_prompts=args.local_prompts,
                                       mode=args.mode)
            _write(run / "prompt.json", graph)
            phase = asyncio.run(pdd._submit_prompt_capture(server=f"http://{args.host}:{args.port}",
                prompt=graph, timeout_seconds=args.timeout_seconds))
            _write(run / "phase.json", phase)
            report["phase"] = {"pid": server.process.pid, "elapsed_seconds": phase["elapsed_seconds"],
                               "terminal": phase["terminal"]}
            reports = {label: json.loads(pdd._phase_text(phase, node))
                       for label, node in pins["reports"].items()}
            _write(run / "relay-reports.json", reports)
            report["relay"] = reports["audit"]
            executing = {str(event.get("node")) for event in phase["events"]
                         if event["type"] == "executing"}
            progress = sum(event["type"] == "progress" and str(event.get("node")) == pins["sampler"]
                           for event in phase["events"])
            report["checks"].update(
                terminal_success=phase["terminal"]["type"] == "execution_success",
                plan_encode_apply_audit_executed={pins[k] for k in ("plan", "encode", "apply", "audit")}.issubset(executing),
                native_three_step_sampler=progress == 3,
                real_video_cross_attention_observed=all(v > 0 for v in reports["audit"]["observed_calls"]),
                relay_mode_effect=(all(v > 0 for v in reports["audit"]["applied_calls"])
                    if args.mode == "apply_exp" else all(v == 0 for v in reports["audit"]["applied_calls"])),
                original_backend_delegated=(reports["audit"]["selected_backend_delegate_calls"] > 0
                    if args.mode == "apply_exp" else True),
                candidate_not_misrepresented_as_qualified=reports["audit"]["candidate_provenance_verified"] is False,
            )
            if args.combined_eav:
                report["checks"]["independent_eav_measured"] = all(
                    value > 0 for value in reports["eav"]["measured_calls"])
            saved = list((run / "output").glob("ltx_relay_candidate_*.latent"))
            report["checks"]["one_owned_candidate_latent"] = len(saved) == 1
            if saved:
                report["candidate_latent"] = {"path": str(saved[0]), "sha256": source_runner.sha(saved[0])}
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["checks"].update(source_stable=source_snapshot(ROOT) == before,
                                input_unchanged=source_runner.sha(args.source_video) == source_sha,
                                owned_server_stopped=server.process is None or server.process.poll() is not None,
                                owned_port_stopped=not shared.port_is_listening(args.host, args.port))
        report["status"] = ("real_weight_relay_observed_not_media_or_quality" if
                            "error" not in report and all(report["checks"].values()) else "fail")
        _write(run / "report.json", report)
        print(json.dumps({"run_root": str(run), "status": report["status"],
                          "failed": [key for key, value in report["checks"].items() if not value]}), flush=True)
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
