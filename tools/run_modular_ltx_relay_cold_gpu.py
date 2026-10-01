"""Owned full-save → fresh-process cold LTX Relay comparison.

Read-only preflight by default. With --confirm-run, the RGB source Save/Load
nodes execute. --include-media also executes the isolated candidate and source
reference exporters in both processes; default remains latent-only. Neither
mode qualifies canvas interaction or image quality.
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
from tools import run_modular_ltx_relay_gpu as probe  # noqa: E402
from tools import build_modular_ltx_rgb_source_workflows as source  # noqa: E402
from tools import run_modular_ltx_rgb_source_gpu as source_runner  # noqa: E402
from tools.run_modular_s26_pdd_resume_gpu import shared, pdd  # noqa: E402
from tools.run_candidate_cpu_regression import source_snapshot  # noqa: E402
from tools.vdn_probe_environment import probe_resource_config  # noqa: E402


def _cold_graph(api, receipt, *, global_prompt, local_prompts, include_media=False):
    graph = deepcopy(api)
    store = probe._one(graph, source.LOAD)
    setup = next(key for key, node in graph.items() if node["class_type"] in probe.builder.SETUPS)
    plan = probe._one(graph, probe.builder.PLAN)
    apply = probe._one(graph, probe.builder.APPLY)
    audit = probe._one(graph, probe.builder.RELAY_AUDIT)
    sampler = probe._one(graph, "SamplerCustomAdvanced")
    exporter = probe._one(graph, source.ISOLATED_WRITER)
    if not include_media:
        graph.pop(exporter)
    graph[store]["inputs"].update(artifact_path=receipt["path"], artifact_sha256=receipt["sha"])
    graph[setup]["inputs"]["attention_backend"] = "dense_reference"
    graph[plan]["inputs"].update(global_prompt=global_prompt, local_prompts=local_prompts)
    graph[apply]["inputs"]["mode"] = "apply_exp"
    next_id = max(map(int, graph)) + 1
    save = str(next_id)
    preview = str(next_id + 1)
    graph[save] = {"class_type": "SaveLatent", "inputs": {
        "samples": [audit, 0], "filename_prefix": "cold_relay_candidate"}}
    graph[preview] = {"class_type": "PreviewAny", "inputs": {"source": [audit, 1]}}
    pins = {"store": store, "plan": plan, "apply": apply, "audit": audit,
            "sampler": sampler, "save": save, "report": preview,
            "exporter": exporter if include_media else None}
    if any(node["class_type"] == probe.eav.EFFECT_AUDIT for node in graph.values()):
        eav_audit = probe._one(graph, probe.eav.EFFECT_AUDIT)
        eav_preview = str(next_id + 2)
        graph[eav_preview] = {"class_type": "PreviewAny", "inputs": {"source": [eav_audit, 1]}}
        pins["eav_report"] = eav_preview
    return graph, pins


def _attach_source_reference(graph, pins):
    """Mux the frozen original frames/audio through the same isolated writer."""
    if pins["exporter"] is None:
        raise ValueError("Source reference requires the isolated media exporter")
    trim = probe._one(graph, "MiniMaxH3OutputTrimT8")
    create = probe._one(graph, "CreateVideo")
    next_id = max(map(int, graph)) + 1
    prep, ref_trim, ref_create, ref_export = (str(next_id + offset) for offset in range(4))
    graph[prep] = {"class_type": "PreviewAny", "inputs": {"source": [pins["store"], 3]}}
    graph[ref_trim] = deepcopy(graph[trim])
    graph[ref_trim]["inputs"].update(frames=[pins["store"], 0], audio=[pins["store"], 1])
    graph[ref_create] = deepcopy(graph[create])
    graph[ref_create]["inputs"].update(images=[ref_trim, 0], audio=[ref_trim, 1])
    graph[ref_export] = deepcopy(graph[pins["exporter"]])
    graph[ref_export]["inputs"].update(video=[ref_create, 0], filename_prefix="source_reference")
    pins["prep_report"] = prep
    pins["reference_exporter"] = ref_export


def _latent(root, prefix):
    from safetensors.torch import load_file
    files = list((root / "output").glob(prefix + "_*.latent"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one owned {prefix} candidate latent")
    return load_file(str(files[0]))["latent_tensor"], {"path": str(files[0]),
                                                      "sha256": source_runner.sha(files[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("ordinary", "identity"), required=True)
    parser.add_argument("--combined-eav", action="store_true")
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--global-prompt", required=True)
    parser.add_argument("--local-prompts", required=True)
    parser.add_argument("--saved-graphs", type=Path, default=probe.SAVED)
    parser.add_argument("--comfy-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--port", type=int, default=8956)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--include-media", action="store_true",
                        help="Also export and strictly compare complete candidate/reference MP4+audio")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    args.comfy_root = args.comfy_root.resolve()
    args.source_video = args.source_video.resolve()
    args.saved_graphs = args.saved_graphs.resolve()
    full_name, full_frontend, full_api = probe._saved(args.route, args.combined_eav, args.saved_graphs)
    cold_name, cold_frontend, cold_api = probe._saved(args.route, args.combined_eav, args.saved_graphs,
                                                     variant="resume_ltx")
    args.host = "127.0.0.1"
    args.use_pytorch_cross_attention = True
    ready = source_runner.preflight(args, {"full_save": (full_frontend, full_api)})
    print(json.dumps({"schema": "t8.s27.ltx-relay-cold.preflight.v1", **ready}), flush=True)
    if not args.confirm_run or not ready["ready"]:
        return 0 if ready["ready"] else 2
    run = ROOT / "artifacts/development/modular-ltx-relay-20260928/trained" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.route +
        ("-eav-relay-cold" if args.combined_eav else "-cold"))
    run.mkdir(parents=True, exist_ok=False)
    print("RUN_ROOT=" + str(run), flush=True)
    before = source_snapshot(ROOT)
    probe._write(run / "sources-before.json", before)
    source_sha = source_runner.sha(args.source_video)
    config = probe_resource_config(args.comfy_root, ROOT)
    config["t8_runtime_models"]["taehv"] = str(args.comfy_root / "models/taehv")
    args.extra_model_paths_config = run / "paths.json"
    probe._write(args.extra_model_paths_config, config)
    report = {"schema": "t8.s27.ltx-relay-full-cold.v2", "status": "started",
              "saved_graphs": [full_name, cold_name], "combined_eav": args.combined_eav,
              "include_media": args.include_media,
              "preflight": ready,
              "source_sha256": source_sha, "phases": {}, "checks": {}, "boundary":
              ("Real new-process LTX candidate and source-reference complete-media checks; "
               "not canvas or human quality." if args.include_media else
               "Real new-process LTX candidate-latent parity only, no complete media/canvas/quality.")}
    servers = []
    receipt = None
    try:
        for phase_name, frontend, api in (("full_save", full_frontend, full_api),
                                          ("resume_ltx", cold_frontend, cold_api)):
            owned = run / ("full" if phase_name == "full_save" else "cold")
            owned.mkdir()
            args.input_directory = owned / "input"
            args.input_directory.mkdir()
            if phase_name == "full_save":
                shutil.copyfile(args.source_video, args.input_directory / "source.mp4")
                if source_runner.sha(args.input_directory / "source.mp4") != source_sha:
                    raise ValueError("Owned source copy differs")
            else:
                source_runner.copy_source(receipt, owned)
            server = shared.IsolatedServer(args, owned, phase_name)
            servers.append(server)
            with server:
                with urllib.request.urlopen(f"http://{args.host}:{args.port}/object_info", timeout=30) as response:
                    info = json.load(response)
                live = source.split_api(frontend, info)
                if phase_name == "full_save":
                    live[probe._one(live, "LoadVideo")]["inputs"]["file"] = "0.6.mp4"
                if live != api:
                    raise ValueError("Live saved graph/schema differs: " + phase_name)
                if phase_name == "full_save":
                    graph, pins = probe._probe_graph(api, width=args.width, height=args.height,
                        global_prompt=args.global_prompt, local_prompts=args.local_prompts,
                        mode="apply_exp", include_media=args.include_media)
                    graph[pins["store"]]["inputs"].update(confirm_save=True,
                                                            filename_prefix="relay_full_source")
                    graph[pins["save"]]["inputs"]["filename_prefix"] = "full_relay_candidate"
                    if args.include_media:
                        graph[pins["exporter"]]["inputs"]["filename_prefix"] = "full_relay"
                        _attach_source_reference(graph, pins)
                    graph["701"] = {"class_type": "PreviewAny", "inputs": {"source": [pins["store"], 8]}}
                    graph["702"] = {"class_type": "PreviewAny", "inputs": {"source": [pins["store"], 9]}}
                    preview = pins["reports"]["audit"]
                    eav_preview = pins["reports"].get("eav")
                    expected_no_source = False
                    prefix = "full_relay_candidate"
                else:
                    graph, pins = _cold_graph(api, receipt, global_prompt=args.global_prompt,
                                              local_prompts=args.local_prompts,
                                              include_media=args.include_media)
                    if args.include_media:
                        graph[pins["exporter"]]["inputs"]["filename_prefix"] = "cold_relay"
                        _attach_source_reference(graph, pins)
                    preview = pins["report"]
                    eav_preview = pins.get("eav_report")
                    expected_no_source = True
                    prefix = "cold_relay_candidate"
                probe._write(run / (phase_name + ".prompt.json"), graph)
                phase = asyncio.run(pdd._submit_prompt_capture(server=f"http://{args.host}:{args.port}",
                    prompt=graph, timeout_seconds=args.timeout_seconds))
                probe._write(run / (phase_name + ".phase.json"), phase)
                audit = json.loads(pdd._phase_text(phase, preview))
                eav_audit = json.loads(pdd._phase_text(phase, eav_preview)) if eav_preview else None
                prep = json.loads(pdd._phase_text(phase, pins["prep_report"])) if args.include_media else None
                executing = {str(event.get("node")) for event in phase["events"]
                             if event["type"] == "executing"}
                progress = sum(event["type"] == "progress" and str(event.get("node")) == pins["sampler"]
                               for event in phase["events"])
                latent, file_info = _latent(owned, prefix)
                import torch
                checks = {"terminal_success": phase["terminal"]["type"] == "execution_success",
                          "source_store_executed": pins["store"] in executing,
                          "three_sampler_steps": progress == 3,
                          "relay_apply_executed": pins["apply"] in executing,
                          "all_video_blocks_biased": all(v > 0 for v in audit["applied_calls"]),
                          "candidate_finite": bool(torch.isfinite(latent).all()),
                          "combined_eav_measured": (all(value > 0 for value in eav_audit["measured_calls"])
                              if args.combined_eav else True),
                          "no_original_video_preparation_on_cold": (not executing.intersection(
                              {"1", "2", "3", "4", "5", "6", "7"}) if expected_no_source else True)}
                if args.include_media:
                    checks.update(candidate_export_executed=pins["exporter"] in executing,
                                  source_reference_export_executed=pins["reference_exporter"] in executing,
                                  original_audio_bypass_reported=(
                                      prep["audio_policy"] == "bypass_stage2_and_preserve_original_h3_audio_object"))
                report["phases"][phase_name] = {"pid": server.process.pid,
                                                "elapsed_seconds": phase["elapsed_seconds"],
                                                "candidate": file_info, "audit": audit,
                                                "eav_audit": eav_audit, "prep": prep,
                                                "checks": checks}
                print(json.dumps({"phase": phase_name, "checks": checks}), flush=True)
                if not all(checks.values()):
                    raise RuntimeError("LTX Relay " + phase_name + " failed its execution checks")
                if phase_name == "full_save":
                    receipt = source_runner.source_receipt(owned, phase)
                    report["source_receipt"] = receipt
        from safetensors.torch import load_file
        import torch
        full = load_file(report["phases"]["full_save"]["candidate"]["path"])["latent_tensor"]
        cold = load_file(report["phases"]["resume_ltx"]["candidate"]["path"])["latent_tensor"]
        report["candidate_comparison"] = {"shape": list(full.shape),
            "equal": bool(torch.equal(full, cold)),
            "max_abs": float((full.float() - cold.float()).abs().max())}
        report["checks"].update(new_process=report["phases"]["full_save"]["pid"] !=
                               report["phases"]["resume_ltx"]["pid"],
                               exact_full_cold_candidate_latent=report["candidate_comparison"]["equal"])
        if args.include_media:
            report["media"] = {label: source_runner.media(folder, prefix, isolated=True)
                               for label, folder, prefix in (
                                   ("full", run / "full", "full_relay"),
                                   ("cold", run / "cold", "cold_relay"),
                                   ("source_full", run / "full", "source_reference"),
                                   ("source_cold", run / "cold", "source_reference"))}
            full_prep = report["phases"]["full_save"]["prep"]
            cold_prep = report["phases"]["resume_ltx"]["prep"]
            report["checks"].update(source_prep_identical=full_prep == cold_prep,
                                    source_reference_video_identical=(
                                        report["media"]["source_full"]["fully_decoded"] and
                                        report["media"]["source_cold"]["fully_decoded"] and
                                        report["media"]["source_full"]["decoded_sha"]["video"] ==
                                        report["media"]["source_cold"]["decoded_sha"]["video"]))
            report["checks"].update(source_runner.media_checks(report["media"], full_prep))
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["checks"].update(source_stable=source_snapshot(ROOT) == before,
                                source_video_unchanged=source_runner.sha(args.source_video) == source_sha,
                                owned_servers_stopped=all(s.process is None or s.process.poll() is not None for s in servers),
                                owned_port_stopped=not shared.port_is_listening(args.host, args.port))
        passing_status = ("full_cold_relay_media_mechanical_not_canvas_or_quality" if args.include_media
                          else "full_cold_relay_latent_parity_not_media_or_quality")
        report["status"] = (passing_status if
                            "error" not in report and all(report["checks"].values()) else "fail")
        probe._write(run / "report.json", report)
        print(json.dumps({"run_root": str(run), "status": report["status"],
                          "failed": [key for key, value in report["checks"].items() if not value]}), flush=True)
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
