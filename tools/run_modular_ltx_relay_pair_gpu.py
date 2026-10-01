"""One owned Core process: report-only → apply-exp → report-only LTX effect.

This tests whether the numeric effect exceeds a same-mode repeat under the
same loaded runtime. It samples a latent only; it never writes finished media.
Read-only preflight unless --confirm-run is provided.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
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


def _delta(a, b):
    import torch
    diff = a.float() - b.float()
    return {"equal": bool(torch.equal(a, b)), "max_abs": float(diff.abs().max()),
            "rmse": float(diff.square().mean().sqrt())}


def _effect_graph(api, *, width, height, global_prompt, local_prompts, effect, mode, tau):
    graph, pins = probe._probe_graph(api, width=width, height=height,
        global_prompt=global_prompt, local_prompts=local_prompts,
        mode=mode if effect == "relay" else "apply_exp")
    if effect == "eav":
        config_id = probe._one(graph, "MiniMaxH3StageEAVConfigEXPT8")
        graph[config_id]["inputs"].update(mode=mode, tau=tau)
    return graph, pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("ordinary", "identity"), required=True)
    parser.add_argument("--effect", choices=("relay", "eav"), default="relay")
    parser.add_argument("--tau", type=float, default=4.)
    parser.add_argument("--expected-source-sha256", default="")
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
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    args.comfy_root = args.comfy_root.resolve()
    args.source_video = args.source_video.resolve()
    args.saved_graphs = args.saved_graphs.resolve()
    name, frontend, api = probe._saved(args.route, args.effect == "eav", args.saved_graphs)
    args.host = "127.0.0.1"
    args.use_pytorch_cross_attention = True
    ready = source_runner.preflight(args, {"full_save": (frontend, api)})
    if args.effect == "eav":
        ready["checks"]["fixed_source_sha256"] = (
            args.source_video.is_file() and len(args.expected_source_sha256) == 64
            and source_runner.sha(args.source_video) == args.expected_source_sha256)
        ready["checks"]["finite_bounded_tau"] = math.isfinite(args.tau) and -32. <= args.tau <= 32.
        ready["ready"] = all(ready["checks"].values())
    print(json.dumps({"schema": "t8.s27.ltx-relay-aba.preflight.v1", **ready}), flush=True)
    if not args.confirm_run or not ready["ready"]:
        return 0 if ready["ready"] else 2

    run = ROOT / "artifacts/development/modular-ltx-relay-20260928/trained" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.route
        + ("-eav-aba" if args.effect == "eav" else "-aba"))
    run.mkdir(parents=True, exist_ok=False)
    print("RUN_ROOT=" + str(run), flush=True)
    before = source_snapshot(ROOT)
    probe._write(run / "sources-before.json", before)
    args.input_directory = run / "input"
    args.input_directory.mkdir()
    shutil.copyfile(args.source_video, args.input_directory / "source.mp4")
    source_sha = source_runner.sha(args.source_video)
    if source_runner.sha(args.input_directory / "source.mp4") != source_sha:
        raise ValueError("Owned source copy differs")
    config = probe_resource_config(args.comfy_root, ROOT)
    config["t8_runtime_models"]["taehv"] = str(args.comfy_root / "models/taehv")
    args.extra_model_paths_config = run / "paths.json"
    probe._write(args.extra_model_paths_config, config)
    report = {"schema": "t8.s27.ltx-" + args.effect + "-aba.v1", "status": "started",
              "saved_graph": name, "preflight": ready, "source_sha256": source_sha,
              "effect": args.effect, "eav_tau": args.tau if args.effect == "eav" else None,
              "modes": ["report_only", "apply_exp", "report_only"],
              "phases": [], "checks": {}, "boundary":
              "Same owned Core loaded runtime, small latent-only API A/B/A; no media/canvas/quality claim."}
    server = shared.IsolatedServer(args, run, "relay-aba")
    try:
        with server:
            with urllib.request.urlopen(f"http://{args.host}:{args.port}/object_info", timeout=30) as response:
                info = json.load(response)
            live = source.split_api(frontend, info)
            live[probe._one(live, "LoadVideo")]["inputs"]["file"] = "0.6.mp4"
            if live != api:
                raise ValueError("Live saved Relay graph/schema differs")
            graphs, latents = [], []
            for index, mode in enumerate(report["modes"]):
                graph, pins = _effect_graph(api, width=args.width, height=args.height,
                    global_prompt=args.global_prompt, local_prompts=args.local_prompts,
                    effect=args.effect, mode=mode, tau=args.tau)
                prefix = f"{args.effect}_aba_{index}_{mode}"
                graph[pins["save"]]["inputs"]["filename_prefix"] = prefix
                graphs.append(graph)
                probe._write(run / f"prompt-{index}.json", graph)
                phase = asyncio.run(pdd._submit_prompt_capture(server=f"http://{args.host}:{args.port}",
                    prompt=graph, timeout_seconds=args.timeout_seconds))
                probe._write(run / f"phase-{index}.json", phase)
                audit = json.loads(pdd._phase_text(phase, pins["reports"]["audit"]))
                eav_audit = (json.loads(pdd._phase_text(phase, pins["reports"]["eav"]))
                             if args.effect == "eav" else None)
                executing = {str(event.get("node")) for event in phase["events"]
                             if event["type"] == "executing"}
                progress = sum(event["type"] == "progress" and str(event.get("node")) == pins["sampler"]
                               for event in phase["events"])
                files = list((run / "output").glob(prefix + "_*.latent"))
                if len(files) != 1:
                    raise RuntimeError(f"Expected one owned latent for {mode} phase {index}")
                from safetensors.torch import load_file
                import torch
                latent = load_file(str(files[0]))["latent_tensor"]
                latents.append(latent)
                checks = {"terminal_success": phase["terminal"]["type"] == "execution_success",
                          "sampler_reexecuted_three_steps": progress == 3,
                          "relay_audit_executed": pins["audit"] in executing,
                          "all_video_blocks_observed": all(v > 0 for v in audit["observed_calls"]),
                          "relay_mode_effect": (all(v > 0 for v in audit["applied_calls"])
                              if args.effect == "eav" or mode == "apply_exp" else
                              all(v == 0 for v in audit["applied_calls"])),
                          "candidate_finite": bool(torch.isfinite(latent).all())}
                if args.effect == "eav":
                    checks.update(
                        relay_always_applied=all(v > 0 for v in audit["applied_calls"]),
                        eav_all_video_blocks_measured=all(v > 0 for v in eav_audit["measured_calls"]),
                        eav_mode_effect=(all(v > 0 for v in eav_audit["applied_calls"])
                            if mode == "apply_exp" else all(v == 0 for v in eav_audit["applied_calls"])),
                        eav_nonidentity_gain=(eav_audit["gain_max"] > 1.
                                              if mode == "apply_exp" else True))
                report["phases"].append({"mode": mode, "pid": server.process.pid,
                                         "elapsed_seconds": phase["elapsed_seconds"],
                                         "latent_sha256": source_runner.sha(files[0]),
                                         "latent_shape": list(latent.shape),
                                         "audit": audit, "eav_audit": eav_audit, "checks": checks})
                print(json.dumps({"phase": index, "mode": mode, "checks": checks}), flush=True)
                if not all(checks.values()):
                    raise RuntimeError(f"A/B/A phase {index} failed its execution checks")
            normalized = []
            for graph, mode in zip(graphs, report["modes"]):
                graph = json.loads(json.dumps(graph))
                if args.effect == "eav":
                    graph[probe._one(graph, "MiniMaxH3StageEAVConfigEXPT8")]["inputs"]["mode"] = "normalized"
                else:
                    graph[probe._one(graph, probe.builder.APPLY)]["inputs"]["mode"] = "normalized"
                graph[probe._one(graph, "SaveLatent")]["inputs"]["filename_prefix"] = "normalized"
                normalized.append(graph)
            report["checks"]["only_mode_and_output_prefix_differ"] = normalized[0] == normalized[1] == normalized[2]
            report["repeat_report_only"] = _delta(latents[0], latents[2])
            report["apply_vs_report_only"] = _delta(latents[0], latents[1])
            report["checks"]["same_mode_exact_repeat"] = report["repeat_report_only"]["equal"]
            report["checks"]["apply_changes_latent"] = not report["apply_vs_report_only"]["equal"]
            report["checks"]["same_owned_pid"] = len({phase["pid"] for phase in report["phases"]}) == 1
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["checks"].update(source_stable=source_snapshot(ROOT) == before,
                                source_video_unchanged=source_runner.sha(args.source_video) == source_sha,
                                server_stopped=server.process is None or server.process.poll() is not None,
                                port_stopped=not shared.port_is_listening(args.host, args.port))
        report["status"] = ("paired_real_weight_numeric_effect_not_media_or_quality" if
                            "error" not in report and all(report["checks"].values()) else "fail")
        probe._write(run / "report.json", report)
        print(json.dumps({"run_root": str(run), "status": report["status"],
                          "failed": [key for key, value in report["checks"].items() if not value]}), flush=True)
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
