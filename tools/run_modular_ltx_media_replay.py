"""Replay saved real LTX latents through explicit serial media I/O, with zero NFE.

Read-only preflight unless confirmed. Uses new owned Core processes, never the
user service. Old failing media/receipts remain intact. This is an output-only
experiment, not full sampling, browser execution, or a quality certification.
"""
import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_modular_ltx_rgb_source_gpu as prior  # noqa: E402

READER = "MiniMaxH3VideoComponentsSerialEXPT8"
WRITER = "MiniMaxH3SaveVideoSerialEXPT8"
ISOLATED_WRITER = "MiniMaxH3SaveVideoIsolatedEXPT8"


def preview(source):
    return {"class_type": "PreviewAny", "inputs": {"source": source}}


def read_graph():
    return {"1": {"class_type": "LoadVideo", "inputs": {"file": "source.mp4"}},
            "2": {"class_type": READER, "inputs": {"video": ["1", 0]}},
            "702": preview(["2", 5])}


def replay_graph(original, isolated=False):
    # Reuse the exact decoder, trim, fps, bit depth and frozen source receipt.
    graph = {key: deepcopy(original[key]) for key in ("17", "18", "19", "25", "29", "51", "52")}
    graph["55"] = {"class_type": "LoadLatent", "inputs": {"latent": "candidate.latent"}}
    graph["17"]["inputs"]["latent"] = ["55", 0]
    graph["18"]["inputs"]["audio"] = ["29", 1]
    for key, input_id, prefix in (("20", "19", "candidate"), ("53", "52", "source_reference")):
        graph[key] = {"class_type": WRITER, "inputs": {"video": [input_id, 0], "filename_prefix": prefix}}
        if isolated:
            graph[key]["class_type"] = ISOLATED_WRITER
            graph[key]["inputs"].update(timeout_seconds=300, max_staging_gib=16., min_free_disk_gib=2.)
    graph.update({"720": preview(["20", 1]), "721": preview(["20", 2]),
                  "753": preview(["53", 1]), "754": preview(["53", 2])})
    return graph


def phase_checks(phase, graph):
    executed = {row["node"] for row in phase["events"] if row["type"] == "executing" and row.get("node")}
    return {"terminal_success": phase["terminal"]["type"] == "execution_success",
            "every_explicit_node_executed": set(graph).issubset(executed),
            "only_explicit_nodes_executed": executed.issubset(graph),
            "no_diffusion_sampler": not any("Sampler" in node["class_type"] for node in graph.values())}


def inspect_media(path, output_root):
    path = Path(path).resolve()
    if not path.is_relative_to(output_root.resolve()) or path.name != "video.mp4" or not path.is_file():
        raise ValueError("Replay output must be a completed owned video")
    details = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json",
                                                 str(path)], timeout=30))
    hashes, errors = {}, {}
    for kind, options in (("video", ["-map", "0:v:0", "-pix_fmt", "rgb24", "-f", "rawvideo"]),
                          ("audio", ["-map", "0:a:0", "-f", "f32le"])):
        result = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), *options, "-"],
                                capture_output=True, timeout=120)
        hashes[kind] = hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 and result.stdout else None
        if hashes[kind] is None:
            errors[kind] = {"returncode": result.returncode, "stderr": result.stderr.decode(errors="replace")}
    return {"path": str(path), "sha": prior.sha(path), "streams": details["streams"],
            "decoded_sha": hashes, "strict_decode_errors": errors, "fully_decoded": not errors}


def evidence(root):
    report = prior.read(root / "report.json")
    prep = json.loads(prior.pdd._phase_text(prior.read(root / "full_save.phase.json"), "704"))
    if "prep" in report and report["prep"] != prep:
        raise ValueError("Prior preparation report differs from its actual graph output")
    report["prep"] = prep
    checks = report["checks"]
    required = ("full_save:terminal_success", "resume_ltx:terminal_success", "exact_full_cold_candidate_latent")
    if not all(checks.get(key) is True for key in required):
        raise ValueError("Prior trained sampler and exact full/cold latent evidence required")
    current = prior.latent_comparison(root / "full", root / "cold")
    if current != report["latent"] or not current["finite"] or not current["equal"]:
        raise ValueError("Saved native latent changed from prior real execution")
    from safetensors import safe_open
    for item in current["files"]:
        with safe_open(item["path"], framework="pt", device="cpu") as handle:
            if "latent_format_version_0" not in handle.keys():
                raise ValueError("Native LoadLatent must preserve the saved scaling")
    source = Path(report["source"]["path"])
    if prior.sha(source) != report["source"]["sha"]:
        raise ValueError("Prior original video changed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8955)
    parser.add_argument("--min-free-vram-mib", type=int, default=4000)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--timeout-seconds", type=float, default=300.)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--isolated-export", action="store_true")
    args = parser.parse_args()
    root, old = args.output.resolve(), args.prior_run.resolve()
    if root.exists() or not root.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("New private replay directory required")
    previous = evidence(old)
    original = prior.read(old / "resume_ltx.prompt.json")
    receipt = previous["phases"]["full_save"]["source_receipt"]
    gpu = prior.shared.gpu_memory_mib()
    ready = {"port_free": args.port != 8940 and not prior.shared.port_is_listening("127.0.0.1", args.port),
             "gpu_ready": gpu.get("available") and gpu.get("free_mib", 0) >= args.min_free_vram_mib,
             "decoder_exists": (args.comfy_root / "models/taehv" / original["25"]["inputs"]["model_name"]).is_file(),
             "tools_exist": all(shutil.which(x) for x in ("ffmpeg", "ffprobe")) and args.python.is_file()}
    print(json.dumps({"preflight": ready, "gpu": gpu, "diffusion_nfe": 0}), flush=True)
    if not all(ready.values()) or not args.confirm_run:
        return 0 if all(ready.values()) else 2
    root.mkdir(parents=True)
    before = prior.source_snapshot(ROOT)
    prior.write(root / "sources-before.json", before)
    args.host, args.use_pytorch_cross_attention = "127.0.0.1", True
    config = prior.probe_resource_config(args.comfy_root, ROOT)
    config["t8_runtime_models"]["taehv"] = str(args.comfy_root / "models/taehv")
    args.extra_model_paths_config = root / "paths.json"
    prior.write(args.extra_model_paths_config, config)
    report = {"schema": "t8.s27.serial-media-output-replay.v1", "status": "started",
              "created_utc": datetime.now(timezone.utc).isoformat(), "prior_run": str(old),
              "preflight": ready, "phases": {}, "media": {}, "checks": {},
              "diffusion_nfe": 0, "quality_accepted": False,
              "isolated_export": args.isolated_export,
              "boundary": "Output-only replay of saved trained latents. Not full sampler rerun, canvas or all-route acceptance."}
    servers, identities, media_pending = [], [], {}
    try:
        for index, label in enumerate(("full", "cold")):
            owned = root / label
            owned.mkdir()
            args.input_directory = owned / "input"
            args.input_directory.mkdir()
            prior.copy_source(receipt, owned)
            for source, destination, expected in (
                (previous["source"]["path"], "source.mp4", previous["source"]["sha"]),
                (previous["latent"]["files"][index]["path"], "candidate.latent", previous["latent"]["files"][index]["sha"])):
                shutil.copyfile(source, args.input_directory / destination)
                if prior.sha(args.input_directory / destination) != expected:
                    raise ValueError("Owned copy changed")
            server = prior.shared.IsolatedServer(args, owned, label)
            servers.append(server)
            with server:
                for stage, graph in (("read_before", read_graph()), ("replay", replay_graph(original, args.isolated_export)), ("read_after", read_graph())):
                    prior.write(owned / (stage + ".prompt.json"), graph)
                    phase = asyncio.run(prior.pdd._submit_prompt_capture(server=f"http://{args.host}:{args.port}",
                        prompt=graph, timeout_seconds=args.timeout_seconds))
                    prior.write(owned / (stage + ".phase.json"), phase)
                    checks = phase_checks(phase, graph)
                    key = label + ":" + stage
                    report["checks"].update({key + ":" + name: value for name, value in checks.items()})
                    report["phases"][key] = {"pid": server.process.pid, "elapsed": phase["elapsed_seconds"]}
                    if not all(checks.values()):
                        raise RuntimeError("Explicit replay graph failed: " + key)
                    if stage.startswith("read_"):
                        read_report = json.loads(prior.pdd._phase_text(phase, "702"))
                        report["phases"][key]["reader"] = read_report
                        identities.append(read_report["data_identity"])
                    else:
                        for media_label, path_node, report_node in ((label, "720", "721"), ("source_" + label, "753", "754")):
                            path = prior.pdd._phase_text(phase, path_node)
                            media_pending[media_label] = (path, owned / "output",
                                json.loads(prior.pdd._phase_text(phase, report_node)))
                    print(json.dumps({"phase": key, "checks": checks}), flush=True)
        # Finish both owned GPU processes before external full-media decoders;
        # all raw execution receipts remain available even if ffprobe crashes.
        for label, (path, output_root, writer) in media_pending.items():
            report["media"][label] = inspect_media(path, output_root)
            report["media"][label]["writer"] = writer
        report["checks"].update(prior.media_checks(report["media"], previous["prep"]))
        report["checks"]["exact_full_cold_pre_encode_data"] = (
            report["media"]["full"]["writer"]["input_identity"] == report["media"]["cold"]["writer"]["input_identity"])
        for field in ("quantized_rgb_sha256", "converted_yuv_sha256"):
            report["checks"]["exact_full_cold_" + field] = (
                report["media"]["full"]["writer"][field] == report["media"]["cold"]["writer"][field])
        report["checks"]["four_source_reads_equal"] = len(identities) == 4 and all(x == identities[0] for x in identities)
        report["checks"]["fresh_replay_processes"] = report["phases"]["full:replay"]["pid"] != report["phases"]["cold:replay"]["pid"]
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["checks"].update(sources_stable=before == prior.source_snapshot(ROOT),
            old_source_unchanged=prior.sha(Path(previous["source"]["path"])) == previous["source"]["sha"],
            old_latents_unchanged=all(prior.sha(Path(item["path"])) == item["sha"] for item in previous["latent"]["files"]),
            old_frozen_data_unchanged=prior.sha(Path(receipt["state_file"])) == receipt["state_sha"]
                and prior.sha(Path(receipt["manifest_file"])) == receipt["sha"],
            owned_servers_stopped=all(s.process is None or s.process.poll() is not None for s in servers),
            owned_port_stopped=not prior.shared.port_is_listening(args.host, args.port))
        report["status"] = "output_replay_mechanical_pass_not_quality" if "error" not in report and all(report["checks"].values()) else "fail"
        prior.write(root / "report.json", report)
        print(json.dumps({"output": str(root), "status": report["status"],
                          "failed": [k for k, v in report["checks"].items() if not v]}), flush=True)
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
