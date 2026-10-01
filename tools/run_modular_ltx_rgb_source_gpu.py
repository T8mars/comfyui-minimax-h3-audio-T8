"""Owned real-weight S27 full/freeze/cold comparison; read-only by default.

No user service, public graph or production sampler is modified. A successful
receipt proves only the selected graphs/weights/media, not quality acceptance,
LTX effects, browser execution or a portable completed-sampling checkpoint.
"""
import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from tools import build_modular_ltx_rgb_source_workflows as builder  # noqa: E402
from tools.run_modular_s26_pdd_resume_gpu import shared, pdd  # noqa: E402
from tools.run_candidate_cpu_regression import source_snapshot  # noqa: E402
from tools.vdn_probe_environment import probe_resource_config  # noqa: E402

SAVED = ROOT / "artifacts/development/modular-ltx-rgb-source-20260928/candidate-v2-final"
VARIANTS = ("full_save", "freeze_source", "resume_ltx")
REPORTS = {"path": "701", "sha": "702", "storage": "703", "prep": "704", "audit": "705"}


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def write(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def saved_graphs(route, *, isolated_media=False, saved_root=None):
    if route not in ("ordinary", "identity"):
        raise ValueError("Unknown RGB LTX route")
    result = {}
    if isolated_media and saved_root is None:
        raise ValueError("Isolated media requires an explicitly saved candidate directory")
    directory = Path(saved_root) if saved_root is not None else SAVED
    for name, frontend in builder.generated(isolated_media=isolated_media).items():
        if ("Identity_Preserve" in name) != (route == "identity"):
            continue
        variant = next(v for v in VARIANTS if name.endswith(v))
        path = directory / (name + ".json")
        if read(path) != frontend:
            raise ValueError("Saved frontend differs from current builder")
        result[variant] = (frontend, read(directory / (name + ".api.json")))
    if set(result) != set(VARIANTS):
        raise ValueError("All three saved graphs are required")
    return result


def execution_graph(original, variant, *, prompt, width, height, receipt=None):
    graph = deepcopy(original)
    if variant not in VARIANTS or width % 32 or height % 32 or min(width, height) < 32:
        raise ValueError("Explicit supported variant and 32-aligned geometry required")
    if variant == "resume_ltx":
        if not receipt:
            raise ValueError("Cold graph requires an explicit source receipt")
        graph["29"]["inputs"].update(artifact_path=receipt["path"], artifact_sha256=receipt["sha"])
    else:
        graph["1"]["inputs"]["file"] = "source.mp4"
        graph["3"]["inputs"].update(target_width=width, target_height=height)
        graph["29"]["inputs"].update(confirm_save=True, filename_prefix=variant)
    outputs = {"storage": ["29", 8 if variant == "resume_ltx" else 10], "prep": ["29", 3]}
    if variant != "resume_ltx":
        outputs.update(path=["29", 8], sha=["29", 9])
    if variant != "freeze_source":
        # Explicit test setting, identical in full and cold. No default graph changes.
        graph["10"]["inputs"]["attention_backend"] = "dense_reference"
        graph["12"]["inputs"]["text"] = prompt
        graph["20"]["inputs"]["filename_prefix"] = variant
        graph["50"] = {"class_type": "SaveLatent", "inputs": {
            "samples": ["28", 0], "filename_prefix": "candidate"}}
        outputs["audit"] = ["28", 2]
        # Independent original-picture mux through the same trim, proving bypass audio.
        graph["51"] = deepcopy(graph["18"])
        graph["51"]["inputs"].update(frames=["29", 0], audio=["29", 1])
        graph["52"] = deepcopy(graph["19"])
        graph["52"]["inputs"].update(images=["51", 0], audio=["51", 1])
        graph["53"] = deepcopy(graph["20"])
        graph["53"]["inputs"].update(video=["52", 0], filename_prefix="source_reference")
    for name, source in outputs.items():
        graph[REPORTS[name]] = {"class_type": "PreviewAny", "inputs": {"source": source}}
    return graph


def phase_checks(phase, variant):
    executing = {row["node"] for row in phase["events"] if row["type"] == "executing"}
    progress = sum(row["type"] == "progress" and row["node"] == "16" for row in phase["events"])
    checks = {"terminal_success": phase["terminal"]["type"] == "execution_success",
              "source_io_executed": "29" in executing}
    if variant == "freeze_source":
        checks.update(no_sampler_or_model_executed=not executing.intersection({"8", "9", "10", "11", "16"}),
                      encode_and_upscale_executed={"1", "3", "5", "7"}.issubset(executing))
    else:
        checks.update(sampler_audit_decode_save_executed={"16", "28", "17", "20", "50"}.issubset(executing),
                      three_native_sampler_progress_callbacks=progress == 3)
        if variant == "resume_ltx":
            checks["no_source_decode_encode_upscale"] = not executing.intersection({"1", "2", "3", "4", "5", "6", "7"})
    return checks


def source_receipt(root, phase):
    relative = pdd._phase_text(phase, REPORTS["path"])
    expected = pdd._phase_text(phase, REPORTS["sha"])
    store = root / "output/MiniMaxH3/ltx_rgb_sources"
    manifest = (store / relative).resolve()
    if not manifest.is_relative_to(store.resolve()) or manifest.name != "manifest.json" or sha(manifest) != expected:
        raise ValueError("Source receipt manifest path/SHA differs")
    data = read(manifest)
    state = manifest.parent / "state.safetensors"
    if data.get("data_only") is not True or data["state_sha256"] != sha(state) or data["state_bytes"] != state.stat().st_size:
        raise ValueError("Source state differs from its data-only manifest")
    return {"path": relative, "sha": expected, "state_sha": data["state_sha256"],
            "manifest_file": str(manifest), "state_file": str(state)}


def copy_source(receipt, root):
    store = root / "output/MiniMaxH3/ltx_rgb_sources"
    destination = (store / receipt["path"]).resolve()
    if not destination.is_relative_to(store.resolve()) or destination.exists():
        raise ValueError("Cold destination must be new and inside its owned store")
    destination.parent.mkdir(parents=True, exist_ok=False)
    for source, target, expected in ((receipt["state_file"], destination.parent / "state.safetensors", receipt["state_sha"]),
                                     (receipt["manifest_file"], destination, receipt["sha"])):
        if sha(Path(source)) != expected:
            raise ValueError("Frozen source changed before copying")
        shutil.copyfile(source, target)
        if sha(target) != expected:
            raise ValueError("Copied source changed")


def media(root, prefix, *, isolated=False):
    files = list((root / "output").glob(prefix + ("-*/video.mp4" if isolated else "_*.mp4")))
    if len(files) != 1:
        raise ValueError("Expected exactly one owned " + prefix + " MP4")
    path = files[0]
    details = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json",
                                                 str(path)], timeout=30))
    streams = details["streams"]
    if len([s for s in streams if s["codec_type"] == "video"]) != 1 or len([s for s in streams if s["codec_type"] == "audio"]) != 1:
        raise ValueError("Expected complete video and original audio")
    hashes, errors = {}, {}
    for kind, options in (("video", ["-map", "0:v:0", "-pix_fmt", "rgb24", "-f", "rawvideo"]),
                          ("audio", ["-map", "0:a:0", "-f", "f32le"])):
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), *options, "-"],
                                 capture_output=True, timeout=120)
        if decoded.returncode != 0 or not decoded.stdout:
            errors[kind] = {"returncode": decoded.returncode, "stderr": decoded.stderr.decode(errors="replace")}
            hashes[kind] = None
        else:
            hashes[kind] = hashlib.sha256(decoded.stdout).hexdigest()
    return {"path": str(path), "sha": sha(path), "decoded_sha": hashes, "streams": streams,
            "strict_decode_errors": errors, "fully_decoded": not errors}


def media_checks(files, prep):
    checks = {label + "_strict_complete_av": item["fully_decoded"] for label, item in files.items()}
    target, source = prep["target"], prep["source"]
    for label, item in files.items():
        stream = next(s for s in item["streams"] if s["codec_type"] == "video")
        size = source if label.startswith("source_") else target
        num, den = stream["avg_frame_rate"].split("/")
        checks[label + "_geometry_duration"] = (
            (stream["width"], stream["height"]) == (size["width"], size["height"])
            and int(stream.get("nb_frames", 0)) == target["frames"]
            and math.isclose(float(num) / float(den), prep["fps"], abs_tol=1e-6)
            and math.isclose(float(stream.get("duration", 0)), prep["output_duration_seconds"], abs_tol=1e-5))
    hashes = {k: v["decoded_sha"] for k, v in files.items()}
    checks["exact_full_cold_media"] = (files["full"]["fully_decoded"] and files["cold"]["fully_decoded"]
                                      and hashes["full"] == hashes["cold"])
    checks["original_audio_bypassed"] = (all(h["audio"] for h in hashes.values())
                                          and len({h["audio"] for h in hashes.values()}) == 1)
    return checks


def latent_comparison(full, cold):
    from safetensors import safe_open
    import torch
    tensors = []
    files = []
    for root in (full, cold):
        found = list((root / "output").glob("candidate_*.latent"))
        if len(found) != 1:
            raise ValueError("Expected one native saved candidate LATENT")
        with safe_open(str(found[0]), framework="pt", device="cpu") as handle:
            tensors.append(handle.get_tensor("latent_tensor"))
        files.append({"path": str(found[0]), "sha": sha(found[0])})
    a, b = tensors
    return {"files": files, "shape": list(a.shape), "dtype": str(a.dtype),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            "equal": torch.equal(a, b), "max_abs": float((a.float() - b.float()).abs().max())}


def preflight(args, saved):
    full = saved["full_save"][1]
    assets = {}
    fields = {"unet_name": "diffusion_models", "vae_name": "vae", "clip_name": "text_encoders", "lora_name": "loras"}
    for key, node in full.items():
        for field, folder in fields.items():
            if field in node["inputs"]:
                assets[key] = args.comfy_root / "models" / folder / node["inputs"][field]
    assets["6"] = args.comfy_root / "models/latent_upscale_models" / full["6"]["inputs"]["model_name"]
    assets["25"] = args.comfy_root / "models/taehv" / full["25"]["inputs"]["model_name"]
    gpu = shared.gpu_memory_mib()
    checks = {"port_free": args.port != 8940 and not shared.port_is_listening("127.0.0.1", args.port),
              "gpu_ready": gpu.get("available") and gpu.get("free_mib", 0) >= args.min_free_vram_mib,
              "assets_exist": all(p.is_file() for p in assets.values()),
              "source_exists": args.source_video.is_file(), "python_exists": args.python.is_file(),
              "media_tools": all(shutil.which(x) for x in ("ffmpeg", "ffprobe")),
              "geometry_valid": min(args.width, args.height) >= 32 and args.width % 32 == args.height % 32 == 0}
    return {"checks": checks, "ready": all(checks.values()), "gpu": gpu,
            "assets": {k: str(v) for k, v in assets.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("ordinary", "identity"), required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--comfy-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--port", type=int, default=8955)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--isolated-media", action="store_true")
    parser.add_argument("--saved-graphs", type=Path)
    args = parser.parse_args()
    args.comfy_root, args.source_video = args.comfy_root.resolve(), args.source_video.resolve()
    saved = saved_graphs(args.route, isolated_media=args.isolated_media, saved_root=args.saved_graphs)
    ready = preflight(args, saved)
    print(json.dumps(ready), flush=True)
    if not args.confirm_run or not ready["ready"]:
        return 0 if ready["ready"] else 2
    run = ROOT / "artifacts/development/modular-ltx-rgb-source-20260928/trained" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.route)
    run.mkdir(parents=True, exist_ok=False)
    print("RUN_ROOT=" + str(run), flush=True)
    before = source_snapshot(ROOT)
    write(run / "sources-before.json", before)
    args.host = "127.0.0.1"
    args.use_pytorch_cross_attention = True
    args.input_directory = run / "input"
    args.input_directory.mkdir()
    shutil.copyfile(args.source_video, args.input_directory / "source.mp4")
    source_sha = sha(args.source_video)
    if sha(args.input_directory / "source.mp4") != source_sha:
        raise ValueError("Owned source copy differs")
    config = probe_resource_config(args.comfy_root, ROOT)
    config["t8_runtime_models"]["taehv"] = str(args.comfy_root / "models/taehv")
    args.extra_model_paths_config = run / "paths.json"
    write(args.extra_model_paths_config, config)
    report = {"schema": "t8.s27.rgb-source-trained-full-cold.v1", "status": "started",
              "source": {"path": str(args.source_video), "sha": source_sha}, "preflight": ready,
              "checks": {}, "phases": {}, "settings": vars(args).copy(),
              "boundary": "Real saved-graph API execution; not canvas, effects, all-material quality, or H3 first-pass rerun."}
    report["settings"] = {k: str(v) if isinstance(v, Path) else v for k, v in report["settings"].items()}
    servers = []
    receipt = None
    try:
        for group in (("full_save", "freeze_source"), ("resume_ltx",)):
            owned = run / ("full" if group[0] == "full_save" else "cold")
            owned.mkdir()
            if receipt:
                copy_source(receipt, owned)
            server = shared.IsolatedServer(args, owned, group[0])
            servers.append(server)
            with server:
                with urllib.request.urlopen(f"http://{args.host}:{args.port}/object_info", timeout=30) as response:
                    info = json.load(response)
                for variant in group:
                    frontend, original = saved[variant]
                    serialized = builder.split_api(frontend, info)
                    if "1" in serialized:
                        serialized["1"]["inputs"]["file"] = "0.6.mp4"  # Saved static schema fixture only.
                    if serialized != original:
                        raise ValueError("Live saved graph/schema differs")
                    graph = execution_graph(original, variant, prompt=args.prompt, width=args.width,
                                            height=args.height, receipt=receipt)
                    write(run / (variant + ".prompt.json"), graph)
                    phase = asyncio.run(pdd._submit_prompt_capture(server=f"http://{args.host}:{args.port}",
                        prompt=graph, timeout_seconds=args.timeout_seconds))
                    write(run / (variant + ".phase.json"), phase)
                    checks = phase_checks(phase, variant)
                    report["checks"].update({variant + ":" + key: value for key, value in checks.items()})
                    report["phases"][variant] = {"pid": server.process.pid, "elapsed": phase["elapsed_seconds"]}
                    if not all(checks.values()):
                        raise RuntimeError("Graph execution failed: " + variant)
                    if variant != "resume_ltx":
                        frozen = source_receipt(owned, phase)
                        report["phases"][variant]["source_receipt"] = frozen
                        if variant == "full_save":
                            receipt = frozen
                            report["prep"] = json.loads(pdd._phase_text(phase, REPORTS["prep"]))
                        else:
                            report["checks"]["freeze_only_matches_full_source"] = frozen["state_sha"] == receipt["state_sha"]
                    print(json.dumps({"phase": variant, "checks": checks}), flush=True)
        full, cold = run / "full", run / "cold"
        report["latent"] = latent_comparison(full, cold)
        report["checks"]["exact_full_cold_candidate_latent"] = report["latent"]["finite"] and report["latent"]["equal"]
        report["media"] = {label: media(folder, prefix, isolated=args.isolated_media)
                           for label, folder, prefix in (("full", full, "full_save"), ("cold", cold, "resume_ltx"),
                               ("source_full", full, "source_reference"), ("source_cold", cold, "source_reference"))}
        report["checks"].update(media_checks(report["media"], report["prep"]))
        report["checks"]["fresh_process"] = report["phases"]["full_save"]["pid"] != report["phases"]["resume_ltx"]["pid"]
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["checks"].update(sources_stable=source_snapshot(ROOT) == before,
            original_video_unchanged=sha(args.source_video) == source_sha,
            owned_servers_stopped=all(s.process is None or s.process.poll() is not None for s in servers),
            owned_port_stopped=not shared.port_is_listening(args.host, args.port))
        report["status"] = "mechanical_pass_not_quality_acceptance" if "error" not in report and all(report["checks"].values()) else "fail"
        write(run / "report.json", report)
        print(json.dumps({"run_root": str(run), "status": report["status"],
                          "failed": [k for k, v in report["checks"].items() if not v]}), flush=True)
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
