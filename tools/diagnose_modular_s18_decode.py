"""Decode saved equal S18 final AV in isolated Cores without resampling."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_modular_s18_full_parity_gpu as pair  # noqa: E402
import run_modular_s18_chunked_gpu as s18  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SOURCE = pair.RUNS / "20260925T055402Z-pair"
EXTENSION = TOOLS / "s18_decode_probe_extension"


def build_graph(original: dict, final: dict, *, with_media: bool = False,
                kind: str = "full") -> dict:
    graph = {key: dict(node) for key, node in original.items()}
    graph["55"] = {"class_type": "MiniMaxH3ChunkedV1SegmentLoadEXPT8", "inputs": {
        "source_segment": ["29", 0], "segment_spec": ["29", 1],
        "pass2_context": ["21", 0], "plan": ["14", 0],
        "artifact_path": final["path"], "artifact_sha256": final["sha256"]}}
    graph["18"] = {"class_type": "MiniMaxH3AVDecodeT8", "inputs": {
        "av_latent": ["55", 0], "video_vae": ["1", 0], "audio_vae": ["2", 0]}}
    graph["300"] = {"class_type": "T8S18DecodeProbe", "inputs": {
        "frames": ["18", 0], "audio": ["18", 1]}}
    graph["301"] = {"class_type": "PreviewAny", "inputs": {"source": ["300", 0]}}
    if with_media:
        graph["19"] = {"class_type": "VHS_VideoCombine", "inputs": {
            **original["19"]["inputs"], "images": ["18", 0], "audio": ["18", 1],
            "filename_prefix": f"MiniMaxH3/S18_Decode_Probe/{kind}"}}
    wanted = set()

    def visit(key):
        if key in wanted:
            return
        wanted.add(key)
        for value in graph[key]["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                visit(str(value[0]))

    visit("301")
    if with_media:
        visit("19")
    selected = {key: graph[key] for key in wanted}
    if any(key in selected for key in ("12", "25", "28", "31", "41")):
        raise ValueError("S18 decode-only graph includes a sampler or prior segment")
    return selected


def _media(root: Path, kind: str) -> dict:
    files = list((root / "output/MiniMaxH3/S18_Decode_Probe").glob(f"{kind}_*-audio.mp4"))
    if len(files) != 1:
        raise ValueError("S18 decode probe expected one private audio MP4")
    file = files[0]
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(file)],
                           capture_output=True, text=True, timeout=30, check=True)
    streams = json.loads(probe.stdout)["streams"]
    video = [item for item in streams if item.get("codec_type") == "video"]
    audio = [item for item in streams if item.get("codec_type") == "audio"]
    decode = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(file),
                             "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                            capture_output=True, text=True, timeout=120, check=False)
    return {"path": str(file), "sha256": s18._sha(file),
            "checks": {"h264_56_frames": len(video) == 1 and video[0].get("codec_name") == "h264"
                       and int(video[0].get("nb_frames", 0)) == 56,
                       "one_aac_audio": len(audio) == 1 and audio[0].get("codec_name") == "aac",
                       "full_av_decode": decode.returncode == 0},
            "decoder_stderr": decode.stderr[-1000:],
            "decoded_md5": {stream: s18._decoded_md5(file, stream)
                            for stream in ("video", "audio")}}


def _copy(source_root: Path, target_root: Path, subdir: str, path: str, sha: str,
          state_sha: str | None = None) -> None:
    root = (source_root / "output/MiniMaxH3" / subdir).resolve()
    selected = (root / path).resolve()
    if not selected.is_relative_to(root) or not selected.is_file() or s18._sha(selected).lower() != sha.lower():
        raise ValueError("S18 decode probe source receipt changed")
    paths = [(selected, sha)]
    if state_sha is not None:
        paths.append((selected.parent / "state.safetensors", state_sha))
    for file, digest in paths:
        target = target_root / "output/MiniMaxH3" / subdir / file.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
        if s18._sha(target).lower() != digest.lower():
            raise ValueError("S18 decode probe copied receipt changed")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=SOURCE)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8879)
    parser.add_argument("--with-media", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.source_run, args.comfy_root, args.python = (
        args.source_run.resolve(), args.comfy_root.resolve(), args.python.resolve())
    prior = json.loads((args.source_run / "report.json").read_text(encoding="utf-8"))
    if prior.get("status") != "fail" or prior.get("checks", {}).get("final_av_tensor_identity_equal") is not True:
        raise ValueError("S18 source run is not the equal-latent media mismatch under investigation")
    ready = EXTENSION.is_dir() and not shared.port_is_listening("127.0.0.1", args.port)
    print(json.dumps({"ready": ready, "source_run": str(args.source_run),
                      "full_final_sha": prior["full"]["final"]["sha256"],
                      "cold_final_sha": prior["cold"]["final"]["sha256"]}, indent=2), flush=True)
    if not args.confirm_run or not ready:
        return 0 if ready else 2
    root = args.source_run / ("decode-diagnostic-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    root.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "source_run": str(args.source_run),
              "with_media": args.with_media}
    try:
        for kind in ("full", "cold"):
            source = args.source_run / kind
            target = root / kind
            target.mkdir()
            (target / "input").mkdir()
            (target / "paths.json").write_text(json.dumps(
                probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
            prompt = json.loads((args.source_run / "cold/prompt.json").read_text(encoding="utf-8"))
            graph = build_graph(prompt, prior[kind]["final"],
                                with_media=args.with_media, kind=kind)
            (target / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
            native = prior["full"]["receipts"]["native"]
            _copy(source, target, "latent_checkpoints", native["path"], native["sha256"])
            final = prior[kind]["final"]
            _copy(source, target, "chunked_v1_segment_artifacts", final["path"], final["sha256"],
                  final["manifest"]["state_sha256"])
            args.host = "127.0.0.1"
            args.input_directory = target / "input"
            args.extra_model_paths_config = target / "paths.json"
            args.extra_whitelist_custom_nodes = (EXTENSION.name,)
            args.server_start_timeout = 240.
            with shared.IsolatedServer(args, target, f"s18-decode-{kind}") as server:
                report[kind + "_core_pid"] = server.process.pid
                phase = asyncio.run(capture._submit_prompt_capture(
                    server=f"http://127.0.0.1:{args.port}", prompt=graph, timeout_seconds=600.))
            (target / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
            if (phase.get("terminal") or {}).get("type") != "execution_success":
                raise RuntimeError(f"S18 {kind} decode-only graph did not complete")
            report[kind] = json.loads(capture._phase_text(phase, "301"))
            if args.with_media:
                report[kind + "_media"] = _media(target, kind)
        first = report["full"]
        second = report["cold"]
        report["checks"] = {
            "same_raw_frames": first["frames"]["sha256"] == second["frames"]["sha256"],
            "same_audio_waveform": first["audio"]["sha256"] == second["audio"]["sha256"],
            "all_finite": first["frames"]["nonfinite"] == second["frames"]["nonfinite"] == 0,
            "expected_56_frames": first["frames"]["shape"][0] == second["frames"]["shape"][0] == 56,
            "owned_core_stopped": not shared.port_is_listening("127.0.0.1", args.port),
        }
        report["different_frame_indexes"] = [index for index, (one, two) in enumerate(zip(
            first["frame_sha256"], second["frame_sha256"])) if one != two]
        if args.with_media:
            full_media, cold_media = report["full_media"], report["cold_media"]
            report["checks"]["both_media_valid"] = all(full_media["checks"].values()) and all(
                cold_media["checks"].values())
            report["checks"]["decoded_video_audio_equal"] = (
                full_media["decoded_md5"] == cold_media["decoded_md5"])
        report["status"] = "diagnosed"
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
