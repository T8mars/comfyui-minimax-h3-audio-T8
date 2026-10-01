"""Real accepted-parent V2 second window: full versus new-Core frozen LOW HIGH.

Copy only a previously completed first candidate into a fresh, owned output
scope, review/accept it there, then execute the same second-window contract
once in full and once with LOW restored from its typed bundle. This is a
mechanical identity probe, not a human picture/sound approval.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tools import build_modular_fast_h3_v2_continuation_workflow as second  # noqa: E402
from tools import run_modular_s08_frozen_first_gpu as frozen_first  # noqa: E402
from tools import run_modular_s08_old_new_pair_gpu as pair  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


ROOT = PROJECT / "artifacts/development/modular-sampling-s08-frozen-real-gpu-20260926"
SCHEMA = "t8.modular-sampling.s08-frozen-second-gpu.v1"
FLAGS = {**pair.FLAGS, "with_candidate_save": False, "with_frozen_low_bundle": True}


def _first_candidate(source_root: Path) -> Path:
    terminal = json.loads((source_root / "terminal.json").read_text(encoding="utf-8"))
    if terminal.get("status") != (
            "split_8s_colored_mechanical_chain_pass_same_source_parity_and_human_review_pending"):
        raise ValueError("Frozen second needs a completed source-bound split chain")
    relative = (Path("output/minimax_h3_t8_long_video") / pair.CHAIN /
                "candidates/segment_00000/split_gpu_first_20260925/candidate.json")
    candidate = (source_root / relative).resolve(strict=True)
    if not candidate.is_relative_to(source_root.resolve()):
        raise ValueError("First candidate leaves the selected split evidence root")
    data = json.loads(candidate.read_text(encoding="utf-8"))
    if (data.get("chain_id") != pair.CHAIN or data.get("index") != 0
            or data.get("frame_count") != 124 or data.get("status") != "candidate"
            or data.get("video_sha256") != shared._sha256_file(candidate.parent / "candidate.mp4").lower()
            or data.get("context_sha256") != shared._sha256_file(
                candidate.parent / "candidate.context.safetensors").lower()):
        raise ValueError("First candidate payload/actual media is not the accepted source")
    return candidate


def _copy_first_candidate(source: Path, run_root: Path) -> Path:
    destination = (run_root / "output/minimax_h3_t8_long_video" / pair.CHAIN /
                   "candidates/segment_00000" / source.parent.name)
    if destination.exists() or source.parent.is_symlink():
        raise ValueError("First candidate destination exists or source is a link")
    files = {item.name: item for item in source.parent.iterdir()}
    expected = {"candidate.json", "candidate.mp4", "candidate.context.safetensors",
                "source-provenance.json"}
    if set(files) != expected or any(not item.is_file() or item.is_symlink()
                                     for item in files.values()):
        raise ValueError("First candidate folder has unknown or linked files")
    shutil.copytree(source.parent, destination)
    for name, original in files.items():
        if shared._sha256_file(original) != shared._sha256_file(destination / name):
            raise RuntimeError("Copied first candidate differs: " + name)
    return destination / "candidate.json"


def _graph(parent: dict, *, cold: bool = False, manifest: Path | None = None,
           run_root: Path | None = None) -> dict:
    graph = second.continuation_graph(first_frame=pair.FIRST_FRAME,
        resume_high=cold, with_delivery=True, **FLAGS)
    graph["70"]["inputs"].update(chain_id=pair.CHAIN, segment_index=1,
        parent_candidate_id=parent["candidate_id"], parent_revision=parent["revision"],
        previous_job_sha256=parent["job_sha256"])
    if not cold:
        graph["83"]["inputs"]["continuation_render_frames"] = 124
    graph["77"]["inputs"]["render_policy"] = "old_fixed_124"
    graph["78"]["inputs"]["render_policy"] = "old_fixed_124"
    for key in ("74", "75"):
        if key in graph:
            graph[key]["inputs"]["accepted_end_frame"] = 192
    graph["16"]["inputs"]["filename_prefix"] = (
        "MiniMaxH3/S08_Frozen_Second_Cold" if cold else "MiniMaxH3/S08_Frozen_Second_Full")
    if cold:
        if manifest is None or run_root is None:
            raise ValueError("Cold second HIGH needs one exact saved LOW manifest")
        stage_root = run_root / "output/MiniMaxH3/stage_artifacts"
        relative = manifest.relative_to(stage_root).as_posix()
        digest = shared._sha256_file(manifest).lower()
        for key in ("60", "102"):
            graph[key]["inputs"].update(artifact_path=relative, artifact_sha256=digest)
    graph = pair._memory_wrappers(graph)
    if cold:
        for key in ("105", "106"):
            graph.pop(key)
        if any(key in graph for key in ("1", "13", "83", "85", "89")):
            raise RuntimeError("Cold second graph retained a LOW sampler/recipe source")
    return graph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split-root", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8894)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=float, default=2400)
    parser.add_argument("--server-start-timeout", type=float, default=240)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve(strict=True)
    args.python = args.python.resolve(strict=True)
    source_root = args.source_split_root.resolve(strict=True)
    source = _first_candidate(source_root)
    gpu = shared.gpu_memory_mib()
    checks = {"first_candidate_present": source.is_file(),
              "port_free": not shared.port_is_listening("127.0.0.1", args.port),
              "gpu_headroom": bool(gpu.get("available") and
                                   gpu["free_mib"] >= args.min_free_vram_mib)}
    readiness = {"gpu": gpu, "checks": checks, "ready": all(checks.values())}
    print(json.dumps(readiness, ensure_ascii=False), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = ROOT / f"frozen-second-{run_id}"
    run_root.mkdir(parents=True, exist_ok=False)
    copied = _copy_first_candidate(source, run_root)
    args.host = "127.0.0.1"
    args.use_pytorch_cross_attention = True
    args.extra_model_paths_config = run_root / "paths.json"
    pair._write(args.extra_model_paths_config, probe_resource_config(args.comfy_root, PROJECT))
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "source_split_root": str(source_root), "source_candidate": str(source),
              "copied_candidate": str(copied), "preflight": readiness,
              "boundary": "Accepted-parent second-window frozen LOW mechanical identity only; "
                          "no automatic human picture/sound approval"}
    try:
        server = f"http://127.0.0.1:{args.port}"
        with shared.IsolatedServer(args, run_root, "s08-frozen-second-full") as owned:
            report["full_core_pid"] = owned.process.pid
            parent = pair._accept(server, copied, args.timeout_seconds, run_root, "parent-accept")
            report["parent_mechanical_accept"] = parent
            if parent["job_sha256"] != json.loads(source.read_text(encoding="utf-8"))["sampling_summary"]:
                raise ValueError("Copied accepted parent differs from its original job")
            report["full_phase"] = pair._phase(server, _graph(parent), args.timeout_seconds,
                                                run_root, "full")["terminal"]["type"]
        low_manifest = frozen_first._manifest(run_root, "LOW")
        full_high = frozen_first._manifest(run_root, "HIGH")
        report["low"] = frozen_first._stage_record(low_manifest)
        report["full_high"] = frozen_first._stage_record(full_high)
        sidecar = low_manifest.with_name("current-v2-low-bundle.json")
        envelope = json.loads(sidecar.read_text(encoding="utf-8"))
        report["low_bundle"] = {"path": str(sidecar),
            "sha256": shared._sha256_file(sidecar).lower(),
            "schema": envelope.get("schema"), "raw_bytes": envelope.get("raw_bytes"),
            "compressed_bytes": envelope.get("compressed_bytes")}
        before_high = set(full_high.parent.parent.glob("HIGH-*/manifest.json"))
        with shared.IsolatedServer(args, run_root, "s08-frozen-second-cold") as owned:
            report["cold_core_pid"] = owned.process.pid
            report["cold_phase"] = pair._phase(server,
                _graph(parent, cold=True, manifest=low_manifest, run_root=run_root),
                args.timeout_seconds, run_root, "cold")["terminal"]["type"]
        cold_high = frozen_first._manifest(run_root, "HIGH", prior=before_high)
        report["cold_high"] = frozen_first._stage_record(cold_high)
        report["stage_equal"] = all(report["full_high"][key] == report["cold_high"][key]
            for key in ("request_sha256", "receipt_sha256", "state_sha256"))
        report["status"] = ("frozen_second_high_exact_mechanical_pass_human_review_pending"
                            if report["stage_equal"] else "failed_stage_identity_difference")
        return 0 if report["stage_equal"] else 3
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        pair._write(run_root / "terminal.json", report)
        print(json.dumps({"status": report["status"], "root": str(run_root)},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
