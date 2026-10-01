"""Real V2 first-window full LOW/HIGH versus a new-Core frozen-LOW HIGH run.

This is an isolated mechanical probe, not human picture/sound approval. It
exercises the optional bounded LOW witness sidecar without changing Stage
manifests or any old one-piece workflow.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tools import build_modular_fast_h3_v2_first_segment_workflow as first  # noqa: E402
from tools import run_modular_s08_old_new_pair_gpu as pair  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


ROOT = PROJECT / "artifacts/development/modular-sampling-s08-frozen-real-gpu-20260926"
CHAIN = "s08_frozen_first_20260926"
SCHEMA = "t8.modular-sampling.s08-frozen-first-gpu.v1"
FLAGS = {**pair.FLAGS, "with_candidate_save": False, "with_frozen_low_bundle": True}


def _manifest(run_root: Path, stage: str, *, prior: set[Path] | None = None) -> Path:
    root = run_root / "output/MiniMaxH3/stage_artifacts/FastH3V2"
    paths = set(root.glob(f"{stage}-*/manifest.json"))
    selected = paths if prior is None else paths - prior
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one new {stage} Stage manifest, found {len(selected)}")
    return next(iter(selected))


def _graph(*, cold=False, manifest: Path | None = None, run_root: Path | None = None):
    graph = first.first_segment_graph(pair.FIRST_FRAME, resume_high=cold, **FLAGS)
    graph["70"]["inputs"]["chain_id"] = CHAIN
    graph["16"]["inputs"]["filename_prefix"] = (
        "MiniMaxH3/S08_Frozen_First_Cold" if cold else "MiniMaxH3/S08_Frozen_First_Full")
    if cold:
        if manifest is None or run_root is None:
            raise ValueError("Cold HIGH needs the exact selected LOW manifest")
        stage_root = run_root / "output/MiniMaxH3/stage_artifacts"
        relative = manifest.relative_to(stage_root).as_posix()
        digest = shared._sha256_file(manifest).lower()
        for key in ("60", "102"):
            graph[key]["inputs"].update(artifact_path=relative, artifact_sha256=digest)
    graph = pair._memory_wrappers(graph)
    if cold:
        # The generic wrapper helper emits both branches. The cold graph has
        # no LOW UNET, so remove only its two orphaned adapter nodes.
        for key in ("105", "106"):
            graph.pop(key)
    return graph


def _stage_record(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = path.parent / "state.safetensors"
    if (payload.get("state_sha256") != shared._sha256_file(state).lower()
            or payload.get("portable_identity") is not True):
        raise ValueError("Saved stage is not a complete portable Stage result")
    return {"manifest": str(path), "manifest_sha256": shared._sha256_file(path).lower(),
            "request_sha256": payload["request_sha256"],
            "receipt_sha256": payload["receipt_sha256"],
            "state_sha256": payload["state_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    image = args.comfy_root / "input" / pair.FIRST_FRAME
    model = args.comfy_root / "models/diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors"
    gpu = shared.gpu_memory_mib()
    checks = {"model_present": model.is_file(), "first_frame_present": image.is_file(),
              "port_free": not shared.port_is_listening("127.0.0.1", args.port),
              "gpu_headroom": bool(gpu.get("available") and
                                   gpu["free_mib"] >= args.min_free_vram_mib)}
    readiness = {"gpu": gpu, "checks": checks, "ready": all(checks.values())}
    print(json.dumps(readiness, ensure_ascii=False), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = ROOT / f"frozen-first-{run_id}"
    run_root.mkdir(parents=True, exist_ok=False)
    args.host = "127.0.0.1"
    args.extra_model_paths_config = run_root / "paths.json"
    pair._write(args.extra_model_paths_config, probe_resource_config(args.comfy_root, PROJECT))
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "preflight": readiness,
              "boundary": "First-window 124-frame frozen LOW mechanical identity only; "
                          "no second segment, color match, old-loop parity or human review"}
    try:
        server = f"http://127.0.0.1:{args.port}"
        with shared.IsolatedServer(args, run_root, "s08-frozen-full") as owned:
            report["full_core_pid"] = owned.process.pid
            report["full_phase"] = pair._phase(server, _graph(), args.timeout_seconds,
                                                run_root, "full")["terminal"]["type"]
        low_manifest = _manifest(run_root, "LOW")
        full_high = _manifest(run_root, "HIGH")
        report["low"] = _stage_record(low_manifest)
        report["full_high"] = _stage_record(full_high)
        sidecar = low_manifest.with_name("current-v2-low-bundle.json")
        envelope = json.loads(sidecar.read_text(encoding="utf-8"))
        report["low_bundle"] = {"path": str(sidecar),
            "sha256": shared._sha256_file(sidecar).lower(),
            "schema": envelope.get("schema"),
            "raw_bytes": envelope.get("raw_bytes"),
            "compressed_bytes": envelope.get("compressed_bytes")}
        before_high = set(full_high.parent.parent.glob("HIGH-*/manifest.json"))
        with shared.IsolatedServer(args, run_root, "s08-frozen-cold") as owned:
            report["cold_core_pid"] = owned.process.pid
            report["cold_phase"] = pair._phase(server,
                _graph(cold=True, manifest=low_manifest, run_root=run_root),
                args.timeout_seconds, run_root, "cold")["terminal"]["type"]
        cold_high = _manifest(run_root, "HIGH", prior=before_high)
        report["cold_high"] = _stage_record(cold_high)
        report["stage_equal"] = all(report["full_high"][key] == report["cold_high"][key]
            for key in ("request_sha256", "receipt_sha256", "state_sha256"))
        report["status"] = ("frozen_first_high_exact_mechanical_pass_human_review_pending"
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
