"""Run the source-bound FastH3 V2 split 124+68 chain against an old-loop baseline.

This is an isolated, mechanical GPU probe. The first candidate is explicitly
accepted by this controller for the sole purpose of exercising its continuation;
that action is not a human picture/sound approval. No public workflow is edited.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
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
from tools import build_modular_fast_h3_v2_continuation_workflow as second  # noqa: E402
from tools import build_modular_fast_h3_v2_review_workflow as review  # noqa: E402
from tools import build_modular_fast_h3_v2_compose_workflow as compose  # noqa: E402
from tools import build_modular_fast_h3_v2_workflow as base  # noqa: E402
import run_fast_h3_v2_loop_probe as old  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as pdd  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s08-old-new-pair-gpu.v1"
CHAIN = "s08_split_pair_20260925"
ROOT = PROJECT / "artifacts/development/modular-sampling-s08-old-new-gpu-20260925"
FIRST_FRAME = "t8_fasth3_v2_loop_a89496c9bdc8cada29af5951.png"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _memory_wrappers(graph: dict) -> dict:
    """Use precisely the old loop's 4-head/2-FFN external memory stack."""
    graph = deepcopy(graph)
    memory_ids = (("105", "106", "107", "108")
                  if not any(key in graph for key in ("105", "106", "107", "108"))
                  else tuple(str(max(map(int, graph)) + offset) for offset in range(1, 5)))
    for raw, attention, ffn in (("1", memory_ids[0], memory_ids[1]),
                                ("22", memory_ids[2], memory_ids[3])):
        for item in graph.values():
            for name, value in item["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and value == [raw, 0]:
                    item["inputs"][name] = [ffn, 0]
        graph[attention] = {"class_type": old.LOWVRAM,
                            "inputs": {"model": [raw, 0], "head_chunks": 4}}
        graph[ffn] = {"class_type": old.FFN,
                      "inputs": {"model": [attention, 0], "chunks": 2,
                                 "seq_threshold": 4096}}
    return graph


FLAGS = {"with_current_recipe": True, "with_stage_attestation": True,
         "with_condition_provenance": True, "with_handoff_provenance": True,
         "with_job_binding": True, "with_media_provenance": True,
         "with_candidate_save": True, "with_frozen_low_bundle": False}


def first_graph(*, with_color_match=False) -> dict:
    graph = first.first_segment_graph(FIRST_FRAME, **FLAGS, with_color_match=with_color_match)
    for node in graph.values():
        if node["class_type"] == "MiniMaxH3FastH3V2CurrentRecipeEXPT8":
            node["inputs"]["continuation_render_frames"] = 124
    graph["70"]["inputs"]["chain_id"] = CHAIN
    graph["16"]["inputs"]["filename_prefix"] = "MiniMaxH3/S08_PairSplit_First_Preview"
    graph["98"]["inputs"]["candidate_id"] = "split_gpu_first_20260925"
    return _memory_wrappers(graph)


def second_graph(parent: dict, *, with_color_match=False) -> dict:
    graph = second.continuation_graph(first_frame=FIRST_FRAME, with_delivery=True,
                                      **FLAGS, with_color_match=with_color_match)
    graph["70"]["inputs"].update(chain_id=CHAIN, segment_index=1,
        parent_candidate_id=parent["candidate_id"], parent_revision=parent["revision"],
        previous_job_sha256=parent["job_sha256"])
    for node in graph.values():
        if node["class_type"] == "MiniMaxH3FastH3V2CurrentRecipeEXPT8":
            node["inputs"]["continuation_render_frames"] = 124
    # The old loop always samples its 124-frame second window, even though
    # only frames 22..89 become the 68-frame accepted remainder. The compact
    # 90-frame split remains the default in saved/private example builders.
    graph["77"]["inputs"]["render_policy"] = "old_fixed_124"
    graph["78"]["inputs"]["render_policy"] = "old_fixed_124"
    graph["74"]["inputs"]["accepted_end_frame"] = 192
    graph["75"]["inputs"]["accepted_end_frame"] = 192
    graph["16"]["inputs"]["filename_prefix"] = "MiniMaxH3/S08_PairSplit_Second_Preview"
    graph["98"]["inputs"]["candidate_id"] = "split_gpu_second_20260925"
    return _memory_wrappers(graph)


def _candidate(run_root: Path, segment: int, candidate_id: str) -> Path:
    root = run_root / "output/minimax_h3_t8_long_video" / CHAIN / "candidates"
    path = root / f"segment_{segment:05d}" / candidate_id / "candidate.json"
    if not path.is_file() or not path.resolve().is_relative_to(run_root.resolve()):
        raise FileNotFoundError(f"Expected source-bound candidate: {path}")
    return path


def _phase(server: str, graph: dict, timeout: float, run_root: Path, name: str) -> dict:
    _write(run_root / f"{name}-prompt.json", graph)
    result = asyncio.run(pdd._submit_prompt_capture(
        server=server, prompt=graph, timeout_seconds=timeout))
    _write(run_root / f"{name}-phase.json", result)
    terminal = (result.get("terminal") or {}).get("type")
    if terminal != "execution_success":
        raise RuntimeError(f"{name}: {terminal}; inspect {name}-phase.json and owned Core logs")
    return result


def _accept(server: str, candidate: Path, timeout: float, run_root: Path, name: str) -> dict:
    graph = review.review_graph(str(candidate))
    graph["1"]["inputs"]["accept_candidate"] = True
    phase = _phase(server, graph, timeout, run_root, name)
    texts = (phase.get("executed_outputs") or {}).get("2", {}).get("text") or []
    if not texts:
        raise ValueError(f"{name}: missing explicit acceptance report")
    report = json.loads(texts[0])
    if report.get("status") != "accepted" or report.get("chain_id") != CHAIN:
        raise ValueError(f"{name}: candidate not accepted")
    return {"candidate_id": report["candidate_id"],
            "revision": report["manifest_revision"],
            "job_sha256": report["current_job_sha256"], "report": report}


def preflight(args: argparse.Namespace, old_terminal: dict, old_loop: dict) -> dict:
    gpu = shared.gpu_memory_mib()
    model = args.comfy_root / "models/diffusion_models" / old.MODEL
    image = args.comfy_root / "input" / FIRST_FRAME
    checks = {
        "old_loop_mechanical_pass": old_terminal.get("status") ==
            "mechanical_8s_AV_stage_contract_pass_human_review_pending",
        "old_loop_same_chain": old_loop.get("chain_id") == CHAIN and
            old_loop.get("status") == "complete",
        "actual_model_present": model.is_file(),
        "byte_identical_reference_present": image.is_file() and
            shared._sha256_file(image).lower() == old_terminal["reference"]["sha256"],
        "isolated_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "gpu_headroom": bool(gpu.get("available") and
                             gpu["free_mib"] >= args.min_free_vram_mib),
    }
    return {"schema": SCHEMA + ".preflight", "gpu": gpu,
            "checks": checks, "ready": all(checks.values())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8894)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=float, default=2400)
    parser.add_argument("--server-start-timeout", type=float, default=240)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--with-color-match", action="store_true",
                        help="Use the versioned external old-default RGB Color Match and colored candidate writer")
    parser.add_argument("--first-only", action="store_true",
                        help="Run only the full first-segment graph for same-backend numeric diagnosis")
    parser.add_argument("--diagnostic-low", action="store_true",
                        help="Run only the first LOW Stage Save for identity diagnosis")
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve(strict=True)
    args.python = args.python.resolve(strict=True)
    old_root = ROOT / "old-dense-current-source-v2"
    old_terminal = json.loads((old_root / "terminal.json").read_text(encoding="utf-8"))
    old_loop = json.loads((old_root / "loop-report.json").read_text(encoding="utf-8"))
    readiness = preflight(args, old_terminal, old_loop)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = ROOT / f"split-{'colored' if args.with_color_match else 'dense'}-{run_id}"
    run_root.mkdir(parents=True, exist_ok=False)
    paths = run_root / "paths.json"
    _write(paths, probe_resource_config(args.comfy_root, PROJECT))
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    # The old-loop probe explicitly selects this Core backend. A paired probe
    # cannot claim numeric parity if the new Core silently defaults to xformers.
    args.use_pytorch_cross_attention = True
    report = {"schema": SCHEMA, "status": "started", "old_root": str(old_root),
              "run_root": str(run_root), "preflight": readiness,
              "frozen_low_bundle": "not_exercised; frozen LOW is a separate explicit test route",
              "parity_boundary": ("External bounded-motion Color Match and old fixed second window enabled; "
                  "numeric old/new parity and human approval require independent evidence"
                  if args.with_color_match else
                  "Split candidate has no external bounded-motion color match; "
                  "do not claim default-old final RGB parity or human approval")}
    try:
        with shared.IsolatedServer(args, run_root, "s08-pair-split") as owned:
            report["owned_core_pid"] = owned.process.pid
            server = f"http://127.0.0.1:{args.port}"
            if args.diagnostic_low:
                graph = base.split_graph_with_results(base.split_graph_with_effects())
                graph["9"]["inputs"].update(width=128, height=64)
                graph["40"]["inputs"]["length"] = 22
                graph["10"]["inputs"]["min_tokens"] = 0
                graph = second._prune(_memory_wrappers(graph), ["50"])
                _phase(server, graph, args.timeout_seconds, run_root, "diagnostic-low")
                report["status"] = "diagnostic_low_pass"
                return 0
            report["first_phase"] = _phase(server, first_graph(with_color_match=args.with_color_match), args.timeout_seconds,
                                           run_root, "first")["terminal"]["type"]
            if args.first_only:
                report["status"] = "first_segment_same_backend_diagnostic_pass_parity_pending"
                return 0
            first_candidate = _candidate(run_root, 0, "split_gpu_first_20260925")
            report["first_candidate"] = str(first_candidate)
            parent = _accept(server, first_candidate, args.timeout_seconds, run_root, "first-accept")
            report["first_mechanical_accept"] = parent
            report["second_phase"] = _phase(server, second_graph(parent, with_color_match=args.with_color_match), args.timeout_seconds,
                                            run_root, "second")["terminal"]["type"]
            second_candidate = _candidate(run_root, 1, "split_gpu_second_20260925")
            report["second_candidate"] = str(second_candidate)
            report["second_mechanical_accept"] = _accept(
                server, second_candidate, args.timeout_seconds, run_root, "second-accept")
            report["compose_phase"] = _phase(server, compose.compose_graph(CHAIN),
                                             args.timeout_seconds, run_root, "compose")["terminal"]["type"]
        report["status"] = ("split_8s_colored_mechanical_chain_pass_same_source_parity_and_human_review_pending"
                            if args.with_color_match else
                            "split_8s_mechanical_chain_pass_color_parity_and_human_review_pending")
        return 0
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        _write(run_root / "terminal.json", report)
        print(json.dumps({"status": report["status"], "root": str(run_root)},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
