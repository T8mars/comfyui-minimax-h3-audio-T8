"""Isolated real-weight S08 trained VSA + external stage EAV probe.

This makes a private execution copy of the saved split graph. Prompt Relay is
removed explicitly: the native sparse VSA kernel cannot yet express its
query-varying timeline bias. No accepted graph or user Core is changed.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_modular_s08_fastv2_gpu as base  # noqa: E402

SCHEMA = "t8.modular-sampling.s08-vsa-eav-real-gpu.v1"


def replace_relay_with_native_conditioning(graph: dict, node_id: str, model_id: str,
                                           plan_id: str) -> None:
    """Explicitly remove one unsupported sparse Relay branch in a probe copy."""
    node = graph[node_id]
    plan = graph[plan_id]
    if (node["class_type"] != "MiniMaxH3PromptRelayConditioningT8Advanced"
            or plan["class_type"] != "MiniMaxH3PromptRelayPlanT8Advanced"
            or node["inputs"]["model"] != [model_id, 0]
            or node["inputs"]["prompt_relay_plan"] != [plan_id, 0]):
        raise ValueError("Saved independent Relay branch changed")
    # Native conditioning gives (positive, AV); Relay gives (MODEL,
    # positive, AV). Rewrite every consumer, including audits and handoff.
    for target in graph.values():
        for name, value in list(target["inputs"].items()):
            if isinstance(value, list) and len(value) == 2 and value[0] == node_id:
                slot = value[1]
                if slot not in (0, 1, 2):
                    raise ValueError("Unknown Relay output slot")
                target["inputs"][name] = ([model_id, 0] if slot == 0
                                           else [node_id, slot - 1])
    inputs = node["inputs"]
    for name in ("model", "prompt_relay_plan", "execution_mode", "query_chunk_rows"):
        inputs.pop(name)
    inputs["prompt"] = plan["inputs"]["global_prompt"]
    inputs["length"] = 22
    node["class_type"] = "MiniMaxH3AudioConditioningT8"
    if any(isinstance(value, list) and len(value) == 2 and value[0] == plan_id
           for target in graph.values() for value in target["inputs"].values()):
        raise ValueError("An external Relay Plan remains connected")
    graph.pop(plan_id)


def build_probe_graph() -> tuple[dict, str]:
    graph, source_sha = base.build_probe_graph()
    for node_id, model_id, plan_id in (("9", "1", "40"), ("24", "22", "47")):
        replace_relay_with_native_conditioning(graph, node_id, model_id, plan_id)
    for stage_id, config_id in (("10", "41"), ("26", "42")):
        graph[stage_id]["inputs"].update(profile="trained_vsa_exp", min_tokens=0)
        graph[config_id]["inputs"].update(mode="apply_exp", tau=4.0,
            start_video_progress=0.0, end_video_progress=1.0, g_hard_limit=3.0)
    return graph, source_sha


def _audit(phase: dict, node_id: str) -> dict:
    text = (phase.get("executed_outputs") or {}).get(node_id, {}).get("text")
    if not isinstance(text, list) or not text:
        raise ValueError(f"Missing actual stage EAV audit {node_id}")
    return json.loads(text[0])


def _effect_checks(audit: dict) -> dict[str, bool]:
    feta = audit.get("feta") or {}
    dispatch = audit.get("v2_dispatch") or {}
    return {
        "observed_apply_exp": audit.get("status") == "observed_apply_exp",
        "native_sparse_dispatched": dispatch.get("actual_vsa_dispatched") is True
                                    and (dispatch.get("counts") or {}).get("vsa") == 200
                                    and not (dispatch.get("dense_reasons") or {}),
        "all_four_forwards": audit.get("completed_forwards") == audit.get("planned_forwards") == 4,
        "sparse_producer_covered": audit.get("sparse_producer_calls", 0) >= 200,
        "gain_above_identity": float(feta.get("g_max") or 0) > 1.0001,
        "no_unqualified_relay": audit.get("relay_required") is False
                                and audit.get("relay_attention_calls") == 0,
        "clock_matches": audit.get("clock_match") is True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=base.PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8864)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--server-start-timeout", type=float, default=240)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    graph, source_sha = build_probe_graph()
    readiness = base.preflight(args, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (base.PROJECT / "artifacts/development/modular-sampling-m1-v2-vsa-eav-20260924"
                / f"{run_id}-full")
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(base.probe_resource_config(args.comfy_root, base.PROJECT), indent=2),
                     encoding="utf8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "candidate_sha256": source_sha, "preflight": readiness,
              "test_canvas": [128, 64, 22], "profile": "trained_vsa_exp",
              "relay": "not_connected_unsupported_sparse_bias", "status": "started"}
    try:
        with base.shared.IsolatedServer(args, run_root, "s08-fastv2-vsa-eav") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(base.pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf8")
        checks = base._stage_checks(phase)
        report["checks"] = checks
        if checks["terminal_success"]:
            report["low_save"] = base._stage_save_receipt(phase, "50")
            report["high_save"] = base._stage_save_receipt(phase, "51")
            checks["low_artifact_file_sha"] = base._check_stage_file(run_root, report["low_save"])
            checks["high_artifact_file_sha"] = base._check_stage_file(run_root, report["high_save"])
            report["low_eav"] = _audit(phase, "45")
            report["high_eav"] = _audit(phase, "46")
            checks.update({"low_" + key: value for key, value in _effect_checks(report["low_eav"]).items()})
            checks.update({"high_" + key: value for key, value in _effect_checks(report["high_eav"]).items()})
            checks.update(base._media_checks(run_root))
        checks["source_candidate_unchanged"] = base.shared._sha256_file(base.CANDIDATE) == source_sha
        checks["server_stopped"] = not base.shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("small_canvas_real_vsa_eav_mechanical_pass_not_quality_acceptance"
                            if all(checks.values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("S08 trained VSA/EAV mechanical qualification failed")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    raise SystemExit(main())
