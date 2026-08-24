#!/usr/bin/env python3
"""Run one targeted Creator background cancellation probe against ComfyUI.

This is intentionally a single-run diagnostic, not a load or stress harness. It waits until the
requested sampler node reports real progress, cancels the exact bound background prompt, and
records queue/history/state plus conservative GPU-memory observations.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import quote
import uuid


TERMINAL_EVENTS = {"execution_success", "execution_error", "execution_interrupted"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gpu_memory_mib() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        values = [int(float(item.strip())) for item in completed.stdout.splitlines()[0].split(",")]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"available": False}
    return {
        "available": True,
        "total_mib": values[0],
        "used_mib": values[1],
        "free_mib": values[2],
        "utilization_percent": values[3],
        "temperature_c": values[4],
    }


async def _json_request(session, method: str, url: str, **kwargs) -> dict:
    async with session.request(method, url, **kwargs) as response:
        text = await response.text()
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {"raw_text": text}
        if response.status >= 400:
            raise RuntimeError(f"{method} {url} returned HTTP {response.status}: {payload}")
        if not isinstance(payload, dict):
            raise RuntimeError(f"{method} {url} returned a non-object JSON response")
        return payload


async def run_probe(args) -> dict:
    try:
        import aiohttp
    except ImportError as error:
        raise RuntimeError("aiohttp is required in the ComfyUI Python environment") from error

    prompt = json.loads(args.workflow.read_text(encoding="utf-8"))
    if not isinstance(prompt, dict) or not prompt:
        raise ValueError("workflow must contain a non-empty ComfyUI API prompt object")
    base = args.server.rstrip("/")
    client_id = uuid.uuid4().hex
    requested_prompt_id = str(uuid.uuid4())
    chain_url = quote(args.chain_id, safe="")
    baseline = _gpu_memory_mib()
    started = time.monotonic()
    events = []
    cancel_response = None
    terminal = None
    cancelled_at_progress = None

    timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_connect=20, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await _json_request(session, "GET", f"{base}/system_stats")
        ws_url = base.replace("http://", "ws://").replace("https://", "wss://")
        async with session.ws_connect(f"{ws_url}/ws?clientId={client_id}", heartbeat=30) as ws:
            submitted = await _json_request(
                session,
                "POST",
                f"{base}/prompt",
                json={
                    "prompt": prompt,
                    "client_id": client_id,
                    "prompt_id": requested_prompt_id,
                },
            )
            prompt_id = str(submitted.get("prompt_id") or "")
            if prompt_id != requested_prompt_id:
                raise RuntimeError(
                    f"ComfyUI returned prompt_id={prompt_id!r}, expected {requested_prompt_id!r}"
                )
            deadline = time.monotonic() + args.timeout_seconds
            while terminal is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for target node {args.target_node} progress"
                    )
                try:
                    message = await asyncio.wait_for(ws.receive(), timeout=min(1.0, remaining))
                except asyncio.TimeoutError:
                    continue
                if message.type == aiohttp.WSMsgType.BINARY:
                    continue
                if message.type in {
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                }:
                    raise RuntimeError("ComfyUI WebSocket closed before terminal execution event")
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    packet = json.loads(message.data)
                except json.JSONDecodeError:
                    continue
                event_type = packet.get("type")
                data = packet.get("data") if isinstance(packet.get("data"), dict) else {}
                event_prompt_id = data.get("prompt_id")
                if event_prompt_id not in {None, prompt_id}:
                    continue
                if event_type in {
                    "execution_start",
                    "execution_cached",
                    "executing",
                    "executed",
                    "progress_state",
                    *TERMINAL_EVENTS,
                }:
                    events.append(
                        {
                            "elapsed_seconds": round(time.monotonic() - started, 4),
                            "type": event_type,
                            "node": data.get("node"),
                        }
                    )
                if event_type == "progress_state" and cancel_response is None:
                    nodes = data.get("nodes")
                    progress = nodes.get(str(args.target_node)) if isinstance(nodes, dict) else None
                    if isinstance(progress, dict) and progress.get("state") == "running":
                        value = int(progress.get("value") or 0)
                        maximum = int(progress.get("max") or 0)
                        if value >= args.cancel_at_progress:
                            cancelled_at_progress = {"value": value, "max": maximum}
                            cancel_response = await _json_request(
                                session,
                                "POST",
                                f"{base}/minimax_h3_t8/long_video/background/{chain_url}/cancel",
                            )
                if event_type in TERMINAL_EVENTS and event_prompt_id == prompt_id:
                    terminal = {"type": event_type, "data": data}

        # The queue consumes release flags after interruption. Observe, do not force, for a
        # bounded interval so the report distinguishes signal delivery from actual memory drop.
        memory_samples = []
        for _ in range(args.release_observation_samples):
            state = await _json_request(
                session,
                "GET",
                f"{base}/minimax_h3_t8/long_video/background/{chain_url}",
            )
            queue = await _json_request(session, "GET", f"{base}/queue")
            memory_samples.append(
                {
                    "elapsed_seconds": round(time.monotonic() - started, 4),
                    "gpu": _gpu_memory_mib(),
                    "state": state.get("state"),
                    "runtime_location": state.get("runtime_location"),
                    "queue_running": len(queue.get("queue_running") or []),
                    "queue_pending": len(queue.get("queue_pending") or []),
                }
            )
            if memory_samples[-1]["queue_running"] or memory_samples[-1]["queue_pending"]:
                await asyncio.sleep(args.release_observation_interval)
                continue
            await asyncio.sleep(args.release_observation_interval)
        history = await _json_request(session, "GET", f"{base}/history/{prompt_id}")
        state = await _json_request(
            session,
            "GET",
            f"{base}/minimax_h3_t8/long_video/background/{chain_url}",
        )
        queue = await _json_request(session, "GET", f"{base}/queue")

    history_record = history.get(prompt_id) if isinstance(history, dict) else None
    status = history_record.get("status") if isinstance(history_record, dict) else {}
    history_events = []
    for item in status.get("messages", []) if isinstance(status, dict) else []:
        if isinstance(item, list) and item:
            history_events.append(str(item[0]))
    final_gpu = memory_samples[-1]["gpu"] if memory_samples else _gpu_memory_mib()
    memory_return_delta = None
    if baseline.get("available") and final_gpu.get("available"):
        memory_return_delta = int(final_gpu["used_mib"]) - int(baseline["used_mib"])
    checks = {
        "cancel_was_sent": cancel_response is not None,
        "cancel_bound_prompt_matched": (
            isinstance(cancel_response, dict)
            and str(cancel_response.get("last_cancelled_prompt_id") or "")
            == prompt_id
        ),
        "interrupt_signalled": bool(
            isinstance(cancel_response, dict)
            and cancel_response.get("last_control_result", {}).get("interrupt_signalled")
        ),
        "terminal_was_interrupted": (terminal or {}).get("type") == "execution_interrupted",
        "history_contains_interruption": "execution_interrupted" in history_events,
        "background_state_cancelled": state.get("state") == "cancelled",
        "release_policy_recorded": state.get("last_release_policy") == args.expected_release_policy,
        "release_error_absent": not bool(state.get("last_release_error")),
        "queue_empty": not (queue.get("queue_running") or queue.get("queue_pending")),
        "accepted_count_unchanged": int(state.get("accepted_count", -1)) == 0,
        "retry_count_unchanged": int(state.get("retry_count", -1)) == 0,
    }
    return {
        "schema": "t8.creator_background_h3_cancel_probe.v1",
        "created_at": _utc_now(),
        "workflow": str(args.workflow.resolve()),
        "server": base,
        "chain_id": args.chain_id,
        "prompt_id": prompt_id,
        "target_node": str(args.target_node),
        "cancelled_at_progress": cancelled_at_progress,
        "baseline_gpu": baseline,
        "final_gpu": final_gpu,
        "final_minus_baseline_used_mib": memory_return_delta,
        "terminal": terminal,
        "history_status": status,
        "history_events": history_events,
        "background_state": state,
        "cancel_response": cancel_response,
        "queue": {
            "running": len(queue.get("queue_running") or []),
            "pending": len(queue.get("queue_pending") or []),
        },
        "memory_samples": memory_samples,
        "events": events,
        "checks": checks,
        "passed": all(checks.values()),
        "boundary": (
            "Single low-resolution H3 interruption/release observation. This does not prove "
            "media-quality parity, repeated-run stability, or crash recovery."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8197")
    parser.add_argument("--chain-id", required=True)
    parser.add_argument("--target-node", default="12")
    parser.add_argument("--cancel-at-progress", type=int, default=1)
    parser.add_argument("--expected-release-policy", default="unload_all_models")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--release-observation-samples", type=int, default=8)
    parser.add_argument("--release-observation-interval", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = asyncio.run(run_probe(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
