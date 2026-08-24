#!/usr/bin/env python3
"""Run one Creator background reattach/auto-accept completion probe.

The workflow is expected to contain the real Candidate Save and Auto Accept terminal. This tool
submits it once, observes automatically queued prompts, and verifies the final accepted manifest
and composed media. It is a functional probe, not a repeated-load harness.
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gpu_memory_mib() -> dict:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
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
            raise RuntimeError(f"{method} {url} returned non-object JSON")
        return payload


def _probe_media(path: Path) -> dict:
    if not path.is_file():
        return {"exists": False, "path": str(path)}
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,width,height,sample_rate,channels",
                "-show_entries",
                "format=duration,size",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        return {"exists": True, "path": str(path), "probe_error": str(error)}
    return {"exists": True, "path": str(path), **payload}


async def run_probe(args) -> dict:
    try:
        import aiohttp
    except ImportError as error:
        raise RuntimeError("aiohttp is required in the ComfyUI Python environment") from error

    prompt = json.loads(args.workflow.read_text(encoding="utf-8"))
    if not isinstance(prompt, dict) or not prompt:
        raise ValueError("workflow must contain a non-empty ComfyUI API prompt object")
    base = args.server.rstrip("/")
    chain_url = quote(args.chain_id, safe="")
    client_id = uuid.uuid4().hex
    prompt_id = str(uuid.uuid4())
    baseline_gpu = _gpu_memory_mib()
    started = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_connect=20, sock_read=None)
    timeline = []
    memory_samples = []
    observed_prompt_ids = {prompt_id}

    async with aiohttp.ClientSession(timeout=timeout) as session:
        await _json_request(session, "GET", f"{base}/system_stats")
        previous = await _json_request(
            session,
            "GET",
            f"{base}/minimax_h3_t8/long_video/background/{chain_url}",
        )
        previous_job_id = str(previous.get("job_id") or "")
        submitted = await _json_request(
            session,
            "POST",
            f"{base}/prompt",
            json={"prompt": prompt, "client_id": client_id, "prompt_id": prompt_id},
        )
        if str(submitted.get("prompt_id") or "") != prompt_id:
            raise RuntimeError("ComfyUI returned an unexpected prompt ID")

        deadline = time.monotonic() + args.timeout_seconds
        seen_new_job = False
        final_state = None
        final_queue = None
        previous_signature = None
        while time.monotonic() < deadline:
            state = await _json_request(
                session,
                "GET",
                f"{base}/minimax_h3_t8/long_video/background/{chain_url}",
            )
            queue = await _json_request(session, "GET", f"{base}/queue")
            active_prompt_id = str(state.get("active_prompt_id") or "")
            if active_prompt_id:
                observed_prompt_ids.add(active_prompt_id)
            accepted_prompt_id = str(state.get("last_accepted_prompt_id") or "")
            if accepted_prompt_id:
                observed_prompt_ids.add(accepted_prompt_id)
            current_job_id = str(state.get("job_id") or "")
            if current_job_id and current_job_id != previous_job_id:
                seen_new_job = True
            signature = (
                current_job_id,
                state.get("state"),
                state.get("accepted_count"),
                state.get("current_segment_index"),
                active_prompt_id,
            )
            if signature != previous_signature:
                timeline.append(
                    {
                        "elapsed_seconds": round(time.monotonic() - started, 4),
                        "job_id": current_job_id,
                        "state": state.get("state"),
                        "accepted_count": state.get("accepted_count"),
                        "current_segment_index": state.get("current_segment_index"),
                        "active_prompt_id": active_prompt_id,
                    }
                )
                previous_signature = signature
            memory_samples.append(
                {
                    "elapsed_seconds": round(time.monotonic() - started, 4),
                    "gpu": _gpu_memory_mib(),
                    "state": state.get("state"),
                }
            )
            running = queue.get("queue_running") or []
            pending = queue.get("queue_pending") or []
            if seen_new_job and state.get("state") == "completed" and not running and not pending:
                final_state = state
                final_queue = queue
                break
            if seen_new_job and state.get("state") in {"failed", "cancelled", "detached"}:
                final_state = state
                final_queue = queue
                break
            await asyncio.sleep(args.poll_interval)
        if final_state is None:
            raise TimeoutError("Timed out waiting for Creator background completion")

        # Observe the selected release policy after the final accepted boundary.
        for _ in range(args.release_observation_samples):
            await asyncio.sleep(args.release_observation_interval)
            memory_samples.append(
                {
                    "elapsed_seconds": round(time.monotonic() - started, 4),
                    "gpu": _gpu_memory_mib(),
                    "state": final_state.get("state"),
                }
            )
        histories = {}
        for observed_id in sorted(observed_prompt_ids):
            response = await _json_request(session, "GET", f"{base}/history/{observed_id}")
            record = response.get(observed_id)
            if isinstance(record, dict):
                status = record.get("status") if isinstance(record.get("status"), dict) else {}
                histories[observed_id] = {
                    "status_str": status.get("status_str"),
                    "completed": status.get("completed"),
                    "events": [
                        str(item[0])
                        for item in status.get("messages", [])
                        if isinstance(item, list) and item
                    ],
                }

    manifest_path = Path(str(final_state.get("last_manifest_path") or ""))
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_video_path = Path(str(final_state.get("final_video_path") or ""))
    media = _probe_media(final_video_path)
    streams = media.get("streams") if isinstance(media.get("streams"), list) else []
    stream_types = {str(item.get("codec_type")) for item in streams if isinstance(item, dict)}
    history_success = bool(histories) and all(
        item.get("status_str") == "success" and item.get("completed") is True
        for item in histories.values()
    )
    final_gpu = memory_samples[-1]["gpu"] if memory_samples else _gpu_memory_mib()
    final_delta = None
    if baseline_gpu.get("available") and final_gpu.get("available"):
        final_delta = int(final_gpu["used_mib"]) - int(baseline_gpu["used_mib"])
    manifest_segments = manifest.get("segments") if isinstance(manifest, dict) else None
    checks = {
        "new_job_attached_after_previous_state": seen_new_job,
        "background_completed": final_state.get("state") == "completed",
        "accepted_segment_count": int(final_state.get("accepted_count", -1))
        == args.expected_segments,
        "manifest_complete": bool(final_state.get("manifest_complete")),
        "manifest_segment_count": isinstance(manifest_segments, list)
        and len(manifest_segments) == args.expected_segments,
        "release_policy_recorded": final_state.get("last_release_policy")
        == args.expected_release_policy,
        "release_error_absent": not bool(final_state.get("last_release_error")),
        "queue_empty": not (
            (final_queue or {}).get("queue_running") or (final_queue or {}).get("queue_pending")
        ),
        "all_observed_histories_success": history_success,
        "final_media_exists": bool(media.get("exists")),
        "final_media_probe_succeeded": not bool(media.get("probe_error")),
        "final_media_has_video_and_audio": {"video", "audio"}.issubset(stream_types),
    }
    return {
        "schema": "t8.creator_background_h3_resume_completion_probe.v1",
        "created_at": _utc_now(),
        "workflow": str(args.workflow.resolve()),
        "server": base,
        "chain_id": args.chain_id,
        "initial_prompt_id": prompt_id,
        "previous_state": previous,
        "final_state": final_state,
        "timeline": timeline,
        "histories": histories,
        "manifest": manifest,
        "media": media,
        "baseline_gpu": baseline_gpu,
        "final_gpu": final_gpu,
        "final_minus_baseline_used_mib": final_delta,
        "maximum_observed_used_mib": max(
            (
                int(item["gpu"]["used_mib"])
                for item in memory_samples
                if item["gpu"].get("available")
            ),
            default=None,
        ),
        "memory_samples": memory_samples,
        "checks": checks,
        "passed": all(checks.values()),
        "boundary": (
            "One low-resolution cancel-then-reattach, Candidate Save, Auto Accept and final "
            "composition path. This is not repeated stability or a general quality/VRAM claim."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8197")
    parser.add_argument("--chain-id", required=True)
    parser.add_argument("--expected-segments", type=int, default=2)
    parser.add_argument("--expected-release-policy", default="unload_all_models")
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--release-observation-samples", type=int, default=6)
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
