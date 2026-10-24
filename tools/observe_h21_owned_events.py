"""Read-only WS observer of one explicitly frozen H21-owned Core process.

Never POST, start/stop a process, submit a graph, load a model or touch a user
browser. Ownership is proved from the listening PID/executable/exact argv.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import time

import psutil
import websocket


TERMINAL_EVENTS = {"execution_success", "execution_error", "execution_interrupted"}


def _invalid_json_constant(token):
    raise ValueError("Non-finite JSON constant on WebSocket: " + token)


def summarize_case_events(
    events, prompt_id, required_nodes, history, *, connected_before_post
):
    """Evidence and generation are separate; counters are not MODEL.forward NFE."""
    required = set(map(str, required_nodes))
    starts, terminals, seen, cached, progress = [], [], set(), set(), {}
    for event in events:
        data = event.get("data") or {}
        if not isinstance(data, dict) or data.get("prompt_id") != prompt_id:
            continue
        kind = event.get("type")
        if kind == "execution_start":
            starts.append(data)
        elif kind in TERMINAL_EVENTS:
            terminals.append(kind)
        elif kind == "execution_cached":
            cached.update(map(str, data.get("nodes", [])))
        elif kind == "executing" and data.get("node") is not None:
            seen.add(str(data["node"]))
        elif kind == "progress" and data.get("node") is not None:
            node = str(data["node"])
            progress.setdefault(node, []).append(
                {"value": data.get("value"), "max": data.get("max")}
            )
    missing = required - seen
    matched = connected_before_post and len(starts) == 1 and len(terminals) == 1
    status = (history or {}).get("status", {})
    return {
        "schema": "t8.h21.case_event_evidence.v1",
        "prompt_id": prompt_id,
        "generation_status": status.get(
            "status_str", "unknown_no_authoritative_history"
        ),
        "history_completed": status.get("completed"),
        "event_evidence_status": "complete_start_terminal" if matched else "incomplete",
        "connected_before_POST": connected_before_post,
        "terminal_events": terminals,
        "execution_start_events": starts,
        "required_dependency_nodes": sorted(required),
        "executed_nodes": sorted(seen),
        "cached_nodes": sorted(cached & required),
        "missing_execution_nodes": sorted(missing),
        "unrelated_cached_nodes": sorted(cached - required),
        "unrelated_executed_nodes": sorted(seen - required),
        "uncached_dependency_coverage": bool(
            matched and not missing and not cached & required
        ),
        "observed_progress": progress,
        "NFE_inferred": False,
        "progress_scope": "Raw Core sampler callbacks only; absent/cached counters are insufficient evidence, not failed generation",
        "media_quality_qualified": False,
    }


class OwnedEventCapture:
    """One case, one pre-POST owned WebSocket. Never queue, reconnect or kill.

    The controller alone submits and polls history. Read errors after submission
    preserve incomplete evidence; no observer failure may silently rerun a graph.
    """

    def __init__(
        self,
        manifest_path,
        plan,
        output,
        *,
        client_id,
        expected_pid,
        handshake_timeout=10,
        owner_reader=None,
    ):
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        if self.manifest_path != Path(plan["run_root"]).resolve() / "manifest.json":
            raise ValueError("Explicit contained H21 manifest required")
        if plan.get("schema") != "t8.h21.control_probe.v1":
            raise ValueError("Expected the frozen H21 controller manifest")
        if (
            not isinstance(client_id, str)
            or not client_id
            or not client_id.replace("-", "").isalnum()
        ):
            raise ValueError("Explicit ordinary case-specific client ID required")
        if type(expected_pid) is not int or expected_pid <= 0:
            raise ValueError("Exact controller-owned PID required")
        if type(handshake_timeout) not in (
            int,
            float,
        ) or not 0 < handshake_timeout < float("inf"):
            raise ValueError("Positive finite handshake deadline required")
        self.plan, self.client_id, self.expected_pid = plan, client_id, expected_pid
        self.output = Path(output).resolve()
        allowed = (
            Path(plan["run_root"]).resolve() / "attempts",
            Path(__file__).resolve().parents[1] / "artifacts",
        )
        if (
            not any(self.output.is_relative_to(root) for root in allowed)
            or self.output.exists()
        ):
            raise ValueError(
                "Use a new contained attempt or private artifact observation directory"
            )
        self.output.mkdir(parents=True)
        self.handshake_timeout, self.owner_reader = (
            handshake_timeout,
            owner_reader or listener_owner,
        )
        self.started, self.events, self.errors = time.monotonic(), [], []
        self.sock, self.owner, self.prompt_id = None, None, None
        self.connected_at, self.post_started_at, self.finished = None, None, False
        self.stream = (self.output / "events.jsonl").open("x", encoding="utf-8")
        write_new(
            self.output / "started.json",
            {
                "manifest": str(self.manifest_path),
                "run_id": plan["run_id"],
                "manifest_sha256": hashlib.sha256(
                    self.manifest_path.read_bytes()
                ).hexdigest(),
                "client_id": client_id,
                "expected_pid": expected_pid,
                "read_only": True,
                "reconnect_or_resubmit": False,
            },
        )

    def _owner(self):
        owner = self.owner_reader(self.manifest_path, self.plan)
        if owner is None or owner["pid"] != self.expected_pid:
            raise RuntimeError(
                "WebSocket listener is not the exact controller-owned PID"
            )
        if self.owner is not None and owner != self.owner:
            raise RuntimeError("Owned listener identity changed across observation")
        return owner

    def _record(self, raw):
        row = {
            "elapsed_seconds": time.monotonic() - self.started,
            "received_at": time.time(),
        }
        if isinstance(raw, str):
            row["raw_json"] = raw
            try:
                packet = json.loads(raw, parse_constant=_invalid_json_constant)
                if not isinstance(packet, dict):
                    raise ValueError("WS returned a non-object JSON packet")
                # Retain exact wire JSON independently; timestamps cannot be
                # overwritten by untrusted packet keys.
                row.update({"type": packet.get("type"), "data": packet.get("data")})
            except (TypeError, ValueError) as error:
                row.update(
                    type="invalid_json_observation", observation_error=str(error)
                )
                self.errors.append(str(error))
        else:
            row.update(
                type="binary_observation",
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                raw_base64=base64.b64encode(raw).decode("ascii"),
            )
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        self.events.append(row)
        return row

    def connect(self):
        """Require Core status.sid ACK before the controller's only POST."""
        try:
            self.owner = self._owner()
            self.sock = websocket.create_connection(
                f"ws://127.0.0.1:{self.plan['port']}/ws?clientId={self.client_id}",
                timeout=0.25,
                http_no_proxy=["127.0.0.1", "localhost"],
                http_proxy_host=None,
            )
            self._owner()
            deadline = time.monotonic() + self.handshake_timeout
            while time.monotonic() < deadline:
                row = self.receive()
                if row is None:
                    if self.sock is None:
                        raise RuntimeError(
                            "Observer disconnected before Core handshake ACK"
                        )
                    continue
                data = row.get("data") or {}
                if (
                    row.get("type") == "status"
                    and isinstance(data, dict)
                    and data.get("sid") == self.client_id
                ):
                    self._owner()
                    self.connected_at = time.time()
                    write_new(
                        self.output / "connected.json",
                        {
                            "connected_at": self.connected_at,
                            "client_id": self.client_id,
                            "owner": self.owner,
                            "Core_status_sid_ack": True,
                            "scope": "This exact case/attempt; older attempts do not affect pre-POST order",
                        },
                    )
                    return
            raise TimeoutError("No matching Core status.sid handshake ACK before POST")
        except Exception as error:
            self.errors.append(f"handshake:{type(error).__name__}:{error}")
            self.close_socket()
            raise

    def mark_post(self):
        if self.connected_at is None or self.post_started_at is not None:
            raise RuntimeError("A single POST requires a completed pre-POST handshake")
        self._owner()
        self.post_started_at = time.time()
        write_new(
            self.output / "pre-post.json",
            {
                "client_id": self.client_id,
                "connected_at": self.connected_at,
                "post_started_at": self.post_started_at,
                "owner": self.owner,
                "Core_status_sid_ack": True,
            },
        )

    def receive(self):
        if self.sock is None:
            return None
        try:
            raw = self.sock.recv()
            if raw in (None, "", b""):
                raise ConnectionError("Owned WebSocket closed")
            return self._record(raw)
        except websocket.WebSocketTimeoutException:
            return None
        except Exception as error:
            self.errors.append(f"receive:{type(error).__name__}:{error}")
            self.close_socket()
            return None

    def terminal_for(self, prompt_id):
        return any(
            row.get("type") in TERMINAL_EVENTS
            and isinstance(row.get("data"), dict)
            and row["data"].get("prompt_id") == prompt_id
            for row in self.events
        )

    def finish(
        self, prompt_id, required_nodes, history=None, error=None, sampler_nodes=None
    ):
        if self.finished:
            raise RuntimeError("Observation evidence may only be finalized once")
        self.finished = True
        try:
            final_owner = self._owner()
        except Exception as owner_error:
            final_owner = None
            self.errors.append(
                f"final_owner:{type(owner_error).__name__}:{owner_error}"
            )
        result = summarize_case_events(
            self.events,
            prompt_id,
            required_nodes,
            history,
            connected_before_post=self.connected_at is not None
            and self.post_started_at is not None
            and self.connected_at <= self.post_started_at,
        )
        result.update(
            owner=self.owner,
            final_owner=final_owner,
            client_id=self.client_id,
            observer_errors=self.errors,
            controller_error=error,
            events=len(self.events),
            read_only=True,
            GPU_model_work_started_by_observer=False,
        )
        samplers = sampler_nodes or {}
        missing_progress = sorted(set(samplers) - set(result["observed_progress"]))
        result.update(
            sampler_nodes=samplers,
            sampler_nodes_without_progress=missing_progress,
            progress_evidence_status="missing_or_cached"
            if missing_progress
            else "raw_callbacks_present_not_MODEL_forward_NFE"
            if samplers
            else "no_sampler_contract",
        )
        if self.errors:
            result["event_evidence_status"] = "incomplete"
            result["uncached_dependency_coverage"] = False
        write_new(self.output / "terminal.json", result)
        self.close_socket()
        self.stream.close()
        return result

    def close_socket(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def process_contract(manifest_path, plan, pid):
    process = psutil.Process(pid)
    expected_python = Path(plan["python"]).resolve(strict=True)
    script = Path(plan["run_root"]).resolve() / "runtime_tools/h21_control_probe.py"
    command = process.cmdline()
    if Path(process.exe()).resolve(strict=True) != expected_python:
        raise RuntimeError("Listening process is not the manifest Python executable")
    if "--_serve" not in command:
        raise RuntimeError("Listening process is not the exact owned frozen server")
    position = command.index("--_serve")
    if (
        position + 1 >= len(command)
        or Path(command[position + 1]).resolve() != Path(manifest_path).resolve()
        or not any(
            Path(arg).resolve() == script
            for arg in command[1:position]
            if not arg.startswith("-")
        )
    ):
        raise RuntimeError(
            "Listening process argv does not bind the exact frozen script and manifest"
        )
    return {
        "pid": pid,
        "create_time": process.create_time(),
        "exe": str(expected_python),
        "argv": command,
    }


def listener_owner(manifest_path, plan):
    port = plan["port"]
    if type(port) is not int or not 1024 <= port <= 65535 or port in {8189, 8878, 8991}:
        raise ValueError("Only an explicit separate owned loopback port is allowed")
    rows = [
        row
        for row in psutil.net_connections(kind="tcp")
        if row.status == psutil.CONN_LISTEN and row.laddr.port == port
    ]
    if not rows:
        return None
    if (
        any(row.laddr.ip != "127.0.0.1" or row.pid is None for row in rows)
        or len({row.pid for row in rows}) != 1
    ):
        raise RuntimeError("Port listener is ambiguous or not owned loopback-only")
    return process_contract(manifest_path, plan, rows[0].pid)


def observe(manifest_path, output, *, wait_seconds=1800, timeout_seconds=7200):
    manifest_path = Path(manifest_path).resolve(strict=True)
    plan = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        plan.get("schema") != "t8.h21.control_probe.v1"
        or manifest_path != Path(plan["run_root"]).resolve() / "manifest.json"
    ):
        raise ValueError("Explicit contained H21 manifest required")
    output = Path(output).resolve()
    private = Path(__file__).resolve().parents[1] / "artifacts"
    if not output.is_relative_to(private) or output.exists():
        raise ValueError("Use a new private artifact observer directory")
    if wait_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("Positive observer deadlines required")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_new(
        output / "started.json",
        {
            "manifest": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "run_id": plan["run_id"],
            "port": plan["port"],
            "observer_pid": psutil.Process().pid,
            "started_at": time.time(),
            "read_only": True,
        },
    )
    owner = None
    while time.monotonic() - started < wait_seconds:
        owner = listener_owner(manifest_path, plan)
        if owner is not None:
            break
        time.sleep(0.25)
    if owner is None:
        raise TimeoutError(
            "Owned server listener did not appear; no connection attempted"
        )
    client = "h21-control-" + plan["run_id"]
    root = Path(plan["run_root"])
    before = [str(path) for path in root.glob("attempts/*/live-runtime.json")]
    submitted_before = [
        str(path) for path in root.glob("attempts/*/executions/*/submitted.json")
    ]
    sock = websocket.create_connection(
        f"ws://127.0.0.1:{plan['port']}/ws?clientId={client}",
        timeout=0.5,
        http_no_proxy=["127.0.0.1", "localhost"],
        http_proxy_host=None,
    )
    connected_at = time.time()
    if listener_owner(manifest_path, plan) != owner:
        sock.close()
        raise RuntimeError("Owned listener identity changed across WS handshake")
    write_new(
        output / "connected.json",
        {
            "connected_at": connected_at,
            "client_id": client,
            "owner": owner,
            "live_runtime_files_existing_before_connect": before,
            "submitted_files_existing_before_connect": submitted_before,
        },
    )
    terminal, prompt, starts, progress, event_count = None, None, [], [], 0
    deadline = time.monotonic() + timeout_seconds
    try:
        with (output / "events.jsonl").open("x", encoding="utf-8") as stream:
            while time.monotonic() < deadline:
                try:
                    raw = sock.recv()
                except websocket.WebSocketTimeoutException:
                    if not psutil.pid_exists(owner["pid"]):
                        break
                    continue
                if raw in (None, ""):
                    break
                if not isinstance(raw, str):
                    event = {
                        "type": "binary_observation",
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                else:
                    event = json.loads(raw)
                    if not isinstance(event, dict):
                        raise ValueError("WS returned a non-object JSON packet")
                    event["raw_json"] = raw
                event = {
                    "elapsed_seconds": time.monotonic() - started,
                    "received_at": time.time(),
                    **event,
                }
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                stream.flush()
                event_count += 1
                data = event.get("data") or {}
                if event.get("type") == "execution_start":
                    starts.append(data)
                if event.get("type") == "progress":
                    progress.append(data)
                if event.get("type") in {
                    "execution_success",
                    "execution_error",
                    "execution_interrupted",
                }:
                    terminal, prompt = event["type"], data.get("prompt_id")
                    break
    finally:
        sock.close()
        later = [
            path
            for path in root.glob("attempts/*/live-runtime.json")
            if path.stat().st_mtime >= connected_at
        ]
        coverage = (
            not before
            and not submitted_before
            and bool(later)
            and terminal is not None
            and any(row.get("prompt_id") == prompt for row in starts)
        )
        result = {
            "status": "terminal_observed" if terminal else "incomplete_no_terminal",
            "owner": owner,
            "client_id": client,
            "event_count": event_count,
            "terminal": terminal,
            "prompt_id": prompt,
            "execution_start_events": starts,
            "progress_events": progress,
            "pre_POST_order_supported": not before
            and not submitted_before
            and bool(later),
            "full_execution_event_coverage_supported": coverage,
            "coverage_basis": "WS connected before controller live-runtime.json creation; controller writes that receipt before POST. Requires execution_start plus terminal for same prompt. No missing progress invented.",
            "later_live_runtime_files": [str(path) for path in later],
            "NFE": "NOT inferred; raw progress value/max/node retained only",
            "read_only": True,
            "GPU_model_work_started_by_observer": False,
        }
        write_new(output / "terminal.json", result)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "terminal": terminal,
                    "events": event_count,
                    "coverage_supported": coverage,
                }
            ),
            flush=True,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait-seconds", type=int, default=1800)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    observe(
        args.manifest,
        args.output,
        wait_seconds=args.wait_seconds,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
