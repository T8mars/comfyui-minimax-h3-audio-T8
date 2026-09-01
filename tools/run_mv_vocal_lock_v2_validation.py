from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import uuid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW = ROOT / "tests/fixtures/api/mv_vocal_lock_v2_clear_speech_api.json"


def _json_request(method: str, url: str, payload: dict | None = None, timeout: int = 30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"local ComfyUI request failed: {method} {url}: {error}") from error


def _local_server(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("validation server must be local http://127.0.0.1 or localhost")
    return value.rstrip("/")


def _wait_history(server: str, prompt_id: str, timeout_seconds: float) -> dict:
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        history = _json_request("GET", f"{server}/history/{prompt_id}", timeout=30)
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2.0)
    raise TimeoutError(f"prompt {prompt_id} did not finish within {timeout_seconds:g}s")


def _execution_error(record: dict) -> str:
    messages = ((record.get("status") or {}).get("messages") or [])
    for event, payload in messages:
        if event == "execution_error":
            return json.dumps(payload, ensure_ascii=False)
    return ""


def _renderer_output(record: dict) -> dict:
    outputs = record.get("outputs") or {}
    value = outputs.get("11") or {}
    return value if isinstance(value, dict) else {"raw": value}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Submit exactly one local ComfyUI prompt for the MV Vocal Lock V2 clear-speech "
            "validation fixture. This is an external test harness; the product node itself "
            "never calls /prompt."
        )
    )
    parser.add_argument("--server", default="http://127.0.0.1:8191")
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--chain-id", default="")
    args = parser.parse_args()

    server = _local_server(args.server)
    workflow = json.loads(args.workflow.resolve().read_text(encoding="utf-8"))
    chain_id = args.chain_id.strip() or (
        "mv_vocal_lock_v2_clear_speech_" + time.strftime("%Y%m%d_%H%M%S")
    )
    workflow["11"]["inputs"]["chain_id"] = chain_id
    required_nodes = (
        "MiniMaxH3MVVocalLockScenePlannerV2T8Advanced",
        "MiniMaxH3MVVocalLockPromptCompilerV2T8Advanced",
        "MiniMaxH3LocalMVVocalLockRendererV2T8Advanced",
    )
    schemas = {}
    for node_id in required_nodes:
        payload = _json_request("GET", f"{server}/object_info/{node_id}", timeout=30)
        if node_id not in payload:
            raise RuntimeError(f"local ComfyUI did not register required node {node_id}")
        schemas[node_id] = payload[node_id]

    requested_prompt_id = str(uuid.uuid4())
    started = time.time()
    queued = _json_request(
        "POST",
        f"{server}/prompt",
        {
            "prompt": workflow,
            "client_id": "h3-t8-mv-vocal-lock-v2-validation",
            "prompt_id": requested_prompt_id,
        },
        timeout=60,
    )
    prompt_id = str(queued.get("prompt_id") or "")
    if prompt_id != requested_prompt_id:
        raise RuntimeError(
            f"local ComfyUI returned prompt_id={prompt_id!r}, expected {requested_prompt_id!r}"
        )
    record = _wait_history(server, prompt_id, args.timeout_seconds)
    elapsed = time.time() - started
    error = _execution_error(record)
    renderer = _renderer_output(record)
    report = {
        "schema": "t8.minimax_h3.mv_vocal_lock_v2_real_validation.v1",
        "status": "failed" if error else "completed_waiting_media_audit",
        "server": server,
        "workflow": str(args.workflow.resolve()),
        "chain_id": chain_id,
        "prompt_id": prompt_id,
        "prompts_submitted": 1,
        "elapsed_seconds": elapsed,
        "required_node_contracts_loaded": list(schemas),
        "external_api_used": False,
        "node_internal_prompt_http_used": False,
        "execution_error": error,
        "renderer_output": renderer,
        "history_status": record.get("status"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
