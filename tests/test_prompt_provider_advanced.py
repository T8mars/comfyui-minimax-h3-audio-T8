from __future__ import annotations

import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import time
from urllib.request import Request

import pytest
import torch

from h3_audio_t8_pkg.nodes_prompt_provider_advanced import (
    MiniMaxH3PromptProviderRouterT8Advanced,
)
from h3_audio_t8_pkg import prompt_provider_advanced as provider


GOOD_OUTPUT = (
    "integrated_multimodal_description: [Shot 1] A woman waves and says "
    "<d>[Chinese] 你好。</d>\n"
    "overall_soundscape: Quiet room tone and soft cloth movement.\n"
    "non_diegetic_music: N/A"
)


def _args(**overrides):
    values = {
        "prompt": "A woman waves and says <d>[Chinese] 你好。</d>",
        "provider_mode": "local_passthrough — 本地原文直通",
        "task": "T2VA — 文生音视频",
        "resolution": "16:9",
        "duration": 10,
        "endpoint": "",
        "provider_model": "test-model",
        "api_key_env": "",
        "confirm_provider_request": False,
        "allow_remote_endpoint": False,
        "max_new_tokens": 1024,
        "temperature": 0.0,
        "top_p": 1.0,
        "maximum_image_edge": 768,
        "jpeg_quality": 85,
        "timeout_seconds": 120.0,
        "maximum_response_bytes": 262144,
        "strict_output_contract": True,
        "ollama_keep_alive": "0",
        "first_frame": None,
        "last_frame": None,
        "contract_repair_attempts": 0,
    }
    values.update(overrides)
    return values


_PROTECTED_SOURCE, _DIALOGUE_BINDINGS = provider._protect_exact_dialogue(_args()["prompt"])
PROTECTED_DIALOGUE = _DIALOGUE_BINDINGS[0][0]
PROTECTED_GOOD_OUTPUT = GOOD_OUTPUT.replace(
    "<d>[Chinese] 你好。</d>", PROTECTED_DIALOGUE
)


def _openai_payload(text=PROTECTED_GOOD_OUTPUT):
    return {"choices": [{"message": {"content": text}}]}


def test_local_passthrough_is_exact_and_never_uses_network(monkeypatch):
    monkeypatch.setattr(
        provider,
        "_open_json",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )
    result = provider.rewrite_prompt_provider(**_args())
    assert result[:4] == (
        _args()["prompt"],
        _args()["prompt"],
        "",
        "",
    )
    report = json.loads(result[-1])
    assert report["network_used"] is False
    assert report["prompt_uploaded"] is False


def test_provider_request_requires_explicit_confirmation_before_endpoint_or_key(monkeypatch):
    called = False

    def fake_open(**_kwargs):
        nonlocal called
        called = True
        return _openai_payload()

    monkeypatch.setattr(provider, "_open_json", fake_open)
    with pytest.raises(ValueError, match="confirm_provider_request"):
        provider.rewrite_prompt_provider(
            **_args(provider_mode="openai_compatible — OpenAI / LM Studio / llama.cpp")
        )
    assert called is False


def test_openai_loopback_request_uses_h3_contract_and_no_key(monkeypatch):
    observed = {}

    def fake_open(**kwargs):
        observed.update(kwargs)
        return _openai_payload()

    monkeypatch.setattr(provider, "_open_json", fake_open)
    result = provider.rewrite_prompt_provider(
        **_args(
            provider_mode="openai_compatible — OpenAI / LM Studio / llama.cpp",
            confirm_provider_request=True,
        )
    )
    request = observed["request"]
    body = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://127.0.0.1:1234/v1/chat/completions"
    assert observed["use_system_proxy"] is False
    assert body["messages"][0]["content"].startswith(
        "You are a professional MiniMax-H3 prompt rewriter"
    )
    assert body["temperature"] == 0.0
    serialized_body = request.data.decode("utf-8")
    assert "<d>[Chinese] 你好。</d>" not in serialized_body
    assert PROTECTED_DIALOGUE in serialized_body
    assert "Authorization" not in request.headers
    assert result[1].startswith("[Shot 1]")
    assert "<d>[Chinese] 你好。</d>" in result[1]
    assert result[2].startswith("Quiet room")
    report = json.loads(result[-1])
    assert report["network_scope"] == "loopback"
    assert report["api_key_serialized"] is False
    assert report["exact_dialogue_text_uploaded"] is False
    assert report["protected_dialogue_tokens"] == 1
    assert report["restored_dialogue_tokens"] == 1
    assert report["provider_release"] == "not_standardized_by_openai_compatible_protocol"


def test_i2va_image_is_downscaled_and_embedded_in_openai_message(monkeypatch):
    observed = {}
    output = (
        "For the target video, at 0.00 seconds into the target video, <Picture 1> "
        "(from [Shot 1]) is fully referenced.\n\n"
        + PROTECTED_GOOD_OUTPUT
    )

    def fake_open(**kwargs):
        observed.update(kwargs)
        return _openai_payload(output)

    monkeypatch.setattr(provider, "_open_json", fake_open)
    frame = torch.zeros((1, 900, 1600, 3), dtype=torch.float32)
    result = provider.rewrite_prompt_provider(
        **_args(
            provider_mode="openai_compatible — OpenAI / LM Studio / llama.cpp",
            task="I2VA — 首帧生音视频",
            confirm_provider_request=True,
            first_frame=frame,
        )
    )
    body = json.loads(observed["request"].data.decode("utf-8"))
    parts = body["messages"][1]["content"]
    images = [item for item in parts if item["type"] == "image_url"]
    assert len(images) == 1
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    report = json.loads(result[-1])
    assert report["reference_images_uploaded"] == 1
    assert report["reference_image_jpeg_bytes"] > 0


def test_ollama_request_asks_for_unload_and_preserves_no_secret(monkeypatch):
    observed = {}

    def fake_open(**kwargs):
        observed.update(kwargs)
        return {"message": {"content": PROTECTED_GOOD_OUTPUT}}

    monkeypatch.setattr(provider, "_open_json", fake_open)
    result = provider.rewrite_prompt_provider(
        **_args(
            provider_mode="ollama_chat — Ollama本地或远程服务",
            confirm_provider_request=True,
        )
    )
    body = json.loads(observed["request"].data.decode("utf-8"))
    assert observed["request"].full_url == "http://127.0.0.1:11434/api/chat"
    assert body["stream"] is False
    assert body["keep_alive"] == "0"
    assert body["options"]["num_predict"] == 1024
    assert json.loads(result[-1])["provider_release"] == "ollama_keep_alive=0"


def test_ollama_thinking_only_response_is_rejected_with_actionable_error():
    with pytest.raises(
        ValueError,
        match=r"message\.thinking.*no final message\.content.*max_new_tokens",
    ):
        provider._response_text(
            "ollama_chat",
            {
                "message": {
                    "content": "",
                    "thinking": "private reasoning must not become the H3 prompt",
                }
            },
        )


def test_empty_ollama_response_keeps_generic_fail_closed_error():
    with pytest.raises(ValueError, match="empty rewritten prompt"):
        provider._response_text("ollama_chat", {"message": {"content": ""}})


def test_contract_repair_is_bounded_text_only_and_restores_dialogue(monkeypatch):
    observed = []
    responses = iter(
        [
            _openai_payload("unstructured candidate without the protected token"),
            _openai_payload(
                "For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
                + PROTECTED_GOOD_OUTPUT
            ),
        ]
    )

    def fake_open(**kwargs):
        observed.append(json.loads(kwargs["request"].data.decode("utf-8")))
        return next(responses)

    monkeypatch.setattr(provider, "_open_json", fake_open)
    frame = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
    result = provider.rewrite_prompt_provider(
        **_args(
            provider_mode="openai_compatible — OpenAI / LM Studio / llama.cpp",
            task="I2VA — 首帧生音视频",
            confirm_provider_request=True,
            first_frame=frame,
            contract_repair_attempts=1,
        )
    )
    assert len(observed) == 2
    first_serialized = json.dumps(observed[0], ensure_ascii=False)
    repair_serialized = json.dumps(observed[1], ensure_ascii=False)
    assert "data:image/jpeg;base64," in first_serialized
    assert "data:image/jpeg;base64," not in repair_serialized
    assert "<d>[Chinese] 你好。</d>" not in repair_serialized
    assert PROTECTED_DIALOGUE in repair_serialized
    assert "<d>[Chinese] 你好。</d>" in result[0]
    report = json.loads(result[-1])
    assert report["provider_request_count"] == 2
    assert report["contract_repair_attempts_used"] == 1
    assert report["contract_repair_succeeded"] is True
    assert report["repair_reference_images_reuploaded"] is False
    assert len(report["request_sha256s"]) == 2


def test_contract_repair_exhaustion_stays_fail_closed(monkeypatch):
    calls = 0

    def fake_open(**_kwargs):
        nonlocal calls
        calls += 1
        return _openai_payload("still invalid")

    monkeypatch.setattr(provider, "_open_json", fake_open)
    with pytest.raises(ValueError, match=r"after 3 provider response\(s\)"):
        provider.rewrite_prompt_provider(
            **_args(
                provider_mode="openai_compatible — OpenAI / LM Studio / llama.cpp",
                confirm_provider_request=True,
                contract_repair_attempts=2,
            )
        )
    assert calls == 3


def test_real_loopback_openai_protocol_roundtrip():
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            received["path"] = self.path
            received["body"] = json.loads(self.rfile.read(length).decode("utf-8"))
            payload = json.dumps(_openai_payload()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = provider.rewrite_prompt_provider(
            **_args(
                provider_mode="openai_compatible — OpenAI / LM Studio / llama.cpp",
                confirm_provider_request=True,
                endpoint=(
                    f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
                ),
                timeout_seconds=5.0,
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert received["path"] == "/v1/chat/completions"
    assert received["body"]["model"] == "test-model"
    assert result[1].startswith("[Shot 1]")
    assert json.loads(result[-1])["network_scope"] == "loopback"


@pytest.mark.parametrize(
    "output",
    [
        GOOD_OUTPUT.replace("<d>[Chinese] 你好。</d>", "dialogue omitted"),
        PROTECTED_GOOD_OUTPUT.replace(
            PROTECTED_DIALOGUE, PROTECTED_DIALOGUE + " " + PROTECTED_DIALOGUE
        ),
        (
            "integrated_multimodal_description: [Shot 1] A woman waves.\n"
            f"overall_soundscape: {PROTECTED_DIALOGUE}\n"
            "non_diegetic_music: N/A"
        ),
    ],
)
def test_dialogue_guard_rejects_missing_duplicate_or_misplaced_token(monkeypatch, output):
    monkeypatch.setattr(provider, "_open_json", lambda **_kwargs: _openai_payload(output))
    with pytest.raises(ValueError, match="output contract failed"):
        provider.rewrite_prompt_provider(
            **_args(
                provider_mode="openai_compatible — OpenAI / LM Studio / llama.cpp",
                confirm_provider_request=True,
            )
        )


def test_real_loopback_stalled_headers_returns_bounded_timeout():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            time.sleep(0.3)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        with pytest.raises(ValueError, match="timed out after 0.1 seconds"):
            provider._open_json(
                request=Request(
                    f"http://127.0.0.1:{server.server_address[1]}/slow",
                    data=b"{}",
                ),
                timeout_seconds=0.1,
                maximum_response_bytes=4096,
                use_system_proxy=False,
            )
    finally:
        elapsed = time.perf_counter() - started
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert elapsed < 0.5


def test_real_loopback_streaming_read_observes_interruption_between_chunks(monkeypatch):
    payload = json.dumps(_openai_payload()).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload[:16])
                self.wfile.flush()
                time.sleep(0.4)
                self.wfile.write(payload[16:])
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, _format, *_args):
            return

    class TestInterrupted(RuntimeError):
        pass

    checks = 0

    def interrupt_after_first_chunk():
        nonlocal checks
        checks += 1
        if checks >= 3:
            raise TestInterrupted("cancelled during streamed response")

    monkeypatch.setattr(provider, "_check_interrupted", interrupt_after_first_chunk)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        with pytest.raises(TestInterrupted, match="cancelled during streamed response"):
            provider._open_json(
                request=Request(
                    f"http://127.0.0.1:{server.server_address[1]}/stream",
                    data=b"{}",
                ),
                timeout_seconds=3.0,
                maximum_response_bytes=4096,
                use_system_proxy=False,
            )
        elapsed = time.perf_counter() - started
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert checks == 3
    # Keep the gate below the server's 0.4-second body stall while allowing a
    # small Windows scheduler margin on an otherwise busy ComfyUI host.
    assert elapsed < 0.35


def test_remote_endpoint_requires_https_permission_and_environment_key(monkeypatch):
    mode = "openai_compatible — OpenAI / LM Studio / llama.cpp"
    with pytest.raises(ValueError, match="allow_remote_endpoint"):
        provider.rewrite_prompt_provider(
            **_args(
                provider_mode=mode,
                confirm_provider_request=True,
                endpoint="https://provider.example/v1/chat/completions",
            )
        )
    with pytest.raises(ValueError, match="require HTTPS"):
        provider.rewrite_prompt_provider(
            **_args(
                provider_mode=mode,
                confirm_provider_request=True,
                allow_remote_endpoint=True,
                endpoint="http://provider.example/v1/chat/completions",
            )
        )
    with pytest.raises(ValueError, match="requires api_key_env"):
        provider.rewrite_prompt_provider(
            **_args(
                provider_mode=mode,
                confirm_provider_request=True,
                allow_remote_endpoint=True,
                endpoint="https://provider.example/v1/chat/completions",
            )
        )
    monkeypatch.setenv("T8_PROVIDER_TEST_KEY", "do-not-serialize")
    monkeypatch.setattr(provider, "_open_json", lambda **_kwargs: _openai_payload())
    result = provider.rewrite_prompt_provider(
        **_args(
            provider_mode=mode,
            confirm_provider_request=True,
            allow_remote_endpoint=True,
            endpoint="https://provider.example/v1/chat/completions",
            api_key_env="T8_PROVIDER_TEST_KEY",
        )
    )
    assert "do-not-serialize" not in result[-1]
    assert json.loads(result[-1])["network_scope"] == "remote_https"


def test_bounded_json_reader_checks_interruptions_and_disables_redirects(monkeypatch):
    raw = json.dumps({"message": {"content": GOOD_OUTPUT}}).encode("utf-8")
    interrupt_checks = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, amount):
            nonlocal raw
            value, raw = raw[:amount], raw[amount:]
            return value

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "http://127.0.0.1:11434/api/chat"
            assert timeout == 5.0
            return Response()

    handlers = []

    def fake_build(*values):
        handlers.extend(values)
        return Opener()

    monkeypatch.setattr(provider, "build_opener", fake_build)
    monkeypatch.setattr(provider, "_check_interrupted", lambda: interrupt_checks.append(True))
    result = provider._open_json(
        request=Request("http://127.0.0.1:11434/api/chat", data=b"{}"),
        timeout_seconds=5.0,
        maximum_response_bytes=4096,
        use_system_proxy=False,
    )
    assert result["message"]["content"] == GOOD_OUTPUT
    assert len(interrupt_checks) >= 3
    assert any(isinstance(item, provider._NoRedirects) for item in handlers)
    assert any(item.__class__.__name__ == "ProxyHandler" for item in handlers)


def test_bounded_json_reader_rejects_oversized_response(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, amount):
            return b"x" * amount

    class Opener:
        def open(self, request, timeout):  # noqa: ARG002
            return Response()

    monkeypatch.setattr(provider, "build_opener", lambda *_handlers: Opener())
    monkeypatch.setattr(provider, "_check_interrupted", lambda: None)
    with pytest.raises(ValueError, match="exceeds maximum_response_bytes"):
        provider._open_json(
            request=Request("http://127.0.0.1:11434/api/chat", data=b"{}"),
            timeout_seconds=5.0,
            maximum_response_bytes=4096,
            use_system_proxy=False,
        )


def test_redirect_handler_refuses_all_redirects():
    assert (
        provider._NoRedirects().redirect_request(
            None, None, 302, "Found", {}, "https://elsewhere.example"
        )
        is None
    )


@pytest.mark.parametrize(
    ("task", "first_frame", "output"),
    [
        ("T2VA — 文生音视频", None, "an unstructured response"),
        (
            "T2VA — 文生音视频",
            None,
            GOOD_OUTPUT.replace("<d>[Chinese] 你好。</d>", "你好。"),
        ),
        (
            "T2VA — 文生音视频",
            None,
            GOOD_OUTPUT.replace("[Shot 1]", "<Picture 9> [Shot 1]"),
        ),
        ("I2VA — 首帧生音视频", torch.zeros((1, 8, 8, 3)), GOOD_OUTPUT),
    ],
)
def test_strict_output_contract_rejects_malformed_or_changed_output(
    monkeypatch, task, first_frame, output
):
    monkeypatch.setattr(provider, "_open_json", lambda **_kwargs: _openai_payload(output))
    with pytest.raises(ValueError, match="output contract failed"):
        provider.rewrite_prompt_provider(
            **_args(
                provider_mode="openai_compatible — OpenAI / LM Studio / llama.cpp",
                confirm_provider_request=True,
                task=task,
                first_frame=first_frame,
            )
        )


def test_provider_node_schema_is_safe_and_workflow_is_documented():
    schema = MiniMaxH3PromptProviderRouterT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.is_experimental is True
    assert inputs["provider_mode"].default.startswith("local_passthrough")
    assert inputs["confirm_provider_request"].default is False
    assert inputs["allow_remote_endpoint"].default is False
    assert inputs["strict_output_contract"].default is True
    assert inputs["ollama_keep_alive"].default == "0"
    assert inputs["contract_repair_attempts"].default == 0
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "14-prompt-relay"
        / "2026-08-23_H3_Prompt_Provider_Router_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    router = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3PromptProviderRouterT8Advanced"
    )
    assert router["widgets_values"][1].startswith("local_passthrough")
    assert router["widgets_values"][8:10] == [False, False]
    assert router["widgets_values"][17:] == [True, "0", 0]
    assert "MarkdownNote" in {node["type"] for node in nodes.values()}
