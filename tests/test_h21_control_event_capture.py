"""Owned observation: real Core schemas and ephemeral CPU websocket fixtures.

Graph-only safetensors are never inference weights. Local test sockets do not
start Core, load trained weights, queue GPU work or contact a user service.
"""
# ruff: noqa: F811 -- imported pytest fixture intentionally appears as arguments

from copy import deepcopy
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import wave

import pytest

from test_h21_control_probe import prepared, probe, CORE  # noqa: F401,F811


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h21_event_capture_tested", ROOT / "tools/observe_h21_owned_events.py"
)
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def test_output_closure_does_not_require_disconnected_legacy_nodes():
    graph = {
        "1": {"class_type": "Source", "inputs": {}},
        "2": {"class_type": "Save", "inputs": {"images": ["1", 0]}},
        "6": {"class_type": "LegacyDisconnected", "inputs": {}},
    }
    assert probe.output_dependency_closure(graph, ["2"]) == {"1", "2"}


def test_append_report_observers_preserves_every_existing_node_and_ignores_orphans():
    graph = {
        "1": {"class_type": "MiniMaxH3DirectorControlSourceEXPT8", "inputs": {}},
        "2": {
            "class_type": "MiniMaxH3SafeAVSaveT8Advanced",
            "inputs": {"images": ["1", 0]},
        },
        "orphan": {"class_type": "MiniMaxH3AudioWindowT8", "inputs": {}},
    }
    before = deepcopy(graph)
    observed, contract = probe.append_report_observers(graph, ["2"])
    assert graph == before
    assert all(observed[key] == value for key, value in before.items())
    assert {(row["producer"], row["output"]) for row in contract["reports"]} == {
        ("1", 3),
        ("2", 2),
    }
    assert contract["sampling_inputs_changed"] is False
    assert contract["sampling_math_changed"] is False
    for row in contract["reports"]:
        assert observed[row["preview"]] == {
            "class_type": "PreviewAny",
            "inputs": {"source": [row["producer"], row["output"]]},
        }


def test_future_cpu_preparation_keeps_base_graph_and_real_validated_observers(prepared):  # noqa: F811
    _, run, _, _, _, plan, _ = prepared
    assert "runtime_tools/observe_h21_owned_events.py" in plan["sealed_files"]
    for case in plan["cases"]:
        base = probe.read_json(run / case["base_prompt"])
        graph = probe.read_json(run / case["prompt"])
        validation = probe.read_json(run / case["validation"])
        assert all(graph[key] == node for key, node in base.items())
        assert validation["valid"] is True and validation["node_errors"] == {}
        assert case["output_nodes"] == validation["outputs"]
        for row in case["observation"]["reports"]:
            assert row["preview"] in validation["outputs"]
            assert graph[row["preview"]]["inputs"]["source"] == [
                row["producer"],
                row["output"],
            ]
        assert set(base) <= set(graph)
        assert case["observation"]["reports"]


def test_success_history_with_missing_progress_is_incomplete_evidence_not_failure():
    summary = observer.summarize_case_events(
        [],
        "own",
        {"2"},
        {"status": {"status_str": "success", "completed": True}},
        connected_before_post=True,
    )
    assert summary["generation_status"] == "success"
    assert summary["event_evidence_status"] == "incomplete"
    assert summary["observed_progress"] == {}
    assert summary["NFE_inferred"] is False


def test_cached_and_foreign_prompt_events_are_separate_from_actual_callback_counts():
    events = [
        {"type": "execution_start", "data": {"prompt_id": "own"}},
        {"type": "execution_cached", "data": {"prompt_id": "own", "nodes": ["1"]}},
        {
            "type": "progress",
            "data": {"prompt_id": "foreign", "node": "2", "value": 8, "max": 8},
        },
        {"type": "executing", "data": {"prompt_id": "own", "node": "2"}},
        {"type": "execution_success", "data": {"prompt_id": "own"}},
    ]
    summary = observer.summarize_case_events(
        events,
        "own",
        {"1", "2"},
        {"status": {"status_str": "success", "completed": True}},
        connected_before_post=True,
    )
    assert summary["generation_status"] == "success"
    assert summary["cached_nodes"] == ["1"]
    assert summary["observed_progress"] == {}
    assert summary["uncached_dependency_coverage"] is False
    assert summary["NFE_inferred"] is False


def test_callback_evidence_retains_real_node_sequences_without_declared_steps():
    events = [{"type": "execution_start", "data": {"prompt_id": "own"}}]
    for node in ("low", "high"):
        events.append({"type": "executing", "data": {"prompt_id": "own", "node": node}})
        for value in range(1, 5):
            events.append(
                {
                    "type": "progress",
                    "data": {
                        "prompt_id": "own",
                        "node": node,
                        "value": value,
                        "max": 4,
                    },
                }
            )
    events.append({"type": "execution_success", "data": {"prompt_id": "own"}})
    summary = observer.summarize_case_events(
        events,
        "own",
        {"low", "high"},
        {"status": {"status_str": "success", "completed": True}},
        connected_before_post=True,
    )
    assert summary["event_evidence_status"] == "complete_start_terminal"
    assert summary["observed_progress"]["low"] == [
        {"value": n, "max": 4} for n in range(1, 5)
    ]
    assert summary["observed_progress"]["high"] == [
        {"value": n, "max": 4} for n in range(1, 5)
    ]
    assert summary["uncached_dependency_coverage"] is True


def test_reports_from_history_preserve_audit_and_source_provenance_not_quality():
    graph = {
        "audit": {"class_type": "MiniMaxH3ControlRegionAuditEXPT8", "inputs": {}},
        "source": {"class_type": "MiniMaxH3DirectorControlSourceEXPT8", "inputs": {}},
    }
    contract = {"reports": [{"producer": "source", "output": 3, "preview": "preview"}]}
    payload = {"schema": "source", "actual_indices": [0, 1], "source_fps": 24}
    history = {
        "outputs": {
            "audit": {"text": [json.dumps({"residual_calls": 5})]},
            "preview": {"text": [json.dumps(payload)]},
        }
    }
    result = probe.collect_case_reports(graph, contract, history)
    assert result["reports"]["source"]["report"] == payload
    assert result["reports"]["audit"]["report"]["residual_calls"] == 5
    assert result["media_quality_qualified"] is False
    assert result["missing_or_invalid"] == []


@pytest.fixture
def wire_server(tmp_path, monkeypatch):
    """Real aiohttp/WebSocket transport; explicit protocol-only CPU emitter."""
    from aiohttp import web
    import requests

    state = {
        "mode": "success",
        "posts": [],
        "ws": {},
        "history": {},
        "ready": threading.Event(),
    }
    loop = asyncio.new_event_loop()

    async def socket_handler(request):
        sock = web.WebSocketResponse()
        await sock.prepare(request)
        client = request.query["clientId"]
        state["ws"][client] = sock
        if state["mode"] != "no_ack":
            await sock.send_json(
                {
                    "type": "status",
                    "data": {
                        "sid": "wrong" if state["mode"] == "wrong_ack" else client
                    },
                }
            )
        async for _ in sock:
            pass
        return sock

    async def prompt_handler(request):
        payload = await request.json()
        state["posts"].append({"at": time.time(), "payload": payload})
        prompt_id = "prompt-" + str(len(state["posts"]))
        sock = state["ws"][payload["client_id"]]
        if state["mode"] == "post_reject":
            return web.json_response({"node_errors": {"9": "protocol test rejection"}})
        if state["mode"] == "post_error":
            raise web.HTTPInternalServerError(text="explicit test transport error")
        await sock.send_json(
            {"type": "execution_start", "data": {"prompt_id": prompt_id}}
        )
        if state["mode"] == "disconnect":
            await sock.close()
        elif state["mode"] != "history_timeout":
            await sock.send_json(
                {
                    "type": "execution_cached",
                    "data": {"prompt_id": prompt_id, "nodes": ["orphan"]},
                }
            )
            await sock.send_json(
                {"type": "execution_success", "data": {"prompt_id": "foreign"}}
            )
            for node_id in ("9", "12"):
                await sock.send_json(
                    {
                        "type": "executing",
                        "data": {"prompt_id": prompt_id, "node": node_id},
                    }
                )
            for value in range(1, 9):
                await sock.send_json(
                    {
                        "type": "progress",
                        "data": {
                            "prompt_id": prompt_id,
                            "node": "9",
                            "value": value,
                            "max": 8,
                        },
                    }
                )
            await sock.send_bytes(b"\x00\x01raw-test-wire")
            if state["mode"] == "invalid_packet":
                await sock.send_str("not JSON from protocol test")
            await sock.send_json(
                {
                    "type": "executed",
                    "data": {
                        "prompt_id": prompt_id,
                        "node": "12",
                        "output": {"text": ['{"actual":true}']},
                    },
                }
            )
            terminal = (
                "execution_error"
                if state["mode"] == "execution_error"
                else "execution_success"
            )
            await sock.send_json({"type": terminal, "data": {"prompt_id": prompt_id}})
        if state["mode"] != "history_timeout":
            state["history"][prompt_id] = {
                "status": {
                    "status_str": "error"
                    if state["mode"] == "execution_error"
                    else "success",
                    "completed": True,
                },
                "outputs": {},
            }
        return web.json_response({"prompt_id": prompt_id, "node_errors": {}})

    async def history_handler(request):
        prompt_id = request.match_info["prompt"]
        return web.json_response(
            {prompt_id: state["history"][prompt_id]}
            if prompt_id in state["history"]
            else {}
        )

    async def setup():
        app = web.Application()
        app.router.add_get("/ws", socket_handler)
        app.router.add_post("/prompt", prompt_handler)
        app.router.add_get("/history/{prompt}", history_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        state.update(port=site._server.sockets[0].getsockname()[1], runner=runner)
        state["ready"].set()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(setup())
        loop.run_forever()

    thread = threading.Thread(target=run, name="cpu-protocol-only-aiohttp", daemon=True)
    thread.start()
    assert state["ready"].wait(5)
    root = tmp_path / "owned"
    root.mkdir()
    manifest = root / "manifest.json"
    plan = {
        "schema": probe.SCHEMA,
        "run_root": str(root),
        "run_id": "testcase",
        "port": state["port"],
        "python": sys.executable,
    }
    manifest.write_text(json.dumps(plan), encoding="utf-8")
    owner = {
        "pid": os.getpid(),
        "create_time": 1.5,
        "exe": sys.executable,
        "argv": ["protocol-only test fixture"],
    }
    monkeypatch.setattr(observer, "listener_owner", lambda *_: dict(owner))
    monkeypatch.setitem(sys.modules, "observe_h21_owned_events", observer)
    session = requests.Session()
    session.trust_env = False

    def request(method, endpoint, **kwargs):
        response = session.request(
            method, f"http://127.0.0.1:{state['port']}" + endpoint, timeout=2, **kwargs
        )
        response.raise_for_status()
        return response.json()

    server = SimpleNamespace(
        process=SimpleNamespace(pid=os.getpid(), poll=lambda: None),
        assert_port_owner=lambda: None,
        request=request,
        stop=lambda: pytest.fail("The observer must never stop/interrupt a process"),
    )
    case = {
        "name": "selected",
        "prompt": "prompt.json",
        "validation": "validation.json",
        "observation": {"reports": []},
    }
    graph = {
        "9": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
        "12": {"class_type": "ProtocolOnlyOutput", "inputs": {"source": ["9", 0]}},
        "orphan": {"class_type": "UnusedProvenance", "inputs": {}},
    }
    (root / "prompt.json").write_text(json.dumps(graph), encoding="utf-8")
    (root / "validation.json").write_text(
        json.dumps({"valid": True, "node_errors": {}, "outputs": ["12"]}),
        encoding="utf-8",
    )
    try:
        yield state, owner, root, manifest, plan, server, case
    finally:
        session.close()
        asyncio.run_coroutine_threadsafe(state["runner"].cleanup(), loop).result(
            timeout=5
        )
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        assert not thread.is_alive()
        loop.close()


def test_actual_wire_prepost_ACK_raw_progress_binary_audit_and_terminal_persist(
    wire_server,
):
    state, _, root, manifest, plan, server, case = wire_server
    # Earlier attempts are irrelevant to this exact pre-POST proof.
    old = root / "attempts/old/executions/previous"
    old.mkdir(parents=True)
    (old / "submitted.json").write_text('{"prompt_id":"old"}', encoding="utf-8")
    folder = root / "attempts/new/executions/selected"
    history = probe.execute_observed_case(
        server, plan, manifest, case, folder, timeout=3
    )
    assert history["status"]["status_str"] == "success"
    assert len(state["posts"]) == 1
    receipt = probe.read_json(folder / "observation/terminal.json")
    prepost = probe.read_json(folder / "observation/pre-post.json")
    assert (
        prepost["connected_at"] <= prepost["post_started_at"] <= state["posts"][0]["at"]
    )
    assert receipt["event_evidence_status"] == "complete_start_terminal"
    assert receipt["uncached_dependency_coverage"] is True
    assert receipt["required_dependency_nodes"] == ["12", "9"]
    assert receipt["unrelated_cached_nodes"] == ["orphan"]
    assert receipt["observed_progress"]["9"] == [
        {"value": n, "max": 8} for n in range(1, 9)
    ]
    assert receipt["terminal_events"] == ["execution_success"]
    events = [
        json.loads(line)
        for line in (folder / "observation/events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    executed = next(row for row in events if row["type"] == "executed")
    assert json.loads(executed["raw_json"])["data"]["output"]["text"] == [
        '{"actual":true}'
    ]
    binary = next(row for row in events if row["type"] == "binary_observation")
    assert binary["raw_base64"] and binary["sha256"]
    assert all("received_at" in row for row in events)
    assert (folder / "reports.json").is_file()
    assert receipt["NFE_inferred"] is False


@pytest.mark.parametrize("mode", ["disconnect", "invalid_packet"])
def test_post_observer_loss_does_not_requeue_or_erase_successful_history(
    wire_server, mode
):
    state, _, root, manifest, plan, server, case = wire_server
    state["mode"] = mode
    folder = root / "attempts/new/executions/selected"
    assert (
        probe.execute_observed_case(server, plan, manifest, case, folder, timeout=3)[
            "status"
        ]["status_str"]
        == "success"
    )
    receipt = probe.read_json(folder / "observation/terminal.json")
    assert receipt["generation_status"] == "success"
    assert receipt["event_evidence_status"] == "incomplete"
    assert receipt["observer_errors"]
    assert len(state["posts"]) == 1


@pytest.mark.parametrize(
    "mode", ["post_reject", "post_error", "execution_error", "history_timeout"]
)
def test_transport_execution_error_timeout_preserve_receipts_without_retry_or_process_stop(
    wire_server, mode
):
    state, _, root, manifest, plan, server, case = wire_server
    state["mode"] = mode
    folder = root / "attempts/new/executions/selected"
    with pytest.raises(Exception):
        probe.execute_observed_case(server, plan, manifest, case, folder, timeout=0.35)
    receipt = probe.read_json(folder / "observation/terminal.json")
    assert receipt["controller_error"]
    assert (folder / "observation/events.jsonl").exists()
    assert len(state["posts"]) == 1
    assert server.process.poll() is None


@pytest.mark.parametrize("mode", ["wrong_ack", "no_ack"])
def test_mismatched_or_missing_Core_ACK_never_submits(wire_server, mode, monkeypatch):
    state, _, root, manifest, plan, server, case = wire_server
    state["mode"] = mode
    original = observer.OwnedEventCapture

    def short_capture(*args, **kwargs):
        return original(*args, **kwargs, handshake_timeout=0.1)

    monkeypatch.setattr(observer, "OwnedEventCapture", short_capture)
    folder = root / "attempts/new/executions/selected"
    with pytest.raises(TimeoutError, match="handshake ACK"):
        probe.execute_observed_case(server, plan, manifest, case, folder, timeout=1)
    assert state["posts"] == []
    receipt = probe.read_json(folder / "observation/terminal.json")
    assert receipt["generation_status"] == "unknown_no_authoritative_history"
    assert receipt["connected_before_POST"] is False


def test_owner_change_across_real_websocket_handshake_is_rejected_before_POST(
    wire_server, monkeypatch
):
    state, owner, root, manifest, plan, server, case = wire_server
    calls = []

    def changed_owner(*_):
        calls.append(1)
        return {**owner, "create_time": 999} if len(calls) > 1 else dict(owner)

    monkeypatch.setattr(observer, "listener_owner", changed_owner)
    folder = root / "attempts/new/executions/selected"
    with pytest.raises(RuntimeError, match="identity changed"):
        probe.execute_observed_case(server, plan, manifest, case, folder)
    assert state["posts"] == []
    assert probe.read_json(folder / "observation/terminal.json")["observer_errors"]


def test_explicit_selected_case_only_queues_that_graph_once(wire_server):
    state, _, root, manifest, plan, server, case = wire_server
    other = {**case, "name": "not-authorized"}
    plan["cases"] = [case, other]
    for selected in probe.select_cases(plan, ["selected"]):
        probe.execute_observed_case(
            server,
            plan,
            manifest,
            selected,
            root / "attempts/new/executions" / selected["name"],
            timeout=3,
        )
    assert len(state["posts"]) == 1
    assert not (root / "attempts/new/executions/not-authorized").exists()


def test_existing_case_artifacts_are_never_overwritten_or_requeued(wire_server):
    state, _, root, manifest, plan, server, case = wire_server
    folder = root / "attempts/new/executions/selected"
    probe.execute_observed_case(server, plan, manifest, case, folder, timeout=3)
    before = (folder / "observation/events.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        probe.execute_observed_case(server, plan, manifest, case, folder, timeout=3)
    assert (folder / "observation/events.jsonl").read_bytes() == before
    assert len(state["posts"]) == 1


def test_capture_preserves_malformed_and_nonfinite_wire_without_claiming_a_progress_event(
    wire_server,
):
    state, _, root, manifest, plan, _, _ = wire_server
    output = root / "attempts/private/observation"
    capture = observer.OwnedEventCapture(
        manifest, plan, output, client_id="test-wire", expected_pid=os.getpid()
    )
    capture.connect()
    capture.mark_post()
    for raw in (
        '{"type":"progress","data":{"value":NaN}}',
        '["not-an-object"]',
        "{broken",
    ):
        row = capture._record(raw)
        assert row["raw_json"] == raw
        assert row["type"] == "invalid_json_observation"
    result = capture.finish("unsubmitted-test", [], None)
    assert result["event_evidence_status"] == "incomplete"
    assert result["observed_progress"] == {}
    assert len(result["observer_errors"]) == 3
    assert state["posts"] == []


def test_single_POST_marker_and_exact_PID_are_required_without_process_actions(
    wire_server,
):
    state, _, root, manifest, plan, _, _ = wire_server
    capture = observer.OwnedEventCapture(
        manifest,
        plan,
        root / "attempts/private/observation",
        client_id="pid-test",
        expected_pid=os.getpid(),
    )
    with pytest.raises(RuntimeError, match="completed pre-POST"):
        capture.mark_post()
    capture.connect()
    capture.mark_post()
    with pytest.raises(RuntimeError, match="single POST"):
        capture.mark_post()
    capture.finish(None, [])
    wrong = observer.OwnedEventCapture(
        manifest,
        plan,
        root / "attempts/other/observation",
        client_id="wrong-pid",
        expected_pid=os.getpid() + 1,
    )
    with pytest.raises(RuntimeError, match="exact controller-owned PID"):
        wrong.connect()
    wrong.finish(None, [])
    assert state["posts"] == []


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_case_deadline_has_no_socket_or_POST_side_effect(wire_server, timeout):
    state, _, root, manifest, plan, server, case = wire_server
    folder = root / "attempts/new/executions/selected"
    with pytest.raises(ValueError, match="positive and finite"):
        probe.execute_observed_case(
            server, plan, manifest, case, folder, timeout=timeout
        )
    assert state["posts"] == [] and not folder.exists()


def test_original_controller_cpu_graphs_match_new_base_byte_structure(
    prepared, tmp_path
):  # noqa: F811
    _, run, _, config, _, plan, _ = prepared
    preimage = (
        ROOT
        / "tests/fixtures/h21_control_preimage/h21_control_probe.py"
    )
    old_spec = importlib.util.spec_from_file_location("h21_preimage_probe", preimage)
    original = importlib.util.module_from_spec(old_spec)
    old_spec.loader.exec_module(original)
    source = tmp_path / "preimage-source"
    for path in original.runtime_sources(ROOT):
        original.copy_exact(path, source / path.relative_to(ROOT))
    for name in original.TOOLS:
        original.copy_exact(
            preimage if name == "h21_control_probe.py" else ROOT / "tools" / name,
            source / "tools" / name,
        )
    config_file = tmp_path / "same-config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    old_run = tmp_path / "old-controller-run"
    old_plan = original.prepare(
        config_file, old_run, CORE, source_root=source, python=sys.executable
    )
    for old_case, new_case in zip(old_plan["cases"], plan["cases"], strict=True):
        old_graph = original.read_json(old_run / old_case["prompt"])
        base_graph = probe.read_json(run / new_case["base_prompt"])
        # The private output namespace alone contains the independently minted
        # run_id. This deliberate pre-existing controller field is not sampling.
        for node in old_graph.values():
            if node["class_type"] == "MiniMaxH3SafeAVSaveT8Advanced":
                node["inputs"]["filename_prefix"] = node["inputs"][
                    "filename_prefix"
                ].replace(old_plan["run_id"], plan["run_id"])
        assert old_graph == base_graph
        assert original.read_json(old_run / old_case["validation"])["valid"] is True
        assert set(probe.read_json(run / new_case["prompt"])) - set(base_graph) == {
            row["preview"] for row in new_case["observation"]["reports"]
        }
    assert probe.verify_prepared(run / "manifest.json")["run_id"] == plan["run_id"]
    assert (
        original.verify_prepared(old_run / "manifest.json")["run_id"]
        == old_plan["run_id"]
    )


@pytest.mark.parametrize("class_name,slot", list(probe.REPORT_OUTPUTS.items()))
def test_report_slot_contract_matches_real_Core_node_schema(class_name, slot):
    import asyncio
    from h3_audio_t8_pkg import comfy_entrypoint

    classes = {
        cls.__name__: cls for cls in asyncio.run(comfy_entrypoint().get_node_list())
    }
    cls = classes[class_name]
    actual = cls.define_schema().outputs[slot]
    assert actual.id == "report_json"
    assert actual.io_type == "STRING"


def test_real_AudioWindow_report_preview_validates_and_consumes_without_audio_change(
    tmp_path, monkeypatch
):
    import execution
    import folder_paths
    import nodes
    import torch
    from comfy_extras.nodes_audio import LoadAudio
    from comfy_extras.nodes_preview_any import PreviewAny
    from h3_audio_t8_pkg.nodes import MiniMaxH3AudioWindowT8

    monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", dict(nodes.NODE_CLASS_MAPPINGS))
    for cls in (LoadAudio, PreviewAny, MiniMaxH3AudioWindowT8):
        nodes.NODE_CLASS_MAPPINGS[cls.__name__] = cls
    monkeypatch.setattr(folder_paths, "input_directory", str(tmp_path))
    audio_file = tmp_path / "owned.wav"
    with wave.open(str(audio_file), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(32000)
        stream.writeframes(b"\x00\x00" * 32000)
    window_inputs = {
        "audio": ["audio", 0],
        "scene_start_seconds": 0.0,
        "scene_duration_seconds": 1.0,
        "warmup_seconds": 0.0,
        "cooldown_seconds": 0.0,
        "ensure_minimum_context": True,
    }
    graph = {
        "audio": {"class_type": "LoadAudio", "inputs": {"audio": "owned.wav"}},
        "window": {"class_type": "MiniMaxH3AudioWindowT8", "inputs": window_inputs},
        "existing": {"class_type": "PreviewAny", "inputs": {"source": ["window", 0]}},
    }
    before = deepcopy(graph)
    observed, contract = probe.append_report_observers(graph, ["existing"])
    valid = asyncio.run(execution.validate_prompt("cpu-audio-report", observed, None))
    assert valid[0] is True and not valid[3], valid
    assert graph == before and observed["window"] == before["window"]
    assert contract["reports"] == [
        {
            "producer": "window",
            "class_type": "MiniMaxH3AudioWindowT8",
            "output": 5,
            "preview": "h21_report_window",
        }
    ]
    waveform = torch.linspace(-0.3, 0.3, 32000).reshape(1, 1, -1)
    raw = {"waveform": waveform, "sample_rate": 32000}
    unchanged = waveform.clone()
    output = MiniMaxH3AudioWindowT8.execute(**{**window_inputs, "audio": raw})
    preview = PreviewAny().main(output.result[5])
    assert preview["ui"]["text"][0] == output.result[5]
    assert isinstance(json.loads(output.result[5]), dict)
    assert torch.equal(waveform, unchanged)
