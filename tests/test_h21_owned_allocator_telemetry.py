"""Stdlib/fake-transport contracts; no real Torch, CUDA, Core or model execution.

Run from an isolated root with --noconftest/PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
CUDA objects below are explicitly protocol doubles, never kernel evidence.
"""
from copy import deepcopy
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(os.environ.get("T8_H21_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
SPEC = importlib.util.spec_from_file_location("owned_allocator_controller", ROOT / "tools/h21_control_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
raw = probe._raw_api_helper()
MISSING = object()
BAD = [None, 0, 1, "false", "true", [], {}]
BASE_EXTENSION_SHA = "a1b28b321cfa3077441b522260dcfbe04a64f21733e8c99ae32bd788b9bddf85"


def bare_plan(root, value=MISSING):
    plan = dict(schema=probe.SCHEMA, run_id="a" * 32, run_root=str(root), core_root=str(root / "unused_core"),
                package_name="owned_runtime", package_root=str(root / "runtime_nodes/owned_runtime"),
                identity_extension="owned_identity", python=str(Path(sys.executable).resolve()),
                port=49157, reserve_vram_gb=5)
    if value is not MISSING:
        plan["allocator_telemetry"] = value
    return plan


def metadata_files(tmp_path, value=True, config_value=MISSING):
    root = tmp_path / "owned"
    root.mkdir()
    (root / "runtime_tools").mkdir()
    extension = root / "runtime_nodes/owned_identity/__init__.py"
    extension.parent.mkdir(parents=True)
    plan = bare_plan(root, value)
    extension.write_text(probe._identity_extension(plan), encoding="utf8")
    prompt = root / "cases/only/prompt.json"
    prompt.parent.mkdir(parents=True)
    prompt.write_text(json.dumps({"1": {"class_type": "PreviewAny", "inputs": {"source": "literal"}}}), encoding="utf8")
    declared = value if config_value is MISSING else config_value
    config = {} if declared is MISSING else {"allocator_telemetry": declared}
    probe.write_json(root / "config.json", config)
    plan.update(cases=[{"name": "only", "prompt": "cases/only/prompt.json"}], resources={"models": {}},
                core_files={}, python_identity=probe.file_identity(sys.executable), graph_class_types=[])
    plan["sealed_files"] = {path.relative_to(root).as_posix(): probe.file_identity(path)
                            for path in (extension, prompt, root / "config.json")}
    plan["server_argv"] = probe.server_command(plan)
    manifest = root / "manifest.json"
    probe.write_json(manifest, plan)
    return plan, manifest, extension, prompt


@pytest.mark.parametrize("value", [MISSING, False])
def test_default_identity_bytes_and_entire_server_argv_are_unchanged(tmp_path, value):
    plan = bare_plan(tmp_path, value)
    assert hashlib.sha256(probe._identity_extension(plan).encode()).hexdigest() == BASE_EXTENSION_SHA
    without = deepcopy(plan)
    without.pop("allocator_telemetry", None)
    assert probe.server_command(plan) == probe.server_command(without)


def test_true_installs_only_two_optin_endpoints_without_argv_or_graph_change(tmp_path):
    plan = bare_plan(tmp_path, True)
    text = probe._identity_extension(plan)
    assert "/h21_probe/allocator/begin" in text and "/h21_probe/allocator/finish" in text
    assert text.startswith(probe._identity_extension({}))
    old = deepcopy(plan)
    old.pop("allocator_telemetry")
    assert probe.server_command(plan) == probe.server_command(old)


@pytest.mark.parametrize("bad", BAD)
def test_explicit_flag_is_strict_bool_before_any_socket_or_source_freeze(tmp_path, monkeypatch, bad):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"allocator_telemetry": bad}), encoding="utf8")
    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid allocator flag reached socket")
    monkeypatch.setattr(probe.socket, "socket", forbidden)
    with pytest.raises(ValueError, match="allocator_telemetry.*bool"):
        probe.prepare(config, tmp_path / "unused", tmp_path)


@pytest.mark.parametrize("value", [MISSING, False, True])
def test_raw_config_flag_does_not_drop_or_change_original_graph_fields(value):
    config = dict(schema="t8.h21.raw_api_config.v1", mode="raw_api", assets={}, model_directories={},
                  models=[dict(node="1", input="unet_name", folder="diffusion_models", name="user-selected.safetensors")],
                  raw_api_cases=[dict(name="only", api="original.json", api_sha256="b" * 64)])
    if value is not MISSING:
        config["allocator_telemetry"] = value
    before = deepcopy(config)
    raw.validate_config(config)
    assert config == before


@pytest.mark.parametrize("bad", BAD)
def test_raw_flag_nonbool_is_not_ignored_or_accepted_by_truthiness(bad):
    with pytest.raises(ValueError, match="allocator_telemetry.*bool"):
        raw.validate_config({"mode": "raw_api", "allocator_telemetry": bad})


@pytest.mark.parametrize("value", [MISSING, False, True])
def test_real_sealed_metadata_verify_preserves_presence_and_old_absence(tmp_path, value):
    plan, manifest, _, _ = metadata_files(tmp_path, value)
    assert probe.verify_prepared(manifest) == plan


@pytest.mark.parametrize("plan_value,config_value", [(True, False), (False, True), (MISSING, True), (True, MISSING)])
def test_plan_and_sealed_config_mismatch_is_rejected(tmp_path, plan_value, config_value):
    plan, manifest, _, _ = metadata_files(tmp_path, plan_value)
    config = {} if config_value is MISSING else {"allocator_telemetry": config_value}
    (manifest.parent / "config.json").write_text(json.dumps(config), encoding="utf8")
    plan["sealed_files"]["config.json"] = probe.file_identity(manifest.parent / "config.json")
    manifest.write_text(json.dumps(plan), encoding="utf8")
    with pytest.raises(RuntimeError, match="allocator_telemetry.*config|config.*allocator_telemetry"):
        probe.verify_prepared(manifest)


class FakeCuda:
    """Protocol counts only; all numbers are synthetic and not GPU evidence."""
    def __init__(self):
        self.calls = []
        self.initialized = True
        self.explode = False

    def is_initialized(self):
        self.calls.append("is_initialized")
        return self.initialized

    def reset_peak_memory_stats(self, device):
        self.calls.append(("reset", device))

    def memory_allocated(self, device):
        self.calls.append(("allocated", device))
        return 10

    def memory_reserved(self, device):
        self.calls.append(("reserved", device))
        return 20

    def max_memory_allocated(self, device):
        self.calls.append(("max_allocated", device))
        return 30

    def max_memory_reserved(self, device):
        self.calls.append(("max_reserved", device))
        if self.explode:
            raise RuntimeError("actual statistics API failed")
        return 40

    def get_allocator_backend(self):
        return "native"


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    plan, manifest, extension, prompt = metadata_files(tmp_path)
    cuda, registered, histories = FakeCuda(), {}, {}
    class Routes:
        def get(self, path):
            return lambda function: function
        def post(self, path):
            def register(function):
                registered[path] = function
                return function
            return register
    def response(value, status=200):
        return SimpleNamespace(status=status, body=json.dumps(value, allow_nan=False).encode())
    aiohttp = ModuleType("aiohttp")
    aiohttp.web = SimpleNamespace(json_response=response)
    server = ModuleType("server")
    server.PromptServer = SimpleNamespace(instance=SimpleNamespace(routes=Routes(),
        prompt_queue=SimpleNamespace(get_history=lambda prompt_id: {prompt_id: histories[prompt_id]} if prompt_id in histories else {})))
    comfy = ModuleType("comfy")
    comfy.cli_args = SimpleNamespace(args=SimpleNamespace(cpu=False))
    comfy.model_management = SimpleNamespace(get_torch_device=lambda: SimpleNamespace(type="cuda", index=0))
    tensor_protocol = ModuleType("torch")
    tensor_protocol.cuda = cuda
    tensor_protocol.__version__ = "synthetic-protocol-not-Torch"
    for name, module in {"aiohttp": aiohttp, "nodes": SimpleNamespace(NODE_CLASS_MAPPINGS={}),
                         "server": server, "comfy": comfy, "comfy.cli_args": comfy.cli_args,
                         "comfy.model_management": comfy.model_management, "torch": tensor_protocol}.items():
        monkeypatch.setitem(sys.modules, name, module)
    namespace = {"__file__": str(extension), "__name__": "owned_allocator_route_test"}
    exec(compile(extension.read_text(encoding="utf8"), str(extension), "exec"), namespace)
    payload = dict(run_id=plan["run_id"], manifest_sha256=probe.file_identity(manifest)["sha256"],
                   expected_pid=os.getpid(), case_name="only", prompt_sha256=probe.file_identity(prompt)["sha256"],
                   interval_id="c" * 32)
    def call(action, body=None):
        selected = deepcopy(payload if body is None else body)
        if action == "finish" and "prompt_id" not in selected:
            selected["prompt_id"] = "actual_prompt_id"
        class Request:
            async def json(self):
                return selected
        value = asyncio.run(registered["/h21_probe/allocator/" + action](Request()))
        return value.status, json.loads(value.body)
    return SimpleNamespace(call=call, cuda=cuda, plan=plan, manifest=manifest, prompt=prompt,
                           extension=extension, namespace=namespace, payload=payload,
                           registered=registered, comfy=comfy, histories=histories)


def add_history(endpoint, prompt_id="actual_prompt_id", changed=False):
    graph = probe.read_json(endpoint.prompt)
    if changed:
        graph["1"]["inputs"]["source"] = "other graph"
    endpoint.histories[prompt_id] = {"prompt": [1, prompt_id, graph, {}, []],
        "status": {"status_str": "success", "completed": True}}


def test_bound_begin_then_actual_terminal_finish_reads_peaks_with_one_reset(endpoint):
    status, started = endpoint.call("begin")
    assert status == 200 and started["status"] == "active"
    add_history(endpoint)
    status, ended = endpoint.call("finish")
    assert status == 200 and ended["status"] == "complete"
    assert ended["scope"] == "whole_prompt_not_DiT_or_stage_peak"
    assert ended["pid"] == os.getpid() and ended["interval_id"] == endpoint.payload["interval_id"]
    assert ended["peaks"]["max_allocated_bytes"] == 30 and ended["peaks"]["max_reserved_bytes"] == 40
    assert len([call for call in endpoint.cuda.calls if isinstance(call, tuple) and call[0] == "reset"]) == 1


@pytest.mark.parametrize("field,value", [("expected_pid", -1), ("run_id", "d" * 32),
    ("manifest_sha256", "0" * 64), ("prompt_sha256", "0" * 64), ("case_name", "../foreign")])
def test_fake_owner_manifest_or_case_is_rejected_before_any_CUDA_read(endpoint, field, value):
    bad = {**endpoint.payload, field: value}
    status, response = endpoint.call("begin", bad)
    assert status >= 400 and response["status"] == "rejected"
    assert endpoint.cuda.calls == []


@pytest.mark.parametrize("kind", ["cpu", "uninitialized"])
def test_CPU_or_missing_existing_CUDA_context_never_initializes_or_reads_stats(endpoint, kind):
    if kind == "cpu":
        endpoint.comfy.cli_args.args.cpu = True
    else:
        endpoint.cuda.initialized = False
    status, response = endpoint.call("begin")
    assert status == 200 and response["status"] == "unavailable"
    assert response["peaks"] is None
    assert endpoint.cuda.calls == ([] if kind == "cpu" else ["is_initialized"])


def test_duplicate_begin_and_wrong_finish_do_not_reset_or_close_valid_interval(endpoint):
    assert endpoint.call("begin")[0] == 200
    calls = list(endpoint.cuda.calls)
    assert endpoint.call("begin")[0] >= 400
    add_history(endpoint)
    assert endpoint.call("finish", {**endpoint.payload, "interval_id": "e" * 32})[0] >= 400
    assert endpoint.cuda.calls == calls
    assert endpoint.call("finish")[1]["status"] == "complete"


@pytest.mark.parametrize("kind", ["no_history", "changed_graph"])
def test_finish_requires_real_owned_terminal_history_for_same_submitted_graph(endpoint, kind):
    endpoint.call("begin")
    before = list(endpoint.cuda.calls)
    if kind == "changed_graph":
        add_history(endpoint, changed=True)
    status, response = endpoint.call("finish")
    assert status >= 400 and response["status"] == "rejected"
    assert endpoint.cuda.calls == before


def test_statistics_API_failure_is_not_zero_or_success_and_never_retried(endpoint):
    endpoint.cuda.explode = True
    status, response = endpoint.call("begin")
    assert status == 200 and response["status"] == "failed"
    assert response["peaks"] is None and "actual statistics API failed" in response["error"]
    calls = list(endpoint.cuda.calls)
    assert endpoint.call("begin")[0] >= 400
    assert endpoint.cuda.calls == calls


@pytest.mark.parametrize("kind", ["extension_path", "extension_bytes", "prompt_bytes", "root", "python", "prompt_escape"])
def test_actual_owned_file_or_interpreter_boundary_precedes_any_stats(endpoint, tmp_path, kind):
    if kind == "extension_path":
        endpoint.namespace["__file__"] = str(endpoint.prompt)
    elif kind == "extension_bytes":
        endpoint.extension.write_text("different sealed extension", encoding="utf8")
    elif kind == "prompt_bytes":
        endpoint.prompt.write_text("{}", encoding="utf8")
    else:
        plan = probe.read_json(endpoint.manifest)
        if kind == "root":
            plan["run_root"] = str(tmp_path)
        elif kind == "python":
            plan["python"] = str(endpoint.prompt)
        else:
            outside = tmp_path / "outside.json"
            outside.write_bytes(endpoint.prompt.read_bytes())
            plan["cases"][0]["prompt"] = "../outside.json"
            plan["sealed_files"]["../outside.json"] = probe.file_identity(outside)
        endpoint.manifest.write_text(json.dumps(plan), encoding="utf8")
        endpoint.payload["manifest_sha256"] = probe.file_identity(endpoint.manifest)["sha256"]
    status, report = endpoint.call("begin")
    assert status >= 400 and report["peaks"] is None
    assert endpoint.cuda.calls == []


@pytest.fixture
def controller_case(tmp_path, monkeypatch):
    plan, manifest, _, prompt = metadata_files(tmp_path)
    root = manifest.parent
    validation = root / "cases/only/validation.json"
    probe.write_json(validation, {"valid": True, "node_errors": {}, "outputs": ["1"]})
    case = {"name": "only", "prompt": "cases/only/prompt.json", "validation": "cases/only/validation.json"}
    trace, posted, captures = [], [], []
    history = {"status": {"status_str": "success", "completed": True}, "outputs": {}}
    class Capture:
        def __init__(self, *_args, **_kwargs):
            self.sock = object()
            self.events = [
                {"type": "execution_start", "received_at": 10.0, "data": {"prompt_id": "owned_prompt"}},
                {"type": "execution_success", "received_at": 12.0, "data": {"prompt_id": "owned_prompt"}},
            ]
            captures.append(self)
        def connect(self):
            trace.append("ack")
        def mark_post(self):
            trace.append("post_mark")
        def receive(self):
            trace.append("receive")
        def terminal_for(self, _prompt):
            return True
        def finish(self, *_args, **_kwargs):
            trace.append("capture_finish")
    fake_observer = ModuleType("observe_h21_owned_events")
    fake_observer.OwnedEventCapture = Capture
    monkeypatch.setitem(sys.modules, "observe_h21_owned_events", fake_observer)
    state = SimpleNamespace(begin_error=None, finish_error=None, owner_error=None, bad_pid=False, reject_prompt=False)
    class Server:
        process = SimpleNamespace(pid=1011, poll=lambda: None)
        def assert_port_owner(self):
            if state.owner_error:
                raise state.owner_error
        def request(self, method, path, **kwargs):
            if path.startswith("/h21_probe/allocator/"):
                action = path.rsplit("/", 1)[-1]
                trace.append(action)
                error = state.begin_error if action == "begin" else state.finish_error
                if error:
                    raise error
                payload = kwargs["json"]
                return {**{key: payload[key] for key in ("run_id", "manifest_sha256", "case_name", "prompt_sha256", "interval_id")},
                    "pid": 999 if state.bad_pid else 1011, "status": "active" if action == "begin" else "complete", "reset_count": 1,
                    "peaks": {"allocated_bytes": 10, "reserved_bytes": 20, "max_allocated_bytes": 30, "max_reserved_bytes": 40}}
            if path == "/prompt":
                trace.append("prompt")
                posted.append(deepcopy(kwargs["json"]["prompt"]))
                return {"node_errors": {"real": "bad input"}} if state.reject_prompt else {"prompt_id": "owned_prompt", "node_errors": {}}
            assert method == "GET" and path == "/history/owned_prompt"
            trace.append("history")
            return {"owned_prompt": history}
    return SimpleNamespace(plan=plan, manifest=manifest, prompt=prompt, case=case, trace=trace, posted=posted,
        server=Server(), state=state, folder=root / "attempts/new/only", history=history, captures=captures)


def run_controller(fixture):
    return probe.execute_observed_case(fixture.server, fixture.plan, fixture.manifest,
                                       fixture.case, fixture.folder, timeout=1)


def test_controller_begins_after_ACK_before_POST_and_finishes_after_actual_history(controller_case):
    value = controller_case
    assert run_controller(value) is value.history
    assert value.trace.index("ack") < value.trace.index("begin") < value.trace.index("post_mark") < value.trace.index("prompt")
    assert value.trace.index("history") < value.trace.index("finish") < value.trace.index("capture_finish")
    assert value.posted == [probe.read_json(value.prompt)]
    report = probe.read_json(value.folder / "allocator-telemetry.json")
    assert report["status"] == "complete" and report["prompt_event_wall_seconds"] == 2.0
    assert report["performance_qualified"] is False and report["NFE_or_full_DiT_calls_inferred"] is False


@pytest.mark.parametrize("value", [MISSING, False])
def test_controller_default_has_no_allocator_request_or_artifact(controller_case, value):
    fixture = controller_case
    fixture.plan.pop("allocator_telemetry")
    if value is not MISSING:
        fixture.plan["allocator_telemetry"] = value
    run_controller(fixture)
    assert "begin" not in fixture.trace and "finish" not in fixture.trace
    assert not list(fixture.folder.glob("allocator-*.json"))
    assert fixture.posted == [probe.read_json(fixture.prompt)]


@pytest.mark.parametrize("action", ["begin", "finish"])
def test_allocator_timeout_keeps_one_real_original_prompt_without_retry(controller_case, action):
    fixture = controller_case
    setattr(fixture.state, action + "_error", TimeoutError("allocator endpoint timeout"))
    assert run_controller(fixture) is fixture.history
    assert fixture.trace.count("prompt") == 1 and fixture.trace.count(action) == 1
    report = probe.read_json(fixture.folder / "allocator-telemetry.json")
    assert report["status"] == "failed_or_incomplete" and report["performance_qualified"] is False
    observed = report[action]
    assert observed["peaks"] is None and "timeout" in observed["error"]


def test_actual_false_response_owner_is_not_softened_into_a_telemetry_warning(controller_case):
    fixture = controller_case
    fixture.state.bad_pid = True
    with pytest.raises(RuntimeError, match="different actual owner"):
        run_controller(fixture)
    assert fixture.posted == [] and "prompt" not in fixture.trace


def test_actual_listener_owner_error_and_prompt_error_are_preserved(controller_case):
    fixture = controller_case
    fixture.state.owner_error = RuntimeError("real listener owner mismatch")
    with pytest.raises(RuntimeError, match="real listener owner mismatch"):
        run_controller(fixture)
    assert fixture.posted == []


def test_actual_prompt_rejection_is_not_retried_or_relabelled_as_stats_failure(controller_case):
    fixture = controller_case
    fixture.state.reject_prompt = True
    with pytest.raises(RuntimeError, match="Real Core rejected"):
        run_controller(fixture)
    assert len(fixture.posted) == 1 and "finish" not in fixture.trace


def test_callback_interval_remains_a_bound_observation_not_NFE_or_GPU_duration():
    rows = [
        {"type": "progress", "received_at": 4.0, "data": {"node": "14", "prompt_id": "p", "value": 1, "max": 20}},
        {"type": "progress", "received_at": 7.0, "data": {"node": "14", "prompt_id": "p", "value": 20, "max": 20}},
        {"type": "progress", "received_at": 99.0, "data": {"node": "14", "prompt_id": "other", "value": 20}},
    ]
    report = probe._allocator_event_intervals(rows, "p", {"14": "SamplerCustomAdvanced"})
    assert report["prompt_event_wall_seconds"] is None
    assert report["sampler_callback_intervals"]["14"]["first_to_last_seconds"] == 3.0
    assert len(report["sampler_callback_intervals"]["14"]["callbacks"]) == 2
    assert report["NFE_or_full_DiT_calls_inferred"] is False


@pytest.mark.parametrize("action", ["begin", "finish"])
@pytest.mark.parametrize("reset_count", [True, False, 1.0])
def test_client_requires_integer_exactly_one_reset_not_bool_or_float(controller_case, action, reset_count):
    fixture = controller_case
    payload = dict(run_id=fixture.plan["run_id"], manifest_sha256=probe.file_identity(fixture.manifest)["sha256"],
                   expected_pid=fixture.server.process.pid, case_name="only",
                   prompt_sha256=probe.file_identity(fixture.prompt)["sha256"], interval_id="f" * 32)
    if action == "finish":
        payload["prompt_id"] = "owned_prompt"
    class InvalidResetResponse:
        process = fixture.server.process
        def assert_port_owner(self):
            fixture.server.assert_port_owner()
        def request(self, *args, **kwargs):
            response = fixture.server.request(*args, **kwargs)
            return {**response, "reset_count": reset_count}
    response = probe._allocator_request(InvalidResetResponse(), action, payload)
    assert response["status"] == "failed"
    assert response["peaks"] is None
    assert type(response["raw_response"]["reset_count"]) is type(reset_count)
    assert fixture.trace.count(action) == 1
