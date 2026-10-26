"""Durable first-observed outcomes; missing Core history is never 'generating'."""
import asyncio
from copy import deepcopy
import json
import sys
from types import SimpleNamespace
import uuid

import pytest

from h3_audio_t8_pkg import director_routes as routes
from h3_audio_t8_pkg.director_project import ProjectStore, atomic_json


def job_route(monkeypatch, store, owner, *, all_routes=False):
    handlers = {}

    class Registry:
        def get(self, path):
            def register(handler):
                handlers[path] = handler
                return handler
            return register

        post = get

    owner.routes = Registry()
    monkeypatch.setitem(sys.modules, "server", SimpleNamespace(PromptServer=SimpleNamespace(instance=owner)))
    monkeypatch.setattr(routes, "_REGISTERED", False)
    monkeypatch.setattr(routes, "_GENERATE_LOCK", asyncio.Lock())
    monkeypatch.setattr(routes, "get_store", lambda: store)
    assert routes.register_director_routes()
    return handlers if all_routes else handlers[routes.PREFIX + "/jobs/{prompt_id}"]


def dispatch_job(handler, prompt):
    response = asyncio.run(handler(SimpleNamespace(match_info={"prompt_id": prompt})))
    assert response.status == 200
    return json.loads(response.text)


def receipt_job(tmp_path, *, epoch="original-core"):
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    project, shot, request, prompt = (str(uuid.uuid4()) for _ in range(4))
    path = store.root / "requests" / (request + ".json")
    original = {"state": "queued", "project_id": project, "shot_id": shot,
                "shot_rev": 3, "prompt_id": prompt, "core_epoch": epoch,
                "submitted_at": 100., "result": {"recipe": "director", "seed": 19}}
    atomic_json(path, original)
    return store, project, path, original


@pytest.mark.parametrize("state", ("success", "error", "cancelled"))
def test_first_observed_terminal_survives_new_core_and_does_not_leak_to_a_copy(tmp_path, monkeypatch, state):
    store, project, path, original = receipt_job(tmp_path)
    outputs = {"12": {"images": [{"filename": "saved.mp4", "subfolder": "owned", "type": "output"}]}} if state == "success" else {}
    calls = []

    def lookup(prompt):
        calls.append(prompt)
        return {"state": state, "outputs": deepcopy(outputs)}

    observed = routes.director_project_results(store, project, lookup)["results"]
    assert len(observed) == 1 and observed[0]["state"] == state
    assert observed[0]["outputs"] == outputs
    assert observed[0]["shot_rev"] == original["shot_rev"]
    assert json.loads(path.read_text(encoding="utf8"))["terminal"] == {"state": state, "outputs": outputs}
    monkeypatch.setattr(routes, "_CORE_EPOCH", "new-core")
    # Simulate reopening with a new store and no live queue/history.
    reopened = ProjectStore(tmp_path / "user", tmp_path / "input")
    def lost_history(_prompt):
        raise AssertionError("A frozen terminal does not require live history")
    assert routes.director_project_results(reopened, project, lost_history)["results"] == observed
    assert routes.director_project_results(reopened, str(uuid.uuid4()), lost_history)["results"] == []
    assert calls == [original["prompt_id"]]


@pytest.mark.parametrize("state", ("queued", "running", "unknown"))
@pytest.mark.parametrize("restart", (False, True))
def test_unobserved_history_keeps_actual_state_without_resubmission_or_fake_terminal(tmp_path, monkeypatch, state, restart):
    store, project, path, original = receipt_job(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(routes, "_CORE_EPOCH", "new-core" if restart else original["core_epoch"])
    row = routes.director_project_results(store, project, lambda _: {"state": state})["results"][0]
    assert row["state"] == state
    assert row["outputs"] == {}
    assert path.read_bytes() == before
    # An unknown result can become observable later, without creating a job.
    success = routes.director_project_results(store, project, lambda _: {"state": "success", "outputs": {}})["results"][0]
    assert success["state"] == "success"
    assert success["request_id"] == row["request_id"] and success["prompt_id"] == row["prompt_id"]


def test_unknown_backend_state_is_not_promoted_to_running(tmp_path):
    store, project, path, _ = receipt_job(tmp_path)
    before = path.read_bytes()
    row = routes.director_project_results(store, project, lambda _: {"state": "unrecognized"})["results"][0]
    assert row["state"] == "unknown" and row["outputs"] == {}
    assert path.read_bytes() == before


@pytest.mark.parametrize("terminal", ([], {"state": "success", "outputs": []},
                                      {"state": "invented", "outputs": {}}, {"state": [], "outputs": {}}))
def test_malformed_terminal_is_visible_as_needing_review_not_a_playable_success(tmp_path, terminal):
    store, project, path, original = receipt_job(tmp_path)
    atomic_json(path, {**original, "terminal": terminal})
    before = path.read_bytes()
    row = routes.director_project_results(store, project, lambda _: {"state": "running"}, output_root=tmp_path / "output")["results"][0]
    assert row["state"] == "needs_review" and row["outputs"] == {}
    assert row["integrity_error"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("damage", ("nested_media", "unhashable_media", "legacy_selection"))
def test_unrelated_damaged_output_metadata_does_not_hide_a_healthy_result(tmp_path, damage):
    store, project, path, original = receipt_job(tmp_path)
    good = {**original, "terminal": {"state": "success", "outputs": {}}}
    atomic_json(path, good)
    other = {**original, "prompt_id": str(uuid.uuid4())}
    if damage == "nested_media":
        other["terminal"] = {"state": "success", "outputs": {"12": {"images": 7}}}
    elif damage == "unhashable_media":
        other["terminal"] = {"state": "success", "outputs": {"12": {"images": [{"filename": {}, "subfolder": []}]}}}
    else:
        other.pop("shot_id")
        other["result"] = {"report": {"project_id": project, "selection": ["damaged"]}}
    atomic_json(path.with_name(str(uuid.uuid4()) + ".json"), other)
    rows = routes.director_project_results(store, project, lambda _: {"state": "unknown"}, output_root=tmp_path / "output")["results"]
    assert any(row["prompt_id"] == original["prompt_id"] and row["state"] == "success" for row in rows)


@pytest.mark.parametrize("state", ("success", "error", "cancelled"))
@pytest.mark.parametrize("first_observed", (False, True))
def test_real_core_queue_terminal_and_history_loss(tmp_path, monkeypatch, state, first_observed):
    import execution
    from h3_audio_t8_pkg.director_generation import director_job_status
    store, project, path, original = receipt_job(tmp_path)
    owner = SimpleNamespace(queue_updated=lambda: None)
    queue = execution.PromptQueue(owner)
    owner.prompt_queue = queue
    monkeypatch.setitem(sys.modules, "server", SimpleNamespace(PromptServer=SimpleNamespace(instance=owner)))
    prompt = original["prompt_id"]
    queue.put((0, prompt, {}, {}, []))
    assert director_job_status(prompt)["state"] == "queued"
    _, task_id = queue.get(timeout=.01)
    assert director_job_status(prompt)["state"] == "running"
    event = {"success": "execution_success", "error": "execution_error", "cancelled": "execution_interrupted"}[state]
    messages = [(event, {"prompt_id": prompt})]
    queue.task_done(task_id, {"outputs": {}}, execution.PromptQueue.ExecutionStatus(
        "success" if state == "success" else "error", state == "success", messages))
    if first_observed:
        assert routes.director_project_results(store, project, director_job_status)["results"][0]["state"] == state
    queue.wipe_history()
    result = routes.director_project_results(store, project, director_job_status)["results"][0]
    assert result["state"] == (state if first_observed else "unknown")
    assert ("terminal" in json.loads(path.read_text(encoding="utf8"))) is first_observed


@pytest.mark.parametrize("status", (None, {}, {"status_str": "invented"}, {"status_str": []}))
def test_history_without_a_known_completion_state_is_not_fabricated_success(tmp_path, monkeypatch, status):
    import execution
    from h3_audio_t8_pkg.director_generation import director_job_status
    store, project, path, original = receipt_job(tmp_path)
    owner = SimpleNamespace(queue_updated=lambda: None)
    queue = execution.PromptQueue(owner)
    owner.prompt_queue = queue
    monkeypatch.setitem(sys.modules, "server", SimpleNamespace(PromptServer=SimpleNamespace(instance=owner)))
    queue.history[original["prompt_id"]] = {"status": status, "outputs": {}}
    before = path.read_bytes()
    assert routes.director_project_results(store, project, director_job_status)["results"][0]["state"] == "unknown"
    assert path.read_bytes() == before


@pytest.mark.parametrize("state", ("success", "error", "cancelled"))
def test_cross_project_jobs_first_observation_persists_before_history_loss(tmp_path, monkeypatch, state):
    import execution
    from h3_audio_t8_pkg.director_generation import director_job_status
    store, project, path, original = receipt_job(tmp_path)
    _, other_project, other_path, _ = receipt_job(tmp_path)
    other_before = other_path.read_bytes()
    owner = SimpleNamespace(queue_updated=lambda: None)
    queue = execution.PromptQueue(owner)
    owner.prompt_queue = queue
    handler = job_route(monkeypatch, store, owner)
    prompt = original["prompt_id"]
    queue.put((0, prompt, {}, {}, []))
    _, task_id = queue.get(timeout=.01)
    outputs = {"12": {"images": [{"filename": "original.mp4", "type": "output"}]}} if state == "success" else {}
    event = {"success": "execution_success", "error": "execution_error", "cancelled": "execution_interrupted"}[state]
    queue.task_done(task_id, {"outputs": outputs}, execution.PromptQueue.ExecutionStatus(
        "success" if state == "success" else "error", state == "success", [(event, {"prompt_id": prompt})]))
    # No current-project UI or /results request is involved in this observation.
    observed = dispatch_job(handler, prompt)
    assert observed["state"] == state and observed["outputs"] == outputs
    saved = json.loads(path.read_text(encoding="utf8"))
    assert saved["terminal"]["state"] == state
    assert saved["terminal"]["outputs"] == outputs
    assert {key: value for key, value in saved.items() if key != "terminal"} == original
    assert saved["project_id"] == project and saved["shot_id"] == original["shot_id"]
    queue.wipe_history()
    reopened = ProjectStore(tmp_path / "user", tmp_path / "input")
    assert routes.director_project_results(reopened, project, director_job_status)["results"][0]["state"] == state
    other_results = routes.director_project_results(reopened, other_project, director_job_status)["results"]
    assert len(other_results) == 1 and other_results[0]["state"] == "unknown"
    assert other_results[0]["prompt_id"] != prompt
    assert other_path.read_bytes() == other_before
