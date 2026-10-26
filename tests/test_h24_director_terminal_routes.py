"""All terminal observers share one original receipt, never another project."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from h3_audio_t8_pkg import director_batch as batches
from h3_audio_t8_pkg import director_routes as routes
from h3_audio_t8_pkg.director_project import atomic_json, new_project, sha
from test_director_batch import _setup, _create, _receipt
from test_h24_director_result_observation import receipt_job, job_route, dispatch_job


def endpoint(tmp_path, monkeypatch, observed):
    store, project, path, original = receipt_job(tmp_path)
    monkeypatch.setattr(routes, "director_job_status", lambda _prompt: deepcopy(observed))
    handler = job_route(monkeypatch, store, SimpleNamespace())
    return store, project, path, original, handler


@pytest.mark.parametrize("state", ("queued", "running", "unknown"))
def test_live_nonterminal_does_not_change_or_create_receipts(tmp_path, monkeypatch, state):
    store, _, path, original, handler = endpoint(tmp_path, monkeypatch, {"state": state})
    before = path.read_bytes()
    assert dispatch_job(handler, original["prompt_id"])["state"] == state
    assert dispatch_job(handler, str(uuid.uuid4()))["state"] == state
    assert path.read_bytes() == before
    assert list((store.root / "requests").glob("*.json")) == [path]


@pytest.mark.parametrize("state", ("success", "error", "cancelled"))
@pytest.mark.parametrize("bad", ([], None, False, 0))
def test_live_terminal_bad_outputs_are_not_frozen_as_success(tmp_path, monkeypatch, state, bad):
    _, _, path, original, handler = endpoint(tmp_path, monkeypatch, {"state": state, "outputs": bad})
    before = path.read_bytes()
    result = dispatch_job(handler, original["prompt_id"])
    assert result["state"] == "needs_review" and result["outputs"] == {}
    assert path.read_bytes() == before


@pytest.mark.parametrize("bad", ([], {}, {"state": "running"}, {"state": "success", "outputs": []}))
def test_bad_saved_terminal_is_never_replaced_with_live_success(tmp_path, monkeypatch, bad):
    _, _, path, original, handler = endpoint(tmp_path, monkeypatch, {"state": "success", "outputs": {}})
    atomic_json(path, {**original, "terminal": bad})
    before = path.read_bytes()
    assert dispatch_job(handler, original["prompt_id"])["state"] == "needs_review"
    assert path.read_bytes() == before


@pytest.mark.parametrize("state", ("success", "error", "cancelled"))
def test_saved_terminal_is_authoritative_after_history_loss_and_keeps_error_details(tmp_path, monkeypatch, state):
    _, _, path, original, handler = endpoint(tmp_path, monkeypatch, {"state": "unknown"})
    terminal = {"state": state, "outputs": {}, "status": {"messages": [["execution_error", {"exception_message": "original failure"}]]}}
    atomic_json(path, {**original, "terminal": terminal, "media_evidence": [{"sentinel": "retain"}]})
    before = path.read_bytes()
    result = dispatch_job(handler, original["prompt_id"])
    assert result["state"] == state and result["status"] == terminal["status"]
    assert result["recovered_by"] == "request_receipt"
    assert path.read_bytes() == before


@pytest.mark.parametrize("completed_first", (False, True))
def test_same_prompt_two_owners_never_assigns_or_cancels_one_by_guess(tmp_path, monkeypatch, completed_first):
    store, _, path, original, handler = endpoint(tmp_path, monkeypatch, {"state": "success", "outputs": {}})
    other_path = path.with_name(str(uuid.uuid4()) + ".json")
    other = {**original, "project_id": str(uuid.uuid4()), "shot_id": str(uuid.uuid4())}
    if completed_first:
        atomic_json(path, {**original, "terminal": {"state": "error", "outputs": {}}})
    atomic_json(other_path, other)
    before = {item: item.read_bytes() for item in (path, other_path)}
    dispatch_job(handler, original["prompt_id"])
    assert routes._record_deleted_queue_receipt(store, original["prompt_id"]) is False
    assert all(item.read_bytes() == value for item, value in before.items())


def test_terminal_write_failure_is_http_error_and_original_receipt_is_untouched(tmp_path, monkeypatch):
    _, _, path, original, handler = endpoint(tmp_path, monkeypatch, {"state": "success", "outputs": {}})
    before = path.read_bytes()
    def fail_write(*_args):
        raise OSError("owned disk write failed")
    monkeypatch.setattr(routes, "atomic_json", fail_write)
    response = asyncio.run(handler(SimpleNamespace(match_info={"prompt_id": original["prompt_id"]})))
    assert response.status == 500
    assert path.read_bytes() == before


@pytest.mark.parametrize("terminal", ([], {}, {"state": "queued"}, {"state": "cancelled", "outputs": []}))
def test_batch_does_not_launder_bad_saved_terminal_through_live_history(tmp_path, terminal):
    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 23)
    path, _ = _receipt(store, batch, 0, terminal=terminal)
    before = path.read_bytes()
    result = batches.batch_status(store, batch_id, lambda _: {"state": "cancelled", "outputs": {}}, output)
    assert result["items"][0]["state"] == "needs_review"
    assert path.read_bytes() == before


def test_bundle_result_observation_takes_generation_lock_but_hashing_does_not(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_bundle
    import folder_paths
    store, _, _, _ = receipt_job(tmp_path)
    project = new_project()
    handlers = job_route(monkeypatch, store, SimpleNamespace(), all_routes=True)
    lock = routes._GENERATE_LOCK
    seen = []
    def records(*_args, **_kwargs):
        seen.append("observe")
        assert lock.locked(), "bundle observation can persist original request receipts"
        return {"results": []}
    def prepare(*_args, **_kwargs):
        seen.append("prepare")
        assert not lock.locked(), "large hashes/copies must not block submissions"
        return {"ready": True}
    monkeypatch.setattr(routes, "director_project_results", records)
    monkeypatch.setattr(director_bundle, "prepare_bundle", prepare)
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path / "output"))
    async def body():
        return {"project": project}
    response = asyncio.run(handlers[routes.PREFIX + "/bundles/prepare"](SimpleNamespace(json=body)))
    assert response.status == 200 and seen == ["observe", "prepare"]


@pytest.mark.parametrize("damage", ("project", "shot", "sha", "legacy_project", "legacy_shot"))
def test_valid_UUID_but_conflicting_receipt_owner_is_not_certified(tmp_path, monkeypatch, damage):
    store, project, path, original, handler = endpoint(tmp_path, monkeypatch, {"state": "success", "outputs": {}})
    snapshot = {"project": {"id": project}, "shot_id": original["shot_id"]}
    damaged = {**original, "snapshot": snapshot, "snapshot_sha256": sha(snapshot)}
    if damage in {"project", "shot"}:
        damaged[damage + "_id"] = str(uuid.uuid4())
    elif damage == "sha":
        damaged["snapshot_sha256"] = "0" * 64
    else:
        damaged["result"] = {"report": {"project_id": project, "selection": {"shot_id": original["shot_id"]}}}
        report = damaged["result"]["report"]
        if damage == "legacy_project":
            report["project_id"] = str(uuid.uuid4())
        else:
            report["selection"]["shot_id"] = str(uuid.uuid4())
    atomic_json(path, damaged)
    before = path.read_bytes()
    assert dispatch_job(handler, original["prompt_id"])["state"] == "needs_review"
    def no_lookup(_prompt):
        raise AssertionError("A conflicting owner cannot acquire a new terminal")
    rows = routes.director_project_results(store, damaged["project_id"], no_lookup)["results"]
    assert rows[0]["state"] == "needs_review"
    assert path.read_bytes() == before


def test_saved_terminal_recovery_does_not_require_a_live_Core_history_API(tmp_path, monkeypatch):
    _, _, path, original, handler = endpoint(tmp_path, monkeypatch, {"state": "unknown"})
    atomic_json(path, {**original, "terminal": {"state": "cancelled", "outputs": {}}})
    def unavailable(_prompt):
        raise ConnectionError("Core history unavailable")
    monkeypatch.setattr(routes, "director_job_status", unavailable)
    assert dispatch_job(handler, original["prompt_id"])["state"] == "cancelled"


@pytest.mark.parametrize("entrypoint", ("results", "batch"))
@pytest.mark.parametrize("state", ("success", "error", "cancelled"))
def test_original_Core_terminal_details_survive_whichever_observer_was_first(tmp_path, monkeypatch, entrypoint, state):
    import execution
    from h3_audio_t8_pkg.director_generation import director_job_status
    if entrypoint == "results":
        store, project, path, original = receipt_job(tmp_path)
        output, batch_id = None, None
    else:
        store, batch_id, selected, output = _setup(tmp_path)
        project = selected["id"]
        batch = _create(store, batch_id, selected, 3)
        path, original = _receipt(store, batch, 0)
    owner = SimpleNamespace(queue_updated=lambda: None)
    queue = execution.PromptQueue(owner)
    owner.prompt_queue = queue
    handler = job_route(monkeypatch, store, owner)
    prompt = original["prompt_id"]
    queue.put((0, prompt, {}, {}, []))
    _, task_id = queue.get(timeout=.01)
    event = {"success": "execution_success", "error": "execution_error", "cancelled": "execution_interrupted"}[state]
    outputs = {}
    if entrypoint == "batch" and state == "success":
        from test_director_batch import _write_av
        video = output / "done.mp4"
        _write_av(video)
        outputs = {"save": {"images": [{"filename": video.name, "type": "output", "subfolder": ""}]}}
    queue.task_done(task_id, {"outputs": outputs}, execution.PromptQueue.ExecutionStatus(
        "success" if state == "success" else "error", state == "success",
        [(event, {"prompt_id": prompt, "exception_message": "original error details"})]))
    if entrypoint == "results":
        assert routes.director_project_results(store, project, director_job_status)["results"][0]["state"] == state
    else:
        assert batches.batch_status(store, batch_id, director_job_status, output)["items"][0]["state"] == state
    queue.wipe_history()
    recovered = dispatch_job(handler, prompt)
    assert recovered["state"] == state
    assert recovered["status"]["messages"][0][1]["exception_message"] == "original error details"


def test_results_terminal_write_error_is_not_silently_dropped(tmp_path, monkeypatch):
    store, project, path, _ = receipt_job(tmp_path)
    before = path.read_bytes()
    def fail(*_args):
        raise OSError("disk failure")
    monkeypatch.setattr(routes, "atomic_json", fail)
    with pytest.raises(OSError, match="disk failure"):
        routes.director_project_results(store, project, lambda _: {"state": "success", "outputs": {}})
    assert path.read_bytes() == before


def test_jobs_batch_bundle_interleaving_preserves_terminal_snapshot_and_media_digest(tmp_path, monkeypatch):
    import threading
    import folder_paths
    from h3_audio_t8_pkg import director_bundle
    from test_director_batch import _write_av
    store, batch_id, _, output = _setup(tmp_path)
    project = new_project()
    batch = _create(store, batch_id, project, 5)
    path, original = _receipt(store, batch, 0)
    snapshot = {"project": project, "shot_id": original["shot_id"]}
    original.update(snapshot=snapshot, snapshot_sha256=sha(snapshot))
    atomic_json(path, original)
    video = output / "completed.mp4"
    _write_av(video)
    terminal = {"state": "success", "status": {"status_str": "success", "messages": []}, "outputs": {
        "save": {"videos": [{"filename": video.name, "subfolder": "", "type": "output"}]}}}
    entered, release = threading.Event(), threading.Event()
    def lookup(prompt):
        assert prompt == original["prompt_id"]
        entered.set()
        assert release.wait(5), "The test must release its own observation barrier"
        return deepcopy(terminal)
    monkeypatch.setattr(routes, "director_job_status", lookup)
    handlers = job_route(monkeypatch, store, SimpleNamespace(), all_routes=True)
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))
    def prepare(*_args, **_kwargs):
        assert not routes._GENERATE_LOCK.locked()
        saved = json.loads(path.read_text(encoding="utf8"))
        assert saved["media_evidence_digest"] == sha(saved["media_evidence"])
        return {"ready": True}
    monkeypatch.setattr(director_bundle, "prepare_bundle", prepare)
    async def body():
        return {"project": project}
    async def interleave():
        task = asyncio.create_task(handlers[routes.PREFIX + "/jobs/{prompt_id}"](
            SimpleNamespace(match_info={"prompt_id": original["prompt_id"]})))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            batch_task = asyncio.create_task(handlers[routes.PREFIX + "/batches/{batch_id}"](
                SimpleNamespace(match_info={"batch_id": batch_id})))
            await asyncio.sleep(0)
            bundle_task = asyncio.create_task(handlers[routes.PREFIX + "/bundles/prepare"](SimpleNamespace(json=body)))
            await asyncio.sleep(0)
            assert not batch_task.done() and not bundle_task.done()
            release.set()
            return await asyncio.gather(task, batch_task, bundle_task)
        finally:
            release.set()
    responses = asyncio.run(interleave())
    assert all(response.status == 200 for response in responses)
    saved = json.loads(path.read_text(encoding="utf8"))
    assert {key: value for key, value in saved.items() if key not in {"terminal", "media_evidence", "media_evidence_digest"}} == original
    assert saved["terminal"] == terminal
    assert saved["media_evidence_digest"] == sha(saved["media_evidence"])


@pytest.mark.parametrize("entrypoint", ("results", "batch"))
@pytest.mark.parametrize("other_completed", (False, True))
def test_every_first_observer_refuses_same_prompt_two_receipts(tmp_path, entrypoint, other_completed):
    if entrypoint == "results":
        store, project, path, original = receipt_job(tmp_path)
        output, batch_id = None, None
    else:
        store, batch_id, selected, output = _setup(tmp_path)
        project = selected["id"]
        batch = _create(store, batch_id, selected, 4)
        path, original = _receipt(store, batch, 0)
    other_path = path.with_name(str(uuid.uuid4()) + ".json")
    other = {**original, "project_id": str(uuid.uuid4()), "shot_id": str(uuid.uuid4())}
    if other_completed:
        other["terminal"] = {"state": "cancelled", "outputs": {}}
    atomic_json(other_path, other)
    before = {item: item.read_bytes() for item in (path, other_path)}
    calls = []
    def lookup(prompt):
        calls.append(prompt)
        return {"state": "cancelled", "outputs": {}}
    if entrypoint == "results":
        row = routes.director_project_results(store, project, lookup)["results"][0]
    else:
        row = batches.batch_status(store, batch_id, lookup, output)["items"][0]
        assert not row["retry_available"]
    assert row["state"] == "needs_review"
    assert calls == []
    assert all(item.read_bytes() == value for item, value in before.items())


def _observe_case(tmp_path, monkeypatch, entrypoint, *, core_epoch=None):
    calls = []

    def lookup(prompt):
        calls.append(prompt)
        return {"state": "cancelled", "outputs": {}}

    if entrypoint == "batch":
        store, batch_id, project, output = _setup(tmp_path)
        batch = _create(store, batch_id, project, 4)
        path, original = _receipt(store, batch, 0)
        def observe():
            return batches.batch_status(store, batch_id, lookup, output, core_epoch=core_epoch)
    else:
        store, project, path, original = receipt_job(tmp_path)
        if entrypoint == "results":
            def observe():
                return routes.director_project_results(store, project, lookup)
        else:
            monkeypatch.setattr(routes, "director_job_status", lookup)
            function = routes._observe_director_job if entrypoint == "jobs" else routes._record_deleted_queue_receipt
            def observe():
                return function(store, original["prompt_id"])
    return store, path, original, calls, observe


@pytest.mark.parametrize("entrypoint", ("jobs", "results", "batch", "cancel"))
@pytest.mark.parametrize("failure", ("read", "enumerate"))
def test_unreadable_request_does_not_establish_a_unique_owner(tmp_path, monkeypatch, entrypoint, failure):
    _, path, original, calls, observe = _observe_case(tmp_path, monkeypatch, entrypoint)
    other = path.with_name(str(uuid.uuid4()) + ".json")
    atomic_json(other, {**original, "project_id": str(uuid.uuid4())})
    before = {item: item.read_bytes() for item in (path, other)}
    if failure == "read":
        read = Path.read_text

        def denied_read(item, *args, **kwargs):
            if item == other:
                raise PermissionError("owned request cannot be read")
            return read(item, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", denied_read)
    else:
        entries = Path.iterdir

        def denied_entries(item):
            if item == path.parent:
                raise PermissionError("owned requests cannot be enumerated")
            return entries(item)

        monkeypatch.setattr(Path, "iterdir", denied_entries)
    with pytest.raises(PermissionError, match="owned request"):
        observe()
    assert calls == []
    assert all(item.read_bytes() == value for item, value in before.items())


@pytest.mark.parametrize("entrypoint", ("jobs", "results", "batch", "cancel"))
def test_incomplete_submitting_receipt_still_counts_as_same_prompt(tmp_path, monkeypatch, entrypoint):
    _, path, original, calls, observe = _observe_case(tmp_path, monkeypatch, entrypoint)
    other = path.with_name(str(uuid.uuid4()) + ".json")
    atomic_json(other, {"state": "submitting", "prompt_id": original["prompt_id"]})
    before = {item: item.read_bytes() for item in (path, other)}
    result = observe()
    if entrypoint == "cancel":
        assert result is False
    else:
        row = result["items"][0] if entrypoint == "batch" else result["results"][0] if entrypoint == "results" else result
        assert row["state"] == "needs_review"
        if entrypoint == "batch":
            assert row["retry_available"] is False
    assert calls == []
    assert all(item.read_bytes() == value for item, value in before.items())


@pytest.mark.parametrize("entrypoint", ("jobs", "results", "batch", "cancel"))
@pytest.mark.parametrize("unrelated", ('{broken', '[]', '{"prompt_id": []}'))
def test_malformed_unrelated_json_does_not_hide_a_healthy_receipt(tmp_path, monkeypatch, entrypoint, unrelated):
    _, path, original, calls, observe = _observe_case(tmp_path, monkeypatch, entrypoint)
    path.with_name("damaged.json").write_text(unrelated, encoding="utf8")
    result = observe()
    if entrypoint == "cancel":
        assert result is True and calls == []
    else:
        row = result["items"][0] if entrypoint == "batch" else result["results"][0] if entrypoint == "results" else result
        assert row["state"] == "cancelled" and calls == [original["prompt_id"]]
    assert json.loads(path.read_text(encoding="utf8"))["terminal"] == {"state": "cancelled", "outputs": {}}


def test_batch_terminal_write_error_propagates_without_retry_or_forged_terminal(tmp_path, monkeypatch):
    _, path, _, _, observe = _observe_case(tmp_path, monkeypatch, "batch")
    before = path.read_bytes()

    def denied(*_args):
        raise PermissionError("owned terminal write denied")

    monkeypatch.setattr(batches, "atomic_json", denied)
    with pytest.raises(PermissionError, match="owned terminal write"):
        observe()
    assert path.read_bytes() == before


@pytest.mark.parametrize("entrypoint", ("jobs", "results", "batch", "cancel"))
def test_observer_uses_one_read_snapshot_for_all_request_paths(tmp_path, monkeypatch, entrypoint):
    _, path, _, _, observe = _observe_case(tmp_path, monkeypatch, entrypoint)
    read = Path.read_text
    reads = []

    def once(item, *args, **kwargs):
        if item.parent == path.parent:
            assert item not in reads, "Observation must not reread another version of a request"
            reads.append(item)
        return read(item, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", once)
    observe()
    assert reads == [path]


@pytest.mark.parametrize("entrypoint", ("jobs", "results", "batch", "cancel"))
@pytest.mark.parametrize("terminal_state", ("success", "error", "cancelled"))
def test_duplicate_retains_original_terminal_and_media_even_after_core_epoch_changes(tmp_path, monkeypatch, entrypoint, terminal_state):
    _, path, original, calls, observe = _observe_case(tmp_path, monkeypatch, entrypoint, core_epoch="new-core")
    terminal = {"state": terminal_state, "outputs": {}, "status": {"original": True}}
    original.update(terminal=terminal, core_epoch="old-core", media_evidence=[{"sentinel": "retain"}],
                    media_evidence_digest="original digest is not laundered")
    atomic_json(path, original)
    other = path.with_name(str(uuid.uuid4()) + ".json")
    atomic_json(other, {"state": "submitting", "prompt_id": original["prompt_id"]})
    before = {item: item.read_bytes() for item in (path, other)}
    result = observe()
    if entrypoint == "cancel":
        assert result is False
    else:
        row = result["items"][0] if entrypoint == "batch" else result["results"][0] if entrypoint == "results" else result
        assert row["state"] == "needs_review" and not row.get("outputs") and not row.get("media")
        if entrypoint == "batch":
            assert row["retry_available"] is False
    assert calls == []
    assert all(item.read_bytes() == value for item, value in before.items())
