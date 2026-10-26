from __future__ import annotations

import asyncio
import json
import sys
import types
import uuid
from copy import deepcopy

import av
import folder_paths
import numpy as np
import pytest

from h3_audio_t8_pkg.director_project import ProjectStore, new_project


def _write_valid_video(path):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        sound = container.add_stream("aac", rate=48000)
        sound.layout = "mono"
        frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        samples = av.AudioFrame.from_ndarray(np.zeros((1, 1024), dtype=np.float32), format="fltp", layout="mono")
        samples.sample_rate = 48000
        for packet in sound.encode(samples):
            container.mux(packet)
        for packet in sound.encode():
            container.mux(packet)


def test_generate_request_id_returns_existing_prompt_without_double_queue(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_routes

    handlers = {}

    class Routes:
        def post(self, path):
            def register(handler):
                handlers[("POST", path)] = handler
                return handler
            return register

        get = post

    server = types.SimpleNamespace(routes=Routes())
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=server)))
    monkeypatch.setattr(director_routes, "_REGISTERED", False)
    monkeypatch.setattr(director_routes, "get_store", lambda: ProjectStore(tmp_path / "user", tmp_path / "input"))
    monkeypatch.setattr(director_routes, "build_director_generation_prompt", lambda *_args, **_kwargs: {
        "prompt": {"1": {"class_type": "Test", "inputs": {}}},
        "recipe": "director_two_pass", "seed": 1, "turbo_lora": None,
        "sampling": {"mode": "two_pass"}, "report": {"ready": True},
    })
    submitted = []

    async def queue(_prompt, _client_id, prompt_id=None):
        submitted.append(prompt_id)
        return prompt_id

    monkeypatch.setattr(director_routes, "queue_director_prompt", queue)
    director_routes.register_director_routes()
    generate = handlers[("POST", director_routes.PREFIX + "/generate")]
    assert ("POST", director_routes.PREFIX + "/sampling_ui.mjs") in handlers
    body = {
        "request_id": "e42bea1f-f961-4ef7-9c48-b181490b6f17",
        "project": {"id": "302b352d-14a3-4f90-a923-e0528e7efed0"},
        "shot_id": "7d2d9d7c-d95b-4ba2-85e5-bd3b4b27fdce", "seed": 1,
    }

    class Request:
        async def json(self):
            return body

    first = asyncio.run(generate(Request()))
    second = asyncio.run(generate(Request()))
    assert first.status == 202 and second.status == 200
    assert len(submitted) == 1
    assert json.loads(first.text)["prompt_id"] == json.loads(second.text)["prompt_id"]
    body["seed"] = 2
    rejected = asyncio.run(generate(Request()))
    assert rejected.status == 400
    assert len(submitted) == 1


def test_ambiguous_queue_exception_retains_reservation_even_if_history_is_unknown(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_routes

    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    project = {"id": str(uuid.uuid4()), "doc": {"shots": []}}
    shot_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    called = []

    async def lost_reply(_prompt, _client_id, prompt_id=None):
        called.append(prompt_id)
        raise ConnectionError("reply lost after queue may have accepted")

    monkeypatch.setattr(director_routes, "queue_director_prompt", lost_reply)
    monkeypatch.setattr(director_routes, "director_job_status", lambda _id: {"state": "unknown"})
    built = {"prompt": {"1": {"class_type": "Test", "inputs": {}}}}
    with pytest.raises(ConnectionError, match="reply lost"):
        asyncio.run(director_routes._submit_director_request_locked(
            store, project, shot_id, 1, request_id, built=built))
    reserved = store.root / "requests" / f"{request_id}.json"
    assert json.loads(reserved.read_text(encoding="utf-8"))["state"] == "submitting"
    response, code = asyncio.run(director_routes._submit_director_request_locked(
        store, project, shot_id, 1, request_id, built=built))
    assert code == 409 and response["prompt_id"] == called[0]
    assert len(called) == 1


def test_results_recover_existing_receipt_and_cache_terminal_media(tmp_path):
    from h3_audio_t8_pkg.director_routes import director_project_results

    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    project_id = "302b352d-14a3-4f90-a923-e0528e7efed0"
    shot_id = "7d2d9d7c-d95b-4ba2-85e5-bd3b4b27fdce"
    prompt_id = "e42bea1f-f961-4ef7-9c48-b181490b6f17"
    request_file = store.root / "requests" / "a.json"
    request_file.parent.mkdir(parents=True)
    request_file.write_text(json.dumps({
        "state": "queued", "prompt_id": prompt_id,
        "result": {"recipe": "director_two_pass", "report": {
            "project_id": project_id, "selection": {"shot_id": shot_id},
        }},
    }), encoding="utf-8")
    other = store.root / "requests" / "b.json"
    # This positive recovery case uses an unrelated task. Incomplete duplicate
    # prompt receipts are explicitly rejected by the H24 first-observer tests.
    other.write_text(json.dumps({"state": "queued", "prompt_id": "4eb5f3d2-6a98-43b4-b4e8-063e51d41205",
                                 "project_id": "316378f7-18b0-456d-bc84-ac089e141fc7"}), encoding="utf-8")
    media = {"12": {"images": [{"filename": "film_00001_.mp4", "subfolder": "T8_Director\\abc", "type": "output"}]}}
    calls = []

    def status(queried):
        calls.append(queried)
        return {"state": "success", "outputs": media}

    results = director_project_results(store, project_id, status)["results"]
    assert len(results) == 1
    assert results[0]["shot_id"] == shot_id
    assert results[0]["outputs"] == media
    assert results[0]["state"] == "success"
    assert calls == [prompt_id]
    assert director_project_results(store, project_id, status)["results"] == results
    assert calls == [prompt_id]  # persisted media remains discoverable after Core restart


def test_results_recover_pre_receipt_video_from_exact_project_and_shot_prefix(tmp_path):
    from h3_audio_t8_pkg.director_routes import director_project_results

    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    project_id = "302b352d-14a3-4f90-a923-e0528e7efed0"
    shot_id = "7d2d9d7c-d95b-4ba2-85e5-bd3b4b27fdce"
    media_dir = tmp_path / "output" / "T8_Director" / project_id[:8]
    media_dir.mkdir(parents=True)
    (media_dir / f"{shot_id[:8]}_00001_.mp4").write_bytes(b"video")
    (media_dir / "ffffffff_00001_.mp4").write_bytes(b"other shot")
    records = director_project_results(store, project_id, lambda _id: {},
                                       shot_ids=[shot_id], output_root=tmp_path / "output")["results"]
    assert len(records) == 1
    assert records[0]["recovered_by"] == "project_and_shot_output_prefix"
    assert records[0]["outputs"]["legacy"]["images"][0]["filename"] == f"{shot_id[:8]}_00001_.mp4"


def test_durable_batch_route_reconciles_two_shots_without_double_queue(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_routes

    handlers = {}

    class Routes:
        def post(self, path):
            def register(handler):
                handlers[("POST", path)] = handler
                return handler
            return register

        get = post

    server = types.SimpleNamespace(routes=Routes())
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=server)))
    monkeypatch.setattr(director_routes, "_REGISTERED", False)
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    monkeypatch.setattr(director_routes, "get_store", lambda: store)
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))
    monkeypatch.setattr(director_routes, "build_director_generation_prompt", lambda project, shot_id, _store, *, seed: {
        "prompt": {"1": {"class_type": "Test", "inputs": {"seed": seed}}},
        "recipe": "director_frozen", "seed": seed, "turbo_lora": None,
        "sampling": {"mode": "single"}, "report": {"ready": True},
        "shot": {"time": {"delivery_trim_frames": 1, "requested_seconds": 1 / 24},
                 "canvas": {"width": 32, "height": 32}},
    })
    statuses, submitted = {}, []

    async def queue(_prompt, _client_id, prompt_id=None):
        submitted.append(prompt_id)
        statuses[prompt_id] = {"state": "running"}
        return prompt_id

    monkeypatch.setattr(director_routes, "queue_director_prompt", queue)
    monkeypatch.setattr(director_routes, "director_job_status", lambda prompt_id: statuses.get(prompt_id, {"state": "unknown"}))
    director_routes.register_director_routes()
    prefix = director_routes.PREFIX
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "first"
    second = deepcopy(project["doc"]["shots"][0])
    second.update(id=str(uuid.uuid4()), name="second", simplePrompt="second")
    project["doc"]["shots"].append(second)
    batch_id = str(uuid.uuid4())

    class Request:
        def __init__(self, body=None, batch_id=None):
            self.body = body or {}
            self.match_info = {"batch_id": batch_id} if batch_id else {}

        async def json(self):
            return self.body

    create = handlers[("POST", prefix + "/batches")]
    read = handlers[("POST", prefix + "/batches/{batch_id}")]
    next_shot = handlers[("POST", prefix + "/batches/{batch_id}/continue")]
    start = asyncio.run(create(Request({"batch_id": batch_id, "project": project, "seed": 77})))
    assert start.status == 200
    frozen = asyncio.run(read(Request(batch_id=batch_id)))
    assert [row["state"] for row in json.loads(frozen.text)["items"]] == ["not_submitted"] * 2
    async def two_tabs_at_once():
        return await asyncio.gather(
            next_shot(Request({"client_id": "tab-A"}, batch_id)),
            next_shot(Request({"client_id": "tab-B"}, batch_id)),
        )

    competing = asyncio.run(two_tabs_at_once())
    assert sorted(response.status for response in competing) == [202, 409]
    abandoned_id = json.loads(next(response.text for response in competing if response.status == 202))["prompt_id"]
    assert len(submitted) == 1
    # Simulate a restarted *owned* Core with no old history. Reading status
    # cannot enqueue; explicit confirmation abandons only this shot attempt.
    statuses[abandoned_id] = {"state": "unknown"}
    monkeypatch.setattr(director_routes, "_CORE_EPOCH", str(uuid.uuid4()))
    stale = json.loads(asyncio.run(read(Request(batch_id=batch_id))).text)
    assert stale["items"][0]["state"] == "unknown"
    assert stale["items"][0]["retry_available"] is True
    retry = handlers[("POST", prefix + "/batches/{batch_id}/retry")]
    restarted = asyncio.run(retry(Request({"confirm_abandoned": True}, batch_id)))
    assert restarted.status == 200
    assert json.loads(restarted.text)["attempt"] == 1
    assert len(submitted) == 1, "read/explicit retry reservation is not a GPU submission"
    first_again = asyncio.run(next_shot(Request({"client_id": "tab-A"}, batch_id)))
    assert first_again.status == 202
    first_id = json.loads(first_again.text)["prompt_id"]
    assert first_id != abandoned_id
    assert len(submitted) == 2
    video1 = output / "one.mp4"
    _write_valid_video(video1)
    statuses[first_id] = {"state": "success", "outputs": {"save": {"images": [
        {"filename": video1.name, "subfolder": "", "type": "output"}
    ]}}}
    assert json.loads(asyncio.run(read(Request(batch_id=batch_id))).text)["next_index"] == 1
    second_response = asyncio.run(next_shot(Request({"client_id": "tab-B"}, batch_id)))
    assert second_response.status == 202 and len(submitted) == 3
    second_id = json.loads(second_response.text)["prompt_id"]
    video2 = output / "two.mp4"
    _write_valid_video(video2)
    statuses[second_id] = {"state": "success", "outputs": {"save": {"images": [
        {"filename": video2.name, "subfolder": "", "type": "output"}
    ]}}}
    assert json.loads(asyncio.run(read(Request(batch_id=batch_id))).text)["complete"] is True
    statuses.clear()  # Core restart: durable terminal receipts still recover.
    assert json.loads(asyncio.run(read(Request(batch_id=batch_id))).text)["complete"] is True
    assert len(submitted) == 3
