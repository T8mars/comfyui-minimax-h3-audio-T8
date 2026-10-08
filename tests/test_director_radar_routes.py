import asyncio
import json
from pathlib import Path
import sys
import types

import pytest

from h3_audio_t8_pkg.director_project import ProjectStore, new_project, sha
from h3_audio_t8_pkg import director_routes


@pytest.fixture
def radar_routes(tmp_path, monkeypatch):
    handlers = {}

    class Routes:
        def post(self, path):
            def register(handler):
                handlers[path] = handler
                return handler

            return register

        get = post

    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    monkeypatch.setitem(
        sys.modules,
        "server",
        types.SimpleNamespace(
            PromptServer=types.SimpleNamespace(
                instance=types.SimpleNamespace(routes=Routes())
            )
        ),
    )
    monkeypatch.setattr(director_routes, "_REGISTERED", False)
    monkeypatch.setattr(director_routes, "get_store", lambda: store)

    def no_queue(*_args, **_kwargs):
        raise AssertionError("Evidence/rules must never queue inference")

    monkeypatch.setattr(director_routes, "queue_director_prompt", no_queue)
    assert director_routes.register_director_routes()
    return store, handlers


def dispatch(handler, body, *, length=1000, match_info=None):
    class Request:
        content_length = length

        async def json(self):
            return body

    Request.match_info = match_info or {}
    response = asyncio.run(handler(Request()))
    return response.status, json.loads(response.body)


def test_recipe_card_route_is_explicit_bounded_and_never_saves_or_queues(radar_routes, monkeypatch):
    from h3_audio_t8_pkg import model_recipe_cards

    store, handlers = radar_routes
    calls = []

    def inspect(category, selection):
        calls.append((category, selection))
        return {"automatic_changes": False, "saved": False, "queued": False}

    monkeypatch.setattr(model_recipe_cards, "inspect_installed_recipe", inspect)
    handler = handlers[director_routes.PREFIX + "/model-recipe/inspect"]
    assert dispatch(handler, {"category": "loras", "selection": "用户.safetensors"})[0] == 200
    assert calls == [("loras", "用户.safetensors")]
    assert dispatch(handler, {"category": "loras", "selection": "auto", "path": "C:/outside"})[0] == 400
    assert dispatch(handler, {}, length=8193)[0] == 400
    assert dispatch(handler, {}, length=None)[0] == 400
    assert store.list() == [] and len(calls) == 1


def test_readonly_candidate_inspection_is_bound_to_the_current_draft_without_store_writes(radar_routes, monkeypatch):
    from copy import deepcopy
    import hashlib
    from h3_audio_t8_pkg.director_radar import apply_operation

    store, handlers = radar_routes
    project = new_project()
    project = apply_operation(project, project['current'], 'candidate_create', {'facts': False, 'intent': True})
    before = deepcopy(project)
    raw = json.dumps(project, ensure_ascii=False)
    body = dict(project=project, shot_id=project['current'], base_json=raw,
                base_sha256=hashlib.sha256(raw.encode('utf8')).hexdigest())
    def no_store():
        raise AssertionError('Read-only draft inspection must not read/write the store or prepare images')
    monkeypatch.setattr(director_routes, 'get_store', no_store)
    handler = handlers[director_routes.PREFIX + '/radar/inspect']
    code, result = dispatch(handler, body)
    assert code == 200
    assert result['candidate']['status'] == 'pending'
    assert result['candidate']['changed_components'] == []
    assert result['saved'] is result['queued'] is result['project_modified'] is False
    assert result['time_plan']['fps'] == 24
    assert result['time_plan']['generated_seconds'] == result['time_plan']['aligned_frames'] / 24
    assert project == before and store.list() == []
    body['base_sha256'] = '0' * 64
    assert dispatch(handler, body)[0] == 409
    body['base_sha256'] = hashlib.sha256(raw.encode('utf8')).hexdigest()
    body['base_json'] = '{}'
    assert dispatch(handler, body)[0] == 409
    assert dispatch(handler, body, length=None)[0] == 400
    assert dispatch(handler, body, length=2 * 1024**2 + 1)[0] == 400
    assert project == before and store.list() == []


def test_session_relative_radar_module_has_actual_native_core_static_route(radar_routes):
    _store, handlers = radar_routes
    handler = handlers[director_routes.PREFIX + "/radar_state.mjs"]
    response = asyncio.run(handler(types.SimpleNamespace()))
    expected = Path(director_routes.__file__).resolve().parents[1] / "web/director/radar_state.mjs"
    assert response.status == 200
    assert response._path.resolve(strict=True) == expected.resolve(strict=True)
    session = (expected.parent / "session.mjs").read_text(encoding="utf8")
    assert "'./radar_state.mjs'" in session
    assert expected.read_bytes()


def test_readonly_reference_plan_redacts_paths_and_separates_video_audio_ordinals(radar_routes, monkeypatch):
    import hashlib
    import uuid

    _store, handlers = radar_routes
    project = new_project()
    aid = str(uuid.uuid4())
    # Deliberately declared metadata only: this endpoint must not read files or
    # grant actual encoding/file-content certification to this untrusted draft.
    project['assets'] = [dict(id=aid, name='declared.mp4', kind='video', size=1,
        server_path=f't8_director/{aid}/declared.mp4', sha256='a' * 64,
        width=512, height=288, duration=5, has_audio=True)]
    project['doc']['shots'][0].update(mode='refs', refs=[aid], simplePrompt='使用 @video1')
    def no_store():
        raise AssertionError('Plan inspector must not verify/read files or preprocess images')
    monkeypatch.setattr(director_routes, 'get_store', no_store)
    raw = json.dumps(project, ensure_ascii=False)
    code, report = dispatch(handlers[director_routes.PREFIX + '/radar/inspect'],
        dict(project=project, shot_id=project['current'], base_json=raw,
             base_sha256=hashlib.sha256(raw.encode()).hexdigest()))
    assert code == 200
    assert [row['native'] for row in report['reference_slots']] == ['<Video 1>', '<Audio 1>']
    assert [row['role'] for row in report['reference_slots']] == ['ref_video', 'ref_video_audio']
    assert all(set(row) == {'asset_id', 'role', 'native', 'sha256'} for row in report['reference_slots'])
    assert report['reference_aliases'] == [{'alias': '@video1', 'asset_id': aid, 'native': '<Video 1>'}]
    assert report['scope'] == 'current_draft_plan_not_actual_encoding_or_runtime_cache'
    assert report['source_time_map'] is None
    assert report['saved'] is report['queued'] is report['project_modified'] is False


def test_actual_guarded_radar_route_preserves_unsaved_draft_and_never_queues(
    radar_routes,
):
    store, handlers = radar_routes
    project = new_project()
    sid = project["current"]
    project["doc"]["shots"][0]["simplePrompt"] = "原文"
    body = {
        "project": project,
        "base_sha256": sha(project),
        "shot_id": sid,
        "operation": "skill_create",
        "value": {"name": "镜头", "text": "slow zoom", "scope": "shot"},
    }
    code, result = dispatch(handlers[director_routes.PREFIX + "/radar/operate"], body)
    assert code == 200 and result["saved"] is result["queued"] is False
    assert result["project"]["doc"]["shots"][0]["simplePrompt"] == "原文"
    assert store.list() == []
    body["base_sha256"] = "0" * 64
    code, result = dispatch(handlers[director_routes.PREFIX + "/radar/operate"], body)
    assert code == 409 and result["code"] == "revision_conflict"


def test_actual_route_rejects_old_saved_revision_large_request_and_json_substitution(
    radar_routes,
):
    store, handlers = radar_routes
    project = new_project()
    handler = handlers[director_routes.PREFIX + "/radar/operate"]
    body = {
        "project": project,
        "base_sha256": sha(project),
        "shot_id": project["current"],
        "operation": "candidate_revert",
        "value": {},
    }
    assert dispatch(handler, body, length=2 * 1024**2 + 1)[0] == 400
    body["base_json"] = "{}"
    assert dispatch(handler, body)[0] == 409
    body.pop("base_json")
    store.save(project, 0)
    assert dispatch(handler, body)[0] == 409


def test_client_raw_json_hash_is_checked_and_utf8_exact(radar_routes):
    import hashlib

    _store, handlers = radar_routes
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "汉字🙂\r\n"
    raw = json.dumps(project, ensure_ascii=False, separators=(",", ":"))
    body = {
        "project": project,
        "base_json": raw,
        "base_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "shot_id": project["current"],
        "operation": "candidate_create",
        "value": {},
    }
    code, result = dispatch(handlers[director_routes.PREFIX + "/radar/operate"], body)
    assert code == 200
    assert (
        result["project"]["doc"]["shots"][0]["promptCandidate"]["text"] == "汉字🙂\r\n"
    )


def test_actual_source_pts_measured_in_isolated_worker_and_bad_declared_clock_rejected(
    radar_routes,
):
    import uuid
    from test_director_idempotency import _write_valid_video
    from h3_audio_t8_pkg.director_radar import PACKET_SCHEMA

    store, handlers = radar_routes
    aid = str(uuid.uuid4())
    path = store.input_root / "t8_director" / aid / "source.mp4"
    path.parent.mkdir(parents=True)
    _write_valid_video(path)
    asset = store.register_asset(path, aid, "source.mp4")
    route = handlers[director_routes.PREFIX + "/radar/assets/{asset_id}/timing"]
    code, timing = dispatch(route, {}, match_info={"asset_id": aid})
    assert (
        code == 200
        and timing["fully_decoded"] is False
        and timing["sha256"] == asset["sha256"]
    )
    stream = next(row for row in timing["streams"] if row["kind"] == "video")
    packet = {
        "schema": PACKET_SCHEMA,
        "id": str(uuid.uuid4()),
        "revision": 1,
        "source": {
            "asset_id": aid,
            "sha256": asset["sha256"],
            "stream": stream["index"],
            "timebase": stream["timebase"],
            "start_pts": stream["start_pts"],
            "end_pts": stream["end_pts"],
        },
        "claims": [],
    }
    project = new_project()
    project["assets"] = [asset]
    body = {
        "project": project,
        "base_sha256": sha(project),
        "shot_id": project["current"],
        "operation": "evidence_import",
        "value": packet,
    }
    handler = handlers[director_routes.PREFIX + "/radar/operate"]
    assert dispatch(handler, body)[0] == 200
    packet["source"]["timebase"] = {"num": 1, "den": stream["timebase"]["den"] + 1}
    code, result = dispatch(handler, body)
    assert code == 400 and "timebase" in result["error"]
    packet["source"]["timebase"] = stream["timebase"]
    packet["source"]["end_pts"] += 1
    assert dispatch(handler, body)[0] == 400
