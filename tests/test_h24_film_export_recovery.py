"""Three real clips, bounded export faults and an actual worker restart."""
import asyncio
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import time
import types
import uuid

import av
import numpy as np
import pytest

from h3_audio_t8_pkg import director_film_export as exporter
from h3_audio_t8_pkg.director_film import prepare_film
from h3_audio_t8_pkg.director_project import file_sha
from test_director_film import setup, video


def case(tmp_path):
    store, output, project, records = setup(tmp_path)
    third = {**deepcopy(project["doc"]["shots"][0]), "id": str(uuid.uuid4()),
             "name": "third", "adoptedResultId": "version-third"}
    project["doc"]["shots"].append(third)
    records.append({"shot_id": third["id"], "prompt_id": third["adoptedResultId"], "state": "success",
                    "outputs": {"save": {"images": [{"filename": "2.mp4", "type": "output"}]}}})
    for index, (value, frequency, frames) in enumerate(((40, 440, 12), (100, 880, 12), (180, 1320, 17))):
        video(output / f"{index}.mp4", value, tone=frequency, frames=frames)
    film = prepare_film(store, project, records, output)
    assert film["ready"] and len(film["entries"]) == 3
    options = {"policy": exporter.POLICY, "confirm_encoding": True, "width": 64, "height": 32}
    return store, output, project, film, options


def reserve(store, project, film, options):
    job = str(uuid.uuid4())
    assert exporter.reserve_export(store, project["id"], film["id"], job, options)[1]
    return job


def run(store, project, film, job):
    return exporter.run_export(store, project["id"], film["id"], job)


def test_three_clip_export_keeps_third_tail_and_sample_precision(tmp_path):
    store, output, project, film, options = case(tmp_path)
    originals = {path: file_sha(path) for path in output.glob("*.mp4")}
    job = reserve(store, project, film, options)
    assert run(store, project, film, job)["state"] == "success"
    path = exporter.export_file(store, project["id"], film["id"], job)
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 41
        assert [Fraction(f.pts) * f.time_base for f in frames] == [Fraction(i, 24) for i in range(41)]
        assert frames[0].to_ndarray(format="rgb24").mean() == pytest.approx(40, abs=3)
        assert frames[24].to_ndarray(format="rgb24").mean() == pytest.approx(180, abs=3)
        assert frames[-1].to_ndarray(format="rgb24").mean() == pytest.approx(196, abs=3)
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        audio_frames = list(container.decode(audio=0))
        cursor = Fraction(0)
        for frame in audio_frames:
            assert Fraction(frame.pts) * frame.time_base == cursor
            cursor += Fraction(frame.samples, frame.sample_rate)
        assert Fraction(stream.duration) * stream.time_base * 48000 == 82000
        samples = np.concatenate([frame.to_ndarray() for frame in audio_frames], axis=1)
        tail = samples.mean(axis=0)[round(1.5 * 48000):round(1.65 * 48000)]
        peak = np.fft.rfftfreq(tail.size, 1 / 48000)[abs(np.fft.rfft(tail)).argmax()]
        assert abs(peak - 1320) < 15
    assert all(file_sha(p) == digest for p, digest in originals.items())


@pytest.mark.parametrize("failure", ("second_clip", "concat", "publish", "receipt"))
def test_faults_preserve_originals_old_success_and_allow_explicit_new_export(tmp_path, monkeypatch, failure):
    store, output, project, film, options = case(tmp_path)
    old = reserve(store, project, film, options)
    assert run(store, project, film, old)["state"] == "success"
    old_path = exporter.export_file(store, project["id"], film["id"], old)
    subtitle = store.root / "film_exports" / "old-captions.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nold subtitle\n", encoding="utf8")
    protected = {p: file_sha(p) for p in [*output.glob("*.mp4"), old_path, subtitle]}
    job = reserve(store, project, film, options)
    original_run, original_replace, original_write = exporter._run, Path.replace, exporter.atomic_json
    calls = []

    def encode(arguments, log):
        name = Path(arguments[-1]).name
        calls.append(name)
        if (failure == "second_clip" and name == "clip-0001.mov") or (failure == "concat" and name == "film.mp4"):
            raise RuntimeError("owned injected encoder failure")
        return original_run(arguments, log)

    def replace(source, destination):
        if failure == "publish" and source.name == "film.mp4":
            raise PermissionError("owned injected publication failure")
        return original_replace(source, destination)

    def write(path, value):
        if failure == "receipt" and value.get("state") == "success":
            raise PermissionError("owned injected receipt failure")
        return original_write(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(exporter, "_run", encode)
        patch.setattr(Path, "replace", replace)
        patch.setattr(exporter, "atomic_json", write)
        if failure == "receipt":
            with pytest.raises(PermissionError, match="owned injected receipt"):
                run(store, project, film, job)
        else:
            result = run(store, project, film, job)
            assert result["state"] == "error" and "owned injected" in result["error"]
    with pytest.raises(ValueError, match="尚未成功"):
        exporter.export_file(store, project["id"], film["id"], job)
    assert all(file_sha(p) == digest for p, digest in protected.items())
    failed_file = store.root / "film_exports" / f"{job}.mp4"
    assert failed_file.exists() is (failure == "receipt")
    # Persisted incomplete/error jobs never silently restart encoding.
    assert not exporter.reserve_export(store, project["id"], film["id"], job, options)[1]
    with monkeypatch.context() as patch:
        patch.setattr(exporter, "_run", lambda *_a: pytest.fail("same job re-encoded"))
        assert run(store, project, film, job)["state"] in {"running", "error"}
    new = reserve(store, project, film, options)
    assert run(store, project, film, new)["state"] == "success"
    assert all(file_sha(p) == digest for p, digest in protected.items())
    assert calls.count("clip-0000.mov") == 1


def test_actual_worker_interruption_reopens_without_reencoding_old_job(tmp_path, monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from h3_audio_t8_pkg import director_routes
    import folder_paths

    store, output, project, film, options = case(tmp_path)
    job = reserve(store, project, film, options)
    before = {p: file_sha(p) for p in output.glob("*.mp4")}
    worker = Path(__file__).with_name("h24_export_process_worker.py")
    command = [sys.executable, str(worker), "--core", str(Path(folder_paths.__file__).parent),
        "--user", str(store.root.parent), "--input", str(store.input_root),
        "--project", project["id"], "--film", film["id"], "--job", job]
    ready = tmp_path / "paused-worker.json"
    process = subprocess.Popen([*command, "--pause", str(ready)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        deadline = time.monotonic() + 30
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert ready.exists(), process.communicate(timeout=5)
        checkpoint = json.loads(ready.read_text(encoding="utf8"))
        assert checkpoint["pid"] == process.pid and checkpoint["calls"] == ["clip-0000.mov"]
        assert checkpoint["before_second_encoder_started"] and not checkpoint["cuda_initialized"]
    finally:
        if process.poll() is None:
            process.terminate()  # Exact owned handle, paused before any next child.
        process.communicate(timeout=10)
    assert process.poll() is not None
    disk = exporter.export_status(store, project["id"], film["id"], job)
    assert disk["state"] == "running" and disk["done"] == 1
    fresh = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    reopened = json.loads(fresh.stdout.strip().splitlines()[-1])
    assert reopened["job"] == disk and reopened["calls"] == [] and not reopened["cuda_initialized"]

    routes = web.RouteTableDef()
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(
        instance=types.SimpleNamespace(routes=routes))))
    monkeypatch.setattr(director_routes, "_REGISTERED", False)
    monkeypatch.setattr(director_routes, "_FILM_EXPORT_TASKS", {})
    monkeypatch.setattr(director_routes, "get_store", lambda: store)
    director_routes.register_director_routes()

    async def inspect():
        app = web.Application()
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as client:
            base = director_routes.PREFIX + f"/films/{project['id']}/{film['id']}/exports/{job}"
            response = await client.get(base)
            assert response.status == 200 and (await response.json())["state"] == "interrupted"
            response = await client.get(base + "/file")
            assert response.status == 400

    asyncio.run(inspect())
    assert all(file_sha(p) == digest for p, digest in before.items())
    new = reserve(store, project, film, options)
    assert run(store, project, film, new)["state"] == "success"
