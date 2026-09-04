from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from h3_audio_t8_pkg import comfy_entrypoint
from h3_audio_t8_pkg.face_refine_window_advanced import (
    build_face_refine_window_plan,
    extract_face_refine_window,
)
from h3_audio_t8_pkg.face_refine_window_studio_advanced import (
    STUDIO_CHAIN_PREFIX,
    commit_face_refine_window_studio,
    compose_face_refine_window_studio,
    face_refine_window_studio_position,
    load_face_refine_window_studio,
    prepare_face_refine_window_studio,
)
from h3_audio_t8_pkg.long_video_background import BackgroundJobManager
from h3_audio_t8_pkg.nodes_face_refine_window_studio_advanced import (
    FACE_REFINE_WINDOW_STUDIO_ADVANCED_NODE_CLASSES,
)


def _bind_output(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))


def _fixture():
    frames = torch.linspace(0.0, 1.0, 124 * 4 * 5 * 3, dtype=torch.float32).reshape(
        124, 4, 5, 3
    )
    plan, _mask, count, _report = build_face_refine_window_plan(
        frames,
        24.0,
        "0-3,80-83",
        "frames_inclusive",
        0,
        0,
        90,
        362,
        1.0,
        "reject",
        "reject",
        True,
    )
    assert count == 2
    return frames, plan


def _candidate(frames, plan, index, value):
    render, _audio, mapping, *_rest = extract_face_refine_window(
        frames, plan, index, "reject"
    )
    candidate = render.clone()
    mask = torch.zeros(render.shape[:3], dtype=render.dtype)
    allowed = {
        frame
        for start, end in mapping["window"]["repair_ranges_abs"]
        for frame in range(start, end + 1)
    }
    for record in mapping["frame_map"]:
        source = record["source_frame"]
        if source in allowed:
            relative = int(record["render_frame"])
            candidate[relative, 1:3, 1:4] = value
            mask[relative, 1:3, 1:4] = 1.0
    return candidate, mask, mapping


def test_studio_prepare_is_source_bound_and_idempotent(monkeypatch, tmp_path):
    _bind_output(monkeypatch, tmp_path)
    frames, plan = _fixture()
    chain_id, index, path, complete, manifest, report = prepare_face_refine_window_studio(
        plan, "studio-a"
    )
    assert chain_id == f"{STUDIO_CHAIN_PREFIX}studio-a"
    assert index == 0
    assert complete is False
    assert Path(path).is_file()
    assert json.loads(report)["resolved_window_count"] == 0
    assert manifest["source_overwrite_allowed"] is False

    repeated = prepare_face_refine_window_studio(plan, "studio-a")
    assert repeated[4] == manifest

    changed = frames.clone()
    changed[0, 0, 0, 0] += 0.01
    other_plan = build_face_refine_window_plan(
        changed, 24.0, "0-3,80-83", "frames_inclusive", 0, 0, 90, 362,
        1.0, "reject", "reject", True
    )[0]
    with pytest.raises(ValueError, match="different source or window plan"):
        prepare_face_refine_window_studio(other_plan, "studio-a")


def test_studio_nodes_are_append_only_after_frozen_p0_nodes():
    import asyncio

    ids = [
        node.define_schema().node_id
        for node in FACE_REFINE_WINDOW_STUDIO_ADVANCED_NODE_CLASSES
    ]
    assert ids == [
        "MiniMaxH3FaceRefineWindowStudioStartT8Advanced",
        "MiniMaxH3FaceRefineWindowStudioCommitT8Advanced",
        "MiniMaxH3FaceRefineWindowStudioComposeT8Advanced",
    ]
    registered = [
        node.define_schema().node_id
        for node in asyncio.run(comfy_entrypoint().get_node_list())
    ]
    assert registered[292:295] == [
        "MiniMaxH3FaceRefineWindowPlanT8Advanced",
        "MiniMaxH3FaceRefineWindowExtractT8Advanced",
        "MiniMaxH3FaceRefineManualReviewT8Advanced",
    ]
    assert registered[295:] == ids


def test_preview_is_non_mutating_then_accept_reject_and_compose(monkeypatch, tmp_path):
    _bind_output(monkeypatch, tmp_path)
    frames, plan = _fixture()
    candidate0, mask0, mapping0 = _candidate(frames, plan, 0, 0.125)
    preview = commit_face_refine_window_studio(
        frames, candidate0, mask0, mapping0, plan, "studio-b",
        "preview_only", "", False, 0
    )
    assert preview[4] is False
    assert preview[6] == 0
    assert torch.equal(preview[1], frames)
    assert face_refine_window_studio_position(f"{STUDIO_CHAIN_PREFIX}studio-b") == (
        0,
        False,
    )

    accepted = commit_face_refine_window_studio(
        frames, candidate0, mask0, mapping0, plan, "studio-b",
        "accept_selected", "", True, 0
    )
    assert accepted[4] is True
    assert accepted[6:8] == (1, False)
    manifest = load_face_refine_window_studio(f"{STUDIO_CHAIN_PREFIX}studio-b")
    assert manifest["windows"][0]["state"] == "accepted"
    assert manifest["windows"][0]["overlay_sha256"]

    with pytest.raises(ValueError, match="expects window 1"):
        commit_face_refine_window_studio(
            frames, candidate0, mask0, mapping0, plan, "studio-b",
            "accept_selected", "", True, 0
        )

    candidate1, mask1, mapping1 = _candidate(frames, plan, 1, 0.875)
    rejected = commit_face_refine_window_studio(
        frames, candidate1, mask1, mapping1, plan, "studio-b",
        "reject", "", False, 0
    )
    assert rejected[4] is True
    assert rejected[6:8] == (2, True)
    assert face_refine_window_studio_position(f"{STUDIO_CHAIN_PREFIX}studio-b") == (
        2,
        True,
    )

    result, combined, complete, report = compose_face_refine_window_studio(
        frames, plan, "studio-b"
    )
    assert complete is True
    assert json.loads(report)["accepted_window_count"] == 1
    assert torch.all(result[0:4, 1:3, 1:4] == 0.125)
    assert torch.equal(result[80:84], frames[80:84])
    outside = combined <= 0
    assert torch.equal(
        result[outside.unsqueeze(-1).expand_as(result)],
        frames[outside.unsqueeze(-1).expand_as(frames)],
    )
    assert torch.equal(frames, _fixture()[0])


def test_unconfirmed_accept_fails_closed_without_manifest_advance(monkeypatch, tmp_path):
    _bind_output(monkeypatch, tmp_path)
    frames, plan = _fixture()
    candidate, mask, mapping = _candidate(frames, plan, 0, 0.25)
    with pytest.raises(ValueError, match="was not accepted"):
        commit_face_refine_window_studio(
            frames, candidate, mask, mapping, plan, "studio-c",
            "accept_selected", "", False, 0
        )
    assert face_refine_window_studio_position(f"{STUDIO_CHAIN_PREFIX}studio-c") == (
        0,
        False,
    )


def test_overlay_first_fault_leaves_pending_manifest_and_retry_succeeds(monkeypatch, tmp_path):
    _bind_output(monkeypatch, tmp_path)
    frames, plan = _fixture()
    candidate, mask, mapping = _candidate(frames, plan, 0, 0.33)
    with pytest.raises(RuntimeError, match="fault injection"):
        commit_face_refine_window_studio(
            frames, candidate, mask, mapping, plan, "studio-fault",
            "accept_selected", "", True, 0,
            fault_inject_after_overlay=True,
        )
    chain_id = f"{STUDIO_CHAIN_PREFIX}studio-fault"
    manifest = load_face_refine_window_studio(chain_id)
    assert manifest["windows"][0]["state"] == "pending"
    assert face_refine_window_studio_position(chain_id) == (0, False)

    committed = commit_face_refine_window_studio(
        frames, candidate, mask, mapping, plan, "studio-fault",
        "accept_selected", "", True, 0
    )
    assert committed[4] is True
    assert committed[6] == 1


def test_compose_rejects_tampered_overlay(monkeypatch, tmp_path):
    _bind_output(monkeypatch, tmp_path)
    frames, plan = _fixture()
    candidate, mask, mapping = _candidate(frames, plan, 0, 0.44)
    commit_face_refine_window_studio(
        frames, candidate, mask, mapping, plan, "studio-tamper",
        "accept_selected", "", True, 0
    )
    chain_id = f"{STUDIO_CHAIN_PREFIX}studio-tamper"
    manifest = load_face_refine_window_studio(chain_id)
    root = Path(tmp_path) / "minimax_h3_t8_long_video" / chain_id
    overlay = root / manifest["windows"][0]["overlay_path"]
    overlay.write_bytes(overlay.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="file/hash"):
        compose_face_refine_window_studio(frames, plan, "studio-tamper")


def test_compose_rejects_self_consistent_overlay_outside_source(monkeypatch, tmp_path):
    _bind_output(monkeypatch, tmp_path)
    frames, plan = _fixture()
    candidate, mask, mapping = _candidate(frames, plan, 0, 0.45)
    commit_face_refine_window_studio(
        frames, candidate, mask, mapping, plan, "studio-bounds",
        "accept_selected", "", True, 0
    )
    import h3_audio_t8_pkg.face_refine_window_studio_advanced as studio

    original = studio._load_overlay

    def outside(root, item):
        crop, overlay_mask, bbox = original(root, item)
        return crop, overlay_mask, (-1, *bbox[1:])

    monkeypatch.setattr(studio, "_load_overlay", outside)
    with pytest.raises(ValueError, match="outside the source"):
        compose_face_refine_window_studio(frames, plan, "studio-bounds")


class _Runtime:
    def __init__(self, prompt_id):
        self.prompt_id = prompt_id
        self.queued = []
        self.locations = {prompt_id: "running"}

    def current_prompt_id(self):
        return self.prompt_id

    @staticmethod
    def current_client_id():
        return "client"

    def queue_prompt(self, prompt, client_id):
        prompt_id = f"queued-{len(self.queued) + 1}"
        self.queued.append((prompt_id, prompt, client_id))
        self.locations[prompt_id] = "queued"
        return prompt_id

    def prompt_location(self, prompt_id):
        return self.locations.get(prompt_id, "missing")

    @staticmethod
    def history_record(_prompt_id):
        return None

    def cancel_prompt(self, prompt_id):
        self.locations[prompt_id] = "missing"
        return {"prompt_id": prompt_id, "deleted_from_queue": True}

    @staticmethod
    def request_release(_release_policy):
        return None


def test_shared_background_manager_resumes_from_studio_manifest(monkeypatch, tmp_path):
    _bind_output(monkeypatch, tmp_path)
    frames, plan = _fixture()
    chain_id = prepare_face_refine_window_studio(plan, "studio-bg")[0]
    binding = {
        "kind": "face-window-test",
        "window_plan_sha256": plan["plan_sha256"],
    }
    runtime = _Runtime("prompt-0")
    manager = BackgroundJobManager(runtime, start_monitors=False)
    state = manager.attach_prompt(
        chain_id, {"1": {"class_type": "Test", "inputs": {}}}, "1", 1, 0,
        "clear_execution_cache", binding_metadata=binding
    )
    candidate, mask, mapping = _candidate(frames, plan, 0, 0.5)
    committed = commit_face_refine_window_studio(
        frames, candidate, mask, mapping, plan, "studio-bg",
        "accept_selected", "", True, 0
    )
    advanced = manager.segment_accepted(
        state["job_id"], candidate_index=0, candidate_json_path=committed[5],
        manifest_path=committed[5], accepted_count=committed[6],
        is_final_segment=False,
    )
    assert advanced["accepted_count"] == 1
    assert [item[0] for item in runtime.queued] == ["queued-1"]
    manager.close()

    restarted_runtime = _Runtime("prompt-restarted")
    restarted = BackgroundJobManager(restarted_runtime, start_monitors=False)
    recovered = restarted.attach_prompt(
        chain_id, {"1": {"class_type": "Test", "inputs": {}}}, "1", 1, 0,
        "clear_execution_cache", binding_metadata=binding
    )
    assert recovered["accepted_count"] == 1
    assert recovered["current_segment_index"] == 1
    assert recovered["previous_job_id"] == state["job_id"]
    restarted.close()


def test_studio_api_and_frontend_workflows_are_serial_source_audio_routes():
    from tools.build_face_refine_window_studio_workflow import (
        API_OUTPUT,
        FRONTEND_OUTPUT,
        build_api,
        build_frontend,
    )

    api = json.loads(API_OUTPUT.read_text(encoding="utf-8"))
    assert api == build_api()
    assert api["28"]["class_type"] == (
        "MiniMaxH3FaceRefineWindowStudioStartT8Advanced"
    )
    assert api["28"]["inputs"]["execution_mode"] == "review_only"
    assert api["5"]["inputs"]["window_index"] == ["28", 0]
    assert api["29"]["class_type"] == (
        "MiniMaxH3FaceRefineWindowStudioCommitT8Advanced"
    )
    assert api["29"]["inputs"]["decision"] == "preview_only"
    assert api["29"]["inputs"]["confirm_accept"] is False
    assert api["29"]["inputs"]["job_id"] == ["28", 3]
    assert api["29"]["inputs"]["auto_continue"] == ["28", 2]
    assert api["30"]["inputs"]["commit_barrier"] == ["29", 9]
    assert api["25"]["inputs"]["images"] == ["30", 0]
    assert api["25"]["inputs"]["audio"] == ["2", 1]

    frontend = json.loads(FRONTEND_OUTPUT.read_text(encoding="utf-8"))
    assert frontend == build_frontend()
    nodes = {int(node["id"]): node for node in frontend["nodes"]}
    assert nodes[28]["widgets_values"][1] == "review_only"
    assert nodes[29]["widgets_values"][1:5] == ["preview_only", "", False, 2]
    assert any(
        link[1:5] == [28, 0, 27, 2] and link[5] == "INT"
        for link in frontend["links"]
    )
    assert any(
        link[1:5] == [29, 9, 30, 2] and link[5] == "STRING"
        for link in frontend["links"]
    )
    assert any(
        link[1:5] == [2, 1, 21, 1] and link[5] == "AUDIO"
        for link in frontend["links"]
    )


def test_studio_compose_only_workflow_recovers_without_h3_generation():
    from tools.build_face_refine_window_studio_compose_workflow import (
        API_OUTPUT,
        FRONTEND_OUTPUT,
        build_api,
        build_frontend,
    )

    api = json.loads(API_OUTPUT.read_text(encoding="utf-8"))
    assert api == build_api()
    assert not {
        "UNETLoader",
        "SamplerCustomAdvanced",
        "MiniMaxH3FaceRefineWindowStudioCommitT8Advanced",
    } & {node["class_type"] for node in api.values()}
    assert api["4"]["inputs"]["window_plan"] == ["3", 0]
    assert api["5"]["inputs"]["audio"] == ["2", 1]

    frontend = json.loads(FRONTEND_OUTPUT.read_text(encoding="utf-8"))
    assert frontend == build_frontend()
    types = {node["type"] for node in frontend["nodes"]}
    assert "MiniMaxH3FaceRefineWindowStudioComposeT8Advanced" in types
    assert "SamplerCustomAdvanced" not in types
    assert "UNETLoader" not in types
    compose = next(
        node
        for node in frontend["nodes"]
        if node["type"] == "MiniMaxH3FaceRefineWindowStudioComposeT8Advanced"
    )
    assert [item["name"] for item in compose["inputs"]] == [
        "base_frames",
        "window_plan",
    ]
    assert compose["widgets_values"] == ["face_refine_project_01"]


def _start_process_owner(tmp_path, chain_id, prompt_id, suffix, hold_seconds):
    worker = Path(__file__).with_name("multiprocess_background_worker.py")
    ready = tmp_path / f"ready-{suffix}"
    result = tmp_path / f"result-{suffix}.json"
    process = subprocess.Popen(
        [
            sys.executable,
            str(worker),
            "--output-dir",
            str(tmp_path),
            "--chain-id",
            chain_id,
            "--prompt-id",
            prompt_id,
            "--ready",
            str(ready),
            "--result",
            str(result),
            "--hold-seconds",
            str(hold_seconds),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    deadline = time.monotonic() + 30
    while not ready.is_file():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"Studio lease worker exited early: {process.returncode}\n{stdout}\n{stderr}"
            )
        if time.monotonic() >= deadline:
            process.kill()
            raise TimeoutError("Studio lease worker did not become ready")
        time.sleep(0.02)
    return process, json.loads(result.read_text(encoding="utf-8"))


def test_studio_same_project_process_lock_and_kill_recovery(monkeypatch, tmp_path):
    _bind_output(monkeypatch, tmp_path)
    frames, plan = _fixture()
    candidate, mask, mapping = _candidate(frames, plan, 0, 0.6)
    committed = commit_face_refine_window_studio(
        frames, candidate, mask, mapping, plan, "studio-process",
        "accept_selected", "", True, 0
    )
    assert committed[6] == 1
    chain_id = f"{STUDIO_CHAIN_PREFIX}studio-process"
    first = second = third = None
    try:
        first, first_result = _start_process_owner(
            tmp_path, chain_id, "prompt-first", "first", 60
        )
        assert first_result["ok"] is True
        assert first_result["state"]["accepted_count"] == 1
        first_job = first_result["state"]["job_id"]

        second, second_result = _start_process_owner(
            tmp_path, chain_id, "prompt-second", "second", 0
        )
        second.communicate(timeout=30)
        assert second_result["ok"] is False
        assert "owned by another ComfyUI process" in second_result["error"]

        first.kill()
        first.wait(timeout=10)
        third, third_result = _start_process_owner(
            tmp_path, chain_id, "prompt-third", "third", 0
        )
        third.communicate(timeout=30)
        assert third_result["ok"] is True
        assert third_result["state"]["accepted_count"] == 1
        assert third_result["state"]["previous_job_id"] == first_job
    finally:
        for process in (first, second, third):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
