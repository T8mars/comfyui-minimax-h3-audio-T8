from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from h3_audio_t8_pkg.long_video_background import (
    BACKGROUND_SCHEMA,
    BACKGROUND_STATE_FORMAT,
    BackgroundJobError,
    BackgroundJobManager,
    ComfyQueueRuntime,
    UnsupportedBackgroundSchemaError,
    load_background_job_state,
)


class FakeRuntime:
    def __init__(self):
        self.current_id = "prompt-0"
        self.client_id = "client-1"
        self.counter = 0
        self.locations = {self.current_id: "running"}
        self.histories = {}
        self.queued = []
        self.cancelled = []
        self.releases = []
        self.release_error = None

    def current_prompt_id(self):
        return self.current_id

    def current_client_id(self):
        return self.client_id

    def queue_prompt(self, prompt, client_id):
        self.counter += 1
        prompt_id = f"queued-{self.counter}"
        self.queued.append((prompt_id, dict(prompt), client_id))
        self.locations[prompt_id] = "queued"
        return prompt_id

    def prompt_location(self, prompt_id):
        return self.locations.get(prompt_id, "missing")

    def history_record(self, prompt_id):
        return self.histories.get(prompt_id)

    def cancel_prompt(self, prompt_id):
        self.cancelled.append(prompt_id)
        location = self.locations.get(prompt_id)
        self.locations[prompt_id] = "missing"
        return {
            "prompt_id": prompt_id,
            "deleted_from_queue": location == "queued",
            "interrupt_signalled": location == "running",
        }

    def request_release(self, release_policy):
        self.releases.append(release_policy)
        if self.release_error is not None:
            raise self.release_error


def _manager(monkeypatch, tmp_path, *, max_retries=1):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    runtime = FakeRuntime()
    manager = BackgroundJobManager(runtime, start_monitors=False)
    state = manager.attach_prompt(
        "background-chain",
        {
            "1": {
                "class_type": "Test",
                "inputs": {"text": "Secret prompt text"},
                "is_changed": ["stale-runtime-cache-fingerprint"],
            }
        },
        "99",
        max_retries,
        0.0,
        "clear_execution_cache",
    )
    return manager, runtime, state


def test_background_state_does_not_persist_prompt_and_advances_one_segment(monkeypatch, tmp_path):
    manager, runtime, state = _manager(monkeypatch, tmp_path)
    job_id = state["job_id"]
    path = (
        Path(tmp_path)
        / "minimax_h3_t8_long_video"
        / "background-chain"
        / "background_job.json"
    )
    assert path.is_file()
    assert "Secret prompt text" not in path.read_text(encoding="utf-8")
    assert len(state["prompt_sha256"]) == 64

    advanced = manager.segment_accepted(
        job_id,
        candidate_index=0,
        candidate_json_path="candidate-0.json",
        manifest_path="manifest.json",
        accepted_count=1,
        is_final_segment=False,
    )
    assert advanced["state"] == "running"
    assert advanced["active_prompt_id"] == "queued-1"
    assert advanced["current_segment_index"] == 1
    assert runtime.releases == ["clear_execution_cache"]
    assert [item[0] for item in runtime.queued] == ["queued-1"]
    assert "is_changed" not in runtime.queued[0][1]["1"]

    completed = manager.segment_accepted(
        job_id,
        candidate_index=1,
        candidate_json_path="candidate-1.json",
        manifest_path="manifest.json",
        accepted_count=2,
        is_final_segment=True,
        final_video_path="final.mp4",
    )
    assert completed["state"] == "completed"
    assert completed["active_prompt_id"] == ""
    assert completed["final_video_path"] == "final.mp4"
    assert len(runtime.queued) == 1
    assert runtime.releases == ["clear_execution_cache", "clear_execution_cache"]
    assert completed["last_release_policy"] == "clear_execution_cache"
    assert completed["release_requested_unix"] > 0


def test_background_error_retries_once_then_stops_without_changing_settings(
    monkeypatch, tmp_path
):
    manager, runtime, state = _manager(monkeypatch, tmp_path, max_retries=1)
    error_record = {
        "status": {
            "status_str": "error",
            "completed": False,
            "messages": [
                [
                    "execution_error",
                    {
                        "node_id": "8",
                        "node_type": "SamplerCustomAdvanced",
                        "exception_type": "OutOfMemoryError",
                        "exception_message": "CUDA out of memory",
                        "current_inputs": {"prompt": "must never be persisted"},
                    },
                ]
            ],
        }
    }
    manager._handle_prompt_history(
        "background-chain", state["job_id"], "prompt-0", error_record
    )
    retried = load_background_job_state("background-chain")
    assert retried["state"] == "running"
    assert retried["retry_count"] == 1
    assert retried["active_prompt_id"] == "queued-1"
    assert runtime.queued[0][1]["1"]["inputs"]["text"] == "Secret prompt text"
    assert runtime.releases == ["clear_execution_cache"]

    manager._handle_prompt_history(
        "background-chain", state["job_id"], "queued-1", error_record
    )
    failed = load_background_job_state("background-chain")
    assert failed["state"] == "failed"
    assert failed["active_prompt_id"] == ""
    assert "CUDA out of memory" in failed["last_error"]
    assert "must never be persisted" not in failed["last_error"]
    assert "current_inputs" not in failed["last_error"]
    assert len(runtime.queued) == 1


def test_pause_after_current_resume_and_targeted_cancel(monkeypatch, tmp_path):
    manager, runtime, state = _manager(monkeypatch, tmp_path)
    paused_requested = manager.pause("background-chain")
    assert paused_requested["state"] == "pausing"
    assert runtime.cancelled == []

    paused = manager.segment_accepted(
        state["job_id"],
        candidate_index=0,
        candidate_json_path="candidate-0.json",
        manifest_path="manifest.json",
        accepted_count=1,
        is_final_segment=False,
    )
    assert paused["state"] == "paused"
    assert runtime.queued == []
    assert runtime.releases == ["clear_execution_cache"]

    resumed = manager.resume("background-chain")
    assert resumed["state"] == "running"
    assert resumed["active_prompt_id"] == "queued-1"
    cancelled = manager.cancel("background-chain")
    assert cancelled["state"] == "cancelled"
    assert cancelled["cancel_requested"] is True
    assert runtime.cancelled == ["queued-1"]
    assert cancelled["last_control_result"]["deleted_from_queue"] is True


def test_release_failure_after_acceptance_stops_without_queueing(monkeypatch, tmp_path):
    manager, runtime, state = _manager(monkeypatch, tmp_path)
    runtime.release_error = RuntimeError("release flag rejected")
    failed = manager.segment_accepted(
        state["job_id"],
        candidate_index=0,
        candidate_json_path="candidate-0.json",
        manifest_path="manifest.json",
        accepted_count=1,
        is_final_segment=False,
    )
    assert failed["state"] == "failed"
    assert failed["accepted_count"] == 1
    assert failed["active_prompt_id"] == ""
    assert "failed to apply release policy" in failed["last_error"]
    assert runtime.queued == []


def test_duplicate_active_chain_is_rejected(monkeypatch, tmp_path):
    manager, runtime, _state = _manager(monkeypatch, tmp_path)
    runtime.current_id = "another-prompt"
    runtime.locations["prompt-0"] = "running"
    with pytest.raises(BackgroundJobError, match="already has an active"):
        manager.attach_prompt(
            "background-chain",
            {"1": {"class_type": "Test", "inputs": {}}},
            "99",
            1,
            0.0,
            "clear_execution_cache",
        )


def test_restart_status_reconciles_stale_prompt_and_reattaches_from_manifest(
    monkeypatch, tmp_path
):
    manager_before_restart, runtime, original = _manager(monkeypatch, tmp_path)
    runtime.locations["prompt-0"] = "missing"
    manager_before_restart.close()
    restarted = BackgroundJobManager(runtime, start_monitors=False)
    monkeypatch.setattr(restarted, "_manifest_position", lambda _chain: (1, False))

    detached = restarted.status("background-chain")
    assert detached["state"] == "detached"
    assert detached["accepted_count"] == 1
    assert detached["manifest_complete"] is False
    assert detached["active_prompt_id"] == ""
    assert detached["runtime_location"] == "none"
    assert detached["orphaned_prompt_id"] == "prompt-0"
    assert detached["orphaned_runtime_location"] == "missing"
    assert detached["recovery_required"] is True
    assert detached["recovery_action"] == "queue_workflow_once"
    assert detached["resumable_in_memory"] is False
    assert "Queue the background workflow once" in detached["last_error"]

    persisted = load_background_job_state("background-chain")
    assert persisted["state"] == "detached"
    assert persisted["accepted_count"] == 1
    with pytest.raises(BackgroundJobError, match="snapshot was lost"):
        restarted.resume("background-chain")

    runtime.current_id = "prompt-after-restart"
    runtime.locations[runtime.current_id] = "running"
    reattached = restarted.attach_prompt(
        "background-chain",
        {"1": {"class_type": "Test", "inputs": {"text": "same workflow"}}},
        "99",
        1,
        0.0,
        "clear_execution_cache",
    )
    assert reattached["state"] == "running"
    assert reattached["accepted_count"] == 1
    assert reattached["current_segment_index"] == 1
    assert reattached["previous_job_id"] == original["job_id"]
    assert reattached["active_prompt_id"] == "prompt-after-restart"


def test_restart_with_complete_manifest_directs_compose_instead_of_generation(
    monkeypatch, tmp_path
):
    manager_before_restart, runtime, _original = _manager(monkeypatch, tmp_path)
    runtime.locations["prompt-0"] = "success"
    manager_before_restart.close()
    restarted = BackgroundJobManager(runtime, start_monitors=False)
    monkeypatch.setattr(restarted, "_manifest_position", lambda _chain: (2, True))

    detached = restarted.status("background-chain")
    assert detached["state"] == "detached"
    assert detached["accepted_count"] == 2
    assert detached["manifest_complete"] is True
    assert detached["recovery_required"] is True
    assert detached["recovery_action"] == "compose_accepted"
    assert "do not generate another segment" in detached["last_error"]
    with pytest.raises(BackgroundJobError, match="already complete"):
        restarted.resume("background-chain")


def test_multiple_background_chains_keep_prompt_and_control_state_isolated(
    monkeypatch, tmp_path
):
    manager, runtime, first = _manager(monkeypatch, tmp_path)
    runtime.current_id = "prompt-b"
    runtime.locations[runtime.current_id] = "running"
    second = manager.attach_prompt(
        "background-chain-b",
        {"1": {"class_type": "TestB", "inputs": {"value": 2}}},
        "100",
        0,
        0.0,
        "keep_loaded",
    )

    advanced_first = manager.segment_accepted(
        first["job_id"],
        candidate_index=0,
        candidate_json_path="candidate-a.json",
        manifest_path="manifest-a.json",
        accepted_count=1,
        is_final_segment=False,
    )
    assert advanced_first["active_prompt_id"] == "queued-1"
    assert manager.status("background-chain-b")["active_prompt_id"] == "prompt-b"

    cancelled_second = manager.cancel("background-chain-b")
    assert cancelled_second["job_id"] == second["job_id"]
    assert runtime.cancelled == ["prompt-b"]
    assert manager.status("background-chain")["active_prompt_id"] == "queued-1"
    assert load_background_job_state("background-chain")["state"] == "running"
    assert load_background_job_state("background-chain-b")["state"] == "cancelled"


def test_unreadable_auxiliary_state_recovers_from_manifest_on_one_requeue(
    monkeypatch, tmp_path
):
    manager_instance, runtime, original = _manager(monkeypatch, tmp_path)
    state_path = (
        Path(tmp_path)
        / "minimax_h3_t8_long_video"
        / "background-chain"
        / "background_job.json"
    )
    state_path.write_text("{broken", encoding="utf-8")
    manager_instance.close()
    restarted = BackgroundJobManager(runtime, start_monitors=False)
    monkeypatch.setattr(restarted, "_manifest_position", lambda _chain: (1, False))

    detached = restarted.status("background-chain")
    assert detached["state"] == "detached"
    assert detached["accepted_count"] == 1
    assert detached["state_file_unreadable"] is True
    assert detached["recovery_action"] == "queue_workflow_once"
    assert state_path.read_text(encoding="utf-8") == "{broken"

    runtime.current_id = "prompt-after-corruption"
    runtime.locations[runtime.current_id] = "running"
    recovered = restarted.attach_prompt(
        "background-chain",
        {"1": {"class_type": "Test", "inputs": {"text": "same workflow"}}},
        "99",
        1,
        0.0,
        "clear_execution_cache",
    )
    assert recovered["state"] == "running"
    assert recovered["accepted_count"] == 1
    assert recovered["job_id"] != original["job_id"]
    assert recovered["last_error"] == ""
    assert "Recovered from unreadable" in recovered["recovery_notice"]
    quarantine = Path(recovered["quarantined_state_path"])
    assert quarantine.is_file()
    assert quarantine.read_text(encoding="utf-8") == "{broken"
    assert load_background_job_state("background-chain")["job_id"] == recovered["job_id"]


def test_newer_background_state_schema_is_never_overwritten_by_downgrade(
    monkeypatch, tmp_path
):
    manager_instance, runtime, _original = _manager(monkeypatch, tmp_path)
    state_path = (
        Path(tmp_path)
        / "minimax_h3_t8_long_video"
        / "background-chain"
        / "background_job.json"
    )
    newer = json.loads(state_path.read_text(encoding="utf-8"))
    newer["schema"] = 999
    state_path.write_text(json.dumps(newer), encoding="utf-8")
    manager_instance.close()

    restarted = BackgroundJobManager(runtime, start_monitors=False)
    with pytest.raises(UnsupportedBackgroundSchemaError, match="schema 999"):
        restarted.status("background-chain")
    runtime.current_id = "prompt-from-older-plugin"
    runtime.locations[runtime.current_id] = "running"
    with pytest.raises(UnsupportedBackgroundSchemaError, match="schema 999"):
        restarted.attach_prompt(
            "background-chain",
            {"1": {"class_type": "Test", "inputs": {}}},
            "99",
            1,
            0.0,
            "clear_execution_cache",
        )
    assert json.loads(state_path.read_text(encoding="utf-8"))["schema"] == 999
    assert not list(state_path.parent.glob("background_job.corrupt.*.json"))


def test_legacy_v1_background_state_migrates_on_restart_reattach(monkeypatch, tmp_path):
    manager, runtime, original = _manager(monkeypatch, tmp_path)
    state_path = (
        Path(tmp_path)
        / "minimax_h3_t8_long_video"
        / "background-chain"
        / "background_job.json"
    )
    legacy = json.loads(state_path.read_text(encoding="utf-8"))
    legacy["schema"] = 1
    legacy.pop("format", None)
    legacy.pop("migrated_from_schema", None)
    state_path.write_text(json.dumps(legacy), encoding="utf-8")
    manager.close()

    loaded = load_background_job_state("background-chain")
    assert loaded["schema"] == BACKGROUND_SCHEMA
    assert loaded["format"] == BACKGROUND_STATE_FORMAT
    assert loaded["migrated_from_schema"] == 1
    assert json.loads(state_path.read_text(encoding="utf-8"))["schema"] == 1

    runtime.locations["prompt-0"] = "missing"
    runtime.current_id = "prompt-after-v1-upgrade"
    runtime.locations[runtime.current_id] = "running"
    restarted = BackgroundJobManager(runtime, start_monitors=False)
    reattached = restarted.attach_prompt(
        "background-chain",
        {"1": {"class_type": "Test", "inputs": {"text": "same workflow"}}},
        "99",
        1,
        0.0,
        "clear_execution_cache",
    )
    assert reattached["job_id"] != original["job_id"]
    assert reattached["previous_job_id"] == original["job_id"]
    assert reattached["previous_state_schema"] == 1
    upgraded_raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert upgraded_raw["schema"] == BACKGROUND_SCHEMA
    assert upgraded_raw["format"] == BACKGROUND_STATE_FORMAT
    assert upgraded_raw["previous_state_schema"] == 1
    restarted.close()


def _background_worker_command(*arguments):
    worker = Path(__file__).with_name("multiprocess_background_worker.py")
    return [sys.executable, str(worker), *map(str, arguments)]


def _wait_for_process_file(path, process, timeout_seconds=30.0):
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"Background worker exited before signalling ready: {process.returncode}\n"
                f"stdout={stdout}\nstderr={stderr}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(0.02)


def _start_background_worker(tmp_path, sync, prompt_id, suffix, hold_seconds):
    ready = sync / f"ready-{suffix}"
    result = sync / f"result-{suffix}.json"
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        _background_worker_command(
            "--output-dir", tmp_path,
            "--chain-id", "process-owned-chain",
            "--prompt-id", prompt_id,
            "--ready", ready,
            "--result", result,
            "--hold-seconds", hold_seconds,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creation_flags,
    )
    _wait_for_process_file(ready, process)
    return process, json.loads(result.read_text(encoding="utf-8"))


def test_background_process_lease_rejects_second_owner_and_recovers_after_kill(
    monkeypatch, tmp_path
):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    sync = tmp_path / "background-process-sync"
    sync.mkdir()
    first = second = third = None
    try:
        first, first_result = _start_background_worker(
            tmp_path, sync, "prompt-first", "first", 60
        )
        assert first_result["ok"] is True
        first_job_id = first_result["state"]["job_id"]

        second, second_result = _start_background_worker(
            tmp_path, sync, "prompt-second", "second", 0
        )
        stdout, stderr = second.communicate(timeout=30)
        assert second.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
        assert second_result["ok"] is False
        assert second_result["error_type"] == "BackgroundJobError"
        assert "owned by another ComfyUI process" in second_result["error"]
        unchanged = load_background_job_state("process-owned-chain")
        assert unchanged["job_id"] == first_job_id
        assert unchanged["active_prompt_id"] == "prompt-first"

        first.kill()
        first.wait(timeout=10)
        third, third_result = _start_background_worker(
            tmp_path, sync, "prompt-third", "third", 0
        )
        stdout, stderr = third.communicate(timeout=30)
        assert third.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
        assert third_result["ok"] is True
        recovered = third_result["state"]
        assert recovered["job_id"] != first_job_id
        assert recovered["previous_job_id"] == first_job_id
        assert recovered["active_prompt_id"] == "prompt-third"
        state_text = (
            tmp_path
            / "minimax_h3_t8_long_video"
            / "process-owned-chain"
            / "background_job.json"
        ).read_text(encoding="utf-8")
        assert "private prompt body" not in state_text
    finally:
        for process in (first, second, third):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)


def test_release_flags_do_not_implicitly_unload_models(monkeypatch):
    flags = []

    class Queue:
        @staticmethod
        def set_flag(name, value):
            flags.append((name, value))

    runtime = ComfyQueueRuntime()
    monkeypatch.setattr(
        runtime, "_server", lambda: SimpleNamespace(prompt_queue=Queue())
    )
    runtime.request_release("clear_execution_cache")
    assert flags == [("unload_models", False), ("free_memory", True)]
    flags.clear()
    runtime.request_release("unload_all_models")
    assert flags == [("unload_models", True), ("free_memory", True)]
