"""Actual LoRA report UI normalization, cache replay and durable Core history."""
from copy import deepcopy
import json
from types import SimpleNamespace
import sys

import execution
import folder_paths
import pytest
from safetensors.torch import save_file
from comfy_execution.asset_enrichment import emit_cached_output

from h3_audio_t8_pkg import director_routes as routes
from h3_audio_t8_pkg.director_generation import director_job_status
from h3_audio_t8_pkg.director_project import atomic_json, sha
from h3_audio_t8_pkg.nodes_h3_lora_compat_advanced import MiniMaxH3LoRACompatibilityLoaderT8Advanced as Loader
from test_h24_director_result_observation import receipt_job
from test_h24_lora_file_alpha import check_forward, direct_pair
from test_progressive_sampling_runtime import tiny_model


def test_each_real_loader_report_survives_core_normalization_cache_and_history(tmp_path, monkeypatch):
    paths, deltas = {}, {}
    for prefix, alpha in (("one", 8.), ("two", -4.)):
        path = tmp_path / prefix / "same.safetensors"
        path.parent.mkdir()
        state, delta = direct_pair(alpha=alpha)
        save_file(state, str(path))
        paths[prefix + "/same.safetensors"] = path
        deltas[prefix] = delta * alpha / 16
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda _kind, name: str(paths[name]))
    base, selected, histories = tiny_model(), None, {}
    total = sum(deltas.values()) * 0
    for index, (name, strength) in enumerate((("one/same.safetensors", 3.), ("two/same.safetensors", -3.),
                                            ("one/same.safetensors", 0.)), 14):
        result = Loader.execute(selected or base, name, strength)
        output, ui, expanded = execution.get_output_from_returns([result], Loader)
        selected = output[0][0]
        assert not expanded
        envelope = ui["t8_lora_loader"][0]
        assert envelope["selected_lora_name"] == name
        assert envelope["report_json"] == output[1][0]
        assert json.loads(envelope["report_json"])["selected_strength"] == strength
        cached = SimpleNamespace(ui={"meta": {"node_id": str(index)}, "output": ui})
        replay = {}
        emit_cached_output(SimpleNamespace(client_id=None), str(index), str(index), cached,
                           "owned-prompt", replay, SimpleNamespace(enabled=False))
        assert replay[str(index)]["output"] == ui
        histories[str(index)] = replay[str(index)]["output"]
        total += deltas[name.split("/")[0]] * strength
    check_forward(selected, base, total)
    assert len(histories) == 3
    assert [json.loads(value["t8_lora_loader"][0]["report_json"])["file"]["name"]
            for value in histories.values()] == ["same.safetensors"] * 3
    before = deepcopy(histories)
    store, project, path, receipt = receipt_job(tmp_path / "history")
    owner = SimpleNamespace(queue_updated=lambda: None)
    queue = execution.PromptQueue(owner)
    owner.prompt_queue = queue
    monkeypatch.setitem(sys.modules, "server", SimpleNamespace(PromptServer=SimpleNamespace(instance=owner)))
    prompt = receipt["prompt_id"]
    queue.put((0, prompt, {}, {}, []))
    _, task = queue.get(timeout=.01)
    queue.task_done(task, {"outputs": histories},
                    execution.PromptQueue.ExecutionStatus("success", True,
                        [("execution_success", {"prompt_id": prompt})]))
    records = routes.director_project_results(store, project, director_job_status)["results"]
    assert records[0]["outputs"] == before
    queue.wipe_history()
    assert routes.director_project_results(store, project, director_job_status)["results"][0]["outputs"] == before
    assert json.loads(path.read_text(encoding="utf8"))["terminal"]["outputs"] == before


def test_disabled_loader_ui_is_explicit_and_never_opens_a_file(monkeypatch):
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise",
                        lambda *_args: pytest.fail("disabled loader opened a file"))
    base = tiny_model()
    result = Loader.execute(base, "disabled", -101.)
    output, ui, _ = execution.get_output_from_returns([result], Loader)
    assert output[0][0] is base
    assert json.loads(ui["t8_lora_loader"][0]["report_json"])["status"] == "disabled"


def test_snapshot_response_exposes_its_actual_request_and_prompt_not_current_draft(tmp_path):
    store, project, path, receipt = receipt_job(tmp_path)
    receipt.update(snapshot={"project": {"id": project}, "shot_id": receipt["shot_id"], "prompt": {}, "report": {}})
    receipt["snapshot_sha256"] = sha(receipt["snapshot"])
    atomic_json(path, receipt)
    response = routes.director_result_snapshot(store, project, path.stem)
    assert response["request_id"] == path.stem
    assert response["project_id"] == project
    assert response["prompt_id"] == receipt["prompt_id"]
    assert response["snapshot"] == receipt["snapshot"]
