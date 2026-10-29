"""Real source bytes, three-shot dependency and selected resource boundaries."""

from copy import deepcopy
import uuid

import pytest
import torch
from safetensors.torch import save_file

from h3_audio_t8_pkg.director_batch import (
    capture_resources,
    compile_batch_selection,
    selected_project,
    verify_resources,
)
from h3_audio_t8_pkg.director_project import ProjectStore
from h3_audio_t8_pkg.director_radar import apply_operation, candidate_status
from test_director_radar import project_with_source, review, accept
from test_director_generation import _store
from test_h24_director_audio_bindings import _video


def test_parent_source_change_invalidates_dependent_not_independent_and_keeps_history(
    tmp_path,
):
    store = _store(tmp_path)
    project, packet, sid = project_with_source()
    asset = _video(store, "red", True)
    project["assets"] = [asset]
    packet["source"].update(
        asset_id=asset["id"],
        sha256=asset["sha256"],
        timebase={"num": 1, "den": 24},
        start_pts=0,
        end_pts=48,
    )
    packet["claims"][0].update(start_pts=0, end_pts=24)
    packet["claims"][1].update(start_pts=24, end_pts=48)
    project = review(apply_operation(project, sid, "evidence_import", packet), sid)
    project = accept(apply_operation(project, sid, "candidate_create", {}), sid)
    parent = project["doc"]["shots"][0]
    dependent = deepcopy(parent)
    dependent.update(id=str(uuid.uuid4()), name="Dependent")
    independent = deepcopy(parent)
    independent.update(
        id=str(uuid.uuid4()), name="Independent", simplePrompt="Unrelated scene."
    )
    for shot in (dependent, independent):
        for key in (
            "sourceEvidence",
            "evidenceHistory",
            "promptCandidate",
            "candidateHistory",
            "creativeIntent",
            "intentHistory",
        ):
            shot.pop(key, None)
    project["doc"]["shots"] += [dependent, independent]
    project = apply_operation(
        project,
        dependent["id"],
        "intent_update",
        {
            "text": "沿用上一镜事实",
            "reason": "",
            "dependencies": [
                {
                    "shot_id": sid,
                    "evidence_sha256": parent["sourceEvidence"]["packet"]["sha256"],
                }
            ],
        },
    )
    project = accept(
        apply_operation(project, dependent["id"], "candidate_create", {}),
        dependent["id"],
    )
    saved = store.save(project, 0)
    reopened = ProjectStore(store.root.parent, store.input_root).load(saved["id"])

    def resolver(*_):
        return None

    resources = {}
    for shot in reopened["doc"]["shots"]:
        assert compile_batch_selection(reopened, store, [shot["id"]])["ready"]
        selected = selected_project(reopened, [shot])
        resources[shot["id"]] = capture_resources(
            store, selected, [{"prompt": {}}], resolver, evidence_context=reopened
        )
    assert resources[independent["id"]]["assets"] == {}
    original_history = deepcopy(reopened["doc"]["shots"][1]["promptCandidate"])
    path = store.input_root / asset["server_path"]
    raw = path.read_bytes()
    path.write_bytes(raw + b"\nactual_same_name_byte_change")
    for shot_id in (sid, dependent["id"]):
        with pytest.raises(ValueError, match="字节改变"):
            verify_resources(store, resources[shot_id], resolver)
    verify_resources(store, resources[independent["id"]], resolver)
    assert compile_batch_selection(reopened, store, [independent["id"]])["ready"]
    assert reopened["doc"]["shots"][1]["promptCandidate"] == original_history
    # Explicit fact revision stales B; unrelated author text alone does not.
    path.write_bytes(raw)
    reopened["doc"]["shots"][0]["simplePrompt"] += " author draft only"
    assert candidate_status(reopened, reopened["doc"]["shots"][1]) == "active"
    packet["revision"] = 2
    packet["claims"][0]["text"] = "事实已改变"
    changed = apply_operation(reopened, sid, "evidence_import", packet)
    assert candidate_status(changed, changed["doc"]["shots"][1]) == "stale"
    assert compile_batch_selection(changed, store, [independent["id"]])["ready"]


def test_actual_selected_lora_content_changes_break_old_receipt_without_mutating_order(
    tmp_path,
):
    store = _store(tmp_path)
    project, _, _ = project_with_source()
    project["assets"] = []
    paths = {}
    prompt = {}
    for i, strength in enumerate((0.35, -0.7)):
        name = f"content-{i}.safetensors"
        paths[name] = tmp_path / name
        save_file({"weight": torch.full((2, 2), i + 0.25)}, str(paths[name]))
        prompt[str(i)] = {
            "class_type": "MiniMaxH3LoRACompatibilityLoaderT8Advanced",
            "inputs": {"lora_name": name, "strength_model": strength},
        }
    built = [{"prompt": prompt}]
    before = deepcopy(built)

    def resolve(folder, name):
        return paths[name]

    resources = capture_resources(store, project, built, resolve)
    verify_resources(store, resources, resolve)
    save_file({"weight": torch.full((2, 2), 2.25)}, str(paths["content-0.safetensors"]))
    with pytest.raises(ValueError, match="模型已变化"):
        verify_resources(store, resources, resolve)
    assert built == before
