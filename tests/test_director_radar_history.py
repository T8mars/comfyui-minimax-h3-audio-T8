"""Immutable content revisions and selected-shot dependency ownership, no GPU."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from h3_audio_t8_pkg.director_radar import (
    apply_operation,
    candidate_status,
    validate_history_progress,
)
from h3_audio_t8_pkg.director_project import (
    new_project,
    ProjectStore,
    validate_project,
    referenced_assets,
)
from h3_audio_t8_pkg.director_batch import (
    compile_batch_selection,
    selected_project,
    capture_resources,
    verify_resources,
)
from h3_audio_t8_pkg.director_bundle import portable_project
from test_director_radar import project_with_source, review, accept, uid


def test_evidence_versions_and_reviews_are_detached_readonly_history():
    project, packet, sid = project_with_source()
    project = apply_operation(project, sid, "evidence_import", packet)
    pending = deepcopy(project["doc"]["shots"][0]["sourceEvidence"])
    project = review(project, sid)
    confirmed = deepcopy(project["doc"]["shots"][0]["sourceEvidence"])
    packet["revision"] = 2
    packet["claims"][0]["text"] = "另一份观察"
    updated = apply_operation(project, sid, "evidence_import", packet)
    shot = updated["doc"]["shots"][0]
    assert shot["evidenceHistory"] == [pending, confirmed]
    assert shot["sourceEvidence"]["review"] == confirmed["review"]
    assert project["doc"]["shots"][0]["sourceEvidence"] == confirmed
    updated = review(updated, sid)
    assert updated["doc"]["shots"][0]["evidenceHistory"][:2] == [pending, confirmed]
    assert updated["doc"]["shots"][0]["sourceEvidence"]["review"]["revision"] == 2
    validate_history_progress(project, updated)
    for revision in [1, 2]:
        invalid = deepcopy(packet)
        invalid["revision"] = revision
        invalid["claims"][0]["text"] = "同版本改写"
        before = deepcopy(updated)
        with pytest.raises(ValueError, match="版本不可改写|回退"):
            apply_operation(updated, sid, "evidence_import", invalid)
        assert updated == before
    # Importing exactly the current packet does not invent a new revision.
    repeated = apply_operation(updated, sid, "evidence_import", packet)
    assert (
        repeated["doc"]["shots"][0]["evidenceHistory"]
        == updated["doc"]["shots"][0]["evidenceHistory"]
    )


def test_saved_intent_candidate_history_cannot_be_rewritten_or_dropped(tmp_path):
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    project = new_project()
    sid = project["current"]
    project["doc"]["shots"][0]["simplePrompt"] = "原作者稿"
    project = store.save(project, 0)
    project = apply_operation(
        project,
        sid,
        "intent_update",
        {"text": "蓝衣", "reason": "用户明确修改", "dependencies": []},
    )
    project = accept(apply_operation(project, sid, "candidate_create", {}), sid)
    project = store.save(project, 1)
    old = deepcopy(project["doc"]["shots"][0])
    next_draft = apply_operation(
        project,
        sid,
        "intent_update",
        {"text": "蓝衣，缓慢推近", "reason": "新增运动", "dependencies": []},
    )
    next_draft = accept(apply_operation(next_draft, sid, "candidate_create", {}), sid)
    saved = store.save(next_draft, 2)
    shot = saved["doc"]["shots"][0]
    assert shot["intentHistory"] == [old["creativeIntent"]]
    assert shot["candidateHistory"] == [old["promptCandidate"]]
    assert (
        ProjectStore(tmp_path / "user", tmp_path / "input").load(project["id"]) == saved
    )
    assert (
        portable_project(saved, include_radar=True)["doc"]["shots"][0]["intentHistory"]
        == shot["intentHistory"]
    )
    assert "intentHistory" not in portable_project(saved)["doc"]["shots"][0]
    for changed in ["rewrite", "drop"]:
        invalid = deepcopy(saved)
        if changed == "rewrite":
            invalid["doc"]["shots"][0]["intentHistory"][0]["text"] = "伪造旧修改"
        else:
            invalid["doc"]["shots"][0].pop("candidateHistory")
        with pytest.raises(ValueError, match="历史不可改写"):
            store.save(invalid, 3)
        assert store.load(project["id"]) == saved
    # Explicit revert is a state change, not destructive editing of a candidate.
    reverted = apply_operation(saved, sid, "candidate_revert", {})
    assert (
        store.save(reverted, 3)["doc"]["shots"][0]["candidateHistory"]
        == shot["candidateHistory"]
    )


def test_no_silent_history_truncation_and_no_change_to_legacy_shape():
    project = new_project()
    sid = project["current"]
    project["doc"]["shots"][0]["simplePrompt"] = "稿"
    assert not any(
        "History" in key for key in validate_project(project)["doc"]["shots"][0]
    )
    project = apply_operation(project, sid, "candidate_create", {})
    shot = project["doc"]["shots"][0]
    shot["candidateHistory"] = [deepcopy(shot["promptCandidate"]) for _ in range(256)]
    before = deepcopy(project)
    with pytest.raises(ValueError, match="256"):
        apply_operation(project, sid, "candidate_create", {})
    assert project == before


def cross_project():
    project, packet, sid = project_with_source()
    project = review(apply_operation(project, sid, "evidence_import", packet), sid)
    first = project["doc"]["shots"][0]
    second = deepcopy(first)
    second.update(id=uid(), name="第二镜")
    second.pop("sourceEvidence")
    second.pop("evidenceHistory")
    project["doc"]["shots"].append(second)
    dependency = {
        "shot_id": sid,
        "evidence_sha256": first["sourceEvidence"]["packet"]["sha256"],
    }
    project = apply_operation(
        project,
        second["id"],
        "intent_update",
        {"text": "沿用上一镜事实", "reason": "", "dependencies": [dependency]},
    )
    return accept(
        apply_operation(project, second["id"], "candidate_create", {}), second["id"]
    ), second["id"]


def test_batch_subset_keeps_only_consumed_cross_shot_facts_and_resource_bytes():
    project, sid = cross_project()
    project["doc"]["shots"][0]["simplePrompt"] = (
        ""  # unfinished unselected shot must not block
    )
    unused = {"id": uid(), "name": "无关草稿", "kind": "invalid-unused-kind"}
    project["assets"].append(unused)
    selected = selected_project(project, [project["doc"]["shots"][1]])
    assert [row["id"] for row in selected["assets"]] == [project["assets"][0]["id"]]
    report = compile_batch_selection(project, None, [sid])
    assert report["ready"], report["errors"]
    assert report["selection"] == {"shot_ids": [sid]}
    assert [row["id"] for row in report["shots"]] == [sid]
    aid = project["assets"][0]["id"]
    calls = []

    def asset(key, verify=True):
        calls.append(key)
        return {**project["assets"][0], "server_path": "owned/source.mp4"}

    store = SimpleNamespace(asset=asset)
    resources = capture_resources(
        store, selected, [{"prompt": {}}], lambda *_: None, evidence_context=project
    )
    assert calls == [aid] and set(resources["assets"]) == {aid}
    assert referenced_assets(project) == {aid}
    verify_resources(store, resources, lambda *_: None)
    project["assets"][0]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="素材已变化"):
        verify_resources(store, resources, lambda *_: None)
    assert not compile_batch_selection(project, None, [sid])["ready"]


def test_history_and_unconsumed_facts_intent_do_not_stale_candidate():
    project, packet, sid = project_with_source()
    project = apply_operation(project, sid, "evidence_import", packet)
    project = accept(
        apply_operation(
            project, sid, "candidate_create", {"facts": False, "intent": False}
        ),
        sid,
    )
    packet["revision"] = 2
    packet["claims"][0]["text"] = "未消费的新观察"
    project = apply_operation(project, sid, "evidence_import", packet)
    project = apply_operation(
        project,
        sid,
        "intent_update",
        {"text": "未消费的修改", "reason": "", "dependencies": []},
    )
    assert candidate_status(project, project["doc"]["shots"][0]) == "active"


def test_history_must_be_preserved_before_changing_saved_current_fields():
    project = new_project()
    sid = project["current"]
    project = apply_operation(
        project,
        sid,
        "intent_update",
        {"text": "旧修改", "reason": "", "dependencies": []},
    )
    invalid = deepcopy(project)
    invalid["doc"]["shots"][0]["creativeIntent"].update(revision=2, text="无留档替换")
    with pytest.raises(ValueError, match="完整历史"):
        validate_history_progress(project, validate_project(invalid))
