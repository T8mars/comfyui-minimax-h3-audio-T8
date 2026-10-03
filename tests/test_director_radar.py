from copy import deepcopy
import json
import uuid

import pytest

from h3_audio_t8_pkg.director_project import (
    new_project,
    validate_project,
    compile_project,
    sha,
    ProjectStore,
)
from h3_audio_t8_pkg.director_radar import (
    PACKET_SCHEMA,
    make_skill,
    bind_skill,
    render_binding,
    seal_packet,
    validate_packet,
    apply_operation,
    candidate_status,
    evidence_status,
    evidence_time_map,
    validate_bindings,
)
from h3_audio_t8_pkg.director_bundle import (
    portable_project,
    prepare_bundle,
    build_bundle,
    inspect_bundle,
    import_bundle,
)


def uid():
    return str(uuid.uuid4())


def project_with_source():
    project = new_project()
    asset_id = uid()
    project["assets"] = [
        {"id": asset_id, "kind": "video", "name": "source.mp4", "sha256": "a" * 64}
    ]
    shot = project["doc"]["shots"][0]
    shot["simplePrompt"] = "原作者稿：台词“今天晴天”。"
    packet = {
        "schema": PACKET_SCHEMA,
        "id": uid(),
        "revision": 1,
        "source": {
            "asset_id": asset_id,
            "sha256": "a" * 64,
            "stream": 0,
            "timebase": {"num": 1, "den": 90000},
            "start_pts": 900000,
            "end_pts": 990000,
        },
        "claims": [
            {
                "id": uid(),
                "kind": "visual",
                "text": "红衣女子站在窗边",
                "start_pts": 900000,
                "end_pts": 945000,
            },
            {
                "id": uid(),
                "kind": "ocr",
                "text": "字幕“今天下雨”",
                "start_pts": 945000,
                "end_pts": 990000,
            },
        ],
    }
    return project, packet, shot["id"]


def review(project, sid):
    packet = project["doc"]["shots"][0]["sourceEvidence"]["packet"]
    return apply_operation(
        project,
        sid,
        "evidence_review",
        {
            "confirm": True,
            "packet_sha256": packet["sha256"],
            "claim_ids": [row["id"] for row in packet["claims"]],
        },
    )


def accept(project, sid):
    candidate = next(row for row in project["doc"]["shots"] if row["id"] == sid)[
        "promptCandidate"
    ]
    return apply_operation(
        project,
        sid,
        "candidate_accept",
        {"confirm": True, "text_sha256": candidate["text_sha256"]},
    )


def test_legacy_project_and_compilation_exact_without_new_fields():
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "  原稿\r\n“台词”\t🙂  "
    assert validate_project(project) == project
    compiled = compile_project(project)
    assert (
        compiled["ready"]
        and compiled["shots"][0]["prompt"] == project["doc"]["shots"][0]["simplePrompt"]
    )
    assert not any(
        key.startswith("skill") or key == "promptCandidate"
        for key in project["doc"]["shots"][0]
    )
    assert portable_project(project) == portable_project(project, include_radar=True)


def test_evidence_import_review_intent_and_candidate_are_separate_no_auto_accept():
    project, packet, sid = project_with_source()
    original = deepcopy(project)
    project = apply_operation(project, sid, "evidence_import", packet)
    shot = project["doc"]["shots"][0]
    assert (
        evidence_status(project, shot) == "unreviewed"
        and "review" not in shot["sourceEvidence"]
    )
    with pytest.raises(ValueError, match="审核"):
        apply_operation(project, sid, "candidate_create", {"facts": True})
    project = review(project, sid)
    project = apply_operation(
        project,
        sid,
        "intent_update",
        {"text": "改成蓝衣，原台词不改", "reason": "用户创作修改", "dependencies": []},
    )
    project = apply_operation(project, sid, "candidate_create", {"facts": True})
    shot = project["doc"]["shots"][0]
    assert candidate_status(project, shot) == "pending"
    assert (
        shot["sourceEvidence"]["packet"]["claims"][0]["text"]
        == packet["claims"][0]["text"]
    )
    assert (
        "红衣" in shot["promptCandidate"]["text"]
        and "蓝衣" in shot["promptCandidate"]["text"]
    )
    assert shot["simplePrompt"] == original["doc"]["shots"][0]["simplePrompt"]
    assert original["doc"]["shots"][0].get("sourceEvidence") is None
    project = accept(project, sid)
    assert candidate_status(project, project["doc"]["shots"][0]) == "active"
    reverted = apply_operation(project, sid, "candidate_revert", {})
    assert candidate_status(reverted, reverted["doc"]["shots"][0]) == "pending"
    assert (
        reverted["doc"]["shots"][0]["simplePrompt"]
        == original["doc"]["shots"][0]["simplePrompt"]
    )


def test_pts_mapping_exact_not_local_seconds_or_rounded_frames():
    _project, packet, _sid = project_with_source()
    packet["claims"][0]["start_pts"] = 900001
    mapping = evidence_time_map(seal_packet(packet), output_start_frame=120)
    start = mapping["claims"][0]["start"]
    assert start["absolute_seconds"] == {"num": 900001, "den": 90000}
    assert start["local_seconds"] == {"num": 1, "den": 90000}
    assert start["output_frame_position"] == {"num": 450001, "den": 3750}
    assert (
        mapping["automatic_rounding"] is False
        and mapping["sampling_frames_modified"] is False
    )


def test_negative_pts_and_audio_header_bounds_are_not_rebased_or_guessed():
    from h3_audio_t8_pkg.director_radar import verify_audio_timing

    _project, packet, _sid = project_with_source()
    packet["source"].update(start_pts=-90000, end_pts=0)
    packet["claims"] = [
        {
            "id": uid(),
            "kind": "visual",
            "text": "观察",
            "start_pts": -45000,
            "end_pts": 0,
        }
    ]
    mapping = evidence_time_map(seal_packet(packet))
    assert mapping["claims"][0]["start"]["absolute_seconds"] == {"num": -1, "den": 2}
    assert mapping["claims"][0]["start"]["local_seconds"] == {"num": 1, "den": 2}
    audio = {
        "asset_id": uid(),
        "sha256": "a" * 64,
        "sample_rate": 48000,
        "start_sample": 0,
        "end_sample": 1024,
    }
    stream = {
        "kind": "audio",
        "index": 1,
        "sample_rate": 48000,
        "duration_pts": 1024,
        "timebase": {"num": 1, "den": 48000},
    }
    verify_audio_timing(audio, {"streams": [stream]})
    with pytest.raises(ValueError, match="唯一"):
        verify_audio_timing(audio, {"streams": [stream, {**stream, "index": 2}]})
    with pytest.raises(ValueError, match="sample_rate"):
        verify_audio_timing({**audio, "sample_rate": 24000}, {"streams": [stream]})
    with pytest.raises(ValueError, match="超出"):
        verify_audio_timing({**audio, "end_sample": 1025}, {"streams": [stream]})


def test_asr_requires_independent_audio_and_never_becomes_dialogue_automatically():
    project, packet, sid = project_with_source()
    packet["claims"][0]["kind"] = "asr"
    with pytest.raises(ValueError, match="音频来源"):
        seal_packet(packet)
    aid = uid()
    project["assets"].append(
        {"id": aid, "kind": "audio", "name": "voice.wav", "sha256": "b" * 64}
    )
    packet["audio_source"] = {
        "asset_id": aid,
        "sha256": "b" * 64,
        "sample_rate": 48000,
        "start_sample": 0,
        "end_sample": 48000,
    }
    packet["claims"][0].update(text="可能是“今天晴天”", confidence=1.0)
    project = apply_operation(project, sid, "evidence_import", packet)
    assert evidence_status(project, project["doc"]["shots"][0]) == "unreviewed"
    project = review(project, sid)
    project = apply_operation(project, sid, "candidate_create", {"facts": True})
    assert "[已确认asr证据]" in project["doc"]["shots"][0]["promptCandidate"]["text"]
    assert project["doc"]["shots"][0]["events"] == []


def test_updated_packet_keeps_old_confirmation_stale_not_resigned():
    project, packet, sid = project_with_source()
    project = review(apply_operation(project, sid, "evidence_import", packet), sid)
    old = deepcopy(project["doc"]["shots"][0]["sourceEvidence"]["review"])
    packet["revision"] = 2
    packet["claims"][0].update(id=uid(), text="不同观察")
    project = apply_operation(project, sid, "evidence_import", packet)
    shot = project["doc"]["shots"][0]
    assert shot["sourceEvidence"]["review"] == old
    assert evidence_status(project, shot) == "review_stale"
    with pytest.raises(ValueError, match="审核"):
        apply_operation(project, sid, "candidate_create", {"facts": True})


def test_source_same_name_new_bytes_invalidates_review_and_candidate():
    project, packet, sid = project_with_source()
    project = review(apply_operation(project, sid, "evidence_import", packet), sid)
    project = accept(
        apply_operation(project, sid, "candidate_create", {"facts": True}), sid
    )
    project["assets"][0]["sha256"] = "c" * 64
    shot = project["doc"]["shots"][0]
    assert evidence_status(project, shot) == "source_changed"
    assert candidate_status(project, shot) == "stale"
    with pytest.raises(ValueError, match="改变"):
        review(project, sid)


def test_rules_order_inheritance_local_disable_snapshots_survive_missing_library():
    project = new_project()
    sid = project["current"]
    project["doc"]["shots"][0]["simplePrompt"] = "作者原文"
    shared = bind_skill(
        make_skill(
            "全片",
            "cinematic ${style}",
            {"style": {"type": "string", "default": "night"}},
        )
    )
    first = bind_skill(make_skill("本镜一", "FIRST"))
    second = bind_skill(make_skill("本镜二", "SECOND"))
    project = apply_operation(
        project, sid, "skills_update", {"shared": [shared], "local": [first, second]}
    )
    project = apply_operation(project, sid, "candidate_create", {})
    assert (
        project["doc"]["shots"][0]["promptCandidate"]["text"]
        == "作者原文\n\ncinematic night\n\nFIRST\n\nSECOND"
    )
    project = apply_operation(
        project,
        sid,
        "skills_update",
        {"local": [second, first], "disabled": [shared["id"]], "library": []},
    )
    assert candidate_status(project, project["doc"]["shots"][0]) == "stale"
    project = apply_operation(project, sid, "candidate_create", {})
    assert (
        project["doc"]["shots"][0]["promptCandidate"]["text"]
        == "作者原文\n\nSECOND\n\nFIRST"
    )
    assert project["doc"]["sharedSkills"] == [shared]
    project = apply_operation(
        project, sid, "skills_update", {"inherit": False, "disabled": []}
    )
    assert project["doc"]["sharedSkills"] == [shared]


def test_same_id_version_conflict_rejected_explicit_new_version_allowed():
    project = new_project()
    sid = project["current"]
    first = make_skill("旧", "one")
    second = make_skill("新", "two", skill_id=first["id"])
    with pytest.raises(ValueError, match="冲突"):
        apply_operation(
            project,
            sid,
            "skills_update",
            {"shared": [bind_skill(first)], "local": [bind_skill(second)]},
        )
    second = make_skill("新", "two", skill_id=first["id"], version=2)
    assert validate_project(
        apply_operation(
            project,
            sid,
            "skills_update",
            {"shared": [bind_skill(first)], "local": [bind_skill(second)]},
        )
    )


def test_literal_template_parameters_one_pass_no_eval_or_url_fetch():
    skill = make_skill(
        "规则",
        '${word} {{__import__("os")}} https://example.invalid',
        {"word": {"type": "string", "max_length": 100}},
    )
    row = bind_skill(skill, {"word": "${word}; $(command)"})
    assert (
        render_binding(row)
        == '${word}; $(command) {{__import__("os")}} https://example.invalid'
    )
    for params in (
        {},
        {"word": 2},
        {"word": "a" * 101},
        {"word": "ok", "extra": "bad"},
    ):
        with pytest.raises(ValueError):
            bind_skill(skill, params)


def test_active_candidate_executes_explicitly_and_stale_refuses_without_altering_raw_or_clock():
    project = new_project()
    sid = project["current"]
    shot = project["doc"]["shots"][0]
    shot.update(
        writingMode="advanced",
        simplePrompt="非活动原稿",
        prompt="原高级稿",
        events=[{"id": uid(), "start": 0, "end": 1, "text": "他说：“你好”。"}],
    )
    baseline = compile_project(project)
    project = apply_operation(
        project,
        sid,
        "skill_create",
        {"name": "镜头", "text": "slow zoom", "scope": "shot"},
    )
    project = accept(apply_operation(project, sid, "candidate_create", {}), sid)
    compiled = compile_project(project)
    assert compiled["ready"]
    assert "slow zoom" in compiled["shots"][0]["local_prompt"]
    for key in ("events", "time", "canvas", "audio_mode", "media_map"):
        assert compiled["shots"][0][key] == baseline["shots"][0][key]
    assert (
        compiled["shots"][0]["source_drafts"] == baseline["shots"][0]["source_drafts"]
    )
    project["doc"]["shots"][0]["prompt"] = "新作者稿"
    assert not compile_project(project)["ready"]
    assert "失效" in str(compile_project(project)["errors"]) or "改变" in str(
        compile_project(project)["errors"]
    )
    project = apply_operation(project, sid, "candidate_revert", {})
    assert (
        compile_project(project)["ready"]
        and compile_project(project)["shots"][0]["local_prompt"] == "新作者稿"
    )


def test_candidate_uses_real_existing_reference_alias_translation():
    project = new_project()
    sid = project["current"]
    aid = uid()
    shot = project["doc"]["shots"][0]
    project["assets"] = [
        {
            "id": aid,
            "name": "ref.png",
            "kind": "image",
            "sha256": "a" * 64,
            "width": 32,
            "height": 32,
        }
    ]
    shot.update(mode="refs", refs=[aid], tray=[aid], simplePrompt="@image1 微笑")
    project = apply_operation(
        project,
        sid,
        "skill_create",
        {"name": "参考", "text": "@image1 保持衣着", "scope": "shot"},
    )
    project = accept(apply_operation(project, sid, "candidate_create", {}), sid)
    result = compile_project(project)
    assert result["ready"] and result["shots"][0]["prompt"].count("<Picture 1>") == 2
    project["doc"]["shots"][0]["refs"] = []
    assert not compile_project(project)["ready"]


def test_explicit_cross_shot_dependency_is_scoped_and_shot_selection_preserves_context():
    project, packet, sid = project_with_source()
    project = review(apply_operation(project, sid, "evidence_import", packet), sid)
    second = deepcopy(project["doc"]["shots"][0])
    second.update(id=uid(), name="第二镜")
    second.pop("sourceEvidence")
    project["doc"]["shots"].append(second)
    dependency = {
        "shot_id": sid,
        "evidence_sha256": project["doc"]["shots"][0]["sourceEvidence"]["packet"][
            "sha256"
        ],
    }
    project = apply_operation(
        project,
        second["id"],
        "intent_update",
        {"text": "沿用已确认红衣", "reason": "", "dependencies": [dependency]},
    )
    project = accept(
        apply_operation(project, second["id"], "candidate_create", {}), second["id"]
    )
    assert compile_project(project, shot_id=second["id"])["ready"]
    project["doc"]["shots"][0]["simplePrompt"] = "无关作者文字修改"
    assert candidate_status(project, project["doc"]["shots"][1]) == "active"
    changed = deepcopy(packet)
    changed["revision"] = 2
    changed["claims"][0]["text"] = "改变事实"
    project = apply_operation(project, sid, "evidence_import", changed)
    assert candidate_status(project, project["doc"]["shots"][1]) == "stale"
    with pytest.raises(ValueError, match="跨镜事实"):
        apply_operation(project, second["id"], "candidate_create", {})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(schema="other"),
        lambda p: p.update(revision=True),
        lambda p: p.update(secret="do not silently serialize"),
        lambda p: p["source"].update(asset_id="../escape"),
        lambda p: p["source"].update(sha256="bad"),
        lambda p: p["source"]["timebase"].update(den=0),
        lambda p: p["source"].update(end_pts=1),
        lambda p: p["claims"][0].update(kind=[]),
        lambda p: p["claims"][0].update(confidence=float("nan")),
        lambda p: p["claims"][0].update(start_pts=1),
        lambda p: p["claims"][0].update(text="x" * 32769),
        lambda p: p["claims"].append(deepcopy(p["claims"][0])),
    ],
)
def test_bad_evidence_rejected_without_touching_original(mutation):
    project, packet, sid = project_with_source()
    original = deepcopy(project)
    mutation(packet)
    with pytest.raises(ValueError):
        apply_operation(project, sid, "evidence_import", packet)
    assert project == original


def test_sha_corruption_review_adoption_and_large_arrays_rejected():
    _project, packet, _sid = project_with_source()
    packet = seal_packet(packet)
    packet["claims"][0]["text"] = "损坏"
    with pytest.raises(ValueError, match="SHA"):
        validate_packet(packet)
    row = bind_skill(make_skill("规则", "文字"))
    row["snapshot"]["text_snapshot"] = "不同字节"
    with pytest.raises(ValueError, match="SHA"):
        validate_bindings([row])
    with pytest.raises(ValueError, match="256"):
        validate_bindings([row] * 257)
    project = new_project()
    sid = project["current"]
    project["doc"]["shots"][0]["simplePrompt"] = "original"
    project = apply_operation(project, sid, "candidate_create", {})
    for value in (
        {
            "confirm": False,
            "text_sha256": project["doc"]["shots"][0]["promptCandidate"]["text_sha256"],
        },
        {"confirm": True, "text_sha256": "f" * 64},
    ):
        with pytest.raises(ValueError, match="显式采用"):
            apply_operation(project, sid, "candidate_accept", value)


def test_legacy_save_receipt_and_new_skill_snapshot_survive_real_store_and_bundle(
    tmp_path,
):
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    project = new_project()
    sid = project["current"]
    project["doc"]["shots"][0]["simplePrompt"] = "作者稿"
    saved = store.save(project, 0)
    old_sha = sha(saved)
    next_draft = apply_operation(
        saved,
        sid,
        "skill_create",
        {
            "name": "规则",
            "text": "cinematic ${mood}",
            "parameter_schema": {"mood": {"type": "string", "default": "calm"}},
            "scope": "shot",
        },
    )
    next_draft = accept(apply_operation(next_draft, sid, "candidate_create", {}), sid)
    assert (
        sha(store.load(project["id"])) == old_sha
    )  # Operations never write existing frozen project.
    saved2 = store.save(next_draft, 1)
    restored = ProjectStore(tmp_path / "user", tmp_path / "input").load(project["id"])
    assert (
        restored == saved2 and restored["doc"]["shots"][0]["simplePrompt"] == "作者稿"
    )
    ordinary = portable_project(saved2)
    assert (
        "promptCandidate" not in ordinary["doc"]["shots"][0]
        and "skillLibrary" not in ordinary["doc"]
    )
    output = tmp_path / "output"
    output.mkdir()
    plan = prepare_bundle(store, saved2, [], output, False, include_radar=True)
    assert plan["ready"], plan
    archive = build_bundle(store, project["id"], plan["id"])
    target = ProjectStore(tmp_path / "new-user", tmp_path / "new-input")
    output2 = tmp_path / "new-output"
    output2.mkdir()
    from test_director_bundle import upload

    upload_id = upload(target, archive)
    assert inspect_bundle(target, upload_id)["ready"]
    new_id = uid()
    import_bundle(target, output2, upload_id, new_id)
    imported = target.load(new_id)
    newshot = imported["doc"]["shots"][0]
    assert (
        newshot["id"] != sid
        and newshot["skillBindings"] == restored["doc"]["shots"][0]["skillBindings"]
    )
    assert newshot["promptCandidate"]["active"] is False
    assert newshot["simplePrompt"] == "作者稿"
    assert (
        newshot["promptCandidate"]["text"]
        == restored["doc"]["shots"][0]["promptCandidate"]["text"]
    )
    imported["doc"]["skillLibrary"] = []
    new_id = newshot["id"]
    imported = accept(apply_operation(imported, new_id, "candidate_create", {}), new_id)
    assert (
        compile_project(imported)["ready"]
        and "cinematic calm" in compile_project(imported)["shots"][0]["prompt"]
    )


def test_evidence_bundle_true_media_new_ids_retains_review_stale_and_no_sampling_ancestor(
    tmp_path,
):
    from test_director_idempotency import _write_valid_video
    from test_director_bundle import upload

    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    aid = uid()
    source = store.input_root / "t8_director" / aid / "source.mp4"
    source.parent.mkdir(parents=True)
    _write_valid_video(source)
    asset = store.register_asset(source, aid, "source.mp4")
    project, packet, sid = project_with_source()
    project["assets"] = [asset]
    packet["source"].update(asset_id=aid, sha256=asset["sha256"])
    project = review(apply_operation(project, sid, "evidence_import", packet), sid)
    old_review = deepcopy(project["doc"]["shots"][0]["sourceEvidence"]["review"])
    old_evidence = deepcopy(project["doc"]["shots"][0]["sourceEvidence"])
    packet["revision"] = 2
    packet["claims"][0]["text"] = "新的独立观察"
    project = review(apply_operation(project, sid, "evidence_import", packet), sid)
    history = deepcopy(project["doc"]["shots"][0]["evidenceHistory"])
    assert old_evidence in history
    old_review = deepcopy(project["doc"]["shots"][0]["sourceEvidence"]["review"])
    saved = store.save(project, 0)
    output = tmp_path / "output"
    output.mkdir()
    plan = prepare_bundle(store, saved, [], output, False, include_radar=True)
    assert plan["ready"], plan
    archive = build_bundle(store, project["id"], plan["id"])
    target = ProjectStore(tmp_path / "fresh-user", tmp_path / "fresh-input")
    output2 = tmp_path / "fresh-output"
    output2.mkdir()
    upload_id = upload(target, archive)
    inspect_bundle(target, upload_id)
    new_id = uid()
    import_bundle(target, output2, upload_id, new_id)
    imported = target.load(new_id)
    shot = imported["doc"]["shots"][0]
    assert (
        shot["sourceEvidence"]["packet"]["source"]["asset_id"]
        == imported["assets"][0]["id"]
        != aid
    )
    assert (
        shot["sourceEvidence"]["review"] == old_review
        and evidence_status(imported, shot) == "review_stale"
    )
    assert shot["evidenceHistory"][: len(history)] == history
    assert old_evidence in shot["evidenceHistory"]
    assert shot["sourceEvidence"]["packet"]["revision"] == 3
    assert "StageResult" not in json.dumps(imported) and "seed" not in shot


def test_default_bundle_omits_evidence_only_original_media_and_private_transcript():
    project, packet, sid = project_with_source()
    project = review(apply_operation(project, sid, "evidence_import", packet), sid)
    packet_sha = project["doc"]["shots"][0]["sourceEvidence"]["packet"]["sha256"]
    ordinary = portable_project(project)
    assert ordinary["assets"] == []
    assert "sourceEvidence" not in ordinary["doc"]["shots"][0]
    assert packet_sha not in json.dumps(ordinary)
    explicit = portable_project(project, include_radar=True)
    assert explicit["assets"] == project["assets"]
    assert (
        explicit["doc"]["shots"][0]["sourceEvidence"]
        == project["doc"]["shots"][0]["sourceEvidence"]
    )
    # A legacy reference retains its media, even if it is also cited by evidence.
    project["doc"]["shots"][0]["tray"] = [project["assets"][0]["id"]]
    assert portable_project(project)["assets"] == project["assets"]
