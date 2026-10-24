"""Frozen-film caption sidecars using actual tiny PyAV media, CPU only.

No ASR/model, browser, media re-encoding by the caption exporter, or adoption.
The over-one-hour renderer check is a rational-clock unit test, not hour video.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import shutil
import uuid

import pytest
import torch

from h3_audio_t8_pkg import director_captions as captions
from h3_audio_t8_pkg.director_film import film_media, prepare_film
from h3_audio_t8_pkg.director_project import file_sha, sha
from h3_audio_t8_pkg.subtitle_text import render_subtitles, subtitle_timestamp
from test_director_film import setup, video


def frozen(tmp_path, *, trim=False, silent_second=False, third=False):
    store, output, project, records = setup(tmp_path)
    project["doc"]["global"] = "这不是字幕，不能自动抄写全片提示词"
    if trim:
        project["doc"]["shots"][0]["filmTrim"] = {"in_frame": 2, "out_frame": 10}
    if silent_second:
        video(output / "1.mp4", 150, audio=False)
    if third:
        shot = deepcopy(project["doc"]["shots"][1])
        shot.update(id=str(uuid.uuid4()), name="third-no-guessed-dialogue", adoptedResultId="version-third")
        project["doc"]["shots"].append(shot)
        video(output / "2.mp4", 110)
        records.append({"shot_id": shot["id"], "prompt_id": shot["adoptedResultId"], "state": "success",
                        "outputs": {"save": {"images": [{"filename": "2.mp4", "type": "output"}]}}})
    film = prepare_film(store, project, records, output)
    assert film["ready"], film.get("errors")
    return store, output, project, records, film


@pytest.fixture
def prepared(tmp_path):
    return frozen(tmp_path)


def track(entry, rows, *, unit="frames", rate=None, source=None):
    return {"shot_id": entry["shot_id"], "version_id": entry["version_id"],
            "media_sha256": entry["media_sha256"],
            "clock": {"unit": unit, **({"rate": rate} if rate is not None else {})},
            "source": source or {"kind": "manual_planned_text"}, "cues": rows}


def cue(start, end, text="明确计划字幕"):
    return {"start": start, "end": end, "text": text}


def payload(film, tracks=()):
    return {"schema": captions.INPUT_SCHEMA, "film_sha256": film["sha256"], "tracks": list(tracks)}


def rational(value):
    return Fraction(value["num"], value["den"])


def snapshot(store, output, film):
    paths = [store.root / "films" / (film["id"] + ".json"), *output.glob("*.mp4")]
    paths.extend(film_media(store, film["project_id"], film["id"], index) for index in range(len(film["entries"])))
    return {path: path.read_bytes() for path in paths}


def assert_unchanged(before):
    assert all(path.read_bytes() == value for path, value in before.items())


def prepare(store, film, data, caption_id=None):
    caption_id = caption_id or str(uuid.uuid4())
    result, created = captions.prepare_captions(store, film["project_id"], film["id"], caption_id, data)
    return caption_id, result, created


def test_old_frozen_film_empty_subtitles_do_not_import_prompts_or_change_media(prepared):
    store, output, _, _, film = prepared
    before = snapshot(store, output, film)
    assert "captions" not in film
    cid, result, created = prepare(store, film, payload(film))
    assert created and result["cues"] == [] and result["notices"] == []
    assert rational(result["duration"]) == Fraction(1)
    assert result["text_status"] == "planned_text_not_asr_or_spoken_alignment_verified"
    assert result["media_modified"] is False
    assert captions.caption_file(store, film["project_id"], film["id"], cid, "srt").read_bytes() == b""
    assert captions.caption_file(store, film["project_id"], film["id"], cid, "vtt").read_text() == "WEBVTT\n\n"
    assert_unchanged(before)
    assert not torch.cuda.is_initialized()


def test_trim_clips_times_not_characters_and_silent_no_dialogue_shot_advances(tmp_path):
    store, output, _, _, film = frozen(tmp_path, trim=True, silent_second=True, third=True)
    before = snapshot(store, output, film)
    first, silent, third = film["entries"]
    assert not silent["media"]["has_audio"]
    text = "整句保留，不能按裁切比例删字。"
    data = payload(film, [track(first, [cue(0, 4, text), cue(8, 12, text), cue(11, 12, "裁外")]),
                          track(third, [cue(0, 12, "第三镜")])])
    _, result, _ = prepare(store, film, data)
    assert len(result["cues"]) == 3
    assert result["cues"][0]["text"] == result["cues"][1]["text"] == text
    assert [(rational(row["start"]), rational(row["end"])) for row in result["cues"]] == [
        (Fraction(0), Fraction(1, 12)), (Fraction(1, 4), Fraction(1, 3)),
        (Fraction(5, 6), Fraction(4, 3)),
    ]
    assert sum(item["reason"] == "partially_trimmed_text_needs_review" for item in result["notices"]) == 2
    assert any(item["reason"] == "outside_adopted_trim_omitted" for item in result["notices"])
    assert rational(result["duration"]) == Fraction(4, 3)
    assert_unchanged(before)


def test_sample_clock_is_exact_rational_then_frame_trim_and_absolute_offset(tmp_path):
    store, _, _, _, film = frozen(tmp_path, trim=True)
    first, second = film["entries"]
    data = payload(film, [track(first, [cue(6001, 10002)], unit="samples", rate=44100),
                          track(second, [cue(1, 8001, "后镜")], unit="samples", rate=48000)])
    cid, result, _ = prepare(store, film, data)
    expected = [(Fraction(6001, 44100) - Fraction(1, 12), Fraction(10002, 44100) - Fraction(1, 12)),
                (Fraction(1, 3) + Fraction(1, 48000), Fraction(1, 3) + Fraction(8001, 48000))]
    assert [(rational(row["start"]), rational(row["end"])) for row in result["cues"]] == expected
    srt = captions.caption_file(store, film["project_id"], film["id"], cid, "srt").read_text(encoding="utf-8")
    assert all(subtitle_timestamp(start) + " --> " + subtitle_timestamp(end) in srt for start, end in expected)


def test_unicode_multiline_markup_and_paragraphs_are_literal_not_new_records(prepared):
    store, output, _, _, film = prepared
    before = snapshot(store, output, film)
    text = '中文🙂 <b>不是加粗</b> & "引号"\r\n\r\n第二段 <script>alert(1)</script>'
    data = payload(film, [track(film["entries"][0], [cue(0, 12, text)])])
    original = deepcopy(data)
    cid, result, _ = prepare(store, film, data)
    assert data == original
    assert result["cues"][0]["text"] == text.replace("\r\n", "\n")
    for format in ("srt", "vtt"):
        path = captions.caption_file(store, film["project_id"], film["id"], cid, format)
        body = path.read_text(encoding="utf-8")
        assert "中文🙂 &lt;b&gt;不是加粗&lt;/b&gt; &amp;" in body
        assert "\n\u200b\n第二段 &lt;script&gt;" in body
        assert "<script>" not in body and body.count(" --> ") == 1
        assert file_sha(path) == result["formats"][format]["sha256"]
    data["tracks"][0]["cues"][0]["text"] = "改当前草稿不改附件"
    assert captions.load_captions(store, film["project_id"], film["id"], cid) == result
    assert_unchanged(before)


def test_overlaps_are_retained_and_empty_submillisecond_cues_are_reported(prepared):
    store, _, _, _, film = prepared
    data = payload(film, [track(film["entries"][0], [cue(0, 8, "第一条"), cue(4, 12, "第二条"),
                                                       cue(0, 1, " \n ")]),
                          track(film["entries"][1], [cue(1, 2, "不足毫秒")], unit="samples", rate=48000)])
    _, result, _ = prepare(store, film, data)
    assert [row["text"] for row in result["cues"]] == ["第一条", "第二条"]
    reasons = {row["reason"] for row in result["notices"]}
    assert {"overlapping_cues_retained", "empty_text_omitted", "below_millisecond_resolution_omitted"} <= reasons
    assert all(rational(row["end"]) <= rational(result["duration"]) for row in result["cues"])


def test_pure_serializer_crosses_hour_on_rational_clock_without_float_accumulation():
    start, end = Fraction(3600) + Fraction(1, 24), Fraction(3601) + Fraction(1, 3)
    rows = [{"text": "跨小时", "start_seconds": start, "end_seconds": end}]
    assert "01:00:00,042 --> 01:00:01,333" in render_subtitles(rows, "srt", literal_text=True)
    assert "01:00:00.042 --> 01:00:01.333" in render_subtitles(rows, "vtt", literal_text=True)


def test_actual_speech_timeline_requires_explicit_adopted_audio_binding(prepared):
    from h3_audio_t8_pkg.speech import assemble_speech_audio
    from test_speech import described_profile, make_audio, one_segment_plan

    store, _, _, _, film = prepared
    plan = one_segment_plan(described_profile())
    _, report_text, _, _ = assemble_speech_audio(
        plan, {"audio_segment_0": make_audio(.5, 48000, value=0.)}, 48000, 0., -1.,
    )
    report = json.loads(report_text)
    original = deepcopy(report)
    entry = film["entries"][0]
    with pytest.raises(ValueError, match="明确确认"):
        captions.track_from_speech_report(entry, report)
    bound = captions.track_from_speech_report(entry, report, confirmed_media_match=True)
    assert bound["source"]["report_sha256"] == sha(report)
    assert bound["source"]["binding"] == "user_declared_media_match_not_independently_verified"
    cid, result, _ = prepare(store, film, payload(film, [bound]))
    assert result["cues"][0]["text"].startswith(report["timeline"][0]["speaker_id"] + ": ")
    assert "not_asr" in result["text_status"]
    assert captions.load_captions(store, film["project_id"], film["id"], cid) == result
    assert report == original
    silent_entry = deepcopy(entry)
    silent_entry["media"]["has_audio"] = False
    with pytest.raises(ValueError, match="没有音轨"):
        captions.track_from_speech_report(silent_entry, report, confirmed_media_match=True)


@pytest.mark.parametrize("damage", ["film_sha", "media_sha", "version", "shot", "source", "speech_unbound"])
def test_wrong_film_track_provenance_never_publishes(prepared, damage):
    store, output, _, _, film = prepared
    before = snapshot(store, output, film)
    row = track(film["entries"][0], [cue(0, 12)])
    data = payload(film, [row])
    if damage == "film_sha":
        data["film_sha256"] = "0" * 64
    elif damage == "media_sha":
        row["media_sha256"] = "0" * 64
    elif damage == "version":
        row["version_id"] = "some-newer-unadopted-version"
    elif damage == "shot":
        row["shot_id"] = str(uuid.uuid4())
    elif damage == "source":
        row["source"] = {"kind": "automatically_extracted_prompt"}
    else:
        row["source"] = {"kind": "speech_planned_text", "report_sha256": "a" * 64}
    cid = str(uuid.uuid4())
    with pytest.raises(ValueError):
        prepare(store, film, data, cid)
    assert not (store.root / "film_captions" / cid).exists()
    assert_unchanged(before)


def test_owner_version_reorder_and_idempotent_collision_keep_old_attachment(prepared):
    store, output, project, records, film = prepared
    before = snapshot(store, output, film)
    data = payload(film, [track(film["entries"][0], [cue(0, 12, "旧字幕")])])
    cid, first, created = prepare(store, film, data)
    assert created
    assert prepare(store, film, data, cid) == (cid, first, False)
    root = store.root / "film_captions" / cid
    original = {path.name: path.read_bytes() for path in root.iterdir()}
    changed = deepcopy(data)
    changed["tracks"][0]["cues"][0]["text"] = "同身份不能覆盖新稿"
    with pytest.raises(ValueError, match="另一份稿件"):
        prepare(store, film, changed, cid)
    with pytest.raises(ValueError):
        captions.load_captions(store, str(uuid.uuid4()), film["id"], cid)
    project["doc"]["shots"].reverse()
    new_film = prepare_film(store, project, records, output)
    with pytest.raises(ValueError, match="冻结整片"):
        prepare(store, new_film, data)
    remapped = deepcopy(data)
    remapped["film_sha256"] = new_film["sha256"]
    _, after_reorder, _ = prepare(store, new_film, remapped)
    assert rational(after_reorder["cues"][0]["start"]) == Fraction(1, 2)
    assert captions.load_captions(store, film["project_id"], film["id"], cid) == first
    assert {path.name: path.read_bytes() for path in root.iterdir()} == original
    assert_unchanged(before)


def test_competing_complete_same_revision_returns_existing_without_overwrite(prepared, monkeypatch):
    store, output, _, _, film = prepared
    before = snapshot(store, output, film)
    cid = str(uuid.uuid4())
    destination = store.root / "film_captions" / cid
    data = payload(film, [track(film["entries"][0], [cue(0, 12)])])
    original_rename = Path.rename
    observed = []

    def competing_commit(path, target):
        if path.name == "ready" and Path(target) == destination:
            # Exercise the real collision-validation branch with a complete
            # competing artifact; this is not a multiprocess stress claim.
            shutil.copytree(path, target)
            observed.append({child.name: child.read_bytes() for child in Path(target).iterdir()})
            raise FileExistsError("explicit-competing-complete-revision")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", competing_commit)
    actual_cid, result, created = prepare(store, film, data, cid)
    assert actual_cid == cid and created is False
    assert observed == [{child.name: child.read_bytes() for child in destination.iterdir()}]
    assert captions.load_captions(store, film["project_id"], film["id"], cid) == result
    assert_unchanged(before)


def test_unsafe_caption_identity_or_format_cannot_escape_attachment_directory(prepared):
    store, _, _, _, film = prepared
    data = payload(film)
    with pytest.raises(ValueError, match="UUID|身份"):
        prepare(store, film, data, "../escaped")
    cid, _, _ = prepare(store, film, data)
    with pytest.raises(ValueError, match="格式"):
        captions.caption_file(store, film["project_id"], film["id"], cid, "../captions.srt")


def test_corrupt_frozen_media_before_first_prepare_creates_no_attachment(prepared):
    store, output, _, _, film = prepared
    originals = {path: path.read_bytes() for path in output.glob("*.mp4")}
    cid = str(uuid.uuid4())
    data = payload(film, [track(film["entries"][0], [cue(0, 12)])])
    film_media(store, film["project_id"], film["id"], 0).write_bytes(b"explicit-corrupted-frozen-copy")
    with pytest.raises(ValueError, match="变化"):
        prepare(store, film, data, cid)
    assert not (store.root / "film_captions" / cid).exists()
    assert_unchanged(originals)


@pytest.mark.parametrize("damage", ["srt", "vtt_missing", "manifest", "media"])
def test_corrupt_or_half_attachment_is_never_delivered_or_overwritten(prepared, damage):
    store, _, _, _, film = prepared
    data = payload(film, [track(film["entries"][0], [cue(0, 12)])])
    cid, _, _ = prepare(store, film, data)
    root = store.root / "film_captions" / cid
    if damage == "srt":
        (root / "captions.srt").write_bytes(b"half-file")
    elif damage == "vtt_missing":
        (root / "captions.vtt").unlink()
    elif damage == "manifest":
        value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        value["cues"][0]["text"] = "tampered"
        (root / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
    else:
        film_media(store, film["project_id"], film["id"], 0).write_bytes(b"changed-media")
    broken = {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises((ValueError, FileNotFoundError)):
        captions.caption_file(store, film["project_id"], film["id"], cid, "srt")
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare(store, film, data, cid)
    assert {path.name: path.read_bytes() for path in root.iterdir()} == broken


@pytest.mark.parametrize("failure", ["second_format_write", "manifest_write"])
def test_io_failure_before_atomic_publish_leaves_no_downloadable_partial(prepared, monkeypatch, failure):
    store, output, _, _, film = prepared
    before = snapshot(store, output, film)
    data = payload(film, [track(film["entries"][0], [cue(0, 12)])])
    cid = str(uuid.uuid4())
    original_write = Path.write_text
    if failure == "second_format_write":
        def interrupted(path, text, *args, **kwargs):
            if path.name == "captions.vtt":
                original_write(path, "WEBVTT", *args, **kwargs)
                raise OSError("explicit-second-format-write-failure")
            return original_write(path, text, *args, **kwargs)
        monkeypatch.setattr(Path, "write_text", interrupted)
    else:
        monkeypatch.setattr(captions, "atomic_json", lambda *_: (_ for _ in ()).throw(OSError("explicit-manifest-write-failure")))
    with pytest.raises(OSError, match="explicit-"):
        prepare(store, film, data, cid)
    assert not (store.root / "film_captions" / cid).exists()
    with pytest.raises(FileNotFoundError):
        captions.caption_file(store, film["project_id"], film["id"], cid, "srt")
    assert not list((store.root / "film_caption_work").iterdir())
    assert_unchanged(before)


@pytest.mark.parametrize("row", [cue(True, 2), cue(0, 0), cue(2, 1), cue(0, 2, "bad\x00text")])
def test_invalid_cue_coordinates_and_text_fail_before_publication(prepared, row):
    store, _, _, _, film = prepared
    cid = str(uuid.uuid4())
    with pytest.raises(ValueError):
        prepare(store, film, payload(film, [track(film["entries"][0], [row])]), cid)
    assert not (store.root / "film_captions" / cid).exists()
