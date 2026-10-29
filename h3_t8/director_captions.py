"""Independent, immutable planned-text captions on a frozen film's frame axis.

    No model, ASR, mixing, encoding or adoption. A Speech timeline import is an
    explicit user-declared audio binding, not proof the delivered actor spoke it.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import tempfile
import threading

from .director_film import film_media, load_film
from .director_project import atomic_json, contained, file_sha, identity, sha
from .subtitle_text import render_subtitles


SCHEMA = "t8.director.captions.v1"
INPUT_SCHEMA = "t8.director.caption_input.v1"
_LOCK = threading.RLock()


def _rational(value):
    return {"num": value.numerator, "den": value.denominator}


def _fps(entry):
    value = entry["media"]["fps"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("冻结成片缺少有效实测帧率")
    actual = entry.get("decoded_clock", {}).get("video", {}).get("fps")
    if actual is not None:
        rate = Fraction(actual["num"], actual["den"])
        if rate <= 0 or float(rate) != value:
            raise ValueError("冻结成片帧率与实测时钟不一致")
        return rate
    return Fraction(str(value))


def _integer(value, label, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{label}必须是{'正' if positive else '非负'}整数")
    return value


def _text(value):
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("字幕正文必须是无NUL字符的文字")
    return value.replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def track_from_speech_report(entry, report, *, confirmed_media_match=False):
    """Bind an explicitly chosen Speech report without fabricating media proof."""
    if confirmed_media_match is not True:
        raise ValueError("请明确确认Speech时间表对应此采用音轨；系统未独立核实")
    if not entry["media"].get("has_audio"):
        raise ValueError("此采用成片没有音轨，不能绑定Speech音频时间表")
    if not isinstance(report, dict) or report.get("schema") != "minimax_h3_t8_speech_timeline_v1":
        raise ValueError("请选择Speech拼接输出的完整时间表报告")
    rate = _integer(report.get("sample_rate"), "Speech采样率", positive=True)
    if not isinstance(report.get("timeline"), list):
        raise ValueError("Speech报告缺少timeline列表")
    cues = []
    for item in report["timeline"]:
        if not isinstance(item, dict):
            raise ValueError("Speech时间表条目无效")
        speaker = item.get("speaker_id")
        if not isinstance(speaker, str):
            raise ValueError("Speech时间表缺少明确说话人")
        cues.append({"start": item.get("start_sample"), "end": item.get("end_sample"),
                     "text": f"{speaker}: {_text(item.get('text'))}"})
    return {"shot_id": entry["shot_id"], "version_id": entry["version_id"],
            "media_sha256": entry["media_sha256"], "clock": {"unit": "samples", "rate": rate},
            "source": {"kind": "speech_planned_text", "report_sha256": sha(report),
                       "binding": "user_declared_media_match_not_independently_verified"}, "cues": cues}


def _compile(film, payload):
    if not isinstance(payload, dict) or payload.get("schema") != INPUT_SCHEMA:
        raise ValueError("字幕输入schema无效")
    if payload.get("film_sha256") != film["sha256"]:
        raise ValueError("字幕稿不属于当前冻结整片，未自动套用到新版本")
    tracks = payload.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("字幕tracks必须是列表；无台词可使用空列表")
    entries = {entry["shot_id"]: entry for entry in film["entries"]}
    if len(entries) != len(film["entries"]):
        raise ValueError("冻结整片存在重复镜头身份")
    selected = {}
    for track in tracks:
        if not isinstance(track, dict) or track.get("shot_id") not in entries:
            raise ValueError("字幕条目缺少冻结整片中的明确镜头身份")
        sid = track["shot_id"]
        if sid in selected:
            raise ValueError("同镜字幕轨重复；请把该镜多条字幕放入同一cues列表")
        entry = entries[sid]
        if track.get("version_id") != entry["version_id"] or track.get("media_sha256") != entry["media_sha256"]:
            raise ValueError("字幕采用版本或媒体SHA已变化，拒绝重新定向")
        selected[sid] = track
    cursor, cues, notices = Fraction(0), [], []
    for entry in film["entries"]:
        rate = _fps(entry)
        begin = Fraction(_integer(entry["in_frame"], "采用起始帧"), 1) / rate
        finish = Fraction(_integer(entry["out_frame"], "采用结束帧", positive=True), 1) / rate
        if finish <= begin:
            raise ValueError("采用结束帧必须大于起始帧")
        track = selected.get(entry["shot_id"])
        if track is not None:
            source, clock = track.get("source"), track.get("clock")
            if not isinstance(source, dict) or source.get("kind") not in {"manual_planned_text", "speech_planned_text", "recognized_audio"}:
                raise ValueError("字幕须有明确的手写、计划或实际识别来源，不自动提取提示词")
            if source['kind'] == 'recognized_audio':
                identity(source.get('asr_job_id'))
                if (source.get('binding') != 'actual_frozen_soundtrack_full_source_zero'
                        or any(not isinstance(source.get(key), str) or len(source[key]) != 64
                               or any(c not in '0123456789abcdef' for c in source[key])
                               for key in ('report_sha256', 'audio_sha256'))):
                    raise ValueError('识别字幕缺少实际音轨与识别回执绑定')
            if source["kind"] == "speech_planned_text":
                if (not entry["media"].get("has_audio")
                        or source.get("binding") != "user_declared_media_match_not_independently_verified"
                        or not isinstance(source.get("report_sha256"), str)
                        or len(source["report_sha256"]) != 64
                        or any(char not in "0123456789abcdef" for char in source["report_sha256"])):
                    raise ValueError("Speech字幕缺少明确的采用音轨绑定与来源摘要")
            if not isinstance(clock, dict) or clock.get("unit") not in {"frames", "samples"}:
                raise ValueError("字幕clock须明确frames或samples")
            clock_rate = rate if clock["unit"] == "frames" else Fraction(_integer(clock.get("rate"), "字幕采样率", positive=True))
            rows = track.get("cues")
            if not isinstance(rows, list):
                raise ValueError("字幕cues必须是列表")
            for index, raw in enumerate(rows):
                if not isinstance(raw, dict):
                    raise ValueError("字幕cue必须是对象")
                start = Fraction(_integer(raw.get("start"), "字幕开始坐标"), 1) / clock_rate
                end = Fraction(_integer(raw.get("end"), "字幕结束坐标", positive=True), 1) / clock_rate
                if end <= start:
                    raise ValueError("字幕结束坐标必须大于开始坐标")
                text = _text(raw.get("text"))
                tag = {"shot_id": entry["shot_id"], "cue_index": index}
                if not text.strip():
                    notices.append({**tag, "reason": "empty_text_omitted"})
                    continue
                left, right = max(start, begin), min(end, finish)
                if right <= left:
                    notices.append({**tag, "reason": "outside_adopted_trim_omitted"})
                    continue
                absolute_start, absolute_end = cursor + left - begin, cursor + right - begin
                if round(absolute_end * 1000) <= round(absolute_start * 1000):
                    notices.append({**tag, "reason": "below_millisecond_resolution_omitted"})
                    continue
                clipped = left != start or right != end
                if clipped:
                    notices.append({**tag, "reason": "partially_trimmed_text_needs_review"})
                cues.append({**tag, "version_id": entry["version_id"], "media_sha256": entry["media_sha256"],
                             "text": text, "start": _rational(absolute_start), "end": _rational(absolute_end),
                             "source": deepcopy(source), "partially_trimmed": clipped})
        cursor += finish - begin  # A shot without dialogue still advances the film clock.
    cues.sort(key=lambda cue: (Fraction(cue["start"]["num"], cue["start"]["den"]),
                               Fraction(cue["end"]["num"], cue["end"]["den"])))
    furthest = Fraction(0)
    for cue in cues:
        start, end = (Fraction(cue[key]["num"], cue[key]["den"]) for key in ("start", "end"))
        if start < furthest:
            notices.append({"shot_id": cue["shot_id"], "cue_index": cue["cue_index"], "reason": "overlapping_cues_retained"})
        furthest = max(furthest, end)
    return cues, notices, cursor


def _build(film, caption_id, payload):
    cues, notices, duration = _compile(film, payload)
    renderable = [{"text": cue["text"],
                   "start_seconds": Fraction(cue["start"]["num"], cue["start"]["den"]),
                   "end_seconds": Fraction(cue["end"]["num"], cue["end"]["den"])} for cue in cues]
    texts = {format: render_subtitles(renderable, format, literal_text=True) for format in ("srt", "vtt")}
    result = {"schema": SCHEMA, "id": identity(caption_id), "project_id": film["project_id"],
              "film_id": film["id"], "film_sha256": film["sha256"], "input": deepcopy(payload),
              "input_sha256": sha(payload), "cues": cues, "notices": notices, "duration": _rational(duration),
              "text_status": "planned_text_not_asr_or_spoken_alignment_verified", "media_modified": False,
              "serializer": "literal_text_v1", "formats": {
                  format: {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                           "bytes": len(text.encode("utf-8"))} for format, text in texts.items()}}
    if any(track['source']['kind'] == 'recognized_audio' for track in payload['tracks']):
        result['text_status'] = 'recognized_audio_draft_editable_not_speaker_or_lipsync_verified'
    result["sha256"] = sha(result)
    return result, texts


def _root(store, caption_id):
    return contained(store.root, f"film_captions/{identity(caption_id)}")


def load_captions(store, project_id, film_id, caption_id):
    film = load_film(store, project_id, film_id)
    root = _root(store, caption_id)
    result = json.loads(contained(root, "manifest.json").read_text(encoding="utf-8"))
    if (not isinstance(result, dict) or result.get("schema") != SCHEMA or result.get("id") != caption_id
            or result.get("project_id") != identity(project_id) or result.get("film_id") != identity(film_id)
            or result.get("film_sha256") != film["sha256"]):
        raise ValueError("字幕附件身份或冻结整片不一致")
    _verify_recognized_tracks(store, film, result.get('input'))
    expected, _ = _build(film, caption_id, result.get("input"))
    if result != expected:
        raise ValueError("字幕附件内容或校验已损坏")
    for index in range(len(film["entries"])):
        film_media(store, project_id, film_id, index)
    for format, evidence in result["formats"].items():
        path = contained(root, f"captions.{format}")
        if not path.is_file() or path.stat().st_size != evidence["bytes"] or file_sha(path) != evidence["sha256"]:
            raise ValueError("字幕文件已缺失或变化，不发布半文件")
    return result


def prepare_captions(store, project_id, film_id, caption_id, payload):
    film = load_film(store, project_id, film_id)
    caption_id = identity(caption_id)
    _verify_recognized_tracks(store, film, payload)
    result, texts = _build(film, caption_id, payload)
    for index in range(len(film["entries"])):
        film_media(store, project_id, film_id, index)
    destination = _root(store, caption_id)
    with _LOCK:
        if destination.exists():
            existing = load_captions(store, project_id, film_id, caption_id)
            if existing != result:
                raise ValueError("字幕任务身份已用于另一份稿件，请明确新建附件")
            return existing, False
        destination.parent.mkdir(parents=True, exist_ok=True)
        work = contained(store.root, "film_caption_work")
        work.mkdir(parents=True, exist_ok=True)
        created = True
        with tempfile.TemporaryDirectory(prefix=caption_id + "-", dir=work) as temporary:
            stage = Path(temporary) / "ready"
            stage.mkdir()
            for format, text in texts.items():
                (stage / f"captions.{format}").write_text(text, encoding="utf-8", newline="\n")
            atomic_json(stage / "manifest.json", result)
            # Publish the complete manifest and both formats in one directory rename.
            try:
                stage.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
                existing = load_captions(store, project_id, film_id, caption_id)
                if existing != result:
                    raise ValueError("字幕任务身份已用于另一份稿件") from None
                created = False
        return load_captions(store, project_id, film_id, caption_id), created


def _verify_recognized_tracks(store, film, payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('tracks'), list):
        raise ValueError('字幕输入缺少tracks列表')
    for track in payload['tracks']:
        if isinstance(track, dict) and isinstance(track.get('source'), dict) and track['source'].get('kind') == 'recognized_audio':
            from .director_caption_asr import verify_track_source
            verify_track_source(store, film, track)


def caption_file(store, project_id, film_id, caption_id, format):
    if format not in {"srt", "vtt"}:
        raise ValueError("字幕格式必须是 srt 或 vtt")
    load_captions(store, project_id, film_id, caption_id)
    return contained(_root(store, caption_id), f"captions.{format}")
