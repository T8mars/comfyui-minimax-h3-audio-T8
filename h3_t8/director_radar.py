"""Opt-in, non-executable source evidence and per-shot prompt rule snapshots.

No network, model imports, downloads, queue access or automatic acceptance.
Existing projects have no new fields and retain their original compilation.
Hashes bind content, not factual correctness, authorship or quality.
"""

from __future__ import annotations

from copy import deepcopy
import math
import re
import uuid
from fractions import Fraction

from .director_project import canonical, identity, sha

PACKET_SCHEMA = "t8.director.source_evidence.v1"
SKILL_SCHEMA = "t8.director.prompt_skill.v1"
CANDIDATE_SCHEMA = "t8.director.prompt_candidate.v1"
CANDIDATE_COMPONENTS = {"schema", "global", "draft", "references", "assets", "evidence",
                        "intent", "dependencies", "bindings", "options"}
MAX_BYTES = 512 * 1024
MAX_ITEMS = 256
MAX_TEXT = 32768
SHA = re.compile(r"[0-9a-f]{64}")
PARAMETER = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,63}")
PLACEHOLDER = re.compile(r"\$\{([a-zA-Z][a-zA-Z0-9_]{0,63})\}")
KINDS = {"visual", "ocr", "asr", "audio", "end_state"}
DOC_FIELDS = {"skillLibrary", "sharedSkills"}
SHOT_FIELDS = {
    "sourceEvidence",
    "creativeIntent",
    "skillBindings",
    "skillInherit",
    "skillDisabled",
    "promptCandidate",
    "evidenceHistory",
    "intentHistory",
    "candidateHistory",
}
HISTORY_FIELDS = {
    "sourceEvidence": "evidenceHistory",
    "creativeIntent": "intentHistory",
    "promptCandidate": "candidateHistory",
}


def _bounded(value, depth=0):
    if depth > 16:
        raise ValueError("证据/规则 JSON 嵌套超过16层")
    if isinstance(value, dict):
        if len(value) > MAX_ITEMS or not all(isinstance(k, str) for k in value):
            raise ValueError("证据/规则 JSON 对象过量或键无效")
        for key, item in value.items():
            _bounded(key, depth + 1)
            _bounded(item, depth + 1)
    elif isinstance(value, list):
        if len(value) > MAX_ITEMS:
            raise ValueError("证据/规则列表最多256项")
        for item in value:
            _bounded(item, depth + 1)
    elif isinstance(value, str):
        if len(value) > MAX_TEXT:
            raise ValueError("证据/规则单段文字超过32768字")
    elif value is not None and type(value) not in (bool, int, float):
        raise ValueError("证据/规则仅接受非执行 JSON 数据")
    elif type(value) is float and not math.isfinite(value):
        raise ValueError("证据/规则不接受 NaN/Infinity")


def _data(value):
    _bounded(value)
    if len(canonical(value).encode("utf-8")) > MAX_BYTES:
        raise ValueError("证据/规则数据超过512KiB")


def _object(value, fields, required=()):
    if (
        not isinstance(value, dict)
        or set(value) - set(fields)
        or set(required) - set(value)
    ):
        raise ValueError("证据/规则字段无效或缺失")


def _text(value, label, *, empty=True, limit=MAX_TEXT):
    if (
        not isinstance(value, str)
        or len(value) > limit
        or (not empty and not value.strip())
    ):
        raise ValueError(label + "必须是有效文字")


def _digest(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError("证据/规则 SHA256 无效")


def _integer(value, label, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**53 - 1:
        raise ValueError(label + "必须是可精确表达的整数")


def _boolean(value):
    if type(value) is not bool:
        raise ValueError("证据/规则开关必须是布尔值")


def _pts(value, label):
    if type(value) is not int or not -(2**53 - 1) <= value <= 2**53 - 1:
        raise ValueError(label + "必须是可精确表达的整数 PTS")


def _unique(rows, key="id"):
    values = [row[key] for row in rows]
    if len(values) != len(set(values)):
        raise ValueError("证据/规则身份重复")


def _media(value):
    _object(value, {"asset_id", "sha256"}, {"asset_id", "sha256"})
    identity(value["asset_id"])
    _digest(value["sha256"])


def validate_packet(value):
    _data(value)
    _object(
        value,
        {"schema", "id", "revision", "source", "claims", "audio_source", "sha256"},
        {"schema", "id", "revision", "source", "claims", "sha256"},
    )
    if value["schema"] != PACKET_SCHEMA:
        raise ValueError("不支持的源视频证据格式")
    identity(value["id"])
    _integer(value["revision"], "证据 revision", 1)
    source = value["source"]
    _object(
        source,
        {"asset_id", "sha256", "stream", "timebase", "start_pts", "end_pts"},
        {"asset_id", "sha256", "stream", "timebase", "start_pts", "end_pts"},
    )
    _media({k: source[k] for k in ("asset_id", "sha256")})
    _integer(source["stream"], "stream")
    _object(source["timebase"], {"num", "den"}, {"num", "den"})
    for key in ("num", "den"):
        _integer(source["timebase"][key], "timebase." + key, 1)
    for key in ("start_pts", "end_pts"):
        _pts(source[key], key)
    if source["end_pts"] <= source["start_pts"]:
        raise ValueError("源证据结束 PTS 必须晚于开始")
    audio = value.get("audio_source")
    if audio is not None:
        _object(
            audio,
            {
                "asset_id",
                "sha256",
                "sample_rate",
                "start_sample",
                "end_sample",
                "stream",
            },
            {"asset_id", "sha256", "sample_rate", "start_sample", "end_sample"},
        )
        _media({k: audio[k] for k in ("asset_id", "sha256")})
        _integer(audio["sample_rate"], "sample_rate", 1)
        if "stream" in audio:
            _integer(audio["stream"], "audio stream")
        for key in ("start_sample", "end_sample"):
            _integer(audio[key], key)
        if (
            audio["end_sample"] <= audio["start_sample"]
            or audio["sample_rate"] > 384000
        ):
            raise ValueError("音频证据采样区间无效")
    claims = value["claims"]
    if not isinstance(claims, list):
        raise ValueError("claims 必须是列表")
    for claim in claims:
        _object(
            claim,
            {
                "id",
                "kind",
                "text",
                "start_pts",
                "end_pts",
                "frame",
                "provider",
                "confidence",
            },
            {"id", "kind", "text", "start_pts", "end_pts"},
        )
        identity(claim["id"])
        if not isinstance(claim["kind"], str) or claim["kind"] not in KINDS:
            raise ValueError("画面/OCR/ASR/声音/末态证据必须分别标记")
        _text(claim["text"], "证据文字", empty=False)
        for key in ("start_pts", "end_pts"):
            _pts(claim[key], key)
        if (
            not source["start_pts"]
            <= claim["start_pts"]
            <= claim["end_pts"]
            <= source["end_pts"]
        ):
            raise ValueError("证据时间超出源 PTS 范围")
        if claim["kind"] == "asr" and audio is None:
            raise ValueError("ASR 候选必须有独立真实音频来源，不允许视觉猜台词")
        if "provider" in claim:
            _text(claim["provider"], "provider", limit=200)
        if "confidence" in claim:
            confidence = claim["confidence"]
            if (
                type(confidence) not in (int, float)
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                raise ValueError("证据 confidence 必须在0–1之间，不能自动确认")
        if "frame" in claim:
            frame = claim["frame"]
            _object(
                frame,
                {"asset_id", "sha256", "frame", "pts"},
                {"asset_id", "sha256", "frame", "pts"},
            )
            _media({k: frame[k] for k in ("asset_id", "sha256")})
            _integer(frame["frame"], "frame")
            _pts(frame["pts"], "frame.pts")
            if not claim["start_pts"] <= frame["pts"] <= claim["end_pts"]:
                raise ValueError("证据帧 PTS 不在声明区间")
    _unique(claims)
    _digest(value["sha256"])
    if sha({k: v for k, v in value.items() if k != "sha256"}) != value["sha256"]:
        raise ValueError("源证据内容 SHA 不符")
    return deepcopy(value)


def seal_packet(value):
    """Explicit import seals content, never promotes a claim to confirmed."""
    value = deepcopy(value)
    value.pop("sha256", None)
    _data(value)
    value["sha256"] = sha(value)
    return validate_packet(value)


def _parameter_value(value, spec):
    kind = spec["type"]
    valid = (
        isinstance(value, str)
        if kind == "string"
        else type(value) is bool
        if kind == "boolean"
        else type(value) is int
        if kind == "integer"
        else type(value) in (int, float)
    )
    if not valid:
        raise ValueError("规则参数类型不匹配")
    if isinstance(value, str) and len(value) > spec.get("max_length", 4096):
        raise ValueError("规则参数文字过长")
    if kind in ("number", "integer"):
        if not math.isfinite(value) or not spec.get("min", -1e12) <= value <= spec.get(
            "max", 1e12
        ):
            raise ValueError("规则参数数字越界")
    if "choices" in spec and value not in spec["choices"]:
        raise ValueError("规则参数不在允许选项")


def validate_skill(value):
    _data(value)
    _object(
        value,
        {
            "schema",
            "id",
            "version",
            "name",
            "text_snapshot",
            "parameter_schema",
            "content_sha256",
        },
        {
            "schema",
            "id",
            "version",
            "name",
            "text_snapshot",
            "parameter_schema",
            "content_sha256",
        },
    )
    if value["schema"] != SKILL_SCHEMA:
        raise ValueError("不支持的 Prompt Skill 格式")
    identity(value["id"])
    _integer(value["version"], "Skill version", 1)
    _text(value["name"], "规则名称", empty=False, limit=200)
    _text(value["text_snapshot"], "规则快照", empty=False)
    params = value["parameter_schema"]
    if not isinstance(params, dict) or len(params) > 32:
        raise ValueError("规则最多32个声明参数")
    for key, spec in params.items():
        if not PARAMETER.fullmatch(key):
            raise ValueError("规则参数名称无效")
        _object(
            spec, {"type", "default", "max_length", "min", "max", "choices"}, {"type"}
        )
        if not isinstance(spec["type"], str) or spec["type"] not in {
            "string",
            "number",
            "integer",
            "boolean",
        }:
            raise ValueError("规则参数只能是标量，不执行模板或工具")
        if "max_length" in spec:
            _integer(spec["max_length"], "max_length", 1)
            if spec["max_length"] > 4096:
                raise ValueError("参数最长4096字")
        for bound in ("min", "max"):
            if bound in spec and (
                type(spec[bound]) not in (int, float) or not math.isfinite(spec[bound])
            ):
                raise ValueError("参数数值界限无效")
        if spec.get("min", -1e12) > spec.get("max", 1e12):
            raise ValueError("参数 min 大于 max")
        if "choices" in spec:
            if (
                not isinstance(spec["choices"], list)
                or not 1 <= len(spec["choices"]) <= 32
            ):
                raise ValueError("参数 choices 无效")
            for choice in spec["choices"]:
                _parameter_value(
                    choice, {k: v for k, v in spec.items() if k != "choices"}
                )
        if "default" in spec:
            _parameter_value(spec["default"], spec)
    if set(PLACEHOLDER.findall(value["text_snapshot"])) - params.keys():
        raise ValueError("模板引用未声明的参数")
    _digest(value["content_sha256"])
    if (
        sha({k: v for k, v in value.items() if k != "content_sha256"})
        != value["content_sha256"]
    ):
        raise ValueError("规则快照 SHA 不符")
    return deepcopy(value)


def make_skill(name, text, parameter_schema=None, *, skill_id=None, version=1):
    value = {
        "schema": SKILL_SCHEMA,
        "id": skill_id or str(uuid.uuid4()),
        "version": version,
        "name": name,
        "text_snapshot": text,
        "parameter_schema": deepcopy(parameter_schema or {}),
    }
    _data(value)
    value["content_sha256"] = sha(value)
    return validate_skill(value)


def validate_bindings(rows):
    _data(rows)
    if not isinstance(rows, list):
        raise ValueError("规则绑定必须是有序列表")
    for row in rows:
        _object(
            row,
            {"id", "snapshot", "enabled", "parameters"},
            {"id", "snapshot", "enabled", "parameters"},
        )
        identity(row["id"])
        _boolean(row["enabled"])
        skill = validate_skill(row["snapshot"])
        params = row["parameters"]
        if (
            not isinstance(params, dict)
            or set(params) - skill["parameter_schema"].keys()
        ):
            raise ValueError("绑定包含未声明参数")
        for key, spec in skill["parameter_schema"].items():
            if key not in params and "default" not in spec:
                raise ValueError("绑定缺少规则参数 " + key)
            _parameter_value(params.get(key, spec.get("default")), spec)
    _unique(rows)
    return deepcopy(rows)


def bind_skill(snapshot, parameters=None, *, enabled=True, binding_id=None):
    row = {
        "id": binding_id or str(uuid.uuid4()),
        "snapshot": deepcopy(snapshot),
        "enabled": enabled,
        "parameters": deepcopy(parameters or {}),
    }
    return validate_bindings([row])[0]


def effective_bindings(doc, shot):
    inherited = doc.get("sharedSkills", []) if shot.get("skillInherit", True) else []
    disabled = set(shot.get("skillDisabled", []))
    return [
        row
        for row in inherited + shot.get("skillBindings", [])
        if row["enabled"] and row["id"] not in disabled
    ]


def render_binding(row):
    snapshot = row["snapshot"]
    params = {
        key: row["parameters"].get(key, spec.get("default"))
        for key, spec in snapshot["parameter_schema"].items()
    }

    def literal(match):
        value = params[match.group(1)]
        return str(value).lower() if type(value) is bool else str(value)

    # One substitution pass: a parameter cannot inject a second template operation.
    return PLACEHOLDER.sub(literal, snapshot["text_snapshot"])


def _review(value, packet):
    _object(
        value,
        {"packet_sha256", "revision", "claim_ids"},
        {"packet_sha256", "revision", "claim_ids"},
    )
    _digest(value["packet_sha256"])
    _integer(value["revision"], "review revision", 1)
    ids = value["claim_ids"]
    if not isinstance(ids, list):
        raise ValueError("确认条目必须是 UUID 列表")
    for claim_id in ids:
        identity(claim_id)
    if len(ids) != len(set(ids)) or (
        value["packet_sha256"] == packet["sha256"]
        and set(ids) - {row["id"] for row in packet["claims"]}
    ):
        raise ValueError("确认记录引用不存在或重复的证据")
    # Mismatched review remains historical/stale; it is never silently re-signed.


def _evidence(value):
    _object(value, {"packet", "review"}, {"packet"})
    packet = validate_packet(value["packet"])
    if value.get("review") is not None:
        _review(value["review"], packet)


def _intent(value):
    _object(
        value,
        {"revision", "text", "reason", "dependencies"},
        {"revision", "text", "reason", "dependencies"},
    )
    _integer(value["revision"], "intent revision", 1)
    _text(value["text"], "创作意图")
    _text(value["reason"], "修改原因")
    if not isinstance(value["dependencies"], list):
        raise ValueError("事实依赖必须是明确列表")
    for dep in value["dependencies"]:
        _object(dep, {"shot_id", "evidence_sha256"}, {"shot_id", "evidence_sha256"})
        identity(dep["shot_id"])
        _digest(dep["evidence_sha256"])
    _unique(value["dependencies"], "shot_id")


def _archive(shot, field):
    """Append a detached historical record; never truncate or re-sign it."""
    if field in shot:
        shot.setdefault(HISTORY_FIELDS[field], []).append(deepcopy(shot[field]))


def validate_history_progress(previous, current):
    """Saved history is append-only. A new project ID may retain an imported archive."""
    old_shots = {row["id"]: row for row in previous["doc"]["shots"]}
    for shot in current["doc"]["shots"]:
        old = old_shots.get(shot["id"], {})
        for field, history_key in HISTORY_FIELDS.items():
            history = shot.get(history_key, [])
            old_history = old.get(history_key, [])
            if history[: len(old_history)] != old_history:
                raise ValueError("已保存修订历史不可改写或删除；请另存新工程")
            before, after = old.get(field), shot.get(field)
            # Adoption/revert changes execution state, not the candidate text/receipt.
            same_candidate = (
                field == "promptCandidate"
                and before
                and after
                and {k: v for k, v in before.items() if k != "active"}
                == {k: v for k, v in after.items() if k != "active"}
            )
            if before is not None and before != after and not same_candidate:
                if before not in history[len(old_history) :]:
                    raise ValueError("修改已有证据/创作/候选必须先保留完整历史修订")


def dependency_media(project, shots):
    """Only facts explicitly consumed by an active candidate own extra resources."""
    lookup = {row["id"]: row for row in project["doc"]["shots"]}
    result = set()
    for shot in shots:
        candidate = shot.get("promptCandidate")
        if (
            not candidate
            or not candidate["active"]
            or not candidate["options"]["intent"]
        ):
            continue
        for dep in shot.get("creativeIntent", {}).get("dependencies", []):
            evidence = lookup.get(dep["shot_id"], {}).get("sourceEvidence")
            if evidence:
                result.update(
                    row["asset_id"] for row in _packet_media(evidence["packet"])
                )
    return result


def validate_fields(project):
    doc = project["doc"]
    data = {k: doc[k] for k in DOC_FIELDS if k in doc}
    data["shots"] = [
        {k: row[k] for k in SHOT_FIELDS if k in row}
        for row in doc["shots"]
        if any(k in row for k in SHOT_FIELDS)
    ]
    if not data["shots"] and len(data) == 1:
        return
    _data(data)
    library = doc.get("skillLibrary", [])
    if not isinstance(library, list):
        raise ValueError("skillLibrary 必须是快照列表")
    skills = [validate_skill(row) for row in library]
    versions = [(row["id"], row["version"]) for row in skills]
    if len(versions) != len(set(versions)):
        raise ValueError("规则库 ID/version 重复")
    shared = validate_bindings(doc.get("sharedSkills", []))
    skills.extend(row["snapshot"] for row in shared)
    shared_ids = {row["id"] for row in shared}
    for shot in doc["shots"]:
        if "skillInherit" in shot:
            _boolean(shot["skillInherit"])
        disabled = shot.get("skillDisabled", [])
        if not isinstance(disabled, list) or len(disabled) != len(set(disabled)):
            raise ValueError("局部禁用规则列表无效")
        for binding_id in disabled:
            identity(binding_id)
        rows = validate_bindings(shot.get("skillBindings", []))
        if shared_ids & {row["id"] for row in rows}:
            raise ValueError("本镜与全片绑定 UUID 冲突")
        skills.extend(row["snapshot"] for row in rows)
        packet_versions = {}
        packet_revisions = {}
        for field, check in [
            ("sourceEvidence", _evidence),
            ("creativeIntent", _intent),
        ]:
            history = shot.get(HISTORY_FIELDS[field], [])
            if not isinstance(history, list):
                raise ValueError("修订历史必须是只读记录列表")
            records = history + ([shot[field]] if field in shot else [])
            last_revision = 0
            for record in records:
                check(record)
                if field == "sourceEvidence":
                    packet = record["packet"]
                    key = (packet["id"], packet["revision"])
                    if packet_versions.get(key, packet["sha256"]) != packet[
                        "sha256"
                    ] or packet["revision"] < packet_revisions.get(packet["id"], 0):
                        raise ValueError("同一证据版本不可改写或回退，请增加 revision")
                    packet_versions[key] = packet["sha256"]
                    packet_revisions[packet["id"]] = packet["revision"]
                else:
                    if record["revision"] <= last_revision:
                        raise ValueError("创作修订历史 revision 必须递增")
                    last_revision = record["revision"]
        candidate_history = shot.get("candidateHistory", [])
        if not isinstance(candidate_history, list):
            raise ValueError("候选历史必须是只读记录列表")
        for candidate in candidate_history + (
            [shot["promptCandidate"]] if "promptCandidate" in shot else []
        ):
            _object(
                candidate,
                {
                    "schema",
                    "id",
                    "text",
                    "text_sha256",
                    "inputs_sha256",
                    "active",
                    "options",
                    "receipt",
                },
                {
                    "schema",
                    "id",
                    "text",
                    "text_sha256",
                    "inputs_sha256",
                    "active",
                    "options",
                    "receipt",
                },
            )
            if candidate["schema"] != CANDIDATE_SCHEMA:
                raise ValueError("不支持的候选稿格式")
            identity(candidate["id"])
            _text(candidate["text"], "候选稿")
            for key in ("text_sha256", "inputs_sha256"):
                _digest(candidate[key])
            if candidate["text_sha256"] != sha(candidate["text"]):
                raise ValueError("候选稿文字 SHA 不符")
            _boolean(candidate["active"])
            _object(candidate["options"], {"facts", "intent"}, {"facts", "intent"})
            for flag in candidate["options"].values():
                _boolean(flag)
            if not isinstance(candidate["receipt"], dict):
                raise ValueError("候选稿需要消费回执")
            _object(
                candidate["receipt"],
                {"binding_order", "claim_ids", "facts_revision", "intent_revision", "input_component_sha256"},
                {"binding_order", "claim_ids", "facts_revision", "intent_revision"},
            )
            if not isinstance(
                candidate["receipt"]["binding_order"], list
            ) or not isinstance(candidate["receipt"]["claim_ids"], list):
                raise ValueError("候选回执条目必须是列表")
            for entry in candidate["receipt"]["binding_order"]:
                _object(
                    entry,
                    {"binding_id", "content_sha256"},
                    {"binding_id", "content_sha256"},
                )
                identity(entry["binding_id"])
                _digest(entry["content_sha256"])
            for claim_id in candidate["receipt"]["claim_ids"]:
                identity(claim_id)
            for key in ("facts_revision", "intent_revision"):
                if candidate["receipt"][key] is not None:
                    _integer(candidate["receipt"][key], key, 1)
            components = candidate["receipt"].get("input_component_sha256")
            if "input_component_sha256" in candidate["receipt"]:
                _object(components, CANDIDATE_COMPONENTS, CANDIDATE_COMPONENTS)
                for digest in components.values():
                    _digest(digest)
    fingerprints = {}
    for skill in skills:
        key = (skill["id"], skill["version"])
        if key in fingerprints and fingerprints[key] != skill["content_sha256"]:
            raise ValueError("同 ID/version 的规则快照内容冲突，请显式提升版本")
        fingerprints[key] = skill["content_sha256"]


def _packet_media(packet):
    rows = [packet["source"]]
    if packet.get("audio_source"):
        rows.append(packet["audio_source"])
    rows.extend(claim["frame"] for claim in packet["claims"] if "frame" in claim)
    return rows


def evidence_status(project, shot):
    evidence = shot.get("sourceEvidence")
    if evidence is None:
        return "legacy"
    assets = {row["id"]: row for row in project["assets"]}
    packet = evidence["packet"]
    for source in _packet_media(packet):
        if assets.get(source["asset_id"], {}).get("sha256") != source["sha256"]:
            return "source_changed"
    review = evidence.get("review")
    if review is None:
        return "unreviewed"
    return (
        "confirmed" if review["packet_sha256"] == packet["sha256"] else "review_stale"
    )


def evidence_time_map(packet, *, output_start_frame=0, output_fps=24):
    """Exact rational mappings; no truncation, snapping, VFR assumption or trim."""
    validate_packet(packet)
    _integer(output_start_frame, "output_start_frame")
    _integer(output_fps, "output_fps", 1)
    if output_fps > 1000:
        raise ValueError("output_fps 超出范围")
    source = packet["source"]
    timebase = Fraction(source["timebase"]["num"], source["timebase"]["den"])

    def rational(value):
        return {"num": value.numerator, "den": value.denominator}

    rows = []
    for claim in packet["claims"]:
        row = {"claim_id": claim["id"], "kind": claim["kind"]}
        for side in ("start", "end"):
            absolute = claim[side + "_pts"] * timebase
            local = (claim[side + "_pts"] - source["start_pts"]) * timebase
            row[side] = {
                "source_pts": claim[side + "_pts"],
                "absolute_seconds": rational(absolute),
                "local_seconds": rational(local),
                "output_frame_position": rational(
                    local * output_fps + output_start_frame
                ),
            }
        rows.append(row)
    return {
        "source_sha256": source["sha256"],
        "source_timebase": source["timebase"],
        "output_fps": output_fps,
        "output_start_frame": output_start_frame,
        "claims": rows,
        "automatic_rounding": False,
        "sampling_frames_modified": False,
        "audio_source": deepcopy(packet.get("audio_source")),
    }


def verify_packet_timing(packet, timing):
    """Actual header constraints, never a declaration that content is truthful."""
    source = packet["source"]
    stream = next(
        (
            row
            for row in timing["streams"]
            if row["index"] == source["stream"] and row["kind"] == "video"
        ),
        None,
    )
    if stream is None:
        raise ValueError("证据声明的视频轨道不存在")
    if Fraction(source["timebase"]["num"], source["timebase"]["den"]) != Fraction(
        stream["timebase"]["num"], stream["timebase"]["den"]
    ):
        raise ValueError("证据 timebase 与实际视频轨道不符")
    if stream["start_pts"] is not None and source["start_pts"] < stream["start_pts"]:
        raise ValueError("证据起始 PTS 早于真实轨道")
    if stream["end_pts"] is not None and source["end_pts"] > stream["end_pts"]:
        raise ValueError("证据结束 PTS 超出真实轨道")


def verify_audio_timing(audio_source, timing):
    streams = [row for row in timing["streams"] if row["kind"] == "audio"]
    if "stream" in audio_source:
        streams = [row for row in streams if row["index"] == audio_source["stream"]]
    if len(streams) != 1:
        raise ValueError("音频证据必须绑定唯一真实音频轨道；多轨需明确 stream")
    stream = streams[0]
    if stream["sample_rate"] != audio_source["sample_rate"]:
        raise ValueError("音频证据 sample_rate 与实际轨道不符")
    if stream["duration_pts"] is not None:
        capacity = Fraction(
            stream["duration_pts"] * stream["timebase"]["num"] * stream["sample_rate"],
            stream["timebase"]["den"],
        )
        if audio_source["end_sample"] > capacity:
            raise ValueError("音频证据采样区间超出真实轨道")


def _inputs(project, shot, options):
    doc = project["doc"]
    assets = {row["id"]: row.get("sha256") for row in project["assets"]}
    referenced = set(doc["sharedRefs"] + shot["tray"] + shot["refs"])
    referenced.update(shot[key] for key in ("first", "last", "audio") if shot.get(key))
    evidence = shot.get("sourceEvidence") if options["facts"] else None
    if evidence:
        referenced.update(row["asset_id"] for row in _packet_media(evidence["packet"]))
    intent = shot.get("creativeIntent") if options["intent"] else None
    dependencies = []
    for dep in (intent or {}).get("dependencies", []):
        other = next((row for row in doc["shots"] if row["id"] == dep["shot_id"]), None)
        actual = (other or {}).get("sourceEvidence", {})
        dependencies.append(
            {
                "declared": dep,
                "actual_packet_sha256": actual.get("packet", {}).get("sha256"),
                "review": actual.get("review"),
                "status": evidence_status(project, other) if other else "missing",
            }
        )
    return {
        "schema": "t8.director.prompt_inputs.v1",
        "global": doc["global"],
        "draft": {
            k: shot.get(k)
            for k in (
                "writingMode",
                "simplePrompt",
                "prompt",
                "events",
                "first",
                "last",
                "audio",
                "mode",
                "sound",
            )
        },
        "references": {
            "shared": doc["sharedRefs"],
            "tray": shot["tray"],
            "refs": shot["refs"],
        },
        "assets": {aid: assets.get(aid) for aid in sorted(referenced)},
        "evidence": evidence,
        "intent": intent,
        "dependencies": dependencies,
        "bindings": effective_bindings(doc, shot),
        "options": options,
    }


def candidate_status(project, shot):
    candidate = shot.get("promptCandidate")
    if candidate is None:
        return "legacy"
    if sha(_inputs(project, shot, candidate["options"])) != candidate["inputs_sha256"]:
        return "stale"
    return "active" if candidate["active"] else "pending"


def candidate_diagnostics(project, shot):
    """Read-only draft-signature explanation, never Stage/MODEL certification.

    New candidates record component digests computed by create_candidate. Old
    candidates have only the aggregate digest: do not invent the missing prior
    values or change their compilation/adoption policy.
    """
    status = candidate_status(project, shot)
    candidate = shot.get("promptCandidate")
    result = {"schema": "t8.director.candidate_diagnostics.v1", "status": status,
              "scope": "draft_signatures_not_runtime_MODEL_or_Stage_cache_certification",
              "changed_components": None, "component_snapshot": "legacy_missing",
              "saved": False, "queued": False, "adopted": False, "cache_hit_certified": False}
    if candidate is None:
        result.update(reason="no_candidate_original_author_draft", inputs_hash_match=None)
        return result
    actual_inputs = _inputs(project, shot, candidate["options"])
    actual_sha = sha(actual_inputs)
    match = actual_sha == candidate["inputs_sha256"]
    result.update(expected_inputs_sha256=candidate["inputs_sha256"],
                  actual_inputs_sha256=actual_sha, inputs_hash_match=match,
                  reason="candidate_inputs_changed" if not match else
                         "explicitly_adopted_current_draft" if candidate["active"] else
                         "current_candidate_not_adopted")
    old = candidate["receipt"].get("input_component_sha256")
    if old is not None:
        current = {key: sha(value) for key, value in actual_inputs.items()}
        changed = sorted(key for key in CANDIDATE_COMPONENTS if old[key] != current[key])
        # A caller can edit metadata; inconsistent declarations are not evidence
        # of a specific changed input. Never silently repair/re-sign the receipt.
        if bool(changed) == (not match):
            result.update(changed_components=changed,
                          component_snapshot="recorded_component_digest_comparison")
        else:
            result.update(component_snapshot="inconsistent_receipt_declaration",
                          diagnostic="Aggregate and component declarations disagree; no exact field attribution.")
    if result["component_snapshot"] == "legacy_missing":
        result["diagnostic"] = "Old candidate has no component snapshot; aggregate staleness is known, exact prior values are unavailable."
    return result


def create_candidate(project, shot_id, *, facts=False, intent=True):
    validate_fields(project)
    _boolean(facts)
    _boolean(intent)
    shot = next(row for row in project["doc"]["shots"] if row["id"] == shot_id)
    options = {"facts": facts, "intent": intent}
    inputs = _inputs(project, shot, options)
    for dependency in inputs["dependencies"]:
        if (
            dependency["status"] != "confirmed"
            or dependency["declared"]["evidence_sha256"]
            != dependency["actual_packet_sha256"]
        ):
            raise ValueError("明确依赖的跨镜事实已失效，请重新确认")
    text = shot["simplePrompt"] if shot["writingMode"] == "simple" else shot["prompt"]
    blocks = [text]
    receipt = {
        "binding_order": [],
        "claim_ids": [],
        "facts_revision": None,
        "intent_revision": None,
        "input_component_sha256": {key: sha(value) for key, value in inputs.items()},
    }
    if facts and shot.get("sourceEvidence"):
        if evidence_status(project, shot) != "confirmed":
            raise ValueError("请先审核源证据；ASR/OCR置信度不自动确认事实")
        evidence = shot["sourceEvidence"]
        selected = set(evidence["review"]["claim_ids"])
        # Verbatim evidence is not cast dialogue; label its type explicitly.
        blocks.extend(
            f"[已确认{claim['kind']}证据] {claim['text']}"
            for claim in evidence["packet"]["claims"]
            if claim["id"] in selected
        )
        receipt.update(
            claim_ids=evidence["review"]["claim_ids"],
            facts_revision=evidence["review"]["revision"],
        )
    if intent and shot.get("creativeIntent"):
        blocks.append("[用户创作意图] " + shot["creativeIntent"]["text"])
        receipt["intent_revision"] = shot["creativeIntent"]["revision"]
    for binding in inputs["bindings"]:
        blocks.append(render_binding(binding))
        receipt["binding_order"].append(
            {
                "binding_id": binding["id"],
                "content_sha256": binding["snapshot"]["content_sha256"],
            }
        )
    result = "\n\n".join(block for block in blocks if block)
    _text(result, "编译候选稿")
    return {
        "schema": CANDIDATE_SCHEMA,
        "id": str(uuid.uuid4()),
        "text": result,
        "text_sha256": sha(result),
        "inputs_sha256": sha(inputs),
        "active": False,
        "options": options,
        "receipt": receipt,
    }


def compiled_local(project, shot):
    """Only explicitly adopted, current candidates replace the execution draft."""
    candidate = shot.get("promptCandidate")
    if candidate is not None and candidate["active"]:
        if candidate_status(project, shot) != "active":
            raise ValueError("已采用候选稿依赖已改变，请重新编译审核或退回原稿")
        return candidate["text"]
    return shot["simplePrompt"] if shot["writingMode"] == "simple" else shot["prompt"]


def apply_operation(project, shot_id, operation, value):
    from .director_project import validate_project

    result = validate_project(project)
    _data(value)
    shot = next((row for row in result["doc"]["shots"] if row["id"] == shot_id), None)
    if shot is None:
        raise ValueError("请选择项目内有效镜头")
    if operation == "evidence_import":
        packet = seal_packet(value)
        assets = {row["id"]: row for row in result["assets"]}
        for source in _packet_media(packet):
            actual = assets.get(source["asset_id"], {})
            if actual.get("sha256") != source["sha256"]:
                raise ValueError("证据素材必须先登记且SHA与项目一致")
        if assets[packet["source"]["asset_id"]].get("kind") != "video":
            raise ValueError("源证据必须绑定视频素材")
        if packet.get("audio_source") and assets[
            packet["audio_source"]["asset_id"]
        ].get("kind") not in ("audio", "video"):
            raise ValueError("音频证据必须绑定音频或含音频的视频素材")
        for claim in packet["claims"]:
            if "frame" in claim:
                frame = claim["frame"]
                asset = assets[frame["asset_id"]]
                if asset.get("kind") != "image":
                    raise ValueError("证据帧必须是已登记图片")
                provenance = asset.get("source_frame")
                if provenance and (
                    provenance["media_sha256"] != packet["source"]["sha256"]
                    or provenance["frame"] != frame["frame"]
                ):
                    raise ValueError("证据帧真实取帧来源与声明不符")
        # Preserve prior review as stale, never confirm a new packet automatically.
        old = shot.get("sourceEvidence")
        if old and old["packet"] != packet:
            _archive(shot, "sourceEvidence")
        shot["sourceEvidence"] = {
            "packet": packet,
            **(
                {"review": shot["sourceEvidence"]["review"]}
                if shot.get("sourceEvidence", {}).get("review")
                else {}
            ),
        }
    elif operation == "evidence_review":
        _object(
            value,
            {"confirm", "packet_sha256", "claim_ids"},
            {"confirm", "packet_sha256", "claim_ids"},
        )
        evidence = shot.get("sourceEvidence")
        if (
            value["confirm"] is not True
            or not evidence
            or value["packet_sha256"] != evidence["packet"]["sha256"]
        ):
            raise ValueError("请显式确认当前证据 SHA 与所选条目")
        if evidence_status(result, shot) == "source_changed":
            raise ValueError("证据来源字节已经改变")
        review = {
            "packet_sha256": value["packet_sha256"],
            "claim_ids": value["claim_ids"],
            "revision": evidence.get("review", {}).get("revision", 0) + 1,
        }
        _review(review, evidence["packet"])
        _archive(shot, "sourceEvidence")
        evidence["review"] = review
    elif operation == "intent_update":
        _object(
            value,
            {"text", "reason", "dependencies"},
            {"text", "reason", "dependencies"},
        )
        _archive(shot, "creativeIntent")
        shot["creativeIntent"] = {
            **deepcopy(value),
            "revision": shot.get("creativeIntent", {}).get("revision", 0) + 1,
        }
    elif operation == "skills_update":
        _object(value, {"library", "shared", "local", "inherit", "disabled"})
        for source, target in [("library", "skillLibrary"), ("shared", "sharedSkills")]:
            if source in value:
                result["doc"][target] = deepcopy(value[source])
        for source, target in [
            ("local", "skillBindings"),
            ("inherit", "skillInherit"),
            ("disabled", "skillDisabled"),
        ]:
            if source in value:
                shot[target] = deepcopy(value[source])
    elif operation == "skill_create":
        _object(
            value,
            {
                "name",
                "text",
                "parameter_schema",
                "parameters",
                "scope",
                "skill_id",
                "version",
            },
            {"name", "text", "scope"},
        )
        if not isinstance(value["scope"], str) or value["scope"] not in {
            "project",
            "shot",
            "library",
        }:
            raise ValueError("规则作用域只能是全片、本镜或仅收藏")
        snapshot = make_skill(
            value["name"],
            value["text"],
            value.get("parameter_schema"),
            skill_id=value.get("skill_id"),
            version=value.get("version", 1),
        )
        result["doc"].setdefault("skillLibrary", []).append(snapshot)
        if value["scope"] != "library":
            row = bind_skill(snapshot, value.get("parameters"))
            owner, key = (
                (result["doc"], "sharedSkills")
                if value["scope"] == "project"
                else (shot, "skillBindings")
            )
            owner.setdefault(key, []).append(row)
    elif operation == "candidate_create":
        _object(value, {"facts", "intent"})
        _archive(shot, "promptCandidate")
        shot["promptCandidate"] = create_candidate(
            result,
            shot_id,
            facts=value.get("facts", False),
            intent=value.get("intent", True),
        )
    elif operation == "candidate_accept":
        _object(value, {"confirm", "text_sha256"}, {"confirm", "text_sha256"})
        candidate = shot.get("promptCandidate")
        if (
            value["confirm"] is not True
            or candidate is None
            or value["text_sha256"] != candidate["text_sha256"]
            or candidate_status(result, shot) == "stale"
        ):
            raise ValueError("请核对当前候选稿文字/依赖并显式采用")
        candidate["active"] = True
    elif operation == "candidate_revert":
        _object(value, set())
        if shot.get("promptCandidate"):
            shot["promptCandidate"]["active"] = False
    else:
        raise ValueError("未知证据/规则操作")
    shot["rev"] = shot.get("rev", 0) + 1
    return validate_project(result)


def remap_project(project, asset_map, shot_map):
    """Fresh bundle identities: retain original receipts, don't re-sign candidates."""
    for shot in project["doc"]["shots"]:
        evidence = shot.get("sourceEvidence")
        if evidence:
            # Remapping explicitly creates a new packet; retain the prior review
            # SHA/revision unchanged and stale, rather than re-signing approval.
            _archive(shot, "sourceEvidence")
            packet = evidence["packet"]
            for source in _packet_media(packet):
                source["asset_id"] = asset_map[source["asset_id"]]
            packet["revision"] += 1
            evidence["packet"] = seal_packet(packet)
        intent = shot.get("creativeIntent")
        if intent:
            original = deepcopy(intent)
            for dep in intent["dependencies"]:
                dep["shot_id"] = shot_map.get(dep["shot_id"], dep["shot_id"])
            if original != intent:
                shot.setdefault("intentHistory", []).append(original)
                intent["revision"] += 1
        # Candidates are historical drafts after import. Explicitly revert them;
        # original author fields and generated text are both retained.
        if shot.get("promptCandidate"):
            shot["promptCandidate"]["active"] = False
