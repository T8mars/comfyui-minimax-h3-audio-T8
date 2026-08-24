from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import re

import torch

from .prompt_tags import MEDIA_TAG_RE, canonicalize_media_tags, prepare_prompt


PROMPT_BUDGET_SCHEMA = "minimax_h3_t8_prompt_budget_v1"
OFFICIAL_H3_SUBMISSION_CHARACTER_LIMIT = 7000
OFFICIAL_H3_SUBMISSION_LIMIT_SOURCE = (
    "https://github.com/MiniMax-AI/cli/blob/main/skill/h3-video/SKILL.md#core-limits"
)
_SAFE_SUBJECT_ID = re.compile(r"^[\w\-\u3400-\u9fff]{1,64}$", flags=re.UNICODE)
_CJK = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def _json(value: Mapping) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _estimated_token_count(text: str) -> int:
    """Return a labelled planning proxy, never an alleged tokenizer-exact value."""

    cjk_count = len(_CJK.findall(text))
    non_cjk = _CJK.sub(" ", text)
    pieces = re.findall(r"[A-Za-z0-9_]+|[^\w\s]", non_cjk, flags=re.UNICODE)
    latin_proxy = sum(max(1, math.ceil(len(piece.encode("utf-8")) / 4)) for piece in pieces)
    return int(cjk_count + latin_proxy)


def _sequence_token_count(value) -> int | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0
        return int(value.shape[-1]) if value.ndim >= 1 else 1
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return 0
        if all(
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes, bytearray))
            and len(item) >= 1
            and isinstance(item[0], int)
            for item in value
        ):
            return len(value)
        child_counts = [count for item in value if (count := _sequence_token_count(item)) is not None]
        return max(child_counts) if child_counts else None
    return None


def _clip_token_count(clip, text: str) -> tuple[int | None, dict, str | None]:
    if clip is None:
        return None, {}, None
    try:
        tokenized = clip.tokenize(text)
    except Exception as error:  # Tokenizers differ across ComfyUI/model revisions.
        return None, {}, f"{type(error).__name__}: {error}"
    if isinstance(tokenized, Mapping):
        per_encoder = {
            str(key): count
            for key, value in tokenized.items()
            if (count := _sequence_token_count(value)) is not None
        }
    else:
        count = _sequence_token_count(tokenized)
        per_encoder = {"default": count} if count is not None else {}
    return (max(per_encoder.values()) if per_encoder else None), per_encoder, None


def _parse_assignments(value: str) -> list[dict]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"media_assignments_json is invalid JSON: {error}") from error
    if isinstance(payload, Mapping):
        payload = payload.get("subjects", payload.get("assignments", []))
    if not isinstance(payload, list):
        raise ValueError("media_assignments_json must be a list or an object with subjects")
    assignments = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ValueError(f"media assignment {index} must be an object")
        subject_id = str(raw.get("subject_id") or raw.get("id") or "").strip()
        if not _SAFE_SUBJECT_ID.fullmatch(subject_id):
            raise ValueError(
                f"media assignment {index} subject_id must contain only letters, numbers, "
                "CJK characters, underscores or hyphens"
            )
        assignment = {"subject_id": subject_id}
        for key in ("picture_ordinal", "video_ordinal", "audio_ordinal"):
            ordinal = int(raw.get(key, 0) or 0)
            if ordinal < 0:
                raise ValueError(f"media assignment {index} {key} cannot be negative")
            assignment[key] = ordinal
        if not any(assignment[key] for key in assignment if key.endswith("_ordinal")):
            raise ValueError(f"media assignment {index} has no media ordinal")
        role = str(raw.get("role") or "subject_reference").strip()[:128]
        assignment["role"] = role or "subject_reference"
        assignments.append(assignment)
    return assignments


def _assignment_lines(assignments: list[dict]) -> list[str]:
    lines = []
    for item in assignments:
        media = []
        if item["picture_ordinal"]:
            media.append(f"visual identity from <Picture {item['picture_ordinal']}>")
        if item["video_ordinal"]:
            media.append(f"motion/context from <Video {item['video_ordinal']}>")
        if item["audio_ordinal"]:
            media.append(f"voice identity from <Audio {item['audio_ordinal']}>")
        lines.append(
            f"- Subject {item['subject_id']} ({item['role']}): " + "; ".join(media) + "."
        )
    return lines


def _extract_media_tags(text: str) -> list[dict]:
    canonical = canonicalize_media_tags(text)
    result = []
    for index, match in enumerate(MEDIA_TAG_RE.finditer(canonical)):
        media_type = (match.group(1) or match.group(3)).lower()
        result.append(
            {
                "occurrence": index,
                "type": "Picture" if media_type in {"image", "picture"} else media_type.title(),
                "ordinal": int(match.group(2) or match.group(4)),
                "text": match.group(0),
            }
        )
    return result


def _tag_ordinals(tags: list[dict]) -> dict[str, list[int]]:
    result: dict[str, set[int]] = {"Picture": set(), "Video": set(), "Audio": set()}
    for item in tags:
        result[item["type"]].add(int(item["ordinal"]))
    return {media_type: sorted(ordinals) for media_type, ordinals in result.items()}


def _assignment_ordinals(assignments: list[dict]) -> dict[str, list[int]]:
    fields = {
        "Picture": "picture_ordinal",
        "Video": "video_ordinal",
        "Audio": "audio_ordinal",
    }
    return {
        media_type: sorted(
            {
                int(item[field])
                for item in assignments
                if int(item[field]) > 0
            }
        )
        for media_type, field in fields.items()
    }


def compile_prompt_budget(
    prompt: str,
    character_limit: int,
    token_limit: int,
    picture_count: int,
    video_count: int,
    audio_count: int,
    media_assignments_json: str,
    append_role_bindings: bool,
    allow_shared_audio: bool,
    require_exact_token_count: bool,
    clip=None,
) -> tuple[str, bool, str, int, int, int, str, str]:
    if character_limit <= 0:
        raise ValueError("character_limit must be positive")
    if token_limit < 0:
        raise ValueError("token_limit cannot be negative")
    counts = {
        "pictures": max(0, int(picture_count)),
        "videos": max(0, int(video_count)),
        "audios": max(0, int(audio_count)),
    }
    assignments = _parse_assignments(media_assignments_json)
    findings: list[dict] = []
    warnings: list[dict] = []
    subject_ids = [item["subject_id"] for item in assignments]
    if len(subject_ids) != len(set(subject_ids)):
        findings.append({"code": "duplicate_subject_id", "subject_ids": subject_ids})

    limits = {
        "picture_ordinal": counts["pictures"],
        "video_ordinal": counts["videos"],
        "audio_ordinal": counts["audios"],
    }
    audio_owners: dict[int, list[str]] = {}
    for item in assignments:
        for key, limit in limits.items():
            ordinal = item[key]
            if ordinal > limit:
                findings.append(
                    {
                        "code": "assignment_media_ordinal_out_of_range",
                        "subject_id": item["subject_id"],
                        "field": key,
                        "ordinal": ordinal,
                        "connected_count": limit,
                    }
                )
        if item["audio_ordinal"]:
            audio_owners.setdefault(item["audio_ordinal"], []).append(item["subject_id"])
    if not allow_shared_audio:
        for ordinal, owners in audio_owners.items():
            if len(owners) > 1:
                findings.append(
                    {
                        "code": "audio_reference_assigned_to_multiple_subjects",
                        "audio_ordinal": ordinal,
                        "subject_ids": owners,
                    }
                )

    role_lines = _assignment_lines(assignments)
    source_prompt = str(prompt or "")
    source_prompt_tags = _extract_media_tags(source_prompt)
    candidate = source_prompt
    if append_role_bindings and role_lines:
        separator = "\n\n" if candidate else ""
        candidate = candidate + separator + "Reference role bindings:\n" + "\n".join(role_lines)
    canonical_candidate = canonicalize_media_tags(candidate)
    tags_before = _extract_media_tags(candidate)
    invalid_prompt_tags = []
    for item in tags_before:
        limit = counts[item["type"].lower() + "s"]
        if not 1 <= item["ordinal"] <= limit:
            finding = {
                "code": "prompt_media_tag_out_of_range",
                "tag": item["text"],
                "connected_count": limit,
            }
            findings.append(finding)
            invalid_prompt_tags.append(finding)
    if invalid_prompt_tags:
        # Preserve the requested ordinals so the report cannot be mistaken for a
        # successful remap.  The node abstains and never guesses a replacement.
        normalized = canonical_candidate
        tag_warnings = []
    else:
        normalized, tag_warnings = prepare_prompt(canonical_candidate, counts, strict=False)
    tags_after = _extract_media_tags(normalized)
    if tag_warnings:
        warnings.extend({"code": "prompt_tag_normalization", "message": item} for item in tag_warnings)

    per_type_order: dict[str, list[int]] = {"Picture": [], "Video": [], "Audio": []}
    for item in tags_after:
        per_type_order[item["type"]].append(item["ordinal"])
    for media_type, order in per_type_order.items():
        if order and order != sorted(order):
            warnings.append(
                {
                    "code": "nonascending_media_tag_order",
                    "media_type": media_type,
                    "observed_order": order,
                    "message": "Nonascending order is reported for review but is not inherently invalid.",
                }
            )

    assigned_ordinals = _assignment_ordinals(assignments)
    source_prompt_ordinals = _tag_ordinals(source_prompt_tags)
    connected_ordinals = {
        "Picture": list(range(1, counts["pictures"] + 1)),
        "Video": list(range(1, counts["videos"] + 1)),
        "Audio": list(range(1, counts["audios"] + 1)),
    }
    unassigned_connected_ordinals = {
        media_type: sorted(set(connected) - set(assigned_ordinals[media_type]))
        for media_type, connected in connected_ordinals.items()
    }
    prompt_only_ordinals = {
        media_type: sorted(
            set(source_prompt_ordinals[media_type]) - set(assigned_ordinals[media_type])
        )
        for media_type in source_prompt_ordinals
    }
    if assignments and not append_role_bindings:
        warnings.append(
            {
                "code": "role_bindings_not_appended_to_prompt",
                "message": (
                    "Assignments remain in media_map_json/report_json only; H3 Conditioning "
                    "does not consume that JSON as a role instruction."
                ),
            }
        )
    for media_type, ordinals in unassigned_connected_ordinals.items():
        if ordinals:
            warnings.append(
                {
                    "code": "connected_media_without_subject_assignment",
                    "media_type": media_type,
                    "ordinals": ordinals,
                    "message": (
                        "This can be intentional for scene, ambience or motion-only references; "
                        "review the role contract."
                    ),
                }
            )
    for media_type, ordinals in prompt_only_ordinals.items():
        if ordinals:
            warnings.append(
                {
                    "code": "prompt_media_without_subject_assignment",
                    "media_type": media_type,
                    "ordinals": ordinals,
                    "message": (
                        "The source prompt references this media, but no subject assignment owns it."
                    ),
                }
            )

    character_count = len(normalized)
    utf8_bytes = len(normalized.encode("utf-8"))
    estimated_tokens = _estimated_token_count(normalized)
    exact_tokens, per_encoder, tokenizer_error = _clip_token_count(clip, normalized)
    official_submission_compatible = (
        character_count <= OFFICIAL_H3_SUBMISSION_CHARACTER_LIMIT
    )
    character_limit_semantics = (
        "official_h3_submission_compatibility_ceiling"
        if character_limit == OFFICIAL_H3_SUBMISSION_CHARACTER_LIMIT
        else "user_configured_audit_ceiling"
    )
    if character_count > character_limit:
        findings.append(
            {
                "code": "character_budget_exceeded",
                "actual": character_count,
                "limit": int(character_limit),
                "limit_semantics": character_limit_semantics,
            }
        )
    if character_limit > OFFICIAL_H3_SUBMISSION_CHARACTER_LIMIT:
        warnings.append(
            {
                "code": "configured_character_limit_exceeds_official_h3_submission_limit",
                "configured_limit": int(character_limit),
                "official_submission_limit": OFFICIAL_H3_SUBMISSION_CHARACTER_LIMIT,
                "actual": character_count,
                "official_submission_compatible": official_submission_compatible,
                "message": (
                    "This user override can pass prompts that the current official MiniMax H3 "
                    "CLI submission contract rejects. It does not raise that official limit."
                ),
            }
        )
    budget_tokens = exact_tokens if exact_tokens is not None else estimated_tokens
    token_method = "connected_clip_tokenizer" if exact_tokens is not None else "planning_estimate"
    if token_limit and budget_tokens > token_limit:
        findings.append(
            {
                "code": "token_budget_exceeded",
                "actual": budget_tokens,
                "limit": int(token_limit),
                "method": token_method,
            }
        )
    if require_exact_token_count and exact_tokens is None:
        findings.append(
            {
                "code": "exact_token_count_unavailable",
                "tokenizer_error": tokenizer_error,
            }
        )
    elif exact_tokens is None:
        warnings.append(
            {
                "code": "exact_token_count_unavailable",
                "message": "Connect the actual CLIP to replace the planning estimate when supported.",
                "tokenizer_error": tokenizer_error,
            }
        )

    decision = "ABSTAIN" if findings else "PASS"
    media_payload = {
        "counts": counts,
        "assignments": assignments,
        "role_binding_lines": role_lines,
        "role_bindings_appended": bool(append_role_bindings and role_lines),
        "source_prompt_tags": source_prompt_tags,
        "source_prompt_ordinals": source_prompt_ordinals,
        "assigned_ordinals": assigned_ordinals,
        "connected_ordinals": connected_ordinals,
        "unassigned_connected_ordinals": unassigned_connected_ordinals,
        "prompt_only_ordinals": prompt_only_ordinals,
        "tags_in_compiled_prompt": tags_after,
        "tag_order_by_type": per_type_order,
    }
    report = {
        "schema": PROMPT_BUDGET_SCHEMA,
        "decision": decision,
        "prompt_truncated": False,
        "character_count_unicode_codepoints": character_count,
        "utf8_byte_count": utf8_bytes,
        "character_limit": int(character_limit),
        "character_limit_semantics": character_limit_semantics,
        "official_h3_submission_character_limit": OFFICIAL_H3_SUBMISSION_CHARACTER_LIMIT,
        "official_h3_submission_limit_source": OFFICIAL_H3_SUBMISSION_LIMIT_SOURCE,
        "official_h3_submission_compatible": official_submission_compatible,
        "local_tokenizer_runtime_boundary_probed": False,
        "estimated_token_count": estimated_tokens,
        "exact_token_count": exact_tokens,
        "exact_token_count_per_encoder": per_encoder,
        "token_count_method_for_limit": token_method,
        "token_count_scope": "compiled_text_only_without_connected_visual_embeddings",
        "token_limit": int(token_limit) if token_limit else None,
        "media": media_payload,
        "findings": findings,
        "warnings": warnings,
        "limitations": [
            "The default 7000-character value matches the current official MiniMax H3 CLI submission contract.",
            "That official CLI submission rule is not evidence of a 7000-character hard limit in the open-weight tokenizer or local ComfyUI runtime.",
            "This node does not probe the active runtime's tokenizer boundary; a connected CLIP is used only for an exact compiled-text token count when supported.",
            "Raising character_limit is a local audit override and does not raise the current official CLI submission limit.",
            "The estimate is not tokenizer-exact; only a successful connected-CLIP count is labelled exact.",
            "Even an exact connected-CLIP count here covers compiled text only; image/video references add vision and timestamp tokens during H3 Conditioning.",
            "Role binding text is a prompt contract and does not prove model identity/voice adherence.",
            "This node never truncates or silently drops prompt content.",
        ],
    }
    return (
        normalized,
        not findings,
        decision,
        character_count,
        estimated_tokens,
        exact_tokens if exact_tokens is not None else -1,
        _json(media_payload),
        _json(report),
    )
