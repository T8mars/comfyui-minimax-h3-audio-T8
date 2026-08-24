from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
import unicodedata
from typing import Any, Mapping

from .prompt_rewriter_8b import parse_rewritten_prompt
from .prompt_tags import MEDIA_TAG_RE, canonicalize_media_tags


SCHEMA = "minimax_h3_prompt_semantic_contract_audit/v1"
ALLOWED_SCOPES = frozenset({"full", "integrated", "soundscape", "music"})
GROUP_KEYS = frozenset({"id", "any_of", "scope"})
ROOT_KEYS = frozenset({"required_groups", "forbidden_groups"})
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
DIALOGUE_RE = re.compile(r"<d>.*?</d>", flags=re.IGNORECASE | re.DOTALL)
SPACE_RE = re.compile(r"\s+")

MAX_PROMPT_UTF8_BYTES = 400_000
MAX_CONTRACT_UTF8_BYTES = 65_536
MAX_GROUPS_PER_KIND = 64
MAX_PHRASES_PER_GROUP = 32
MAX_PHRASE_CHARACTERS = 256


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} cannot be empty")
    if len(value.encode("utf-8")) > MAX_PROMPT_UTF8_BYTES:
        raise ValueError(
            f"{name} exceeds the {MAX_PROMPT_UTF8_BYTES}-byte UTF-8 safety limit"
        )
    return value


def _normalize(value: str) -> str:
    return SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _uses_substring_matching(value: str) -> bool:
    for character in value:
        codepoint = ord(character)
        if (
            0x3040 <= codepoint <= 0x30FF  # Hiragana / Katakana
            or 0x3400 <= codepoint <= 0x9FFF  # CJK
            or 0xAC00 <= codepoint <= 0xD7AF  # Hangul syllables
            or 0xF900 <= codepoint <= 0xFAFF  # CJK compatibility
        ):
            return True
    return False


def _contains_phrase(haystack: str, phrase: str) -> bool:
    normalized_haystack = _normalize(haystack)
    normalized_phrase = _normalize(phrase)
    if _uses_substring_matching(normalized_phrase):
        return normalized_phrase in normalized_haystack
    # ASCII action anchors may legitimately touch CJK text without a space. Keep
    # ASCII word boundaries strict enough to reject "turn" inside "return", but
    # do not treat a neighboring CJK codepoint as part of the same Latin token.
    word_class = r"[0-9a-z_]" if normalized_phrase.isascii() else r"\w"
    left = (
        rf"(?<!{word_class})"
        if normalized_phrase[0].isalnum() or normalized_phrase[0] == "_"
        else ""
    )
    right = (
        rf"(?!{word_class})"
        if normalized_phrase[-1].isalnum() or normalized_phrase[-1] == "_"
        else ""
    )
    return re.search(left + re.escape(normalized_phrase) + right, normalized_haystack) is not None


def _parse_groups(value: Any, name: str, seen_ids: set[str]) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    if len(value) > MAX_GROUPS_PER_KIND:
        raise ValueError(f"{name} may contain at most {MAX_GROUPS_PER_KIND} groups")

    groups: list[dict[str, Any]] = []
    for index, raw_group in enumerate(value):
        location = f"{name}[{index}]"
        if not isinstance(raw_group, dict):
            raise ValueError(f"{location} must be a JSON object")
        unknown = sorted(set(raw_group) - GROUP_KEYS)
        if unknown:
            raise ValueError(f"{location} contains unknown keys: {', '.join(unknown)}")
        missing = sorted(GROUP_KEYS - set(raw_group))
        if missing:
            raise ValueError(f"{location} is missing keys: {', '.join(missing)}")

        group_id = raw_group["id"]
        if not isinstance(group_id, str) or SAFE_ID.fullmatch(group_id) is None:
            raise ValueError(
                f"{location}.id must match {SAFE_ID.pattern} and be at most 64 characters"
            )
        if group_id in seen_ids:
            raise ValueError(f"duplicate semantic group id: {group_id}")
        seen_ids.add(group_id)

        scope = raw_group["scope"]
        if not isinstance(scope, str) or scope not in ALLOWED_SCOPES:
            raise ValueError(
                f"{location}.scope must be one of: {', '.join(sorted(ALLOWED_SCOPES))}"
            )

        raw_phrases = raw_group["any_of"]
        if not isinstance(raw_phrases, list) or not raw_phrases:
            raise ValueError(f"{location}.any_of must be a non-empty JSON array")
        if len(raw_phrases) > MAX_PHRASES_PER_GROUP:
            raise ValueError(
                f"{location}.any_of may contain at most {MAX_PHRASES_PER_GROUP} phrases"
            )
        phrases: list[str] = []
        normalized_seen: set[str] = set()
        for phrase_index, phrase in enumerate(raw_phrases):
            phrase_location = f"{location}.any_of[{phrase_index}]"
            if not isinstance(phrase, str) or not phrase.strip():
                raise ValueError(f"{phrase_location} must be a non-empty string")
            if len(phrase) > MAX_PHRASE_CHARACTERS:
                raise ValueError(
                    f"{phrase_location} exceeds {MAX_PHRASE_CHARACTERS} characters"
                )
            normalized_phrase = _normalize(phrase)
            if not normalized_phrase:
                raise ValueError(f"{phrase_location} normalizes to an empty string")
            if normalized_phrase in normalized_seen:
                raise ValueError(f"{location}.any_of contains duplicate phrases")
            normalized_seen.add(normalized_phrase)
            phrases.append(phrase)
        groups.append({"id": group_id, "scope": scope, "any_of": phrases})
    return groups


def _parse_contract(value: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, str):
        raise ValueError("semantic_contract_json must be a string")
    payload = value.encode("utf-8")
    if len(payload) > MAX_CONTRACT_UTF8_BYTES:
        raise ValueError(
            "semantic_contract_json exceeds the "
            f"{MAX_CONTRACT_UTF8_BYTES}-byte UTF-8 safety limit"
        )
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"semantic_contract_json is invalid JSON at line {error.lineno}, column {error.colno}"
        ) from error
    if not isinstance(parsed, dict):
        raise ValueError("semantic_contract_json root must be a JSON object")
    unknown = sorted(set(parsed) - ROOT_KEYS)
    if unknown:
        raise ValueError(
            "semantic_contract_json contains unknown root keys: " + ", ".join(unknown)
        )
    seen_ids: set[str] = set()
    return {
        "required_groups": _parse_groups(
            parsed.get("required_groups"), "required_groups", seen_ids
        ),
        "forbidden_groups": _parse_groups(
            parsed.get("forbidden_groups"), "forbidden_groups", seen_ids
        ),
    }


def _media_tag_facts(value: str) -> Counter[tuple[str, int]]:
    facts: Counter[tuple[str, int]] = Counter()
    for match in MEDIA_TAG_RE.finditer(canonicalize_media_tags(value)):
        media_type = (match.group(1) or match.group(3)).lower()
        media_type = "picture" if media_type in {"image", "picture"} else media_type
        facts[(media_type, int(match.group(2) or match.group(4)))] += 1
    return facts


def _counter_rows(counter: Counter[Any]) -> list[dict[str, Any]]:
    rows = []
    for value, count in sorted(counter.items(), key=lambda item: repr(item[0])):
        if isinstance(value, tuple):
            rows.append({"type": value[0], "ordinal": value[1], "count": count})
        else:
            rows.append({"value": value, "count": count})
    return rows


def audit_prompt_semantics(
    *,
    original_prompt: str,
    candidate_prompt: str,
    semantic_contract_json: str,
    accept_candidate_after_review: bool,
    preserve_exact_dialogue: bool = True,
    preserve_source_media_tags: bool = True,
    allow_new_media_tags: bool = True,
) -> tuple[str, str, bool, str, str]:
    """Audit explicit user-authored phrase anchors without claiming semantic equivalence."""

    source = _required_text(original_prompt, "original_prompt")
    candidate = _required_text(candidate_prompt, "candidate_prompt")
    findings: list[dict[str, Any]] = []
    contract_error = ""
    try:
        contract = _parse_contract(semantic_contract_json)
    except ValueError as error:
        contract = {"required_groups": [], "forbidden_groups": []}
        contract_error = str(error)
        findings.append({"code": "invalid_semantic_contract", "message": contract_error})

    integrated, soundscape, music, parse_warnings = parse_rewritten_prompt(candidate)
    scoped_text = {
        "full": candidate,
        "integrated": integrated,
        "soundscape": soundscape,
        "music": music,
    }
    non_full_scopes = {
        group["scope"]
        for kind in ("required_groups", "forbidden_groups")
        for group in contract[kind]
        if group["scope"] != "full"
    }
    if non_full_scopes:
        for warning in parse_warnings:
            findings.append(
                {"code": "candidate_structure_warning", "message": warning}
            )
        for scope in sorted(non_full_scopes):
            if not scoped_text[scope]:
                findings.append(
                    {
                        "code": "candidate_scope_empty",
                        "scope": scope,
                        "message": f"candidate {scope} field is empty",
                    }
                )

    required_results: list[dict[str, Any]] = []
    forbidden_results: list[dict[str, Any]] = []
    for group in contract["required_groups"]:
        matches = [
            phrase
            for phrase in group["any_of"]
            if _contains_phrase(scoped_text[group["scope"]], phrase)
        ]
        row = {"id": group["id"], "scope": group["scope"], "matched": matches}
        required_results.append(row)
        if not matches:
            findings.append(
                {
                    "code": "required_group_missing",
                    "id": group["id"],
                    "scope": group["scope"],
                    "any_of": group["any_of"],
                }
            )
    for group in contract["forbidden_groups"]:
        matches = [
            phrase
            for phrase in group["any_of"]
            if _contains_phrase(scoped_text[group["scope"]], phrase)
        ]
        row = {"id": group["id"], "scope": group["scope"], "matched": matches}
        forbidden_results.append(row)
        if matches:
            findings.append(
                {
                    "code": "forbidden_group_present",
                    "id": group["id"],
                    "scope": group["scope"],
                    "matched": matches,
                }
            )

    source_dialogue = Counter(DIALOGUE_RE.findall(source))
    candidate_dialogue = Counter(DIALOGUE_RE.findall(candidate))
    if bool(preserve_exact_dialogue) and source_dialogue != candidate_dialogue:
        findings.append(
            {
                "code": "exact_dialogue_mismatch",
                "missing": _counter_rows(source_dialogue - candidate_dialogue),
                "added": _counter_rows(candidate_dialogue - source_dialogue),
            }
        )

    source_tags = _media_tag_facts(source)
    candidate_tags = _media_tag_facts(candidate)
    if bool(preserve_source_media_tags):
        missing_tags = source_tags - candidate_tags
        if missing_tags:
            findings.append(
                {
                    "code": "source_media_tags_missing",
                    "tags": _counter_rows(missing_tags),
                }
            )
    if not bool(allow_new_media_tags):
        added_tags = candidate_tags - source_tags
        if added_tags:
            findings.append(
                {
                    "code": "new_media_tags_present",
                    "tags": _counter_rows(added_tags),
                }
            )

    anchor_count = len(contract["required_groups"]) + len(contract["forbidden_groups"])
    mechanical_pass = bool(anchor_count) and not findings
    if findings:
        decision = "REJECT"
    elif not anchor_count:
        decision = "ABSTAIN_NO_SEMANTIC_ANCHORS"
    elif not bool(accept_candidate_after_review):
        decision = "REVIEW_REQUIRED"
    else:
        decision = "ACCEPTED"
    safe_prompt = candidate if decision == "ACCEPTED" else source

    report = {
        "schema": SCHEMA,
        "decision": decision,
        "mechanical_pass": mechanical_pass,
        "accept_candidate_after_review": bool(accept_candidate_after_review),
        "safe_prompt_source": "candidate" if decision == "ACCEPTED" else "original",
        "contract_valid": not bool(contract_error),
        "contract_error": contract_error,
        "contract_sha256": hashlib.sha256(
            str(semantic_contract_json).encode("utf-8")
        ).hexdigest(),
        "original_prompt_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "candidate_prompt_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
        "anchor_count": anchor_count,
        "required_results": required_results,
        "forbidden_results": forbidden_results,
        "findings": findings,
        "preservation": {
            "preserve_exact_dialogue": bool(preserve_exact_dialogue),
            "preserve_source_media_tags": bool(preserve_source_media_tags),
            "allow_new_media_tags": bool(allow_new_media_tags),
            "source_dialogue_blocks": sum(source_dialogue.values()),
            "candidate_dialogue_blocks": sum(candidate_dialogue.values()),
            "source_media_tags": _counter_rows(source_tags),
            "candidate_media_tags": _counter_rows(candidate_tags),
        },
        "limits": {
            "prompt_utf8_bytes": MAX_PROMPT_UTF8_BYTES,
            "contract_utf8_bytes": MAX_CONTRACT_UTF8_BYTES,
            "groups_per_kind": MAX_GROUPS_PER_KIND,
            "phrases_per_group": MAX_PHRASES_PER_GROUP,
            "phrase_characters": MAX_PHRASE_CHARACTERS,
        },
        "boundary": (
            "This node checks only explicit user-authored phrase anchors plus exact dialogue "
            "and media-tag preservation. A mechanical pass is not proof of semantic "
            "equivalence, visual quality, provider quality, or MiniMax H3 generation quality."
        ),
    }
    return safe_prompt, candidate, mechanical_pass, decision, _json(report)
