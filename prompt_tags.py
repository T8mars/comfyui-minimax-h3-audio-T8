from __future__ import annotations

import json
import re


MEDIA_TAG_RE = re.compile(
    r"<\s*(Image|Picture|Video|Audio)\s*(\d+)\s*>|"
    r"(?<![\w<])(Image|Picture|Video|Audio)\s*#?\s*(\d+)\b(?!\s*>)",
    re.IGNORECASE,
)


def canonicalize_media_tags(prompt: str) -> str:
    def replacement(match: re.Match) -> str:
        media_type = (match.group(1) or match.group(3)).lower()
        ordinal = int(match.group(2) or match.group(4))
        official_type = "Picture" if media_type in {"image", "picture"} else media_type.title()
        return f"<{official_type} {ordinal}>"

    return MEDIA_TAG_RE.sub(replacement, prompt or "")


def prepare_prompt(
    prompt: str,
    counts: dict[str, int],
    source_audio_ordinal: int = 0,
    prompt_primary_audio_ordinal: int = 1,
    strict: bool = True,
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    fatal_warnings: list[str] = []
    limits = {
        "picture": max(0, int(counts.get("pictures", 0))),
        "video": max(0, int(counts.get("videos", 0))),
        "audio": max(0, int(counts.get("audios", 0))),
    }

    def add_warning(message: str) -> None:
        if message not in warnings:
            warnings.append(message)

    def replacement(match: re.Match) -> str:
        explicit = match.group(1) is not None
        media_type = (match.group(1) or match.group(3)).lower()
        original_ordinal = int(match.group(2) or match.group(4))
        official_type = "Picture" if media_type in {"image", "picture"} else media_type.title()
        limit_key = official_type.lower()
        limit = limits[limit_key]
        ordinal = original_ordinal

        if (
            limit_key == "audio"
            and source_audio_ordinal > 0
            and prompt_primary_audio_ordinal > 0
            and ordinal == int(prompt_primary_audio_ordinal)
        ):
            ordinal = int(source_audio_ordinal)

        if ordinal == 0 and limit >= 1:
            ordinal = 1
            add_warning(
                f"{match.group(0)} used legacy zero-based numbering and was mapped to "
                f"<{official_type} 1>"
            )

        if 1 <= ordinal <= limit:
            return f"<{official_type} {ordinal}>"

        if limit == 1 and ordinal > 1:
            add_warning(
                f"{match.group(0)} was mapped to <{official_type} 1> because exactly one "
                f"{limit_key} is connected"
            )
            return f"<{official_type} 1>"

        message = (
            f"<{official_type} {ordinal}> is not connected; available {limit_key} count is {limit}"
        )
        add_warning(message)
        if strict and (explicit or limit > 0):
            fatal_warnings.append(message)
            return f"<{official_type} {ordinal}>"

        # When no media of this type exists, plain prose such as "Video 2" is
        # common in generated prompts and is not necessarily a reference request.
        # Once at least one same-type item is connected, an ambiguous ordinal still
        # fails in strict mode. Non-strict explicit tags are demoted to prose so
        # Qwen never receives a dangling reference tag.
        return match.group(0) if not explicit else f"{official_type} {ordinal}"

    normalized = MEDIA_TAG_RE.sub(replacement, prompt or "")
    if fatal_warnings:
        raise ValueError(
            "MiniMax H3 prompt media tag validation failed: "
            + "; ".join(dict.fromkeys(fatal_warnings))
            + ". Connect the referenced media, correct the ordinal, or disable "
            "strict_prompt_tags to treat it as plain prompt text."
        )
    return normalized, warnings


def media_map_json(
    pictures: list[str], videos: list[str], audios: list[str], source_audio_ordinal: int = 0
) -> str:
    return json.dumps(
        {
            "pictures": {str(index + 1): label for index, label in enumerate(pictures)},
            "videos": {str(index + 1): label for index, label in enumerate(videos)},
            "audios": {str(index + 1): label for index, label in enumerate(audios)},
            "source_audio_ordinal": source_audio_ordinal or None,
        },
        ensure_ascii=False,
        indent=2,
    )
