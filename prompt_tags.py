from __future__ import annotations

import json
import re


MEDIA_TAG_RE = re.compile(
    r"<\s*(Image|Picture|Video|Audio)\s*(\d+)\s*>|"
    r"(?<![\w<])(Image|Picture|Video|Audio)\s*#?\s*(\d+)\b(?!\s*>)",
    re.IGNORECASE,
)
OFFICIAL_TAG_RE = re.compile(r"<(Picture|Video|Audio)\s+(\d+)>", re.IGNORECASE)


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
    normalized = canonicalize_media_tags(prompt)
    if source_audio_ordinal > 0 and prompt_primary_audio_ordinal > 0:
        primary = re.compile(rf"<Audio\s+{int(prompt_primary_audio_ordinal)}>", re.IGNORECASE)
        normalized = primary.sub(f"<Audio {int(source_audio_ordinal)}>", normalized)

    warnings: list[str] = []
    limits = {
        "picture": int(counts.get("pictures", 0)),
        "video": int(counts.get("videos", 0)),
        "audio": int(counts.get("audios", 0)),
    }
    for match in OFFICIAL_TAG_RE.finditer(normalized):
        media_type, ordinal = match.group(1).lower(), int(match.group(2))
        if ordinal < 1 or ordinal > limits[media_type]:
            warnings.append(
                f"{match.group(0)} is not connected; available {media_type} count is {limits[media_type]}"
            )
    if strict and warnings:
        raise ValueError("MiniMax H3 prompt media tag validation failed: " + "; ".join(warnings))
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
