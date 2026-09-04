#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REVEAL_SCHEMA = "t8.minimax_h3.world.action_pair_reveal.v1"
REVIEW_SCHEMA = "t8.minimax_h3.world.action_pair_human_review.v1"
SCREENING_SCHEMA = "t8.minimax_h3.world.action_pair_screening.v1"
RESULT_SCHEMA = "t8.minimax_h3.world.action_pair_result.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load(path: Path, schema: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"{path.name} does not use schema {schema}")
    return value


def _validate_mechanics(screening: dict[str, Any]) -> None:
    if screening.get("mechanical_gate") != "PASS":
        raise ValueError("H3-World mechanical screening did not pass")
    clips = screening.get("clips")
    if not isinstance(clips, dict) or set(clips) != {"forward", "still"}:
        raise ValueError("screening must contain exactly forward and still clips")
    for name, clip in clips.items():
        if not isinstance(clip, dict) or clip.get("strict_av_decode") is not True:
            raise ValueError(f"{name} did not pass strict A/V decode")
        motion = clip.get("motion") or {}
        audio = clip.get("audio") or {}
        if int(motion.get("decoded_frames", 0)) != 124:
            raise ValueError(f"{name} does not contain exactly 124 decoded frames")
        if int(motion.get("black_frame_count", -1)) != 0:
            raise ValueError(f"{name} contains black frames")
        if int(motion.get("frozen_pair_count_at_1e-5", -1)) != 0:
            raise ValueError(f"{name} contains frozen frame pairs")
        if audio.get("finite") is not True:
            raise ValueError(f"{name} audio is not finite")
        if int(audio.get("clipped_sample_values", -1)) != 0:
            raise ValueError(f"{name} audio contains clipped sample values")


def analyze_review(
    *, review_path: Path, screening_path: Path, reveal_path: Path
) -> dict[str, Any]:
    review = _load(review_path, REVIEW_SCHEMA)
    screening = _load(screening_path, SCREENING_SCHEMA)
    reveal = _load(reveal_path, REVEAL_SCHEMA)
    screening_sha256 = _sha256(screening_path)
    if review.get("screening_sha256") != screening_sha256:
        raise ValueError("human review is not bound to this screening.json")
    if reveal.get("screening_sha256") != screening_sha256:
        raise ValueError("reveal.json is not bound to this screening.json")
    _validate_mechanics(screening)

    mapping = reveal.get("mapping")
    if not isinstance(mapping, dict) or set(mapping) != {"A", "B"}:
        raise ValueError("reveal mapping must contain exactly A and B")
    if set(mapping.values()) != {"forward", "still"}:
        raise ValueError("reveal mapping must contain one forward and one still clip")
    if review.get("watched_full_length") is not True:
        raise ValueError("human reviewer did not confirm full-length viewing")
    votes = review.get("votes")
    allowed = {
        "forward": {"A", "B", "tie"},
        "stable": {"yes", "no"},
        "visual": {"A", "B", "tie"},
        "audio": {"both_ok", "A", "B", "problem"},
    }
    if not isinstance(votes, dict) or set(votes) != set(allowed):
        raise ValueError("human review must resolve all four vote fields")
    for key, choices in allowed.items():
        if votes[key] not in choices:
            raise ValueError(f"invalid human review vote {key}={votes[key]!r}")

    forward_label = next(label for label, value in mapping.items() if value == "forward")
    reasons = []
    if votes["forward"] != forward_label:
        reasons.append("reviewer did not identify the forward-controlled candidate")
    if votes["stable"] != "yes":
        reasons.append("reviewer did not accept character motion as stable and usable")
    if votes["audio"] == "problem":
        reasons.append("reviewer reported an audio problem")
    gate = "PASS" if not reasons else "FAIL"
    return {
        "schema": RESULT_SCHEMA,
        "p3_fixed_material_gate": gate,
        "review_sha256": _sha256(review_path),
        "screening_sha256": screening_sha256,
        "reveal_sha256": _sha256(reveal_path),
        "human_votes": votes,
        "human_notes": str(review.get("notes", "")),
        "forward_label_after_reveal": forward_label,
        "mechanical_gate": screening["mechanical_gate"],
        "reasons": reasons,
        "promotion_allowed": gate == "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hash-bind and close the H3-World forward/still human review."
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--screening", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_review(
        review_path=args.review,
        screening_path=args.screening,
        reveal_path=args.reveal,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["p3_fixed_material_gate"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
