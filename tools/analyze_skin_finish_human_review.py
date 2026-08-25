#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import build_skin_finish_human_review as builder
import run_skin_finish_live_sam31_validation as base


ASSESSABILITY = {
    "assessable",
    "source_insufficient",
    "playback_problem",
    "unsure",
}
VOTES = {"A", "B", "tie", "abstain"}
FAILURES = {
    "identity",
    "mouth_eye",
    "flicker",
    "halo",
    "cross_person",
    "av_sync",
}


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def analyze_review(
    submission: dict[str, Any],
    public: dict[str, Any],
    private: dict[str, Any],
) -> dict[str, Any]:
    submission = _require_dict(submission, "submission")
    public = _require_dict(public, "public manifest")
    private = _require_dict(private, "private key")
    if public.get("schema") != builder.SCHEMA:
        raise ValueError("unsupported public manifest schema")
    if submission.get("schema") != builder.SCHEMA:
        raise ValueError("unsupported submission schema")
    review_id = str(public.get("review_id", ""))
    if not review_id or submission.get("review_id") != review_id:
        raise ValueError("submission review_id mismatch")
    if private.get("review_id") != review_id:
        raise ValueError("private-key review_id mismatch")
    public_without_hash = dict(public)
    public_hash = str(public_without_hash.pop("sha256", ""))
    if not public_hash or builder._hash_json(public_without_hash) != public_hash:
        raise ValueError("public manifest hash is invalid")
    if submission.get("public_manifest_sha256") != public_hash:
        raise ValueError("submission is bound to a different public manifest")
    if private.get("public_manifest_sha256") != public_hash:
        raise ValueError("private key is bound to a different public manifest")
    private_without_hash = dict(private)
    private_hash = str(private_without_hash.pop("sha256", ""))
    if not private_hash or builder._hash_json(private_without_hash) != private_hash:
        raise ValueError("private key hash is invalid")
    mapping = _require_dict(private.get("mapping"), "private mapping")
    if set(mapping) != {"A", "B"} or set(mapping.values()) != {
        "source",
        "candidate",
    }:
        raise ValueError("private mapping is not a source/candidate bijection")
    assessability = str(submission.get("assessability", ""))
    if assessability not in ASSESSABILITY:
        raise ValueError("submission assessability is missing or invalid")
    raw_criteria = _require_dict(submission.get("criteria"), "criteria")
    expected_criteria = {key for key, _ in builder.CRITERIA}
    if set(raw_criteria) != expected_criteria:
        raise ValueError("submission criteria set does not match the review contract")
    invalid_votes = {
        key: value
        for key, value in raw_criteria.items()
        if value not in VOTES and not (assessability != "assessable" and value is None)
    }
    if invalid_votes:
        raise ValueError(f"submission has invalid or missing votes: {invalid_votes}")
    if assessability == "assessable":
        missing_votes = {
            key: value for key, value in raw_criteria.items() if value not in VOTES
        }
        if missing_votes:
            raise ValueError(
                f"submission has invalid or missing votes: {missing_votes}"
            )
    normalized_criteria = {
        key: ("abstain" if value is None else value)
        for key, value in raw_criteria.items()
    }
    raw_failures = _require_dict(submission.get("hard_failures"), "hard_failures")
    if set(raw_failures) != {"A", "B"}:
        raise ValueError("hard_failures must contain separate A and B arrays")
    for label in ("A", "B"):
        values = raw_failures[label]
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ValueError(f"hard_failures.{label} must be a unique array")
        if not set(values).issubset(FAILURES):
            raise ValueError(f"hard_failures.{label} contains an unknown failure")

    revealed_votes: dict[str, str] = {}
    counts = {"candidate": 0, "source": 0, "tie": 0, "abstain": 0}
    for criterion, vote in normalized_criteria.items():
        revealed = mapping[vote] if vote in {"A", "B"} else vote
        revealed_votes[criterion] = revealed
        counts[revealed] += 1
    revealed_failures = {
        mapping[label]: list(raw_failures[label]) for label in ("A", "B")
    }
    candidate_hard_fail = bool(revealed_failures["candidate"])
    source_hard_fail = bool(revealed_failures["source"])
    if assessability != "assessable":
        status = f"ABSTAIN_{assessability.upper()}"
        recommendation = "NO_AESTHETIC_DECISION"
    elif candidate_hard_fail:
        status = "HUMAN_REVIEW_COMPLETE_CANDIDATE_HARD_FAIL"
        recommendation = "REJECT_CANDIDATE"
    else:
        status = "HUMAN_REVIEW_COMPLETE"
        recommendation = "HUMAN_JUDGMENT_RECORDED_NO_AUTO_ACCEPT"
    report: dict[str, Any] = {
        "schema": f"{builder.SCHEMA}/analysis",
        "created_at": base._utc_now(),
        "review_id": review_id,
        "status": status,
        "assessability": assessability,
        "normalized_unanswered_criteria": [
            key for key, value in raw_criteria.items() if value is None
        ],
        "revealed_votes": revealed_votes,
        "vote_counts": counts,
        "hard_failures": revealed_failures,
        "candidate_hard_fail": candidate_hard_fail,
        "source_hard_fail": source_hard_fail,
        "recommendation": recommendation,
        "notes": {
            "left": str(submission.get("left_notes", "")),
            "right": str(submission.get("right_notes", "")),
            "overall": str(submission.get("notes", "")),
        },
        "claim_boundary": (
            "One human review records this fixed media pair only. It does not establish "
            "automatic acceptance, universal superiority, fairness, identity truth, "
            "long-video continuity or a Safety Audit float-mask pass."
        ),
        "automatic_accept": False,
    }
    report["sha256"] = builder._hash_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    submission = json.loads(args.submission.read_text(encoding="utf-8-sig"))
    public = json.loads(args.public_manifest.read_text(encoding="utf-8-sig"))
    private = json.loads(args.private_key.read_text(encoding="utf-8-sig"))
    report = analyze_review(submission, public, private)
    if args.output:
        base._json_write(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
